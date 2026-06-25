"""Environment worker.

Owns a single environment instance, a client cache, and three ZMQ
sockets (PULL for requests, PUSH for responses, PUSH for stats).
Receives requests from the router, runs rollouts, and pushes
responses + stats back.
"""

import asyncio
import ctypes
import gc
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import msgpack
import zmq
import zmq.asyncio
from pydantic import BaseModel

import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.serve.server._routed_frames import detach_routed_attachments
from verifiers.serve.types import (
    BaseResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.types import ClientConfig
from verifiers.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.process_utils import monitor_death_pipe, set_proc_title
from verifiers.utils.serve_utils import msgpack_encoder
from verifiers.utils.thread_utils import register_executor

# --- routed_experts multipart transport config (issue #76, PR1) ---------------
#
# Bulk routed_experts blobs ride on raw ZMQ multipart frames (frame 2..2+K-1)
# instead of inside the msgpack control body. Two knobs govern the response leg:

# Dedicated serializer pool for the (now lightened) control-body packb. Kept OFF
# the 512-thread default executor so multi-MB packb GIL contention can't starve
# the ~256 concurrent parse_response_tokens calls sharing the default pool.
# packb is GIL-bound, so >16 threads buys nothing.
_SER_EXECUTOR_THREADS = 16
_SER_EXECUTOR_NAME = "vf-ser"

# Byte-budget semaphore: bounds the O(workers x inflight) peak materialization of
# attachment bytes in flight (extract -> pack -> send). The mallopt fix returns
# pages to the OS on free but does NOT bound how many blobs are simultaneously
# live; this does. Documented default below; override via env (fail-loud on a
# misconfigured value — never a silently-clamped magic number).
_BYTE_BUDGET_ENV = "VF_ROUTED_BYTE_BUDGET_MB"
_DEFAULT_BYTE_BUDGET_MB = 512  # ~512 inflight x ~1 MB amortized blob, conservative
                               # vs the unbounded ~2.3-3.6 GB pre-cutover peak.


def _resolve_byte_budget_bytes() -> int:
    """Resolve the routed-attachment byte budget (bytes) for this worker.

    Reads ``VF_ROUTED_BYTE_BUDGET_MB`` if set, else the documented default.
    Fails loud (refuses to start) on a misconfigured override rather than
    silently degrading to an unbounded or clamped budget.
    """
    raw = os.environ.get(_BYTE_BUDGET_ENV)
    if raw is None:
        mb = _DEFAULT_BYTE_BUDGET_MB
    else:
        try:
            mb = int(raw)
        except ValueError as e:
            raise RuntimeError(
                f"{_BYTE_BUDGET_ENV}={raw!r} is not an integer; "
                "refusing to start with a misconfigured routed byte budget"
            ) from e
        if mb <= 0:
            raise RuntimeError(
                f"{_BYTE_BUDGET_ENV}={raw!r} must be a positive number of MiB; "
                "refusing to start with a misconfigured routed byte budget"
            )
    return mb << 20


class _ByteBudget:
    """Async credit gate over a fixed pool of *bytes* (not a count).

    A coroutine acquires ``n`` bytes before materializing+sending its
    attachments and releases exactly ``n`` in a ``finally`` — including on
    cancellation — so a cancel storm cannot leak credits into a deadlock (G2).
    Acquiring more than the whole pool is allowed (it simply waits until the
    pool is otherwise empty) so a single oversized message can never wedge.

    ``release`` is **synchronous and never awaits**: it bumps the counter and
    resolves waiter futures directly. This is the G2 load-bearing property — the
    release call in a cancelled coroutine's ``finally`` cannot itself be
    interrupted by the pending cancellation, so credits always return.
    ``acquire`` is cancel-safe too: a cancellation while waiting drops the
    pending request without ever having decremented the pool.

    The charged amount is ``min(n, capacity)`` (an oversized message waits for an
    otherwise-empty pool, then borrows the whole pool). ``acquire`` returns that
    charge and the caller passes the **same value** back to ``release`` so the
    debit/credit always net to zero.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        # FIFO of (charge, future) for coroutines blocked on insufficient credit.
        self._waiters: list[tuple[int, asyncio.Future[None]]] = []

    @property
    def available(self) -> int:
        return self._available

    async def acquire(self, n: int) -> int:
        """Acquire credit for ``n`` bytes; return the charged amount to release."""
        charge = min(n, self._capacity)
        if not self._waiters and self._available >= charge:
            self._available -= charge
            return charge

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append((charge, fut))
        try:
            await fut
        except asyncio.CancelledError:
            # Cancelled while waiting: if granted concurrently with the cancel
            # (fut already resolved), the charge was debited — give it back.
            # Otherwise drop the still-pending request untouched.
            if fut.done() and not fut.cancelled():
                self.release(charge)
            else:
                try:
                    self._waiters.remove((charge, fut))
                except ValueError:
                    pass
            raise
        # Granted: the waker already debited `charge` on our behalf.
        return charge

    def release(self, charge: int) -> None:
        self._available += charge
        self._wake()

    def _wake(self) -> None:
        """Grant credit to head-of-line waiters while the pool can satisfy them.

        Strict FIFO (no overtaking) keeps the oversized-message case live: the
        head waiter is served first the moment the pool is otherwise empty.
        """
        while self._waiters:
            charge, fut = self._waiters[0]
            if fut.cancelled():
                self._waiters.pop(0)
                continue
            if self._available < charge:
                break
            self._waiters.pop(0)
            self._available -= charge
            fut.set_result(None)


class EnvWorkerStats(BaseModel):
    worker_id: int
    timestamp: float
    active_tasks: int
    lag: EventLoopLagStats = EventLoopLagStats()

    def __str__(self) -> str:
        parts = []
        if self.lag.n > 0:
            parts.append(f"Lag: {self.lag}")
        return " | ".join(parts) if parts else "no lag data"


class EnvWorker:
    """Executes environment logic."""

    def __init__(
        self,
        env_id: str,
        env_args: dict[str, Any] | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
        console_logging: bool = True,
        json_logging: bool = False,
        *,
        worker_id: int,
        worker_name: str,
        request_address: str,
        response_address: str,
        stats_address: str,
        death_pipe: Connection | None = None,
    ):
        set_proc_title(f"EnvWorker{worker_id}")
        self.death_pipe = death_pipe
        self.env_id = env_id
        self.worker_id = worker_id
        self.worker_name = worker_name

        # setup logging — each worker gets its own log file
        logger_kwargs: dict[str, Any] = {
            "console_logging": console_logging,
            "file_logging": log_dir is not None,
            "json_logging": json_logging,
        }
        if log_level is not None:
            logger_kwargs["level"] = log_level
        if log_dir is not None:
            worker_log_file = EnvWorker.get_log_file(log_dir, worker_id)
            worker_log_file.parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["log_file"] = str(worker_log_file)
        vf.setup_logging(**logger_kwargs)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # setup env
        self.logger.debug(
            f"Loading environment {env_id} for worker {worker_name} ({env_args=}, {extra_env_kwargs=})"
        )
        self.env = vf.load_environment(env_id, **(env_args or {}))
        if extra_env_kwargs:
            self.env.set_kwargs(**extra_env_kwargs)

        # setup zmq sockets
        self.ctx = zmq.asyncio.Context()

        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 0)
        self.pull_socket.setsockopt(zmq.LINGER, 0)
        self.pull_socket.bind(request_address)

        self.response_socket = self.ctx.socket(zmq.PUSH)
        self.response_socket.setsockopt(zmq.SNDHWM, 0)
        self.response_socket.setsockopt(zmq.LINGER, 5000)
        self.response_socket.connect(response_address)

        self.stats_socket = self.ctx.socket(zmq.PUSH)
        self.stats_socket.setsockopt(zmq.SNDHWM, 100)
        self.stats_socket.setsockopt(zmq.LINGER, 0)
        self.stats_socket.connect(stats_address)

        # Dedicated serializer pool for the lightened control body (issue #76).
        # Registered so scale_executors keeps its own scaling — pinned at 16
        # threads (1:1 would track the 512 default, defeating the isolation).
        self.ser_executor = ThreadPoolExecutor(
            max_workers=_SER_EXECUTOR_THREADS,
            thread_name_prefix=_SER_EXECUTOR_NAME,
        )
        register_executor(
            _SER_EXECUTOR_NAME,
            self.ser_executor,
            scaling_fn=lambda _concurrency: _SER_EXECUTOR_THREADS,
        )

        # Byte-budget gate over routed-attachment bytes in flight (G2).
        self.byte_budget = _ByteBudget(_resolve_byte_budget_bytes())

        # state tracking
        self.clients: dict[str, Client] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.shutting_down: bool = False

        # stats
        self.lag_monitor = EventLoopLagMonitor()

        self.logger.info(f"Initialized worker {worker_name} on {request_address}")

    async def resolve_client(self, client_config: ClientConfig) -> Client:
        """Resolve the client instance given the request client config."""
        resolved = resolve_client_config(client_config)
        key = resolved.model_dump_json()
        if key not in self.clients:
            self.clients[key] = resolve_client(resolved)
        return self.clients[key]

    async def handle_run_rollout(
        self, request: RunRolloutRequest
    ) -> RunRolloutResponse:
        client = await self.resolve_client(request.client_config)
        output = await self.env.run_rollout(
            input=request.input,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
            generation=request.generation,
        )
        return RunRolloutResponse(output=output)

    async def handle_run_group(self, request: RunGroupRequest) -> RunGroupResponse:
        client = await self.resolve_client(request.client_config)
        outputs = await self.env.run_group(
            group_inputs=request.group_inputs,
            client=client,
            model=request.model,
            sampling_args=request.sampling_args,
            max_retries=request.max_retries,
            state_columns=request.state_columns,
            generation=request.generation,
        )
        return RunGroupResponse(outputs=outputs)

    async def process_request(
        self,
        client_id: bytes,
        request_id_bytes: bytes,
        payload_bytes: bytes,
    ) -> None:
        request_id = request_id_bytes.decode()
        response: BaseResponse

        async def send_error_response(error: str) -> None:
            """Serialize and send an error response. Best-effort."""
            try:
                response_bytes = cast(
                    bytes,
                    msgpack.packb(
                        BaseResponse(success=False, error=error).model_dump(
                            mode="python", warnings=False
                        ),
                        default=msgpack_encoder,
                        use_bin_type=True,
                    ),
                )
                await self.response_socket.send_multipart(
                    [client_id, request_id.encode(), response_bytes]
                )
            except Exception:
                pass

        try:
            raw = await asyncio.to_thread(msgpack.unpackb, payload_bytes, raw=False)
            request_type = raw.get("request_type")
            request_id = raw.get("request_id", request_id)

            if request_type == "run_rollout":
                request = await asyncio.to_thread(RunRolloutRequest.model_validate, raw)
                response = await self.handle_run_rollout(request)
            elif request_type == "run_group":
                request = await asyncio.to_thread(RunGroupRequest.model_validate, raw)
                response = await self.handle_run_group(request)
            else:
                self.logger.warning(f"Unknown request type: {request_type}")
                response = BaseResponse(
                    success=False, error=f"Unknown request type: {request_type}"
                )

        except asyncio.CancelledError:
            if self.shutting_down:
                return
            # shield prevents the still-set cancellation flag from killing the await
            await asyncio.shield(send_error_response("Request was cancelled"))
            return

        except Exception as e:
            self.logger.error(
                f"Error processing request {request_id}: {e}", exc_info=True
            )
            await send_error_response(repr(e))
            return

        # Detach bulk routed_experts blobs onto an ordered attachment list, pack
        # the lightened control body on the dedicated serializer pool, and send
        # [client_id, request_id, control, *attachments] (issue #76, PR1). The
        # whole extract -> pack -> send span runs under the byte-budget gate with
        # a try/finally release so a cancel storm can't leak credits (G2). The
        # packb thread cannot be cancelled, so its bytes stay accounted until it
        # returns — we release only after send (or after a send failure).
        try:
            control_payload = response.model_dump(mode="python", warnings=False)
            _, attachments = detach_routed_attachments(control_payload)
        except Exception as e:
            self.logger.error(
                f"Failed to detach routed_experts for {request_id}: {e}",
                exc_info=True,
            )
            await send_error_response(f"Response detach failed: {repr(e)}")
            return

        attachment_bytes = sum(len(a) for a in attachments)
        charge = await self.byte_budget.acquire(attachment_bytes)
        try:
            try:
                control_bytes = await asyncio.get_running_loop().run_in_executor(
                    self.ser_executor,
                    lambda: cast(
                        bytes,
                        msgpack.packb(
                            control_payload,
                            default=msgpack_encoder,
                            use_bin_type=True,
                        ),
                    ),
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to serialize response for {request_id}: {e}",
                    exc_info=True,
                )
                await send_error_response(f"Response serialization failed: {repr(e)}")
                return

            try:
                await self.response_socket.send_multipart(
                    [client_id, request_id.encode(), control_bytes, *attachments]
                )
            except zmq.ZMQError as e:
                self.logger.warning(f"Failed to send response for {request_id[:7]}: {e}")
        finally:
            # Synchronous release in the finally — G2: cannot be interrupted by a
            # pending cancellation, so credits always return (no deadlock).
            self.byte_budget.release(charge)

    async def stats_loop(self, interval: float = 10.0) -> None:
        """Loop to push worker stats to the router."""
        while True:
            await asyncio.sleep(interval)

            stats = EnvWorkerStats(
                worker_id=self.worker_id,
                timestamp=time.time(),
                active_tasks=len(self.active_tasks),
                lag=EventLoopLagStats.from_monitor(self.lag_monitor),
            )

            try:
                data = msgpack.packb(
                    stats.model_dump(mode="python"),
                    default=msgpack_encoder,
                    use_bin_type=True,
                )
                await self.stats_socket.send(data, zmq.NOBLOCK)
            except zmq.Again:
                pass  # best-effort

            self._reclaim_memory()

    def _reclaim_memory(self) -> None:
        """Trim each arena's contiguous top free space at the stats cadence (a
        backstop for small-allocation slack, NOT the primary reclaim).

        The env-worker analog of the orchestrator's per-step trim — these are
        separate child processes the orchestrator's malloc_trim can't reach.
        NOTE: malloc_trim only returns each arena's *contiguous top* free space.
        Under sustained inflight an arena almost always has a live block pinning
        its top, so once many per-thread arenas exist trim reclaims little — it is
        NOT sufficient to prevent the RSS ratchet (confirmed by repro). The actual
        page-return mechanism for the large routed_experts buffers is the
        M_MMAP_THRESHOLD policy set in ``run_worker`` (those allocs go through
        mmap -> munmap on free). This trim stays as a cheap backstop for
        small-allocation arena slack.

        We do NOT force a full ``gc.collect()`` here. A separate, already-fixed
        retention bug (the live-trajectory registry and the interception-server
        body store) was an acyclic tree freed by refcounting the moment its sole
        holder is popped (see ``Runtime.cleanup_rollout`` / ``Endpoint.discard_request``);
        a forced full collection reclaims nothing extra and a synchronous
        collect over a large heap stalls the event loop past the worker
        heartbeat timeout under load. Automatic generational gc stays enabled at
        the default threshold for the rare genuine cycle."""
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError) as e:
            self.logger.debug(f"malloc_trim(0) unavailable: {e}")

    async def serve(self, stop_event: asyncio.Event | None = None) -> None:
        """Main worker loop."""
        self.logger.info(f"Starting worker {self.worker_name}")

        # Freeze the permanent startup set so every future young-gen scan stays
        # cheap. The env-worker host-RAM ramp was NOT cyclic garbage: it was
        # unbounded *retention* (the live-trajectory registry + interception body
        # store, both acyclic and refcount-freed once their holder is popped --
        # fixed in cleanup_rollout / discard_request). So we leave automatic
        # generational gc at the default threshold (cheap, catches the rare real
        # cycle) and do the one-time freeze below to move long-lived init objects
        # out of every future scan. We do NOT force a full gc.collect() per stats
        # tick -- it reclaims nothing extra here and stalls the loop past the
        # worker heartbeat under load (see _reclaim_memory).
        gc.collect()
        gc.freeze()

        lag_task = asyncio.create_task(self.lag_monitor.run())
        stats_task = asyncio.create_task(self.stats_loop())

        poller = zmq.asyncio.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                try:
                    events = dict(await poller.poll(timeout=100))
                    if self.pull_socket not in events:
                        continue

                    frames = await self.pull_socket.recv_multipart()
                    if len(frames) != 3:
                        self.logger.warning(
                            f"Invalid message: expected 3 frames, got {len(frames)}"
                        )
                        continue

                    raw_client_id, raw_request_id, raw_payload = frames
                    request_id = raw_request_id.decode()

                    if not raw_payload:
                        # Cancel signal
                        task = self.active_tasks.get(request_id)
                        if task is not None:
                            task.cancel()
                        continue

                    task = asyncio.create_task(
                        self.process_request(raw_client_id, raw_request_id, raw_payload)
                    )
                    self.active_tasks[request_id] = task

                    def cleanup_task(task: asyncio.Task, request_id: str) -> None:
                        if self.active_tasks.get(request_id) is task:
                            self.active_tasks.pop(request_id, None)

                    task.add_done_callback(
                        lambda t, rid=request_id: cleanup_task(t, rid)
                    )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in serve loop: {e}", exc_info=True)
        finally:
            poller.unregister(self.pull_socket)
            for t in (stats_task, lag_task):
                t.cancel()
            await asyncio.gather(stats_task, lag_task, return_exceptions=True)

    async def close(self) -> None:
        self.shutting_down = True
        if self.active_tasks:
            tasks = list(self.active_tasks.values())
            self.logger.info(
                f"Cancelling {len(tasks)} active tasks on worker {self.worker_name}"
            )
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

        for client in self.clients.values():
            await client.close()
        self.clients.clear()

        await self.env._teardown()

        self.ser_executor.shutdown(wait=False, cancel_futures=True)

        self.pull_socket.close()
        self.response_socket.close()
        self.stats_socket.close()
        self.ctx.term()

        self.logger.info(f"Shut down worker {self.worker_name}")

    async def run(self) -> None:
        if self.death_pipe is not None:
            monitor_death_pipe(self.death_pipe)

        from verifiers.utils.thread_utils import (
            install_default_executor,
            scale_executors,
        )

        # Scale the default executor BEFORE install_default_executor so the
        # event loop picks up a properly-sized pool. Python's default
        # `min(32, cpu_count+4)` caps to_thread; with ~256 concurrent rollouts
        # per worker each calling asyncio.to_thread(parse_response_tokens),
        # the wait queue bottlenecks the threaded path. 512 gives 2x headroom.
        scale_executors(concurrency=512)
        install_default_executor()

        stop_event = asyncio.Event()

        def signal_handler(sig, _frame):
            stop_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            await self.serve(stop_event=stop_event)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            await self.close()

    @staticmethod
    def get_log_file(log_dir: str, worker_id: int) -> Path:
        """Return the log file path for a given worker."""
        return Path(log_dir) / f"env_worker_{worker_id}.log"

    @classmethod
    def run_worker(cls, *args, **kwargs) -> None:
        # Route large transient allocations (the ~3.4 MB msgpack-packed routed_experts
        # payload per rollout) through mmap, so free() -> munmap() hands them straight
        # back to the OS instead of high-water-marking per-thread glibc arenas that
        # malloc_trim cannot reclaim under sustained inflight (an arena's top stays
        # pinned by live work). Pinning the threshold also disables glibc's dynamic
        # auto-raise. Must precede the 512-thread executor and any rollout allocation.
        # Root cause: per-thread-arena fragmentation. Deeper "bulk tensors off the
        # message bus" cleanup tracked in joanvelja/prime-rl#76.
        _M_MMAP_THRESHOLD = -3
        _MMAP_THRESHOLD_BYTES = 1 << 20  # 1 MiB: below the routed_experts payload, above small control allocs
        rc = ctypes.CDLL("libc.so.6").mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD_BYTES)
        if rc != 1:
            raise RuntimeError(
                f"mallopt(M_MMAP_THRESHOLD, {_MMAP_THRESHOLD_BYTES}) returned {rc}; "
                "env-worker arena fix inactive — refusing to start rather than silently OOM later"
            )
        try:
            import uvloop

            uvloop.install()
        except ImportError:
            pass
        worker = cls(*args, **kwargs)
        asyncio.run(worker.run())
