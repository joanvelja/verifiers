"""Shared routed_experts attachment traversal (issue #76, PR1).

The bulk ``routed_experts`` tensor used to ride *inside* the control-plane
msgpack body. PR1 lifts it onto raw uint8 ZMQ multipart frames: the env-worker
``detach``es each blob into an ordered attachment list (replacing the in-body
``"data"`` with a tiny descriptor), and the zmq env-client ``reattach``es the
frames after unpacking the control body.

This module is the ONE shared traversal used by both the producer
(``env_worker``) and the consumer (``zmq_env_client``). Keeping a single walk
is guardrail G3: detach assigns dense ``frame`` ordinals in a deterministic
walk, and reattach looks up ``frames[frame]`` by that ordinal — it never
re-derives the order. Same walk, same order, both ends.

Wire/descriptor contract (per routed_experts site, inside the msgpack control):

    {
        "frame":    int,          # 0-based ordinal into the attachment list
        "shape":    [int, ...],   # tensor shape; shape[0] may be 0
        "dtype":    str,          # numpy dtype name, e.g. "uint8" (G5: explicit)
        "start":    int,          # token offset, carried verbatim (G6 downstream)
        "encoding": "raw",        # only "raw" is valid (G5: fail-loud otherwise)
    }

The descriptor has NO ``"data"`` key — the bytes live in the attachment list.

Dict path walked (post ``model_dump(mode="python")``):
  - RunRolloutResponse -> ``payload["output"]`` (single RolloutOutput dict | None)
  - RunGroupResponse   -> ``payload["outputs"]`` (list[RolloutOutput dict] | None)
  - RolloutOutput      -> ``["trajectory"][i]["tokens"]["routed_experts"]``
    where each routed_experts is a ``RoutedExpertsPayload`` dict
    (``{"data", "shape", "start"}``) or absent/None.
"""

from __future__ import annotations

import math
from typing import Any

# Routed-experts blobs are emitted as uint8 by the vLLM capturer; the
# descriptor carries the dtype explicitly (G5) so np.frombuffer is
# parameterized downstream. This is the documented default stamped when a
# site dict does not already carry its own "dtype" key — NOT a magic number
# baked into the length check (that reads desc["dtype"]).
DEFAULT_ROUTED_DTYPE = "uint8"

# The only valid encoding on the wire. msgpack ``bin`` is encoding-blind, so we
# carry this explicitly and fail loud on anything else (G5).
RAW_ENCODING = "raw"


def _itemsize(dtype: str) -> int:
    import numpy as np

    return np.dtype(dtype).itemsize


def _iter_routed_sites(control: dict) -> Any:
    """Yield every ``routed_experts`` site dict in a single deterministic walk.

    Yields the *container* ``tokens`` dict and the key ``"routed_experts"`` so
    callers can read-or-rebind the site in place. Order is fixed: outputs in
    list order, then trajectory steps in order. Sites that are absent or
    ``None`` are skipped (they carry no bytes).
    """
    if "output" in control:
        outputs = [control["output"]]
    elif "outputs" in control:
        outputs = control["outputs"] or []
    else:
        # Error responses (BaseResponse) carry neither key and no attachments.
        return

    for output in outputs:
        if output is None:
            continue
        trajectory = output.get("trajectory")
        if not trajectory:
            continue
        for step in trajectory:
            tokens = step.get("tokens")
            if tokens is None:
                continue
            site = tokens.get("routed_experts")
            if site is None:
                continue
            yield tokens


def detach_routed_attachments(control_payload: dict) -> tuple[dict, list[bytes]]:
    """Pop every routed_experts blob's raw bytes into an ordered list.

    Walks ``control_payload`` once, and for each routed_experts site pops the
    ``"data"`` bytes into ``attachments`` and replaces the site with a
    descriptor ``{frame, shape, dtype, start, encoding}`` (no ``"data"``).
    ``frame`` ordinals are dense ``0..K-1`` in traversal order (G3).

    Mutates ``control_payload`` in place (the model_dump'd dict is owned by the
    caller and discarded after packing) and returns it for convenience.
    """
    attachments: list[bytes] = []
    for tokens in _iter_routed_sites(control_payload):
        site = tokens["routed_experts"]
        data = site["data"]
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        elif not isinstance(data, bytes):
            raise TypeError(
                f"routed_experts data must be raw bytes by detach time, got {type(data).__name__}"
            )

        frame = len(attachments)
        attachments.append(data)
        tokens["routed_experts"] = {
            "frame": frame,
            "shape": list(site["shape"]),
            "dtype": site.get("dtype", DEFAULT_ROUTED_DTYPE),
            "start": site["start"],
            "encoding": RAW_ENCODING,
        }

    return control_payload, attachments


def reattach_routed_attachments(control: dict, attachments: list[bytes]) -> dict:
    """Inverse of ``detach``: rebind each site's ``"data"`` from the frames.

    Walks ``control`` the same way, and for each site sets
    ``site["data"] = attachments[desc["frame"]]``. Asserts (fail-loud):
      - ``encoding == "raw"`` (G5)
      - ``len(frame) == prod(shape) * itemsize(dtype)`` (G1/G5) — this is the
        ONLY length check; it NEVER cross-checks token/completion counts
        (``align_routed_experts`` is the sole dim-0 arbiter).
      - every frame ordinal is in range and each attachment is consumed exactly
        once (catches dropped/short/extra frames — e.g. a router that forwarded
        too few frames).

    Mutates ``control`` in place and returns it.
    """
    consumed = [False] * len(attachments)

    for tokens in _iter_routed_sites(control):
        desc = tokens["routed_experts"]

        encoding = desc.get("encoding")
        if encoding != RAW_ENCODING:
            raise ValueError(
                f"routed_experts descriptor has encoding={encoding!r}, expected {RAW_ENCODING!r}"
            )

        frame = desc["frame"]
        if not (0 <= frame < len(attachments)):
            raise ValueError(
                f"routed_experts frame ordinal {frame} out of range for {len(attachments)} attachment(s)"
            )
        if consumed[frame]:
            raise ValueError(f"routed_experts frame ordinal {frame} referenced more than once")

        data = attachments[frame]
        shape = desc["shape"]
        dtype = desc["dtype"]
        expected = math.prod(shape) * _itemsize(dtype)
        if len(data) != expected:
            raise ValueError(
                f"routed_experts frame {frame} length {len(data)} != prod(shape={shape})"
                f" * itemsize({dtype})={expected}"
            )

        consumed[frame] = True
        tokens["routed_experts"] = {
            "data": data,
            "shape": shape,
            "dtype": dtype,
            "start": desc["start"],
        }

    if not all(consumed):
        missing = [i for i, c in enumerate(consumed) if not c]
        raise ValueError(
            f"received {len(attachments)} routed_experts frame(s) but {len(missing)} were "
            f"never referenced by a descriptor (ordinals {missing})"
        )

    return control
