"""Cancel-safety tests for the env_worker byte-budget gate (issue #76, G2).

``_ByteBudget`` is the credit gate spanning extract->pack->send. The load-bearing
property is G2: a cancel storm must never leak credits into a deadlock. The class
is pure async (no sockets / env), so it is unit-testable without mocking.
"""

import asyncio

from verifiers.serve.server.env_worker import _ByteBudget


async def test_byte_budget_basic_acquire_release():
    budget = _ByteBudget(100)
    charge = await budget.acquire(40)
    assert charge == 40
    assert budget.available == 60
    budget.release(charge)
    assert budget.available == 100


async def test_byte_budget_oversized_borrows_whole_pool():
    """A message larger than the pool waits for an empty pool then borrows it all
    (charge == capacity) — it can never wedge."""
    budget = _ByteBudget(100)
    charge = await budget.acquire(250)
    assert charge == 100
    assert budget.available == 0
    budget.release(charge)
    assert budget.available == 100


async def test_byte_budget_backpressure_then_grant():
    budget = _ByteBudget(100)
    first = await budget.acquire(80)
    # Second can't fit (80 + 50 > 100) -> blocks.
    waiter = asyncio.create_task(budget.acquire(50))
    await asyncio.sleep(0)
    assert not waiter.done()
    # Releasing the first frees enough credit -> waiter granted.
    budget.release(first)
    granted = await waiter
    assert granted == 50
    assert budget.available == 50


async def test_byte_budget_cancel_storm_no_credit_leak():
    """T3 (G2): a storm of blocked waiters all cancelled mid-pack must return the
    pool to full — no leaked credit, no deadlock."""
    budget = _ByteBudget(100)
    held = await budget.acquire(100)  # pool now empty; all newcomers block.

    waiters = [asyncio.create_task(budget.acquire(10)) for _ in range(20)]
    await asyncio.sleep(0)
    assert all(not w.done() for w in waiters)

    # Cancel the entire storm.
    for w in waiters:
        w.cancel()
    for w in waiters:
        try:
            await w
        except asyncio.CancelledError:
            pass

    # Release the original holder; pool must come back to exactly full.
    budget.release(held)
    assert budget.available == 100

    # And the gate is still live: a fresh acquire succeeds immediately.
    again = await budget.acquire(100)
    assert again == 100
    assert budget.available == 0


async def test_byte_budget_grant_concurrent_with_cancel_no_leak():
    """If a waiter is granted in the same tick it is cancelled, the charge debited
    on its behalf must be returned (acquire's finally path), not leaked."""
    budget = _ByteBudget(100)
    held = await budget.acquire(100)

    waiter = asyncio.create_task(budget.acquire(60))
    await asyncio.sleep(0)
    assert not waiter.done()

    # Release (which synchronously grants the waiter via set_result) and cancel in
    # the same tick before the waiter coroutine resumes -> the grant raced a cancel.
    budget.release(held)
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass

    # Whether the cancel won (request dropped) or the grant won (charge returned by
    # acquire's finally), the pool must net back to full.
    assert budget.available == 100
