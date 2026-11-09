"""Regression coverage for state and budgets scoped to one DAG execution."""

import asyncio
import threading

import pytest

from mmar_carl import ReasoningChain, ReasoningContext
from mmar_carl import executor
from mmar_carl.models.config import StepCache, ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


def tool_step(number, name, *, dependencies=None, cache=None):
    return ToolStepDescription(
        number=number, title=name, dependencies=dependencies or [], cache=cache,
        config=ToolStepConfig(tool_name=name),
    )


def context(**kwargs):
    return ReasoningContext(outer_context="same input", api=None, model="stub", **kwargs)


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_share_step_cache():
    started = asyncio.Event()
    resume = asyncio.Event()
    first, second = context(), context()

    async def wait():
        started.set()
        await resume.wait()
        return "ready"

    first.register_tool("gate", lambda: "ready")
    first.register_tool("answer", lambda: "first")
    second.register_tool("gate", wait)
    second.register_tool("answer", lambda: "second")
    chain = ReasoningChain(steps=[
        tool_step(1, "gate"), tool_step(2, "answer", dependencies=[1], cache=StepCache()),
    ])
    pending = asyncio.create_task(chain.execute_async(second))
    try:
        await asyncio.wait_for(started.wait(), 2)
        result = await chain.execute_async(first)
        assert result.step_results[-1].result == "first"
    finally:
        resume.set()
        result = await asyncio.wait_for(pending, 2)
    assert result.success
    assert result.step_results[-1].result == "second"


@pytest.mark.asyncio
async def test_last_step_uses_remaining_chain_budget():
    ctx = context()
    completed = []

    async def slow():
        await asyncio.sleep(0.12)
        completed.append(True)
        return "done"

    ctx.register_tool("slow", slow)
    chain = ReasoningChain(
        steps=[tool_step(1, "slow"), tool_step(2, "slow", dependencies=[1])],
        timeout=0.20,
    )
    with pytest.raises(TimeoutError, match="Chain execution timed out"):
        await chain.execute_async(ctx)
    assert completed == [True]


@pytest.mark.asyncio
async def test_expired_active_step_releases_paused_queue_without_fail_fast():
    ctx = context()
    entered = asyncio.Event()

    async def blocked():
        entered.set()
        await asyncio.Event().wait()

    ctx.register_tool("blocked", blocked)
    ctx.register_tool("queued", lambda: pytest.fail("expired queue started"))
    chain = ReasoningChain(
        steps=[tool_step(1, "blocked"), tool_step(2, "queued")],
        max_workers=1,
        timeout=0.10,
    )
    task = asyncio.create_task(chain.execute_async(ctx))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        ctx.request_pause()
        with pytest.raises(TimeoutError, match="Chain execution timed out"):
            await asyncio.wait_for(task, 2)
    finally:
        ctx.clear_pause()


@pytest.mark.asyncio
async def test_run_without_replan_does_not_deepcopy_memory():
    ctx = context(memory={"default": {"lock": threading.Lock()}})
    ctx.register_tool("noop", lambda: "done")
    result = await ReasoningChain(steps=[tool_step(1, "noop")]).execute_async(ctx)
    assert result.success


@pytest.mark.asyncio
async def test_memory_size_reuses_unchanged_scalar_payload(monkeypatch):
    ctx = context(memory={"default": {"payload": "x" * 1024}})
    ctx.register_tool("noop", lambda: "done")
    count = 0
    estimate = executor._estimate_memory_bytes

    def count_estimate(memory):
        nonlocal count
        count += 1
        return estimate(memory)

    monkeypatch.setattr(executor, "_estimate_memory_bytes", count_estimate)
    chain = ReasoningChain(steps=[tool_step(1, "noop"), tool_step(2, "noop", dependencies=[1])])
    result = await chain.execute_async(ctx)
    sizes = [item.profiling["memory_bytes_after"] for item in result.step_results]
    assert sizes == [estimate(ctx.memory)] * 2
    assert count == 1
