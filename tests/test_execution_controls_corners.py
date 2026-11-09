"""Offline execution-control regressions with actual tools and event barriers."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
import pytest_asyncio

from mmar_carl import LLMClientBase, ReasoningChain, ReasoningContext
from mmar_carl.executor import ExecutionCancelledError
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


class _NoProvider(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        raise AssertionError("These local tool tests must never call a provider")

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


@pytest_asyncio.fixture
async def context():
    context = ReasoningContext(outer_context="synthetic test", api=_NoProvider())
    yield context
    await context.close()


def _step(number: int, deps=()):
    return ToolStepDescription(
        number=number, title=f"Step {number}", dependencies=list(deps),
        config=ToolStepConfig(tool_name=f"tool_{number}", timeout=3),
    )


async def _cleanup(task):
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_ten_tools_actually_overlap_and_join_once(context, fail_fast):
    entered, release = asyncio.Event(), asyncio.Event()
    active = peak = 0
    calls = []

    def tool(number):
        async def run():
            nonlocal active, peak
            calls.append(number)
            active += 1
            peak = max(peak, active)
            if active == 10:
                entered.set()
            try:
                await release.wait()
                return {"number": number, "value": number * 10}
            finally:
                active -= 1
        return run

    async def join():
        assert active == 0
        calls.append(11)
        return "joined all ten"

    for number in range(1, 11):
        context.register_tool(f"tool_{number}", tool(number))
    context.register_tool("tool_11", join)
    chain = ReasoningChain(
        steps=[*[_step(n) for n in range(1, 11)], _step(11, range(1, 11))],
        max_workers=10, fail_fast=fail_fast,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert active == peak == 10
        assert calls == list(range(1, 11))
        assert not task.done()
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        await _cleanup(task)
    assert result.success is True
    assert Counter(calls) == Counter(range(1, 12))
    assert len(result.step_results) == 11
    assert all(step.success and not step.skipped for step in result.step_results)


@pytest.mark.asyncio
async def test_middle_failure_skips_queue_and_join_but_drains_nine_siblings(context):
    entered, fail, release, failed = (asyncio.Event() for _ in range(4))
    calls, completed = [], []
    active = peak = 0

    async def seed():
        calls.append(1)
        return "seed"

    def tool(number):
        async def run():
            nonlocal active, peak
            calls.append(number)
            active += 1
            peak = max(peak, active)
            if active == 10:
                entered.set()
            try:
                if number == 7:
                    await fail.wait()
                    raise RuntimeError("middle step disconnected")
                await release.wait()
                return f"real result {number}"
            finally:
                active -= 1
        return run

    def complete(step):
        completed.append(step.step_number)
        if step.step_number == 7:
            failed.set()

    context.on_step_complete = complete
    context.register_tool("tool_1", seed)
    for number in range(2, 15):
        context.register_tool(f"tool_{number}", tool(number))
    chain = ReasoningChain(
        steps=[_step(1), *[_step(n, [1]) for n in range(2, 14)], _step(14, range(2, 14))],
        max_workers=10, fail_fast=True,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert active == peak == 10
        fail.set()
        await asyncio.wait_for(failed.wait(), timeout=2)
        await _wait_until(lambda: 12 in completed and 13 in completed)
        assert calls == list(range(1, 12))
        assert active == 9
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        fail.set()
        release.set()
        await _cleanup(task)
    outcomes = {step.step_number: step for step in result.step_results}
    assert result.success is False
    assert outcomes[7].success is False and not outcomes[7].skipped
    assert {n for n, step in outcomes.items() if step.skipped} == {12, 13, 14}
    assert {n for n, step in outcomes.items() if step.success} == set(range(1, 12)) - {7}
    assert Counter(completed) == Counter(range(1, 15))
    assert active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_pause_ten_active_tools_stops_queued_work_and_resumes_exactly_once(context, fail_fast):
    entered, release, drained = (asyncio.Event() for _ in range(3))
    calls = []
    active = 0

    def tool(number):
        async def run():
            nonlocal active
            calls.append(number)
            active += 1
            if active == 10:
                entered.set()
            try:
                if number <= 10:
                    await release.wait()
                return {"step": number}
            finally:
                active -= 1
                if active == 0:
                    drained.set()
        return run

    for number in range(1, 14):
        context.register_tool(f"tool_{number}", tool(number))
    chain = ReasoningChain(
        steps=[*[_step(n) for n in range(1, 13)], _step(13, range(1, 13))],
        max_workers=10, fail_fast=fail_fast,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        context.request_pause()
        assert not context.is_paused(), "running tools must not be reported as paused"
        release.set()
        await asyncio.wait_for(drained.wait(), timeout=2)
        await _wait_until(context.is_paused)
        await asyncio.sleep(0.02)
        assert calls == list(range(1, 11))
        assert not task.done()
        context.clear_pause()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release.set()
        context.clear_pause()
        await _cleanup(task)
    assert result.success is True
    assert Counter(calls) == Counter(range(1, 14))
    assert not context.is_paused()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resume", "cancel", "request_cancellation"])
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_pause_at_batch_boundary_then_resume_or_cancel(context, action, fail_fast):
    calls = []

    async def run():
        calls.append(len(calls) + 1)
        return "finished"

    def complete(step):
        if step.step_number == 1:
            context.request_pause()

    context.register_tool("tool_1", run)
    context.register_tool("tool_2", run)
    context.on_step_complete = complete
    chain = ReasoningChain(steps=[_step(1), _step(2, [1])], fail_fast=fail_fast)
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await _wait_until(context.is_paused)
        assert calls == [1]
        assert not task.done()
        if action == "resume":
            context.clear_pause()
            result = await asyncio.wait_for(task, timeout=2)
            assert result.success and calls == [1, 2]
        else:
            getattr(context, action)()
            with pytest.raises(ExecutionCancelledError) as error:
                await asyncio.wait_for(task, timeout=2)
            assert error.value.result.metadata["cancelled"] is True
            assert calls == [1]
    finally:
        context.clear_pause()
        await _cleanup(task)


@pytest.mark.asyncio
async def test_failure_wakes_paused_queue_without_requiring_resume(context):
    entered, release_success, release_failure, first_done = (asyncio.Event() for _ in range(4))
    calls = []

    async def first():
        calls.append(1)
        await release_success.wait()
        return "success"

    async def failing():
        calls.append(2)
        entered.set()
        await release_failure.wait()
        raise RuntimeError("failure while queued work is paused")

    async def forbidden():
        calls.append(3)
        raise AssertionError("queued tool must not start")

    context.on_step_complete = lambda step: first_done.set() if step.step_number == 1 else None
    for n, tool in enumerate((first, failing, forbidden), 1):
        context.register_tool(f"tool_{n}", tool)
    chain = ReasoningChain(steps=[_step(n) for n in range(1, 4)], max_workers=2, fail_fast=True)
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        context.request_pause()
        release_success.set()
        await asyncio.wait_for(first_done.wait(), timeout=2)
        await asyncio.sleep(0.02)
        assert calls == [1, 2]
        assert not context.is_paused(), "second tool is still active"
        release_failure.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release_success.set()
        release_failure.set()
        context.clear_pause()
        await _cleanup(task)
    assert result.success is False
    assert result.step_results[-1].skipped is True
    assert calls == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_cancel_while_queue_paused_does_not_start_remaining_tools(context, fail_fast):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def first():
        calls.append(1)
        entered.set()
        await release.wait()
        return "completed before cancellation"

    async def forbidden():
        calls.append(2)
        raise AssertionError("cancelled queue must not start")

    context.register_tool("tool_1", first)
    context.register_tool("tool_2", forbidden)
    chain = ReasoningChain(steps=[_step(1), _step(2)], max_workers=1, fail_fast=fail_fast)
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        context.request_pause()
        release.set()
        await _wait_until(context.is_paused)
        context.cancel()
        with pytest.raises(ExecutionCancelledError) as error:
            await asyncio.wait_for(task, timeout=2)
        assert error.value.result.metadata["cancelled"] is True
        assert calls == [1]
        assert context.snapshot().completed_step_numbers == [1]
    finally:
        release.set()
        context.clear_pause()
        await _cleanup(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_idle_pause_time_does_not_exhaust_chain_timeout(context, queued, fail_fast):
    calls = []

    async def run():
        calls.append(len(calls) + 1)
        return "done"

    def complete(step):
        if step.step_number == 1:
            context.request_pause()

    for number in range(1, 4):
        context.register_tool(f"tool_{number}", run)
    context.on_step_complete = complete
    chain = ReasoningChain(
        steps=[_step(1), _step(2, [] if queued else [1]), _step(3, [2])],
        max_workers=1, fail_fast=fail_fast, timeout=0.15,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await _wait_until(context.is_paused)
        await asyncio.sleep(0.2)
        assert calls == [1]
        context.clear_pause()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        context.clear_pause()
        await _cleanup(task)
    assert result.success is True
    assert calls == [1, 2, 3]
    assert context.paused_duration() >= 0.2


@pytest.mark.asyncio
async def test_pause_does_not_extend_an_inflight_provider_or_tool_timeout(context):
    entered, never = asyncio.Event(), asyncio.Event()
    calls = []

    async def blocked():
        calls.append(1)
        entered.set()
        await never.wait()
        return "unreachable"

    async def forbidden():
        calls.append(2)
        raise AssertionError("timed-out work must stop the queue")

    context.register_tool("tool_1", blocked)
    context.register_tool("tool_2", forbidden)
    chain = ReasoningChain(
        steps=[_step(1), _step(2)], max_workers=1, fail_fast=True, timeout=0.05,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        context.request_pause()
        assert not context.is_paused()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        context.clear_pause()
        await _cleanup(task)
    assert not result.success
    assert "timed out" in result.step_results[0].error_message.lower()
    assert result.step_results[1].skipped
    assert calls == [1]
    assert context.paused_duration() < 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_external_task_cancellation_drains_paused_waiters(context, fail_fast):
    async def forbidden():
        raise AssertionError("paused chain must not start tools")

    context.register_tool("tool_1", forbidden)
    context.request_pause()
    task = asyncio.create_task(ReasoningChain(steps=[_step(1)], fail_fast=fail_fast).execute_async(context))
    try:
        await _wait_until(context.is_paused)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not context.is_paused()
        assert context._pause_token.waiters == 0
        assert context._pause_token.active_steps == 0
    finally:
        context.clear_pause()
        await _cleanup(task)


@pytest.mark.asyncio
async def test_expired_chain_deadline_does_not_wait_for_resume(context, monkeypatch):
    import time
    from types import SimpleNamespace

    from mmar_carl import executor

    now = 0.0
    monkeypatch.setattr(executor, "time", SimpleNamespace(monotonic=lambda: now, time=time.time))
    started = []

    def pause_after_deadline(*_args):
        nonlocal now
        now = 2.0
        context.request_pause()

    context.on_progress = pause_after_deadline
    context.on_step_start = lambda number, _title: started.append(number)
    chain = ReasoningChain(steps=[_step(1)], timeout=1.0)
    task = asyncio.create_task(chain.execute_async(context))
    try:
        done, _ = await asyncio.wait([task], timeout=0.2)
        assert task in done
        with pytest.raises(TimeoutError, match="Completed 0/1 steps"):
            await task
    finally:
        context.clear_pause()
        await _cleanup(task)

    assert not started
