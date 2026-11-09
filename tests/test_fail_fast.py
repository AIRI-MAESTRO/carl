"""Provider-free fail-fast contracts exercised through the actual CARL scheduler.

The API object deliberately rejects every LLM call. Real registered async tools,
event barriers and the normal step-recovery implementation drive these cases.
"""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from typing import Any

import pytest
import pytest_asyncio

from mmar_carl import (
    ChainBuilder,
    LLMClientBase,
    LoopConfig,
    ReasoningChain,
    ReasoningContext,
    ReplanPolicy,
    RuleBasedReplanCheckerConfig,
)
from mmar_carl.executor import DAGExecutor, ExecutionCancelledError
from mmar_carl.models.config import ToolErrorRecovery, ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


class _NoProvider(LLMClientBase):
    def __init__(self) -> None:
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        self.calls += 1
        raise AssertionError("Tool-only fail-fast tests must not call a provider")

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


@pytest_asyncio.fixture
async def contexts():
    created: list[ReasoningContext] = []
    api = _NoProvider()

    def make(**kwargs: Any) -> ReasoningContext:
        context = ReasoningContext(outer_context="synthetic local task", api=api, **kwargs)
        created.append(context)
        return context

    yield make
    for context in created:
        await context.close()
    assert api.calls == 0


def _step(number: int, name: str, deps=(), **kwargs: Any) -> ToolStepDescription:
    recovery = kwargs.pop("recovery", None)
    return ToolStepDescription(
        number=number,
        title=f"Step {number}: {name}",
        dependencies=list(deps),
        config=ToolStepConfig(
            tool_name=name,
            timeout=2.0,
            error_recovery=recovery,
        ),
        **kwargs,
    )


async def _failed() -> str:
    raise RuntimeError("synthetic terminal tool failure")


def _outcomes(result):
    return {step.step_number: step for step in result.step_results}


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("typed", [True, False])
def test_fail_fast_survives_chain_serialization(enabled, typed):
    chain = ReasoningChain(steps=[_step(1, "local")], fail_fast=enabled)
    payload = chain.to_dict()
    assert payload["fail_fast"] is enabled
    restored = ReasoningChain.from_dict(copy.deepcopy(payload), use_typed_steps=typed)
    assert restored.fail_fast is enabled
    assert restored.executor.fail_fast is enabled
    assert restored.to_dict()["fail_fast"] is enabled
    assert ReasoningChain.from_json(chain.to_json()).fail_fast is enabled


def test_absent_fail_fast_retains_legacy_false_and_builder_is_explicit():
    payload = ReasoningChain(steps=[_step(1, "local")]).to_dict()
    payload.pop("fail_fast", None)
    assert ReasoningChain.from_dict(payload).fail_fast is False
    assert ReasoningChain(steps=[_step(1, "local")]).executor.fail_fast is False
    assert DAGExecutor().fail_fast is False
    builder = ChainBuilder().add_tool_step(1, "Local", "local")
    assert builder.build().fail_fast is False
    assert builder.with_fail_fast().build().fail_fast is True
    assert builder.with_fail_fast(True).with_fail_fast(False).build().fail_fast is False


@pytest.mark.parametrize("invalid", [None, "true", "false", "", 0, 1, 0.0, [], {}])
@pytest.mark.parametrize("surface", ["chain", "executor", "dict", "builder"])
def test_non_boolean_fail_fast_is_rejected(invalid, surface):
    with pytest.raises((TypeError, ValueError), match="fail_fast"):
        if surface == "chain":
            ReasoningChain(steps=[], fail_fast=invalid)
        elif surface == "executor":
            DAGExecutor(fail_fast=invalid)
        elif surface == "dict":
            ReasoningChain.from_dict({"steps": [], "fail_fast": invalid})
        else:
            ChainBuilder().with_fail_fast(invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [True, False, "absent"])
async def test_terminal_failure_blocks_dependents_only_when_enabled(contexts, setting):
    called: list[str] = []
    started: list[int] = []
    completed: list[Any] = []
    chain_completed: list[Any] = []

    async def downstream() -> str:
        called.append("downstream")
        return "downstream result"

    context = contexts(
        on_step_start=lambda number, _title: started.append(number),
        on_step_complete=completed.append,
        on_chain_complete=chain_completed.append,
    )
    context.register_tool("failed", _failed)
    context.register_tool("downstream", downstream)
    kwargs = {} if setting == "absent" else {"fail_fast": setting}
    chain = ReasoningChain(
        steps=[_step(1, "failed"), _step(2, "downstream", [1]), _step(3, "downstream", [2])],
        **kwargs,
    )
    result = await chain.execute_async(context)

    assert result.success is False
    assert len(chain_completed) == 1 and chain_completed[0] is result
    outcomes = _outcomes(result)
    assert outcomes[1].success is False and outcomes[1].skipped is False
    assert Counter(item.step_number for item in completed) == {1: 1, 2: 1, 3: 1}
    assert Counter(item.step_number for item in result.trace.events) == {1: 1, 2: 1, 3: 1}
    if setting is True:
        assert called == []
        assert started == [1]
        for number in (2, 3):
            skipped = outcomes[number]
            assert skipped.skipped is True and skipped.success is False
            assert skipped.result == ""
            assert skipped.execution_time in (None, 0)
            assert not skipped.token_usage
            assert skipped.profiling["fail_fast"] is True
            assert skipped.profiling["failed_step_number"] == 1
        assert result.metadata["fail_fast"] == {
            "enabled": True,
            "triggered": True,
            "failed_step_number": 1,
            "skipped_step_numbers": [2, 3],
        }
        assert result.trace.metadata["fail_fast"] == result.metadata["fail_fast"]
        stats = result.metadata["execution_stats"]
        assert stats["executed_steps"] == 1
        assert stats["failed_steps"] == 1
        assert stats["successful_steps"] == 0
        assert stats["skipped_steps"] == 2
        assert result.get_failed_steps() == [outcomes[1]]
        assert result.get_successful_steps() == []
        assert result.partial_outputs == {}
        assert result.get_partial_final_output() is None
        assert context.snapshot().completed_step_numbers == []
        assert context.is_cancelled() is False
    else:
        assert called == ["downstream", "downstream"]
        assert started == [1, 2, 3]
        assert outcomes[2].success and outcomes[3].success
        assert not any(step.skipped for step in result.step_results)


@pytest.mark.asyncio
async def test_single_worker_does_not_dispatch_queued_independent_roots(contexts):
    calls: list[str] = []

    async def queued() -> str:
        calls.append("queued")
        return "must not run"

    context = contexts()
    context.register_tool("failed", _failed)
    context.register_tool("queued", queued)
    chain = ReasoningChain(
        steps=[_step(1, "failed"), _step(2, "queued"), _step(3, "queued", [2])],
        fail_fast=True,
        max_workers=1,
    )
    result = await chain.execute_async(context)
    assert calls == []
    assert {step.step_number for step in result.get_skipped_steps()} == {2, 3}


@pytest.mark.asyncio
async def test_inflight_sibling_settles_but_queued_roots_and_join_do_not_start(contexts):
    sibling_started = asyncio.Event()
    failure_observed = asyncio.Event()
    release_sibling = asyncio.Event()
    calls: list[str] = []

    async def failed() -> str:
        await sibling_started.wait()
        calls.append("failed")
        raise RuntimeError("synthetic terminal failure")

    async def sibling() -> str:
        calls.append("sibling start")
        sibling_started.set()
        await release_sibling.wait()
        calls.append("sibling finish")
        return "retained independent work"

    async def queued() -> str:
        calls.append("queued")
        return "must not run"

    def completed(result):
        if result.step_number == 1:
            failure_observed.set()

    context = contexts(on_step_complete=completed)
    for name, function in (("failed", failed), ("sibling", sibling), ("queued", queued)):
        context.register_tool(name, function)
    chain = ReasoningChain(
        steps=[
            _step(1, "failed"), _step(2, "sibling"), _step(3, "queued"),
            _step(4, "queued", [1, 2]), _step(5, "queued", [3]),
        ],
        fail_fast=True,
        max_workers=2,
    )
    task = asyncio.create_task(chain.execute_async(context))
    try:
        await asyncio.wait_for(failure_observed.wait(), timeout=2)
        assert "sibling start" in calls and "sibling finish" not in calls
        assert "queued" not in calls
        assert not task.done(), "Already-started sibling must be allowed to settle"
        release_sibling.set()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        release_sibling.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert calls == ["sibling start", "failed", "sibling finish"]
    assert result.success is False
    assert {step.step_number for step in result.get_skipped_steps()} == {3, 4, 5}
    assert result.partial_outputs == {2: "retained independent work"}
    assert context.snapshot().completed_step_numbers == [2]
    assert result.metadata["execution_stats"]["executed_steps"] == 2


@pytest.mark.asyncio
async def test_auto_workers_drain_all_started_roots_but_never_dispatch_join(contexts):
    all_started = asyncio.Event()
    roots: set[int] = set()
    joins: list[int] = []

    def root(number):
        async def run():
            roots.add(number)
            if roots == {1, 2, 3}:
                all_started.set()
            await all_started.wait()
            if number == 1:
                raise RuntimeError("root failed")
            return f"root {number}"
        return run

    async def join():
        joins.append(4)
        return "must not run"

    context = contexts()
    for number in (1, 2, 3):
        context.register_tool(f"root{number}", root(number))
    context.register_tool("join", join)
    chain = ReasoningChain(
        steps=[*[_step(n, f"root{n}") for n in (1, 2, 3)], _step(4, "join", [1, 2, 3])],
        max_workers="auto", fail_fast=True,
    )
    result = await asyncio.wait_for(chain.execute_async(context), timeout=3)
    assert roots == {1, 2, 3} and joins == []
    assert result.success is False
    assert {step.step_number for step in result.get_successful_steps()} == {2, 3}
    assert {step.step_number for step in result.get_skipped_steps()} == {4}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retry_recovers", "fallback_recovers", "retry_exhausted", "fallback_fails"])
async def test_step_recovery_finishes_before_fail_fast_decision(contexts, mode):
    calls = Counter()

    async def primary():
        calls["primary"] += 1
        if mode == "retry_recovers" and calls["primary"] == 3:
            return "recovered primary"
        raise RuntimeError("primary failure")

    async def fallback():
        calls["fallback"] += 1
        if mode == "fallback_fails":
            raise RuntimeError("fallback failure")
        return "recovered fallback"

    async def downstream():
        calls["downstream"] += 1
        return "final result"

    context = contexts()
    for name, function in (("primary", primary), ("fallback", fallback), ("downstream", downstream)):
        context.register_tool(name, function)
    recovery = ToolErrorRecovery(
        retry_max=2, retry_delay=0,
        on_exception="fallback" if mode.startswith("fallback") else None,
    )
    chain = ReasoningChain(
        steps=[_step(1, "primary", recovery=recovery), _step(2, "downstream", [1])],
        fail_fast=True,
    )
    result = await chain.execute_async(context)
    recovered = mode.endswith("recovers")
    assert calls["primary"] == 3
    assert calls["fallback"] == int(mode.startswith("fallback"))
    assert calls["downstream"] == int(recovered)
    assert result.success is recovered
    assert _outcomes(result)[2].skipped is not recovered
    assert result.metadata["fail_fast"]["triggered"] is not recovered


@pytest.mark.asyncio
async def test_terminal_failure_precedes_chain_level_replan(contexts):
    calls = 0

    async def transient():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("replan would retry this")
        return "should not reach chain-level retry"

    context = contexts()
    context.register_tool("transient", transient)
    chain = ReasoningChain(
        steps=[_step(1, "transient")], fail_fast=True,
        replan_policy=ReplanPolicy(checkers=[RuleBasedReplanCheckerConfig()]),
    )
    result = await chain.execute_async(context)
    assert calls == 1
    assert result.success is False
    assert result.replan_events == []
    assert result.metadata["fail_fast"]["triggered"] is True


@pytest.mark.asyncio
async def test_sibling_loop_back_cannot_hide_failure_or_restart_work(contexts):
    calls = 0
    tail_started = asyncio.Event()

    async def failed():
        await tail_started.wait()
        raise RuntimeError("failure must survive sibling loop-back")

    async def loop_tail():
        nonlocal calls
        calls += 1
        tail_started.set()
        return "loop tail completed"

    context = contexts()
    context.register_tool("failed", failed)
    context.register_tool("loop_tail", loop_tail)
    chain = ReasoningChain(
        steps=[
            _step(1, "failed"),
            _step(2, "loop_tail", loop_back_to=1, loop_config=LoopConfig(max_iterations=2)),
        ],
        max_workers=2, fail_fast=True,
    )
    result = await asyncio.wait_for(chain.execute_async(context), timeout=3)
    assert calls == 1
    assert _outcomes(result)[2].success is True
    assert result.success is False
    assert len(result.get_failed_steps()) == 1
    assert result.metadata["fail_fast"]["triggered"] is True
    assert not context.metadata.get("loop_iteration_history")


@pytest.mark.asyncio
async def test_fail_fast_state_is_run_local_on_same_chain(contexts):
    chain = ReasoningChain(steps=[_step(1, "work"), _step(2, "work", [1])], fail_fast=True)
    failed_context = contexts()
    failed_context.register_tool("work", _failed)
    failed_result = await chain.execute_async(failed_context)

    calls = 0

    async def succeeded():
        nonlocal calls
        calls += 1
        return "ok"

    successful_context = contexts()
    successful_context.register_tool("work", succeeded)
    successful_result = await chain.execute_async(successful_context)
    assert failed_result.success is False
    assert successful_result.success is True and calls == 2
    assert successful_result.metadata["fail_fast"]["triggered"] is False
    assert successful_result.get_skipped_steps() == []


@pytest.mark.asyncio
async def test_concurrent_chains_do_not_share_stop_state(contexts):
    healthy_started = asyncio.Event()
    release_healthy = asyncio.Event()
    healthy_calls = 0

    async def healthy():
        nonlocal healthy_calls
        healthy_calls += 1
        healthy_started.set()
        await release_healthy.wait()
        return "healthy work"

    async def failed():
        await healthy_started.wait()
        raise RuntimeError("only this context fails")

    chain = ReasoningChain(steps=[_step(1, "work"), _step(2, "work", [1])], fail_fast=True)
    bad_context, good_context = contexts(), contexts()
    bad_context.register_tool("work", failed)
    good_context.register_tool("work", healthy)
    bad_task = asyncio.create_task(chain.execute_async(bad_context))
    good_task = asyncio.create_task(chain.execute_async(good_context))
    try:
        bad_result = await asyncio.wait_for(bad_task, timeout=2)
        assert not good_task.done()
        release_healthy.set()
        good_result = await asyncio.wait_for(good_task, timeout=2)
    finally:
        release_healthy.set()
        for task in (bad_task, good_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(bad_task, good_task, return_exceptions=True)
    assert bad_result.success is False and good_result.success is True
    assert healthy_calls == 2
    assert good_result.get_skipped_steps() == []


@pytest.mark.asyncio
async def test_user_cancellation_is_not_reclassified_as_fail_fast(contexts):
    context = contexts()

    async def work():
        return "completed before cancellation"

    context.register_tool("work", work)
    context.on_step_complete = lambda _result: context.cancel()
    chain = ReasoningChain(
        steps=[_step(1, "work"), _step(2, "work", [1])], fail_fast=True,
    )
    with pytest.raises(ExecutionCancelledError) as error:
        await chain.execute_async(context)
    assert error.value.result.success is False
    assert error.value.result.metadata["cancelled"] is True
    assert [step.step_number for step in error.value.result.step_results] == [1]
    assert error.value.result.metadata["fail_fast"]["triggered"] is False


@pytest.mark.asyncio
async def test_child_cancellation_drains_sibling_before_return_and_stops_queue(contexts):
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()
    never_release = asyncio.Event()
    queued_calls: list[str] = []
    started: list[int] = []

    async def cancelled():
        await sibling_started.wait()
        raise asyncio.CancelledError

    async def sibling():
        sibling_started.set()
        try:
            await never_release.wait()
            return "must be cancelled"
        finally:
            sibling_finished.set()

    async def queued():
        queued_calls.append("queued")
        return "must never start"

    context = contexts(on_step_start=lambda number, _title: started.append(number))
    for name, function in (("cancelled", cancelled), ("sibling", sibling), ("queued", queued)):
        context.register_tool(name, function)
    chain = ReasoningChain(
        steps=[_step(1, "cancelled"), _step(2, "sibling"), _step(3, "queued")],
        max_workers=2, fail_fast=True,
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(chain.execute_async(context), timeout=2)
    assert sibling_finished.is_set(), "Return only after all sibling cleanup completes"
    assert queued_calls == []
    assert set(started) == {1, 2}


@pytest.mark.asyncio
async def test_cooperative_cancel_stops_queued_siblings_in_current_batch(contexts):
    calls: list[str] = []
    started: list[int] = []
    completed: list[Any] = []
    context = contexts(on_step_start=lambda number, _title: started.append(number))

    async def first():
        calls.append("first")
        return "completed"

    async def queued():
        calls.append("queued")
        return "must not execute after cancellation"

    def on_complete(result):
        completed.append(result)
        if result.step_number == 1:
            context.cancel()

    context.on_step_complete = on_complete
    context.register_tool("first", first)
    context.register_tool("queued", queued)
    chain = ReasoningChain(
        steps=[_step(1, "first"), _step(2, "queued"), _step(3, "queued")],
        max_workers=1, fail_fast=True,
    )
    with pytest.raises(ExecutionCancelledError) as error:
        await chain.execute_async(context)
    result = error.value.result
    assert calls == ["first"] and started == [1]
    assert [item.step_number for item in completed] == [1, 2, 3]
    assert all(item.skipped and item.profiling.get("cancelled") is True for item in completed[1:])
    assert all("fail_fast" not in item.profiling for item in completed[1:])
    assert result.metadata["cancelled"] is True
    assert result.metadata["fail_fast"]["triggered"] is False
    assert result.metadata["fail_fast"]["skipped_step_numbers"] == []
    assert result.metadata["execution_stats"]["executed_steps"] == 1
    assert result.metadata["execution_stats"]["failed_steps"] == 0
    assert result.metadata["execution_stats"]["skipped_steps"] == 2
    assert [event.step_number for event in result.trace.events] == [1, 2, 3]
    assert context.snapshot().completed_step_numbers == [1]


@pytest.mark.asyncio
async def test_failed_loop_iteration_is_not_completed_in_resume_snapshot(contexts):
    counts = {"before": 0, "looped": 0, "after": 0}

    async def before():
        counts["before"] += 1
        return "completed outside loop"

    async def looped():
        counts["looped"] += 1
        if counts["looped"] == 2:
            raise RuntimeError("final failed loop iteration")
        return "completed iteration"

    async def after():
        counts["after"] += 1
        return "completed dependent"

    def register(context):
        for name, function in (("before", before), ("looped", looped), ("after", after)):
            context.register_tool(name, function)

    context = contexts()
    register(context)
    chain = ReasoningChain(
        steps=[
            _step(1, "before"),
            _step(2, "looped", [1], loop_back_to=2, loop_config=LoopConfig(max_iterations=1)),
            _step(3, "after", [2]),
        ],
        fail_fast=True,
    )
    result = await chain.execute_async(context)
    assert result.success is False and counts == {"before": 1, "looped": 2, "after": 0}
    snapshot = context.snapshot()
    assert snapshot.completed_step_numbers == [1]
    assert _outcomes(result)[2].success is False
    assert _outcomes(result)[3].skipped is True

    resumed_context = contexts()
    register(resumed_context)
    resumed_result = await chain.execute_async(resumed_context, resume_from=snapshot)
    assert resumed_result.success is True
    assert counts == {"before": 1, "looped": 4, "after": 1}


@pytest.mark.parametrize("workers", [0, -1])
def test_fail_fast_rejects_nonpositive_worker_count_instead_of_hanging(workers):
    with pytest.raises(ValueError, match="max_workers"):
        DAGExecutor(max_workers=workers, fail_fast=True)
    with pytest.raises(ValueError, match="max_workers"):
        ReasoningChain(steps=[_step(1, "local")], max_workers=workers, fail_fast=True)
