import asyncio

import pytest

from mmar_carl.eager_tools import execute_eager_tools
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


def tool_step(number, tool, dependencies=(), **inputs):
    return ToolStepDescription(
        number=number,
        title=tool,
        dependencies=list(dependencies),
        config=ToolStepConfig(tool_name=tool, input_mapping=inputs),
    )


@pytest.mark.asyncio
async def test_child_starts_while_unrelated_root_is_running():
    sibling_started = asyncio.Event()
    child_started = asyncio.Event()
    release_sibling = asyncio.Event()

    async def parent():
        await sibling_started.wait()
        return '{"items": [4]}'

    async def sibling():
        sibling_started.set()
        await release_sibling.wait()
        return "sibling"

    async def child(value):
        child_started.set()
        return value + 1

    steps = [
        tool_step(3, "child", [1], value="$steps.1.result_data.items.0"),
        tool_step(2, "sibling"),
        tool_step(1, "parent"),
    ]
    async with asyncio.TaskGroup() as group:
        run = group.create_task(
            execute_eager_tools(steps, tools={"parent": parent, "sibling": sibling, "child": child}, max_concurrency=2)
        )
        try:
            await asyncio.wait_for(child_started.wait(), 1)
            assert not run.done()
        finally:
            release_sibling.set()

    assert [(r.step_number, r.result_data) for r in run.result()] == [(3, 5), (2, "sibling"), (1, {"items": [4]})]


@pytest.mark.asyncio
async def test_transitive_values_are_isolated_and_failures_block_only_descendants():
    async def original(value):
        return value

    async def fail(value):
        value["items"].append("mutated")
        raise ValueError("broken")

    async def independent(value):
        return value

    async def unreachable():
        pytest.fail("blocked tool was called")

    steps = [
        tool_step(1, "original", value="$outer_context"),
        tool_step(2, "fail", [1], value="$steps.1.result_data"),
        tool_step(3, "independent", [1], value='"literal"'),
        tool_step(4, "unreachable", [2, 3]),
        tool_step(5, "independent", [3], value="$steps.1.result_data.items"),
        tool_step(6, "unreachable", [4]),
    ]
    outer = {"items": [1]}
    results = await execute_eager_tools(
        steps,
        tools={"original": original, "fail": fail, "independent": independent, "unreachable": unreachable},
        outer_context=outer,
    )

    assert [r.success for r in results] == [True, False, True, False, True, False]
    assert [r.skipped for r in results] == [False, False, False, True, False, True]
    assert results[1].error_message == "broken"
    assert results[0].result_data == outer == {"items": [1]}
    assert results[4].result_data == [1]


@pytest.mark.parametrize(
    "reference", ["$steps.1.result_data", "$history", "$steps.1.execution_time", "$outer_context."]
)
@pytest.mark.asyncio
async def test_invalid_reference_fails_preflight(reference):
    async def tool(**kwargs):
        pytest.fail("tool called before admission completed")

    steps = [tool_step(1, "tool"), tool_step(2, "tool", value=reference)]
    with pytest.raises(ValueError):
        await execute_eager_tools(steps, tools={"tool": tool})


@pytest.mark.parametrize("cancel_run", [False, True])
@pytest.mark.asyncio
async def test_timeout_and_cancellation_drain_running_tools(cancel_run):
    started = asyncio.Event()
    drained = asyncio.Event()

    async def hang():
        started.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            drained.set()

    run = asyncio.create_task(
        execute_eager_tools([tool_step(1, "hang")], tools={"hang": hang}, timeout=None if cancel_run else 0.1)
    )
    await asyncio.wait_for(started.wait(), 1)
    if cancel_run:
        run.cancel()

    with pytest.raises(asyncio.CancelledError if cancel_run else TimeoutError):
        await run
    assert drained.is_set()


@pytest.mark.asyncio
async def test_invalid_json_results_fail_the_producer():
    async def malformed():
        return {1: "bad"}

    async def invalid_structured_string():
        return '{"number": NaN}'

    results = await execute_eager_tools(
        [tool_step(1, "malformed"), tool_step(2, "invalid")],
        tools={"malformed": malformed, "invalid": invalid_structured_string},
    )
    assert all(not result.success for result in results)


@pytest.mark.asyncio
async def test_missing_reference_fails_before_dispatch_and_null_is_passed():
    seen = []

    async def capture(value):
        seen.append(value)
        return value

    steps = [
        tool_step(1, "capture", value="$outer_context.null"),
        tool_step(2, "capture", value="$outer_context.missing"),
    ]
    results = await execute_eager_tools(steps, tools={"capture": capture}, outer_context={"null": None})
    assert seen == [None]
    assert results[0].success and results[0].result_data is None
    assert not results[1].success
    assert "Missing eager tool input" in results[1].error_message


@pytest.mark.parametrize("invalid_graph", ["duplicate", "missing", "cycle", "trigger", "tags", "sync", "nan"])
@pytest.mark.asyncio
async def test_invalid_graph_or_host_input_fails_before_dispatch(invalid_graph):
    async def tool():
        pytest.fail("tool called before admission completed")

    steps = [tool_step(1, "tool"), tool_step(2, "tool")]
    tools = {"tool": tool}
    outer = {}
    if invalid_graph == "duplicate":
        steps[1].number = 1
    elif invalid_graph == "missing":
        steps[1].dependencies = [3]
    elif invalid_graph == "cycle":
        steps[1].dependencies = [2]
    elif invalid_graph == "trigger":
        steps[1].triggered_by = ["ready"]
    elif invalid_graph == "tags":
        steps[1].config.allowed_tool_tags = ["allowed"]
    elif invalid_graph == "sync":
        steps[1].config.tool_name = "sync"
        tools["sync"] = lambda: pytest.fail("synchronous tool called")
    else:
        outer = {"bad": float("nan")}
    with pytest.raises(ValueError):
        await execute_eager_tools(steps, tools=tools, outer_context=outer)


@pytest.mark.asyncio
async def test_concurrency_limit_and_per_tool_timeout():
    active = 0
    high_water = 0
    saturated = asyncio.Event()
    released = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def concurrent():
        nonlocal active, high_water
        active += 1
        high_water = max(high_water, active)
        if active == 3:
            saturated.set()
        await released.wait()
        active -= 1
        return "ok"

    async def timeout_tool():
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    timed_step = tool_step(9, "timeout", [8])
    timed_step.config.timeout = 0.01
    steps = [tool_step(number, "concurrent") for number in range(1, 9)] + [timed_step]
    async with asyncio.TaskGroup() as group:
        run = group.create_task(
            execute_eager_tools(steps, tools={"concurrent": concurrent, "timeout": timeout_tool}, max_concurrency=3)
        )
        try:
            await asyncio.wait_for(saturated.wait(), 1)
        finally:
            released.set()

    results = run.result()
    assert high_water == 3
    assert [r.step_number for r in results] == list(range(1, 10))
    assert all(r.success for r in results[:-1])
    assert "timed out" in results[-1].error_message
    assert cleaned_up.is_set()


@pytest.mark.parametrize("cancel_run", [False, True])
@pytest.mark.asyncio
async def test_run_cancellation_does_not_interrupt_tool_timeout_cleanup(cancel_run):
    cleaning_up = asyncio.Event()
    cleaned_up = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def hang():
        try:
            await asyncio.Future()
        finally:
            cleaning_up.set()
            await release_cleanup.wait()
            cleaned_up.set()

    step = tool_step(1, "hang")
    step.config.timeout = 0.01
    run = asyncio.create_task(execute_eager_tools([step], tools={"hang": hang}, timeout=None if cancel_run else 0.05))
    await asyncio.wait_for(cleaning_up.wait(), 1)
    if cancel_run:
        run.cancel()
    try:
        # Shielding the run lets this wait observe whether it returned while the
        # tool was still cleaning up, without introducing another cancellation.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(run), 0.1)
        assert not run.done()
    finally:
        release_cleanup.set()

    with pytest.raises(asyncio.CancelledError if cancel_run else TimeoutError):
        await run
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_tool_that_finishes_past_deadline_fails_its_step(monkeypatch):
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)

    async def slow():
        # The loop cannot deliver its timeout callback during this work.
        nonlocal now
        now += 2
        return "late"

    step = tool_step(1, "slow")
    step.config.timeout = 1
    results = await execute_eager_tools([step], tools={"slow": slow})

    assert not results[0].success
    assert "timed out" in results[0].error_message


@pytest.mark.parametrize("task_factory", [None, asyncio.eager_task_factory])
@pytest.mark.asyncio
async def test_run_deadline_prevents_further_dispatch(task_factory, monkeypatch):
    calls = []
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)

    async def slow():
        nonlocal now
        calls.append("slow")
        now += 2

    async def later():
        calls.append("later")

    previous_factory = loop.get_task_factory()
    loop.set_task_factory(task_factory)
    try:
        with pytest.raises(TimeoutError):
            await execute_eager_tools(
                [tool_step(1, "slow"), tool_step(2, "later")],
                tools={"slow": slow, "later": later},
                timeout=1,
            )
    finally:
        loop.set_task_factory(previous_factory)

    assert calls == ["slow"]


@pytest.mark.asyncio
async def test_tool_deadline_prevents_dispatch_of_a_queued_call(monkeypatch):
    calls = []
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)

    async def slow():
        nonlocal now
        now += 2

    async def later():
        calls.append("later")

    later_step = tool_step(2, "later")
    later_step.config.timeout = 1
    results = await execute_eager_tools(
        [tool_step(1, "slow"), later_step],
        tools={"slow": slow, "later": later},
    )

    assert results[0].success
    assert "timed out" in results[1].error_message
    assert calls == []
