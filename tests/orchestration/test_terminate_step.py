"""TerminateStep stops at a batch boundary and returns an explicit output."""

import asyncio
import json

import pytest

from mmar_carl import (
    LoopConfig,
    ReasoningChain,
    ReasoningContext,
    StepType,
    TerminateStepConfig,
    TerminateStepDescription,
    ToolStepConfig,
    ToolStepDescription,
)


def terminate(number=1, dependencies=(), **config):
    return TerminateStepDescription(
        number=number, title="Finish", dependencies=list(dependencies),
        config=TerminateStepConfig(**config),
    )


def tool(number, name, dependencies=(), **kwargs):
    return ToolStepDescription(
        number=number, title=name, dependencies=list(dependencies),
        config=ToolStepConfig(tool_name=name), **kwargs,
    )


@pytest.mark.asyncio
async def test_terminal_output_survives_sibling_and_history_trimming():
    async def sibling():
        await asyncio.sleep(0)
        return "sibling output"

    def tail():
        pytest.fail("the dependent tool must never start")

    context = ReasoningContext(outer_context="", api=None, max_history_entries=1)
    context.register_tool("sibling", sibling)
    context.register_tool("tail", tail)
    result = await ReasoningChain(steps=[
        terminate(output={"answer": 42}, reason="enough evidence"),
        tool(2, "sibling"),
        tool(3, "tail", (1, 2)),
    ], max_workers=2).execute_async(context)
    assert result.success and result.error is None
    assert json.loads(result.get_final_output()) == {"answer": 42}
    assert result.metadata["terminated"] == {"step_number": 1, "reason": "enough evidence"}
    assert result.trace.metadata["terminated"] == result.metadata["terminated"]
    assert [step.skipped for step in result.step_results] == [False, False, True]
    assert result.metadata["execution_stats"]["successful_steps"] == 2


@pytest.mark.asyncio
async def test_lowest_step_number_wins_and_prevents_loop_back():
    calls = []
    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("loop", lambda: calls.append("loop") or "continue")
    result = await ReasoningChain(steps=[
        terminate(2, output="chosen"), terminate(3, output="ignored"),
        tool(4, "loop", loop_back_to=4, loop_config=LoopConfig(max_iterations=3)),
    ], max_workers=3).execute_async(context)
    assert result.success
    assert result.get_final_output() == "chosen"
    assert calls == ["loop"]


@pytest.mark.asyncio
async def test_missing_output_fails_without_terminating_and_null_is_valid():
    context = ReasoningContext(outer_context="", runtime_input={"present": None}, api=None)
    result = await ReasoningChain(steps=[terminate(output_source="$outer_context.present")]).execute_async(context)
    assert result.success and result.get_final_output() == "null"

    result = await ReasoningChain(steps=[terminate(output_source="$outer_context.missing")]).execute_async(context)
    assert not result.success
    assert "unresolved" in result.error
    assert "terminated" not in result.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_terminate_never_masks_a_sibling_failure(fail_fast):
    def fail():
        raise RuntimeError("sibling failed")

    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("fail", fail)
    result = await ReasoningChain(
        steps=[terminate(output="answer"), tool(2, "fail")], max_workers=2, fail_fast=fail_fast,
    ).execute_async(context)
    assert not result.success
    assert "sibling failed" in result.error
    assert ("terminated" in result.metadata) is not fail_fast


@pytest.mark.asyncio
async def test_loaded_terminate_reads_a_prior_step_result():
    chain = ReasoningChain(steps=[tool(1, "source"), terminate(2, (1,), output_source="$steps.1.result_data")])
    restored = ReasoningChain.from_dict(chain.to_dict())
    assert restored.to_dict()["format_version"] == 15
    assert restored.steps[1].step_type == StepType.TERMINATE
    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("source", lambda: {"answer": 42})
    assert json.loads((await restored.execute_async(context)).get_final_output()) == {"answer": 42}


def test_terminate_requires_exactly_one_output():
    with pytest.raises(ValueError, match="exactly one"):
        TerminateStepConfig()
    with pytest.raises(ValueError, match="exactly one"):
        TerminateStepConfig(output=None, output_source="$outer_context")


@pytest.mark.asyncio
async def test_resume_cannot_run_tail_after_completed_termination():
    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("tail", lambda: pytest.fail("the tail must stay stopped"))
    chain = ReasoningChain(steps=[terminate(output="done"), tool(2, "tail", (1,))])
    await chain.execute_async(context)

    fresh = ReasoningContext(outer_context="", api=None, history=["keep this"])
    with pytest.raises(ValueError, match="already executed TerminateStep"):
        await chain.execute_async(fresh, resume_from=context.snapshot())
    assert fresh.history == ["keep this"]


@pytest.mark.asyncio
async def test_handoff_preserves_explicit_empty_terminal_output():
    from mmar_carl import AgentHandoffStepConfig, AgentHandoffStepDescription

    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("sibling", lambda: "sibling output")
    child = ReasoningChain(steps=[terminate(output=""), tool(2, "sibling")], max_workers=2)
    step = AgentHandoffStepDescription(
        number=1, title="Delegate", config=AgentHandoffStepConfig(), sub_chain=child,
    )

    result = await ReasoningChain(steps=[step]).execute_async(context)
    assert result.success
    assert result.step_results[0].result == ""


def test_catalog_rejects_terminal_output_reference_without_dependency():
    from mmar_carl.meta import validate_spec

    spec = ReasoningChain(steps=[tool(1, "source"), terminate(2, (1,), output_source="$steps.1.result")]).to_dict()
    assert validate_spec(spec).valid
    spec["steps"][1]["dependencies"] = []
    assert not validate_spec(spec).valid


@pytest.mark.asyncio
async def test_same_batch_planner_does_not_schedule_actions():
    from mmar_carl import PlannerStepConfig, PlannerStepDescription

    class Client:
        async def get_response_with_retries(self, prompt, **kwargs):
            return '{"actions":["tail"]}'

    events = []
    context = ReasoningContext(
        outer_context="", api=Client(),
        on_step_event=lambda number, name, payload: events.append((number, name)),
    )
    context.register_tool("tail", lambda: pytest.fail("planner action must not run"))
    planner = PlannerStepDescription(
        number=2, title="Plan",
        config=PlannerStepConfig(instruction="Choose tail", actions={"tail": ToolStepConfig(tool_name="tail")}),
    )
    result = await ReasoningChain(steps=[terminate(output="done"), planner], max_workers=2).execute_async(context)
    assert result.success and result.get_final_output() == "done"
    assert result.get_step_result(2).success
    assert (2, "planner_actions_scheduled") not in events
    assert result.metadata["execution_stats"]["total_steps"] == 2


@pytest.mark.asyncio
async def test_failed_producer_cannot_terminate_with_output_from_a_previous_run():
    calls = []
    context = ReasoningContext(outer_context="", api=None)
    context.register_tool("source", lambda: "previous run")
    context.register_tool("tail", lambda: calls.append("tail") or "fallback output")
    chain = ReasoningChain(steps=[
        tool(1, "source"), terminate(2, (1,), output_source="$steps.1.result"), tool(3, "tail", (2,)),
    ])
    first = await chain.execute_async(context)
    assert first.success and first.get_final_output() == "previous run"

    def fail():
        raise RuntimeError("producer failed")

    context.register_tool("source", fail)
    second = await chain.execute_async(context)
    assert not second.success
    assert not second.get_step_result(2).success
    assert "terminated" not in second.metadata
    assert calls == ["tail"]
