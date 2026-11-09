"""runtime_input must survive DAGExecutor's per-step context snapshots."""

import pytest

from mmar_carl import ReasoningChain, ReasoningContext, ToolParameter, ToolStepConfig, ToolStepDescription


class _NoProvider:
    async def get_response(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("tool-only chain must not call an LLM")


def _lookup_step(number: int, deps=()) -> ToolStepDescription:
    return ToolStepDescription(
        number=number,
        title=f"Lookup {number}",
        dependencies=list(deps),
        config=ToolStepConfig(
            tool_name="lookup",
            input_mapping={"city": "$outer_context.city"},
            parameters=[ToolParameter(name="city", type="string", required=True)],
        ),
    )


@pytest.mark.asyncio
async def test_outer_context_field_reads_runtime_input_inside_chain():
    # outer_context is prose (as a host would send), not JSON.
    context = ReasoningContext(
        outer_context="Find hotels.\n<untrusted_runtime_input>{...}</untrusted_runtime_input>",
        runtime_input={"city": "Paris"},
        api=_NoProvider(),
    )
    seen = []

    async def lookup(city):
        seen.append(city)
        return {"city": city}

    context.register_tool("lookup", lookup)
    try:
        # Two batches: step 2 runs on a fresh snapshot after step 1 merges back.
        result = await ReasoningChain(steps=[_lookup_step(1), _lookup_step(2, [1])]).execute_async(context)
    finally:
        await context.close()

    assert [r.success for r in result.step_results] == [True, True]
    assert seen == ["Paris", "Paris"]


@pytest.mark.asyncio
async def test_missing_required_input_fails_before_tool_dispatch():
    context = ReasoningContext(outer_context="Prose", runtime_input={}, api=_NoProvider())

    async def lookup(city):
        pytest.fail(f"Tool received unresolved input: {city!r}")

    context.register_tool("lookup", lookup)
    try:
        result = await ReasoningChain(steps=[_lookup_step(1)]).execute_async(context)
    finally:
        await context.close()

    assert not result.success
    assert "CARL_TOOL_INPUT_UNRESOLVED" in result.step_results[0].error_message
