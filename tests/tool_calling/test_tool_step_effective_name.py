"""The tool that actually answered a tool step is exposed as ``$steps.N.tool_name``."""

import pytest

from mmar_carl import Language, LLMClientBase, ReasoningChain, ReasoningContext
from mmar_carl.models.config import ToolErrorRecovery, ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription
from mmar_carl.step_executors import resolve_context_reference


class _MockLLMClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _context() -> ReasoningContext:
    return ReasoningContext(outer_context="test", api=_MockLLMClient(), model="test", language=Language.ENGLISH)


def _chain(config: ToolStepConfig) -> ReasoningChain:
    return ReasoningChain(steps=[ToolStepDescription(number=1, title="Rate", config=config)])


@pytest.mark.asyncio
async def test_tool_name_is_the_configured_tool_when_it_succeeds():
    ctx = _context()
    ctx.register_tool("primary", lambda: {"rate": 0.07})
    ctx.register_tool("fallback", lambda: {"rate": 0.05})
    config = ToolStepConfig(tool_name="primary", input_mapping={}, error_recovery=ToolErrorRecovery(on_exception="fallback"))
    result = await _chain(config).execute_async(ctx)
    assert result.success
    assert result.step_results[0].tool_name == "primary"
    assert resolve_context_reference("$steps.1.tool_name", ctx) == "primary"


@pytest.mark.asyncio
async def test_tool_name_reports_the_fallback_that_answered():
    def primary() -> dict:
        raise PermissionError("denied")

    ctx = _context()
    ctx.register_tool("primary", primary)
    ctx.register_tool("fallback", lambda: {"rate": 0.05})
    config = ToolStepConfig(tool_name="primary", input_mapping={}, error_recovery=ToolErrorRecovery(on_exception="fallback"))
    result = await _chain(config).execute_async(ctx)
    assert result.success
    assert result.step_results[0].tool_name == "fallback"
    assert resolve_context_reference("$steps.1.tool_name", ctx) == "fallback"
    assert resolve_context_reference("$steps.1.result", ctx) == '{"rate": 0.05}'


@pytest.mark.parametrize("failure", ["on_timeout", "on_exception"])
@pytest.mark.parametrize("fallback_tags", [[], ["destructive"], ["information"]])
@pytest.mark.asyncio
async def test_tool_tag_policy_applies_to_fallback(failure, fallback_tags):
    async def primary():
        if failure == "on_timeout":
            raise TimeoutError()
        raise RuntimeError("primary failed")

    fallback_calls = []

    async def fallback():
        fallback_calls.append(True)
        return "recovered"

    ctx = _context()
    ctx.register_tool("primary", primary, tags=["information"])
    ctx.register_tool("fallback", fallback, tags=fallback_tags)
    config = ToolStepConfig(
        tool_name="primary",
        allowed_tool_tags=["information"],
        error_recovery=ToolErrorRecovery(**{failure: "fallback"}),
    )
    result = await _chain(config).execute_async(ctx)
    if "information" in fallback_tags:
        assert result.success
        assert fallback_calls == [True]
        assert result.step_results[0].tool_name == "fallback"
    else:
        assert not result.success
        assert not fallback_calls
        assert "Tool 'fallback' tags" in result.step_results[0].error_message
