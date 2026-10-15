"""Typed, process-local HumanInputStep examples with no provider calls."""

import asyncio

from examples.utils import print_execution_summary
from mmar_carl import (
    HumanInputRequest,
    HumanInputResponse,
    HumanInputStepConfig,
    HumanInputStepDescription,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


class MockClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return await self.get_response(prompt)


def response(
    request: HumanInputRequest,
    value: str,
    *,
    actor_id: str = "example-human",
) -> HumanInputResponse:
    """Build the typed response expected by HumanInputStep."""
    return HumanInputResponse(
        request_id=request.request_id,
        value=value,
        actor_id=actor_id,
        provenance={"channel": "example"},
    )


def record_feedback(feedback: str) -> str:
    return f"Recorded human feedback: {feedback}"


async def example1_async_human() -> None:
    """An asynchronous host callback waits for a simulated human."""
    print("\nExample 1: asynchronous human response")
    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Request review",
                config=HumanInputStepConfig(
                    prompt="Review the draft:",
                    timeout=2,
                    min_length=1,
                    output_memory_key="review",
                ),
            ),
            ToolStepDescription(
                number=2,
                title="Record review",
                dependencies=[1],
                config=ToolStepConfig(
                    tool_name="record_feedback",
                    input_mapping={
                        "feedback": "$memory.human_input.review",
                    },
                ),
            ),
        ],
    )
    context = ReasoningContext(outer_context="Review workflow", api=MockClient())
    context.register_tool("record_feedback", record_feedback)

    async def reviewer(request: HumanInputRequest) -> HumanInputResponse:
        print(f"  Human sees: {request.prompt}")
        await asyncio.sleep(0.05)
        return response(request, "Looks good; add one citation.")

    context.on_human_input_requested = reviewer
    result = await chain.execute_async(context)
    print_execution_summary(result)

    assert result.success
    assert context.memory_read("review", namespace="human_input") == (
        "Looks good; add one citation."
    )


async def example2_timeout_is_not_an_answer() -> None:
    """Timeout is a typed non-success outcome and never auto-approves."""
    print("\nExample 2: timeout is non-success")
    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Request input",
                config=HumanInputStepConfig(
                    prompt="Provide a decision:",
                    timeout=0.05,
                    output_memory_key="decision",
                ),
            ),
        ],
    )
    context = ReasoningContext(outer_context="Timeout workflow", api=MockClient())

    async def absent_human(request: HumanInputRequest) -> HumanInputResponse:
        await asyncio.Future()

    context.on_human_input_requested = absent_human
    result = await chain.execute_async(context)
    print_execution_summary(result)

    outcome = result.step_results[0].as_human_input_outcome()
    assert result.success is False
    assert outcome is not None and outcome.status == "timed_out"
    assert context.memory_read("decision", namespace="human_input") is None


async def example3_missing_provider_is_unavailable() -> None:
    """Batch mode must install an explicit fake human instead of a fallback."""
    print("\nExample 3: missing provider is unavailable")
    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Request input",
                config=HumanInputStepConfig(prompt="Provide input:"),
            ),
        ],
    )
    result = await chain.execute_async(
        ReasoningContext(outer_context="No provider", api=MockClient()),
    )
    print_execution_summary(result)

    outcome = result.step_results[0].as_human_input_outcome()
    assert result.success is False
    assert outcome is not None and outcome.status == "unavailable"


async def example4_explicit_fake_human() -> None:
    """Tests and batch jobs remain deterministic with an explicit callback."""
    print("\nExample 4: explicit fake human")
    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Test input",
                config=HumanInputStepConfig(
                    prompt="Synthetic response required:",
                    output_memory_key="answer",
                ),
            ),
        ],
    )
    context = ReasoningContext(outer_context="Test workflow", api=MockClient())
    context.on_human_input_requested = lambda request: response(
        request,
        "fixture answer",
        actor_id="test-fixture",
    )
    result = await chain.execute_async(context)
    print_execution_summary(result)

    assert result.success
    assert context.memory_read("answer", namespace="human_input") == "fixture answer"


async def main() -> None:
    await example1_async_human()
    await example2_timeout_is_not_an_answer()
    await example3_missing_provider_is_unavailable()
    await example4_explicit_fake_human()
    print("\nAll HumanInputStep examples passed.")


if __name__ == "__main__":
    asyncio.run(main())
