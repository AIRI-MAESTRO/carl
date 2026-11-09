"""Embed one CARL chain as a typed, isolated tool in another chain."""

from __future__ import annotations

import asyncio
from typing import Any

from mmar_carl import (
    ChainToolDefinition,
    ChainToolOutcome,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


class NoLLMClient(LLMClientBase):
    """This example uses only deterministic tools."""

    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "unused"


def normalize(payload: dict[str, Any]) -> dict[str, str]:
    """Normalize a text value for the embedded chain."""
    return {"normalized": payload["text"].strip().lower()}


def build_child_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Normalize text",
                config=ToolStepConfig(
                    tool_name="normalize",
                    input_mapping={"payload": "$outer_context"},
                ),
            )
        ],
        trace_name="normalizer-child",
    )


def build_parent_chain() -> ReasoningChain:
    child_tool = ChainToolDefinition.from_chain(
        name="normalize_with_chain",
        description="Run the pinned normalizer child chain.",
        chain=build_child_chain(),
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"normalized": {"type": "string"}},
            "required": ["normalized"],
            "additionalProperties": False,
        },
        output_reference="$steps.1.result_data",
        allowed_tools=["normalize"],
        timeout_seconds=5,
    )
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Invoke child chain",
                config=ToolStepConfig(
                    tool_name=child_tool.name,
                    input_mapping={"text": "$outer_context"},
                ),
            )
        ],
        chain_tools=[child_tool],
        trace_name="chain-as-tool-parent",
    )


async def main() -> None:
    chain = build_parent_chain()
    context = ReasoningContext(outer_context="  CARL Composition  ", api=NoLLMClient())
    context.register_tool("normalize", normalize, tags=["deterministic"])

    preflight = chain.preflight(context)
    assert preflight.all_present
    result = await chain.execute_async(context)
    outcome = ChainToolOutcome.model_validate(result.step_results[0].result_data)

    assert outcome.success
    assert outcome.output == {"normalized": "carl composition"}
    print(f"status={outcome.status}")
    print(f"output={outcome.output}")
    print(f"snapshot_sha256={outcome.snapshot_sha256}")


if __name__ == "__main__":
    asyncio.run(main())
