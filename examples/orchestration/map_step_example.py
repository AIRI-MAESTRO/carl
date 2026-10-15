"""Run a bounded MapStep and preserve input order across parallel tool calls."""

from __future__ import annotations

import asyncio
import json

from mmar_carl import (
    LLMClientBase,
    MapStepConfig,
    MapStepDescription,
    ReasoningChain,
    ReasoningContext,
)


class _UnusedClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "unused"


async def enrich(item: dict[str, object], index: int, source: str) -> dict[str, object]:
    # Deliberately finish later input items first. MapStep still aggregates by
    # their original input indexes.
    await asyncio.sleep(0.01 * (3 - index))
    return {**item, "index": index, "source": source}


async def main() -> None:
    rows = [{"name": "alpha"}, {"name": "beta"}, {"name": "gamma"}]
    context = ReasoningContext(
        outer_context=json.dumps(rows),
        api=_UnusedClient(),
        metadata={"source": "example"},
    )
    context.register_tool("enrich", enrich)

    chain = ReasoningChain(
        steps=[
            MapStepDescription(
                number=1,
                title="Enrich rows",
                config=MapStepConfig(
                    items_source="$outer_context",
                    tool_name="enrich",
                    item_parameter="item",
                    index_parameter="index",
                    input_mapping={"source": "$metadata.source"},
                    max_items=10,
                    max_concurrency=2,
                    item_timeout_seconds=1,
                    output_memory_key="enriched",
                ),
            )
        ]
    )

    result = await chain.execute_async(context)
    outcome = result.step_results[0].as_map_outcome()
    assert result.success and outcome is not None
    print(json.dumps(outcome.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
