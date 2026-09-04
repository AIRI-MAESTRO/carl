"""Strict CodeStep construction and optional Docker execution with no LLM calls."""

from __future__ import annotations

import argparse
import asyncio
import os

from examples.utils import print_execution_summary
from mmar_carl import (
    CodeExecutionPolicy,
    CodeRuntimeProfile,
    CodeStepConfig,
    CodeStepDescription,
    ReasoningChain,
    ReasoningContext,
)

SOURCE = """def run(inputs):
    values = inputs["values"]
    return {"sum": sum(values), "count": len(values)}
"""

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "values": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["values"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sum": {"type": "number"},
        "count": {"type": "integer"},
    },
    "required": ["sum", "count"],
    "additionalProperties": False,
}


def build_chain() -> ReasoningChain:
    """Build a chain that consumes source generated earlier in the run."""

    return ReasoningChain(
        steps=[
            CodeStepDescription(
                number=1,
                title="Execute generated calculation",
                config=CodeStepConfig(
                    source="$memory.generated.python_source",
                    runtime_profile="python-safe-v1",
                    input_mapping={"values": "$memory.input.values"},
                    input_schema=INPUT_SCHEMA,
                    output_schema=OUTPUT_SCHEMA,
                    timeout_seconds=5,
                    max_source_bytes=20_000,
                    max_input_bytes=100_000,
                    max_output_bytes=100_000,
                    output_key="statistics",
                ),
            ),
        ],
    )


def build_context(image: str) -> ReasoningContext:
    """Attach host authority and pre-populate prior runtime values."""

    context = ReasoningContext(
        outer_context="No-provider CodeStep example",
        api=None,
        code_execution_policy=CodeExecutionPolicy(
            profiles={
                "python-safe-v1": CodeRuntimeProfile(
                    runtime="docker",
                    revision=image,
                    interpreter=("python", "-I", "-B"),
                    prepare_config={"image": image},
                    cpu_limit=1,
                    mem_limit="256m",
                    pids_limit=32,
                    max_timeout_seconds=5,
                ),
            },
        ),
    )
    # In a real chain an earlier generation step writes this exact string.
    context.memory_write("python_source", SOURCE, namespace="generated")
    context.memory_write("values", [1, 2, 3, 4], namespace="input")
    return context


async def execute(image: str) -> None:
    """Execute using a caller-supplied, digest-pinned Python Docker image."""

    chain = build_chain()
    context = build_context(image)
    preflight = chain.preflight(context)
    if not preflight.all_present:
        raise RuntimeError(preflight.format_text())

    result = await chain.execute_async(context)
    print_execution_summary(result)
    outcome = result.step_results[0].as_code_execution_outcome()
    assert outcome is not None and outcome.status == "completed"
    assert outcome.output == {"sum": 10, "count": 4}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a CodeStep or execute it in a strict Docker runtime.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute with the digest-pinned image in CARL_CODE_IMAGE.",
    )
    args = parser.parse_args()

    if not args.execute:
        print(build_chain().to_json(indent=2))
        print("\nPass --execute and set CARL_CODE_IMAGE to run without an LLM provider.")
        return

    image = os.environ.get("CARL_CODE_IMAGE", "")
    if not image:
        raise SystemExit(
            "CARL_CODE_IMAGE is required, for example "
            "python@sha256:<64-lowercase-hex-digest>"
        )
    asyncio.run(execute(image))


if __name__ == "__main__":
    main()
