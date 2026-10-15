"""Codex-as-step: delegate chain steps to a local Codex agent.

Demonstrates ``CodexStepDescription`` — a step type that runs a local Codex
agent (via the optional ``openai-codex`` SDK) as a CARL chain step:

1. Step 1 starts a read-only Codex thread that inspects the repository.
2. Step 2 resumes step 1's thread (``resume_session="$memory.codex.step_1"``)
   and asks a follow-up that must be answered as JSON matching ``output_schema``;
   the structured outcome is written to memory via ``output_memory_key``.

Requires ``pip install 'mmar-carl[codex]'`` and a logged-in local Codex
runtime (``codex login``). Skips gracefully when the runtime is missing.

Run: make example-codex
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mmar_carl import (
    CodexStepConfig,
    CodexStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    check_codex_runtime,
)


def build_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        CodexStepDescription(
            number=1,
            title="Repository orientation",
            config=CodexStepConfig(
                task=(
                    "Inspect the repository at a high level. Do not modify files. "
                    "Return three concise bullets describing its architecture. "
                    "Keep this request in mind: {request}"
                ),
                input_mapping={"request": "$outer_context"},
                cwd=str(Path.cwd()),
                sandbox="read-only",
                reasoning_effort="medium",
                ephemeral=False,  # keep the thread resumable by step 2
            ),
        ),
        CodexStepDescription(
            number=2,
            title="Summarise as JSON",
            dependencies=[1],
            config=CodexStepConfig(
                task=(
                    "Condense your previous inspection. Reply with ONLY a JSON "
                    'object: {"summary": "<one sentence>", "keywords": ["<w1>", "<w2>"]}'
                ),
                resume_session="$memory.codex.step_1",  # same Codex thread
                sandbox="read-only",
                output_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "keywords"],
                },
                output_memory_key="orientation",  # parsed object → memory
            ),
        ),
    ])


async def main() -> int:
    status = check_codex_runtime()
    if not status.available:
        print(f"SKIP: {status.error}")
        return 0
    print(f"Using openai-codex {status.sdk_version or '?'} (codex binary: {status.cli_path or 'not on PATH'})")

    context = ReasoningContext(
        outer_context="Explain how CARL executes a reasoning chain.",
        # CodexStep does not use CARL's LLM client; ReasoningContext retains
        # this required field for compatibility with mixed-step chains.
        api=object(),
        language=Language.ENGLISH,
    )

    chain = build_chain()
    print(chain.preflight(context).format_text())

    print("\nRunning 2-step Codex chain (thread resume + JSON contract)...\n")
    result = await chain.execute_async(context)

    for step_result in result.step_results:
        data = step_result.result_data or {}
        print(
            f"\nstep {step_result.step_number} ({step_result.step_title}): "
            f"success={step_result.success} thread={data.get('thread_id')} "
            f"tokens={step_result.token_usage.get('total', 0)}"
        )
        print(f"  answer: {step_result.result[:300]}")
        if not step_result.success:
            print(f"  error: {step_result.error_message}")

    print(f"\nstructured output in memory: {context.memory_read('orientation', namespace='codex')}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
