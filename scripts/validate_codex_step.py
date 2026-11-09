"""Live validation for Codex-as-step.

Runs a real 2-step chain against the local Codex runtime:

1. Step 1 asks the agent to remember a codeword and acknowledge.
2. Step 2 resumes step 1's thread (``resume_session="$memory.codex.step_1"``)
   and asks for the codeword back — proving that one Codex thread can span
   multiple chain steps.

Requires ``pip install 'mmar-carl[codex]'`` and a logged-in local Codex
runtime (``codex login``). Costs a small amount (two agent turns).

Run: PYTHONPATH=src uv run python scripts/validate_codex_step.py
"""

from __future__ import annotations

import asyncio
import sys

from mmar_carl import (
    CodexStepConfig,
    CodexStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    check_codex_runtime,
)

CODEWORD = "CARL-CODEX-OK"


def _make_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        CodexStepDescription(
            number=1,
            title="Set codeword",
            config=CodexStepConfig(
                task=(
                    f"Remember the codeword {CODEWORD}. "
                    "Reply with exactly the single word: ACKNOWLEDGED"
                ),
                sandbox="read-only",
                ephemeral=False,  # keep the thread resumable by step 2
                timeout=120.0,
            ),
        ),
        CodexStepDescription(
            number=2,
            title="Recall codeword",
            dependencies=[1],
            config=CodexStepConfig(
                task=(
                    "What codeword did I ask you to remember earlier in this "
                    "conversation? Reply with only the codeword, nothing else."
                ),
                resume_session="$memory.codex.step_1",
                sandbox="read-only",
                timeout=120.0,
            ),
        ),
    ])


async def main() -> int:
    status = check_codex_runtime()
    if not status.available:
        print(f"SKIP: {status.error}")
        return 1
    print(f"Using openai-codex {status.sdk_version or '?'} (codex binary: {status.cli_path or 'not on PATH'})")

    ctx = ReasoningContext(
        outer_context="codex-step validation",
        api=object(),
        language=Language.ENGLISH,
    )
    result = await _make_chain().execute_async(ctx)

    for step_result in result.step_results:
        data = step_result.result_data or {}
        print(
            f"step {step_result.step_number}: success={step_result.success} "
            f"thread={data.get('thread_id')} status={data.get('status')} "
            f"tokens={step_result.token_usage.get('total', 0)}"
        )
        print(f"  result: {step_result.result[:200]!r}")
        if not step_result.success:
            print(f"  error: {step_result.error_message}")

    if not result.success:
        print("FAIL: chain did not succeed")
        return 1

    recall = result.step_results[1]
    threads = [r.result_data["thread_id"] for r in result.step_results]
    if CODEWORD not in recall.result:
        print(f"FAIL: step 2 did not recall the codeword (got {recall.result!r})")
        return 1

    print(f"OK: codeword recalled across steps (threads: {threads[0]} -> {threads[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
