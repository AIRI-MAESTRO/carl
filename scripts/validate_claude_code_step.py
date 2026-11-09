"""Live validation for the Claude-Code-as-step PoC.

Runs a real 2-step chain against the locally installed ``claude`` CLI:

1. Step 1 asks the agent to remember a codeword and acknowledge.
2. Step 2 resumes step 1's session (``resume_session="$memory.claude_code.step_1"``)
   and asks for the codeword back — proving that one Claude Code session can
   span multiple chain steps.

Requires: Claude Code CLI on PATH with working auth. Costs a few cents
(two headless agent runs).

Run: PYTHONPATH=src uv run python scripts/validate_claude_code_step.py
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

from mmar_carl import (
    ClaudeCodeStepConfig,
    ClaudeCodeStepDescription,
    ReasoningChain,
    ReasoningContext,
    check_claude_code_cli,
)

CODEWORD = "CARL-STEP-OK"


def _make_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        ClaudeCodeStepDescription(
            number=1,
            title="Set codeword",
            config=ClaudeCodeStepConfig(
                task=(
                    f"Remember the codeword {CODEWORD}. "
                    "Reply with exactly the single word: ACKNOWLEDGED"
                ),
                max_turns=1,
                timeout=120.0,
            ),
        ),
        ClaudeCodeStepDescription(
            number=2,
            title="Recall codeword",
            dependencies=[1],
            config=ClaudeCodeStepConfig(
                task=(
                    "What codeword did I ask you to remember earlier in this "
                    "conversation? Reply with only the codeword, nothing else."
                ),
                resume_session="$memory.claude_code.step_1",
                max_turns=1,
                timeout=120.0,
            ),
        ),
    ])


async def main() -> int:
    status = check_claude_code_cli()
    if not status.available:
        print(f"SKIP: {status.error}")
        return 1
    print(f"Using {status.version} at {status.resolved_path}")

    ctx = ReasoningContext(outer_context="claude-code-step validation", api=MagicMock())
    result = await _make_chain().execute_async(ctx)

    total_cost = 0.0
    for step_result in result.step_results:
        data = step_result.result_data or {}
        total_cost += data.get("total_cost_usd") or 0.0
        print(
            f"step {step_result.step_number}: success={step_result.success} "
            f"turns={data.get('num_turns')} session={data.get('session_id')} "
            f"tokens={step_result.token_usage.get('total', 0)} "
            f"model={step_result.model}"
        )
        print(f"  result: {step_result.result[:200]!r}")
        if not step_result.success:
            print(f"  error: {step_result.error_message}")

    print(f"total cost: ${total_cost:.4f}")

    if not result.success:
        print("FAIL: chain did not succeed")
        return 1

    recall = result.step_results[1]
    sessions = [r.result_data["session_id"] for r in result.step_results]
    if CODEWORD not in recall.result:
        print(f"FAIL: step 2 did not recall the codeword (got {recall.result!r})")
        return 1

    print(f"OK: codeword recalled across steps (sessions: {sessions[0]} -> {sessions[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
