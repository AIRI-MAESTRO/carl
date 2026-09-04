"""Claude-Code-as-step: delegate chain steps to a headless Claude Code agent.

Demonstrates ``ClaudeCodeStepDescription`` — a step type that runs the locally
installed Claude Code CLI (``claude -p``) as a fully autonomous sub-agent:

1. Step 1 streams the agent's answer (``stream=True`` forwards each assistant
   text block to ``context.on_llm_chunk`` as it arrives).
2. Step 2 resumes step 1's session (``resume_session="$memory.claude_code.step_1"``)
   and asks a follow-up that must be answered as JSON matching ``output_schema``;
   the parsed object is written to memory via ``output_memory_key``.

Requires the Claude Code CLI on PATH with working auth (costs a few cents per
run — two headless agent calls). Skips gracefully when the CLI is missing.

Run: make example-claude-code
"""

import asyncio
import sys

from mmar_carl import (
    ClaudeCodeStepConfig,
    ClaudeCodeStepDescription,
    ReasoningChain,
    ReasoningContext,
    check_claude_code_cli,
)


def build_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        ClaudeCodeStepDescription(
            number=1,
            title="Explain",
            config=ClaudeCodeStepConfig(
                task="Explain in two short sentences what a {concept} is.",
                input_mapping={"concept": "$outer_context"},
                max_turns=1,
                stream=True,          # forward assistant text to on_llm_chunk
                timeout=120.0,
            ),
        ),
        ClaudeCodeStepDescription(
            number=2,
            title="Summarise as JSON",
            dependencies=[1],
            config=ClaudeCodeStepConfig(
                task=(
                    "Condense your previous explanation. Reply with ONLY a JSON "
                    'object: {"summary": "<one sentence>", "keywords": ["<w1>", "<w2>"]}'
                ),
                resume_session="$memory.claude_code.step_1",  # same agent session
                output_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "keywords"],
                },
                output_memory_key="explanation",  # parsed object → memory
                max_turns=1,
                timeout=120.0,
            ),
        ),
    ])


async def main() -> int:
    status = check_claude_code_cli()
    if not status.available:
        print(f"SKIP: {status.error}")
        return 0
    print(f"Using {status.version} at {status.resolved_path}")

    context = ReasoningContext(
        outer_context="directed acyclic graph",
        api=None,  # no CARL-side LLM needed — the CLI brings its own model
        on_llm_chunk=lambda chunk: print(f"  [stream] {chunk}"),
    )

    chain = build_chain()
    print(chain.preflight(context).format_text())

    print("\nRunning 2-step Claude Code chain (streaming + session resume + JSON contract)...\n")
    result = await chain.execute_async(context)

    total_cost = 0.0
    for step_result in result.step_results:
        data = step_result.result_data or {}
        total_cost += data.get("total_cost_usd") or 0.0
        print(
            f"\nstep {step_result.step_number} ({step_result.step_title}): "
            f"success={step_result.success} session={data.get('session_id')} "
            f"tokens={step_result.token_usage.get('total', 0)}"
        )
        print(f"  answer: {step_result.result[:300]}")

    print(f"\nstructured output in memory: {context.memory_read('explanation', namespace='claude_code')}")
    print(f"total CLI cost: ${total_cost:.4f}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
