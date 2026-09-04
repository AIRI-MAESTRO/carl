"""One-call GLM smoke test: an LLM step feeds a static CommandStep.

The model produces data, not executable code. ``CommandStep`` then invokes a
predeclared Python argv and passes the LLM result through stdin. This is the
safe composition currently supported by CARL; CommandStep itself does not
call an LLM and cannot dynamically replace its executable.

Budget: one request, at most 256 completion tokens. The API key is read from
``OPENROUTER_API_KEY`` or from a non-echoing prompt and is never printed.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import sys

from mmar_carl import (
    CommandPolicy,
    CommandStepConfig,
    CommandStepDescription,
    Language,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    create_openai_client,
)

MODEL = "z-ai/glm-5.2"
STATIC_ANALYZER = (
    "import json,sys; "
    "text=sys.stdin.read().strip(); "
    "print(json.dumps({'text': text, 'chars': len(text), "
    "'words': len(text.split())}, ensure_ascii=False))"
)


def build_chain() -> ReasoningChain:
    return ReasoningChain(
        trace_name="GLM 5.2 -> CommandStep smoke",
        steps=[
            LLMStepDescription(
                number=1,
                title="Generate a short payload",
                aim=(
                    "Return exactly three lowercase English words on one line. "
                    "Return only the words, without quotes, numbering, or explanation."
                ),
                reasoning_questions="Which three harmless words satisfy the requested format?",
                stage_action="Emit the final three-word payload only.",
            ),
            CommandStepDescription(
                number=2,
                title="Analyze the LLM payload with a fixed command",
                dependencies=[1],
                config=CommandStepConfig(
                    command=[sys.executable, "-c", STATIC_ANALYZER],
                    stdin_source="$steps.1.result",
                    runtime="local",
                    # Local execution has host networking in reality. Declaring
                    # it avoids claiming an isolation guarantee that is absent.
                    network="host",
                    timeout=5.0,
                    max_output_bytes=4096,
                ),
            ),
        ],
    )


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key: ")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    client = create_openai_client(
        api_key=api_key,
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=256,
        timeout=60.0,
        extra_headers={"X-Title": "CARL CommandStep smoke test"},
    )
    context = ReasoningContext(
        outer_context="This is a minimal integration smoke test.",
        api=client,
        model=MODEL,
        retry_max=1,
        language=Language.ENGLISH,
        command_policy=CommandPolicy(
            allowed_executables=frozenset({sys.executable}),
            allowed_runtimes=frozenset({"local"}),
            allowed_networks=frozenset({"host"}),
            # The host owns this exact fixed analyzer invocation. Dynamic LLM
            # output is sent through stdin and never becomes Python source.
            require_approval_for_interpreters=False,
        ),
    )

    try:
        result = await build_chain().execute_async(context)
    finally:
        await client.close()

    print(f"model={MODEL}")
    print(f"success={result.success}")
    for step in result.step_results:
        print(f"step={step.step_number} type={step.step_type} success={step.success}")
        print(f"result={step.result!r}")
        if step.token_usage:
            print(f"token_usage={json.dumps(step.token_usage, sort_keys=True)}")
        if step.result_data is not None:
            print(f"result_data={json.dumps(step.result_data, sort_keys=True)}")
        if step.error_message:
            print(f"error={step.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
