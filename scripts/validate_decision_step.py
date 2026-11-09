"""Live validation for the decision-model step.

Sends one real decision request per configured provider and checks that the
answers come back typed and in range:

- ``OPENAI_API_KEY``     → OpenAI Decisions API, ``gpt-6-luna``
- ``OPENROUTER_API_KEY`` → OpenRouter Decisions API, ``typesafe/jev-1.14``
- ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_DECISIONS_BASE_URL``, and
  ``AZURE_OPENAI_DECISIONS_MODEL`` → the host's Azure Decisions deployment

Providers without a key are skipped. Requests use a short synthetic ticket and are billed by the configured provider.

Run: PYTHONPATH=src uv run python scripts/validate_decision_step.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mmar_carl import (
    DecisionChoice,
    DecisionLevel,
    DecisionQuestion,
    DecisionStepConfig,
    DecisionStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
)

ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
}

PROVIDERS = [
    ("openai", "gpt-6-luna", "OPENAI_API_KEY"),
    ("openrouter", "typesafe/jev-1.14", "OPENROUTER_API_KEY"),
]


def _chain(provider: str, model: str, base_url: str | None) -> ReasoningChain:
    return ReasoningChain(steps=[
        DecisionStepDescription(
            number=1,
            title="Classify sentiment",
            config=DecisionStepConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                max_retries=0,
                questions=[
                    DecisionQuestion(
                        name="sentiment",
                        type="choice",
                        instructions="What is the sentiment of the review?",
                        choices=[
                            DecisionChoice(value="positive", description="The reviewer is satisfied."),
                            DecisionChoice(value="negative", description="The reviewer is dissatisfied."),
                        ],
                    ),
                    DecisionQuestion(
                        name="mentions_delivery", type="predicate",
                        instructions="Does the review mention delivery or shipping?",
                    ),
                    DecisionQuestion(
                        name="severity", type="score",
                        instructions="Rate the severity of the complaint.",
                        levels=[DecisionLevel(label="minor"), DecisionLevel(label="severe")],
                    ),
                ],
            ),
        ),
    ])


async def _validate(provider: str, model: str, base_url: str | None = None) -> bool:
    endpoint = ENDPOINTS[provider] if base_url is None else (
        base_url.rstrip("/") + "/decisions", "AZURE_OPENAI_API_KEY",
    )
    context = ReasoningContext(
        outer_context="Arrived two weeks late and the box was crushed. Never ordering again.",
        api=None, model="none", language=Language.ENGLISH,
        decision_endpoints={provider: endpoint},
    )
    result = await _chain(provider, model, base_url).execute_async(context)
    step = result.step_results[0]
    if not step.success:
        print(f"[{provider}] FAIL: {step.error_message}")
        return False
    answers = step.result_data["answers"]
    ok = answers["sentiment"]["value"] == "negative" and answers["mentions_delivery"]["value"] is True
    ok = ok and 0.0 <= answers["severity"]["value"] <= 1.0
    print(f"[{provider}] {'OK' if ok else 'UNEXPECTED'}: model={step.model} answers={answers} "
          f"usage={step.result_data['usage']}")
    return ok


async def main() -> int:
    ran = False
    ok = True
    for provider, model, key_env in PROVIDERS:
        if not os.environ.get(key_env):
            print(f"[{provider}] skipped: {key_env} is not set")
            continue
        ran = True
        ok = await _validate(provider, model) and ok
    azure_url = os.environ.get("AZURE_OPENAI_DECISIONS_BASE_URL")
    azure_model = os.environ.get("AZURE_OPENAI_DECISIONS_MODEL")
    if os.environ.get("AZURE_OPENAI_API_KEY") and azure_url and azure_model:
        ran = True
        ok = await _validate("azure_openai", azure_model, azure_url) and ok
    else:
        print("[azure_openai] skipped: key, base URL or deployment is unset")
    if not ran:
        print("No provider keys set — nothing validated.")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
