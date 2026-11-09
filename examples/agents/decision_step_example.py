"""Decision-model step: triage a support ticket and route the chain on the answer.

Demonstrates ``DecisionStepDescription`` — a step that asks a decision model
(OpenAI Decisions API on ``gpt-6-luna``, or TypeSafe Jev through OpenRouter)
typed questions about the input instead of generating text:

1. Step 1 asks three questions about the ticket in one request: which
   department owns it (``choice``), whether a refund is requested
   (``predicate``), and how severe it is (``score``). Answers land in
   ``$memory.decision.ticket``.
2. Step 2 is a conditional step that routes on ``value['department']``.
3. Steps 3–4 are the per-department handlers (plain tool steps).

Provider selection (first match wins):

- ``OPENROUTER_API_KEY`` set → ``openrouter`` + ``typesafe/jev-1.14``
- ``OPENAI_API_KEY`` set     → ``openai`` + ``gpt-6-luna``
- neither                    → offline: a canned response through a mock transport

A live run sends one decision request (input tokens only; fractions of a cent).

Run: make example-decision
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx

from mmar_carl import (
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    DecisionChoice,
    DecisionLevel,
    DecisionQuestion,
    DecisionStepConfig,
    DecisionStepDescription,
    DecisionStepExecutor,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    register_executor,
)

TICKET = {
    "subject": "Charged twice",
    "body": "I was charged twice for order #4411. Please return the extra charge.",
    "customer_tier": "standard",
}

CANNED_ANSWERS = {
    "answers": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 0.94},
                {"value": "technical", "probability": 0.03},
                {"value": "other", "probability": 0.03},
            ],
            "confidence": 0.91,
        },
        {"type": "predicate", "name": "refund_requested", "probability": 0.97},
        {
            "type": "score",
            "name": "severity",
            "score": 1.2,
            "probabilities": [
                {"value": 0, "label": "Cosmetic", "probability": 0.1},
                {"value": 1, "label": "Degraded", "probability": 0.6},
                {"value": 2, "label": "Blocked", "probability": 0.3},
            ],
            "confidence": 0.6,
        },
    ],
    "usage": {"input_tokens": 210, "output_tokens": 0, "total_tokens": 210},
}


def pick_provider() -> tuple[str, str, bool]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter", "typesafe/jev-1.14", True
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-6-luna", True
    return "openai", "gpt-6-luna", False


def build_chain(provider: str, model: str) -> ReasoningChain:
    return ReasoningChain(
        steps=[
            DecisionStepDescription(
                number=1,
                title="Triage ticket",
                config=DecisionStepConfig(
                    provider=provider,
                    model=model,
                    state_source="$outer_context",
                    questions=[
                        DecisionQuestion(
                            name="department",
                            type="choice",
                            instructions="Which department should handle this ticket?",
                            choices=[
                                DecisionChoice(value="billing", description="Payments, invoices and refunds."),
                                DecisionChoice(value="technical", description="Product bugs and outages."),
                                DecisionChoice(value="other", description="Anything else."),
                            ],
                            min_confidence=0.7,
                            fallback="other",  # low confidence → human-owned queue
                        ),
                        DecisionQuestion(
                            name="refund_requested",
                            type="predicate",
                            instructions="Does the customer ask for money back?",
                            threshold=0.8,
                        ),
                        DecisionQuestion(
                            name="severity",
                            type="score",
                            instructions="How badly is the customer blocked?",
                            levels=[
                                DecisionLevel(label="Cosmetic", description="Nothing is blocked."),
                                DecisionLevel(label="Degraded", description="A workaround exists."),
                                DecisionLevel(label="Blocked", description="No workaround."),
                            ],
                        ),
                    ],
                    output_memory_key="ticket",
                ),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route by department",
                dependencies=[1],
                config=ConditionalStepConfig(
                    condition_context_key="$memory.decision.ticket",
                    branches=[
                        ConditionalBranch(condition="value['department'] == 'billing'", next_step=3),
                        ConditionalBranch(condition="value['department'] == 'technical'", next_step=4),
                    ],
                    default_step=4,
                ),
            ),
            ToolStepDescription(
                number=3, title="Billing queue", config=ToolStepConfig(tool_name="billing_queue", input_mapping={})
            ),
            ToolStepDescription(
                number=4, title="Support queue", config=ToolStepConfig(tool_name="support_queue", input_mapping={})
            ),
        ],
        max_workers=1,
    )


async def main() -> None:
    provider, model, live = pick_provider()
    if not live:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=CANNED_ANSWERS))
        register_executor(StepType.DECISION, DecisionStepExecutor(transport=transport))
        os.environ.setdefault("OPENAI_API_KEY", "offline-demo")
        print("No OPENROUTER_API_KEY / OPENAI_API_KEY — running offline with a canned decision.\n")
    else:
        print(f"Live run: provider={provider} model={model}\n")

    chain = build_chain(provider, model)
    endpoint, key_env = {
        "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
    }[provider]
    context = ReasoningContext(
        outer_context=json.dumps(TICKET), api=None, model="none", language=Language.ENGLISH,
        decision_endpoints={provider: (endpoint, key_env)},
    )
    context.register_tool("billing_queue", lambda: "queued for billing")
    context.register_tool("support_queue", lambda: "queued for support")

    print(chain.preflight(context).format_text())
    result = await chain.execute_async(context)

    decision = next(sr for sr in result.step_results if sr.step_number == 1)
    if not decision.success:
        print(f"Decision step failed: {decision.error_message}")
        return
    print(f"Served model: {decision.model}")
    for name, answer in decision.result_data["answers"].items():
        print(f"  {name:17} → {answer['value']!r}  confidence={answer.get('confidence')}  "
              f"uncertain={answer['uncertain']}")
    routed = next(sr for sr in result.step_results if sr.step_number == 2)
    print(f"Routed to step {routed.result_data['next_step']}; chain success={result.success}")


if __name__ == "__main__":
    asyncio.run(main())
