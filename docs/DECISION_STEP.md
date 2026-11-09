# Decision-model step

`DecisionStepDescription` (`step_type: "decision"`) asks a **decision model**
typed questions about one state and returns typed answers instead of text.

Decision models are a non-generative model family. They do not predict tokens
one by one: they read the whole state at once and choose an answer from the
answer space the caller defines. Examples served today:

| Provider value | Endpoint | Models |
| --- | --- | --- |
| `openai` | `POST https://api.openai.com/v1/decisions` | `gpt-6-luna` |
| `azure_openai` | `POST {base_url}/decisions`, `api-key` header | your `gpt-6-luna` deployment name |
| `openrouter` | `POST https://openrouter.ai/api/alpha/decisions` | `typesafe/jev-1.14`, `~typesafe/jev-latest`, `openai/gpt-6-luna-decisions`, … |

OpenRouter's Decisions API is marked alpha; the request shape may still change.

## Question types

| CARL type | OpenAI wire | OpenRouter / Jev wire | Decided `value` |
| --- | --- | --- | --- |
| `predicate` | `predicate` | `noul` | `probability >= threshold` (bool) |
| `choice` | `choice` + `choices[{value, description}]` | `choice` + `criteria{value: description}` | one of the choice values (str) |
| `score` | `score` + `levels[{label, description}]` | `score` + `criteria["label: description", …]` | probability-weighted level index (float) |

`noul` is accepted as an alias for `predicate` when loading a chain.

Jev's `box` question (image localisation) and image state parts are not wired
yet — state is text or JSON.

## Example

```python
from mmar_carl import (
    DecisionChoice, DecisionQuestion, DecisionStepConfig, DecisionStepDescription,
)

DecisionStepDescription(
    number=1,
    title="Triage ticket",
    config=DecisionStepConfig(
        provider="openrouter",             # or "openai" / "azure_openai"
        model="typesafe/jev-1.14",         # pin a version — see "Thresholds"
        state_source="$outer_context",     # or state_template + input_mapping
        questions=[
            DecisionQuestion(
                name="department",
                type="choice",
                instructions="Which department should handle this ticket?",
                choices=[
                    DecisionChoice(value="billing", description="Payments and refunds."),
                    DecisionChoice(value="other", description="Everything else."),
                ],
                min_confidence=0.7,
                fallback="other",
            ),
            DecisionQuestion(
                name="urgent", type="predicate",
                instructions="Is the customer blocked right now?", threshold=0.8,
            ),
        ],
        output_memory_key="ticket",        # → $memory.decision.ticket
    ),
)
```

Route on the answer with a conditional step:

```python
ConditionalStepConfig(
    condition_context_key="$memory.decision.ticket",
    branches=[ConditionalBranch(condition="value['department'] == 'billing'", next_step=3)],
    default_step=4,
)
```

A runnable version (offline with a canned response when no key is set):
`make example-decision`. Live check against real providers:
`PYTHONPATH=src uv run python scripts/validate_decision_step.py`.

## State

- `state_source` — a `$` reference (`$outer_context` by default). Dicts and
  string lists stay structured for OpenRouter; for the OpenAI protocol any
  non-string state is sent as JSON text.
- `state_template` + `input_mapping` — a `str.format` template rendered into a
  text state, same placeholder rules as `CodexStepConfig`.
- `max_state_bytes` (default 256 KiB) bounds the serialized state before any
  request is sent.

Every question in a step sees the same state. When a question depends on an
earlier answer, put it in a separate step.

## Output

- `result` — the primary question's decided value as text (`primary_question`,
  default the first question). Goes into history with a one-line summary of
  every answer.
- `result_data`:
  - `provider`, `requested_model`, `model` (the snapshot the provider served),
    `request_id`
  - `decision` — the primary decided value
  - `answers` — per question: `type`, `value`, `probabilities`
    (`{choice value | level label | "true"/"false": p}`), `confidence`,
    `probability` / `choice` / `score` + `level`, `refused`, `uncertain`,
    `fallback_applied`
  - `uncertain_questions`, raw `usage`, `cost_usd` (when the provider reports it)
- `token_usage` — provider-reported input and output token counts, when available.
- memory — with `output_memory_key`, `{question name: decided value}` in
  namespace `output_namespace` (default `decision`).

## Failure policy (fail-closed)

The step fails when:

- the API key env var is unset, the state is empty, or it exceeds `max_state_bytes`;
- the provider returns an HTTP error (429, 408 and 5xx are retried up to
  `max_retries` times with exponential backoff and `Retry-After`; 4xx auth and
  validation errors are not retried);
- an answer is malformed: an unknown choice, a probability outside [0, 1], a
  score outside the level range, a non-JSON body;
- an answer is **uncertain** — refused (or missing), or below `min_confidence`
  — and the question has no `fallback`, unless `fail_on_uncertain=False`.

With a `fallback` the uncertain answer is replaced and marked
`fallback_applied=True`; with `fail_on_uncertain=False` it is kept (a refusal
becomes `None`) and marked `uncertain=True`.

## Thresholds

Probabilities and confidences are not calibrated the same way across models,
and they move by a few hundredths between runs. Tune `threshold` and
`min_confidence` on labelled examples for the exact model you pin, set them on
bands rather than exact values, and re-tune after switching `model` or
`provider`. `result_data["model"]` records the served snapshot for that reason.

## Credentials and preflight

The host must authorize an exact pair of request URL and credential name, for
example `ReasoningContext(..., decision_endpoints={"openrouter":
("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY")})`.
The request URL includes the final `/decisions` path. This permission is
runtime-only and is checked before reading credentials or sending state.

The key is read from `api_key_env` at run time (defaults: `OPENAI_API_KEY`,
`AZURE_OPENAI_API_KEY`, `OPENROUTER_API_KEY`). Chains never serialize it, and
`api_key_env` rejects values that look like a key instead of a variable name.
`chain.preflight(ctx)` lists the variables under
`required_decision_credentials` and the unset ones under
`missing_decision_credentials` (names only, never values).

`base_url` must be https (plain http only for localhost) and must not embed
credentials, a query string or a fragment.

## Serialization

Adding the type bumps `ReasoningChain.FORMAT_VERSION` from 13 to 14. The
13 → 14 migration rewrites nothing; older runtimes reject the new type at the
chain boundary instead of failing on an unknown enum later.
