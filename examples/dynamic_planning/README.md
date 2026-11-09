# Dynamic planning demo — multi-hop QA benchmark

Assesses the **dynamic next-step planning** solution (design "A": `DynamicChain` engine on
top of the untouched `DAGExecutor`) on a sample of **HotpotQA (distractor)** validation
questions, against two baselines.

| File | Purpose |
|---|---|
| `dynamic_chain_proto.py` | `DynamicChain` prototype: per-round planner over OCC state cards (`$memory.occ.view`), step vocabulary = existing CARL step types, `meta.validate_spec` + `format_for_llm` repair retries, `r*100+i` round numbering, per-round mini-chains on one persistent `ReasoningContext`, `harvest()` → static `ReasoningChain`. |
| `multihop_data.py` | Fetches N HotpotQA distractor validation cases via the HF datasets-server rows API; caches to `data/hotpot_sample.json` (offline afterwards). |
| `demo_multihop_qa.py` | Benchmark harness: 3 methods × N cases, EM / F1 / tokens / latency / LLM calls table; writes `results.json`. |

## Methods

1. **`single_shot`** (baseline) — one LLM call with all 10 paragraphs in the prompt.
2. **`static_chain`** (baseline) — a *prescriptive* CARL chain fixed up front:
   search(question) → LLM plans follow-up query → search → answer.
3. **`dynamic_chain`** (the solution) — planner LLM sees bounded state cards + trajectory
   digest each round, emits 1..3 validated step specs (`tool` / `llm`) or stops with the
   final answer; environment = a `search_paragraphs` retrieval tool over the case's
   paragraphs.

## Run

```bash
PYTHONPATH=src:examples/dynamic_planning uv run python examples/dynamic_planning/demo_multihop_qa.py --n 12
```

Requires `OPENAI_API_KEY` / `OPENAI_BASE_URL` in the repo `.env` (OpenRouter). Model comes
from `DEMO_MODEL` or `DEFAULT_EXAMPLES_MODEL` (default `qwen/qwen3-8b`; note this model
spends "thinking" tokens, which are included in the token column).

## Metrics

- **EM** — exact match after HotpotQA answer normalization (lowercase, strip articles/punctuation).
- **F1** — token-level overlap F1 against the gold answer.
- **Avg tokens** — total LLM tokens per question (prompt + completion, incl. planner calls).
- **Avg LLM calls / steps** — model invocations and executed chain steps per question.
- Dynamic-only: rounds per question, planner JSON-repair rate, stop-reason breakdown,
  harvested-chain size (the executed trajectory re-validated as a static chain).
