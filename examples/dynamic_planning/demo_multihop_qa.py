"""Multi-hop QA benchmark demo for the dynamic-planning solution.

Compares three methods on a HotpotQA (distractor) sample:
  1. single_shot   — one LLM call with all 10 paragraphs in the prompt.
  2. static_chain  — a fixed prescriptive CARL chain (search -> follow-up query -> search -> answer).
  3. dynamic_chain — the DynamicChain prototype: per-round planner + OCC state cards.

Prints a metrics table (EM / F1 / tokens / latency / LLM calls) and writes results.json.

Usage:
    PYTHONPATH=src:examples/dynamic_planning uv run python examples/dynamic_planning/demo_multihop_qa.py --n 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

def _load_env(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency in the base env)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(Path(__file__).resolve().parents[2] / ".env")
os.environ["ENABLE_LANGFUSE"] = "false"  # demo: avoid optional-dep warning per chain

import logging  # noqa: E402

logging.getLogger("mmar_carl").setLevel(logging.WARNING)

from mmar_carl import (  # noqa: E402
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    ToolStepDescription,
)
from mmar_carl.llm import RetryPolicy  # noqa: E402
from mmar_carl.models.config import ToolStepConfig  # noqa: E402

from dynamic_chain_proto import DynamicChain, strip_think  # noqa: E402
from multihop_data import load_cases  # noqa: E402

# --------------------------------------------------------------------- scoring

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    text = strip_think(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    p_tokens, g_tokens = normalize(pred).split(), normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p_tokens), overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text: str) -> str:
    """Prefer the last 'ANSWER: <span>' marker; fall back to the last short line."""
    text = strip_think(text)
    marked = re.findall(r"ANSWER\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip().strip('"').rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    ans = lines[-1] if len(lines[-1]) <= 100 or len(lines) == 1 else lines[0]
    return ans.strip().strip('"')


# --------------------------------------------------------------------- retrieval tool

_STOP = set("a an the of in on at to for with by is are was were be and or what which who whom whose when where how did does do".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def make_search_tool(paragraphs: list[dict], top_k: int = 2):
    """Token-overlap retrieval over this question's 10 paragraphs (titles boosted)."""

    def search_paragraphs(query: str = "") -> str:
        q = str(query)
        # strip history-entry wrappers like "Step 2. title\nResult: ..." and quotes
        q = re.sub(r"Step \d+\..*?Result:\s*", "", q, flags=re.DOTALL)
        q_tokens = _tokens(strip_think(q))
        if not q_tokens:
            return "No query given."
        scored = []
        for p in paragraphs:
            score = len(q_tokens & _tokens(p["text"])) + 2 * len(q_tokens & _tokens(p["title"]))
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, p in scored[:top_k]:
            if score <= 0:
                continue
            out.append(f"### {p['title']}\n{p['text'][:1400]}")
        return "\n\n".join(out) or "No matching paragraphs."

    return search_paragraphs


# --------------------------------------------------------------------- methods


def make_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        OpenAIClientConfig(
            model=os.environ.get("DEMO_MODEL", os.environ.get("DEFAULT_EXAMPLES_MODEL", "qwen/qwen3-8b")),
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=1200,
            timeout=90.0,  # default 20s is too tight for thinking models on long prompts
            retry_policy=RetryPolicy(max_attempts=4),
        )
    )


async def run_single_shot(case: dict, client) -> dict:
    ctx_text = "\n\n".join(f"### {p['title']}\n{p['text']}" for p in case["paragraphs"])
    prompt = (
        "Answer the question using the paragraphs. End your reply with a final line of the form "
        "'ANSWER: <exact short answer span or yes/no>'.\n\nPARAGRAPHS:\n"
        + ctx_text + "\n\nQUESTION: " + case["question"]
    )
    t0 = time.monotonic()
    reply, usage = await client.get_response_with_usage(prompt)
    return {
        "answer": extract_answer(reply),
        "tokens": int(usage.get("total", usage.get("total_tokens", 0)) or 0),
        "wall_s": time.monotonic() - t0,
        "llm_calls": 1,
        "steps": 1,
    }


def build_static_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Search with the question",
                config=ToolStepConfig(tool_name="search_paragraphs", input_mapping={"query": "$outer_context"}),
            ),
            LLMStepDescription(
                number=2,
                title="Plan follow-up lookup",
                dependencies=[1],
                aim=(
                    "The question is multi-hop. From the retrieved paragraphs identify the bridge "
                    "entity or missing fact still needed. Output ONLY a short search query, nothing else."
                ),
            ),
            ToolStepDescription(
                number=3,
                title="Search follow-up",
                dependencies=[2],
                config=ToolStepConfig(tool_name="search_paragraphs", input_mapping={"query": "$history[-1]"}),
            ),
            LLMStepDescription(
                number=4,
                title="Answer",
                dependencies=[3],
                aim=(
                    "Answer the original question using everything retrieved. End your reply with "
                    "a final line of the form 'ANSWER: <exact short answer span or yes/no>'."
                ),
            ),
        ]
    )


async def run_static(case: dict, client) -> dict:
    ctx = ReasoningContext(outer_context=case["question"], api=client)
    ctx.register_tool("search_paragraphs", make_search_tool(case["paragraphs"]))
    chain = build_static_chain()
    t0 = time.monotonic()
    result = await chain.execute_async(ctx)
    usage = result.token_usage or {}
    return {
        "answer": extract_answer(result.get_final_output()),
        "tokens": int(usage.get("total", usage.get("total_tokens", 0)) or 0),
        "wall_s": time.monotonic() - t0,
        "llm_calls": 2,
        "steps": len(result.step_results),
    }


async def run_dynamic(case: dict, client) -> dict:
    ctx = ReasoningContext(outer_context=case["question"], api=client)
    ctx.register_tool("search_paragraphs", make_search_tool(case["paragraphs"]))
    engine = DynamicChain(
        goal=(
            "Answer this multi-hop question with the exact short answer span (or yes/no): "
            + case["question"]
        ),
        llm_client=client,
        tools_doc=(
            "- search_paragraphs(query: str) -> str : keyword search over this question's 10 "
            "wikipedia paragraphs; returns the top-2 matching paragraphs. Use short entity-style "
            "queries passed as a quoted literal, e.g. {\"query\": \"'Kiss and Tell 1945 film'\"}."
        ),
        allowed_tools=("search_paragraphs",),
        max_rounds=4,
        max_steps_per_round=3,
        spec_retries=2,
    )
    t0 = time.monotonic()
    stats = await engine.execute_async(ctx)
    out = {
        "answer": extract_answer(stats.final_answer),
        "tokens": stats.total_tokens,
        "wall_s": time.monotonic() - t0,
        "llm_calls": stats.llm_calls,
        "steps": stats.steps_executed,
        "rounds": stats.rounds,
        "repairs": stats.planner_repairs,
        "stop_reason": stats.stop_reason,
        "facts": stats.facts_kept,
    }
    try:
        harvested = engine.harvest()
        out["harvested_steps"] = len(harvested.steps)
        out["harvested_spec"] = harvested.to_dict()
        out["harvested_mermaid"] = harvested.to_mermaid()
    except Exception as exc:  # harvest is a bonus artifact; never fail the run on it
        out["harvest_error"] = str(exc)[:200]
    return out


METHODS = {
    "single_shot": run_single_shot,
    "static_chain": run_static,
    "dynamic_chain": run_dynamic,
}

# --------------------------------------------------------------------- harness


async def evaluate(n: int, methods: list[str], concurrency: int) -> dict:
    cases = load_cases(n)
    client = make_client()
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, list[dict]] = {m: [] for m in methods}

    async def one(method: str, case: dict) -> None:
        async with sem:
            try:
                r = await METHODS[method](case, client)
            except Exception as exc:
                r = {"answer": "", "tokens": 0, "wall_s": 0.0, "llm_calls": 0, "steps": 0,
                     "error": f"{type(exc).__name__}: {exc}"}
            r["case_id"] = case["id"]
            r["gold"] = case["answer"]
            r["em"] = em_score(r["answer"], case["answer"])
            r["f1"] = f1_score(r["answer"], case["answer"])
            results[method].append(r)
            mark = "✓" if r["em"] else ("~" if r["f1"] > 0.4 else "✗")
            print(f"  {mark} [{method}] {case['question'][:70]}  ->  {r['answer'][:40]!r} "
                  f"(gold {case['answer'][:30]!r})", flush=True)

    for method in methods:
        print(f"\n=== {method} over {len(cases)} cases ===", flush=True)
        await asyncio.gather(*(one(method, c) for c in cases))
    return results


def summarize(results: dict) -> str:
    header = ("| Method | EM | F1 | Avg tokens | Avg time (s) | Avg LLM calls | Avg steps | Errors |\n"
              "|---|---|---|---|---|---|---|---|")
    lines = [header]
    for method, rows in results.items():
        n = len(rows)
        errs = sum(1 for r in rows if r.get("error"))
        lines.append(
            "| {m} | {em:.2f} | {f1:.2f} | {tok:.0f} | {t:.1f} | {calls:.1f} | {steps:.1f} | {e} |".format(
                m=method,
                em=sum(r["em"] for r in rows) / n,
                f1=sum(r["f1"] for r in rows) / n,
                tok=sum(r["tokens"] for r in rows) / n,
                t=sum(r["wall_s"] for r in rows) / n,
                calls=sum(r["llm_calls"] for r in rows) / n,
                steps=sum(r["steps"] for r in rows) / n,
                e=errs,
            )
        )
    return "\n".join(lines)


def summarize_dynamic(rows: list[dict]) -> str:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return "(all dynamic runs errored)"
    reasons = Counter(r.get("stop_reason", "?") for r in ok)
    return (
        f"dynamic detail: avg rounds {sum(r.get('rounds', 0) for r in ok)/len(ok):.1f}, "
        f"avg planner repairs {sum(r.get('repairs', 0) for r in ok)/len(ok):.2f}, "
        f"avg facts kept {sum(r.get('facts', 0) for r in ok)/len(ok):.1f}, "
        f"stop reasons {dict(reasons)}, "
        f"harvested (avg steps) {sum(r.get('harvested_steps', 0) for r in ok)/len(ok):.1f}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--methods", default="single_shot,static_chain,dynamic_chain")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]
    results = await evaluate(args.n, methods, args.concurrency)
    print("\n" + summarize(results))
    if "dynamic_chain" in results:
        print("\n" + summarize_dynamic(results["dynamic_chain"]))
    out = Path(__file__).parent / args.out
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
