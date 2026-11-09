"""DynamicChain prototype — state-driven per-round step planning on top of the untouched DAGExecutor.

Implements design "A" from the dynamic-planning design work: each round a planner LLM
looks at bounded OCC-style state cards ($memory.occ.view) plus a trajectory digest and
emits 1..K step specs (vocabulary = existing CARL step types). Specs are validated with
``meta.validate_spec`` (with ``format_for_llm`` repair retries), renumbered to ``r*100+i``
so the ``metadata["step_N"]`` mirrors never collide across rounds, built via
``meta.validate_and_build``, and executed as a small per-round ReasoningChain against ONE
persistent ReasoningContext. The executed trajectory is harvestable as an ordinary static
ReasoningChain via :meth:`DynamicChain.harvest`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from mmar_carl import ReasoningContext
from mmar_carl.meta import get_step_catalog, validate_and_build, validate_spec

_FENCE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by hybrid-reasoning models (Qwen3 etc.)."""
    return _THINK.sub("", text).strip()


class DynamicPlanError(RuntimeError):
    pass


@dataclass
class PlannerTurn:
    round_no: int
    attempt: int
    reply: str
    error: Optional[str]
    accepted_steps: list[dict]


@dataclass
class RoundLog:
    round_no: int
    step_dicts: list[dict]
    success_by_step: dict[int, bool]


@dataclass
class DynamicRunStats:
    final_answer: str = ""
    stop_reason: str = ""
    rounds: int = 0
    steps_executed: int = 0
    llm_calls: int = 0
    planner_repairs: int = 0
    total_tokens: int = 0
    wall_s: float = 0.0
    facts_kept: int = 0


class DynamicChain:
    """Round-loop incremental planner engine (prototype for the demo benchmark)."""

    def __init__(
        self,
        goal: str,
        llm_client: Any,
        *,
        tools_doc: str = "",
        allowed_tools: tuple[str, ...] = (),
        vocabulary: tuple[str, ...] = ("tool", "llm"),
        max_rounds: int = 5,
        max_steps_per_round: int = 3,
        spec_retries: int = 2,
        max_facts: int = 10,
        ns: str = "occ",
    ) -> None:
        self.goal = goal
        self.client = llm_client
        self.tools_doc = tools_doc
        self.allowed_tools = set(allowed_tools)
        self.vocab = list(vocabulary)
        self.max_rounds = max_rounds
        self.k = max_steps_per_round
        self.spec_retries = spec_retries
        self.max_facts = max_facts
        self.ns = ns
        self._rounds: list[RoundLog] = []
        self._turns: list[PlannerTurn] = []
        self._digest_lines: list[str] = []
        self.stats = DynamicRunStats()

    # ------------------------------------------------------------------ prompt

    def _round_prompt(self, cards: dict, round_no: int) -> str:
        cat = get_step_catalog()
        menu = "\n".join(f"- {n}: {cat.step(n).purpose}" for n in self.vocab)
        digest = "\n".join(self._digest_lines[-12:]) or "(nothing executed yet)"
        return "\n\n".join(
            [
                "You are an incremental planner for a running reasoning task. Plan ONLY the next "
                f"round: 1..{self.k} steps, or finish. Reply with ONE JSON object, no prose:\n"
                '{"steps": [...], "state_update": {"plan": [".."], "facts_add": [".."]},\n'
                ' "done": false, "final_answer": null}\n'
                'Set "done": true with "steps": [] and a non-null "final_answer" (the exact short '
                "answer only) once the facts already answer the goal.",
                "Rules: within this round number steps 1..K with dependencies among THIS round only. "
                "Steps see the full execution history automatically. To feed an earlier result into a "
                'tool input use "$history[-1]" (last output), a quoted literal like "\'some text\'", '
                'or "$steps.<GLOBAL_N>.result" with a GLOBAL number from the trajectory below. '
                "Do not invent tools or step types.",
                "Allowed step types (ONLY these):\n" + menu,
                "Step type details:\n\n" + cat.tier1(include=self.vocab),
                "Available registered tools:\n" + (self.tools_doc or "(none)"),
                "GOAL:\n" + self.goal,
                "STATE CARDS ($memory.%s.view):\n" % self.ns + json.dumps(cards, ensure_ascii=False),
                "EXECUTED TRAJECTORY (global step numbers):\n" + digest,
                f"This is round {round_no} of at most {self.max_rounds}. Prefer finishing early: "
                "if the answer is already in the facts, stop now.",
            ]
        )

    # ------------------------------------------------------------------ planning

    async def _ask(self, prompt: str) -> str:
        fn = getattr(self.client, "get_response_with_usage", None)
        last_exc: Optional[Exception] = None
        for attempt in range(3):  # connection blips are common under concurrency
            try:
                if fn is not None:
                    reply, usage = await fn(prompt)
                    self.stats.total_tokens += int(
                        usage.get("total", usage.get("total_tokens", 0)) or 0
                    )
                else:
                    reply = await self.client.get_response(prompt)
                self.stats.llm_calls += 1
                return strip_think(reply)
            except Exception as exc:  # noqa: BLE001 — retry transient transport errors only
                if "connection" not in str(exc).lower() and "timeout" not in str(exc).lower():
                    raise
                last_exc = exc
                await asyncio.sleep(2.0 * (attempt + 1))
        raise DynamicPlanError(f"LLM unreachable after retries: {last_exc}")

    @staticmethod
    def _parse_json(reply: str) -> dict:
        text = _FENCE.sub(r"\1", reply.strip())
        start = text.find("{")
        if start == -1:
            raise ValueError("reply contains no JSON object")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError("unbalanced JSON object in reply")

    def _renumber(self, steps: list[dict], round_no: int) -> list[dict]:
        remap = {int(s["number"]): round_no * 100 + i for i, s in enumerate(steps, 1)}
        for s in steps:
            s["number"] = remap[int(s["number"])]
            s["dependencies"] = [remap.get(int(d), int(d)) for d in (s.get("dependencies") or [])]
        return steps

    async def _plan_round(self, ctx: ReasoningContext, round_no: int) -> tuple[Optional[list[dict]], dict]:
        cards = ctx.memory.get(self.ns, {}).get("view", {"goal": self.goal, "plan": [], "facts": []})
        base = prompt = self._round_prompt(cards, round_no)
        last_exc: Optional[str] = None
        for attempt in range(self.spec_retries + 1):
            reply = await self._ask(prompt)
            try:
                obj = self._parse_json(reply)
                steps = obj.get("steps") or []
                if obj.get("done") and not steps:
                    self._turns.append(PlannerTurn(round_no, attempt, reply, None, []))
                    return None, obj
                if len(steps) > self.k:
                    steps = steps[: self.k]
                # heal: assign missing local numbers; drop cross-round numeric deps
                # (prior-round context flows via $history/$memory/$steps refs instead)
                for i, s in enumerate(steps, 1):
                    s.setdefault("number", i)
                local = {int(s["number"]) for s in steps}
                for s in steps:
                    s["dependencies"] = [d for d in (s.get("dependencies") or []) if int(d) in local]
                bad = [s.get("step_type") for s in steps if s.get("step_type") not in self.vocab]
                if bad:
                    raise ValueError(f"step_type(s) {bad} are not in the allowed list {self.vocab}")
                if self.allowed_tools:
                    bad_tools = [
                        (s.get("step_config") or {}).get("tool_name")
                        for s in steps
                        if s.get("step_type") == "tool"
                        and (s.get("step_config") or {}).get("tool_name") not in self.allowed_tools
                    ]
                    if bad_tools:
                        raise ValueError(
                            f"tool_name(s) {bad_tools} are not registered; "
                            f"use one of {sorted(self.allowed_tools)}"
                        )
                report = validate_spec(list(steps))  # LIST form => dep/cycle checks run
                if not report.valid:
                    raise ValueError("\n".join(i.format_for_llm() for i in report.errors))
                self._turns.append(PlannerTurn(round_no, attempt, reply, None, steps))
                return self._renumber([dict(s) for s in steps], round_no), obj
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_exc = str(exc)
                self.stats.planner_repairs += 1
                self._turns.append(PlannerTurn(round_no, attempt, reply, last_exc, []))
                prompt = (
                    base
                    + "\n\n---\nYour previous reply was rejected:\n"
                    + last_exc
                    + "\nReply again with corrected JSON only."
                )
        raise DynamicPlanError(f"planner failed after {self.spec_retries + 1} attempts: {last_exc}")

    # ------------------------------------------------------------------ state cards

    def _merge_state(self, ctx: ReasoningContext, update: dict) -> None:
        cards = ctx.memory.get(self.ns, {}).get("view", {"goal": self.goal, "plan": [], "facts": []})
        cards = {
            "goal": cards.get("goal", self.goal),
            "plan": list(update.get("plan") or cards.get("plan") or [])[:6],
            "facts": list(cards.get("facts") or []),
        }
        for fact in update.get("facts_add") or []:
            fact = str(fact)[:300]
            if fact and fact not in cards["facts"]:
                cards["facts"].append(fact)
        cards["facts"] = cards["facts"][-self.max_facts :]
        ctx.memory_write("view", cards, namespace=self.ns)
        self.stats.facts_kept = len(cards["facts"])

    # ------------------------------------------------------------------ main loop

    async def execute_async(self, ctx: ReasoningContext) -> DynamicRunStats:
        t0 = time.monotonic()
        r = 0
        while r < self.max_rounds:
            r += 1
            try:
                steps, verdict = await self._plan_round(ctx, r)
            except DynamicPlanError:
                self.stats.stop_reason = "planner_error"  # degrade to finalize, don't fail the run
                break
            self._merge_state(ctx, verdict.get("state_update") or {})
            if steps is None:
                self.stats.stop_reason = "planner_stop"
                self.stats.final_answer = strip_think(str(verdict.get("final_answer") or ""))
                break
            report, round_chain = validate_and_build({"steps": steps}, use_typed_steps=True, quiet=True)
            if not report.valid:
                raise DynamicPlanError(report.format_text())
            result = await round_chain.execute_async(ctx)
            usage = result.token_usage or {}
            self.stats.total_tokens += int(usage.get("total", usage.get("total_tokens", 0)) or 0)
            self.stats.llm_calls += sum(
                1 for sr in result.step_results if getattr(sr, "step_type", "") == "llm"
            )
            self.stats.steps_executed += len(result.step_results)
            success_by_step: dict[int, bool] = {}
            for sr in result.step_results:
                success_by_step[sr.step_number] = sr.success
                out = strip_think(str(sr.result or sr.error_message or ""))[:500]
                self._digest_lines.append(
                    f"[{sr.step_number}] {sr.step_title} ({sr.step_type}) ok={sr.success}: {out}"
                )
            self._rounds.append(RoundLog(r, steps, success_by_step))
        self.stats.rounds = r
        if not self.stats.final_answer:  # budget exhausted -> finalize explicitly
            self.stats.stop_reason = self.stats.stop_reason or "budget"
            cards = ctx.memory.get(self.ns, {}).get("view", {})
            reply = await self._ask(
                "Answer the goal using ONLY the gathered facts. Output the exact short answer "
                "span only, no explanation.\nGOAL:\n" + self.goal
                + "\nFACTS:\n" + json.dumps(cards, ensure_ascii=False)
                + "\nTRAJECTORY:\n" + "\n".join(self._digest_lines[-8:])
            )
            self.stats.final_answer = reply.strip()
        self.stats.wall_s = time.monotonic() - t0
        return self.stats

    # ------------------------------------------------------------------ harvest

    def harvest(self) -> Any:
        """Stitch the executed trajectory into an ordinary static ReasoningChain."""
        steps: list[dict] = []
        prev: list[int] = []
        for rnd in self._rounds:
            for d in rnd.step_dicts:
                d = dict(d)
                if not d.get("dependencies") and prev:
                    d["dependencies"] = list(prev)
                steps.append(d)
            prev = [s["number"] for s in rnd.step_dicts]
        spec = {
            "steps": steps,
            "metadata": {
                "dynamic_provenance": {
                    "goal": self.goal,
                    "rounds": len(self._rounds),
                    "planner_turns": len(self._turns),
                }
            },
        }
        report, chain = validate_and_build(spec, use_typed_steps=True, quiet=True)
        if not report.valid:
            raise DynamicPlanError(report.format_text())
        return chain
