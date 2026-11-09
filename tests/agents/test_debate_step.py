"""
Tests for ``DebateStepDescription``.

A debate step runs ``len(roles) * rounds`` LLM calls in strict round-robin
order, then a single judge synthesis call. The transcript and verdict are
both surfaced in ``result_data``. Also covers the opt-in ``judge_llm_config``
override and the opt-in convergence criteria (``stop_when_unchanged`` /
``convergence_predicate``), both of which leave default behaviour unchanged.
"""

import pytest

from mmar_carl import (
    DebateStepConfig,
    DebateStepDescription,
    Language,
    LLMClientBase,
    LLMStepConfig,
    ReasoningChain,
    ReasoningContext,
)


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #


class _ScriptedLLM(LLMClientBase):
    """Returns canned replies in order; records every prompt seen."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.i = 0

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.i >= len(self._replies):
            raise RuntimeError(f"Ran out of canned replies after {self.i} calls")
        reply = self._replies[self.i]
        self.i += 1
        return reply

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _basic_debate(
    *,
    replies: list[str],
    roles: list[str] = ["proponent", "critic"],
    rounds: int = 2,
    role_prompts: dict[str, str] | None = None,
    role_llm_configs: dict[str, LLMStepConfig] | None = None,
    judge_prompt: str = "Topic: {task}\n\nDebate:\n{transcript}\n\nVerdict:",
    output_memory_key: str = "verdict",
    task: str = "Should we add caching?",
    judge_llm_config: LLMStepConfig | None = None,
    stop_when_unchanged: int | None = None,
    convergence_predicate=None,
) -> tuple[ReasoningChain, ReasoningContext, _ScriptedLLM]:
    llm = _ScriptedLLM(replies=replies)
    chain = ReasoningChain(
        steps=[
            DebateStepDescription(
                number=1,
                title="Debate",
                config=DebateStepConfig(
                    roles=roles,
                    rounds=rounds,
                    role_prompts=role_prompts or {},
                    role_llm_configs=role_llm_configs or {},
                    judge_prompt=judge_prompt,
                    output_memory_key=output_memory_key,
                    judge_llm_config=judge_llm_config,
                    stop_when_unchanged=stop_when_unchanged,
                    convergence_predicate=convergence_predicate,
                ),
            )
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context=task, api=llm, language=Language.ENGLISH)
    return chain, ctx, llm


# --------------------------------------------------------------------------- #
# Call counts and turn order
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_total_llm_call_count_matches_formula() -> None:
    """``len(roles) * rounds + 1`` total LLM calls (last is judge)."""
    replies = [f"reply-{i}" for i in range(7)]  # 2 roles * 3 rounds + 1 judge
    chain, ctx, llm = _basic_debate(replies=replies, rounds=3)
    await chain.execute_async(ctx)
    assert llm.i == 7
    # 6 role calls + 1 judge
    role_calls = len([p for p in llm.prompts if "Topic: " in p and "Verdict:" not in p])
    judge_calls = len([p for p in llm.prompts if p.endswith("Verdict:")])
    assert role_calls == 6
    assert judge_calls == 1


@pytest.mark.asyncio
async def test_round_robin_order_proponent_critic_proponent_critic() -> None:
    replies = ["A", "B", "C", "D", "FINAL"]
    chain, ctx, _ = _basic_debate(replies=replies, rounds=2)
    result = await chain.execute_async(ctx)
    transcript = result.step_results[0].result_data["transcript"]
    assert [t["role"] for t in transcript] == ["proponent", "critic", "proponent", "critic"]
    assert [t["round"] for t in transcript] == [1, 1, 2, 2]
    assert [t["argument"] for t in transcript] == ["A", "B", "C", "D"]


@pytest.mark.asyncio
async def test_three_role_round_robin() -> None:
    replies = ["a", "b", "c", "a2", "b2", "c2", "FINAL"]
    chain, ctx, _ = _basic_debate(
        replies=replies, roles=["alpha", "beta", "gamma"], rounds=2
    )
    result = await chain.execute_async(ctx)
    transcript = result.step_results[0].result_data["transcript"]
    assert [t["role"] for t in transcript] == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]


# --------------------------------------------------------------------------- #
# Transcript carried into subsequent prompts
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_each_role_sees_prior_arguments_in_transcript() -> None:
    """Round 2's proponent prompt must include round 1's transcript."""
    replies = ["R1-pro arg", "R1-crit arg", "R2-pro arg", "R2-crit arg", "FINAL"]
    chain, ctx, llm = _basic_debate(replies=replies, rounds=2)
    await chain.execute_async(ctx)
    # Prompt index 2 is round-2 proponent → must contain round-1 arguments
    third_prompt = llm.prompts[2]
    assert "R1-pro arg" in third_prompt
    assert "R1-crit arg" in third_prompt
    # First prompt is round-1 proponent → transcript should be the placeholder
    assert "no prior arguments yet" in llm.prompts[0]


@pytest.mark.asyncio
async def test_judge_prompt_contains_full_transcript_and_task() -> None:
    replies = ["A", "B", "C", "D", "FINAL"]
    chain, ctx, llm = _basic_debate(
        replies=replies, rounds=2, task="Use caching?"
    )
    await chain.execute_async(ctx)
    judge_prompt = llm.prompts[-1]
    assert judge_prompt.startswith("Topic: Use caching?")
    for arg in ("A", "B", "C", "D"):
        assert arg in judge_prompt


@pytest.mark.asyncio
async def test_transcript_format_includes_round_and_role_labels() -> None:
    replies = ["arg1", "arg2", "FINAL"]
    chain, ctx, llm = _basic_debate(replies=replies, rounds=1)
    await chain.execute_async(ctx)
    judge_prompt = llm.prompts[-1]
    assert "[Round 1 · proponent] arg1" in judge_prompt
    assert "[Round 1 · critic] arg2" in judge_prompt


# --------------------------------------------------------------------------- #
# Custom role prompts
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_custom_role_prompts_used_when_provided() -> None:
    chain, ctx, llm = _basic_debate(
        replies=["A", "B", "FINAL"],
        rounds=1,
        role_prompts={
            "proponent": "PRO TEMPLATE for {role} round {round}: {task} | history:{transcript}",
            "critic":    "CRIT TEMPLATE for {role} round {round}: {task} | history:{transcript}",
        },
    )
    await chain.execute_async(ctx)
    assert llm.prompts[0].startswith("PRO TEMPLATE for proponent round 1: Should we add caching?")
    assert llm.prompts[1].startswith("CRIT TEMPLATE for critic round 1: Should we add caching?")


@pytest.mark.asyncio
async def test_missing_role_prompt_falls_back_to_default() -> None:
    chain, ctx, llm = _basic_debate(
        replies=["A", "B", "FINAL"],
        rounds=1,
        role_prompts={"proponent": "CUSTOM: {role}"},  # critic absent
    )
    await chain.execute_async(ctx)
    assert llm.prompts[0] == "CUSTOM: proponent"
    # Default has 'structured debate' language
    assert "structured debate" in llm.prompts[1]


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_verdict_written_to_memory() -> None:
    chain, ctx, _ = _basic_debate(replies=["A", "B", "VERDICT TEXT"], rounds=1)
    result = await chain.execute_async(ctx)
    assert result.step_results[0].result == "VERDICT TEXT"
    assert ctx.memory.get("debate", {}).get("verdict") == "VERDICT TEXT"


@pytest.mark.asyncio
async def test_result_data_includes_transcript_and_metadata() -> None:
    chain, ctx, _ = _basic_debate(replies=["A", "B", "C", "D", "V"], rounds=2)
    result = await chain.execute_async(ctx)
    rd = result.step_results[0].result_data
    assert rd["verdict"] == "V"
    assert rd["rounds_executed"] == 2
    assert rd["role_call_count"] == 4
    assert len(rd["transcript"]) == 4
    assert rd["topic"] == "Should we add caching?"


@pytest.mark.asyncio
async def test_history_entry_names_roles_and_rounds() -> None:
    chain, ctx, _ = _basic_debate(replies=["A", "B", "V"], rounds=1)
    result = await chain.execute_async(ctx)
    entry = result.step_results[0].updated_history[-1]
    assert "DEBATE" in entry
    assert "proponent" in entry
    assert "critic" in entry
    assert "1 rounds" in entry
    assert "V" in entry


@pytest.mark.asyncio
async def test_empty_output_memory_key_does_not_write() -> None:
    chain, ctx, _ = _basic_debate(
        replies=["A", "B", "V"], rounds=1, output_memory_key=""
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert "debate" not in ctx.memory or "verdict" not in ctx.memory["debate"]


# --------------------------------------------------------------------------- #
# Task source
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_task_source_can_pull_from_memory() -> None:
    llm = _ScriptedLLM(replies=["A", "B", "V"])
    chain = ReasoningChain(
        steps=[
            DebateStepDescription(
                number=1, title="d",
                config=DebateStepConfig(
                    roles=["a", "b"], rounds=1,
                    judge_prompt="T={task}|TR={transcript}",
                    task_source="$memory.input.topic",
                ),
            )
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="ignored", api=llm)
    ctx.memory_write("topic", "Adopt edge caching?", namespace="input")
    await chain.execute_async(ctx)
    # Every role prompt + judge prompt should see the resolved topic
    for p in llm.prompts[:-1]:
        assert "Adopt edge caching?" in p
    # Judge prompt includes both topic and transcript
    assert "T=Adopt edge caching?" in llm.prompts[-1]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_at_least_two_roles_required() -> None:
    with pytest.raises(Exception):
        DebateStepConfig(roles=["solo"], rounds=1, judge_prompt="x")


def test_rounds_must_be_positive() -> None:
    with pytest.raises(Exception):
        DebateStepConfig(roles=["a", "b"], rounds=0, judge_prompt="x")


def test_judge_prompt_required() -> None:
    with pytest.raises(Exception):
        DebateStepConfig(roles=["a", "b"], rounds=1, judge_prompt="")


# --------------------------------------------------------------------------- #
# Per-role LLM config override
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_role_llm_configs_dispatch_to_role_specific_client() -> None:
    """``role_llm_configs[role]`` is honoured for that role's turns.

    We don't have a way to introspect *which* client was used per call without
    an OpenAI-compatible mock; we just confirm the path doesn't blow up and
    every call still produces a transcript entry.
    """
    chain, ctx, _ = _basic_debate(
        replies=["A", "B", "C", "D", "V"],
        rounds=2,
        role_llm_configs={
            "proponent": LLMStepConfig(temperature=0.0),
            "critic": LLMStepConfig(temperature=1.0),
        },
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert len(result.step_results[0].result_data["transcript"]) == 4


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_failure_during_debate_marks_step_failed() -> None:
    class _Boom(LLMClientBase):
        async def get_response(self, prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return await self.get_response(prompt)

    chain = ReasoningChain(
        steps=[
            DebateStepDescription(
                number=1, title="d",
                config=DebateStepConfig(
                    roles=["a", "b"], rounds=1, judge_prompt="x: {transcript}",
                ),
            )
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="topic", api=_Boom(), retry_max=1)
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "LLM unavailable" in sr.error_message


# --------------------------------------------------------------------------- #
# Cost estimation interop
# --------------------------------------------------------------------------- #


def test_debate_step_appears_in_cost_estimate_as_non_llm_row() -> None:
    """``chain.estimate_cost`` doesn't model debate calls; the step appears
    with ``calls_llm=False`` because debate isn't in the LLM-call classifier.

    This documents the *current* behaviour and guards against an accidental
    cost-estimator change that would inflate the chain total by misclassifying
    debate steps (whose call count depends on roles × rounds + 1)."""
    chain = ReasoningChain(
        steps=[
            DebateStepDescription(
                number=1, title="d",
                config=DebateStepConfig(
                    roles=["a", "b"], rounds=2, judge_prompt="x",
                ),
            )
        ],
        max_workers=1,
    )

    class _Stub(LLMClientBase):
        async def get_response(self, p):
            return ""
        async def get_response_with_retries(self, p, retries=3):
            return ""

    ctx = ReasoningContext(outer_context="x", api=_Stub())
    est = chain.estimate_cost(ctx)
    assert est.steps[0].calls_llm is False


# --------------------------------------------------------------------------- #
# Judge LLM config override
# --------------------------------------------------------------------------- #


def _spy_on_client_selection(monkeypatch) -> list:
    """Record the ``LLMStepConfig`` passed to every ``get_llm_client_for_step`` call."""
    seen: list = []
    original = ReasoningContext.get_llm_client_for_step

    def _spy(self, llm_config=None):
        seen.append(llm_config)
        return original(self, llm_config)

    monkeypatch.setattr(ReasoningContext, "get_llm_client_for_step", _spy)
    return seen


@pytest.mark.asyncio
async def test_judge_uses_shared_llm_config_by_default(monkeypatch) -> None:
    """Without ``judge_llm_config`` the judge keeps using ``llm_config`` (None here)."""
    seen = _spy_on_client_selection(monkeypatch)
    chain, ctx, _ = _basic_debate(replies=["A", "B", "V"], rounds=1)
    await chain.execute_async(ctx)
    # 2 role calls + 1 judge call, all with the shared (unset) config.
    assert len(seen) == 3
    assert seen[-1] is None


@pytest.mark.asyncio
async def test_judge_llm_config_overrides_only_the_judge_call(monkeypatch) -> None:
    seen = _spy_on_client_selection(monkeypatch)
    judge_cfg = LLMStepConfig(temperature=0.0)
    role_cfg = LLMStepConfig(temperature=0.9)
    chain, ctx, _ = _basic_debate(
        replies=["A", "B", "V"],
        rounds=1,
        role_llm_configs={"proponent": role_cfg, "critic": role_cfg},
        judge_llm_config=judge_cfg,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert len(seen) == 3
    assert seen[0] is role_cfg
    assert seen[1] is role_cfg
    assert seen[2] is judge_cfg


# --------------------------------------------------------------------------- #
# Convergence / early stopping (opt-in)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_early_stop_by_default_runs_full_round_budget() -> None:
    """Identical arguments every round still burn the whole budget by default."""
    chain, ctx, llm = _basic_debate(replies=["SAME"] * 6 + ["V"], rounds=3)
    result = await chain.execute_async(ctx)
    rd = result.step_results[0].result_data
    assert rd["rounds_executed"] == 3
    assert rd["rounds_configured"] == 3
    assert rd["role_call_count"] == 6
    assert rd["converged"] is False
    assert rd["convergence_reason"] is None
    assert llm.i == 7


@pytest.mark.asyncio
async def test_stop_when_unchanged_ends_the_debate_early() -> None:
    """Round 2 repeats round 1 verbatim → stop after round 2 of a 3-round budget."""
    # 2 roles × 2 rounds = 4 role calls, then the judge. A 5th role call would
    # exhaust the canned replies and raise, proving the early stop happened.
    chain, ctx, llm = _basic_debate(
        replies=["SAME", "SAME", "SAME", "SAME", "V"],
        rounds=3,
        stop_when_unchanged=1,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result == "V"
    rd = sr.result_data
    assert rd["rounds_executed"] == 2
    assert rd["rounds_configured"] == 3
    assert rd["role_call_count"] == 4
    assert rd["converged"] is True
    assert "stop_when_unchanged" in rd["convergence_reason"]
    assert llm.i == 5


@pytest.mark.asyncio
async def test_stop_when_unchanged_does_not_fire_when_arguments_differ() -> None:
    chain, ctx, _ = _basic_debate(
        replies=["A", "B", "C", "D", "V"],
        rounds=2,
        stop_when_unchanged=1,
    )
    result = await chain.execute_async(ctx)
    rd = result.step_results[0].result_data
    assert rd["rounds_executed"] == 2
    assert rd["converged"] is False


@pytest.mark.asyncio
async def test_stop_when_unchanged_requires_the_full_streak() -> None:
    """With ``stop_when_unchanged=2`` a single repeated round is not enough."""
    # r1: A/B, r2: A/B (unchanged, streak=1), r3: A/B (unchanged, streak=2 → stop)
    chain, ctx, llm = _basic_debate(
        replies=["A", "B"] * 3 + ["V"],
        rounds=5,
        stop_when_unchanged=2,
    )
    result = await chain.execute_async(ctx)
    rd = result.step_results[0].result_data
    assert rd["rounds_executed"] == 3
    assert rd["converged"] is True
    assert llm.i == 7


@pytest.mark.asyncio
async def test_convergence_predicate_stops_the_debate() -> None:
    chain, ctx, llm = _basic_debate(
        replies=["A", "B", "V"],
        rounds=4,
        convergence_predicate=lambda transcript: len(transcript) >= 2,
    )
    result = await chain.execute_async(ctx)
    rd = result.step_results[0].result_data
    assert rd["rounds_executed"] == 1
    assert rd["converged"] is True
    assert rd["convergence_reason"] == "convergence_predicate returned True"
    assert llm.i == 3


@pytest.mark.asyncio
async def test_broken_convergence_predicate_is_swallowed() -> None:
    """A raising predicate must not fail the step — the debate just runs on."""

    def _boom(_transcript):
        raise ValueError("bad predicate")

    chain, ctx, _ = _basic_debate(
        replies=["A", "B", "C", "D", "V"],
        rounds=2,
        convergence_predicate=_boom,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["rounds_executed"] == 2
    assert sr.result_data["converged"] is False


@pytest.mark.asyncio
async def test_history_entry_reports_actual_rounds_after_early_stop() -> None:
    chain, ctx, _ = _basic_debate(
        replies=["SAME", "SAME", "SAME", "SAME", "V"],
        rounds=5,
        stop_when_unchanged=1,
    )
    result = await chain.execute_async(ctx)
    entry = result.step_results[0].updated_history[-1]
    assert "2 rounds" in entry


@pytest.mark.asyncio
async def test_typed_transcript_view_surfaces_convergence() -> None:
    chain, ctx, _ = _basic_debate(
        replies=["SAME", "SAME", "SAME", "SAME", "V"],
        rounds=3,
        stop_when_unchanged=1,
    )
    result = await chain.execute_async(ctx)
    view = result.step_results[0].as_debate_transcript()
    assert view is not None
    assert view.rounds_executed == 2
    assert view.rounds_configured == 3
    assert view.converged is True
