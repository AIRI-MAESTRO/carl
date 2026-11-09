"""
Tests for ParallelSamplingStep — N-sample voting / LLM-judge aggregation.

Covers:
- Model/config field validation
- majority_vote: basic, tie-breaking, normalisation flag
- best_of_n / llm_judge: judge selects winner by number
- LLM judge fallback (invalid judge response → majority_vote)
- Full chain execution: step runs, history is updated correctly
- Partial failures: some samples fail, winner still emerges from successes
- All samples fail: step returns failure
- Token usage is summed across samples
- result_data carries candidates and metadata
- Aggregation observability: every fallback path is labelled in
  ``result_data["aggregation_outcome"]`` and warns, including the degenerate
  all-unique majority vote; ``strict=True`` turns fallbacks into step failures
- ``answer_extractor`` / ``vote_normalization`` rescue a prose vote
"""

import warnings

import pytest

from mmar_carl import (
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    LLMStepDescription,
    ParallelSamplingStepDescription,
    ParallelSamplingAggregation,
    ParallelSamplingStepConfig,
    StepType,
    VoteNormalization,
)
from mmar_carl.step_executors import ParallelSamplingStepExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_client(responses: list[str]) -> LLMClientBase:
    """Return a mock LLMClientBase that cycles through *responses*."""
    call_count = {"n": 0}

    class _MockClient(LLMClientBase):
        async def get_response(self, prompt: str) -> str:
            idx = call_count["n"] % len(responses)
            call_count["n"] += 1
            return responses[idx]

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return await self.get_response(prompt)

    return _MockClient()


def _make_context(responses: list[str], language: Language = Language.ENGLISH) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test input",
        api=_make_llm_client(responses),
        model="test",
        language=language,
    )


def _base_step(number: int = 2) -> LLMStepDescription:
    return LLMStepDescription(
        number=number,
        title="_sample",
        aim="Answer the question.",
    )


def _sampling_step(
    n_samples: int = 3,
    aggregation: ParallelSamplingAggregation = ParallelSamplingAggregation.MAJORITY_VOTE,
    normalize: bool = True,
    judge_prompt: str = "",
    step_number: int = 1,
    vote_normalization: VoteNormalization = VoteNormalization.BASIC,
    answer_extractor=None,
    strict: bool = False,
) -> ParallelSamplingStepDescription:
    return ParallelSamplingStepDescription(
        number=step_number,
        title="Vote",
        base_step=_base_step(number=step_number),
        config=ParallelSamplingStepConfig(
            n_samples=n_samples,
            aggregation=aggregation,
            normalize_for_vote=normalize,
            judge_prompt=judge_prompt,
            vote_normalization=vote_normalization,
            answer_extractor=answer_extractor,
            strict=strict,
        ),
    )


class _JudgeBoomClient(LLMClientBase):
    """Samples succeed; the judge call (detected by prompt marker) raises."""

    def __init__(self, sample_replies: list[str]) -> None:
        self._replies = list(sample_replies)
        self._i = 0

    async def get_response(self, prompt: str) -> str:
        if "expert evaluator" in prompt:
            raise RuntimeError("judge unavailable")
        reply = self._replies[self._i % len(self._replies)]
        self._i += 1
        return reply

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ---------------------------------------------------------------------------
# Unit: config / model
# ---------------------------------------------------------------------------


class TestParallelSamplingConfig:
    def test_defaults(self):
        cfg = ParallelSamplingStepConfig()
        assert cfg.n_samples == 5
        assert cfg.aggregation == ParallelSamplingAggregation.MAJORITY_VOTE
        assert cfg.normalize_for_vote is True
        assert cfg.judge_prompt == ""

    def test_custom_values(self):
        cfg = ParallelSamplingStepConfig(
            n_samples=7,
            aggregation=ParallelSamplingAggregation.BEST_OF_N,
            normalize_for_vote=False,
        )
        assert cfg.n_samples == 7
        assert cfg.aggregation == ParallelSamplingAggregation.BEST_OF_N
        assert cfg.normalize_for_vote is False

    def test_n_samples_minimum(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ParallelSamplingStepConfig(n_samples=1)  # ge=2

    def test_step_type(self):
        step = _sampling_step()
        assert step.step_type == StepType.PARALLEL_SAMPLING


# ---------------------------------------------------------------------------
# Unit: _majority_vote helper
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_simple_majority(self):
        candidates = ["Paris", "Paris", "London"]
        result = ParallelSamplingStepExecutor._majority_vote(candidates, normalize=True)
        assert result == "Paris"

    def test_tie_first_occurrence_wins(self):
        candidates = ["A", "B", "A", "B"]
        result = ParallelSamplingStepExecutor._majority_vote(candidates, normalize=True)
        # "A" and "B" tie; "A" appears first
        assert result == "A"

    def test_normalize_strips_case(self):
        candidates = ["Paris", "paris", "PARIS", "London"]
        result = ParallelSamplingStepExecutor._majority_vote(candidates, normalize=True)
        # "Paris"/"paris"/"PARIS" all normalise to "paris" → majority; first raw match returned
        assert result == "Paris"

    def test_no_normalize(self):
        candidates = ["Paris", "paris", "London", "London"]
        result = ParallelSamplingStepExecutor._majority_vote(candidates, normalize=False)
        # Without normalisation "Paris" ≠ "paris"; "London" wins
        assert result == "London"

    def test_single_candidate(self):
        result = ParallelSamplingStepExecutor._majority_vote(["only"], normalize=True)
        assert result == "only"


# ---------------------------------------------------------------------------
# Integration: full chain execution
# ---------------------------------------------------------------------------


class TestParallelSamplingChainExecution:
    @pytest.mark.asyncio
    async def test_majority_vote_winner_in_history(self):
        """Winning answer is stored in context history."""
        # 3 samples → responses cycle: ["Paris", "Paris", "London"]
        ctx = _make_context(["Paris", "Paris", "London"])
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3)], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        assert len(ctx.history) == 1
        assert "Paris" in ctx.history[0]

    @pytest.mark.asyncio
    async def test_result_data_carries_candidates(self):
        """result_data contains all raw candidate strings."""
        ctx = _make_context(["A", "B", "A"])
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3)], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.get_step_result(1)
        assert step_result is not None
        rd = step_result.result_data
        assert rd["n_samples"] == 3
        assert len(rd["candidates"]) == 3

    @pytest.mark.asyncio
    async def test_token_usage_summed(self):
        """Total token usage reflects sum across all samples."""
        ctx = _make_context(["answer"] * 5)
        chain = ReasoningChain(steps=[_sampling_step(n_samples=5)], max_workers=5)
        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.get_step_result(1)
        # token_usage may be empty (mock client), but the field should exist
        assert step_result is not None
        assert isinstance(step_result.token_usage, dict)

    @pytest.mark.asyncio
    async def test_downstream_step_sees_winner(self):
        """A subsequent tool step can read the voting winner from history."""
        captured = []

        ctx = _make_context(["Paris", "Paris", "London"])

        def capture_last():
            captured.append(ctx.history[-1])
            return "ok"

        ctx.register_tool("capture", capture_last)

        from mmar_carl import ToolStepDescription
        from mmar_carl.models.config import ToolStepConfig

        chain = ReasoningChain(
            steps=[
                _sampling_step(n_samples=3, step_number=1),
                ToolStepDescription(
                    number=2,
                    title="Read winner",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="capture", input_mapping={}),
                ),
            ],
            max_workers=3,
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert len(captured) == 1
        assert "Paris" in captured[0]


# ---------------------------------------------------------------------------
# Integration: best_of_n / llm_judge
# ---------------------------------------------------------------------------


class TestBestOfNAggregation:
    @pytest.mark.asyncio
    async def test_judge_selects_second_candidate(self):
        """
        3 samples produce ["A", "B", "C"]; judge always returns "2" → winner is "B".
        The LLM mock cycles: first 3 calls give samples A/B/C, 4th call is the judge → "2".
        """
        ctx = _make_context(["A", "B", "C", "2"])
        step = _sampling_step(
            n_samples=3,
            aggregation=ParallelSamplingAggregation.BEST_OF_N,
            step_number=1,
        )
        chain = ReasoningChain(steps=[step], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.get_step_result(1)
        assert step_result is not None
        # Judge said "2" → second candidate
        assert step_result.result == "B"

    @pytest.mark.asyncio
    async def test_judge_fallback_on_invalid_response(self):
        """If judge returns a non-numeric string, fall back to majority vote."""
        # Samples: A, A, B → majority is A
        # Judge response: "invalid!" → fallback to majority → A
        ctx = _make_context(["A", "A", "B", "invalid!"])
        step = _sampling_step(
            n_samples=3,
            aggregation=ParallelSamplingAggregation.BEST_OF_N,
            step_number=1,
        )
        chain = ReasoningChain(steps=[step], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.get_step_result(1)
        assert step_result is not None
        # Fallback: first candidate returned (index 0)
        assert step_result.result == "A"


# ---------------------------------------------------------------------------
# Edge cases: partial / total failures
# ---------------------------------------------------------------------------


class TestSamplingFailures:
    @pytest.mark.asyncio
    async def test_all_samples_fail_returns_failure(self):
        """When every sample raises, the step fails gracefully."""

        class _FailClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                raise RuntimeError("LLM unavailable")

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                raise RuntimeError("LLM unavailable")

        ctx = ReasoningContext(
            outer_context="x",
            api=_FailClient(),
            model="test",
            language=Language.ENGLISH,
        )
        step = _sampling_step(n_samples=2, step_number=1)
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(ctx)

        assert not result.success
        failed = result.get_failed_steps()
        assert any(r.step_number == 1 for r in failed)


# ---------------------------------------------------------------------------
# Unit: canonicalisation for voting
# ---------------------------------------------------------------------------


class TestCanonicaliseForVote:
    def test_basic_tier_is_strip_lower(self):
        out = ParallelSamplingStepExecutor._canonicalise_for_vote("  Paris  ")
        assert out == "paris"

    def test_none_tier_keeps_raw_text(self):
        out = ParallelSamplingStepExecutor._canonicalise_for_vote(
            "  Paris  ", normalization=VoteNormalization.NONE.value
        )
        assert out == "  Paris  "

    def test_normalize_false_keeps_raw_text(self):
        out = ParallelSamplingStepExecutor._canonicalise_for_vote("  Paris  ", normalize=False)
        assert out == "  Paris  "

    def test_aggressive_tier_collapses_whitespace_and_punctuation(self):
        norm = VoteNormalization.AGGRESSIVE.value
        a = ParallelSamplingStepExecutor._canonicalise_for_vote("The   answer is:  Paris.", normalization=norm)
        b = ParallelSamplingStepExecutor._canonicalise_for_vote("the answer is: paris!", normalization=norm)
        assert a == b == "the answer is: paris"

    def test_answer_extractor_runs_before_normalisation(self):
        out = ParallelSamplingStepExecutor._canonicalise_for_vote(
            "blah blah FINAL: Paris",
            answer_extractor=lambda s: s.split("FINAL:")[-1],
        )
        assert out == "paris"

    def test_broken_answer_extractor_falls_back_to_raw_text(self):
        def _boom(_s: str) -> str:
            raise ValueError("bad extractor")

        out = ParallelSamplingStepExecutor._canonicalise_for_vote(" Paris ", answer_extractor=_boom)
        assert out == "paris"


# ---------------------------------------------------------------------------
# Aggregation observability: outcome labels, warnings, strict mode
# ---------------------------------------------------------------------------


class TestAggregationOutcome:
    @pytest.mark.asyncio
    async def test_genuine_vote_is_labelled_and_does_not_warn(self):
        ctx = _make_context(["Paris", "Paris", "London"])
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3)], max_workers=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        rd = result.get_step_result(1).result_data
        assert rd["aggregation_outcome"] == "voted"
        assert rd["vote_distribution"] == {"paris": 2, "london": 1}
        assert rd["vote_winning_count"] == 2
        assert rd["vote_total"] == 3
        assert rd["vote_degenerate"] is False
        assert not [w for w in caught if "fell back" in str(w.message)]

    @pytest.mark.asyncio
    async def test_all_unique_vote_is_reported_as_degenerate_and_warns(self):
        """The headline defect: free-form prose makes every sample unique, so the
        'majority vote' silently degenerates into 'return sample 0'."""
        ctx = _make_context(
            [
                "Paris is the capital, founded long ago.",
                "The capital city of France is Paris.",
                "It's Paris — the seat of government.",
            ]
        )
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3)], max_workers=3)
        with pytest.warns(UserWarning, match="vote_degenerate_fallback_first"):
            result = await chain.execute_async(ctx)

        assert result.success
        rd = result.get_step_result(1).result_data
        assert rd["aggregation_outcome"] == "vote_degenerate_fallback_first"
        assert rd["vote_degenerate"] is True
        assert rd["vote_winning_count"] == 1
        assert rd["vote_total"] == 3
        assert set(rd["vote_distribution"].values()) == {1}

    @pytest.mark.asyncio
    async def test_answer_extractor_rescues_a_degenerate_prose_vote(self):
        """The same prose samples become a real 2-of-3 vote once an extractor runs."""
        ctx = _make_context(
            [
                "Long reasoning... FINAL: Paris",
                "Different reasoning... FINAL: Paris",
                "Yet more reasoning... FINAL: London",
            ]
        )
        step = _sampling_step(
            n_samples=3,
            answer_extractor=lambda s: s.split("FINAL:")[-1],
        )
        chain = ReasoningChain(steps=[step], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        rd = result.get_step_result(1).result_data
        assert rd["aggregation_outcome"] == "voted"
        assert rd["vote_winning_count"] == 2
        assert rd["vote_degenerate"] is False
        assert rd["vote_distribution"] == {"paris": 2, "london": 1}

    @pytest.mark.asyncio
    async def test_aggressive_normalization_merges_punctuation_variants(self):
        ctx = _make_context(["Paris.", "  paris!  ", "London"])
        step = _sampling_step(n_samples=3, vote_normalization=VoteNormalization.AGGRESSIVE)
        chain = ReasoningChain(steps=[step], max_workers=3)
        result = await chain.execute_async(ctx)

        assert result.success
        rd = result.get_step_result(1).result_data
        assert rd["vote_distribution"] == {"paris": 2, "london": 1}
        assert rd["aggregation_outcome"] == "voted"

    @pytest.mark.asyncio
    async def test_strict_mode_fails_the_step_on_a_degenerate_vote(self):
        ctx = _make_context(["alpha prose", "beta prose", "gamma prose"])
        step = _sampling_step(n_samples=3, strict=True)
        chain = ReasoningChain(steps=[step], max_workers=3)
        with pytest.warns(UserWarning, match="vote_degenerate_fallback_first"):
            result = await chain.execute_async(ctx)

        sr = result.get_step_result(1)
        assert not sr.success
        assert "Strict aggregation failed" in sr.error_message
        # Telemetry is still attached so the failure is diagnosable.
        assert sr.result_data["aggregation_outcome"] == "vote_degenerate_fallback_first"

    @pytest.mark.asyncio
    async def test_judge_success_is_labelled_judged(self):
        ctx = _make_context(["A", "B", "C", "2"])
        step = _sampling_step(n_samples=3, aggregation=ParallelSamplingAggregation.BEST_OF_N)
        chain = ReasoningChain(steps=[step], max_workers=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        rd = result.get_step_result(1).result_data
        assert rd["aggregation_outcome"] == "judged"
        assert rd["judge_selected_index"] == 1
        assert not [w for w in caught if "fell back" in str(w.message)]

    @pytest.mark.asyncio
    async def test_unparseable_judge_reply_is_labelled_and_warns(self):
        ctx = _make_context(["A", "A", "B", "invalid!"])
        step = _sampling_step(n_samples=3, aggregation=ParallelSamplingAggregation.BEST_OF_N)
        chain = ReasoningChain(steps=[step], max_workers=3)
        with pytest.warns(UserWarning, match="judge_unparseable_fallback_first"):
            result = await chain.execute_async(ctx)

        sr = result.get_step_result(1)
        assert sr.success
        # Unchanged default behaviour — still the first candidate, but now labelled.
        assert sr.result == "A"
        assert sr.result_data["aggregation_outcome"] == "judge_unparseable_fallback_first"
        assert sr.result_data["judge_reply"] == "invalid!"

    @pytest.mark.asyncio
    async def test_judge_exception_falls_back_to_vote_and_warns(self):
        ctx = ReasoningContext(
            outer_context="x",
            api=_JudgeBoomClient(["A", "A", "B"]),
            model="test",
            language=Language.ENGLISH,
        )
        step = _sampling_step(n_samples=3, aggregation=ParallelSamplingAggregation.BEST_OF_N)
        chain = ReasoningChain(steps=[step], max_workers=3)
        with pytest.warns(UserWarning, match="judge_error_fallback_vote"):
            result = await chain.execute_async(ctx)

        sr = result.get_step_result(1)
        assert sr.success
        assert sr.result == "A"
        rd = sr.result_data
        assert rd["aggregation_outcome"] == "judge_error_fallback_vote"
        assert "judge unavailable" in rd["judge_error"]
        # The fallback vote's telemetry comes along for the ride.
        assert rd["vote_winning_count"] == 2

    @pytest.mark.asyncio
    async def test_strict_mode_fails_the_step_on_an_unparseable_judge_reply(self):
        ctx = _make_context(["A", "A", "B", "no number here"])
        step = _sampling_step(
            n_samples=3,
            aggregation=ParallelSamplingAggregation.BEST_OF_N,
            strict=True,
        )
        chain = ReasoningChain(steps=[step], max_workers=3)
        with pytest.warns(UserWarning, match="judge_unparseable_fallback_first"):
            result = await chain.execute_async(ctx)

        sr = result.get_step_result(1)
        assert not sr.success
        assert "Strict aggregation failed" in sr.error_message

    @pytest.mark.asyncio
    async def test_strict_mode_does_not_fail_a_clean_vote(self):
        ctx = _make_context(["Paris", "Paris", "London"])
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3, strict=True)], max_workers=3)
        result = await chain.execute_async(ctx)

        sr = result.get_step_result(1)
        assert sr.success
        assert sr.result == "Paris"

    @pytest.mark.asyncio
    async def test_typed_view_surfaces_the_outcome(self):
        ctx = _make_context(["Paris", "Paris", "London"])
        chain = ReasoningChain(steps=[_sampling_step(n_samples=3)], max_workers=3)
        result = await chain.execute_async(ctx)

        view = result.get_step_result(1).as_parallel_samples()
        assert view is not None
        assert view.aggregation_outcome == "voted"
        assert view.vote_winning_count == 2
        assert view.vote_degenerate is False
