"""Rewinding a chain must undo its state, not its bill.

A RE-PLAN rollback and a loop-back restart both rewind ``all_results``, the
list ``ReasoningResult.token_usage`` used to be summed from. That made a
discarded attempt free: a chain measured at two model calls billed one call's
worth of tokens. Downstream consumers spend that number — a cost-aware search
(evolutionary chain search bins and Pareto-ranks on ``total_tokens``) would
file a chain that retried in a *cheaper* bucket than a chain that got it right
first time, and a token budget would never trip. These tests pin the split:
token totals and execution counts come from the append-only ledger
(``result.trace.events``), while ``step_results`` and the success verdict stay
rewindable state.

Lives at the top level of ``tests/`` rather than in ``tests/replan/`` because
it also covers the loop-back rewind in ``tests/orchestration/``'s territory.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ExecutionMode,
    Language,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    LoopConfig,
    ReasoningChain,
    ReasoningContext,
    ReplanAction,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    RuleBasedReplanCheckerConfig,
)
from mmar_carl.chain import ChainBuilder

# Every mocked model call costs exactly this much, so token totals are a
# direct, exact multiple of the number of calls the chain really made.
PER_CALL_USAGE = {"prompt": 10, "completion": 5, "total": 15}


class CountingUsageMockLLMClient(LLMClientBase):
    """Deterministic client that reports a fixed usage for every call."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        result, _ = await self.get_response_with_usage(prompt)
        return result

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        result, _ = await self.get_response_with_usage(prompt, retries)
        return result

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        _ = prompt, retries
        self.calls += 1
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index], dict(PER_CALL_USAGE)


def _llm_step(number: int, dependencies: list[int] | None = None, **kwargs):
    return LLMStepDescription(
        number=number,
        title=f"Step {number}",
        aim="Produce answer",
        reasoning_questions="What is the answer?",
        stage_action="Generate",
        example_reasoning="Example",
        dependencies=dependencies or [],
        llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        **kwargs,
    )


def _retry_on_draft_v1_policy() -> ReplanPolicy:
    return ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="draft_guard",
                result_substrings=["draft-v1"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            )
        ],
    )


def _run_single_step(*, replanning: bool):
    """Run the same one-step chain with the retry guard armed or not.

    ``replanning=True`` makes the step emit ``draft-v1`` first, which the
    guard rejects, so the step runs twice. ``replanning=False`` emits the
    accepted answer immediately, so it runs once. Everything else — chain,
    context, per-call usage — is identical.
    """
    responses = ["draft-v1", "draft-v2"] if replanning else ["draft-v2"]
    client = CountingUsageMockLLMClient(responses)
    context = ReasoningContext(
        outer_context="input", api=client, model="unused", language=Language.ENGLISH
    )
    chain = ReasoningChain(
        steps=[_llm_step(1)],
        max_workers=1,
        replan_policy=_retry_on_draft_v1_policy(),
    )
    return client, chain.execute(context)


class TestRolledBackAttemptsAreStillBilled:
    def test_replanning_chain_reports_more_tokens_than_one_that_does_not(self) -> None:
        """The two-call run must cost more than the one-call run.

        Both runs end on the same answer (``draft-v2``) via the same chain and
        the same policy; only the number of model calls differs. Before the
        ledger split both reported 15 tokens, so a search minimising
        ``total_tokens`` saw a retrying chain as exactly as cheap as one that
        never retried and would rank it as the better of the two.
        """
        plain_client, plain = _run_single_step(replanning=False)
        replan_client, replanned = _run_single_step(replanning=True)

        assert plain_client.calls == 1
        assert replan_client.calls == 2
        assert plain.get_final_output() == replanned.get_final_output() == "draft-v2"

        assert replanned.token_usage["total"] > plain.token_usage["total"]
        # Exact, not merely greater: one discarded attempt, billed in full.
        assert replanned.token_usage == {
            "prompt": PER_CALL_USAGE["prompt"] * 2,
            "completion": PER_CALL_USAGE["completion"] * 2,
            "total": PER_CALL_USAGE["total"] * 2,
        }

    def test_token_total_tracks_model_calls_across_a_checkpoint_rollback(self) -> None:
        """A multi-step rollback bills every re-executed step, not just the retried one.

        Step 3 trips the guard, so ``REPLAN_FROM_CHECKPOINT`` rewinds to the
        checkpoint after step 1 and runs steps 2 and 3 again: five model calls
        in total. Summing the rewound state saw only the three steps that
        survived, so the chain looked 40% cheaper than the API it actually
        charged against.
        """
        client = CountingUsageMockLLMClient(["cp-ok", "mid", "draft-v1", "mid-again", "final"])
        policy = ReplanPolicy(
            enabled=True,
            checkers=[
                RuleBasedReplanCheckerConfig(
                    name="draft_guard",
                    result_substrings=["draft-v1"],
                    action_on_match=ReplanAction.REPLAN_FROM_CHECKPOINT,
                    rollback_target_on_match=ReplanRollbackTarget(
                        target_type=ReplanTargetType.NAMED_CHECKPOINT,
                        checkpoint_name="cp1",
                    ),
                )
            ],
        )
        steps = [
            _llm_step(1, checkpoint=True, checkpoint_name="cp1"),
            _llm_step(2, dependencies=[1]),
            _llm_step(3, dependencies=[2]),
        ]
        context = ReasoningContext(
            outer_context="input", api=client, model="unused", language=Language.ENGLISH
        )

        result = ReasoningChain(steps=steps, max_workers=1, replan_policy=policy).execute(context)

        assert result.success
        assert client.calls == 5, "expected 3 steps + a 2-step re-run after the rollback"
        assert len(result.step_results) == 3, "state keeps one record per step"
        assert result.token_usage["total"] == PER_CALL_USAGE["total"] * client.calls

    def test_loop_back_bills_every_iteration(self) -> None:
        """A loop that runs its body three times must bill three times.

        ``loop_back_to`` drops the previous iteration's results from
        ``all_results`` the same way a rollback does, so a 2-step body over 3
        iterations — six real model calls — billed as if the body had run once.
        Loops are the cheapest way to hide spend from a cost-aware caller.
        """
        client = CountingUsageMockLLMClient(["a"])
        steps = [
            _llm_step(1),
            _llm_step(
                2,
                dependencies=[1],
                loop_back_to=1,
                loop_config=LoopConfig(max_iterations=2, condition_key="$memory.ctl.again"),
            ),
        ]
        context = ReasoningContext(
            outer_context="input", api=client, model="unused", language=Language.ENGLISH
        )
        context.memory.setdefault("ctl", {})["again"] = True

        result = ReasoningChain(steps=steps, max_workers=1).execute(context)

        assert client.calls == 6, "2-step body, initial pass + 2 loop iterations"
        assert len(result.step_results) == 2, "state keeps only the final iteration"
        assert result.token_usage["total"] == PER_CALL_USAGE["total"] * 6


class TestNonRewindingChainsAreUnaffected:
    @pytest.mark.parametrize("replan_policy_enabled", [False, True])
    def test_successful_chain_that_never_rewinds_reports_the_step_sum(
        self, replan_policy_enabled: bool
    ) -> None:
        """With nothing discarded, the ledger total must equal the old state total.

        This is the compatibility floor for the fix: every existing caller that
        cross-checks ``result.token_usage`` against ``token_usage_by_step`` (cost
        estimators, per-step token pies, budget guards) keeps agreeing exactly.
        Armed but never-firing RE-PLAN is included because that is the common
        production shape — a policy configured for a bad day that never has one.
        """
        client = CountingUsageMockLLMClient(["clean"])
        context = ReasoningContext(
            outer_context="input", api=client, model="unused", language=Language.ENGLISH
        )
        chain = ReasoningChain(
            steps=[_llm_step(1), _llm_step(2, dependencies=[1])],
            max_workers=1,
            replan_policy=_retry_on_draft_v1_policy() if replan_policy_enabled else None,
        )

        result = chain.execute(context)

        assert result.success
        assert client.calls == 2
        assert result.replan_events == [] or all(
            event.final_action == ReplanAction.CONTINUE for event in result.replan_events
        )

        state_sum = {
            "prompt": sum(s.token_usage.get("prompt", 0) for s in result.step_results),
            "completion": sum(s.token_usage.get("completion", 0) for s in result.step_results),
        }
        state_sum["total"] = state_sum["prompt"] + state_sum["completion"]
        assert result.token_usage == state_sum
        assert result.token_usage["total"] == PER_CALL_USAGE["total"] * 2

    def test_tool_only_chain_still_reports_no_tokens(self) -> None:
        """Steps that never call a model must not acquire a bill.

        Skipped conditional branches and tool steps reach ``all_results`` by
        paths that never produce a ledger entry; if the ledger were treated as
        authoritative for anything but usage, a free chain could start
        reporting a cost.
        """
        chain = ChainBuilder().add_tool_step(number=1, title="Free", tool_name="free").build()
        context = ReasoningContext(
            outer_context="input", api=CountingUsageMockLLMClient(["unused"]), model="unused"
        )
        context.register_tool("free", lambda: "done")

        result = chain.execute(context)

        assert result.success
        assert result.token_usage == {"prompt": 0, "completion": 0, "total": 0}


class TestRewoundStateStaysRewound:
    def test_discarded_failed_attempt_does_not_fail_the_chain(self) -> None:
        """A retried transient failure must still be a successful chain.

        The tempting fix for the lost bill is to append discarded attempts back
        into ``step_results`` — but the chain verdict is
        ``len(failed_steps) == 0`` over that same list, so doing it would turn
        every successful RE-PLAN recovery into a chain-level failure and mark
        the retried step's error as ``result.error``. Tokens accumulate;
        results do not.
        """
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            return "ok"

        chain = (
            ChainBuilder()
            .add_tool_step(number=1, title="Flaky", tool_name="flaky")
            .with_replan_policy(
                ReplanPolicy(enabled=True, checkers=[RuleBasedReplanCheckerConfig(name="guard")])
            )
            .build()
        )
        context = ReasoningContext(
            outer_context="input", api=CountingUsageMockLLMClient(["unused"]), model="unused"
        )
        context.register_tool("flaky", flaky)

        result = chain.execute(context)

        assert calls["n"] == 2
        assert result.success is True
        assert result.error is None
        assert [(r.step_number, r.success) for r in result.step_results] == [(1, True)]
        # The failed attempt is not erased, only demoted to the ledger.
        assert [(e.step_number, e.success) for e in result.trace.events] == [(1, False), (1, True)]

    def test_execution_stats_separates_work_done_from_state_reached(self) -> None:
        """``step_executions`` answers "how many runs", ``executed_steps`` "how many steps".

        A caller reconstructing per-call cost from a persisted run has no other
        source: ``ReasoningResult.trace`` is an untyped optional field that
        serialisers routinely drop, and ``len(step_results)`` counts the final
        state, which is exactly what the rollback rewound.
        """
        _, replanned = _run_single_step(replanning=True)
        _, plain = _run_single_step(replanning=False)

        assert replanned.metadata["execution_stats"]["step_executions"] == 2
        assert replanned.metadata["execution_stats"]["executed_steps"] == 1
        assert plain.metadata["execution_stats"]["step_executions"] == 1
        assert plain.metadata["execution_stats"]["executed_steps"] == 1
