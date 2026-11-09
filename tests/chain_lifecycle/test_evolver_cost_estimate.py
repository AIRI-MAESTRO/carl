"""Tests for ``ChainEvolver.estimate_cost`` — pre-flight USD/token projection.

From the live benchmark: a 3-pop × 2-gen × 5-case run made ~90 LLM
calls and took 8 minutes wall-clock against `qwen/qwen3-8b`. Users need
to know that *up front* before clicking go — especially on larger model
benchmarks where 90 calls translates to several dollars.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ChainEvolver,
    DataCase,
    EvolutionCostEstimate,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    ToolStepConfig,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstMetric(MetricBase):
    @property
    def name(self) -> str:
        return "c"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 1.0


def _ctx_factory(case: DataCase) -> ReasoningContext:
    return ReasoningContext(outer_context=case.input, api=None, model="gpt-4o-mini")


def _make_llm_chain(model: str | None = None) -> ReasoningChain:
    step = LLMStepDescription(number=1, title="solve", aim="Solve the problem")
    if model is not None:
        from mmar_carl import LLMStepConfig

        step = LLMStepDescription(
            number=1, title="solve", aim="Solve", llm_config=LLMStepConfig(model=model)
        )
    return ReasoningChain(steps=[step])


def _make_tool_only_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="noop", config=ToolStepConfig(tool_name="noop")
            )
        ],
    )


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


class TestBasicShape:
    def test_returns_evolution_cost_estimate(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=1,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory)
        assert isinstance(est, EvolutionCostEstimate)

    def test_total_runs_matches_pop_times_gens_times_cases(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset([DataCase(input=f"c{i}") for i in range(4)]),
            _ConstMetric(),
            population_size=5,
            generations=3,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory)
        assert est.total_chain_runs == 5 * 3 * 4
        assert est.population_size == 5
        assert est.generations == 3
        assert est.cases_per_evaluation == 4

    def test_smoke_check_adds_one_run(self) -> None:
        cases = [DataCase(input="x")]
        ev_on = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset(cases),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=True,
        )
        ev_off = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset(cases),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
        )
        est_on = ev_on.estimate_cost(_ctx_factory)
        est_off = ev_off.estimate_cost(_ctx_factory)
        assert est_on.total_chain_runs == est_off.total_chain_runs + 1
        assert est_on.smoke_check_enabled is True
        assert est_off.smoke_check_enabled is False


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


class TestMultiplication:
    def test_total_tokens_equals_per_chain_times_runs(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset([DataCase(input="x"), DataCase(input="y")]),
            _ConstMetric(),
            population_size=3,
            generations=2,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory)
        n_runs = 3 * 2 * 2
        assert est.total_chain_runs == n_runs
        assert est.total_tokens == est.per_chain_total_tokens * n_runs
        assert est.total_input_tokens > 0
        assert est.total_output_tokens > 0

    def test_total_cost_equals_per_chain_times_runs_with_pricing(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="gpt-4o-mini"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
        )
        est = ev.estimate_cost(
            _ctx_factory, pricing={"gpt-4o-mini": (0.00015, 0.0006)}
        )
        n_runs = 2 * 2 * 1
        assert est.total_cost_usd == pytest.approx(
            est.per_chain_cost_usd * n_runs
        )
        assert est.per_chain_cost_usd > 0  # actual cost computed
        assert est.total_cost_usd > 0


# ---------------------------------------------------------------------------
# Pricing missing detection
# ---------------------------------------------------------------------------


class TestPricingMissing:
    def test_unpriced_model_flagged(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="exotic-model-v3"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=1,
            generations=1,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory, pricing={"gpt-4o-mini": (0.0001, 0.0001)})
        assert "exotic-model-v3" in est.pricing_missing_models
        # No cost when the only model isn't priced
        assert est.total_cost_usd == 0.0

    def test_all_models_priced_no_missing_list(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="gpt-4o-mini"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=1,
            generations=1,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory, pricing={"gpt-4o-mini": (0.0001, 0.0001)})
        assert est.pricing_missing_models == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_dataset_raises(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(),
            SimpleDataset([]),
            _ConstMetric(),
            population_size=1,
            generations=1,
            smoke_check=False,
        )
        with pytest.raises(RuntimeError, match="dataset is empty"):
            ev.estimate_cost(_ctx_factory)


# ---------------------------------------------------------------------------
# Tool-only chain — zero LLM cost
# ---------------------------------------------------------------------------


def test_tool_only_chain_estimates_zero_llm_cost() -> None:
    ev = ChainEvolver(
        _make_tool_only_chain(),
        SimpleDataset([DataCase(input="x"), DataCase(input="y")]),
        _ConstMetric(),
        population_size=4,
        generations=2,
        smoke_check=False,
    )
    est = ev.estimate_cost(_ctx_factory)
    assert est.total_chain_runs == 4 * 2 * 2  # still counted as runs
    # ...but no LLM tokens since there are no LLM steps
    assert est.per_chain_total_tokens == 0
    assert est.total_tokens == 0
    assert est.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_summary_includes_key_numbers(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="gpt-4o-mini"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=3,
            smoke_check=True,
        )
        est = ev.estimate_cost(
            _ctx_factory, pricing={"gpt-4o-mini": (0.00015, 0.0006)}
        )
        text = est.format_summary()
        assert "ChainEvolver cost projection" in text
        assert "population_size:" in text
        assert "generations:" in text
        assert "total cost:" in text
        # Smoke check status mentioned
        assert "smoke check:" in text
        # Numbers appear
        assert "2" in text  # pop
        assert "3" in text  # gens

    def test_summary_warns_about_missing_pricing(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="exotic-x"),
            SimpleDataset([DataCase(input="a")]),
            _ConstMetric(),
            population_size=1,
            generations=1,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory, pricing={"gpt-4o-mini": (0.0001, 0.0001)})
        text = est.format_summary()
        assert "missing pricing" in text
        assert "exotic-x" in text


# ---------------------------------------------------------------------------
# Per-model awareness (§5 #8) — MODEL_SWAP routes onto pooled models
# ---------------------------------------------------------------------------


class TestPerModelAwareness:
    def _evolver_with_pool(self, pool: list[str]) -> ChainEvolver:
        from mmar_carl.chain_evolution import ChainMutator

        return ChainEvolver(
            _make_llm_chain(model="gpt-4o-mini"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            mutator=ChainMutator(model_pool=pool),
            population_size=2,
            generations=2,
            smoke_check=False,
        )

    _PRICING = {"gpt-4o-mini": (0.00015, 0.0006), "premium-model": (0.005, 0.015)}

    def test_pooled_model_priced_separately(self) -> None:
        est = self._evolver_with_pool(["premium-model"]).estimate_cost(
            _ctx_factory, pricing=self._PRICING
        )
        # both the base chain's model and the mutator's pooled model are reachable
        assert set(est.per_chain_cost_by_model) == {"gpt-4o-mini", "premium-model"}
        assert (
            est.per_chain_cost_by_model["premium-model"]
            > est.per_chain_cost_by_model["gpt-4o-mini"]
        )

    def test_upper_bound_uses_priciest_reachable_model(self) -> None:
        est = self._evolver_with_pool(["premium-model"]).estimate_cost(
            _ctx_factory, pricing=self._PRICING
        )
        # the point estimate is the base model; the upper bound is the premium model
        assert est.total_cost_usd_upper > est.total_cost_usd
        assert est.total_cost_usd_upper == pytest.approx(
            est.per_chain_cost_by_model["premium-model"] * est.total_chain_runs
        )

    def test_single_model_upper_equals_point_estimate(self) -> None:
        ev = ChainEvolver(
            _make_llm_chain(model="gpt-4o-mini"),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=1,
            smoke_check=False,
        )
        est = ev.estimate_cost(_ctx_factory, pricing={"gpt-4o-mini": (0.00015, 0.0006)})
        assert list(est.per_chain_cost_by_model) == ["gpt-4o-mini"]
        assert est.total_cost_usd_upper == pytest.approx(est.total_cost_usd)

    def test_no_pricing_no_per_model_breakdown(self) -> None:
        est = self._evolver_with_pool(["premium-model"]).estimate_cost(_ctx_factory)
        assert est.per_chain_cost_by_model == {}
        assert est.total_cost_usd_upper == 0.0

    def test_summary_surfaces_upper_bound_and_breakdown(self) -> None:
        est = self._evolver_with_pool(["premium-model"]).estimate_cost(
            _ctx_factory, pricing=self._PRICING
        )
        text = est.format_summary()
        assert "total cost (upper):" in text
        assert "per-chain by model:" in text
        assert "premium-model" in text
