"""Dataset subsample/split/select + correct/incorrect partition + cross-run diff,
plus bounded-concurrency evaluation.
"""

from __future__ import annotations

import asyncio
import random

from mmar_carl import LLMStepDescription, ReasoningChain, ReasoningContext
from mmar_carl.dataset_evaluator import DatasetEvaluator
from mmar_carl.metrics import MetricBase
from mmar_carl.models.dataset import (
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    SimpleDataset,
    ThresholdStrategy,
    select_cases,
    split,
    subsample,
)
from mmar_carl.models.llm_client_base import LLMClientBase


def _ds(n: int = 10) -> SimpleDataset:
    return SimpleDataset([DataCase(input=f"in{i}", label=f"c{i}") for i in range(n)])


# --- subsample / split / select --------------------------------------------


def test_subsample_deterministic_and_bounded() -> None:
    rng = random.Random(0)
    sub = list(subsample(_ds(), 3, rng=rng))
    assert len(sub) == 3
    # deterministic given the same seed
    assert [c.label for c in subsample(_ds(), 3, rng=random.Random(0))] == [c.label for c in sub]


def test_subsample_head_and_oversize() -> None:
    rng = random.Random(0)
    assert [c.label for c in subsample(_ds(), 3, rng=rng, strategy="head")] == ["c0", "c1", "c2"]
    assert len(list(subsample(_ds(3), 99, rng=rng))) == 3  # k >= size returns all


def test_split_partitions_all_cases() -> None:
    a, b = split(_ds(10), 0.7, rng=random.Random(1))
    assert len(list(a)) == 7 and len(list(b)) == 3


def test_select_cases_by_label() -> None:
    sel = select_cases(_ds(), {"c1", "c4"})
    assert sorted(c.label for c in sel) == ["c1", "c4"]


# --- correct/incorrect partition (score-derived, NOT success) --------------


def _result(label: str, score: float, success: bool = True) -> CaseEvaluationResult:
    return CaseEvaluationResult(case=DataCase(input="x", label=label), score=score, chain_output="o", success=success)


def _report(results) -> DatasetEvaluationReport:
    scores = [r.score for r in results]
    return DatasetEvaluationReport(
        metric_name="m",
        strategy=ThresholdStrategy(threshold=1.0),
        all_results=results,
        selected_cases=[],
        mean_score=sum(scores) / len(scores),
        min_score=min(scores),
        max_score=max(scores),
    )


def test_partition_is_score_derived_not_success() -> None:
    # c3 ran fine (success=True) but scored 0 -> INCORRECT
    rep = _report([_result("c0", 1.0), _result("c1", 1.0), _result("c2", 0.0), _result("c3", 0.0, success=True)])
    sp = rep.partition()
    assert {r.case.label for r in sp.correct} == {"c0", "c1"}
    assert {r.case.label for r in sp.incorrect} == {"c2", "c3"}


def test_partition_custom_threshold_and_predicate() -> None:
    rep = _report([_result("a", 0.6), _result("b", 0.4)])
    assert {r.case.label for r in rep.partition(threshold=0.5).correct} == {"a"}
    assert {r.case.label for r in rep.partition(correct=lambda r: r.score > 0.5).correct} == {"a"}


def test_diff_by_case_finds_regressions() -> None:
    old = _report([_result("c0", 1.0), _result("c1", 1.0), _result("c2", 0.0), _result("c3", 0.0)])
    new = _report([_result("c0", 1.0), _result("c1", 0.0), _result("c2", 1.0), _result("c3", 0.0)])
    diff = new.diff_by_case(old)
    assert diff.regressed == ["c1"]  # was correct, now wrong
    assert diff.improved == ["c2"]
    assert set(diff.unchanged) == {"c0", "c3"}
    assert diff.score_delta_by_label["c1"] == -1.0


def test_by_label_uses_index_fallback() -> None:
    rep = _report([CaseEvaluationResult(case=DataCase(input="x"), score=1.0, chain_output="o", success=True)])
    assert "#0" in rep.by_label()


# --- bounded-concurrency evaluation ----------------------------------------


class _Mock(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


class _One(MetricBase):
    @property
    def name(self) -> str:
        return "one"

    async def compute_async(self, output) -> float:
        return 1.0


def test_concurrent_evaluation_preserves_order() -> None:
    ds = _ds(6)
    chain = ReasoningChain(steps=[LLMStepDescription(number=1, title="A", aim="a")])
    ev = DatasetEvaluator(
        chain=chain, dataset=ds, metric=_One(), strategy=ThresholdStrategy(threshold=1.0), max_concurrent=4
    )
    rep = asyncio.run(ev.evaluate_async(lambda c: ReasoningContext(outer_context=c.input, api=_Mock())))
    assert [r.case.label for r in rep.all_results] == [f"c{i}" for i in range(6)]
    assert all(r.score == 1.0 for r in rep.all_results)
