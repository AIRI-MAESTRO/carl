"""RunDescriptor / ComplexityDescriptor extraction + N-D Pareto kernel."""

from __future__ import annotations

import pytest

from mmar_carl import LLMStepDescription, ReasoningChain, ReasoningContext
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.pareto import dominates, pareto_front, pareto_rank
from mmar_carl.run_descriptor import (
    ComplexityDescriptor,
    RunDescriptor,
    extract_complexity,
    extract_run_descriptor,
)


class _Mock(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _diamond() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="A", aim="a"),
            LLMStepDescription(number=2, title="B", aim="b"),  # parallel with 1
            LLMStepDescription(number=3, title="C", aim="c", dependencies=[1, 2]),
        ]
    )


# --- complexity ------------------------------------------------------------


def test_extract_complexity() -> None:
    c = extract_complexity(_diamond())
    assert c.num_steps == 3
    assert c.num_edges == 2
    assert c.max_depth == 2  # {1,2} then {3}
    assert c.max_parallel_batch == 2
    assert c.distinct_step_types == 1
    assert c.has_loops is False


# --- run descriptor --------------------------------------------------------


def test_extract_run_descriptor() -> None:
    chain = _diamond()
    result = __import__("asyncio").run(chain.execute_async(ReasoningContext(outer_context="x", api=_Mock())))
    d = extract_run_descriptor(
        chain=chain, result=result, quality=0.8, scores_by_metric={"acc": 0.8}, fault_tolerance=0.9
    )
    assert d.success is True
    assert d.quality == 0.8
    assert d.complexity.num_steps == 3
    assert d.fault_tolerance == 0.9
    assert d.value("metric.acc") == 0.8
    assert d.value("complexity.max_parallel_batch") == 2.0


def test_descriptor_as_objectives_and_behavior_key() -> None:
    d = RunDescriptor(
        success=True,
        quality=0.9,
        total_tokens=500,
        wall_time_s=1.2,
        llm_calls=3,
        complexity=ComplexityDescriptor(
            num_steps=4, num_edges=3, max_depth=3, max_parallel_batch=2, distinct_step_types=2, has_loops=False
        ),
        fault_tolerance=0.7,
    )
    assert d.as_objectives(["quality", "total_tokens", "complexity.num_steps"]) == (0.9, 500.0, 4.0)
    # niche key: tokens bin index + steps bin index
    key = d.behavior_key({"total_tokens": [100, 1000], "complexity.num_steps": [2, 5]})
    assert key == (1, 1)


def test_descriptor_value_default_for_missing() -> None:
    d = RunDescriptor(success=False, complexity=extract_complexity(_diamond()))
    assert d.value("quality", default=-1.0) == -1.0  # quality is None


# --- pareto ----------------------------------------------------------------


def test_dominates() -> None:
    # quality max, tokens min
    assert dominates((0.9, 50), (0.8, 100), ["max", "min"]) is True
    assert dominates((0.8, 100), (0.9, 50), ["max", "min"]) is False
    assert dominates((0.9, 50), (0.9, 50), ["max", "min"]) is False  # equal, no strict better


def test_pareto_front_and_rank() -> None:
    pts = [(0.9, 100), (0.8, 50), (0.7, 200), (0.9, 100)]
    front = pareto_front(pts, senses=["max", "min"])
    assert set(front) == {0, 1, 3}  # (0.7,200) is dominated
    ranks = pareto_rank(pts, ["max", "min"])
    assert ranks[2] == 1  # dominated point is on the second front
    assert ranks[0] == 0 and ranks[1] == 0


@pytest.mark.parametrize("senses", [["max", "min"], ["min", "max"]])
def test_pareto_front_never_empty(senses) -> None:
    pts = [(1.0, 2.0), (2.0, 1.0), (1.5, 1.5)]
    assert len(pareto_front(pts, senses=senses)) >= 1
