"""SelectionStep — pick item(s) from a collection by a strategy (no LLM call).

Covers every strategy (top_k / bottom_k / weighted / random / filter / all), the
count==1 scalar vs count>1 list contract, score_key extraction, deterministic
seeding, the filter predicate, memory write-through, empty/non-list sources, and
spec round-tripping through the typed-step machinery + catalog.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    SelectionStepConfig,
    SelectionStepDescription,
)
from mmar_carl.meta import validate_spec


def _ctx(collection, namespace="pool", key="cands"):
    ctx = ReasoningContext(outer_context="x", api=None)
    ctx.memory_write(key, collection, namespace)
    return ctx


def _step(**cfg):
    cfg.setdefault("source", "$memory.pool.cands")
    return SelectionStepDescription(number=1, title="pick", config=SelectionStepConfig(**cfg))


async def _run(step, ctx):
    res = await ReasoningChain(steps=[step]).execute_async(ctx)
    assert res.success, res.step_results[0].error_message
    return res.step_results[0].result_data


SCORED = [{"id": "a", "fitness": 0.1}, {"id": "b", "fitness": 0.9}, {"id": "c", "fitness": 0.5}]


@pytest.mark.asyncio
async def test_top_k_single_returns_scalar():
    got = await _run(_step(strategy="top_k", count=1, score_key="fitness"), _ctx(SCORED))
    assert got == {"id": "b", "fitness": 0.9}  # scalar, not a list


@pytest.mark.asyncio
async def test_top_k_multi_returns_ordered_list():
    got = await _run(_step(strategy="top_k", count=2, score_key="fitness"), _ctx(SCORED))
    assert [it["id"] for it in got] == ["b", "c"]


@pytest.mark.asyncio
async def test_bottom_k():
    got = await _run(_step(strategy="bottom_k", count=2, score_key="fitness"), _ctx(SCORED))
    assert [it["id"] for it in got] == ["a", "c"]


@pytest.mark.asyncio
async def test_top_k_bare_numeric_items():
    got = await _run(_step(strategy="top_k", count=2), _ctx([3, 1, 5, 2]))
    assert got == [5, 3]


@pytest.mark.asyncio
async def test_weighted_is_deterministic_under_seed():
    a = await _run(_step(strategy="weighted", count=1, score_key="fitness", seed=0), _ctx(SCORED))
    b = await _run(_step(strategy="weighted", count=1, score_key="fitness", seed=0), _ctx(SCORED))
    assert a == b


@pytest.mark.asyncio
async def test_weighted_favours_high_score():
    # Over many seeds, the high-fitness item should be chosen most often.
    wins = {"a": 0, "b": 0, "c": 0}
    for s in range(60):
        got = await _run(_step(strategy="weighted", count=1, score_key="fitness", seed=s), _ctx(SCORED))
        wins[got["id"]] += 1
    assert wins["b"] > wins["a"]  # 0.9 beats 0.1


@pytest.mark.asyncio
async def test_weighted_all_zero_scores_falls_back_to_uniform():
    zeros = [{"id": "a", "fitness": 0.0}, {"id": "b", "fitness": 0.0}]
    got = await _run(_step(strategy="weighted", count=2, score_key="fitness", seed=1), _ctx(zeros))
    assert {it["id"] for it in got} == {"a", "b"}  # both selected, no crash


@pytest.mark.asyncio
async def test_random_deterministic_and_count():
    a = await _run(_step(strategy="random", count=2, seed=3), _ctx([1, 2, 3, 4, 5]))
    b = await _run(_step(strategy="random", count=2, seed=3), _ctx([1, 2, 3, 4, 5]))
    assert a == b and len(a) == 2


@pytest.mark.asyncio
async def test_filter_predicate_keeps_matching():
    got = await _run(_step(strategy="filter", predicate="item['fitness'] >= 0.5", score_key="fitness"), _ctx(SCORED))
    assert {it["id"] for it in got} == {"b", "c"}


@pytest.mark.asyncio
async def test_filter_bare_numeric_predicate():
    got = await _run(_step(strategy="filter", predicate="x > 2"), _ctx([1, 2, 3, 4]))
    assert got == [3, 4]


@pytest.mark.asyncio
async def test_all_returns_whole_collection():
    got = await _run(_step(strategy="all"), _ctx([1, 2, 3]))
    assert got == [1, 2, 3]


@pytest.mark.asyncio
async def test_empty_collection_scalar_strategy_returns_none():
    got = await _run(_step(strategy="top_k", count=1, score_key="fitness"), _ctx([]))
    assert got is None


@pytest.mark.asyncio
async def test_empty_collection_list_strategy_returns_empty_list():
    got = await _run(_step(strategy="filter", predicate="x > 0"), _ctx([]))
    assert got == []


@pytest.mark.asyncio
async def test_non_list_source_is_wrapped():
    got = await _run(_step(strategy="top_k", count=1), _ctx(42))
    assert got == 42


@pytest.mark.asyncio
async def test_count_larger_than_collection_clamps():
    got = await _run(_step(strategy="top_k", count=10), _ctx([3, 1, 2]))
    assert got == [3, 2, 1]


@pytest.mark.asyncio
async def test_output_key_writes_selection_to_memory():
    ctx = _ctx(SCORED)
    step = _step(strategy="top_k", count=1, score_key="fitness", output_key="chosen", namespace="sel")
    await _run(step, ctx)
    assert ctx.memory_read("chosen", "sel") == {"id": "b", "fitness": 0.9}


@pytest.mark.asyncio
async def test_source_from_history():
    ctx = ReasoningContext(outer_context="x", api=None)
    ctx.history = ["[3, 9, 1]"]  # note: history entries are strings
    # a bare list literal string isn't a Python list, so wrap-as-single-item is expected;
    # use memory for real collections. Here we assert graceful handling (single wrapped item).
    got = await _run(_step(source="$history[-1]", strategy="all"), ctx)
    assert got == ["[3, 9, 1]"]


def test_spec_validates_and_round_trips():
    step = _step(strategy="weighted", count=1, score_key="fitness", seed=0)
    spec = ReasoningChain(steps=[step]).to_dict()
    assert validate_spec(spec).valid
    chain2 = ReasoningChain.from_dict(spec, use_typed_steps=True)
    assert chain2.steps[0].step_type.value == "selection"
    assert chain2.steps[0].step_config.strategy == "weighted"


def test_catalog_exposes_selection():
    from mmar_carl.meta import get_step_catalog

    cat = get_step_catalog()
    assert "selection" in cat.tier0()
    card = cat.tier1(["selection"])
    assert "strategy" in str(card)  # the card mentions the strategy field
