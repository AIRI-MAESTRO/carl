"""Structural + content mutation primitives, DAG queries, and frozen guards."""

from __future__ import annotations

import random

import pytest

from mmar_carl import ReasoningChain, LLMStepDescription
from mmar_carl.meta import (
    apply_records,
    auto_generate_minimal_step as gen,
    change_step_type,
    deletable_steps,
    dependents_of,
    expand_field_path,
    find_leaf_steps,
    find_root_steps,
    frozen_step_numbers,
    insert_step_at,
    mutable_targets,
    mutate_content,
    remove_step,
    rewire_dependencies,
    sample_field_value,
    steps_of_type,
    validate_spec,
)
from mmar_carl.meta.mutations import MutationError


def _chain3():
    return {
        "steps": [
            gen("llm", number=1),
            gen("tool", number=2, dependencies=[1]),
            gen("memory", number=3, dependencies=[2]),
        ]
    }


# --- frozen field round-trip -----------------------------------------------


def test_frozen_round_trips() -> None:
    ch = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="A", aim="a", frozen=True),
            LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
        ]
    )
    d = ch.to_dict()
    assert d["steps"][0]["frozen"] is True
    restored = ReasoningChain.from_dict(d, use_typed_steps=True)
    assert restored.steps[0].frozen is True
    assert restored.steps[1].frozen is False
    assert frozen_step_numbers(d["steps"]) == {1}


# --- DAG queries -----------------------------------------------------------


def test_dag_queries() -> None:
    steps = _chain3()["steps"]
    assert find_root_steps(steps) == {1}
    assert find_leaf_steps(steps) == {3}
    assert dependents_of(steps, 1) == {2}
    assert steps_of_type(steps, "tool") == [2]


# --- topology mutations ----------------------------------------------------


def test_rewire_dependencies() -> None:
    spec = _chain3()
    new, rec = rewire_dependencies(spec, 3, [1])
    assert new["steps"][2]["dependencies"] == [1]
    assert rec.op == "rewire" and rec.after == [1]
    assert validate_spec(new).valid


def test_rewire_rejects_bad_dep_and_cycle() -> None:
    spec = _chain3()
    with pytest.raises(MutationError):
        rewire_dependencies(spec, 3, [99])
    with pytest.raises(MutationError):
        rewire_dependencies(spec, 1, [3])  # would create a cycle 1->3->2->1


def test_change_step_type() -> None:
    spec = _chain3()
    new, rec = change_step_type(spec, 2, "structured_output")
    assert new["steps"][1]["step_type"] == "structured_output"
    assert rec.new_step_type == "structured_output"
    assert validate_spec(new).valid


def test_change_step_type_with_field_map() -> None:
    spec = {"steps": [gen("tool", number=1)]}
    spec["steps"][0]["step_config"]["tool_name"] = "foo"
    new, _ = change_step_type(spec, 1, "mcp", field_map={"step_config.tool_name": "step_config.tool_name"})
    assert new["steps"][0]["step_config"]["tool_name"] == "foo"


def test_insert_step_at_splice() -> None:
    spec = _chain3()
    new, rec = insert_step_at(spec, gen("transform"), after=1, rewire_dependents=True)
    # new step (4) depends on 1; step 2 now depends on 4
    step4 = next(s for s in new["steps"] if s["number"] == 4)
    step2 = next(s for s in new["steps"] if s["number"] == 2)
    assert step4["dependencies"] == [1]
    assert step2["dependencies"] == [4]
    assert validate_spec(new).valid


def test_insert_step_before_gate() -> None:
    spec = {"steps": [gen("llm", number=1), gen("llm", number=2, dependencies=[1])]}
    new, _ = insert_step_at(spec, gen("evaluation")["step_config"] and gen("llm"), before=2)
    step2 = next(s for s in new["steps"] if s["number"] == 2)
    assert step2["dependencies"] == [3]  # gated behind the new step
    assert validate_spec(new).valid


def test_remove_step_heals_dependents() -> None:
    spec = _chain3()
    new, rec = remove_step(spec, 2, heal=True)
    step3 = next(s for s in new["steps"] if s["number"] == 3)
    assert step3["dependencies"] == [1]  # bypassed removed step 2
    assert rec.op == "remove"
    assert validate_spec(new).valid


def test_remove_last_step_rejected() -> None:
    with pytest.raises(MutationError):
        remove_step({"steps": [gen("llm", number=1)]}, 1)


def test_remove_heals_dangling_loop() -> None:
    spec = {
        "steps": [
            gen("llm", number=1),
            {**gen("llm", number=2, dependencies=[1]), "loop_back_to": 1, "loop_config": {"max_iterations": 3}},
        ]
    }
    # removing step 1 (the loop target) must clear the loop edge on step 2
    new, _ = remove_step(spec, 1, heal=True)
    step2 = new["steps"][0]
    assert step2["loop_back_to"] is None
    assert validate_spec(new).valid


# --- frozen guards ---------------------------------------------------------


def test_frozen_guards() -> None:
    spec = {"steps": [{**gen("llm", number=1), "frozen": True}, gen("tool", number=2, dependencies=[1])]}
    with pytest.raises(MutationError):
        remove_step(spec, 1)
    with pytest.raises(MutationError):
        change_step_type(spec, 1, "tool")
    assert deletable_steps(spec["steps"]) == {2}
    # respect_frozen=False overrides
    new, _ = remove_step(spec, 1, respect_frozen=False)
    assert validate_spec(new).valid


# --- content sampling ------------------------------------------------------


def test_sample_field_value_choices_and_range() -> None:
    rng = random.Random(1)
    # enum/Literal: returns a valid choice != current
    v = sample_field_value("transform", "step_config.transform_type", rng=rng, current="format")
    assert v in ["extract", "aggregate", "filter", "map"]
    # numeric range: temperature within [0, 2]
    t = sample_field_value("llm", "llm_config.temperature", rng=rng, current=0.5, mode="random")
    assert 0.0 <= t <= 2.0


def test_mutate_content_produces_valid_step() -> None:
    rng = random.Random(2)
    spec = {"steps": [gen("llm", number=1), gen("evaluation", number=2, dependencies=[1])]}
    new, rec = mutate_content(spec, 2, "step_config.evaluation_method", rng=rng, mode="random")
    assert rec.op == "set_field"
    assert new["steps"][1]["step_config"]["evaluation_method"] in ("rule", "llm")
    assert validate_spec(new).valid


def test_mutable_targets_skips_frozen() -> None:
    spec = {"steps": [{**gen("llm", number=1), "frozen": True}, gen("tool", number=2, dependencies=[1])]}
    targets = mutable_targets(spec)
    assert all(step_num == 2 for step_num, _, _ in targets)


# --- wildcard + replay -----------------------------------------------------


def test_expand_field_path_wildcard() -> None:
    cond = gen("conditional", number=1)
    paths = expand_field_path(cond, "step_config.branches.*.condition")
    assert paths == ["step_config.branches.0.condition"]


def test_sample_unset_numeric_field_yields_valid_number() -> None:
    # tool.timeout is optional (absent from the minimal step) and unbounded —
    # sampling must still produce a number (uses the field default), not None.
    rng = random.Random(3)
    spec = {"steps": [gen("llm", number=1), gen("tool", number=2, dependencies=[1])]}
    new, _ = mutate_content(spec, 2, "step_config.timeout", rng=rng, mode="perturb")
    assert isinstance(new["steps"][1]["step_config"]["timeout"], (int, float))
    assert validate_spec(new).valid


def test_validate_and_build_does_not_corrupt_spec() -> None:
    from mmar_carl import validate_and_build

    spec = {"steps": [gen("llm", number=1), gen("tool", number=2, dependencies=[1])]}
    report, chain = validate_and_build(spec)
    assert report.valid and chain is not None
    # from_dict rewrites step_config in place — validate_and_build must isolate it
    assert isinstance(spec["steps"][1]["step_config"], dict)


def test_apply_records_replays_lineage() -> None:
    base = {"steps": [gen("llm", number=1), gen("tool", number=2, dependencies=[1])]}
    _, rec = rewire_dependencies(base, 2, [])
    replayed = apply_records(base, [rec])
    assert replayed["steps"][1]["dependencies"] == []
    assert validate_spec(replayed).valid


# --- int perturb + unknown-path guard (regressions from live E2E, 2026-07) --


def test_perturb_changes_small_int_with_only_lower_bound() -> None:
    # selection.count is ``int ge=1`` with no upper bound — ±10% of the tiny
    # synthesised domain used to round back to `current` on every seed,
    # making perturb a guaranteed silent no-op for small ints.
    for seed in range(10):
        v = sample_field_value(
            "selection", "step_config.count", rng=random.Random(seed), current=2, mode="perturb"
        )
        assert isinstance(v, int)
        assert v >= 1
        assert v != 2


def test_perturb_int_at_lower_bound_moves_up() -> None:
    v = sample_field_value(
        "selection", "step_config.count", rng=random.Random(0), current=1, mode="perturb"
    )
    assert v == 2  # the -1 candidate is out of bounds; +1 is the only legal move


def test_perturb_int_ignores_synthesised_upper_bound() -> None:
    # No ``le`` on count: perturbing 50 must stay near 50, not get clamped
    # into the synthesised random-sampling window right above ``ge``.
    v = sample_field_value(
        "selection", "step_config.count", rng=random.Random(0), current=50, mode="perturb"
    )
    assert v in (49, 51)


def test_mutate_content_unknown_path_raises() -> None:
    spec = {"steps": [gen("selection", number=1)]}
    with pytest.raises(MutationError, match="unknown field path"):
        mutate_content(spec, 1, "config.count", rng=random.Random(0))
