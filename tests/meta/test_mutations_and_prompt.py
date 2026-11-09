"""Mutation/template helpers and the auto-generated generation prompt."""

from __future__ import annotations

import pytest

from mmar_carl import (
    ReasoningChain,
    auto_generate_minimal_step,
    get_step_catalog,
    mutable_fields,
    validate_spec,
)
from mmar_carl.chain import _create_typed_step_from_dict
from mmar_carl.models.enums import StepType


@pytest.mark.parametrize("step_type", list(StepType), ids=lambda s: s.value)
def test_minimal_step_is_constructible_and_valid(step_type: StepType) -> None:
    spec = auto_generate_minimal_step(step_type.value, number=3, dependencies=[1])
    assert spec["number"] == 3
    assert spec["step_type"] == step_type.value
    # validates without execution ...
    assert validate_spec(spec).valid, validate_spec(spec).format_text()
    # ... and reconstructs into a real typed step (INSERT_STEP ready)
    step = _create_typed_step_from_dict(spec)
    assert step.step_type == step_type


def test_minimal_step_overrides() -> None:
    spec = auto_generate_minimal_step("llm", number=7, title="Custom", dependencies=[2, 3])
    assert spec["number"] == 7
    assert spec["title"] == "Custom"
    assert spec["dependencies"] == [2, 3]


def test_mutable_fields_never_include_runtime_only() -> None:
    for name in get_step_catalog().names():
        spec = get_step_catalog().step(name)
        runtime_only = set(spec.runtime_only_fields)
        for mf in mutable_fields(name):
            leaf = mf.split(".")[-1]
            assert leaf not in runtime_only


def test_insert_step_into_existing_chain_round_trips() -> None:
    base = {"steps": [auto_generate_minimal_step("llm", number=1)]}
    base["steps"].append(auto_generate_minimal_step("parallel_sampling", number=2, dependencies=[1]))
    assert validate_spec(base).valid, validate_spec(base).format_text()
    chain = ReasoningChain.from_dict(base, use_typed_steps=True)
    assert len(chain.steps) == 2
    assert chain.steps[1].step_type == StepType.PARALLEL_SAMPLING


# --- generation prompt ------------------------------------------------------


def test_generation_prompt_covers_all_types() -> None:
    cat = get_step_catalog()
    prompt = cat.generation_prompt(max_steps=8)
    for name in cat.names():
        assert name in prompt


def test_generation_prompt_includes_reference_grammar_and_rules() -> None:
    prompt = get_step_catalog().generation_prompt()
    assert "$history" in prompt and "$memory" in prompt
    assert "NOT an array" in prompt
    assert "triggered_by" in prompt


def test_generation_prompt_menu_is_smaller_than_cards() -> None:
    cat = get_step_catalog()
    menu = cat.generation_prompt(detail="menu")
    cards = cat.generation_prompt(detail="cards")
    assert len(menu) < len(cards)


def test_generation_prompt_can_filter_types() -> None:
    prompt = get_step_catalog().generation_prompt(include=["llm", "tool"], detail="cards")
    # tier-1 cards only for the included types
    assert "### llm" in prompt
    assert "### debate" not in prompt
