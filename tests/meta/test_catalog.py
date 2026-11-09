"""Introspection correctness, progressive-disclosure tiers, schemas, and
serialization for the step-type catalog.
"""

from __future__ import annotations

import json

import pytest

from mmar_carl.meta import StepCatalog, get_step_catalog, schema_for
from mmar_carl.meta._introspect import render_type
from mmar_carl.models.enums import StepType


@pytest.fixture(scope="module")
def catalog():
    return get_step_catalog()


# --- discovery / required vs optional --------------------------------------


def test_llm_is_inlined_with_required_aim(catalog) -> None:
    spec = catalog.step("llm")
    assert spec.inlined is True
    assert spec.config_class is None
    # aim is field-optional (default="") but validator-required -> curated override
    assert any(f.name == "aim" and f.required for f in spec.required_fields)


def test_tool_requires_only_tool_name(catalog) -> None:
    spec = catalog.step("tool")
    assert spec.inlined is False
    assert spec.config_class == "ToolStepConfig"
    assert [f.name for f in spec.required_fields] == ["tool_name"]


def test_runtime_only_fields_are_not_offered_as_config(catalog) -> None:
    # handoff/supervisor/parallel_sampling carry an exclude=True field that must
    # never appear as a fillable config field.
    assert "sub_chain" in catalog.step("agent_handoff").runtime_only_fields
    assert "agents" in catalog.step("supervisor").runtime_only_fields
    assert "base_step" in catalog.step("parallel_sampling").runtime_only_fields
    for name in ("agent_handoff", "supervisor", "parallel_sampling"):
        spec = catalog.step(name)
        field_names = {f.name for f in spec.config_fields}
        assert not (field_names & set(spec.runtime_only_fields))


def test_excluded_base_fields_absent_from_required(catalog) -> None:
    # metrics / cache are exclude=True base fields — never required anywhere.
    for spec in catalog.step_types.values():
        names = {f.name for f in spec.config_fields}
        assert "metrics" not in names
        assert "cache" not in names


# --- compact type rendering ------------------------------------------------


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (int, "int"),
        (list[int], "list[int]"),
        (dict[str, str], "dict[str, str]"),
    ],
)
def test_render_type_basic(annotation, expected) -> None:
    assert render_type(annotation) == expected


def test_render_type_optional_and_union() -> None:
    from typing import Optional, Union

    assert render_type(Optional[int]) == "int?"
    assert render_type(Union[int, str]) == "int | str"


def test_render_type_enum_and_literal() -> None:
    from typing import Literal

    from mmar_carl.models.enums import MemoryOperation

    rendered = render_type(MemoryOperation)
    assert rendered.startswith("enum[") and "read" in rendered and "write" in rendered
    assert render_type(Literal["rule", "llm"]) == "Literal['rule', 'llm']"


# --- tiers ------------------------------------------------------------------


def test_tier0_is_small_and_complete(catalog) -> None:
    t0 = catalog.tier0()
    # one line per type, and small (~hundreds of tokens, well under any budget)
    assert t0.count("\n") + 1 == len(catalog.names())
    assert len(t0) < 3000


def test_tier1_cards_contain_required_fields_and_example(catalog) -> None:
    card = catalog.step("tool").card(full=False)
    assert "tool_name" in card
    assert "example:" in card
    # the example is valid JSON in chain-serialization shape
    example_json = card.split("example: ", 1)[1]
    spec = json.loads(example_json)
    assert spec["step_type"] == "tool"
    assert "step_config" in spec


def test_tier2_includes_descriptions_and_validators(catalog) -> None:
    full = catalog.tier2("evaluation")
    assert "criteria" in full
    assert "min_length 1" in full  # constraint extracted from the model


# --- tier 3 schemas ---------------------------------------------------------


@pytest.mark.parametrize("step_type", list(StepType), ids=lambda s: s.value)
def test_schema_for_works_for_every_type(step_type: StepType) -> None:
    schema = schema_for(step_type.value)
    assert isinstance(schema, dict)
    assert "properties" in schema or "type" in schema


def test_schema_for_llm_synthesizes_inlined_fields() -> None:
    # llm description class can't generate a schema (inherited Callable field),
    # so the catalog synthesises one from the inlined fields.
    schema = schema_for("llm")
    assert "aim" in schema.get("properties", {})


# --- serialization ----------------------------------------------------------


def test_catalog_json_round_trip(catalog) -> None:
    restored = StepCatalog.from_json(catalog.to_json())
    assert set(restored.names()) == set(catalog.names())
    assert restored.schema_version == catalog.schema_version
    assert restored.step("tool").required_fields[0].name == "tool_name"


def test_catalog_to_json_is_versioned(catalog) -> None:
    payload = json.loads(catalog.to_json())
    assert "schema_version" in payload
    assert "carl_version" in payload
