"""Usability surface: structured choices/constraints, nested-model expansion,
field-path get/set, DAG mutation helpers, validate_and_build, and polish.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    auto_generate_minimal_step,
    can_delete_step,
    config_class_names,
    find_leaf_steps,
    get_field_by_path,
    get_step_catalog,
    iter_mutable_values,
    reference_fields,
    schema_for,
    set_field_by_path,
    validate_and_build,
    validate_spec,
)
from mmar_carl.meta.models import CATALOG_SCHEMA_VERSION
from mmar_carl.models.enums import StepType


@pytest.fixture(scope="module")
def catalog():
    return get_step_catalog()


# --- structured choices + constraints --------------------------------------


def test_choices_unified_across_enum_and_literal(catalog) -> None:
    # Literal field
    assert catalog.step("transform").field_choices("transform_type") == [
        "extract",
        "format",
        "aggregate",
        "filter",
        "map",
    ]
    # Enum field — same shape (list[str]), not a different grammar
    assert catalog.step("evaluation").field_choices("on_fail") == ["continue", "abort", "retry_with_feedback"]


def test_numeric_constraints_are_structured(catalog) -> None:
    assert catalog.step("evaluation").field_constraint("max_retries", "ge") == 0
    # nested: llm_config.temperature carries ge/le
    lc = catalog.step("agent_skill").field("llm_config")
    temp = next(f for f in lc.nested_fields if f.name == "temperature")
    assert temp.constraints == {"ge": 0.0, "le": 2.0}


def test_field_choices_none_for_plain_field(catalog) -> None:
    assert catalog.step("tool").field_choices("tool_name") is None


# --- nested model expansion ------------------------------------------------


def test_nested_scalar_model_expanded(catalog) -> None:
    server = catalog.step("mcp").nested_spec("server")
    names = {f.name for f in server}
    assert {"server_name", "transport", "command", "url"} <= names
    transport = next(f for f in server if f.name == "transport")
    assert transport.choices == ["stdio", "http", "sse"]


def test_nested_list_model_expanded(catalog) -> None:
    branches = catalog.step("conditional").field("branches")
    assert branches.container == "list"
    assert branches.nested_model == "ConditionalBranch"
    assert {f.name for f in branches.nested_fields} == {"condition", "next_step"}


def test_union_of_models_not_expanded(catalog) -> None:
    # ToolDiscoveryStepConfig.source is a union of 3 sources — ambiguous, skip.
    src = catalog.step("tool_discovery").field("source")
    assert src.nested_model is None
    assert src.nested_fields is None


def test_nested_expansion_is_depth_bounded(catalog) -> None:
    # No infinite recursion; every FieldSpec is a finite tree.
    def depth(fields, d=0):
        return max([d] + [depth(f.nested_fields, d + 1) for f in fields if f.nested_fields])

    for spec in catalog.step_types.values():
        assert depth(spec.config_fields) <= 2


# --- accepts_references ----------------------------------------------------


def test_accepts_references_flag(catalog) -> None:
    assert catalog.step("conditional").field("condition_context_key").accepts_references is True
    assert catalog.step("tool").field("input_mapping").accepts_references is True
    assert catalog.step("tool").field("output_key").accepts_references is False


# --- field-path get/set ----------------------------------------------------


def test_get_field_by_path_variants() -> None:
    cond = auto_generate_minimal_step("conditional", number=1)
    assert get_field_by_path(cond, "step_config.branches.0.next_step") == 3
    assert get_field_by_path(cond, "step_config.nope", default="D") == "D"
    with pytest.raises(KeyError):
        get_field_by_path(cond, "step_config.nope")


def test_set_field_by_path_creates_nested() -> None:
    step = auto_generate_minimal_step("llm", number=1)
    set_field_by_path(step, "llm_config.temperature", 0.9)
    assert step["llm_config"]["temperature"] == 0.9
    set_field_by_path(step, "llm_config.model", "anthropic/claude-3.5-sonnet")
    assert step["llm_config"]["model"] == "anthropic/claude-3.5-sonnet"


def test_set_field_by_path_list_index() -> None:
    cond = auto_generate_minimal_step("conditional", number=1)
    set_field_by_path(cond, "step_config.branches.0.next_step", 9)
    assert cond["step_config"]["branches"][0]["next_step"] == 9


def test_set_field_by_path_validate_rejects_out_of_range() -> None:
    # temperature is constrained to [0, 2]; validate=True must reject 9.9 and
    # leave the step untouched.
    step = auto_generate_minimal_step("llm", number=1)
    with pytest.raises(ValueError, match="invalid"):
        set_field_by_path(step, "llm_config.temperature", 9.9, validate=True)
    # step wasn't mutated by the rejected write
    assert get_field_by_path(step, "llm_config.temperature", default=None) != 9.9


def test_set_field_by_path_validate_commits_valid_value() -> None:
    step = auto_generate_minimal_step("llm", number=1)
    set_field_by_path(step, "llm_config.temperature", 1.0, validate=True)
    assert step["llm_config"]["temperature"] == 1.0
    assert validate_spec(step).valid


def test_set_field_by_path_default_is_unguarded() -> None:
    # back-compat: without validate=True the writer stays "dumb" (no exception)
    step = auto_generate_minimal_step("llm", number=1)
    set_field_by_path(step, "llm_config.temperature", 9.9)
    assert step["llm_config"]["temperature"] == 9.9


def test_iter_mutable_values() -> None:
    tool = auto_generate_minimal_step("tool", number=1)
    pairs = dict(iter_mutable_values(tool))
    assert "step_config.input_mapping" in pairs
    assert pairs["step_config.input_mapping"] == {"query": "$history[-1]"}


def test_mutation_loop_produces_valid_step() -> None:
    # read -> mutate within constraints -> set -> validate (the full public loop)
    step = auto_generate_minimal_step("llm", number=1)
    temp = next(f for f in get_step_catalog().step("llm").field("llm_config").nested_fields if f.name == "temperature")
    lo, hi = temp.constraints["ge"], temp.constraints["le"]
    set_field_by_path(step, "llm_config.temperature", (lo + hi) / 2)  # 1.0, within [0, 2]
    assert validate_spec(step).valid


# --- DAG mutation-safety ---------------------------------------------------


def test_find_leaf_and_can_delete() -> None:
    steps = [
        auto_generate_minimal_step("llm", number=1),
        auto_generate_minimal_step("llm", number=2, dependencies=[1]),
        auto_generate_minimal_step("tool", number=3, dependencies=[1]),
    ]
    assert find_leaf_steps(steps) == {2, 3}
    assert can_delete_step(steps, 3) is True
    assert can_delete_step(steps, 1) is False  # depended upon


def test_cannot_delete_last_step() -> None:
    steps = [auto_generate_minimal_step("llm", number=1)]
    assert can_delete_step(steps, 1) is False


# --- validate_and_build ----------------------------------------------------


def test_validate_and_build_success() -> None:
    report, chain = validate_and_build(
        {
            "steps": [
                auto_generate_minimal_step("llm", number=1),
                auto_generate_minimal_step("tool", number=2, dependencies=[1]),
            ]
        }
    )
    assert report.valid
    assert chain is not None
    assert len(chain.steps) == 2


def test_validate_and_build_invalid_returns_none() -> None:
    report, chain = validate_and_build({"number": 1, "title": "x", "step_type": "llm"})
    assert not report.valid
    assert chain is None
    assert "aim" in report.errors[0].format_for_llm()


def test_validate_and_build_single_step_runs_cross_checks() -> None:
    # a lone step with an unresolved dep is caught once normalised to chain form
    report, chain = validate_and_build({"number": 2, "title": "x", "step_type": "llm", "aim": "y", "dependencies": [1]})
    assert not report.valid
    assert chain is None
    assert any(e.code == "missing_dependency" for e in report.errors)


# --- reference_fields / schema_for(config class) ---------------------------


def test_reference_fields_accessor() -> None:
    assert reference_fields("conditional") == ["step_config.condition_context_key", "step_config.branches.*.condition"]
    assert reference_fields("llm") == []


def test_schema_for_accepts_config_class_names() -> None:
    assert "MCPServerConfig" in config_class_names()
    assert "ConditionalBranch" in config_class_names()
    schema = schema_for("MCPServerConfig")
    assert "transport" in schema.get("properties", {})


def test_schema_for_unknown_raises() -> None:
    with pytest.raises(KeyError):
        schema_for("NotAThing")


# --- format_for_llm --------------------------------------------------------


@pytest.mark.parametrize(
    "issue_code,keyword",
    [
        ("missing_required_field", "missing"),
        ("duplicate_number", "unique"),
        ("cycle", "circular"),
    ],
)
def test_format_for_llm_covers_codes(issue_code, keyword) -> None:
    from mmar_carl import ValidationIssue

    issue = ValidationIssue(severity="error", code=issue_code, message="cycle involving step 1", step=1, field="aim")
    assert keyword in issue.format_for_llm().lower()


# --- notebook rendering + version ------------------------------------------


def test_step_type_spec_repr_markdown(catalog) -> None:
    md = catalog.step("conditional")._repr_markdown_()
    assert "conditional" in md
    assert "```text" in md


def test_catalog_schema_version_is_2(catalog) -> None:
    assert CATALOG_SCHEMA_VERSION == 2
    assert catalog.schema_version == 2


@pytest.mark.parametrize("step_type", list(StepType), ids=lambda s: s.value)
def test_every_field_serializes_with_new_fields(step_type: StepType, catalog) -> None:
    # new FieldSpec fields must round-trip through the catalog JSON
    from mmar_carl import StepCatalog

    restored = StepCatalog.from_json(catalog.to_json())
    assert restored.step(step_type.value).config_fields is not None
