"""No-execution validation of step / chain specs via ``validate_spec``."""

from __future__ import annotations

import pytest

from mmar_carl import auto_generate_minimal_step, get_step_catalog, validate_spec
from mmar_carl.models.enums import StepType


def test_valid_two_step_chain_passes() -> None:
    chain = {
        "steps": [
            auto_generate_minimal_step("llm", number=1),
            auto_generate_minimal_step("tool", number=2, dependencies=[1]),
        ]
    }
    report = validate_spec(chain)
    assert report.valid, report.format_text()
    assert report.errors == []


@pytest.mark.parametrize("step_type", list(StepType), ids=lambda s: s.value)
def test_every_catalog_example_validates(step_type: StepType) -> None:
    example = get_step_catalog().step(step_type.value).example
    # evaluate the single step (dependency targets aren't required for one step)
    report = validate_spec(example)
    assert report.valid, f"{step_type.value}: {report.format_text()}"


def test_missing_aim_is_an_error() -> None:
    report = validate_spec({"number": 1, "title": "x", "step_type": "llm"})
    assert not report.valid
    assert any(e.code == "missing_required_field" and e.field == "aim" for e in report.errors)


def test_missing_required_config_field_is_an_error() -> None:
    report = validate_spec({"number": 1, "title": "x", "step_type": "tool", "step_config": {}})
    assert not report.valid
    assert any("tool_name" in (e.field or "") for e in report.errors)


def test_unknown_step_type_is_an_error() -> None:
    report = validate_spec({"number": 1, "title": "x", "step_type": "nope"})
    assert not report.valid
    assert any(e.code == "unknown_step_type" for e in report.errors)


def test_duplicate_numbers_and_bad_dependency() -> None:
    chain = {
        "steps": [
            {"number": 1, "title": "a", "step_type": "llm", "aim": "x"},
            {"number": 1, "title": "b", "step_type": "llm", "aim": "y", "dependencies": [9]},
        ]
    }
    report = validate_spec(chain)
    assert not report.valid
    codes = {e.code for e in report.errors}
    assert "duplicate_number" in codes
    assert "missing_dependency" in codes


def test_cycle_is_detected() -> None:
    chain = {
        "steps": [
            {"number": 1, "title": "a", "step_type": "llm", "aim": "x", "dependencies": [2]},
            {"number": 2, "title": "b", "step_type": "llm", "aim": "y", "dependencies": [1]},
        ]
    }
    report = validate_spec(chain)
    assert not report.valid
    assert any(e.code == "cycle" for e in report.errors)


def test_empty_chain_is_invalid() -> None:
    report = validate_spec({"steps": []})
    assert not report.valid
    assert any(e.code == "empty" for e in report.errors)


def test_memory_read_without_writer_warns() -> None:
    chain = {
        "steps": [
            {
                "number": 1,
                "title": "use",
                "step_type": "tool",
                "step_config": {"tool_name": "t", "input_mapping": {"q": "$memory.ns.k"}},
            },
        ]
    }
    report = validate_spec(chain)
    assert report.valid  # warning, not error
    assert any(w.code == "memory_no_writer" for w in report.warnings)


def test_no_execution_required_context() -> None:
    # validate_spec must work with no ReasoningContext / no LLM client at all.
    report = validate_spec(auto_generate_minimal_step("supervisor"))
    assert report.valid, report.format_text()


@pytest.mark.parametrize("spec, field", [
    (None, "steps.0"),
    ({"steps": "not a list"}, "steps"),
    ({"steps": [False]}, "steps.0"),
    ({"steps": [{"number": [], "title": "a", "aim": "a"}]}, "steps.0.number"),
    ({"steps": [{"number": 1, "title": "a", "aim": "a", "dependencies": [[]]}]}, "steps.0.dependencies"),
    ({"steps": [{"number": 1, "title": "a", "aim": "a", "loop_back_to": []}]}, "steps.0.loop_back_to"),
    ({"steps": [{"number": 1, "title": "a", "aim": "a", "step_type": []}]}, "step_type"),
    ({"steps": [{"number": 1, "title": "a", "aim": "a"}], "metadata": "invalid"}, "metadata"),
])
def test_malformed_json_returns_a_validation_report(spec, field):
    report = validate_spec(spec)
    assert not report.valid
    assert report.errors[0].field == field


def test_non_object_llm_config_is_rejected():
    report = validate_spec({"number": 1, "title": "a", "aim": "a", "llm_config": "invalid"})
    assert not report.valid
    assert report.errors[0].code == "invalid_config"
    assert report.errors[0].field == "llm_config"
