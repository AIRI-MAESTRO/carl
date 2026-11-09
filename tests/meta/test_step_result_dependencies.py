"""Executable data references must have a declared predecessor in a full spec."""

import pytest

from mmar_carl import MutationError, rewire_dependencies, validate_spec


def spec():
    return {"steps": [
        {"number": 1, "title": "Produce", "step_type": "llm", "aim": "Produce data"},
        {"number": 2, "title": "Bridge", "step_type": "llm", "aim": "Bridge", "dependencies": [1]},
        {"number": 3, "title": "Consume", "step_type": "tool", "dependencies": [2], "step_config": {
            "tool_name": "consume", "input_mapping": {"data": "$steps.1.result_data"},
            "tool_description": "$steps.999.result_data is a literal in this description",
        }},
    ]}


def test_mutation_cannot_detach_indirect_producer():
    original = spec()
    assert validate_spec(original).valid
    with pytest.raises(MutationError, match="not a dependency"):
        rewire_dependencies(original, 2, [])
    assert original["steps"][1]["dependencies"] == [1]


def test_missing_step_reference_is_invalid_in_a_chain_only():
    candidate = spec()
    step = candidate["steps"][-1]
    step["step_config"]["input_mapping"]["data"] = "$steps.999.result"
    assert validate_spec(step).valid
    report = validate_spec(candidate)
    assert not report.valid
    assert report.errors[0].code == "step_result_missing_producer"


def test_scoped_structured_inputs_are_checked_with_config_alias():
    candidate = spec()
    candidate["steps"][-1] = {
        "number": 3, "title": "Read", "step_type": "structured_output",
        "config": {"output_schema": {"type": "object"}, "input_mapping": {"data": "$steps.2.result"}},
    }
    report = validate_spec(candidate)
    assert not report.valid
    assert report.errors[0].code == "step_result_without_dependency"


@pytest.mark.parametrize(
    "condition",
    ["value >= $steps.1.result_data.score", "equals:$steps.1.result", "$steps.1 == value"],
)
def test_conditional_reference_requires_dependency(condition):
    candidate = spec()
    step = candidate["steps"][-1]
    step.update(
        step_type="conditional",
        dependencies=[],
        step_config={"branches": [{"condition": condition, "next_step": 2}]},
    )
    report = validate_spec(candidate)
    assert [error.code for error in report.errors] == ["step_result_without_dependency"]
    step["dependencies"] = [2]
    assert validate_spec(candidate).valid


@pytest.mark.parametrize("condition", ['value == "$steps.999.result"', r"matches:^\$steps.999.result$"])
def test_conditional_literals_do_not_require_dependencies(condition):
    candidate = spec()
    candidate["steps"][-1].update(
        step_type="conditional",
        step_config={"branches": [{"condition": condition, "next_step": 2}]},
    )
    assert validate_spec(candidate).valid


def test_codex_reference_alias_and_config_model():
    from mmar_carl import CodexStepConfig

    candidate = spec()
    step = candidate["steps"][-1]
    config = {"task": "Continue", "thread_id_source": "$steps.1.result_data.thread_id"}
    step.update(step_type="codex", dependencies=[])
    for wire_config in (config, CodexStepConfig.model_validate(config)):
        step["step_config"] = wire_config
        report = validate_spec(candidate)
        assert [error.code for error in report.errors] == ["step_result_without_dependency"]


def test_unused_memory_value_source_does_not_require_dependency():
    candidate = spec()
    step = candidate["steps"][-1]
    config = {"operation": "read", "memory_key": "key", "value_source": "$steps.999.result"}
    step.update(step_type="memory", step_config=config)
    assert validate_spec(candidate).valid
    config["operation"] = "write"
    assert validate_spec(candidate).errors[0].code == "step_result_missing_producer"


def test_evaluation_reads_explicit_step_result():
    candidate = spec()
    candidate["steps"][-1].update(
        step_type="evaluation",
        dependencies=[],
        step_config={"evaluates_step": 1, "criteria": ["len(value) > 0"], "input_source": "$steps.1.result"},
    )
    assert validate_spec(candidate).errors[0].code == "step_result_without_dependency"


def test_whole_result_map_remains_a_valid_reference():
    candidate = spec()
    candidate["steps"][-1]["step_config"]["input_mapping"]["data"] = "$steps."
    assert validate_spec(candidate).valid
