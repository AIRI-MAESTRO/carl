"""Dependency validation handles graphs deeper than Python's call stack."""

import pytest

from mmar_carl import LLMStepDescription, ReasoningChain, validate_spec


@pytest.fixture
def deep_steps() -> list[dict]:
    return [
        {
            "number": number,
            "title": f"Step {number}",
            "step_type": "llm",
            "aim": "Validate dependencies",
            "dependencies": [number + 1] if number < 1200 else [],
        }
        for number in range(1, 1201)
    ]


def test_runtime_accepts_deep_dag(deep_steps: list[dict]) -> None:
    deep_steps[0]["dependencies"].append(1200)
    chain = ReasoningChain([LLMStepDescription.model_validate(step) for step in deep_steps])
    assert len(chain.executor.build_execution_graph(chain.steps)) == 1200


def test_spec_accepts_deep_dag(deep_steps: list[dict]) -> None:
    deep_steps[0]["dependencies"].append(1200)
    report = validate_spec(deep_steps)
    assert report.valid, report.format_text()


def test_runtime_reports_deep_dependency_cycle(deep_steps: list[dict]) -> None:
    deep_steps[-1]["dependencies"] = [2]
    steps = [LLMStepDescription.model_validate(step) for step in deep_steps]
    with pytest.raises(ValueError, match="Cycle detected involving step 1"):
        ReasoningChain(steps)


def test_spec_reports_deep_dependency_cycle(deep_steps: list[dict]) -> None:
    deep_steps[-1]["dependencies"] = [2]
    report = validate_spec(deep_steps)
    assert not report.valid
    assert [(issue.code, issue.step) for issue in report.errors] == [("cycle", 2)]
