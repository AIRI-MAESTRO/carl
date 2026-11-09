import json
from types import SimpleNamespace

import pytest

from mmar_carl import ReasoningContext, StructuredOutputStepConfig, StructuredOutputStepDescription
from mmar_carl.step_executors import StructuredOutputStepExecutor, resolve_context_reference


@pytest.mark.parametrize("outer", [{"nested": {"text": "CANARY"}}, '{"nested":{"text":"CANARY"}}'])
def test_nested_outer_context(outer):
    context = SimpleNamespace(outer_context=outer, runtime_input=None)
    assert resolve_context_reference("$outer_context.nested.text", context) == "CANARY"


def test_json_literal_round_trip():
    value = 'line\n"quoted"\\path'
    assert resolve_context_reference(json.dumps(value), SimpleNamespace()) == value


def test_conflicting_structured_sources_rejected():
    with pytest.raises(ValueError, match="cannot be combined"):
        StructuredOutputStepConfig(input_mapping={"x": "$outer_context.x"},
                                   input_source="$outer_context", output_schema={"type": "object"})


@pytest.mark.asyncio
async def test_structured_mapping_only_exposes_selected_values():
    prompts = []

    class API:
        async def get_response_with_retries(self, prompt, **kwargs):
            prompts.append(prompt)
            return '{"answer":"ok"}'

    context = ReasoningContext(
        outer_context='{"selected":"INPUT_CANARY","private":"PRIVATE_CANARY"}',
        api=API(), retry_max=0,
    )
    step = StructuredOutputStepDescription(number=1, title="Extract", config=StructuredOutputStepConfig(
        input_mapping={"selected": "$outer_context.selected"},
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    ))
    result = await StructuredOutputStepExecutor().execute(step, context)
    assert result.success, result
    assert prompts and "INPUT_CANARY" in prompts[0]
    assert "PRIVATE_CANARY" not in prompts[0]
