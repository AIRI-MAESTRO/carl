"""structured_output re-asks once with the schema mismatch, then fails closed."""

from __future__ import annotations

import asyncio

from mmar_carl import ReasoningChain, ReasoningContext


class _ScriptedLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def get_response(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _chain() -> dict:
    return {
        "max_workers": 1,
        "steps": [
            {
                "number": 1,
                "title": "shape",
                "step_type": "structured_output",
                "dependencies": [],
                "step_config": {
                    "output_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                    "input_source": "$outer_context",
                    "schema_name": "shape",
                },
            }
        ],
    }


def _run(answers: list[str]):
    llm = _ScriptedLLM(answers)
    chain = ReasoningChain.from_dict_typed(_chain())
    result = asyncio.run(chain.execute_async(ReasoningContext(outer_context="input", api=llm)))
    return result, llm


def test_schema_mismatch_is_repaired_once() -> None:
    result, llm = _run(['{"y": 1}', '{"x": 3}'])
    step = result.step_results[0]
    assert step.success and step.result_data == {"x": 3}
    assert len(llm.prompts) == 2 and "previous output" in llm.prompts[1]


def test_schema_mismatch_after_retry_fails_closed() -> None:
    result, llm = _run(['{"y": 1}', '{"y": 2}'])
    step = result.step_results[0]
    assert not step.success
    assert "schema" in (step.error_message or "").lower()
    assert len(llm.prompts) == 2


def test_valid_output_needs_one_call() -> None:
    result, llm = _run(['{"x": 1}'])
    assert result.step_results[0].success and len(llm.prompts) == 1
