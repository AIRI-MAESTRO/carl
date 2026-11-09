"""
Tests for StructuredOutputStep per-step llm_config and output validation.

Covers two previously confirmed gaps:

1. ``StructuredOutputStepDescription`` had no ``llm_config`` / ``retry_max``
   fields, so pydantic's default ``extra='ignore'`` silently discarded them —
   even though the executor already honours a per-step ``llm_config`` via
   ``context.get_llm_client_for_step``.

2. ``StructuredOutputStepExecutor`` pasted ``output_schema`` into the prompt
   but never validated the parsed reply against it. With
   ``validate_output=True`` the executor now validates and re-prompts with the
   validation error up to ``repair_retries`` times before failing the step.
   The default (``validate_output=False``) preserves legacy behaviour.
"""

from unittest import mock

import pytest

from mmar_carl import (
    Language,
    LLMStepConfig,
    ReasoningChain,
    ReasoningContext,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
)
from mmar_carl.models.llm_client_base import LLMClientBase


SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "value": {"type": "integer"},
    },
    "required": ["status", "value"],
}

VALID_REPLY = '{"status": "ok", "value": 42}'
MISSING_FIELD_REPLY = '{"value": 42}'


class SequencedMockClient(LLMClientBase):
    """Returns queued responses in order; repeats the last one when exhausted."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def make_chain(config: StructuredOutputStepConfig, **step_kwargs) -> ReasoningChain:
    step = StructuredOutputStepDescription(number=1, title="Extract", config=config, **step_kwargs)
    return ReasoningChain(steps=[step], max_workers=1)


def make_context(client: LLMClientBase) -> ReasoningContext:
    return ReasoningContext(
        outer_context="some input text",
        api=client,
        model="test",
        language=Language.ENGLISH,
    )


# ============================================================================
# Gap 1: llm_config / retry_max on StructuredOutputStepDescription
# ============================================================================


class TestPerStepLLMConfig:
    def test_llm_config_retained_on_description(self):
        step = StructuredOutputStepDescription(
            number=1,
            title="Extract",
            config=StructuredOutputStepConfig(output_schema=SCHEMA),
            llm_config=LLMStepConfig(model="teacher-model"),
        )
        assert step.llm_config is not None
        assert step.llm_config.model == "teacher-model"

    def test_retry_max_retained_on_description(self):
        step = StructuredOutputStepDescription(
            number=1,
            title="Extract",
            config=StructuredOutputStepConfig(output_schema=SCHEMA),
            retry_max=5,
        )
        assert step.retry_max == 5

    @pytest.mark.asyncio
    async def test_llm_config_reaches_client_resolution(self):
        client = SequencedMockClient([VALID_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(output_schema=SCHEMA),
            llm_config=LLMStepConfig(model="teacher-model"),
        )
        context = make_context(client)

        captured: list = []
        real = ReasoningContext.get_llm_client_for_step

        def spy(self, llm_config=None):
            captured.append(llm_config)
            return real(self, llm_config)

        with mock.patch.object(ReasoningContext, "get_llm_client_for_step", spy):
            result = await chain.execute_async(context)

        assert result.success
        assert any(cfg is not None and cfg.model == "teacher-model" for cfg in captured)

    def test_roundtrip_preserves_llm_config(self):
        chain = make_chain(
            StructuredOutputStepConfig(output_schema=SCHEMA),
            llm_config=LLMStepConfig(model="teacher-model"),
            retry_max=4,
        )
        restored = ReasoningChain.from_dict_typed(chain.to_dict())
        step = restored.steps[0]
        assert isinstance(step, StructuredOutputStepDescription)
        assert step.llm_config is not None
        assert step.llm_config.model == "teacher-model"
        assert step.retry_max == 4


# ============================================================================
# Gap 2: output validation + repair re-prompts
# ============================================================================


class TestOutputValidation:
    @pytest.mark.asyncio
    async def test_default_enforces_schema(self):
        """Invalid output is rejected even without an explicit validation flag."""
        client = SequencedMockClient([MISSING_FIELD_REPLY])
        chain = make_chain(StructuredOutputStepConfig(output_schema=SCHEMA))

        result = await chain.execute_async(make_context(client))

        assert not result.success
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_valid_reply_passes_validation(self):
        client = SequencedMockClient([VALID_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(output_schema=SCHEMA, validate_output=True)
        )

        result = await chain.execute_async(make_context(client))

        assert result.success
        assert result.step_results[0].result_data == {"status": "ok", "value": 42}
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_repair_recovers_after_invalid_reply(self):
        client = SequencedMockClient([MISSING_FIELD_REPLY, VALID_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(
                output_schema=SCHEMA, validate_output=True, repair_retries=1
            )
        )

        result = await chain.execute_async(make_context(client))

        assert result.success
        assert result.step_results[0].result_data == {"status": "ok", "value": 42}
        assert client.call_count == 2
        # The repair prompt carries the validation error and the invalid reply
        assert "missing required properties: status" in client.prompts[1]
        assert MISSING_FIELD_REPLY in client.prompts[1]

    @pytest.mark.asyncio
    async def test_repair_recovers_after_unparseable_reply(self):
        client = SequencedMockClient(["this is not json", VALID_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(
                output_schema=SCHEMA, validate_output=True, repair_retries=1
            )
        )

        result = await chain.execute_async(make_context(client))

        assert result.success
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_fails_when_repairs_exhausted(self):
        client = SequencedMockClient([MISSING_FIELD_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(
                output_schema=SCHEMA, validate_output=True, repair_retries=2
            )
        )

        result = await chain.execute_async(make_context(client))

        step_result = result.step_results[0]
        assert not step_result.success
        assert "StructuredOutput" in step_result.error_message
        assert "repair attempt" in step_result.error_message
        assert "missing required properties: status" in step_result.error_message
        assert client.call_count == 3  # 1 initial + 2 repairs

    @pytest.mark.asyncio
    async def test_repair_retries_zero_fails_immediately(self):
        client = SequencedMockClient([MISSING_FIELD_REPLY])
        chain = make_chain(
            StructuredOutputStepConfig(
                output_schema=SCHEMA, validate_output=True, repair_retries=0
            )
        )

        result = await chain.execute_async(make_context(client))

        assert not result.step_results[0].success
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_wrong_type_reported_with_path(self):
        client = SequencedMockClient(['{"status": "ok", "value": "not-an-int"}'])
        chain = make_chain(
            StructuredOutputStepConfig(
                output_schema=SCHEMA, validate_output=True, repair_retries=0
            )
        )

        result = await chain.execute_async(make_context(client))

        step_result = result.step_results[0]
        assert not step_result.success
        assert "$.value" in step_result.error_message

    def test_from_pydantic_model_forwards_validation_flags(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        config = StructuredOutputStepConfig.from_pydantic_model(
            Item, validate_output=True, repair_retries=3
        )
        assert config.validate_output is True
        assert config.repair_retries == 3
