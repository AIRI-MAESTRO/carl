"""Runtime admission tests for structured-output JSON Schema contracts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mmar_carl import (
    LLMClientBase,
    ReasoningContext,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
    StructuredOutputStepExecutor,
)


class _StaticResponseClient(LLMClientBase):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        return self.response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        del retries
        return await self.get_response(prompt)


async def _execute(response: str, schema: dict[str, Any]):
    step = StructuredOutputStepDescription(
        number=1,
        title="Validate output",
        config=StructuredOutputStepConfig(
            input_source="$outer_context",
            output_schema=schema,
        ),
    )
    context = ReasoningContext(
        outer_context="{}",
        api=_StaticResponseClient(response),
    )
    return await StructuredOutputStepExecutor().execute(step, context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "schema", "error_fragment"),
    [
        (
            '"not-an-object"',
            {"type": "object"},
            'expected type "object"',
        ),
        (
            '{"value": {"nested": true}}',
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            "$.value",
        ),
        (
            '{"name": "Ada"}',
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
            "missing required properties: age",
        ),
        (
            '{"name": "Ada", "debug": true}',
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            "additional properties are not allowed",
        ),
        (
            '{"items": [{"score": 1}, {"score": "high"}]}',
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"score": {"type": "number"}},
                            "required": ["score"],
                        },
                    }
                },
                "required": ["items"],
            },
            "$.items[1].score",
        ),
    ],
)
async def test_schema_violation_marks_step_failed(
    response: str,
    schema: dict[str, Any],
    error_fragment: str,
) -> None:
    result = await _execute(response, schema)

    assert result.success is False
    assert result.result == ""
    assert error_fragment in (result.error_message or "")
    assert result.updated_history == []


@pytest.mark.asyncio
async def test_valid_nested_output_succeeds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                    "required": ["score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    result = await _execute('{"items": [{"score": 1}, {"score": 2.5}]}', schema)

    assert result.success is True
    assert result.result_data == {"items": [{"score": 1}, {"score": 2.5}]}
    assert len(result.updated_history) == 1


@pytest.mark.asyncio
async def test_local_defs_reference_succeeds() -> None:
    schema = {
        "$defs": {"score": {"type": "integer", "minimum": 0}},
        "type": "object",
        "properties": {"score": {"$ref": "#/$defs/score"}},
        "required": ["score"],
        "additionalProperties": False,
    }

    result = await _execute('{"score": 7}', schema)

    assert result.success is True
    assert result.result_data == {"score": 7}


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef", "$recursiveRef"])
async def test_remote_reference_is_rejected_before_provider_call(keyword: str) -> None:
    client = _StaticResponseClient("{}")
    step = StructuredOutputStepDescription(
        number=1,
        title="Reject remote reference",
        config=StructuredOutputStepConfig(
            input_source="$outer_context",
            output_schema={keyword: "https://example.invalid/schema.json"},
        ),
    )
    context = ReasoningContext(outer_context="{}", api=client)

    result = await StructuredOutputStepExecutor().execute(step, context)

    assert result.success is False
    assert "Remote JSON Schema references are not allowed" in (result.error_message or "")
    assert "example.invalid" not in (result.error_message or "")
    assert client.calls == 0


@pytest.mark.asyncio
async def test_declared_format_is_enforced() -> None:
    schema = {"type": "string", "format": "email"}

    invalid = await _execute('"not-an-email"', schema)
    valid = await _execute('"ada@example.org"', schema)

    assert invalid.success is False
    assert "expected format 'email'" in (invalid.error_message or "")
    assert valid.success is True


@pytest.mark.asyncio
async def test_validation_error_does_not_echo_rejected_instance() -> None:
    secret = "TOP_SECRET_INSTANCE_CANARY"
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }

    result = await _execute(json.dumps({"count": secret}), schema)

    assert result.success is False
    assert "$.count" in (result.error_message or "")
    assert secret not in (result.error_message or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema",
    [
        {"type": 42},
        {"$schema": "https://example.invalid/unknown-schema", "type": "object"},
    ],
)
async def test_malformed_or_unsupported_schema_fails_closed(schema: dict[str, Any]) -> None:
    client = _StaticResponseClient("{}")
    step = StructuredOutputStepDescription(
        number=1,
        title="Reject invalid schema",
        config=StructuredOutputStepConfig(
            input_source="$outer_context",
            output_schema=schema,
        ),
    )
    context = ReasoningContext(outer_context="{}", api=client)

    result = await StructuredOutputStepExecutor().execute(step, context)

    assert result.success is False
    assert "Schema" in (result.error_message or "")
    assert client.calls == 0
