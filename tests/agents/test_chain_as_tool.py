"""Focused contracts for serialized, isolated chain-as-tool composition."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from mmar_carl import (
    AfterWaitCondition,
    AgentStepConfig,
    AgentStepDescription,
    ChainToolDefinition,
    ChainToolOutcome,
    ChainToolStatus,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
    WaitStepConfig,
    WaitStepDescription,
)


class StubClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "unused"


class ScriptedClient(StubClient):
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def get_response_with_tools_and_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        self.requests.append({"tools": tools, "messages": list(messages or [])})
        return self.responses.pop(0)


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def context(api: LLMClientBase | None = None) -> ReasoningContext:
    return ReasoningContext(outer_context="hello", api=api or StubClient())


def tool_step(
    name: str,
    *,
    input_mapping: dict[str, str] | None = None,
) -> ToolStepDescription:
    return ToolStepDescription(
        number=1,
        title=name,
        config=ToolStepConfig(tool_name=name, input_mapping=input_mapping or {}),
    )


def echo_child() -> ReasoningChain:
    return ReasoningChain(steps=[tool_step("echo", input_mapping={"payload": "$outer_context"})])


def definition(
    *,
    child: ReasoningChain | None = None,
    output_schema: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    timeout_seconds: float = 2.0,
    max_input_bytes: int = 1_000_000,
    max_output_bytes: int = 1_000_000,
) -> ChainToolDefinition:
    return ChainToolDefinition.from_chain(
        name="answer_with_child",
        description="Run the embedded answer chain.",
        chain=child or echo_child(),
        input_schema=INPUT_SCHEMA,
        output_schema=output_schema or OUTPUT_SCHEMA,
        output_reference="$steps.1.result_data",
        allowed_tools=["echo"] if allowed_tools is None else allowed_tools,
        timeout_seconds=timeout_seconds,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )


def parent(spec: ChainToolDefinition) -> ReasoningChain:
    return ReasoningChain(
        steps=[tool_step(spec.name, input_mapping={"text": "$outer_context"})],
        chain_tools=[spec],
    )


def echo(payload: dict[str, Any]) -> dict[str, str]:
    return {"answer": payload["text"]}


def wait_child() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            WaitStepDescription(
                number=1,
                title="wait",
                config=WaitStepConfig(condition=AfterWaitCondition(seconds=5)),
            )
        ]
    )


def test_contracts_are_pydantic_and_require_object_input() -> None:
    assert not dataclasses.is_dataclass(ChainToolDefinition)
    assert not dataclasses.is_dataclass(ChainToolOutcome)
    with pytest.raises(ValidationError, match="type='object'"):
        ChainToolDefinition.from_chain(
            name="bad",
            description="bad",
            chain=echo_child(),
            input_schema={"type": "string"},
            output_schema={},
            output_reference="$history[-1]",
        )
    with pytest.raises(ValidationError, match="unsupported schema keyword 'minimum'"):
        definition(output_schema={"type": "integer", "minimum": 0})


def test_snapshot_is_copied_hashed_serialized_and_migrated() -> None:
    raw = echo_child().to_dict()
    spec = ChainToolDefinition(
        name="answer_with_child",
        description="Run child.",
        chain_snapshot=raw,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        output_reference="$steps.1.result_data",
        allowed_tools=["echo"],
    )
    raw["steps"].clear()
    restored = ReasoningChain.from_dict_typed(parent(spec).to_dict())

    assert len(spec.chain_snapshot["steps"]) == 1
    assert len(spec.snapshot_sha256) == 64
    assert restored.chain_tools == [spec]
    assert restored.to_dict()["format_version"] == 10

    legacy = {"format_version": 6, "steps": echo_child().to_dict()["steps"]}
    migrated = ReasoningChain.migrate(legacy)
    assert "chain_tools" not in legacy
    assert migrated["format_version"] == 10
    assert migrated["chain_tools"] == []


async def test_mutated_snapshot_fails_digest_check() -> None:
    spec = definition()
    ctx = context()
    ctx.register_tool("echo", echo)
    runtime = ctx.register_chain_tool(spec)
    spec.chain_snapshot["steps"].clear()

    outcome = await runtime.invoke(text="hello")

    assert outcome.status is ChainToolStatus.UNAVAILABLE
    assert outcome.error_code == "snapshot_digest_mismatch"


def test_construction_requires_explicit_nested_capabilities() -> None:
    with pytest.raises(ValueError, match="undeclared host tools: echo"):
        ReasoningChain(
            steps=[tool_step("answer_with_child")],
            chain_tools=[definition(allowed_tools=[])],
        )


async def test_auto_registered_tool_returns_typed_provenance() -> None:
    spec = definition()
    chain = parent(spec)
    ctx = context()
    ctx.memory_write("private", "parent-only", namespace="secret")
    ctx.register_tool("echo", echo)

    result = await chain.execute_async(ctx)

    outcome = ChainToolOutcome.model_validate(result.step_results[0].result_data)
    assert result.success and outcome.success
    assert outcome.status is ChainToolStatus.COMPLETED
    assert outcome.output == {"answer": "hello"}
    assert outcome.snapshot_sha256 == spec.snapshot_sha256
    assert len(outcome.input_sha256 or "") == 64
    assert len(outcome.output_sha256 or "") == 64
    assert ctx.memory_read("private", namespace="secret") == "parent-only"

    second = await chain.execute_async(ctx)
    assert second.success
    assert ctx.get_tool(spec.name) is not None


async def test_invalid_input_and_output_are_explicit() -> None:
    calls = 0

    def wrong(payload: dict[str, Any]) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"wrong": payload["text"]}

    ctx = context()
    ctx.register_tool("echo", wrong)
    runtime = ctx.register_chain_tool(definition())

    bad_input = await runtime.invoke(text=7)
    assert bad_input.status is ChainToolStatus.INVALID_INPUT
    assert calls == 0

    bad_output = await runtime.invoke(text="hello")
    assert bad_output.status is ChainToolStatus.INVALID_OUTPUT
    assert bad_output.output is None
    assert calls == 1


async def test_output_byte_limit_is_enforced() -> None:
    input_context = context()
    input_context.register_tool("echo", echo)
    input_runtime = input_context.register_chain_tool(definition(max_input_bytes=15))

    input_outcome = await input_runtime.invoke(text="hello")

    assert input_outcome.status is ChainToolStatus.INVALID_INPUT
    assert input_outcome.error_code == "input_too_large"

    ctx = context()
    ctx.register_tool("echo", lambda payload: {"answer": payload["text"] * 100})
    runtime = ctx.register_chain_tool(definition(max_output_bytes=32))

    outcome = await runtime.invoke(text="hello")

    assert outcome.status is ChainToolStatus.INVALID_OUTPUT
    assert outcome.error_code == "output_too_large"
    assert "output exceeds 32 bytes" in (outcome.error_message or "")


async def test_nested_failure_is_explicit() -> None:
    def fail(payload: dict[str, Any]) -> None:
        raise RuntimeError(f"cannot echo {payload['text']}")

    ctx = context()
    ctx.register_tool("echo", fail)
    outcome = await ctx.register_chain_tool(definition()).invoke(text="hello")

    assert outcome.status is ChainToolStatus.FAILED
    assert outcome.error_code == "nested_chain_failed"
    assert "cannot echo hello" in (outcome.error_message or "")


async def test_parent_cancellation_stops_owned_child() -> None:
    ctx = context()
    runtime = ctx.register_chain_tool(
        definition(
            child=wait_child(),
            output_schema={"type": "object"},
            allowed_tools=[],
        )
    )
    task = asyncio.create_task(runtime.invoke(text="hello"))
    await asyncio.sleep(0.08)
    ctx.cancel()

    outcome = await asyncio.wait_for(task, timeout=1)
    assert outcome.status is ChainToolStatus.CANCELLED


async def test_timeout_stops_owned_child() -> None:
    ctx = context()
    runtime = ctx.register_chain_tool(
        definition(
            child=wait_child(),
            output_schema={"type": "object"},
            allowed_tools=[],
            timeout_seconds=0.02,
        )
    )

    outcome = await asyncio.wait_for(runtime.invoke(text="hello"), timeout=1)
    assert outcome.status is ChainToolStatus.TIMED_OUT


async def test_recursive_reentry_is_bounded() -> None:
    child = ReasoningChain(steps=[tool_step("reenter", input_mapping={"payload": "$outer_context"})])
    spec = definition(
        child=child,
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
        allowed_tools=["reenter"],
    )
    ctx = context()
    holder: dict[str, Any] = {}

    async def reenter(payload: dict[str, Any]) -> dict[str, Any]:
        return await holder["runtime"](**payload)

    ctx.register_tool("reenter", reenter)
    holder["runtime"] = ctx.register_chain_tool(spec)
    outcome = await holder["runtime"].invoke(text="hello")

    assert outcome.status is ChainToolStatus.COMPLETED
    assert outcome.output["status"] == ChainToolStatus.RECURSION_LIMIT
    assert outcome.output["success"] is False


def test_preflight_reports_host_capability_not_embedded_name() -> None:
    report = parent(definition()).preflight(context())
    assert report.required_tools == ["echo"]
    assert report.missing_tools == ["echo"]


async def test_missing_capability_and_name_collision_fail_before_steps() -> None:
    calls = 0

    def side_effect() -> str:
        nonlocal calls
        calls += 1
        return "ran"

    chain = ReasoningChain(
        steps=[tool_step("side_effect")],
        chain_tools=[definition()],
    )
    ctx = context()
    ctx.register_tool("side_effect", side_effect)
    with pytest.raises(ValueError, match="missing host tools: echo"):
        await chain.execute_async(ctx)
    assert calls == 0

    ctx.register_tool("echo", echo)
    def original(text):
        return text

    ctx.register_tool("answer_with_child", original)
    with pytest.raises(ValueError, match="already registered"):
        await chain.execute_async(ctx)
    assert ctx.get_tool("answer_with_child") is original

    contract_context = context()
    contract_context.register_tool("echo", echo)
    contract_context.register_chain_tool(definition())
    with pytest.raises(ValueError, match="already registered"):
        contract_context.register_chain_tool(definition(output_schema={"type": "object"}))


async def test_agent_uses_declared_schema_and_rejects_bad_arguments() -> None:
    client = ScriptedClient(
        [
            (
                "",
                [{"id": "bad", "name": "answer_with_child", "arguments": {"text": 7}}],
                {},
            ),
            (
                "",
                [{"id": "ok", "name": "answer_with_child", "arguments": {"text": "hello"}}],
                {},
            ),
            (
                "",
                [{"id": "finish", "name": "finish", "arguments": {"result": "done"}}],
                {},
            ),
        ]
    )
    chain = ReasoningChain(
        steps=[
            AgentStepDescription(
                number=1,
                title="agent",
                config=AgentStepConfig(
                    goal="Use the child chain.",
                    tools=["answer_with_child"],
                    max_iterations=3,
                    timeout_seconds=2,
                    model_timeout_seconds=1,
                    tool_timeout_seconds=1,
                ),
            )
        ],
        chain_tools=[definition()],
    )
    ctx = context(client)
    ctx.register_tool("echo", echo)

    result = await chain.execute_async(ctx)

    declared = next(
        tool["function"] for tool in client.requests[0]["tools"] if tool["function"]["name"] == "answer_with_child"
    )
    transcript = result.step_results[0].result_data["transcript"]
    assert result.success
    assert declared["parameters"] == INPUT_SCHEMA
    assert any(event.get("error", {}).get("code") == "invalid_arguments" for event in transcript)
    assert any(
        event.get("kind") == "tool_observation"
        and isinstance(event.get("value"), dict)
        and event["value"]["status"] == ChainToolStatus.COMPLETED
        for event in transcript
    )
