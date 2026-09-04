"""Focused contract tests for the bounded AgentStep ReAct loop."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from mmar_carl import (
    AgentStepConfig,
    AgentStepDescription,
    AgentStepExecutor,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepType,
)
from mmar_carl.models.llm_client_base import LLMClientBase


def _call(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": arguments}


class ScriptedClient(LLMClientBase):
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "scripted-agent-model"

    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "unused"

    async def get_response_with_tools_and_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        self.requests.append({"tools": tools, "messages": list(messages or [])})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return await response()
        return response


def _context(client: LLMClientBase) -> ReasoningContext:
    return ReasoningContext(
        outer_context="unused unless explicitly mapped",
        api=client,
        model="test",
        language=Language.ENGLISH,
    )


def _step(**overrides: Any) -> AgentStepDescription:
    config_data: dict[str, Any] = {
        "goal": "Find the answer and return it.",
        "tools": ["lookup"],
        "max_iterations": 4,
        "max_tool_calls": 3,
        "timeout_seconds": 2,
        "model_timeout_seconds": 1,
        "tool_timeout_seconds": 1,
    }
    config_data.update(overrides)
    return AgentStepDescription(
        number=1,
        title="Research",
        config=AgentStepConfig(**config_data),
    )


async def _lookup(query: str) -> dict[str, str]:
    """Look up one query."""
    return {"answer": f"found:{query}"}


class TestAgentStepConfig:
    @pytest.mark.parametrize("tools", [[], [""], ["lookup", "lookup"], ["finish"]])
    def test_requires_explicit_valid_tool_allowlist(self, tools: list[str]) -> None:
        with pytest.raises(ValidationError):
            AgentStepConfig(goal="g", tools=tools)

    @pytest.mark.parametrize(
        "input_mapping",
        [{"": "$outer_context"}, {"question": "literal text"}],
    )
    def test_input_mapping_requires_named_context_references(
        self, input_mapping: dict[str, str]
    ) -> None:
        with pytest.raises(ValidationError):
            AgentStepConfig(goal="g", tools=["lookup"], input_mapping=input_mapping)

    def test_public_type_and_json_round_trip(self) -> None:
        step = _step(
            input_mapping={"question": "$outer_context"},
            output_schema={"type": "object", "required": ["answer"]},
            output_key="answer",
        )
        chain = ReasoningChain(steps=[step])
        payload = chain.to_dict()

        assert payload["steps"][0]["step_type"] == "agent"
        restored = ReasoningChain.from_dict_typed(payload)
        restored_step = restored.steps[0]
        assert isinstance(restored_step, AgentStepDescription)
        assert restored_step.step_type is StepType.AGENT
        assert restored_step.config == step.config

        legacy = ReasoningChain.from_dict(payload)
        assert legacy.steps[0].step_type is StepType.AGENT
        assert isinstance(legacy.steps[0].step_config, AgentStepConfig)

        report = restored.preflight(_context(ScriptedClient([])))
        assert report.required_tools == ["lookup"]
        assert report.missing_tools == ["lookup"]

    async def test_saved_chain_executes_after_typed_load(self) -> None:
        original = ReasoningChain(steps=[_step(output_key="answer")])
        restored = ReasoningChain.from_dict_typed(original.to_dict())
        client = ScriptedClient([
            ("", [_call("finish", {"result": {"answer": "ok"}})], {})
        ])
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await restored.execute_async(context)

        assert result.success is True
        assert result.step_results[0].step_type is StepType.AGENT
        assert result.step_results[0].result_data["outcome"] == "completed"
        assert context.memory_read("answer", namespace="agent") == {"answer": "ok"}


class TestAgentStepLoop:
    async def test_tool_observation_then_explicit_finish(self) -> None:
        client = ScriptedClient([
            ("", [_call("lookup", {"query": "carl"})], {"prompt": 4, "completion": 2, "total": 6}),
            ("", [_call("finish", {"result": {"answer": "found:carl"}}, "call-2")],
             {"prompt": 5, "completion": 1, "total": 6}),
        ])
        context = _context(client)
        context.register_tool("lookup", _lookup)
        step = _step(
            output_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
            output_key="research_result",
        )

        result = await AgentStepExecutor().execute(step, context)

        assert result.success is True
        assert result.result_data["outcome"] == "completed"
        assert result.result_data["output"] == {"answer": "found:carl"}
        assert result.result_data["iterations"] == 2
        assert result.result_data["tool_calls"] == 1
        assert result.token_usage == {"prompt": 9, "completion": 3, "total": 12}
        assert result.model == "scripted-agent-model"
        assert context.memory_read("research_result", namespace="agent") == {
            "answer": "found:carl"
        }
        assert [message["role"] for message in client.requests[0]["messages"]] == [
            "system",
            "user",
        ]
        second_history = client.requests[1]["messages"]
        tool_message = next(message for message in second_history if message["role"] == "tool")
        assert "UNTRUSTED TOOL OUTPUT" in tool_message["content"]
        assert "found:carl" in tool_message["content"]

    async def test_text_only_response_is_corrected_not_completed(self) -> None:
        client = ScriptedClient([
            ("I think I am done", [], {}),
            ("", [_call("finish", {"result": "done"}, "call-2")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is True
        assert result.result == "done"
        assert any(
            "missing_call" in str(message.get("content"))
            for message in client.requests[1]["messages"]
        )

    async def test_multiple_calls_execute_nothing(self) -> None:
        executed: list[str] = []

        async def lookup(query: str) -> str:
            executed.append(query)
            return query

        client = ScriptedClient([
            ("", [
                _call("lookup", {"query": "a"}, "a"),
                _call("lookup", {"query": "b"}, "b"),
            ], {}),
            ("", [_call("finish", {"result": "no side effects"}, "finish")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", lookup)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is True
        assert executed == []
        assert result.result_data["tool_calls"] == 0
        assert any(
            event.get("kind") == "protocol_error"
            and event.get("error", {}).get("code") == "multiple_calls"
            for event in result.result_data["transcript"]
        )

    async def test_invalid_arguments_execute_nothing_and_can_recover(self) -> None:
        executed = False

        async def lookup(query: str) -> str:
            nonlocal executed
            executed = True
            return query

        client = ScriptedClient([
            ("", [_call("lookup", {"query": 42})], {}),
            ("", [_call("finish", {"result": "recovered"}, "finish")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", lookup)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is True
        assert executed is False
        assert result.result_data["tool_calls"] == 0
        assert any(
            event.get("error", {}).get("code") == "invalid_arguments"
            for event in result.result_data["transcript"]
        )

    async def test_non_json_arguments_execute_nothing(self) -> None:
        executed = False

        async def lookup(query: Any) -> str:
            nonlocal executed
            executed = True
            return str(query)

        client = ScriptedClient([
            ("", [_call("lookup", {"query": float("nan")})], {}),
            ("", [_call("finish", {"result": "recovered"}, "finish")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", lookup)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is True
        assert executed is False
        assert result.result_data["tool_calls"] == 0
        assert any(
            event.get("error", {}).get("code") == "invalid_arguments"
            for event in result.result_data["transcript"]
        )

    async def test_missing_tool_fails_before_model_call(self) -> None:
        client = ScriptedClient([])

        result = await AgentStepExecutor().execute(_step(), _context(client))

        assert result.success is False
        assert result.result_data["outcome"] == "failed"
        assert "not registered" in result.result_data["stop_reason"]
        assert client.requests == []

    async def test_unsupported_tool_signature_fails_before_model_call(self) -> None:
        async def variadic(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = ScriptedClient([])
        context = _context(client)
        context.register_tool("lookup", variadic)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is False
        assert "unsupported parameter" in result.result_data["stop_reason"]
        assert client.requests == []

    async def test_invalid_finish_is_observed_until_valid_finish(self) -> None:
        client = ScriptedClient([
            ("", [_call("finish", {"result": {"wrong": 1}})], {}),
            ("", [_call("finish", {"result": {"answer": "ok"}}, "call-2")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", _lookup)
        step = _step(output_schema={"type": "object", "required": ["answer"]})

        result = await AgentStepExecutor().execute(step, context)

        assert result.success is True
        assert result.result_data["iterations"] == 2
        assert any(
            event.get("kind") == "invalid_finish"
            for event in result.result_data["transcript"]
        )

    async def test_iteration_exhaustion_is_incomplete_not_success(self) -> None:
        client = ScriptedClient([("text only", [], {})])
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await AgentStepExecutor().execute(_step(max_iterations=1), context)

        assert result.success is False
        assert result.result_data["outcome"] == "incomplete"
        assert result.result_data["stop_reason"] == "max_iterations"
        assert result.result == ""

    async def test_tool_call_budget_stops_before_side_effect(self) -> None:
        executed = False

        async def lookup(query: str) -> str:
            nonlocal executed
            executed = True
            return query

        client = ScriptedClient([("", [_call("lookup", {"query": "x"})], {})])
        context = _context(client)
        context.register_tool("lookup", lookup)

        result = await AgentStepExecutor().execute(_step(max_tool_calls=0), context)

        assert result.success is False
        assert result.result_data["outcome"] == "budget_exhausted"
        assert result.result_data["stop_reason"] == "tool_call_limit"
        assert executed is False

    async def test_reported_token_budget_stops_before_tool(self) -> None:
        client = ScriptedClient([
            ("", [_call("lookup", {"query": "x"})], {"prompt": 4, "completion": 1, "total": 5})
        ])
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await AgentStepExecutor().execute(_step(max_tokens=5), context)

        assert result.success is False
        assert result.result_data["outcome"] == "budget_exhausted"
        assert result.result_data["stop_reason"] == "token_limit"
        assert result.result_data["tool_calls"] == 0

    async def test_tool_timeout_is_typed_observation_and_can_recover(self) -> None:
        async def slow(query: str) -> str:
            await asyncio.sleep(1)
            return query

        client = ScriptedClient([
            ("", [_call("lookup", {"query": "x"})], {}),
            ("", [_call("finish", {"result": "fallback"}, "call-2")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", slow)

        result = await AgentStepExecutor().execute(
            _step(tool_timeout_seconds=0.01), context
        )

        assert result.success is True
        timeout_observation = next(
            event for event in result.result_data["transcript"]
            if event.get("kind") == "tool_observation"
        )
        assert timeout_observation["error"]["type"] == "timeout"

    async def test_large_tool_result_is_bounded_in_message_and_trace(self) -> None:
        async def large_lookup(query: str) -> str:
            return query * 1000

        client = ScriptedClient([
            ("", [_call("lookup", {"query": "x"})], {}),
            ("", [_call("finish", {"result": "bounded"}, "call-2")], {}),
        ])
        context = _context(client)
        context.register_tool("lookup", large_lookup)

        result = await AgentStepExecutor().execute(
            _step(max_tool_result_chars=256), context
        )

        assert result.success is True
        observation = next(
            event for event in result.result_data["transcript"]
            if event.get("kind") == "tool_observation"
        )
        assert observation["value"]["truncated"] is True
        observation_payload = {
            key: value
            for key, value in observation.items()
            if key not in {"iteration", "kind"}
        }
        assert len(json.dumps(observation_payload, ensure_ascii=False)) <= 256
        tool_message = next(
            message for message in client.requests[1]["messages"]
            if message["role"] == "tool"
        )
        assert len(tool_message["content"]) <= 256

    async def test_diagnostic_trace_respects_aggregate_limit(self) -> None:
        client = ScriptedClient([("x" * 2000, [], {})] * 4)
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await AgentStepExecutor().execute(
            _step(
                max_iterations=4,
                max_tool_result_chars=256,
                max_transcript_chars=2048,
            ),
            context,
        )

        assert result.success is False
        assert len(json.dumps(result.result_data["transcript"], ensure_ascii=False)) <= 2048

    async def test_host_cancellation_interrupts_model_wait(self) -> None:
        async def delayed_response() -> tuple[str, list[dict[str, Any]], dict[str, int]]:
            await asyncio.sleep(2)
            return "", [_call("finish", {"result": "late"})], {}

        client = ScriptedClient([delayed_response])
        context = _context(client)
        context.register_tool("lookup", _lookup)
        task = asyncio.create_task(AgentStepExecutor().execute(_step(), context))
        await asyncio.sleep(0.08)
        context.cancel()

        result = await task

        assert result.success is False
        assert result.skipped is True
        assert result.result_data["outcome"] == "cancelled"
        assert result.result_data["stop_reason"] == "cancelled_by_host"

    async def test_provider_failure_is_failed(self) -> None:
        client = ScriptedClient([RuntimeError("provider unavailable")])
        context = _context(client)
        context.register_tool("lookup", _lookup)

        result = await AgentStepExecutor().execute(_step(), context)

        assert result.success is False
        assert result.result_data["outcome"] == "failed"
        assert "provider unavailable" in result.result_data["stop_reason"]
