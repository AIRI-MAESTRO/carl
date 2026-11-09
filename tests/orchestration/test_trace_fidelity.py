"""
Tests for per-agent attribution fidelity of ExecutionTrace.

Covers the data a trace must carry to be usable as a research dataset:

- ``TraceEvent.inputs`` is populated from the step's resolved input_mapping
- ``TraceEvent.model`` records which LLM actually served the step
- ``TraceEvent.prompt`` is opt-in via ``ReasoningContext.trace_prompts``
- ``sensitive=True`` human input never reaches the trace, directly or via a
  downstream step's captured inputs
- ``started_at`` / ``ended_at`` are measured, and ``_compute_step_intervals``
  falls back to the synthesised batch layout for traces without them
- ``TraceEvent.sub_trace`` preserves nested sub-chain runs across to_json /
  from_json
- Backward compatibility: pre-change trace JSON still deserialises
- ``diff`` / ``TraceAggregator`` / ``format_gantt`` / ``to_html`` keep working
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from mmar_carl import (
    AgentHandoffStepConfig,
    AgentHandoffStepDescription,
    ExecutionTrace,
    HumanInputResponse,
    HumanInputStepConfig,
    HumanInputStepDescription,
    Language,
    LLMClientBase,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    TraceEvent,
)
from mmar_carl.execution_trace import TRACE_TEXT_LIMIT, TraceAggregator
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.enums import StepType
from mmar_carl.models.steps import ToolStepDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockClient(LLMClientBase):
    """Minimal deterministic client with an optional advertised model."""

    def __init__(self, response: str = "ok", model: str | None = None):
        self._response = response
        self._model = model

    @property
    def model_name(self) -> str | None:
        return self._model

    async def get_response(self, prompt: str) -> str:
        return self._response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return self._response


def _ctx(client: LLMClientBase | None = None, **kwargs) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test",
        api=client or _MockClient(),
        model="test",
        language=Language.ENGLISH,
        **kwargs,
    )


def _llm_step(number: int, deps: list[int] | None = None) -> LLMStepDescription:
    return LLMStepDescription(
        number=number,
        title=f"Step {number}",
        aim="Think about it.",
        dependencies=deps or [],
    )


# A trace exactly as CARL serialised it before per-agent attribution existed:
# no model / prompt / started_at / ended_at / sub_trace keys anywhere.
_LEGACY_TRACE_JSON = json.dumps(
    {
        "chain_title": "LegacyChain",
        "executed_at": "2024-01-01T00:00:00+00:00",
        "total_execution_time": 1.5,
        "success": True,
        "metadata": {"parallel_batches": 2},
        "events": [
            {
                "step_number": 1,
                "step_title": "Step 1",
                "step_type": "llm",
                "success": True,
                "skipped": False,
                "result": "legacy result",
                "result_data": None,
                "error_message": None,
                "execution_time": 1.0,
                "token_usage": {"prompt": 7, "completion": 3, "total": 10},
                "batch_index": 0,
                "inputs": {},
            },
            {
                "step_number": 2,
                "step_title": "Step 2",
                "step_type": "tool",
                "success": False,
                "skipped": False,
                "result": "",
                "result_data": None,
                "error_message": "boom",
                "execution_time": 0.5,
                "token_usage": {},
                "batch_index": 1,
                "inputs": {"x": 1},
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_legacy_trace_json_still_loads(self) -> None:
        trace = ExecutionTrace.from_json(_LEGACY_TRACE_JSON)
        assert trace.chain_title == "LegacyChain"
        assert trace.total_execution_time == 1.5
        assert trace.success is True
        assert trace.metadata == {"parallel_batches": 2}
        assert [e.step_number for e in trace.events] == [1, 2]
        assert trace.events[0].result == "legacy result"
        assert trace.events[1].error_message == "boom"
        assert trace.events[1].inputs == {"x": 1}

    def test_legacy_trace_defaults_new_fields(self) -> None:
        trace = ExecutionTrace.from_json(_LEGACY_TRACE_JSON)
        for event in trace.events:
            assert event.model is None
            assert event.prompt is None
            assert event.started_at is None
            assert event.ended_at is None
            assert event.sub_trace is None

    def test_legacy_trace_still_diffs_and_aggregates(self) -> None:
        trace = ExecutionTrace.from_json(_LEGACY_TRACE_JSON)
        assert trace.diff(trace)["unchanged"] == [1, 2]
        assert TraceAggregator([trace]).latency_ms[1]["p50"] == 1000.0

    def test_new_trace_round_trips(self) -> None:
        trace = ExecutionTrace(
            chain_title="C",
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="s",
                    success=True,
                    model="acme/model-1",
                    prompt="rendered prompt",
                    started_at="2026-01-01T00:00:00+00:00",
                    ended_at="2026-01-01T00:00:02+00:00",
                    inputs={"a": 1},
                )
            ],
        )
        restored = ExecutionTrace.from_json(trace.to_json())
        event = restored.events[0]
        assert event.model == "acme/model-1"
        assert event.prompt == "rendered prompt"
        assert event.started_at == "2026-01-01T00:00:00+00:00"
        assert event.ended_at == "2026-01-01T00:00:02+00:00"
        assert event.inputs == {"a": 1}


# ---------------------------------------------------------------------------
# model attribution
# ---------------------------------------------------------------------------


class TestModelAttribution:
    @pytest.mark.asyncio
    async def test_model_recorded_on_event(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(_MockClient(model="acme/model-1")))
        assert result.trace.get_event(1).model == "acme/model-1"

    @pytest.mark.asyncio
    async def test_model_none_when_client_does_not_expose_one(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(_MockClient()))
        assert result.trace.get_event(1).model is None

    @pytest.mark.asyncio
    async def test_model_survives_serialisation(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(_MockClient(model="acme/model-2")))
        restored = ExecutionTrace.from_json(result.trace.to_json())
        assert restored.get_event(1).model == "acme/model-2"


# ---------------------------------------------------------------------------
# prompt capture (opt-in)
# ---------------------------------------------------------------------------


class TestPromptCapture:
    @pytest.mark.asyncio
    async def test_prompt_not_captured_by_default(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        ctx = _ctx()
        assert ctx.trace_prompts is False
        result = await chain.execute_async(ctx)
        assert result.trace.get_event(1).prompt is None
        assert result.step_results[0].prompt is None

    @pytest.mark.asyncio
    async def test_prompt_captured_when_opted_in(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(trace_prompts=True))
        prompt = result.trace.get_event(1).prompt
        assert prompt is not None
        assert "Think about it." in prompt

    @pytest.mark.asyncio
    async def test_prompt_truncated_to_trace_text_limit(self) -> None:
        step = LLMStepDescription(number=1, title="S", aim="x" * (TRACE_TEXT_LIMIT * 2))
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(_ctx(trace_prompts=True))
        assert len(result.trace.get_event(1).prompt) == TRACE_TEXT_LIMIT

    @pytest.mark.asyncio
    async def test_prompt_round_trips(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(trace_prompts=True))
        restored = ExecutionTrace.from_json(result.trace.to_json())
        assert restored.get_event(1).prompt == result.trace.get_event(1).prompt


# ---------------------------------------------------------------------------
# inputs capture
# ---------------------------------------------------------------------------


class TestInputCapture:
    @pytest.mark.asyncio
    async def test_tool_step_inputs_are_recorded(self) -> None:
        ctx = _ctx()
        ctx.register_tool("double", lambda value: value * 2)
        ctx.memory_write("value", 21, namespace="scratch")
        step = ToolStepDescription(
            number=1,
            title="Double",
            config=ToolStepConfig(
                tool_name="double",
                input_mapping={"value": "$memory.scratch.value"},
            ),
        )
        result = await ReasoningChain(steps=[step]).execute_async(ctx)
        assert result.trace.get_event(1).inputs == {"value": 21}

    @pytest.mark.asyncio
    async def test_inputs_capture_reflects_pre_execution_state(self) -> None:
        """A step reading a prior step's output records the value it consumed."""
        ctx = _ctx()
        ctx.register_tool("echo", lambda value: value)
        ctx.memory_write("seed", "first", namespace="scratch")
        steps = [
            ToolStepDescription(
                number=1,
                title="One",
                config=ToolStepConfig(tool_name="echo", input_mapping={"value": "$memory.scratch.seed"}),
            ),
            ToolStepDescription(
                number=2,
                title="Two",
                dependencies=[1],
                config=ToolStepConfig(tool_name="echo", input_mapping={"value": "$history[-1]"}),
            ),
        ]
        result = await ReasoningChain(steps=steps).execute_async(ctx)
        assert result.trace.get_event(1).inputs == {"value": "first"}
        assert "first" in result.trace.get_event(2).inputs["value"]

    @pytest.mark.asyncio
    async def test_step_without_input_mapping_records_empty_inputs(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx())
        assert result.trace.get_event(1).inputs == {}

    @pytest.mark.asyncio
    async def test_unresolvable_reference_does_not_break_the_run(self) -> None:
        ctx = _ctx()
        ctx.register_tool("echo", lambda value: str(value))
        step = ToolStepDescription(
            number=1,
            title="Echo",
            config=ToolStepConfig(
                tool_name="echo",
                input_mapping={"value": "$memory.nope.missing"},
            ),
        )
        result = await ReasoningChain(steps=[step]).execute_async(ctx)
        assert result.success is True
        assert result.trace.get_event(1).inputs == {"value": None}


# ---------------------------------------------------------------------------
# sensitive human input must never reach the trace
# ---------------------------------------------------------------------------


def _human_input_step(number: int, *, sensitive: bool) -> HumanInputStepDescription:
    return HumanInputStepDescription(
        number=number,
        title="Ask",
        config=HumanInputStepConfig(
            prompt="Secret?",
            sensitive=sensitive,
            output_memory_key="token",
        ),
    )


def _consumer_step(number: int, deps: list[int]) -> ToolStepDescription:
    return ToolStepDescription(
        number=number,
        title="Consume",
        dependencies=deps,
        config=ToolStepConfig(
            tool_name="echo",
            input_mapping={"value": "$memory.human_input.token"},
        ),
    )


class TestSensitiveRedaction:
    SECRET = "hunter2-super-secret"

    def _ctx_with_provider(self) -> ReasoningContext:
        ctx = _ctx()
        ctx.register_tool("echo", lambda value: "consumed")
        ctx.on_human_input_requested = lambda request: HumanInputResponse(
            request_id=request.request_id, value=self.SECRET, actor_id="tester"
        )
        return ctx

    @pytest.mark.asyncio
    async def test_sensitive_value_absent_from_serialised_trace(self) -> None:
        chain = ReasoningChain(steps=[_human_input_step(1, sensitive=True), _consumer_step(2, [1])])
        result = await chain.execute_async(self._ctx_with_provider())
        assert self.SECRET not in result.trace.to_json()

    @pytest.mark.asyncio
    async def test_downstream_inputs_are_redacted(self) -> None:
        chain = ReasoningChain(steps=[_human_input_step(1, sensitive=True), _consumer_step(2, [1])])
        result = await chain.execute_async(self._ctx_with_provider())
        assert result.trace.get_event(2).inputs == {"value": "[redacted]"}

    @pytest.mark.asyncio
    async def test_sensitive_step_records_no_inputs_or_prompt(self) -> None:
        chain = ReasoningChain(steps=[_human_input_step(1, sensitive=True)])
        ctx = self._ctx_with_provider()
        ctx.trace_prompts = True
        result = await chain.execute_async(ctx)
        event = result.trace.get_event(1)
        assert event.inputs == {}
        assert event.prompt is None

    @pytest.mark.asyncio
    async def test_non_sensitive_value_is_still_captured(self) -> None:
        """Redaction is targeted — it must not blank ordinary human input."""
        chain = ReasoningChain(steps=[_human_input_step(1, sensitive=False), _consumer_step(2, [1])])
        result = await chain.execute_async(self._ctx_with_provider())
        assert result.trace.get_event(2).inputs == {"value": self.SECRET}


# ---------------------------------------------------------------------------
# measured timestamps + synthesised fallback
# ---------------------------------------------------------------------------


class TestMeasuredTimestamps:
    @pytest.mark.asyncio
    async def test_events_carry_measured_utc_timestamps(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1), _llm_step(2, deps=[1])])
        result = await chain.execute_async(_ctx())
        for event in result.trace.events:
            started = datetime.fromisoformat(event.started_at)
            ended = datetime.fromisoformat(event.ended_at)
            assert started.tzinfo is not None
            assert ended >= started

    @pytest.mark.asyncio
    async def test_serial_steps_do_not_share_a_start_time(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1), _llm_step(2, deps=[1])])
        result = await chain.execute_async(_ctx())
        first, second = result.trace.events[0], result.trace.events[1]
        assert second.started_at >= first.ended_at

    @pytest.mark.asyncio
    async def test_timestamps_round_trip(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx())
        restored = ExecutionTrace.from_json(result.trace.to_json())
        assert restored.get_event(1).started_at == result.trace.get_event(1).started_at
        assert restored.get_event(1).ended_at == result.trace.get_event(1).ended_at

    def test_intervals_prefer_measured_timestamps(self) -> None:
        """Measured timing wins over the batch assumption it contradicts."""
        trace = ExecutionTrace(
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="a",
                    success=True,
                    execution_time=1.0,
                    batch_index=0,
                    started_at="2026-01-01T00:00:00+00:00",
                    ended_at="2026-01-01T00:00:01+00:00",
                ),
                TraceEvent(
                    step_number=2,
                    step_title="b",
                    success=True,
                    execution_time=1.0,
                    batch_index=0,  # same batch — synthesis would start it at 0
                    started_at="2026-01-01T00:00:04+00:00",
                    ended_at="2026-01-01T00:00:06+00:00",
                ),
            ]
        )
        rows = trace._compute_step_intervals()
        assert [(s, e) for _, s, e in rows] == [(0.0, 1.0), (4.0, 6.0)]

    def test_intervals_fall_back_without_timestamps(self) -> None:
        trace = ExecutionTrace(
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="a",
                    success=True,
                    execution_time=1.0,
                    batch_index=0,
                ),
                TraceEvent(
                    step_number=2,
                    step_title="b",
                    success=True,
                    execution_time=2.0,
                    batch_index=1,
                ),
            ]
        )
        rows = trace._compute_step_intervals()
        assert [(s, e) for _, s, e in rows] == [(0.0, 1.0), (1.0, 3.0)]

    def test_partially_timestamped_trace_falls_back(self) -> None:
        """Mixing measured and missing timestamps would mix time origins."""
        trace = ExecutionTrace(
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="a",
                    success=True,
                    execution_time=1.0,
                    batch_index=0,
                    started_at="2026-01-01T00:00:00+00:00",
                    ended_at="2026-01-01T00:00:01+00:00",
                ),
                TraceEvent(
                    step_number=2,
                    step_title="b",
                    success=True,
                    execution_time=2.0,
                    batch_index=1,
                ),
            ]
        )
        rows = trace._compute_step_intervals()
        assert [(s, e) for _, s, e in rows] == [(0.0, 1.0), (1.0, 3.0)]

    def test_unparseable_timestamps_fall_back(self) -> None:
        trace = ExecutionTrace(
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="a",
                    success=True,
                    execution_time=1.0,
                    batch_index=0,
                    started_at="not-a-timestamp",
                    ended_at="also-not",
                ),
            ]
        )
        assert trace._compute_step_intervals() == [(trace.events[0], 0.0, 1.0)]

    @pytest.mark.asyncio
    async def test_gantt_renders_with_measured_timestamps(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1), _llm_step(2, deps=[1])])
        result = await chain.execute_async(_ctx())
        text = result.trace.format_gantt()
        assert "total wall time" in text
        assert "gantt" in result.trace.format_gantt(format="mermaid")


# ---------------------------------------------------------------------------
# nested sub-chain traces
# ---------------------------------------------------------------------------


def _handoff_chain() -> ReasoningChain:
    sub_chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="Sub step", aim="Do sub work."),
        ]
    )
    return ReasoningChain(
        steps=[
            AgentHandoffStepDescription(
                number=1,
                title="Delegate",
                sub_chain=sub_chain,
                config=AgentHandoffStepConfig(
                    agent_name="specialist",
                    task_description="Handle it.",
                    output_memory_key="handoff_result",
                ),
            )
        ]
    )


class TestSubTrace:
    @pytest.mark.asyncio
    async def test_sub_trace_is_attached(self) -> None:
        result = await _handoff_chain().execute_async(_ctx())
        sub_trace = result.trace.get_event(1).sub_trace
        assert isinstance(sub_trace, ExecutionTrace)
        assert [e.step_number for e in sub_trace.events] == [1]
        assert sub_trace.events[0].step_title == "Sub step"

    @pytest.mark.asyncio
    async def test_sub_trace_survives_round_trip(self) -> None:
        result = await _handoff_chain().execute_async(_ctx())
        restored = ExecutionTrace.from_json(result.trace.to_json())
        sub_trace = restored.get_event(1).sub_trace
        assert isinstance(sub_trace, ExecutionTrace)
        assert sub_trace.events[0].step_title == "Sub step"
        assert sub_trace.events[0].step_type == StepType.LLM

    @pytest.mark.asyncio
    async def test_sub_trace_keeps_nested_model_attribution(self) -> None:
        result = await _handoff_chain().execute_async(_ctx(_MockClient(model="sub/m")))
        restored = ExecutionTrace.from_json(result.trace.to_json())
        assert restored.get_event(1).sub_trace.events[0].model == "sub/m"

    @pytest.mark.asyncio
    async def test_plain_step_has_no_sub_trace(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx())
        assert result.trace.get_event(1).sub_trace is None

    def test_nested_sub_trace_dict_is_json_serialisable(self) -> None:
        inner = ExecutionTrace(
            chain_title="inner",
            events=[TraceEvent(step_number=1, step_title="i", success=True)],
        )
        outer = ExecutionTrace(
            chain_title="outer",
            events=[
                TraceEvent(
                    step_number=1,
                    step_title="o",
                    success=True,
                    sub_trace=inner,
                )
            ],
        )
        payload = json.loads(outer.to_json())
        assert payload["events"][0]["sub_trace"]["chain_title"] == "inner"


# ---------------------------------------------------------------------------
# downstream consumers keep working
# ---------------------------------------------------------------------------


class TestConsumersUnaffected:
    @pytest.mark.asyncio
    async def test_to_html_shows_new_fields(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(_ctx(_MockClient(model="acme/model-1"), trace_prompts=True))
        html = result.trace.to_html()
        assert "acme/model-1" in html
        assert "<dt>model</dt>" in html
        assert "<dt>started</dt>" in html

    @pytest.mark.asyncio
    async def test_to_html_handles_nested_sub_trace(self) -> None:
        result = await _handoff_chain().execute_async(_ctx())
        html = result.trace.to_html()
        assert "sub_trace" in html
        assert "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_diff_and_aggregator_still_work(self) -> None:
        chain = ReasoningChain(steps=[_llm_step(1)])
        first = await chain.execute_async(_ctx())
        second = await chain.execute_async(_ctx(_MockClient("different")))
        diff = first.trace.diff(second.trace)
        assert len(diff["changed"]) == 1
        assert TraceAggregator([first.trace, second.trace]).n_traces == 2
