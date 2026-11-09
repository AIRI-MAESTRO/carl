"""Self-contained WaitStep timers, events, composition and cancellation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mmar_carl import (
    AfterWaitCondition,
    AnyOfWaitCondition,
    AtWaitCondition,
    EventWaitCondition,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    WaitStepConfig,
    WaitStepDescription,
    get_executor,
)


class _Stub(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "unused"


def _context() -> ReasoningContext:
    return ReasoningContext(outer_context="wait", api=_Stub())


def _step(condition, *, output_memory_key: str | None = None) -> WaitStepDescription:
    return WaitStepDescription(
        number=1,
        title="wait",
        config=WaitStepConfig(
            condition=condition,
            output_memory_key=output_memory_key,
        ),
    )


async def _execute(step: WaitStepDescription, ctx: ReasoningContext):
    return await get_executor(StepType.WAIT).execute(step, ctx)


class TestWaitConditionValidation:
    @pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
    def test_after_rejects_invalid_duration(self, seconds: float) -> None:
        with pytest.raises(ValidationError):
            AfterWaitCondition(seconds=seconds)

    def test_at_requires_timezone(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            AtWaitCondition(timestamp=datetime.now())  # noqa: DTZ005 - intentionally naive

    def test_event_requires_non_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="cannot be empty"):
            EventWaitCondition(name="   ")

    def test_any_of_requires_at_least_two_leaf_conditions(self) -> None:
        with pytest.raises(ValidationError):
            AnyOfWaitCondition(conditions=[AfterWaitCondition(seconds=0)])

    def test_nested_any_of_is_not_a_leaf(self) -> None:
        with pytest.raises(ValidationError):
            AnyOfWaitCondition.model_validate({
                "conditions": [
                    {"type": "any_of", "conditions": [
                        {"type": "after", "seconds": 0},
                        {"type": "after", "seconds": 1},
                    ]},
                    {"type": "after", "seconds": 2},
                ]
            })


class TestWaitTimers:
    @pytest.mark.asyncio
    async def test_after_zero_completes_and_has_typed_view(self) -> None:
        result = await _execute(_step(AfterWaitCondition(seconds=0)), _context())

        assert result.success
        assert result.result_data["trigger"] == "after"
        assert result.result_data["seconds"] == 0.0
        typed = result.as_wait_outcome()
        assert typed is not None
        assert typed.trigger == "after"

    @pytest.mark.asyncio
    async def test_after_does_not_block_event_loop(self) -> None:
        ctx = _context()
        task = asyncio.create_task(
            _execute(_step(AfterWaitCondition(seconds=0.03)), ctx)
        )
        await asyncio.sleep(0)
        assert not task.done()
        marker: list[str] = []
        marker.append("event-loop-ran")
        result = await task
        assert marker == ["event-loop-ran"]
        assert result.success

    @pytest.mark.asyncio
    async def test_at_past_completes_immediately(self) -> None:
        timestamp = datetime.now(UTC) - timedelta(seconds=1)
        result = await _execute(_step(AtWaitCondition(timestamp=timestamp)), _context())

        assert result.success
        assert result.result_data["trigger"] == "at"
        assert result.result_data["timestamp"] == timestamp.isoformat()


class TestWaitEvents:
    @pytest.mark.asyncio
    async def test_pre_emitted_event_completes_immediately(self) -> None:
        ctx = _context()
        ctx.emit_event("ready", {"rows": 42})

        result = await _execute(_step(EventWaitCondition(name="ready")), ctx)

        assert result.success
        assert result.result_data["trigger"] == "event"
        assert result.result_data["name"] == "ready"
        assert result.result_data["payload"] == {"rows": 42}

    @pytest.mark.asyncio
    async def test_later_event_wakes_without_polling(self, monkeypatch) -> None:
        ctx = _context()

        def forbidden_poll(self, name: str) -> bool:
            raise AssertionError(f"polled {name}")

        monkeypatch.setattr(ReasoningContext, "has_event", forbidden_poll)
        task = asyncio.create_task(
            _execute(_step(EventWaitCondition(name="ready")), ctx)
        )
        await asyncio.sleep(0)
        ctx.emit_event("ready", {"ok": True})
        result = await asyncio.wait_for(task, timeout=1)

        assert result.success
        assert result.result_data["payload"] == {"ok": True}

    @pytest.mark.asyncio
    async def test_event_emission_from_sync_thread_is_safe(self) -> None:
        ctx = _context()
        task = asyncio.create_task(
            _execute(_step(EventWaitCondition(name="thread_ready")), ctx)
        )
        await asyncio.sleep(0)
        await asyncio.to_thread(ctx.emit_event, "thread_ready", {"thread": True})
        result = await asyncio.wait_for(task, timeout=1)

        assert result.success
        assert result.result_data["payload"] == {"thread": True}

    @pytest.mark.asyncio
    async def test_non_json_event_payload_fails(self) -> None:
        ctx = _context()
        ctx.emit_event("bad", object())

        result = await _execute(_step(EventWaitCondition(name="bad")), ctx)

        assert not result.success
        assert "JSON-compatible" in (result.error_message or "")


class TestAnyOf:
    @pytest.mark.asyncio
    async def test_event_wins_event_or_timeout_and_writes_memory(self) -> None:
        ctx = _context()
        step = _step(
            AnyOfWaitCondition(conditions=[
                EventWaitCondition(name="ready"),
                AfterWaitCondition(seconds=1),
            ]),
            output_memory_key="outcome",
        )
        task = asyncio.create_task(_execute(step, ctx))
        await asyncio.sleep(0)
        ctx.emit_event("ready", {"value": 7})
        result = await asyncio.wait_for(task, timeout=1)

        assert result.success
        assert result.result_data["trigger"] == "event"
        assert result.result_data["condition_index"] == 0
        assert ctx.memory["wait"]["outcome"] == result.result_data

    @pytest.mark.asyncio
    async def test_timer_wins_event_or_timeout(self) -> None:
        condition = AnyOfWaitCondition(conditions=[
            EventWaitCondition(name="never"),
            AfterWaitCondition(seconds=0),
        ])
        result = await _execute(_step(condition), _context())

        assert result.success
        assert result.result_data["trigger"] == "after"
        assert result.result_data["condition_index"] == 1

    @pytest.mark.asyncio
    async def test_declaration_order_breaks_same_turn_tie(self) -> None:
        condition = AnyOfWaitCondition(conditions=[
            AfterWaitCondition(seconds=0),
            AfterWaitCondition(seconds=0),
        ])
        result = await _execute(_step(condition), _context())

        assert result.success
        assert result.result_data["condition_index"] == 0


class TestWaitCancellationAndParallelism:
    @pytest.mark.asyncio
    async def test_cancellation_interrupts_event_wait(self) -> None:
        ctx = _context()
        task = asyncio.create_task(
            _execute(_step(EventWaitCondition(name="never")), ctx)
        )
        await asyncio.sleep(0)
        ctx.cancel()
        result = await asyncio.wait_for(task, timeout=1)

        assert not result.success
        assert result.skipped
        assert result.error_message == "cancelled by user"
        await asyncio.sleep(0)
        assert not [
            pending
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
            and pending.get_name().startswith("carl-wait-")
            and not pending.done()
        ]

    @pytest.mark.asyncio
    async def test_parallel_sibling_can_wake_waiter(self) -> None:
        ctx = _context()

        async def emit() -> str:
            await asyncio.sleep(0.01)
            ctx.emit_event("sibling_ready", {"source": "tool"})
            return "emitted"

        ctx.register_tool("emit", emit)
        chain = ReasoningChain(
            steps=[
                WaitStepDescription(
                    number=1,
                    title="wait",
                    config=WaitStepConfig(
                        condition=EventWaitCondition(name="sibling_ready")
                    ),
                ),
                ToolStepDescription(
                    number=2,
                    title="emit",
                    config=ToolStepConfig(
                        tool_name="emit",
                        parameters=[],
                        input_mapping={},
                    ),
                ),
            ],
            max_workers=2,
        )

        result = await asyncio.wait_for(chain.execute_async(ctx), timeout=1)

        assert result.success
        wait_result = next(item for item in result.step_results if item.step_type is StepType.WAIT)
        assert wait_result.result_data["payload"] == {"source": "tool"}

    @pytest.mark.asyncio
    async def test_parallel_event_is_not_rolled_back_when_emitter_fails(self) -> None:
        ctx = _context()

        def emit_then_fail() -> str:
            ctx.emit_event("published", {"committed": False})
            raise RuntimeError("producer failed after emission")

        ctx.register_tool("emit_then_fail", emit_then_fail)
        chain = ReasoningChain(
            steps=[
                WaitStepDescription(
                    number=1,
                    title="wait",
                    config=WaitStepConfig(
                        condition=EventWaitCondition(name="published")
                    ),
                ),
                ToolStepDescription(
                    number=2,
                    title="emit then fail",
                    config=ToolStepConfig(
                        tool_name="emit_then_fail",
                        parameters=[],
                        input_mapping={},
                    ),
                ),
            ],
            max_workers=2,
        )

        result = await asyncio.wait_for(chain.execute_async(ctx), timeout=1)

        wait_result = next(
            item for item in result.step_results if item.step_type is StepType.WAIT
        )
        tool_result = next(
            item for item in result.step_results if item.step_type is StepType.TOOL
        )
        assert wait_result.success
        assert wait_result.result_data["payload"] == {"committed": False}
        assert not tool_result.success
