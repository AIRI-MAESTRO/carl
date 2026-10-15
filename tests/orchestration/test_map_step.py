"""Bounded ordered MapStep fan-out, aggregation, and lifecycle contracts."""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from mmar_carl import (
    LLMClientBase,
    MapItemOutcome,
    MapItemStatus,
    MapOutcome,
    MapStepConfig,
    MapStepDescription,
    MapStepExecutor,
    ReasoningChain,
    ReasoningContext,
    StepType,
    create_step,
    get_executor,
)


class _Stub(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "unused"


_DEFAULT_ITEMS = object()


def _context(items: Any = _DEFAULT_ITEMS) -> ReasoningContext:
    outer = [] if items is _DEFAULT_ITEMS else items
    return ReasoningContext(
        outer_context=json.dumps(outer, ensure_ascii=False, allow_nan=False),
        api=_Stub(),
    )


def _step(**overrides: Any) -> MapStepDescription:
    config = {
        "items_source": "$outer_context",
        "tool_name": "map_tool",
        "max_items": 100,
        "max_concurrency": 3,
        "item_timeout_seconds": 0.5,
    }
    config.update(overrides)
    return MapStepDescription(
        number=1,
        title="map",
        config=MapStepConfig(**config),
    )


async def _execute(step: MapStepDescription, ctx: ReasoningContext):
    return await get_executor(StepType.MAP).execute(step, ctx)


class TestMapModels:
    def test_models_are_pydantic(self) -> None:
        assert issubclass(MapStepConfig, BaseModel)
        assert issubclass(MapItemOutcome, BaseModel)
        assert issubclass(MapOutcome, BaseModel)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"items_source": "$history[-1]"}, "items_source must use"),
            ({"items_source": "$memory."}, "items_source must use"),
            ({"tool_name": " "}, "value cannot be empty"),
            ({"item_parameter": "bad-name"}, "valid Python parameter"),
            ({"index_parameter": "class"}, "valid Python parameter"),
            ({"index_parameter": "item"}, "must differ"),
            ({"input_mapping": {"item": "$metadata.x"}}, "cannot overwrite"),
            ({"input_mapping": {"x": "literal"}}, "must use"),
            ({"item_timeout_seconds": math.inf}, "must be finite"),
            ({"output_memory_key": " "}, "cannot be empty"),
        ],
    )
    def test_config_rejects_invalid_contract(
        self, kwargs: dict[str, Any], message: str,
    ) -> None:
        base = {"items_source": "$outer_context", "tool_name": "map_tool"}
        base.update(kwargs)
        with pytest.raises(ValidationError, match=message):
            MapStepConfig(**base)

    def test_item_outcome_enforces_terminal_invariants(self) -> None:
        with pytest.raises(ValidationError, match="exactly for completed"):
            MapItemOutcome(
                index=0,
                status=MapItemStatus.FAILED,
                success=True,
                error_type="ValueError",
                error_message="bad",
                execution_time=0,
            )

    def test_aggregate_enforces_order_and_counts(self) -> None:
        item = MapItemOutcome(
            index=1,
            status=MapItemStatus.COMPLETED,
            success=True,
            output="ok",
            execution_time=0,
        )
        with pytest.raises(ValidationError, match="contiguous input order"):
            MapOutcome(
                items=[item],
                total_items=1,
                completed_items=1,
                failed_items=0,
                timed_out_items=0,
                cancelled_items=0,
                max_concurrency=1,
            )


class TestMapExecution:
    @pytest.mark.asyncio
    async def test_empty_array_succeeds_without_calls(self) -> None:
        ctx = _context([])
        calls = 0

        def tool(item: Any) -> Any:
            nonlocal calls
            calls += 1
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(), ctx)

        assert result.success
        assert calls == 0
        assert result.as_map_outcome().items == []
        assert result.as_map_outcome().total_items == 0

    @pytest.mark.asyncio
    async def test_any_json_item_shared_arguments_and_index(self) -> None:
        items = [None, True, 3, "x", [1], {"a": 2}]
        ctx = _context(items)
        ctx.metadata["prefix"] = "p"

        def tool(item: Any, index: int, prefix: str) -> dict[str, Any]:
            return {"item": item, "index": index, "prefix": prefix}

        ctx.register_tool("map_tool", tool)
        result = await _execute(
            _step(
                index_parameter="index",
                input_mapping={"prefix": "$metadata.prefix"},
            ),
            ctx,
        )

        assert result.success
        outputs = [item.output for item in result.as_map_outcome().items]
        assert outputs == [
            {"item": item, "index": index, "prefix": "p"}
            for index, item in enumerate(items)
        ]

    @pytest.mark.asyncio
    async def test_async_completion_order_does_not_change_input_order(self) -> None:
        ctx = _context([0.03, 0.0, 0.01])
        completion_order: list[float] = []

        async def tool(item: float) -> float:
            await asyncio.sleep(item)
            completion_order.append(item)
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(), ctx)

        assert completion_order != [0.03, 0.0, 0.01]
        assert [item.output for item in result.as_map_outcome().items] == [
            0.03, 0.0, 0.01,
        ]

    @pytest.mark.asyncio
    async def test_max_concurrency_is_enforced(self) -> None:
        ctx = _context(list(range(12)))
        active = 0
        observed_max = 0

        async def tool(item: int) -> int:
            nonlocal active, observed_max
            active += 1
            observed_max = max(observed_max, active)
            await asyncio.sleep(0.005)
            active -= 1
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(max_concurrency=2), ctx)

        assert result.success
        assert observed_max == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("items", [{"not": "array"}, "[]", None])
    async def test_non_array_fails_before_tool_call(self, items: Any) -> None:
        ctx = _context(items)
        calls = 0

        def tool(item: Any) -> Any:
            nonlocal calls
            calls += 1
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(), ctx)

        assert not result.success
        assert "JSON array" in result.error_message
        assert calls == 0

    @pytest.mark.asyncio
    async def test_non_json_array_fails_before_tool_call(self) -> None:
        ctx = _context([])
        ctx.metadata["items"] = [object()]
        calls = 0

        def tool(item: Any) -> Any:
            nonlocal calls
            calls += 1
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(items_source="$metadata.items"), ctx)

        assert not result.success
        assert "JSON-compatible array" in result.error_message
        assert calls == 0

    @pytest.mark.asyncio
    async def test_max_items_fails_before_tool_call(self) -> None:
        ctx = _context([1, 2])
        calls = 0

        def tool(item: Any) -> Any:
            nonlocal calls
            calls += 1
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(max_items=1), ctx)

        assert not result.success
        assert "exceeding max_items=1" in result.error_message
        assert calls == 0

    @pytest.mark.asyncio
    async def test_missing_tool_fails_before_any_item(self) -> None:
        result = await _execute(_step(), _context([1]))

        assert not result.success
        assert "not registered" in result.error_message

    @pytest.mark.asyncio
    async def test_non_callable_registry_entry_fails_before_items(self) -> None:
        ctx = _context([1])
        ctx._tool_registry["map_tool"] = object()

        result = await _execute(_step(), ctx)

        assert not result.success
        assert "not callable" in result.error_message

    @pytest.mark.asyncio
    async def test_collects_exception_and_non_json_failures(self) -> None:
        ctx = _context(["ok", "raise", "object", "nan"])

        def tool(item: str) -> Any:
            if item == "raise":
                raise RuntimeError("boom")
            if item == "object":
                return object()
            if item == "nan":
                return float("nan")
            return {"value": item}

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(), ctx)
        outcome = result.as_map_outcome()

        assert not result.success
        assert [item.status for item in outcome.items] == [
            MapItemStatus.COMPLETED,
            MapItemStatus.FAILED,
            MapItemStatus.FAILED,
            MapItemStatus.FAILED,
        ]
        assert outcome.failed_items == 3
        assert outcome.items[1].error_type == "RuntimeError"
        assert outcome.items[2].error_type == "NonJsonResult"

    @pytest.mark.asyncio
    async def test_timeout_is_per_item_and_collect_all_continues(self) -> None:
        ctx = _context(["slow", "fast"])

        async def tool(item: str) -> str:
            if item == "slow":
                await asyncio.sleep(1)
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(item_timeout_seconds=0.01), ctx)
        outcome = result.as_map_outcome()

        assert not result.success
        assert [item.status for item in outcome.items] == [
            MapItemStatus.TIMED_OUT,
            MapItemStatus.COMPLETED,
        ]
        assert outcome.enforcement_gaps == []

    @pytest.mark.asyncio
    async def test_sync_timeout_reports_enforcement_gap(self) -> None:
        ctx = _context([1])

        def tool(item: int) -> int:
            time.sleep(0.05)
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(_step(item_timeout_seconds=0.005), ctx)
        outcome = result.as_map_outcome()

        assert outcome.items[0].status == MapItemStatus.TIMED_OUT
        assert outcome.enforcement_gaps == [
            "sync tool 'map_tool' may continue after timeout"
        ]

    @pytest.mark.asyncio
    async def test_registered_sync_wrapper_timeout_is_typed(self) -> None:
        ctx = _context([1])

        def tool(item: int) -> int:
            time.sleep(0.05)
            return item

        ctx.register_tool("map_tool", tool, timeout=0.005)
        result = await _execute(_step(item_timeout_seconds=1), ctx)
        outcome = result.as_map_outcome()

        assert outcome.items[0].status == MapItemStatus.TIMED_OUT
        assert outcome.enforcement_gaps == [
            "sync tool 'map_tool' may continue after timeout"
        ]

    @pytest.mark.asyncio
    async def test_cancellation_stops_owned_async_calls_and_pending_items(self) -> None:
        ctx = _context([0, 1, 2, 3])
        started = asyncio.Event()
        cleaned: list[int] = []

        async def tool(item: int) -> int:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.append(item)

        ctx.register_tool("map_tool", tool)
        execution = asyncio.create_task(
            _execute(_step(max_concurrency=2), ctx)
        )
        await started.wait()
        await asyncio.sleep(0)
        ctx.cancel()
        result = await asyncio.wait_for(execution, timeout=1)
        outcome = result.as_map_outcome()

        assert not result.success
        assert result.skipped
        assert outcome.cancelled_items == 4
        assert sorted(cleaned) == [0, 1]
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("carl-map-")
            and not task.done()
        ]

    @pytest.mark.asyncio
    async def test_writes_aggregate_once_after_partial_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctx = _context([1, 2])
        writes: list[tuple[str, Any, str]] = []
        original = ReasoningContext.memory_write

        def recording_write(
            self: ReasoningContext, key: str, value: Any, namespace: str = "default",
        ) -> None:
            writes.append((key, value, namespace))
            original(self, key, value, namespace)

        monkeypatch.setattr(ReasoningContext, "memory_write", recording_write)

        def tool(item: int) -> int:
            if item == 2:
                raise ValueError("bad")
            return item

        ctx.register_tool("map_tool", tool)
        result = await _execute(
            _step(output_memory_key="out", output_namespace="batch"), ctx,
        )

        assert not result.success
        assert len(writes) == 1
        assert writes[0][0::2] == ("out", "batch")
        assert writes[0][1] == result.result_data

    @pytest.mark.asyncio
    async def test_chain_commits_only_partial_aggregate_output(self) -> None:
        ctx = _context([1, 2])

        def tool(item: int) -> int:
            if item == 2:
                raise ValueError("bad")
            return item

        ctx.register_tool("map_tool", tool)
        chain = ReasoningChain(
            steps=[
                _step(output_memory_key="out", output_namespace="batch")
            ]
        )

        result = await chain.execute_async(ctx)

        assert not result.success
        assert ctx.memory["batch"]["out"] == result.step_results[0].result_data


class TestMapSourcesAndWireContract:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "source",
        [
            "$outer_context",
            "$memory.input.items",
            "$metadata.items",
            "$steps.1.result_data.items",
            "$event.items_ready",
        ],
    )
    async def test_supported_array_sources(self, source: str) -> None:
        ctx = _context([1, 2])
        ctx.memory_write("items", [1, 2], namespace="input")
        ctx.metadata["items"] = [1, 2]
        ctx.metadata["step_results"] = {"1": {"result_data": {"items": [1, 2]}}}
        ctx.emit_event("items_ready", [1, 2])
        ctx.register_tool("map_tool", lambda item: item)

        result = await _execute(_step(items_source=source), ctx)

        assert result.success
        assert [item.output for item in result.as_map_outcome().items] == [1, 2]

    def test_serialization_round_trip_and_preflight(self) -> None:
        chain = ReasoningChain(
            steps=[
                _step(
                    output_memory_key="out",
                )
            ]
        )
        data = chain.to_dict()
        loaded = ReasoningChain.from_dict(data)

        assert data["format_version"] == 10
        assert data["steps"][0]["step_type"] == "map"
        assert isinstance(loaded.steps[0], MapStepDescription)
        assert loaded.steps[0].step_config.max_concurrency == 3
        assert chain.required_tools() == ["map_tool"]
        assert chain.preflight(_context([])).missing_tools == ["map_tool"]

    def test_migration_to_current_does_not_mutate_input(self) -> None:
        raw = {"format_version": 6, "steps": []}

        migrated = ReasoningChain.migrate(raw)

        assert raw == {"format_version": 6, "steps": []}
        assert migrated["format_version"] == 10

    def test_executor_is_registered(self) -> None:
        assert isinstance(get_executor(StepType.MAP), MapStepExecutor)

    def test_create_step_factory(self) -> None:
        config = MapStepConfig(
            items_source="$outer_context",
            tool_name="map_tool",
        )

        step = create_step(
            number=1,
            title="map",
            step_type=StepType.MAP,
            config=config,
        )

        assert isinstance(step, MapStepDescription)
        assert step.config is config
