"""Contract and execution tests for the typed HumanInputStep."""

import asyncio
import math
from datetime import datetime

import pytest
from pydantic import ValidationError

from mmar_carl import (
    HumanInputOutcome,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputStepConfig,
    HumanInputStepDescription,
    HumanInputStepExecutor,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.enums import StepType


class MockClient(LLMClientBase):
    async def get_response(self, prompt):
        return "ok"

    async def get_response_with_retries(self, prompt, retries=3):
        return "ok"


def make_context(language=Language.ENGLISH):
    return ReasoningContext(
        outer_context="test",
        api=MockClient(),
        language=language,
    )


def make_step(
    *,
    number=1,
    prompt="Enter your feedback:",
    timeout=None,
    min_length=0,
    max_length=4096,
    sensitive=False,
    output_memory_key=None,
):
    return HumanInputStepDescription(
        number=number,
        title=f"Human input {number}",
        config=HumanInputStepConfig(
            prompt=prompt,
            timeout=timeout,
            min_length=min_length,
            max_length=max_length,
            sensitive=sensitive,
            output_memory_key=output_memory_key,
        ),
    )


def answer(
    request: HumanInputRequest,
    value: str = "answer",
    *,
    actor_id: str | None = "human-1",
) -> HumanInputResponse:
    return HumanInputResponse(
        request_id=request.request_id,
        value=value,
        actor_id=actor_id,
        provenance={"channel": "test"},
    )


class TestHumanInputModels:
    def test_config_defaults(self):
        config = HumanInputStepConfig()
        assert config.prompt == "Please provide input:"
        assert config.timeout is None
        assert config.min_length == 0
        assert config.max_length == 4096
        assert config.sensitive is False
        assert config.output_memory_key is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"prompt": "   "}, "prompt cannot be empty"),
            ({"output_memory_key": " "}, "output_memory_key cannot be empty"),
            ({"min_length": 3, "max_length": 2}, "min_length cannot exceed"),
            ({"timeout": math.inf}, "timeout must be finite"),
            ({"sensitive": True}, "sensitive input requires output_memory_key"),
            ({"fallback_value": "fake"}, "fallback_value was removed"),
        ],
    )
    def test_config_rejects_invalid_contract(self, kwargs, message):
        with pytest.raises(ValidationError, match=message):
            HumanInputStepConfig(**kwargs)

    def test_request_rejects_naive_deadline(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            HumanInputRequest(
                request_id="r1",
                step_number=1,
                step_title="input",
                prompt="Prompt",
                deadline=datetime(2026, 1, 1),  # noqa: DTZ001 - invalid by design
            )

    def test_response_rejects_blank_actor_id(self):
        with pytest.raises(ValidationError, match="actor_id cannot be empty"):
            HumanInputResponse(request_id="r1", value="ok", actor_id=" ")

    def test_response_rejects_naive_timestamp(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            HumanInputResponse(
                request_id="r1",
                value="ok",
                responded_at=datetime(2026, 1, 1),  # noqa: DTZ001 - invalid by design
            )

    def test_response_rejects_oversized_provenance(self):
        with pytest.raises(ValidationError, match="provenance cannot exceed"):
            HumanInputResponse(
                request_id="r1",
                value="ok",
                provenance={"blob": "x" * 9000},
            )


class TestHumanInputOutcomes:
    @pytest.mark.asyncio
    async def test_missing_provider_is_unavailable_not_success(self):
        result = await HumanInputStepExecutor().execute(make_step(), make_context())

        outcome = result.as_human_input_outcome()
        assert result.success is False
        assert result.result == ""
        assert outcome is not None
        assert outcome.status == "unavailable"
        assert outcome.value is None
        assert result.error_message == "human input provider unavailable"

    @pytest.mark.asyncio
    async def test_sync_provider_returns_typed_response(self):
        ctx = make_context()
        seen: list[HumanInputRequest] = []

        def provider(request: HumanInputRequest) -> HumanInputResponse:
            seen.append(request)
            return answer(request, "Alice")

        ctx.on_human_input_requested = provider
        result = await HumanInputStepExecutor().execute(
            make_step(prompt="Name?", min_length=1), ctx,
        )

        outcome = result.as_human_input_outcome()
        assert result.success is True
        assert result.result == "Alice"
        assert outcome is not None
        assert outcome.status == "answered"
        assert outcome.value == "Alice"
        assert outcome.actor_id == "human-1"
        assert seen[0].kind == "text"
        assert seen[0].prompt == "Name?"
        assert seen[0].deadline is None

    @pytest.mark.asyncio
    async def test_async_provider_is_awaited(self):
        ctx = make_context()

        async def provider(request: HumanInputRequest) -> HumanInputResponse:
            await asyncio.sleep(0)
            return answer(request, "async value")

        ctx.on_human_input_requested = provider
        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is True
        assert result.result == "async value"

    @pytest.mark.asyncio
    async def test_dict_response_is_validated(self):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: {
            "request_id": request.request_id,
            "value": "dict value",
            "actor_id": "human-2",
        }

        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is True
        assert result.result == "dict value"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", ["raw string", 42, None])
    async def test_untyped_response_is_invalid(self, raw):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: raw

        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is False
        assert result.as_human_input_outcome().status == "invalid_response"
        assert result.error_message == "invalid human input response"

    @pytest.mark.asyncio
    async def test_request_id_mismatch_is_invalid(self):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: HumanInputResponse(
            request_id="different-request",
            value="wrong target",
        )

        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is False
        assert result.as_human_input_outcome().status == "invalid_response"
        assert "request_id mismatch" in result.error_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["x", "toolong"])
    async def test_response_length_is_validated(self, value):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: answer(request, value)

        result = await HumanInputStepExecutor().execute(
            make_step(min_length=2, max_length=4), ctx,
        )

        assert result.success is False
        assert result.as_human_input_outcome().status == "invalid_response"
        assert result.error_message == "human input response length is invalid"

    @pytest.mark.asyncio
    async def test_provider_exception_is_failed_without_leaking_message(self):
        ctx = make_context()

        def provider(request):
            raise RuntimeError("secret provider detail")

        ctx.on_human_input_requested = provider
        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is False
        assert result.as_human_input_outcome().status == "failed"
        assert result.error_message == "human input provider failed"
        assert "secret" not in str(result.model_dump())


class TestHumanInputLifecycle:
    @pytest.mark.asyncio
    async def test_timeout_is_non_success_and_cleans_provider(self):
        ctx = make_context()
        provider_finished = asyncio.Event()

        async def provider(request):
            try:
                await asyncio.Future()
            finally:
                provider_finished.set()

        ctx.on_human_input_requested = provider
        result = await HumanInputStepExecutor().execute(
            make_step(timeout=0.01, output_memory_key="answer"), ctx,
        )

        assert result.success is False
        assert result.as_human_input_outcome().status == "timed_out"
        assert result.error_message == "human input timed out"
        assert provider_finished.is_set()
        assert ctx.memory_read("answer", namespace="human_input") is None

    @pytest.mark.asyncio
    async def test_preexisting_cancellation_skips_without_calling_provider(self):
        ctx = make_context()
        called = False

        def provider(request):
            nonlocal called
            called = True
            return answer(request)

        ctx.on_human_input_requested = provider
        ctx.cancel()
        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is False
        assert result.skipped is True
        assert result.as_human_input_outcome().status == "cancelled"
        assert called is False

    @pytest.mark.asyncio
    async def test_inflight_cancellation_stops_owned_provider(self):
        ctx = make_context()
        provider_started = asyncio.Event()
        provider_finished = asyncio.Event()

        async def provider(request):
            provider_started.set()
            try:
                await asyncio.Future()
            finally:
                provider_finished.set()

        ctx.on_human_input_requested = provider
        execution = asyncio.create_task(
            HumanInputStepExecutor().execute(make_step(), ctx),
        )
        await provider_started.wait()
        ctx.cancel()
        result = await asyncio.wait_for(execution, timeout=1)

        assert result.success is False
        assert result.skipped is True
        assert result.as_human_input_outcome().status == "cancelled"
        assert provider_finished.is_set()

    @pytest.mark.asyncio
    async def test_cancellation_wins_same_turn_response_race(self):
        ctx = make_context()

        def provider(request):
            ctx.cancel()
            return answer(request, "must not win")

        ctx.on_human_input_requested = provider
        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert result.success is False
        assert result.skipped is True
        assert result.as_human_input_outcome().status == "cancelled"
        assert "must not win" not in str(result.model_dump())


class TestHumanInputDataFlow:
    @pytest.mark.asyncio
    async def test_answer_writes_memory_and_english_history(self):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: answer(request, "great work")

        result = await HumanInputStepExecutor().execute(
            make_step(output_memory_key="feedback"), ctx,
        )

        assert result.success is True
        assert ctx.memory_read("feedback", namespace="human_input") == "great work"
        assert "Human input (answered): great work" in result.updated_history[-1]

    @pytest.mark.asyncio
    async def test_sensitive_answer_is_redacted_but_written_to_memory(self):
        ctx = make_context()
        ctx.on_human_input_requested = lambda request: HumanInputResponse(
            request_id=request.request_id,
            value="private answer",
            provenance={"unsafe_echo": "private answer"},
        )

        result = await HumanInputStepExecutor().execute(
            make_step(sensitive=True, output_memory_key="secret"), ctx,
        )

        outcome = result.as_human_input_outcome()
        assert result.success is True
        assert result.result == "[redacted]"
        assert outcome.status == "answered"
        assert outcome.value is None
        assert outcome.redacted is True
        assert outcome.provenance == {}
        assert "private answer" not in str(result.model_dump())
        assert "[redacted]" in result.updated_history[-1]
        assert ctx.memory_read("secret", namespace="human_input") == "private answer"

    @pytest.mark.asyncio
    async def test_russian_history_entry(self):
        ctx = make_context(language=Language.RUSSIAN)
        ctx.on_human_input_requested = lambda request: answer(request, "ответ")

        result = await HumanInputStepExecutor().execute(make_step(), ctx)

        assert "Ввод пользователя (получен): ответ" in result.updated_history[-1]

    @pytest.mark.asyncio
    async def test_non_success_never_writes_memory_or_new_history(self):
        ctx = make_context()
        ctx.history = ["existing"]

        result = await HumanInputStepExecutor().execute(
            make_step(output_memory_key="answer"), ctx,
        )

        assert result.updated_history == ["existing"]
        assert ctx.memory_read("answer", namespace="human_input") is None


class TestHumanInputChainIntegration:
    @pytest.mark.asyncio
    async def test_parallel_requests_have_independent_ids_and_outputs(self):
        chain = ReasoningChain(
            steps=[
                make_step(number=1, output_memory_key="one"),
                make_step(number=2, output_memory_key="two"),
            ],
            max_workers=2,
        )
        ctx = make_context()
        request_ids: set[str] = set()

        async def provider(request: HumanInputRequest) -> HumanInputResponse:
            request_ids.add(request.request_id)
            await asyncio.sleep(0)
            return answer(request, f"answer-{request.step_number}")

        ctx.on_human_input_requested = provider
        result = await chain.execute_async(ctx)

        assert result.success is True
        assert len(request_ids) == 2
        assert ctx.memory_read("one", namespace="human_input") == "answer-1"
        assert ctx.memory_read("two", namespace="human_input") == "answer-2"

    @pytest.mark.asyncio
    async def test_chain_failure_when_provider_is_missing(self):
        chain = ReasoningChain(steps=[make_step()])

        result = await chain.execute_async(make_context())

        assert result.success is False
        assert result.step_results[0].as_human_input_outcome().status == "unavailable"

    @pytest.mark.asyncio
    async def test_parent_context_cancels_parallel_snapshot_wait(self):
        chain = ReasoningChain(steps=[make_step()])
        ctx = make_context()
        provider_started = asyncio.Event()
        provider_finished = asyncio.Event()

        async def provider(request):
            provider_started.set()
            try:
                await asyncio.Future()
            finally:
                provider_finished.set()

        ctx.on_human_input_requested = provider
        execution = asyncio.create_task(chain.execute_async(ctx))
        await provider_started.wait()
        ctx.cancel()
        result = await asyncio.wait_for(execution, timeout=1)

        human_result = result.step_results[0]
        assert result.success is False
        assert human_result.skipped is True
        assert human_result.as_human_input_outcome().status == "cancelled"
        assert provider_finished.is_set()

    def test_step_serialization_uses_current_format_version(self):
        chain = ReasoningChain(
            steps=[make_step(output_memory_key="answer", min_length=1)],
        )

        data = chain.to_dict()
        loaded = ReasoningChain.from_dict(data)

        assert data["format_version"] == ReasoningChain.FORMAT_VERSION == 10
        assert data["steps"][0]["step_type"] == "human_input"
        assert "fallback_value" not in data["steps"][0]["step_config"]
        assert loaded.steps[0].step_config.min_length == 1
        assert loaded.steps[0].step_config.output_memory_key == "answer"


def test_step_type_is_human_input():
    assert make_step().step_type is StepType.HUMAN_INPUT


def test_public_models_round_trip_json():
    request = HumanInputRequest(
        request_id="request-1",
        step_number=1,
        step_title="Input",
        prompt="Prompt",
    )
    response = answer(request)
    outcome = HumanInputOutcome(
        status="answered",
        request_id=request.request_id,
        elapsed_seconds=0,
        value=response.value,
        actor_id=response.actor_id,
        responded_at=response.responded_at,
        provenance=response.provenance,
    )

    assert HumanInputRequest.model_validate_json(request.model_dump_json()) == request
    assert HumanInputResponse.model_validate_json(response.model_dump_json()) == response
    assert HumanInputOutcome.model_validate_json(outcome.model_dump_json()) == outcome
