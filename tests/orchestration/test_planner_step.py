"""Bounded planning uses the existing DAG and never accepts model-authored tool arguments."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

try:
    from httpx2 import RemoteProtocolError, Request, Response
except ImportError:
    from httpx import RemoteProtocolError, Request, Response

from mmar_carl import (
    DAGExecutor,
    EventWaitCondition,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    PlannerStepConfig,
    PlannerStepDescription,
    ReasoningChain,
    ReasoningContext,
    ReplanPolicy,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    WaitStepConfig,
    WaitStepDescription,
)


class ScriptedClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    async def get_response_with_retries(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


def make_chain(**kwargs):
    return ReasoningChain(
        steps=[
            PlannerStepDescription(
                number=1, title="Plan",
                config=PlannerStepConfig(
                    instruction="Copy the requested action IDs in the requested order.",
                    input_mapping={"requested": "$outer_context.requested"},
                    actions={
                        "first": ToolStepConfig(tool_name="record", input_mapping={"value": "'FIXED-1'"}),
                        "second": ToolStepConfig(tool_name="record", input_mapping={"value": "'FIXED-2'"}),
                    },
                    max_actions=2,
                ),
            ),
            ToolStepDescription(number=2, title="After plan", dependencies=[1], config=ToolStepConfig(
                tool_name="record", input_mapping={"value": "'AFTER'"},
            )),
        ],
        fail_fast=True, **kwargs,
    )


def make_context(client):
    calls = []
    ctx = ReasoningContext(
        api=client, outer_context="PRIVATE-CANARY",
        runtime_input={"requested": ["second", "first"], "other": "PRIVATE-RUNTIME"},
    )

    async def record(value):
        calls.append(value)
        return value

    ctx.register_tool("record", record)
    return ctx, calls


@pytest.mark.asyncio
async def test_round_trip_executes_plan_before_static_dependents():
    client = ScriptedClient('{"actions":["second","first"]}')
    ctx, calls = make_context(client)
    chain = ReasoningChain.from_json(make_chain().to_json())

    result = await chain.execute_async(ctx)

    assert result.success
    assert result.metadata["execution_stats"]["total_steps"] == 4
    assert calls == ["FIXED-2", "FIXED-1", "AFTER"]
    plan = result.get_step_result(1)
    assert plan.step_type is StepType.PLANNER
    assert plan.result_data == {"actions": ["second", "first"], "step_numbers": [3, 4]}
    assert "PRIVATE-CANARY" not in client.prompts[0]
    assert "PRIVATE-RUNTIME" not in client.prompts[0]
    assert "FIXED-1" not in client.prompts[0]
    assert chain.required_tools() == ["record"]


@pytest.mark.parametrize("reply", [
    '{"actions":["arbitrary_tool"]}',
    '{"actions":["first","first"]}',
    '{"actions":[{"name":"first","arguments":{"value":"injected"}}]}',
    '{"actions":["first"],"args":{"value":"injected"}}',
])
@pytest.mark.asyncio
async def test_rejects_untrusted_plan_without_running_any_tool(reply):
    ctx, calls = make_context(ScriptedClient(reply))
    result = await make_chain().execute_async(ctx)
    assert not result.success
    assert not calls


@pytest.mark.asyncio
async def test_empty_plan_releases_static_dependents():
    ctx, calls = make_context(ScriptedClient('{"actions":[]}'))
    result = await make_chain().execute_async(ctx)
    assert result.success
    assert calls == ["AFTER"]


@pytest.mark.asyncio
async def test_rejects_saved_resume_replan_and_insufficient_budget():
    ctx, calls = make_context(ScriptedClient('{"actions":[]}'))
    with pytest.raises(ValueError, match="saved snapshot"):
        await make_chain().execute_async(ctx, resume_from=ctx.snapshot())
    with pytest.raises(ValueError, match="saved snapshot"):
        await DAGExecutor().execute(make_chain().steps, ctx, resume_from_step_numbers={1})
    with pytest.raises(ValueError, match="RE-PLAN"):
        make_chain(replan_policy=ReplanPolicy())
    with pytest.raises(ValueError, match="max_injections"):
        make_chain(max_injections=1)
    with pytest.raises(ValueError, match="cannot replay"):
        await make_chain().replay(None, ctx)
    with pytest.raises(ValueError, match="cannot replay"):
        await make_chain().execute_from_trace(None, 1, ctx)
    assert not calls


@pytest.mark.live
@pytest.mark.asyncio
async def test_planner_on_free_openrouter():
    model = os.environ.get("LIVE_TEST_MODEL", "")
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key or not model.endswith(":free"):
        pytest.skip("set an OpenRouter API key and explicit free LIVE_TEST_MODEL")
    client = OpenAICompatibleClient(OpenAIClientConfig(
        model=model, api_key=key, base_url="https://openrouter.ai/api/v1",
        temperature=0, max_tokens=4096, timeout=30,
    ))
    ctx, calls = make_context(client)
    try:
        result = await make_chain().execute_async(ctx)
    finally:
        await ctx.close()
    assert result.success
    assert calls == ["FIXED-2", "FIXED-1", "AFTER"]


@pytest.mark.asyncio
async def test_concurrent_runs_have_independent_injection_budgets():
    import asyncio

    both_planning = asyncio.Event()
    planners = 0

    class ConcurrentClient:
        async def get_response_with_retries(self, prompt, **kwargs):
            nonlocal planners
            planners += 1
            if planners == 2:
                both_planning.set()
            await both_planning.wait()
            return '{"actions":["second","first"]}'

    chain = make_chain(max_injections=2)
    first_ctx, first_calls = make_context(ConcurrentClient())
    second_ctx, second_calls = make_context(ConcurrentClient())
    first, second = await asyncio.gather(chain.execute_async(first_ctx), chain.execute_async(second_ctx))

    assert first.success and second.success
    assert first_calls == second_calls == ["FIXED-2", "FIXED-1", "AFTER"]


@pytest.mark.parametrize("reply, stream_error, succeeds", [
    ('{"actions":["second","first"]}', ConnectionResetError, True),
    ('{"actions":["first"]', ConnectionResetError, False),
    ('{"actions":["arbitrary_tool"]}', ConnectionResetError, False),
    ('{"actions":["first"]}', RuntimeError, False),
])
@pytest.mark.asyncio
async def test_planner_only_recovers_a_valid_complete_plan_after_transport_reset(reply, stream_error, succeeds):
    class ResettingClient(ScriptedClient):
        async def stream_response(self, prompt):
            self.prompts.append(prompt)
            yield reply
            raise stream_error("stream failed")

    client = ResettingClient(reply)
    ctx, calls = make_context(client)
    ctx.retry_max = 1
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert result.success is succeeds
    assert calls == (["FIXED-2", "FIXED-1", "AFTER"] if succeeds else [])
    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_planner_completion_can_wake_same_batch_wait_step_and_reports_scheduled_ids():
    ctx, _ = make_context(ScriptedClient('{"actions":["second","first"]}'))
    observed = []

    def on_step_complete(result):
        if result.step_type == StepType.PLANNER:
            observed.append(("complete", dict(result.result_data)))
            ctx.emit_event("plan_complete", {})
        else:
            observed.append(result.step_number)

    ctx.on_step_complete = on_step_complete
    ctx.on_step_event = lambda number, event, data: observed.append((number, event, data))
    chain = make_chain()
    chain.steps.append(WaitStepDescription(
        number=3,
        title="Await planner completion",
        config=WaitStepConfig(condition=EventWaitCondition(name="plan_complete")),
    ))
    result = await asyncio.wait_for(chain.execute_async(ctx), timeout=2)

    assert result.success
    assert observed == [
        ("complete", {"actions": ["second", "first"]}),
        3,
        (1, "planner_actions_scheduled", {"actions": ["second", "first"], "step_numbers": [4, 5]}),
        4, 5, 2,
    ]


@pytest.mark.parametrize("reset_after_usage", [False, True])
@pytest.mark.asyncio
async def test_planner_reports_provider_stream_usage_after_the_json(reset_after_usage):
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def __aiter__(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='{"actions":[]}'))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[], usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5),
            )
            if reset_after_usage:
                raise RemoteProtocolError("stream ended unexpectedly")

    client = OpenAICompatibleClient(OpenAIClientConfig(api_key="test", model="test"))
    create = AsyncMock(return_value=Stream())
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx, _ = make_context(client)
    seen = []
    ctx.on_llm_chunk = seen.append

    result = await make_chain().execute_async(ctx)

    assert result.success
    assert result.get_step_result(1).token_usage == {"prompt": 17, "completion": 5, "total": 22}
    assert result.token_usage == {"prompt": 17, "completion": 5, "total": 22}
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert seen == ['{"actions":[]}']


@pytest.mark.asyncio
async def test_planner_usage_stream_respects_overridden_openai_text_stream():
    class CustomClient(OpenAICompatibleClient):
        async def stream_response(self, prompt):
            yield '{"actions":[]}'

    client = CustomClient(OpenAIClientConfig(api_key="test", model="test"))
    create = AsyncMock(side_effect=AssertionError("custom stream should handle the request"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx, _ = make_context(client)
    ctx.retry_max = 1
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert result.success
    assert result.get_step_result(1).token_usage == {}
    create.assert_not_awaited()


@pytest.mark.parametrize("body, retries_without_option", [
    ({"message": "Unknown parameter: 'stream_options'.", "param": "stream_options", "code": "unknown_parameter"}, True),
    ({"message": "stream_options is not supported by this endpoint."}, True),
    ({"message": "unsupported stream_options"}, True),
    ({"message": "Unsupported model for streaming", "param": "model", "code": "unsupported_parameter"}, False),
    ({"message": "Invalid value", "param": "stream_options", "code": "invalid_value"}, False),
])
@pytest.mark.asyncio
async def test_openai_usage_option_retries_only_on_explicit_unsupported_parameter(body, retries_without_option):
    from openai import BadRequestError

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def __aiter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"actions":[]}'))])

    response = Response(400, request=Request("POST", "https://example.test/v1/chat/completions"))
    error = BadRequestError(body["message"], response=response, body=body)
    create = AsyncMock(side_effect=[error, Stream()])
    client = OpenAICompatibleClient(OpenAIClientConfig(api_key="test", model="test"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx, _ = make_context(client)
    ctx.retry_max = 1
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert result.success is retries_without_option
    assert create.await_count == (2 if retries_without_option else 1)
    assert create.await_args_list[0].kwargs["stream_options"] == {"include_usage": True}
    if retries_without_option:
        assert "stream_options" not in create.await_args_list[1].kwargs
        assert result.get_step_result(1).token_usage == {}


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
@pytest.mark.asyncio
async def test_planner_does_not_retry_permanent_stream_errors(status):
    from openai import APIStatusError

    response = Response(status, request=Request("POST", "https://example.test/v1/chat/completions"))
    create = AsyncMock(side_effect=APIStatusError("request failed", response=response, body=None))
    client = OpenAICompatibleClient(OpenAIClientConfig(api_key="test", model="test"))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx, calls = make_context(client)
    ctx.retry_max = 2
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert not result.success
    assert not calls
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_streams_with_openai_sdk_create_without_stream_options():
    class LegacySDKCompletions:
        def __init__(self):
            self.response = SimpleNamespace(aclose=AsyncMock())

        async def create(self, *, model, messages, temperature, stream):
            return self

        async def __aiter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"actions":[]}'))])

    client = OpenAICompatibleClient(OpenAIClientConfig(api_key="test", model="test"))
    completions = LegacySDKCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    ctx, calls = make_context(client)
    ctx.retry_max = 1
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert result.success
    assert calls == ["AFTER"]
    assert result.get_step_result(1).token_usage == {}
    completions.response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_enforces_configured_action_count_below_catalog_size():
    client = ScriptedClient('{"actions":["first","second"]}')
    ctx, calls = make_context(client)
    chain = make_chain()
    chain.steps[0].step_config.max_actions = 1

    result = await chain.execute_async(ctx)

    assert not result.success
    assert not calls
    assert "maxItems" in result.get_step_result(1).error_message


@pytest.mark.asyncio
async def test_planner_does_not_replay_rejected_host_stream_options():
    from openai import BadRequestError

    body = {
        "message": "Unknown parameter: 'stream_options'.",
        "param": "stream_options",
        "code": "unknown_parameter",
    }
    response = Response(400, request=Request("POST", "https://example.test/v1/chat/completions"))
    create = AsyncMock(side_effect=BadRequestError(body["message"], response=response, body=body))
    stream_options = {"include_usage": False}
    client = OpenAICompatibleClient(OpenAIClientConfig(
        api_key="test", model="test", extra_body={"stream_options": stream_options},
    ))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ctx, calls = make_context(client)
    ctx.on_llm_chunk = lambda chunk: None

    result = await make_chain().execute_async(ctx)

    assert not result.success
    assert not calls
    create.assert_awaited_once()
    assert "stream_options" not in create.await_args.kwargs
    assert create.await_args.kwargs["extra_body"]["stream_options"] == stream_options
