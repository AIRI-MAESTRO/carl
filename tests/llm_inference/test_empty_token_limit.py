"""Reasoning-only completions must not look like successful empty answers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion

from mmar_carl import (
    LLMTokenLimitError,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    RetryPolicy,
)


def completion(content=None, finish_reason="length"):
    return ChatCompletion.model_validate({
        "id": "test", "created": 0, "object": "chat.completion", "model": "reasoner",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content, "reasoning": "Not a final answer",
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 512, "total_tokens": 612,
                  "completion_tokens_details": {"reasoning_tokens": 512}},
    })


def client_with(response, policy=None):
    client = OpenAICompatibleClient(OpenAIClientConfig(
        api_key="test", model="reasoner", max_tokens=512, retry_policy=policy,
    ))
    create = AsyncMock(return_value=response)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, RetryPolicy(max_attempts=3)])
@pytest.mark.parametrize("method,args", [
    ("get_response", ("Question",)),
    ("get_response_with_retries", ("Question",)),
    ("get_response_with_usage", ("Question",)),
    ("get_response_with_system", ("System", "Question")),
    ("get_response_with_messages", ([{"role": "user", "content": "Question"}],)),
])
async def test_reasoning_only_limit_fails_without_retry_or_budget_increase(policy, method, args):
    client, create = client_with(completion(), policy)
    with pytest.raises(LLMTokenLimitError, match="output token limit") as error:
        await getattr(client, method)(*args)
    assert error.value.token_usage == {"prompt": 100, "completion": 512, "total": 612}
    assert error.value.reasoning_tokens == 512
    assert "Not a final answer" not in str(error.value)
    create.assert_awaited_once()
    assert create.call_args.kwargs["max_tokens"] == 512
    assert client.config.max_tokens == 512


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", " \n"])
async def test_chain_fails_and_retains_billed_tokens(content):
    client, create = client_with(completion(content))
    chain = ReasoningChain(steps=[LLMStepDescription(number=1, title="Answer", aim="Answer briefly")])
    result = await chain.execute_async(ReasoningContext(outer_context="Question", api=client))
    assert not result.success
    assert "output token limit" in result.step_results[0].error_message
    assert result.step_results[0].token_usage == {"prompt": 100, "completion": 512, "total": 612}
    assert result.token_usage["total"] == 612
    create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_visible_text_still_returns_with_usage(finish_reason):
    client, create = client_with(completion("Visible answer", finish_reason))
    content, usage = await client.get_response_with_usage("Question")
    assert content == "Visible answer"
    assert usage["total"] == 612
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_only_response_is_not_an_empty_text_failure():
    response = completion()
    response.choices[0].finish_reason = "tool_calls"
    from openai.types.chat import ChatCompletionMessageFunctionToolCall
    response.choices[0].message.tool_calls = [ChatCompletionMessageFunctionToolCall.model_validate({
        "id": "call_1", "type": "function", "function": {"name": "answer", "arguments": "{}"},
    })]
    client, _ = client_with(response)
    content, calls, usage = await client.get_response_with_tools_and_usage("", "Question", [])
    assert content == ""
    assert calls[0]["name"] == "answer"
    assert usage["total"] == 612
