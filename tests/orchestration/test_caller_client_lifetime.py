"""The executor must not close an LLM client it did not create.

``DAGExecutor.execute_batch`` builds one snapshot :class:`ReasoningContext` per
step in a batch, each constructed with ``api=context.api`` — i.e. holding a
*reference* to the caller's client, not a copy of it.  Its ``finally`` block then
called ``snapshot.close()``, and ``ReasoningContext.close()`` closes
``OpenAICompatibleClient``, which closes the underlying ``AsyncOpenAI`` and with
it the shared httpx connection pool.

So every batch of every chain execution tore down the HTTP connection pool of a
client owned by the caller.  On its own that is merely wasteful — the pool is
rebuilt lazily on the next request.  It becomes data loss the moment two chain
executions share one client, which is the normal shape of a dataset evaluation:
the first chain to finish a batch closes the pool out from under every request
the other chains still have in flight, and those surface as
``APIConnectionError("Connection error.")``.  Measured against vLLM: 23-26% of
chain executions lost, flat across client concurrency 1..40, with the server
reporting HTTP 200 for everything it was allowed to finish and
``Request ... aborted`` for the rest.

Clients the *snapshot itself* created — the per-step ``llm_config`` overrides in
``_llm_client_cache`` — are still closed, because nobody else can be using them.
"""

import asyncio
import json
import pytest

from mmar_carl import (
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    RetryPolicy,
)
from mmar_carl.models.config import LLMStepConfig
from mmar_carl.models.steps import LLMStepDescription


# ---------------------------------------------------------------------------
# Arm 1 — the contract, checked without any I/O
# ---------------------------------------------------------------------------


class CloseCountingClient(OpenAICompatibleClient):
    """A caller-owned client that counts every ``close()``, request path intact."""

    def __init__(self, config: OpenAIClientConfig) -> None:
        super().__init__(config)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        await super().close()


class OfflineCloseCountingClient(CloseCountingClient):
    """Same, but answers from memory so the contract can be checked without I/O."""

    async def get_response_with_usage(self, prompt: str, retries: int = 3):  # noqa: ARG002
        return "answer", {"prompt": 1, "completion": 1, "total": 2}

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return "answer"


def offline_config(**overrides) -> OpenAIClientConfig:
    base = dict(base_url="http://127.0.0.1:9/v1", api_key="unused", model="stub")
    base.update(overrides)
    return OpenAIClientConfig(**base)


def two_step_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="First", aim="first"),
        LLMStepDescription(number=2, title="Second", aim="second", dependencies=[1]),
    ])


@pytest.mark.asyncio
async def test_execute_async_does_not_close_the_callers_client():
    client = OfflineCloseCountingClient(offline_config())
    context = ReasoningContext(outer_context="input", api=client)

    result = await two_step_chain().execute_async(context)

    assert result.success, result.get_failed_steps()
    assert client.close_calls == 0, (
        "the executor closed a client it did not create; a caller sharing this "
        "client across concurrent chain executions loses their in-flight requests"
    )


@pytest.mark.asyncio
async def test_context_close_still_closes_the_callers_client():
    """The public, caller-invoked ``close()`` keeps its documented behaviour."""
    client = OfflineCloseCountingClient(offline_config())
    context = ReasoningContext(outer_context="input", api=client)

    await two_step_chain().execute_async(context)
    await context.close()

    assert client.close_calls == 1


# ---------------------------------------------------------------------------
# Arm 2 — over real sockets
# ---------------------------------------------------------------------------


async def _serve_chat_completions(reader, writer):
    """Minimal OpenAI-compatible endpoint. A prompt containing SLOW waits."""
    try:
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = await reader.read(4096)
            if not chunk:
                return
            head += chunk
        headers, _, body = head.partition(b"\r\n\r\n")
        length = 0
        for line in headers.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":", 1)[1])
        while len(body) < length:
            more = await reader.read(length - len(body))
            if not more:
                return
            body += more

        if b"SLOW" in body:
            await asyncio.sleep(1.0)

        payload = json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        return
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_a_finished_chain_does_not_abort_a_sibling_sharing_the_client():
    """One fast chain finishing must not kill the slow chains' in-flight requests.

    Retries are switched off on purpose (``max_attempts=1``, ``retry_max=1``, SDK
    ``max_retries=0``). With them on, the surviving chains usually reconnect and
    the loss shows up as a ~25% rate rather than a failure — which is exactly why
    this went unnoticed for so long. Off, the defect is deterministic.
    """
    server = await asyncio.start_server(_serve_chat_completions, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    client = OpenAICompatibleClient(offline_config(
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=30.0,
        retry_policy=RetryPolicy(max_attempts=1),
    ))
    client.client.max_retries = 0

    def run(aim: str):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="Step", aim=aim)]
        )
        context = ReasoningContext(outer_context=aim, api=client, retry_max=1)
        return chain.execute_async(context)

    async with server:
        # The FAST chain returns first and tears down the batch it shares with
        # nobody — but the pool it releases is the one the SLOW chains are on.
        results = await asyncio.gather(
            run("FAST"), *(run(f"SLOW {i}") for i in range(4))
        )

    await client.close()

    failed = [r for r in results if not r.success]
    assert not failed, (
        "a sibling chain lost its in-flight request when another chain finished: "
        + "; ".join(
            step.error_message or "?" for r in failed for step in r.get_failed_steps()
        )
    )


@pytest.mark.asyncio
async def test_per_step_override_clients_are_still_closed():
    """Clients the snapshot built for a per-step ``llm_config`` are snapshot-owned.

    Nobody outside the snapshot can hold one, so releasing them after the batch
    stays correct — the narrowed cleanup must not turn into no cleanup.
    """
    server = await asyncio.start_server(_serve_chat_completions, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    client = CloseCountingClient(offline_config(base_url=f"http://127.0.0.1:{port}/v1"))
    context = ReasoningContext(outer_context="input", api=client)
    chain = ReasoningChain(steps=[
        LLMStepDescription(
            number=1,
            title="First",
            aim="first",
            llm_config=LLMStepConfig(model="other-model"),
        ),
    ])

    built: list[OpenAICompatibleClient] = []
    original = ReasoningContext.get_llm_client_for_step

    def spy(self, llm_config=None):
        resolved = original(self, llm_config)
        if resolved is not client:
            built.append(resolved)
        return resolved

    ReasoningContext.get_llm_client_for_step = spy  # type: ignore[method-assign]
    try:
        async with server:
            result = await chain.execute_async(context)
    finally:
        ReasoningContext.get_llm_client_for_step = original  # type: ignore[method-assign]

    assert result.success, result.get_failed_steps()
    assert built, "the per-step llm_config should have produced its own client"
    assert all(c._client is None for c in built), (
        "snapshot-owned override clients must still be released after the batch"
    )
    assert client.close_calls == 0
