"""
Abstract base classes for CARL reasoning system.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A single message in a multi-turn conversation.

    Used by :class:`~mmar_carl.models.context.ReasoningContext` to accumulate
    a structured conversation history across LLM steps when
    ``LLMStepConfig.use_message_history=True``.

    Example::

        ctx.messages.append(ChatMessage(role="user", content="What is 2+2?"))
        ctx.messages.append(ChatMessage(role="assistant", content="4"))
    """

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Message role: 'system', 'user', or 'assistant'.",
    )
    content: str = Field(..., description="Message text content.")

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI-style message dict."""
        return {"role": self.role, "content": self.content}


class LLMClientBase(ABC):
    """Abstract base class for LLM clients.

    Concrete subclasses (e.g. ``OpenAICompatibleClient``) carry their own
    configuration objects, but call sites that only need basic
    introspection (current model name, temperature, max_tokens, streaming
    support) should prefer the typed properties exposed here instead of
    digging into subclass-specific attributes via ``getattr`` or
    ``isinstance`` checks. Subclasses override the properties to return
    real values; the base returns ``None`` and ``False`` so callers can
    branch safely.
    """

    # ------------------------------------------------------------------
    # Introspection properties — override in subclasses where available
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> Optional[str]:
        """Return the model identifier this client is configured to use.

        Returns ``None`` when the client has no fixed model (e.g. clients
        that select per-call). Subclasses with a static model should
        override.
        """
        return None

    @property
    def temperature(self) -> Optional[float]:
        """Return the temperature this client uses by default, if any."""
        return None

    @property
    def max_tokens(self) -> Optional[int]:
        """Return the max-tokens cap this client applies, if any."""
        return None

    @property
    def supports_streaming(self) -> bool:
        """Whether this client implements :meth:`stream_response`.

        Default: detect by checking whether the subclass overrode the
        default abstract :meth:`stream_response` (i.e. did not raise
        ``NotImplementedError`` at class definition).  Subclasses that
        provide streaming should leave the default — it will detect them
        automatically. Subclasses that explicitly want to disable streaming
        despite providing the method may override and return ``False``.
        """
        try:
            base_method = LLMClientBase.stream_response  # type: ignore[attr-defined]
            return type(self).stream_response is not base_method  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover — defensive
            return False

    # ------------------------------------------------------------------
    # Per-step override protocol
    # ------------------------------------------------------------------

    def clone_with_overrides(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        step: Optional[str] = None,
    ) -> "Optional[LLMClientBase]":
        """Return a twin of this client with per-step sampling overrides applied.

        This is the protocol
        :meth:`~mmar_carl.models.context.ReasoningContext.get_llm_client_for_step`
        uses to honour ``LLMStepConfig.model`` / ``temperature`` /
        ``max_tokens``.  It lives here, on the base class, on purpose: the
        resolver used to clone only :class:`OpenAICompatibleClient` by
        ``isinstance``, so every client type added afterwards —
        :class:`~mmar_carl.anthropic_client.AnthropicClient`, the
        record/replay wrappers, any user-supplied client — silently ran the
        *un-overridden* model while the chain claimed otherwise.

        The default implementation covers every client that keeps its
        settings in a pydantic model on ``self.config`` and is constructed
        as ``type(self)(config)`` (both shipped clients qualify): it copies
        the config, applies the overrides, and re-validates, so an override
        the provider cannot accept (e.g. ``temperature=1.5`` against a
        client whose config caps it at 1.0) is caught here instead of at
        request time.  Note it *re-validates* rather than using
        ``model_copy(update=…)``, which would happily install an
        out-of-range value.

        Subclasses whose configuration does not live in a pydantic
        ``self.config`` — or whose constructor needs more than the config —
        should override this method.  Returning ``None`` means "this client
        cannot honour the override"; callers must then say so out loud
        rather than quietly falling back, because the silence is what made
        this defect survive.

        Args:
            model / temperature / max_tokens: overrides to apply.  ``None``
                means "keep this client's current value" — note ``0.0`` is a
                real temperature override, so implementations must test
                ``is not None`` rather than truthiness.
            step: diagnostic only — the name of the step whose config is
                being applied, for use in messages.  Ignored by the default
                implementation.  Callers should route it through
                :func:`clone_llm_client_with_overrides`, which only forwards
                it to implementations that accept it.

        Returns:
            A new client carrying the overrides, ``self`` when no override
            was actually requested, or ``None`` when this client cannot be
            cloned.
        """
        del step  # diagnostic-only; the default has nothing to report it in
        updates = _set_overrides(model, temperature, max_tokens)
        if not updates:
            return self

        config = getattr(self, "config", None)
        if not isinstance(config, BaseModel):
            return None
        fields = type(config).model_fields
        if any(name not in fields for name in updates):
            return None

        try:
            base_values = {name: getattr(config, name) for name in fields}
            new_config = type(config)(**{**base_values, **updates})
            return type(self)(new_config)  # type: ignore[call-arg]
        except Exception as exc:
            # Broad on purpose: the contract is "None means the override
            # cannot be honoured", and a resolver that crashed mid-chain
            # would be a worse regression than a refused clone.  The reason
            # is reported rather than swallowed.
            warnings.warn(
                f"{type(self).__name__}.clone_with_overrides() could not build a client "
                f"for {describe_llm_overrides(model, temperature, max_tokens)}: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    async def close(self) -> None:
        """Release any resources this client holds (HTTP pools, sockets).

        Declared here so
        :meth:`~mmar_carl.models.context.ReasoningContext.close` can tear
        down *any* client through one call instead of an ``isinstance``
        ladder that only ever named :class:`OpenAICompatibleClient`.

        The default is a no-op — clients that hold nothing (mocks, replay
        wrappers) need not override.  Clients that lazily build a transport
        should close it and drop the reference, since a context may be
        reused after :meth:`close` and is expected to rebuild on demand.
        """
        return None

    # ------------------------------------------------------------------
    # Optional streaming hook — declared here so call sites can call it
    # without ``hasattr`` / ``getattr`` plumbing.
    # ------------------------------------------------------------------

    def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream a response from the LLM, yielding text chunks.

        Default implementation raises ``NotImplementedError`` (eagerly, the
        moment the method is called — not on first iteration — since
        callers should check :attr:`supports_streaming` first when calling
        defensively).  Subclasses that support streaming (e.g.
        :class:`OpenAICompatibleClient`) override with an ``async def`` that
        yields chunks from the underlying provider's streaming API; their
        broader ``AsyncIterator[str]``-compatible return shape is accepted
        here.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            An async iterator yielding text chunks.

        Example::

            async for chunk in client.stream_response("Hello!"):
                print(chunk, end="", flush=True)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stream_response(). "
            "Either subclass and override, or check supports_streaming before calling."
        )

    @abstractmethod
    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        pass

    @abstractmethod
    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            The LLM response as a string
        """
        pass

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        """
        Get a response using separate system and user messages.

        Default implementation falls back to concatenating system + user into
        a single user message (for backward-compatible mock clients). Concrete
        clients (e.g. OpenAICompatibleClient) should override this to send
        proper system/user message roles.

        Args:
            system_prompt: System-role instructions (e.g. skill instructions)
            user_prompt: User-role task message
            retries: Maximum retry attempts

        Returns:
            The LLM response as a string
        """
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        return await self.get_response_with_retries(combined, retries=retries)

    async def get_response_with_usage(self, prompt: str, retries: int = 3) -> tuple[str, dict[str, int]]:
        """
        Get a response plus token usage information.

        Returns a tuple ``(response_text, usage)`` where ``usage`` is a dict with
        keys ``"prompt"``, ``"completion"``, and ``"total"`` (all ints).

        Default implementation delegates to ``get_response_with_retries`` and returns
        empty usage. Concrete clients should override to populate usage from the API.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            Tuple of (response text, token usage dict)
        """
        result = await self.get_response_with_retries(prompt, retries=retries)
        return result, {}

    async def get_response_with_messages(
        self,
        messages: "list[ChatMessage]",
        retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        """
        Get a response from a structured multi-turn message list.

        Returns ``(response_text, token_usage)`` in the same style as
        :meth:`get_response_with_usage`.

        Default implementation flattens messages to a single string and
        delegates to :meth:`get_response_with_usage`.  Concrete clients
        (e.g. :class:`~mmar_carl.llm.OpenAICompatibleClient`) should
        override this to send the message list natively.

        Args:
            messages: Ordered list of :class:`ChatMessage` objects.
            retries:  Number of retry attempts.

        Returns:
            Tuple ``(response_text, token_usage_dict)``.
        """
        # Flatten: skip system role (prepend), join user/assistant turns
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.insert(0, m.content)
            else:
                parts.append(m.content)
        flat = "\n\n".join(parts)
        return await self.get_response_with_usage(flat, retries=retries)

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Get a response with OpenAI-style tool/function call support.

        Returns a tuple ``(content, tool_calls)`` where:
        - ``content``    — the LLM's text response (may be empty if tool calls returned)
        - ``tool_calls`` — list of ``{"id": str, "name": str, "arguments": dict}``

        Default implementation does NOT support tool calls — it calls
        ``get_response_with_system()`` and returns empty ``tool_calls``.

        Concrete clients that support function calling (e.g. OpenAICompatibleClient)
        should override this method to use the native tool call API.

        Args:
            system_prompt: System-role instructions
            user_prompt:   User-role message (ignored when ``messages`` is provided)
            tools:         OpenAI-style tool definitions
            messages:      Full conversation history (system_prompt/user_prompt ignored
                           when provided)
        """
        if messages:
            # Reconstruct a combined prompt from the last user message
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                user_prompt,
            )
            content = await self.get_response_with_retries(last_user)
        else:
            content = await self.get_response_with_system(system_prompt, user_prompt)
        return content, []

    async def get_response_with_tools_and_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        """Return a tool-calling response plus standard usage counters.

        The compatibility default delegates to :meth:`get_response_with_tools`
        and returns an empty usage mapping. Provider clients that expose usage
        should override this method.
        """
        content, tool_calls = await self.get_response_with_tools(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            messages=messages,
        )
        return content, tool_calls, {}

# ----------------------------------------------------------------------
# Helpers for the per-step override protocol
# ----------------------------------------------------------------------


def _set_overrides(
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> dict[str, Any]:
    """Collect the overrides that were actually requested.

    ``is not None`` rather than truthiness: ``temperature=0.0`` (greedy
    decoding) and ``max_tokens=0`` are real values a caller can ask for.
    """
    requested = (("model", model), ("temperature", temperature), ("max_tokens", max_tokens))
    return {name: value for name, value in requested if value is not None}


def describe_llm_overrides(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Render the requested overrides for a diagnostic message.

    Only the fields actually set are listed, so a message points at the
    knob the caller turned rather than at every knob that exists.
    """
    parts = [f"{name}={value!r}" for name, value in _set_overrides(model, temperature, max_tokens).items()]
    return ", ".join(parts) if parts else "<no overrides set>"


def _accepts_step_kwarg(hook: Any) -> bool:
    """Whether ``hook`` accepts the diagnostic ``step=`` keyword.

    ``clone_with_overrides`` implementations predating ``step`` (and the
    duck-typed clients CARL accepts in ``ReasoningContext.api``) take only
    the three sampling kwargs.  Passing ``step=`` blindly would turn a
    working client into a ``TypeError`` at step-execution time, so the
    parameter is offered, not imposed.
    """
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):  # pragma: no cover — exotic callables
        return False
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "step" and param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def clone_llm_client_with_overrides(
    client: Any,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    step: Optional[str] = None,
) -> Optional[LLMClientBase]:
    """Ask *client* for a twin carrying the given per-step overrides.

    Dispatch is duck-typed on :meth:`LLMClientBase.clone_with_overrides`
    rather than ``isinstance``, for two reasons: ``ReasoningContext.api``
    is typed ``Any`` and accepts anything client-shaped, and an
    ``isinstance`` ladder naming one concrete class is exactly what let
    per-step overrides be dropped for every other client.

    Returns ``None`` when *client* does not implement the protocol or
    declines to clone.  Callers must treat ``None`` as "the override
    cannot be honoured" and report it — never fall back in silence.
    """
    hook = getattr(client, "clone_with_overrides", None)
    if not callable(hook):
        return None
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if step is not None and _accepts_step_kwarg(hook):
        kwargs["step"] = step
    return hook(**kwargs)
