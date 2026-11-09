"""Per-step LLM overrides must survive for *every* client, not just OpenAI's.

``LLMStepConfig`` lets a step override ``model`` / ``temperature`` /
``max_tokens``.  Until the clone protocol existed,
``ReasoningContext.get_llm_client_for_step`` applied those overrides only
after an ``isinstance(..., OpenAICompatibleClient)`` test and returned the
un-overridden client for everything else — so CARL's own
:class:`AnthropicClient` ran the wrong model, at the wrong temperature,
with no warning.  A chain that evolves per-step models (the reason EvoC
drives CARL) therefore measured a configuration it never actually ran.

These tests pin the three halves of the fix:

* the override reaches a **non-OpenAI** client (deliberately: special-casing
  one concrete class is what let the defect exist, so the tests must not
  re-encode that class);
* a client that genuinely cannot clone degrades **loudly** — silence is the
  defect, a named warning is survivable;
* teardown goes through the same protocol, so a non-OpenAI client and every
  cached override clone are actually closed.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import pytest

from mmar_carl.anthropic_client import AnthropicClient, AnthropicClientConfig
from mmar_carl.llm import OpenAIClientConfig, OpenAICompatibleClient, RetryPolicy
from mmar_carl.models.config import LLMStepConfig
from mmar_carl.models.context import ReasoningContext
from mmar_carl.models.llm_client_base import (
    LLMClientBase,
    clone_llm_client_with_overrides,
    describe_llm_overrides,
)


class _UncloneableClient(LLMClientBase):
    """A client whose settings do not live in a pydantic ``self.config``.

    Stands in for every hand-rolled client CARL accepts in
    ``ReasoningContext.api``: the default clone protocol cannot copy it, so
    it must be *reported*, not silently ignored.
    """

    @property
    def model_name(self) -> str:
        return "legacy-model"

    @property
    def temperature(self) -> float:
        return 0.3

    async def get_response(self, prompt: str) -> str:
        return f"legacy:{prompt}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return f"legacy:{prompt}"


class _ThreeKwargCloneClient(LLMClientBase):
    """Implements the clone protocol without the newer diagnostic ``step=``.

    Any client written against the three sampling kwargs — including
    ``RecordingLLMClient``'s inner clients and third-party ones — must keep
    working; passing ``step=`` unconditionally would turn them into a
    ``TypeError`` at step-execution time.
    """

    def __init__(self, *, model: str = "vendor-a", temperature: float = 0.5) -> None:
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    def clone_with_overrides(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "_ThreeKwargCloneClient":
        return _ThreeKwargCloneClient(
            model=model if model is not None else self._model,
            temperature=temperature if temperature is not None else self._temperature,
        )

    async def get_response(self, prompt: str) -> str:
        return prompt

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return prompt


class _ClosingStub:
    """SDK handle double recording that it was awaited closed."""

    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def _anthropic(**overrides: Any) -> AnthropicClient:
    """An ``AnthropicClient`` that never touches the (uninstalled) SDK."""
    config = AnthropicClientConfig(
        **{
            "api_key": "k",
            "model": "claude-base",
            "temperature": 0.1,
            "max_tokens": 100,
            **overrides,
        }
    )
    return AnthropicClient(config, sdk_client=_ClosingStub())


def _context(api: Any) -> ReasoningContext:
    return ReasoningContext(outer_context="x", api=api)


# ---------------------------------------------------------------------------
# The override reaches a non-OpenAI client
# ---------------------------------------------------------------------------


class TestOverrideReachesNonOpenAIClients:
    def test_anthropic_step_override_is_applied(self) -> None:
        """The exact symptom: an AnthropicClient asked for a different model
        used to hand back the unmodified base client."""
        base = _anthropic()
        ctx = _context(base)

        client = ctx.get_llm_client_for_step(
            LLMStepConfig(model="claude-other", temperature=0.9, max_tokens=999)
        )

        assert client is not base
        assert client.model_name == "claude-other"
        assert client.temperature == 0.9
        assert client.max_tokens == 999

    def test_base_client_is_left_untouched_by_the_override(self) -> None:
        """Cloning, not mutating: the next un-overridden step must still get
        the context's own configuration."""
        base = _anthropic()
        ctx = _context(base)

        ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"))

        assert base.model_name == "claude-base"
        assert base.temperature == 0.1
        assert ctx.get_llm_client_for_step(None) is base

    def test_partial_override_keeps_the_rest_of_the_provider_config(self) -> None:
        """Only the overridden knobs change — API key, timeout, thinking budget
        and the rest of the provider config ride along."""
        base = _anthropic(thinking_budget=2048, cache_system=True, timeout=42.0)
        ctx = _context(base)

        client = ctx.get_llm_client_for_step(LLMStepConfig(temperature=0.9))

        assert client.config.temperature == 0.9
        assert client.config.thinking_budget == 2048
        assert client.config.cache_system is True
        assert client.config.timeout == 42.0
        assert client.config.api_key == "k"
        assert client.config.model == "claude-base"

    def test_temperature_zero_is_a_real_override(self) -> None:
        """Greedy decoding is the most common setting an optimiser lands on;
        a truthiness check would drop it."""
        ctx = _context(_anthropic(temperature=0.7))

        client = ctx.get_llm_client_for_step(LLMStepConfig(temperature=0.0))

        assert client.temperature == 0.0

    def test_anthropic_clone_shares_the_injected_sdk_handle(self) -> None:
        """The clone must not open a second connection pool — and an injected
        test double has to survive, or a stubbed client would reach for the
        real SDK the moment a step overrode the model."""
        base = _anthropic()
        client = _context(base).get_llm_client_for_step(LLMStepConfig(model="claude-other"))

        assert client.client is base.client

    def test_duck_typed_client_is_not_required_to_subclass_llmclientbase(self) -> None:
        """``ReasoningContext.api`` is typed ``Any`` and documented as
        "LLMClientBase or compatible"; the protocol is duck-typed to match."""

        class _Compatible:
            def __init__(self, model: str) -> None:
                self.model_name = model

            def clone_with_overrides(
                self,
                *,
                model: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                step: Optional[str] = None,
            ) -> "_Compatible":
                return _Compatible(model or self.model_name)

        client = _context(_Compatible("a")).get_llm_client_for_step(LLMStepConfig(model="b"))
        assert client.model_name == "b"

    def test_openai_path_still_works(self) -> None:
        """The class that used to be special-cased must keep behaving."""
        base = OpenAICompatibleClient(OpenAIClientConfig(api_key="k", model="gpt-base"))
        ctx = _context(base)

        client = ctx.get_llm_client_for_step(
            LLMStepConfig(model="gpt-other", temperature=0.9, max_tokens=99)
        )

        assert isinstance(client, OpenAICompatibleClient)
        assert client.config.model == "gpt-other"
        assert client.config.temperature == 0.9
        assert client.config.max_tokens == 99

    def test_openai_override_keeps_the_retry_policy(self) -> None:
        """The hand-written OpenAI clone listed the config fields one by one and
        never learned about ``retry_policy``, so an overridden step quietly lost
        its status-aware retries and went back to retrying 401s."""
        policy = RetryPolicy(max_attempts=5, backoff="constant", initial_delay_s=0.25)
        base = OpenAICompatibleClient(
            OpenAIClientConfig(api_key="k", model="gpt-base", retry_policy=policy)
        )

        client = _context(base).get_llm_client_for_step(LLMStepConfig(model="gpt-other"))

        assert client.config.retry_policy == policy


# ---------------------------------------------------------------------------
# Loud degradation when cloning is genuinely impossible
# ---------------------------------------------------------------------------


class TestLoudDegradation:
    def test_uncloneable_client_warns_naming_step_and_override(self) -> None:
        ctx = _context(_UncloneableClient())

        with pytest.warns(RuntimeWarning) as caught:
            ctx.get_llm_client_for_step(
                LLMStepConfig(model="vendor-b", temperature=0.9), step="rank_candidates"
            )

        message = str(caught[0].message)
        assert "rank_candidates" in message
        assert "model='vendor-b'" in message
        assert "temperature=0.9" in message
        assert "_UncloneableClient" in message

    def test_warning_names_the_configuration_the_step_actually_gets(self) -> None:
        """A reader has to be able to tell what did run, not only what didn't."""
        ctx = _context(_UncloneableClient())

        with pytest.warns(RuntimeWarning) as caught:
            ctx.get_llm_client_for_step(LLMStepConfig(model="vendor-b"))

        message = str(caught[0].message)
        assert "model='legacy-model'" in message
        assert "temperature=0.3" in message
        assert "an unnamed step" in message

    def test_unset_overrides_are_not_listed_as_dropped(self) -> None:
        ctx = _context(_UncloneableClient())

        with pytest.warns(RuntimeWarning) as caught:
            ctx.get_llm_client_for_step(LLMStepConfig(max_tokens=256), step="s")

        head = str(caught[0].message).split(") requested")[0]
        assert "max_tokens=256" in head
        assert "model=" not in head
        assert "temperature=" not in head

    def test_the_step_still_runs_on_the_default_client(self) -> None:
        """Loud, not fatal: a dropped override must not take the chain down."""
        base = _UncloneableClient()
        ctx = _context(base)

        with pytest.warns(RuntimeWarning):
            client = ctx.get_llm_client_for_step(LLMStepConfig(model="vendor-b"))

        assert client is base

    def test_a_refused_clone_is_not_cached_as_a_success(self) -> None:
        """Caching the fallback under the override's key would make the next
        step with that override silently reuse it — and warn only once."""
        ctx = _context(_UncloneableClient())

        with pytest.warns(RuntimeWarning):
            ctx.get_llm_client_for_step(LLMStepConfig(model="vendor-b"))
        with pytest.warns(RuntimeWarning):
            ctx.get_llm_client_for_step(LLMStepConfig(model="vendor-b"))

        assert ctx._llm_client_cache == {}

    def test_override_rejected_by_the_provider_config_warns_instead_of_crashing(
        self,
    ) -> None:
        """``LLMStepConfig`` allows temperature up to 2.0; Anthropic's config
        caps it at 1.0. Re-validating catches that here — the run continues on
        the base client with a warning rather than dying mid-chain or, worse,
        installing a value the API will reject."""
        base = _anthropic(temperature=0.1)
        ctx = _context(base)

        with pytest.warns(RuntimeWarning) as caught:
            client = ctx.get_llm_client_for_step(LLMStepConfig(temperature=1.5), step="hot")

        assert client is base
        messages = " | ".join(str(w.message) for w in caught)
        assert "temperature=1.5" in messages
        assert "AnthropicClient" in messages

    def test_no_warning_on_the_happy_path(self) -> None:
        """The warning has to stay meaningful; a clone that worked is silent."""
        ctx = _context(_anthropic())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"), step="s")

        assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ---------------------------------------------------------------------------
# The clone cache
# ---------------------------------------------------------------------------


class TestOverrideCache:
    def test_same_override_reuses_one_client(self) -> None:
        base = _anthropic()
        ctx = _context(base)

        first = ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"))
        second = ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"))

        assert first is second
        assert first is not base

    def test_temperature_zero_does_not_collide_with_unset_temperature(self) -> None:
        """The old cache key was ``f"{temperature or ''}"``, which rendered both
        ``0.0`` and ``None`` as the empty string: a greedy step and a
        default-temperature step sharing a model got each other's client."""
        ctx = _context(_anthropic(temperature=0.7))

        greedy = ctx.get_llm_client_for_step(LLMStepConfig(model="m", temperature=0.0))
        inherited = ctx.get_llm_client_for_step(LLMStepConfig(model="m"))

        assert greedy is not inherited
        assert greedy.temperature == 0.0
        assert inherited.temperature == 0.7

    def test_max_tokens_zero_does_not_collide_with_unset_max_tokens(self) -> None:
        """Same shape of collision on the other numeric knob."""
        base = OpenAICompatibleClient(OpenAIClientConfig(api_key="k", model="gpt-base"))
        ctx = _context(base)

        capped = ctx.get_llm_client_for_step(LLMStepConfig(model="m", max_tokens=0))
        uncapped = ctx.get_llm_client_for_step(LLMStepConfig(model="m"))

        assert capped is not uncapped
        assert capped.max_tokens == 0
        assert uncapped.max_tokens is None

    def test_swapping_the_context_client_does_not_serve_a_stale_clone(self) -> None:
        """``api`` can be reassigned on a live context; a cache keyed only on the
        override would hand back a clone of the previous client."""
        ctx = _context(_anthropic())
        assert ctx.get_llm_client_for_step(LLMStepConfig(model="m")).model_name == "m"

        ctx.api = OpenAICompatibleClient(OpenAIClientConfig(api_key="k", model="gpt-base"))
        ctx._llm_client = ctx.api

        assert isinstance(ctx.get_llm_client_for_step(LLMStepConfig(model="m")), OpenAICompatibleClient)


# ---------------------------------------------------------------------------
# Teardown goes through the same protocol
# ---------------------------------------------------------------------------


class TestCloseGoesThroughTheProtocol:
    async def test_non_openai_client_is_closed(self) -> None:
        """``close()`` had the same isinstance shape as the override resolver:
        only ``OpenAICompatibleClient`` was ever torn down."""
        sdk = _ClosingStub()
        base = AnthropicClient(AnthropicClientConfig(api_key="k"), sdk_client=sdk)
        ctx = _context(base)

        await ctx.close()

        assert sdk.closed == 1

    async def test_cached_override_clones_are_closed_too(self) -> None:
        """The cache can now hold non-OpenAI clones; leaving them open would
        leak one connection pool per distinct per-step override."""
        ctx = _context(_anthropic())
        clone = ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"))
        sdk = clone.client

        await ctx.close()

        assert sdk.closed >= 1
        assert ctx._llm_client_cache == {}

    async def test_close_is_a_noop_for_clients_that_hold_nothing(self) -> None:
        """Every mock client in the suite inherits ``close()``; it must not
        require them to implement anything."""
        await _context(_UncloneableClient()).close()

    async def test_anthropic_close_tolerates_a_handle_without_close(self) -> None:
        """Teardown must not raise on a bare test double."""
        client = AnthropicClient(AnthropicClientConfig(api_key="k"), sdk_client=object())
        await client.close()

    async def test_anthropic_close_drops_the_handle_for_reuse(self) -> None:
        """A context is documented as reusable after ``close()``."""
        sdk = _ClosingStub()
        client = AnthropicClient(AnthropicClientConfig(api_key="k"), sdk_client=sdk)

        await client.close()
        await client.close()

        assert sdk.closed == 1


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


class TestProtocolHelpers:
    def test_step_kwarg_is_offered_not_imposed(self) -> None:
        """An implementation taking only the three sampling kwargs must not be
        broken by the diagnostic ``step=``."""
        clone = clone_llm_client_with_overrides(
            _ThreeKwargCloneClient(), model="vendor-b", step="rank"
        )

        assert clone is not None
        assert clone.model_name == "vendor-b"

    def test_step_kwarg_is_forwarded_when_accepted(self) -> None:
        seen: dict[str, Any] = {}

        class _RecordsStep(_ThreeKwargCloneClient):
            def clone_with_overrides(  # type: ignore[override]
                self,
                *,
                model: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                step: Optional[str] = None,
            ) -> "_RecordsStep":
                seen["step"] = step
                return self

        clone_llm_client_with_overrides(_RecordsStep(), model="x", step="rank")
        assert seen == {"step": "rank"}

    def test_client_without_the_protocol_returns_none(self) -> None:
        assert clone_llm_client_with_overrides(object(), model="x") is None

    def test_no_requested_override_returns_the_same_client(self) -> None:
        client = _anthropic()
        assert client.clone_with_overrides() is client

    def test_describe_overrides_lists_only_what_was_set(self) -> None:
        assert describe_llm_overrides(model="m", max_tokens=8) == "model='m', max_tokens=8"
        assert describe_llm_overrides(temperature=0.0) == "temperature=0.0"
        assert describe_llm_overrides() == "<no overrides set>"
