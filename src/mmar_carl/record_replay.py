"""Record-and-replay LLM client wrappers.

Two thin LLMClientBase subclasses that mirror the ergonomics of
``pytest-vcr``:

* :class:`RecordingLLMClient` wraps a real client, forwards every call,
  and appends a JSON-Lines cassette entry capturing the prompt + the
  client's response (plus token usage and tool-call structures when
  applicable). Cassettes are append-only by default — pass
  ``overwrite=True`` to start fresh.

* :class:`PlayingLLMClient` reads a cassette and replays responses by
  key lookup. No API calls. A miss raises :class:`CassetteMissError`
  with a prompt preview so users can re-record.

The cassette key is a stable SHA-256 of
``(method, prompt-or-args, model, temperature, max_tokens)`` — calls
under different sampling configurations don't collide. Each line of the
cassette is a single JSON object, so cassettes are human-readable
and ``git diff``-friendly.

Per-step LLM overrides
----------------------
``LLMStepConfig`` lets one step override ``model`` / ``temperature`` /
``max_tokens``. Both wrappers implement :meth:`clone_with_overrides`,
the clone protocol a dispatcher uses to obtain a twin that records into
(or replays from) the same cassette under the *overridden* identity.
Because the identity is part of the cassette key, the overridden step's
interactions stay separate from the un-overridden ones.

Known upstream gap, measured 2026-07-27 on branch ``evo-metadata``:
``ReasoningContext.get_llm_client_for_step`` still picks the overridden
client with ``isinstance(self._llm_client, OpenAICompatibleClient)`` and
returns ``self.llm_client`` unchanged for every other type, so it never
consults the protocol below. The gap is *not* specific to these
wrappers — a plain ``AnthropicClient`` installed as ``context.api``
loses its per-step overrides the same way, with no warning::

    ctx = ReasoningContext(outer_context="x", api=AnthropicClient(cfg))
    c = ctx.get_llm_client_for_step(LLMStepConfig(model="claude-other"))
    c.model_name  # -> cfg.model, i.e. the override vanished

Closing it needs ``LLMClientBase.clone_with_overrides`` plus a
dispatch site in ``get_llm_client_for_step``; neither lives in this
module. Until then :meth:`RecordingLLMClient.clone_with_overrides`
emits a ``RuntimeWarning`` instead of mislabelling the cassette,
because a cassette that quietly stops matching the configuration it
claims to record is worse than a loud refusal.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Optional

from .models.llm_client_base import ChatMessage, LLMClientBase


class CassetteMissError(KeyError):
    """Raised when :class:`PlayingLLMClient` is asked for a key not in the cassette."""

    pass


def _stable_json(obj: Any) -> str:
    """JSON-serialise with stable key order for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _cassette_key(
    method: str,
    *,
    prompt: Optional[str] = None,
    messages: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Build a stable SHA-256 key for one LLM interaction.

    None-valued fields are dropped so that adding new optional fields
    later doesn't invalidate previously-recorded cassette entries.
    ``max_tokens`` was added to the identity after two steps that
    differed *only* in ``max_tokens`` were found to share one cassette
    entry; the None-dropping rule keeps every pre-existing cassette
    recorded by a client without a max-tokens cap on its original key.
    """
    payload = {
        "method": method,
        "prompt": prompt,
        "messages": messages,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "tools": tools,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _clone_client(
    client: LLMClientBase,
    *,
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Optional[LLMClientBase]:
    """Ask *client* for a copy of itself with the overrides applied.

    Dispatch is duck-typed on ``clone_with_overrides`` rather than
    ``isinstance``: the only reason a per-step override ever reached
    production unhonoured is that the resolver in
    ``ReasoningContext.get_llm_client_for_step`` names one concrete
    class, so every client type added afterwards (``AnthropicClient``,
    these wrappers) silently fell through it.

    Returns ``None`` when *client* does not implement the protocol or
    declines to clone. Callers must treat ``None`` as "the override
    cannot be honoured" and say so out loud.
    """
    hook = getattr(client, "clone_with_overrides", None)
    if not callable(hook):
        return None
    return hook(model=model, temperature=temperature, max_tokens=max_tokens)


def _describe_overrides(
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> str:
    """Render the set overrides for a diagnostic message, e.g. ``model='x'``."""
    parts = [
        f"{name}={value!r}"
        for name, value in (
            ("model", model),
            ("temperature", temperature),
            ("max_tokens", max_tokens),
        )
        if value is not None
    ]
    return ", ".join(parts) if parts else "<no overrides set>"


class RecordingLLMClient(LLMClientBase):
    """Wraps a real LLM client, forwarding calls and appending each
    (prompt, response) pair to a JSON-Lines cassette.

    Args:
        real_client: The underlying client (e.g.
            :class:`~mmar_carl.llm.OpenAICompatibleClient`).
        cassette_path: Path to the JSONL cassette file. Parent
            directories are created on first write.
        overwrite: If ``True`` and the cassette already exists, it is
            removed before the first write. Default ``False`` —
            cassettes are append-only, so a chain re-run can capture
            new interactions without losing old ones.
    """

    def __init__(
        self,
        real_client: LLMClientBase,
        cassette_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self._real = real_client
        self._cassette_path = Path(cassette_path)
        if overwrite and self._cassette_path.exists():
            self._cassette_path.unlink()
        self._cassette_path.parent.mkdir(parents=True, exist_ok=True)

    # --- introspection proxies ---
    @property
    def model_name(self) -> Optional[str]:
        return self._real.model_name

    @property
    def temperature(self) -> Optional[float]:
        return self._real.temperature

    @property
    def max_tokens(self) -> Optional[int]:
        return self._real.max_tokens

    @property
    def supports_streaming(self) -> bool:
        # Streaming is not captured — users can fall back to non-streaming
        # methods, which are all supported.
        return False

    @property
    def cassette_path(self) -> Path:
        """Path to the on-disk cassette file."""
        return self._cassette_path

    @property
    def cassette_size(self) -> int:
        """Number of entries recorded so far (counts file lines)."""
        if not self._cassette_path.exists():
            return 0
        with self._cassette_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _identity(self) -> dict[str, Any]:
        """The sampling configuration that identifies a cassette entry.

        Returned as one dict so the cassette *key* and the cassette
        *entry* are built from the same fields at all six call sites.
        Before this existed the two were spelled out separately in each
        method, and ``max_tokens`` was in neither — a step recorded at
        ``max_tokens=256`` and one at ``4096`` produced the same key.
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def clone_with_overrides(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        step: Optional[str] = None,
    ) -> Optional["RecordingLLMClient"]:
        """Return a recorder for the same cassette under an overridden config.

        The overrides are applied to the *wrapped* client (which must
        itself implement the protocol), so the twin really talks to the
        overridden model and the entries it writes carry the overridden
        identity.

        Args:
            model / temperature / max_tokens: overrides to apply; ``None``
                means "keep the wrapped client's value".
            step: diagnostic only — the step whose config is being
                applied, used to name the offender in the warning below.
                Not part of the clone protocol; callers that have no step
                name may omit it.

        Returns:
            A new :class:`RecordingLLMClient` appending to the same
            cassette, or ``None`` when the wrapped client cannot clone —
            in which case a ``RuntimeWarning`` is raised first.
        """
        inner = _clone_client(
            self._real, model=model, temperature=temperature, max_tokens=max_tokens
        )
        if inner is None:
            where = f"step {step!r}" if step else "an unnamed step"
            warnings.warn(
                f"Per-step LLM override ({_describe_overrides(model, temperature, max_tokens)}) "
                f"requested for {where} was DROPPED: {type(self._real).__name__} does not "
                f"implement clone_with_overrides(), so RecordingLLMClient cannot record under "
                f"the overridden configuration. Interactions for that step would be filed in "
                f"{self._cassette_path} under model={self.model_name!r}, "
                f"temperature={self.temperature!r}, max_tokens={self.max_tokens!r} and collide "
                f"with the un-overridden step's entries — the cassette would stop being a "
                f"faithful record of the run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        # overwrite=False: the twin must append to the same cassette as the
        # original, not truncate the entries recorded before the override.
        return RecordingLLMClient(inner, self._cassette_path)

    def _write(self, entry: dict[str, Any]) -> None:
        with self._cassette_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def get_response(self, prompt: str) -> str:
        result = await self._real.get_response(prompt)
        ident = self._identity()
        key = _cassette_key("get_response", prompt=prompt, **ident)
        self._write({
            "key": key, "method": "get_response",
            "prompt": prompt, "response": result, **ident,
        })
        return result

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        result = await self._real.get_response_with_retries(prompt, retries)
        ident = self._identity()
        key = _cassette_key("get_response_with_retries", prompt=prompt, **ident)
        self._write({
            "key": key, "method": "get_response_with_retries",
            "prompt": prompt, "response": result, **ident,
        })
        return result

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        result = await self._real.get_response_with_system(system_prompt, user_prompt, retries)
        ident = self._identity()
        key = _cassette_key(
            "get_response_with_system",
            system_prompt=system_prompt, user_prompt=user_prompt, **ident,
        )
        self._write({
            "key": key, "method": "get_response_with_system",
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "response": result, **ident,
        })
        return result

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        response, usage = await self._real.get_response_with_usage(prompt, retries)
        ident = self._identity()
        key = _cassette_key("get_response_with_usage", prompt=prompt, **ident)
        self._write({
            "key": key, "method": "get_response_with_usage",
            "prompt": prompt, "response": response, "usage": dict(usage), **ident,
        })
        return response, usage

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        response, usage = await self._real.get_response_with_messages(messages, retries)
        msg_dicts = [m.to_dict() for m in messages]
        ident = self._identity()
        key = _cassette_key("get_response_with_messages", messages=msg_dicts, **ident)
        self._write({
            "key": key, "method": "get_response_with_messages",
            "messages": msg_dicts, "response": response, "usage": dict(usage), **ident,
        })
        return response, usage

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        content, tool_calls, _ = await self.get_response_with_tools_and_usage(
            system_prompt, user_prompt, tools, messages
        )
        return content, tool_calls

    async def get_response_with_tools_and_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        content, tool_calls, usage = await self._real.get_response_with_tools_and_usage(
            system_prompt, user_prompt, tools, messages
        )
        ident = self._identity()
        key = _cassette_key(
            "get_response_with_tools",
            system_prompt=system_prompt, user_prompt=user_prompt,
            messages=messages, tools=tools, **ident,
        )
        self._write({
            "key": key, "method": "get_response_with_tools",
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "messages": messages, "tools": tools,
            "response": content, "tool_calls": tool_calls,
            "usage": dict(usage),
            **ident,
        })
        return content, tool_calls, usage


class PlayingLLMClient(LLMClientBase):
    """Replays a cassette recorded by :class:`RecordingLLMClient`.

    Zero API calls are made. Construction loads the JSONL into an
    in-memory dict keyed by cassette key. A miss raises
    :class:`CassetteMissError` with the method name and a prompt
    preview so the caller can debug or re-record.

    Args:
        cassette_path: Path to the JSONL cassette.
        model: Optional override for the model name returned by
            :attr:`model_name`. Defaults to the model recorded on the
            first cassette entry.
        temperature: Optional override for :attr:`temperature`.
            Defaults to the value recorded on the first cassette entry.
        max_tokens: Optional override for :attr:`max_tokens`.
            Defaults to the value recorded on the first cassette entry.
    """

    def __init__(
        self,
        cassette_path: str | Path,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._cassette_path = Path(cassette_path)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._cassette_path.exists():
            raise FileNotFoundError(f"Cassette not found: {self._cassette_path}")
        with self._cassette_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._entries[entry["key"]] = entry
                # Default model/temperature/max_tokens from first entry if user didn't set
                if self._model is None:
                    self._model = entry.get("model")
                if self._temperature is None:
                    self._temperature = entry.get("temperature")
                if self._max_tokens is None:
                    self._max_tokens = entry.get("max_tokens")

    @property
    def cassette_path(self) -> Path:
        return self._cassette_path

    @property
    def cassette_size(self) -> int:
        return len(self._entries)

    @property
    def model_name(self) -> Optional[str]:
        return self._model

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    @property
    def max_tokens(self) -> Optional[int]:
        return self._max_tokens

    @property
    def supports_streaming(self) -> bool:
        return False

    def clone_with_overrides(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        step: Optional[str] = None,
    ) -> "PlayingLLMClient":
        """Return a player that looks up the same cassette under an overridden config.

        Replay is pure key lookup, so honouring an override means
        changing the identity the keys are built from — a step recorded
        under ``model="B"`` is only replayable by a player whose
        ``model`` is ``"B"``.

        Args:
            model / temperature / max_tokens: overrides to apply; ``None``
                means "keep this player's value".
            step: accepted for symmetry with
                :meth:`RecordingLLMClient.clone_with_overrides` and
                ignored — a player can always clone, so it never has to
                report a dropped override.
        """
        del step  # diagnostic-only; replay never fails to clone
        return PlayingLLMClient(
            self._cassette_path,
            model=model if model is not None else self._model,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

    def _keys(self, method: str, **call_args: Any) -> list[str]:
        """Candidate cassette keys for one call, most specific first.

        The second candidate omits ``max_tokens``, which joined the key
        only after cassettes were already in use. Cassettes are
        append-only by default, so an upgraded run writes max-tokens-bearing
        entries into a file whose older entries have none; without this
        fallback every pre-upgrade entry would become a CassetteMissError.
        """
        primary = _cassette_key(
            method,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **call_args,
        )
        if self._max_tokens is None:
            return [primary]
        legacy = _cassette_key(
            method, model=self._model, temperature=self._temperature, **call_args
        )
        return [primary, legacy]

    def _lookup(self, keys: list[str], method: str, preview: str) -> dict[str, Any]:
        for key in keys:
            entry = self._entries.get(key)
            if entry is not None:
                return entry
        short = preview if len(preview) <= 80 else preview[:77] + "..."
        raise CassetteMissError(
            f"Cassette miss for method='{method}' "
            f"(key={keys[0][:12]}…) preview={short!r}. "
            f"Re-record with RecordingLLMClient to capture this interaction."
        )

    async def get_response(self, prompt: str) -> str:
        keys = self._keys("get_response", prompt=prompt)
        return self._lookup(keys, "get_response", prompt)["response"]

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        keys = self._keys("get_response_with_retries", prompt=prompt)
        return self._lookup(keys, "get_response_with_retries", prompt)["response"]

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        keys = self._keys(
            "get_response_with_system",
            system_prompt=system_prompt, user_prompt=user_prompt,
        )
        return self._lookup(keys, "get_response_with_system", user_prompt)["response"]

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        keys = self._keys("get_response_with_usage", prompt=prompt)
        entry = self._lookup(keys, "get_response_with_usage", prompt)
        return entry["response"], dict(entry.get("usage", {}))

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        msg_dicts = [m.to_dict() for m in messages]
        keys = self._keys("get_response_with_messages", messages=msg_dicts)
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        entry = self._lookup(keys, "get_response_with_messages", last_user)
        return entry["response"], dict(entry.get("usage", {}))

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        content, tool_calls, _ = await self.get_response_with_tools_and_usage(
            system_prompt, user_prompt, tools, messages
        )
        return content, tool_calls

    async def get_response_with_tools_and_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        keys = self._keys(
            "get_response_with_tools",
            system_prompt=system_prompt, user_prompt=user_prompt,
            messages=messages, tools=tools,
        )
        entry = self._lookup(keys, "get_response_with_tools", user_prompt)
        return (
            entry["response"],
            list(entry.get("tool_calls", [])),
            dict(entry.get("usage", {})),
        )
