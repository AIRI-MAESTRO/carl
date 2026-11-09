"""Per-step LLM overrides under the record/replay wrappers.

A chain that evolves per-step ``model`` / ``temperature`` / ``max_tokens``
(the reason EvoC drives CARL) records a cassette and replays it later. If
the wrappers ignore a step's override, the cassette claims to be a record
of the run while actually holding the *un-overridden* step's answer — and
replay silently reproduces the wrong client for exactly the steps whose
configuration is under search. These tests pin the two halves of that:
the clone protocol (:meth:`clone_with_overrides`) keeps an overridden
step's entries separate end-to-end, and a client that cannot clone makes
noise instead of corrupting the tape.

Deliberately exercised with a non-``OpenAICompatibleClient`` client type:
special-casing one concrete class is what let this bug exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from mmar_carl import PlayingLLMClient, RecordingLLMClient
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.record_replay import _cassette_key


class _NonOpenAIClient(LLMClientBase):
    """A client that is not ``OpenAICompatibleClient`` and *can* clone.

    Its answers embed the sampling configuration so a replayed response
    proves which configuration produced it.
    """

    def __init__(
        self,
        *,
        model: str = "vendor-a",
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> Optional[int]:
        return self._max_tokens

    def clone_with_overrides(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "_NonOpenAIClient":
        return _NonOpenAIClient(
            model=model if model is not None else self._model,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

    async def get_response(self, prompt: str) -> str:
        self.call_count += 1
        return f"{self._model}@{self._temperature}/{self._max_tokens}:{prompt}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class _UncloneableClient(LLMClientBase):
    """A client with no clone support — e.g. a hand-rolled test double, or
    any ``LLMClientBase`` subclass predating the clone protocol."""

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


@pytest.fixture
def cassette(tmp_path: Path) -> Path:
    return tmp_path / "cassette.jsonl"


def _entries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# The override survives record -> replay
# ---------------------------------------------------------------------------


class TestOverrideSurvivesRoundTrip:
    async def test_model_override_records_and_replays_separately(
        self, cassette: Path
    ) -> None:
        """Two steps share a prompt but differ in `model`. Without the clone
        protocol both answers land on one cassette key, and replaying the
        overridden step returns the *other* step's answer."""
        base = RecordingLLMClient(_NonOpenAIClient(model="vendor-a"), cassette)
        overridden = base.clone_with_overrides(model="vendor-b", step="rank")
        assert overridden is not None

        assert await base.get_response("q") == "vendor-a@0.5/None:q"
        assert await overridden.get_response("q") == "vendor-b@0.5/None:q"

        # Same prompt, same method — two entries, not one overwrite.
        assert len(_entries(cassette)) == 2

        play_base = PlayingLLMClient(cassette, model="vendor-a", temperature=0.5)
        play_over = play_base.clone_with_overrides(model="vendor-b")
        assert await play_base.get_response("q") == "vendor-a@0.5/None:q"
        assert await play_over.get_response("q") == "vendor-b@0.5/None:q"

    async def test_clone_talks_to_the_overridden_client_not_the_original(
        self, cassette: Path
    ) -> None:
        """The override must reach the wrapped client, not just the wrapper's
        reported metadata — otherwise recording calls the old model while
        labelling the entry with the new one."""
        real = _NonOpenAIClient(model="vendor-a", temperature=0.5)
        rec = RecordingLLMClient(real, cassette)
        clone = rec.clone_with_overrides(temperature=0.9, step="critique")
        assert clone is not None

        await clone.get_response("q")

        # The original client was never called; its twin answered at 0.9.
        assert real.call_count == 0
        entry = _entries(cassette)[0]
        assert entry["response"] == "vendor-a@0.9/None:q"
        assert entry["temperature"] == 0.9

    async def test_temperature_zero_override_is_applied_not_treated_as_unset(
        self, cassette: Path
    ) -> None:
        """`temperature=0.0` is a real override (greedy decoding), the most
        common one an optimiser lands on; a truthiness check would drop it."""
        rec = RecordingLLMClient(_NonOpenAIClient(temperature=0.5), cassette)
        clone = rec.clone_with_overrides(temperature=0.0, step="s")
        assert clone is not None
        assert clone.temperature == 0.0

        await clone.get_response("q")
        play = PlayingLLMClient(cassette).clone_with_overrides(temperature=0.0)
        assert await play.get_response("q") == "vendor-a@0.0/None:q"

    async def test_clone_appends_to_the_same_cassette(self, cassette: Path) -> None:
        """The twin must not truncate the tape: entries recorded before the
        override are still there afterwards."""
        rec = RecordingLLMClient(_NonOpenAIClient(), cassette)
        await rec.get_response("first")
        clone = rec.clone_with_overrides(model="vendor-b", step="s")
        assert clone is not None
        await clone.get_response("second")

        assert clone.cassette_path == cassette
        prompts = {e["prompt"] for e in _entries(cassette)}
        assert prompts == {"first", "second"}


# ---------------------------------------------------------------------------
# Loud degradation when cloning is genuinely impossible
# ---------------------------------------------------------------------------


class TestLoudDegradation:
    def test_uncloneable_client_warns_naming_step_and_override(
        self, cassette: Path
    ) -> None:
        """Silence is the defect: a dropped override with no warning leaves a
        cassette that looks valid and reproduces the wrong configuration."""
        rec = RecordingLLMClient(_UncloneableClient(), cassette)

        with pytest.warns(RuntimeWarning) as caught:
            result = rec.clone_with_overrides(
                model="vendor-b", temperature=0.9, step="rank_candidates"
            )

        assert result is None
        message = str(caught[0].message)
        assert "rank_candidates" in message
        assert "model='vendor-b'" in message
        assert "temperature=0.9" in message
        assert "_UncloneableClient" in message
        assert str(cassette) in message

    def test_warning_names_the_configuration_the_entries_would_get(
        self, cassette: Path
    ) -> None:
        """The warning has to say what the cassette *will* contain, otherwise a
        reader cannot tell whether an existing tape is still trustworthy."""
        rec = RecordingLLMClient(_UncloneableClient(), cassette)

        with pytest.warns(RuntimeWarning) as caught:
            rec.clone_with_overrides(model="vendor-b", step="s")

        message = str(caught[0].message)
        assert "model='legacy-model'" in message
        assert "temperature=0.3" in message

    def test_unset_overrides_are_not_listed_as_dropped(self, cassette: Path) -> None:
        """Only the fields actually requested are reported, so the warning
        points at the knob the caller set rather than at every knob."""
        rec = RecordingLLMClient(_UncloneableClient(), cassette)

        with pytest.warns(RuntimeWarning) as caught:
            rec.clone_with_overrides(max_tokens=256, step="s")

        message = str(caught[0].message)
        head = message.split(") requested")[0]
        assert "max_tokens=256" in head
        assert "model=" not in head
        assert "temperature=" not in head

    async def test_recording_still_works_after_a_refused_clone(
        self, cassette: Path
    ) -> None:
        """A refused clone must not leave the recorder unusable — the caller is
        expected to fall back to the un-overridden client and carry on."""
        rec = RecordingLLMClient(_UncloneableClient(), cassette)
        with pytest.warns(RuntimeWarning):
            assert rec.clone_with_overrides(model="x", step="s") is None

        assert await rec.get_response("q") == "legacy:q"
        assert rec.cassette_size == 1


# ---------------------------------------------------------------------------
# max_tokens is part of the cassette identity
# ---------------------------------------------------------------------------


class TestMaxTokensIdentity:
    async def test_steps_differing_only_in_max_tokens_do_not_share_an_entry(
        self, cassette: Path
    ) -> None:
        """A truncation-budget sweep (same prompt, same model, different
        max_tokens) used to collide on one key, so replay served the first
        step's answer to every budget in the sweep."""
        rec = RecordingLLMClient(_NonOpenAIClient(max_tokens=256), cassette)
        wide = rec.clone_with_overrides(max_tokens=4096, step="expand")
        assert wide is not None

        assert await rec.get_response("q") == "vendor-a@0.5/256:q"
        assert await wide.get_response("q") == "vendor-a@0.5/4096:q"

        keys = {e["key"] for e in _entries(cassette)}
        assert len(keys) == 2

        play = PlayingLLMClient(cassette, model="vendor-a", temperature=0.5)
        assert await play.clone_with_overrides(max_tokens=256).get_response("q") == (
            "vendor-a@0.5/256:q"
        )
        assert await play.clone_with_overrides(max_tokens=4096).get_response("q") == (
            "vendor-a@0.5/4096:q"
        )

    async def test_max_tokens_is_persisted_and_replayed(self, cassette: Path) -> None:
        """`PlayingLLMClient.max_tokens` feeds cost accounting and any
        downstream budget check; it has to come back off the tape."""
        rec = RecordingLLMClient(_NonOpenAIClient(max_tokens=1024), cassette)
        await rec.get_response("q")

        assert _entries(cassette)[0]["max_tokens"] == 1024
        assert PlayingLLMClient(cassette).max_tokens == 1024

    async def test_pre_max_tokens_entries_stay_replayable_in_a_mixed_cassette(
        self, cassette: Path
    ) -> None:
        """Cassettes are append-only by default, so upgrading CARL mid-project
        yields a tape whose old entries are keyed without max_tokens and whose
        new ones are keyed with it. Both must still replay."""
        legacy_key = _cassette_key(
            "get_response", prompt="old", model="vendor-a", temperature=0.5
        )
        cassette.write_text(
            json.dumps(
                {
                    "key": legacy_key,
                    "method": "get_response",
                    "prompt": "old",
                    "response": "legacy-answer",
                    "model": "vendor-a",
                    "temperature": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        rec = RecordingLLMClient(_NonOpenAIClient(max_tokens=512), cassette)
        await rec.get_response("new")

        play = PlayingLLMClient(cassette)
        # First entry has no max_tokens, so the default comes from the newer one.
        assert play.max_tokens == 512
        assert await play.get_response("new") == "vendor-a@0.5/512:new"
        assert await play.get_response("old") == "legacy-answer"

    async def test_miss_still_raises_after_both_key_shapes_are_tried(
        self, cassette: Path
    ) -> None:
        """The legacy-key fallback must not turn a genuine miss into a silent
        wrong answer — an unrecorded prompt still fails loudly."""
        from mmar_carl import CassetteMissError

        rec = RecordingLLMClient(_NonOpenAIClient(max_tokens=512), cassette)
        await rec.get_response("recorded")

        play = PlayingLLMClient(cassette)
        with pytest.raises(CassetteMissError) as exc:
            await play.get_response("never-recorded")
        assert "never-recorded" in str(exc.value)
