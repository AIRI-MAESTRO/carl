"""Legacy chain JSON fixture round-trip + typed
``ChainFormatNewerError`` exception.

Every JSON file dropped under ``tests/fixtures/legacy_chains/`` must
survive ``ReasoningChain.migrate`` + ``from_dict`` without raising. The
test is intentionally parametrised over the directory so CARE
contributors can drop new versioned fixtures here as the format
evolves — running the suite catches missed migration rungs.

Plus: ``from_dict`` raises a typed :class:`ChainFormatNewerError` when
the saved ``format_version`` is greater than the current library's
``FORMAT_VERSION`` — CARE catches it and prompts the user to upgrade
``mmar-carl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmar_carl import ChainFormatNewerError, ReasoningChain

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "legacy_chains"


# ---------------------------------------------------------------------------
# Parametrised fixture round-trip
# ---------------------------------------------------------------------------


def _fixture_paths() -> list[Path]:
    """Return every JSON file under ``tests/fixtures/legacy_chains/``.

    Sorted so test ids are stable across runs.
    """
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "fixture_path",
    _fixture_paths(),
    ids=lambda p: p.name,
)
def test_legacy_chain_fixture_round_trips(fixture_path: Path) -> None:
    """Each ``.json`` fixture under ``tests/fixtures/legacy_chains/``
    survives migration + reconstruction without raising, and the
    rebuilt chain re-serialises into something whose ``to_dict()``
    round-trips back through ``from_dict``.

    This is the CI guard the spec asks for: when a step-config field is
    renamed or removed, the migration ladder MUST cover the legacy
    fixture or this test fires.
    """
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    migrated = ReasoningChain.migrate(raw)
    chain = ReasoningChain.from_dict(migrated)

    # Basic sanity: at least one step and step numbers are positive.
    assert chain.steps, f"{fixture_path.name}: rebuilt chain has no steps"
    for step in chain.steps:
        assert step.number >= 1

    # Re-serialise and round-trip through migrate + from_dict again.
    second = ReasoningChain.from_dict(
        ReasoningChain.migrate(chain.to_dict()),
    )
    assert len(second.steps) == len(chain.steps)
    # Each step's title + number survives the second pass.
    for orig, rt in zip(chain.steps, second.steps):
        assert rt.number == orig.number
        assert rt.title == orig.title


def test_fixtures_directory_is_present() -> None:
    """If the fixtures directory disappears (e.g. accidentally
    deleted), fail loudly so contributors notice — the parametrised
    test would otherwise silently collect zero cases.
    """
    assert FIXTURES_DIR.is_dir(), (
        f"Expected fixtures dir at {FIXTURES_DIR} — has it been moved?"
    )
    assert _fixture_paths(), (
        "tests/fixtures/legacy_chains/ exists but contains no .json files. "
        "Add at least one fixture so the regression test has something to "
        "exercise."
    )


def test_v0_pre_format_version_fixture_migrates_to_current() -> None:
    """The hand-written ``v0_pre_format_version.json`` fixture uses the
    pre-format_version layout (carries a `version` string key, no
    `format_version` integer). The migrate ladder must strip the old
    `version` key and migrate through to the current format version.

    This documents the migration contract: a legacy on-disk chain
    saved by an older mmar-carl release must keep loading.
    """
    raw = json.loads(
        (FIXTURES_DIR / "v0_pre_format_version.json").read_text(encoding="utf-8"),
    )
    assert "version" in raw  # pre-migration shape
    assert "format_version" not in raw

    migrated = ReasoningChain.migrate(raw)
    assert migrated["format_version"] == ReasoningChain.FORMAT_VERSION
    assert "version" not in migrated  # the migration rung dropped it

    chain = ReasoningChain.from_dict(migrated)
    assert len(chain.steps) == 1
    assert chain.steps[0].title == "Pre-format-version step"


def test_v1_chain_migrates_to_current_without_rewriting_steps() -> None:
    raw = json.loads(
        (FIXTURES_DIR / "v1_simple_llm.json").read_text(encoding="utf-8"),
    )
    original_steps = raw["steps"]

    migrated = ReasoningChain.migrate(raw)

    assert migrated["format_version"] == ReasoningChain.FORMAT_VERSION == 10
    assert migrated["steps"] == original_steps
    assert migrated["chain_tools"] == []
    assert ReasoningChain.from_dict(migrated).steps[0].title == original_steps[0]["title"]


def test_v2_chain_migrates_to_current_without_rewriting_steps() -> None:
    raw = json.loads(
        (FIXTURES_DIR / "v1_simple_llm.json").read_text(encoding="utf-8"),
    )
    raw["format_version"] = 2
    original_steps = raw["steps"]

    migrated = ReasoningChain.migrate(raw)

    assert migrated["format_version"] == ReasoningChain.FORMAT_VERSION == 10
    assert migrated["steps"] == original_steps
    assert migrated["chain_tools"] == []
    assert ReasoningChain.from_dict(migrated).steps[0].title == original_steps[0]["title"]


def test_migrate_honors_explicit_target_and_rejects_downgrade() -> None:
    raw_v0 = {"version": "legacy", "steps": []}
    migrated_v1 = ReasoningChain.migrate(raw_v0, to_version=1)
    assert migrated_v1["format_version"] == 1
    assert "version" not in migrated_v1

    migrated_v2 = ReasoningChain.migrate(migrated_v1, to_version=2)
    assert migrated_v2["format_version"] == 2

    migrated_v3 = ReasoningChain.migrate(migrated_v2, to_version=3)
    assert migrated_v3["format_version"] == 3

    migrated_v4 = ReasoningChain.migrate(migrated_v3, to_version=4)
    assert migrated_v4["format_version"] == 4

    migrated_v5 = ReasoningChain.migrate(migrated_v4, to_version=5)
    assert migrated_v5["format_version"] == 5

    migrated_v6 = ReasoningChain.migrate(migrated_v5, to_version=6)
    assert migrated_v6["format_version"] == 6

    migrated_v7 = ReasoningChain.migrate(migrated_v6, to_version=7)
    assert migrated_v7["format_version"] == 7

    migrated_v8 = ReasoningChain.migrate(migrated_v7, to_version=8)
    assert migrated_v8["format_version"] == 8

    migrated_v9 = ReasoningChain.migrate(migrated_v8, to_version=9)
    assert migrated_v9["format_version"] == 9
    assert migrated_v9["chain_tools"] == []

    migrated_v10 = ReasoningChain.migrate(migrated_v9, to_version=10)
    assert migrated_v10["format_version"] == ReasoningChain.FORMAT_VERSION == 10

    with pytest.raises(ValueError, match="downgrades are not supported"):
        ReasoningChain.migrate({"format_version": 7, "steps": []}, to_version=6)


def test_v5_human_input_migration_removes_fallback_with_warning() -> None:
    legacy = {
        "format_version": 5,
        "steps": [
            {
                "number": 1,
                "title": "Human review",
                "step_type": "human_input",
                "dependencies": [],
                "checkpoint": False,
                "checkpoint_name": None,
                "replan_enabled": None,
                "step_config": {
                    "prompt": "Review?",
                    "timeout": 10.0,
                    "fallback_value": "auto-approved",
                    "output_memory_key": "review",
                },
            },
        ],
    }

    with pytest.warns(UserWarning, match="fallback_value was removed"):
        migrated = ReasoningChain.migrate(legacy)

    assert legacy["steps"][0]["step_config"]["fallback_value"] == "auto-approved"
    assert migrated["format_version"] == ReasoningChain.FORMAT_VERSION == 10
    assert migrated["chain_tools"] == []
    assert "fallback_value" not in migrated["steps"][0]["step_config"]
    chain = ReasoningChain.from_dict(migrated)
    assert chain.steps[0].step_config.prompt == "Review?"


# ---------------------------------------------------------------------------
# Newer-format → typed error
# ---------------------------------------------------------------------------


class TestChainFormatNewerError:
    """A chain serialised by a *future* mmar-carl release must raise
    :class:`ChainFormatNewerError` instead of silently parsing on a
    best-effort basis. CARE catches this and surfaces an "upgrade
    mmar-carl" prompt.
    """

    def test_future_format_version_raises_typed_error(self) -> None:
        future = {
            "format_version": ReasoningChain.FORMAT_VERSION + 99,
            "carl_version": "99.0.0",
            "max_workers": 3,
            "enable_progress": False,
            "metadata": {},
            "timeout": None,
            "replan_policy": None,
            "search_config": None,
            "steps": [
                {
                    "number": 1,
                    "title": "x",
                    "step_type": "llm",
                    "aim": "x",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        with pytest.raises(ChainFormatNewerError) as exc_info:
            ReasoningChain.from_dict(future)
        err = exc_info.value
        assert err.required_version == ReasoningChain.FORMAT_VERSION + 99
        assert err.this_version == ReasoningChain.FORMAT_VERSION
        # Message mentions both versions + the upgrade hint.
        msg = str(err)
        assert str(err.required_version) in msg
        assert str(err.this_version) in msg
        assert "upgrade mmar-carl" in msg.lower()

    def test_current_format_version_loads_normally(self) -> None:
        data = {
            "format_version": ReasoningChain.FORMAT_VERSION,
            "carl_version": "unknown",
            "max_workers": 3,
            "enable_progress": False,
            "metadata": {},
            "timeout": None,
            "replan_policy": None,
            "search_config": None,
            "steps": [
                {
                    "number": 1,
                    "title": "ok",
                    "step_type": "llm",
                    "aim": "ok",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        chain = ReasoningChain.from_dict(data)
        assert chain.steps[0].title == "ok"

    def test_missing_format_version_loads_via_migrate(self) -> None:
        """A pre-format_version chain (no ``format_version`` key at all)
        loads after migrate() applies every rung through the current format — no
        ChainFormatNewerError raised.
        """
        legacy = {
            "version": "old-string-version",
            "max_workers": 3,
            "metadata": {},
            "steps": [
                {
                    "number": 1,
                    "title": "legacy",
                    "step_type": "llm",
                    "aim": "x",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        migrated = ReasoningChain.migrate(legacy)
        chain = ReasoningChain.from_dict(migrated)
        assert chain.steps[0].title == "legacy"

    def test_exported_from_top_level(self) -> None:
        """``ChainFormatNewerError`` is importable as
        ``mmar_carl.ChainFormatNewerError`` for CARE's `except` clause."""
        from mmar_carl import ChainFormatNewerError as TopLevel
        assert TopLevel is ChainFormatNewerError
