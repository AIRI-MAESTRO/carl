"""The committed catalog snapshot must stay in sync with the live models.

This is the CI half of the "both" delivery model: external services consume
``src/mmar_carl/meta/catalog_snapshot.json``, and this test fails (with the fix
command) whenever a step-type change makes it stale.
"""

from __future__ import annotations

from mmar_carl.meta import StepCatalog, load_snapshot
from mmar_carl.meta.catalog import SNAPSHOT_PATH, snapshot_is_stale
from mmar_carl.models.enums import StepType


def test_committed_snapshot_is_present() -> None:
    assert SNAPSHOT_PATH.exists(), "run `make meta-snapshot`"


def test_committed_snapshot_is_not_stale() -> None:
    stale, summary = snapshot_is_stale()
    assert not stale, summary


def test_snapshot_loads_and_covers_all_types() -> None:
    snap = load_snapshot()
    assert isinstance(snap, StepCatalog)
    assert set(snap.names()) == {st.value for st in StepType}


def test_snapshot_has_no_timestamp_for_stable_diffs() -> None:
    # generated_at is intentionally omitted so version bumps / re-runs don't
    # produce spurious diffs.
    snap = load_snapshot()
    assert snap is not None
    assert snap.generated_at is None
