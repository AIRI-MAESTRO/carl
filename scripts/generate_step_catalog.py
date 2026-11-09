"""Regenerate the committed step-type catalog snapshot.

Run via ``make meta-snapshot``. The snapshot
(``src/mmar_carl/meta/catalog_snapshot.json``) is the versioned artifact that
external services (MAGE chain generation, GigaEvo evolution) consume, and the
CI drift test (``tests/meta/test_snapshot_drift.py``) compares it against the
live models so a new step type can't ship without a reviewable line-diff here.

The snapshot omits the ``generated_at`` timestamp and is compared structurally,
so version bumps / timestamps never cause spurious "stale snapshot" failures —
only an actual change to the step-type surface does.
"""

from __future__ import annotations

import pathlib
import sys

# Make the script run regardless of how (or whether) mmar-carl is installed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mmar_carl.meta.catalog import snapshot_is_stale, write_snapshot  # noqa: E402


def main() -> int:
    was_stale, summary = snapshot_is_stale()
    path = write_snapshot()
    n_types = len(__import__("mmar_carl.meta", fromlist=["get_step_catalog"]).get_step_catalog().names())
    if was_stale:
        print(f"updated {path} ({n_types} step types) — {summary}")
    else:
        print(f"{path} already up to date ({n_types} step types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
