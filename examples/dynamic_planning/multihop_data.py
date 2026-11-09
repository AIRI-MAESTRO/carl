"""HotpotQA (distractor) sample loader for the dynamic-planning demo.

Fetches a small slice of the validation split through the HuggingFace
datasets-server rows API (no full-dataset download) and caches it locally, so
repeated demo runs are offline after the first fetch.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

_CACHE = Path(__file__).parent / "data" / "hotpot_sample.json"
_API = "https://datasets-server.huggingface.co/rows"


def load_cases(n: int = 12) -> list[dict]:
    """Return n HotpotQA distractor validation cases: question, answer, paragraphs."""
    if _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if len(cached) >= n:
            return cached[:n]
    params = urllib.parse.urlencode(
        {
            "dataset": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "validation",
            "offset": 0,
            "length": max(n, 24),
        }
    )
    req = urllib.request.Request(f"{_API}?{params}", headers={"User-Agent": "carl-demo/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    cases = []
    for row in payload["rows"]:
        r = row["row"]
        paragraphs = [
            {"title": title, "text": " ".join(sents)}
            for title, sents in zip(r["context"]["title"], r["context"]["sentences"])
        ]
        cases.append(
            {
                "id": r.get("id", f"case-{row['row_idx']}"),
                "question": r["question"],
                "answer": r["answer"],
                "type": r.get("type", ""),
                "level": r.get("level", ""),
                "paragraphs": paragraphs,
            }
        )
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    return cases[:n]
