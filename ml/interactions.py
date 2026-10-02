"""Interaction-log schema and I/O.

One JSONL row = one (session, candidate) impression. Labels:

  clicked     user opened / tapped the product
  saved       user favorited / shortlisted it
  purchased   user bought it (or marked "I wear this")
  rated       optional 1–5 match-quality rating

``relevance_score`` collapses those into a single graded target for
NDCG and for the composite ``relevance`` training objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TARGETS = ("clicked", "saved", "purchased", "rated", "relevance")

# Graded relevance used for NDCG and the composite target.
# Purchase dominates; a high rating without a click still counts.
REL_WEIGHTS = {
    "purchased": 8,
    "saved": 4,
    "clicked": 2,
}


@dataclass
class Event:
    session_id: str
    user_id: str
    timestamp: str
    profile: dict
    candidate: dict
    rule_score: int
    components: dict
    rank_rules: int
    shown: bool = True
    clicked: int = 0
    saved: int = 0
    purchased: int = 0
    rated: int | None = None
    source: str = "behavior"
    sample_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "profile": self.profile,
            "candidate": self.candidate,
            "rule_score": self.rule_score,
            "components": self.components,
            "rank_rules": self.rank_rules,
            "shown": self.shown,
            "clicked": int(self.clicked),
            "saved": int(self.saved),
            "purchased": int(self.purchased),
            "rated": self.rated,
            "source": self.source,
            "sample_weight": float(self.sample_weight),
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def relevance_score(event: Event | dict) -> int:
    """Graded label: purchased > saved > clicked, plus optional rating."""
    get = event.get if isinstance(event, dict) else lambda k, d=None: getattr(event, k, d)
    score = 0
    if get("purchased"):
        score += REL_WEIGHTS["purchased"]
    if get("saved"):
        score += REL_WEIGHTS["saved"]
    if get("clicked"):
        score += REL_WEIGHTS["clicked"]
    rated = get("rated")
    if rated is not None:
        score += max(0, int(rated) - 2)  # 3→1, 4→2, 5→3
    return score


def binary_label(event: Event | dict, target: str) -> int:
    """Map an event onto the chosen training target."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}. Choose from {TARGETS}.")
    get = event.get if isinstance(event, dict) else lambda k, d=None: getattr(event, k, d)
    if target == "clicked":
        return 1 if get("clicked") else 0
    if target == "saved":
        return 1 if get("saved") else 0
    if target == "purchased":
        return 1 if get("purchased") else 0
    if target == "rated":
        rated = get("rated")
        return 1 if rated is not None and int(rated) >= 4 else 0
    return 1 if relevance_score(event) > 0 else 0


def append_event(path: Path, event: Event | dict) -> None:
    """Append one interaction to a JSONL log. Safe to call from the UI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = event.to_dict() if isinstance(event, Event) else event
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def group_by_session(events: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for ev in events:
        grouped.setdefault(ev["session_id"], []).append(ev)
    return grouped
