"""Multi-source training labels and the bootstrap merge.

Label sources, weakest → strongest. A stronger source *replaces* a weaker
one for the same (user profile, SKU) key — that is the practical trick:
start from rule-score pseudo-labels, then overwrite with synthetic users,
expert ratings, beta testers, and finally real behavior.

    rule        weak / pseudo-label from the stage-1 score
    synthetic   generated users (skin type × undertone combos)
    expert      manual ratings on a product subset
    tester      friends / small beta (Streamlit feedback buttons)
    behavior    production clicks / saves / purchases
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.interactions import Event, now_iso

SOURCE_PRIORITY = {
    "rule": 10,
    "synthetic": 20,
    "expert": 30,
    "tester": 40,
    "behavior": 50,
}

SOURCE_WEIGHT = {
    "rule": 0.25,
    "synthetic": 0.50,
    "expert": 1.50,
    "tester": 1.50,
    "behavior": 2.00,
}

FINISH_MAPS = {
    "matte": ["Matte"],
    "natural": ["Natural", "Satin"],
    "satin": ["Satin", "Natural"],
    "dewy": ["Dewy", "Radiant", "Luminous"],
}
COVERAGE_MAPS = {
    "light": ["Sheer", "Light", "Light-Medium"],
    "medium": ["Light-Medium", "Medium", "Medium-Full"],
    "full": ["Medium-Full", "Full"],
}


def split_multi(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace("|", ",").split(",") if p.strip()]


def build_profile(
    skin_type: str,
    undertone: str,
    skin_depth: str,
    finish: str,
    coverage: str,
    fragrance_free: str = "no_pref",
    price_max: float = 60,
) -> dict:
    skins = split_multi(skin_type)
    finishes = split_multi(finish)
    maps: list[str] = []
    for f in finishes:
        for m in FINISH_MAPS.get(f, [f.capitalize()]):
            if m not in maps:
                maps.append(m)
    return {
        "skin_type": {"selected": skins, "maps_to_dataset": None},
        "undertone": undertone,
        "skin_depth": skin_depth,
        "finish": {"selected": finishes, "maps_to_dataset": maps},
        "coverage": {
            "selected": coverage,
            "maps_to_dataset": COVERAGE_MAPS.get(coverage, ["Medium"]),
        },
        "fragrance_free": fragrance_free or "no_pref",
        "non_comedogenic": "preferred" if "acne_prone" in skins else "no_pref",
        "cruelty_free": "no_pref",
        "price_range": {"selected": "custom", "max_usd": float(price_max)},
        "spf_needed": "no_pref",
    }


def example_key(profile: dict, candidate: dict) -> tuple:
    """Stable identity for (user, SKU) used when a stronger label replaces a weaker one."""
    st = profile.get("skin_type", {})
    skins = tuple(sorted(st.get("selected", []) if isinstance(st, dict) else (st or [])))
    fin = profile.get("finish", {})
    finishes = tuple(sorted(fin.get("selected", []) if isinstance(fin, dict) else (fin or [])))
    cov = profile.get("coverage", {})
    coverage = cov.get("selected", cov) if isinstance(cov, dict) else cov
    return (
        skins,
        profile.get("undertone"),
        profile.get("skin_depth"),
        finishes,
        coverage,
        candidate.get("brand"),
        candidate.get("product"),
        candidate.get("shade_name"),
    )


def annotate(event: Event | dict, source: str) -> dict:
    payload = event.to_dict() if isinstance(event, Event) else dict(event)
    payload["source"] = source
    payload["sample_weight"] = float(payload.get("sample_weight") or SOURCE_WEIGHT[source])
    return payload


def merge_by_priority(groups: Iterable[list[dict]]) -> list[dict]:
    """Keep the highest-priority source for each (profile, SKU)."""
    chosen: dict[tuple, dict] = {}
    for group in groups:
        for ev in group:
            src = ev.get("source") or "behavior"
            key = example_key(ev["profile"], ev["candidate"])
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = ev
                continue
            if SOURCE_PRIORITY.get(src, 0) > SOURCE_PRIORITY.get(prev.get("source"), 0):
                chosen[key] = ev
    return list(chosen.values())


def rating_to_flags(rating: int) -> dict:
    """Map a 1–5 expert/tester rating onto the click/save/purchase schema."""
    rating = int(rating)
    return {
        "clicked": int(rating >= 4),
        "saved": int(rating >= 4),
        "purchased": int(rating >= 5),
        "rated": rating,
    }


def rule_score_to_flags(score: int) -> dict:
    """Weak / pseudo-label from the stage-1 rule score.

    High rule scores become positive; low ones stay negative. Purchases are
    never invented from a rule score — that label stays reserved for humans.
    """
    if score >= 80:
        rated = 5
    elif score >= 70:
        rated = 4
    elif score >= 55:
        rated = 3
    elif score >= 40:
        rated = 2
    else:
        rated = 1
    return {
        "clicked": int(score >= 70),
        "saved": int(score >= 80),
        "purchased": 0,
        "rated": rated,
    }


def _score_unfiltered(row: rules.Row, profile: dict) -> rules.Scored:
    """Apply scoring rules even if a hard filter would reject the row."""
    scored = rules.Scored(row=row, components={k: 0 for k in rules.COMPONENT_MAX})
    for fn in rules.SCORE_RULES:
        for c in fn(row, profile):
            scored.components[c.component] = scored.components.get(c.component, 0) + c.points
            scored.contributions.append(c)
    return scored


def expert_csv_to_events(path: Path, catalog: list[rules.Row]) -> list[dict]:
    """Read a filled-in expert ratings CSV and join it to the live catalog."""
    index = {(r.brand, r.product, r.shade_name): r for r in catalog}
    events: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if not row.get("brand") or not row.get("rating"):
                continue
            key = (row["brand"].strip(), row["product"].strip(), row["shade_name"].strip())
            sku = index.get(key)
            if sku is None:
                print(f"  skip expert row {i}: SKU not in catalog {key}", file=sys.stderr)
                continue
            profile = build_profile(
                row.get("skin_type", "normal"),
                row.get("undertone", "neutral"),
                row.get("skin_depth", "medium"),
                row.get("finish", "natural"),
                row.get("coverage", "medium"),
                row.get("fragrance_free") or "no_pref",
                float(row.get("price_max") or 60),
            )
            scored, reasons = rules.evaluate(sku, profile)
            if scored is None:
                # Experts rate pairs, not "pairs that pass today's filters".
                # Keep the label and score with the same rules, minus the gate.
                scored = _score_unfiltered(sku, profile)
            flags = rating_to_flags(int(row["rating"]))
            events.append(annotate(Event(
                session_id=f"expert-{row.get('labeler', 'anon')}-{i:03d}",
                user_id=f"expert:{row.get('labeler', 'anon')}",
                timestamp=now_iso(),
                profile=profile,
                candidate={
                    "brand": sku.brand,
                    "product": sku.product,
                    "shade_name": sku.shade_name,
                    "shade_code": sku.shade_code,
                    "finish": sku.finish,
                    "coverage": sku.coverage,
                    "depth": sku.depth,
                    "undertone": sku.undertone,
                    "price_usd": sku.price,
                },
                rule_score=scored.total,
                components=scored.components,
                rank_rules=0,
                **flags,
                source="expert",
                sample_weight=SOURCE_WEIGHT["expert"],
            ), "expert"))
    return events


def rule_weak_labels(catalog: list[rules.Row], top_k: int = 8) -> list[dict]:
    """Grid of realistic skin-type × undertone × depth users, labeled by rule score."""
    skins = ["oily", "dry", "combination", "normal", "sensitive", "oily,acne_prone", "dry,sensitive"]
    undertones = ["cool", "warm", "neutral", "olive"]
    depths = ["fair", "light", "medium", "tan", "deep"]
    finish_by_skin = {
        "oily": "matte,natural",
        "acne_prone": "matte",
        "dry": "dewy,satin",
        "sensitive": "natural,dewy",
        "combination": "natural,matte",
        "normal": "natural",
    }
    events: list[dict] = []
    n = 0
    for skin in skins:
        primary = split_multi(skin)[0]
        finish = finish_by_skin.get(primary, "natural")
        coverage = "medium" if primary != "dry" else "light"
        for ut in undertones:
            for depth in depths:
                profile = build_profile(skin, ut, depth, finish, coverage, "no_pref", 80)
                scored = rules.rank(catalog, profile, top=top_k)
                if not scored:
                    continue
                n += 1
                for rank_i, s in enumerate(scored, start=1):
                    flags = rule_score_to_flags(s.total)
                    events.append(annotate(Event(
                        session_id=f"rule-{n:04d}",
                        user_id="pseudo:rules",
                        timestamp=now_iso(),
                        profile=profile,
                        candidate={
                            "brand": s.row.brand,
                            "product": s.row.product,
                            "shade_name": s.row.shade_name,
                            "shade_code": s.row.shade_code,
                            "finish": s.row.finish,
                            "coverage": s.row.coverage,
                            "depth": s.row.depth,
                            "undertone": s.row.undertone,
                            "price_usd": s.row.price,
                        },
                        rule_score=s.total,
                        components=s.components,
                        rank_rules=rank_i,
                        **flags,
                        source="rule",
                        sample_weight=SOURCE_WEIGHT["rule"],
                    ), "rule"))
    return events


def load_jsonl(path: Path, default_source: str | None = None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    events = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            src = ev.get("source") or default_source
            if src:
                ev = annotate(ev, src)
            events.append(ev)
    return events


def source_counts(events: Iterable[dict]) -> Counter:
    return Counter((e.get("source") or "unknown") for e in events)
