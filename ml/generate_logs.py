"""Generate synthetic interaction logs so the ranker can be trained before
real users exist.

Each session:
  1. Sample a user profile.
  2. Run the *rules* engine and take the top-K candidates (what the MVP shows).
  3. Draw clicks / saves / purchases from a latent preference model that
     includes residual patterns the rules only partly capture, e.g.
     dry users preferring satin over matte, warm users clicking warm shades.

The latent model is *not* the training target the ranker sees — the ranker
only sees the resulting labels — so we can measure whether it recovers those
patterns from the logs.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.interactions import Event, append_event, now_iso

SKIN_POOL = ["oily", "dry", "combination", "normal", "sensitive", "acne_prone"]
DEPTHS = ["fair", "light", "medium", "tan", "deep", "very_deep"]
UNDERTONES = ["cool", "warm", "neutral", "olive"]
FINISHES = ["matte", "natural", "satin", "dewy"]
COVERAGES = ["light", "medium", "full"]
PRICE_TIERS = [
    ("drugstore", 20),
    ("mid", 40),
    ("prestige", 60),
    ("luxury", 1000),
    ("no_pref", 1000),
]

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


def _sample_profile(rng: random.Random) -> dict:
    n_types = rng.randint(1, 2)
    skin = rng.sample(SKIN_POOL, n_types)
    # Mild correlation: oily users lean matte; dry users lean dewy/satin.
    if "oily" in skin or "acne_prone" in skin:
        finish_opts = ["matte", "natural", "satin"]
        weights = [0.45, 0.35, 0.20]
    elif "dry" in skin:
        finish_opts = ["dewy", "satin", "natural", "matte"]
        weights = [0.30, 0.30, 0.30, 0.10]
    else:
        finish_opts = FINISHES
        weights = [0.25] * 4
    n_fin = 1 if rng.random() < 0.4 else 2
    finishes = rng.choices(finish_opts, weights=weights, k=n_fin)
    finishes = list(dict.fromkeys(finishes))

    maps: list[str] = []
    for f in finishes:
        for m in FINISH_MAPS[f]:
            if m not in maps:
                maps.append(m)
    # Often keep finish broad so the MVP actually has a shortlist to re-rank.
    if rng.random() < 0.35:
        maps = ["Matte", "Natural", "Satin", "Radiant", "Dewy", "Luminous"]

    coverage = rng.choice(COVERAGES)
    cov_maps = COVERAGE_MAPS[coverage]
    if rng.random() < 0.4:
        cov_maps = ["Sheer", "Light", "Light-Medium", "Medium", "Medium-Full", "Full"]

    # Slightly richer depths (more catalog coverage) and looser budgets.
    depth = rng.choices(
        DEPTHS, weights=[0.12, 0.22, 0.28, 0.18, 0.12, 0.08]
    )[0]
    tier, cap = rng.choices(
        PRICE_TIERS, weights=[0.15, 0.30, 0.30, 0.10, 0.15]
    )[0]
    ff = rng.choices(["required", "preferred", "no_pref"], weights=[0.12, 0.33, 0.55])[0]
    if "sensitive" in skin and rng.random() < 0.4:
        ff = "required"

    return {
        "skin_type": {"selected": skin, "maps_to_dataset": None},
        "undertone": rng.choice(UNDERTONES),
        "skin_depth": depth,
        "finish": {"selected": finishes, "maps_to_dataset": maps},
        "coverage": {"selected": coverage, "maps_to_dataset": cov_maps},
        "fragrance_free": ff,
        "non_comedogenic": "preferred" if "acne_prone" in skin else "no_pref",
        "cruelty_free": "no_pref",
        "price_range": {"selected": tier, "max_usd": cap},
        "spf_needed": rng.choices(["yes", "yes_30", "no_pref"], weights=[0.15, 0.10, 0.75])[0],
    }


def _latent_logit(profile: dict, scored: rules.Scored, rng: random.Random) -> float:
    """Hidden preference the synthetic user actually has.

    Rules contribute, but residual terms (dry×satin, warm×warm shade, etc.)
    are strong enough that a model can beat a pure rule ranking.
    """
    row = scored.row
    types = rules._user_skin_types(profile)
    ut = profile.get("undertone")
    z = -2.4
    z += 0.018 * scored.total
    z += 0.8 * (scored.components.get("tone_match", 0) / 25)
    z += 0.8 * (scored.components.get("undertone_match", 0) / 25)

    if "dry" in types:
        if row.finish in {"Dewy", "Radiant", "Luminous", "Satin"}:
            z += 1.6
        if row.finish == "Matte":
            z -= 1.4
        if rules._contains_any(row.ingredients, rules.HYDRATION_SIGNALS):
            z += 0.6
    if "oily" in types or "acne_prone" in types:
        if row.finish == "Matte":
            z += 1.3
        if row.finish in {"Dewy", "Luminous"}:
            z -= 0.9
    if "sensitive" in types and not row.fragrance_free:
        z -= 1.5
    if ut == "warm" and row.undertone in {"Warm", "Neutral-Warm"}:
        z += 1.1
    if ut == "cool" and row.undertone in {"Cool", "Neutral-Cool"}:
        z += 1.1
    if ut == "olive" and row.undertone == "Olive":
        z += 1.4
    if ut == "neutral" and "Neutral" in row.undertone:
        z += 0.7

    cap = rules._price_cap(profile)
    if cap < 1e8 and row.price > 0.85 * cap:
        z -= 0.5
    if row.price < 25:
        z += 0.25
    z += rng.gauss(0, 0.45)
    return z


def _sigmoid(z: float) -> float:
    if z >= 20:
        return 1.0
    if z <= -20:
        return 0.0
    return 1.0 / (1.0 + pow(2.718281828, -z))


def simulate_session(
    session_id: str,
    user_id: str,
    profile: dict,
    catalog: list[rules.Row],
    rng: random.Random,
    show_k: int = 12,
) -> list[Event]:
    scored = rules.rank(catalog, profile, top=show_k)
    if not scored:
        return []

    events: list[Event] = []
    for rank_i, s in enumerate(scored, start=1):
        p_click = _sigmoid(_latent_logit(profile, s, rng))
        clicked = int(rng.random() < p_click)
        saved = int(clicked and rng.random() < 0.35)
        purchased = int(saved and rng.random() < 0.25)
        rated = None
        if clicked:
            base = 3 + int(p_click > 0.55) + int(p_click > 0.75)
            rated = max(1, min(5, base + rng.choice([-1, 0, 0, 1])))
        events.append(Event(
            session_id=session_id,
            user_id=user_id,
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
            shown=True,
            clicked=clicked,
            saved=saved,
            purchased=purchased,
            rated=rated,
            source="synthetic",
            sample_weight=0.5,
        ))
    return events


def generate(n_sessions: int, catalog: list[rules.Row], seed: int = 7,
             show_k: int = 12, min_shown: int = 6) -> list[Event]:
    rng = random.Random(seed)
    all_events: list[Event] = []
    i = 0
    attempts = 0
    while i < n_sessions and attempts < n_sessions * 20:
        attempts += 1
        profile = _sample_profile(rng)
        evs = simulate_session(f"s{i:05d}", f"u{rng.randint(1, n_sessions // 2 + 1):04d}",
                               profile, catalog, rng, show_k=show_k)
        if len(evs) < min_shown:
            continue
        all_events.extend(evs)
        i += 1
    return all_events


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sessions", type=int, default=400)
    p.add_argument("--show-k", type=int, default=12,
                   help="Candidates shown per session (the MVP shortlist)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--csv", type=Path, default=ROOT / "foundations.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "interactions.jsonl")
    args = p.parse_args(argv)

    with args.csv.open() as f:
        catalog = [rules.Row.from_dict(d) for d in csv.DictReader(f)]

    if args.out.exists():
        args.out.unlink()
    events = generate(args.n_sessions, catalog, seed=args.seed, show_k=args.show_k)
    for ev in events:
        append_event(args.out, ev)

    n_click = sum(e.clicked for e in events)
    n_save = sum(e.saved for e in events)
    n_buy = sum(e.purchased for e in events)
    print(f"Wrote {len(events)} impressions across {args.n_sessions} sessions → {args.out}")
    print(f"  clicked={n_click} ({n_click / max(len(events), 1):.1%})  "
          f"saved={n_save}  purchased={n_buy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
