"""End-to-end modeling pipeline.

    quiz → rule filter (20–50 candidates) → ML ranker → top 3–5
         → user feedback → training JSONL → retrain

This is the product architecture: rules never get skipped, the model only
reorders the shortlist, and every impression/click is written back so the
next ``python3 -m ml.train`` run sees it.

Usage:
    python3 pipeline.py --profile user_profile.demo.json
    python3 pipeline.py --profile user_profile.demo.json --feedback
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.interactions import Event, append_event, now_iso
from ml.labels import SOURCE_WEIGHT, annotate
from ml.ranker import Ranked, load_ranker

CATALOG_PATH = ROOT / "foundations.csv"
LIVE_LOG = ROOT / "data" / "interactions.live.jsonl"
TRAINING_LOG = ROOT / "data" / "labels" / "training.jsonl"
MODEL_PATH = ROOT / "ml" / "artifacts" / "ranker.joblib"

CANDIDATE_K_MIN, CANDIDATE_K_MAX = 20, 50
DEFAULT_CANDIDATE_K = 30
DEFAULT_TOP_K = 5


@dataclass
class PipelineResult:
    session_id: str
    profile: dict
    n_catalog: int
    n_candidates: int
    used_ml: bool
    shown: list[Ranked]
    candidates: list[rules.Scored] = field(default_factory=list)


def load_catalog(csv_path: Path | None = None) -> list[rules.Row]:
    path = csv_path or CATALOG_PATH
    with path.open() as f:
        return [rules.Row.from_dict(d) for d in csv.DictReader(f)]


def load_profile(path: Path) -> dict:
    with path.open() as f:
        prof = json.load(f)
    return prof.get("resolved", prof)


def _candidate_payload(s: rules.Scored) -> dict:
    r = s.row
    return {
        "brand": r.brand,
        "product": r.product,
        "shade_name": r.shade_name,
        "shade_code": r.shade_code,
        "finish": r.finish,
        "coverage": r.coverage,
        "depth": r.depth,
        "undertone": r.undertone,
        "price_usd": r.price,
    }


def recommend(
    profile: dict,
    catalog: list[rules.Row] | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = 0.7,
    use_ml: bool | None = None,
    session_id: str | None = None,
    user_id: str = "pipeline",
    log_impressions: bool = True,
    source: str = "tester",
) -> PipelineResult:
    """Run the two-stage pipeline and optionally log impressions.

    1. Rule engine filters the catalog and keeps up to ``candidate_k`` (20–50).
    2. ML ranker scores that pool (falls back to rule order if no model).
    3. Return the top ``top_k`` (3–5) products.
    """
    catalog = catalog if catalog is not None else load_catalog()
    candidate_k = max(CANDIDATE_K_MIN, min(CANDIDATE_K_MAX, int(candidate_k)))
    top_k = max(3, min(5, int(top_k)))
    session_id = session_id or f"p-{uuid.uuid4().hex[:10]}"

    pool = rules.rank(catalog, profile, top=candidate_k)
    used_ml = False
    shown: list[Ranked] = []

    want_ml = MODEL_PATH.exists() if use_ml is None else use_ml
    if want_ml and pool:
        try:
            ranker = load_ranker()
            shown = ranker.rerank(profile, pool, top=top_k, alpha=alpha)
            used_ml = True
        except FileNotFoundError:
            want_ml = False

    if not used_ml:
        shown = [
            Ranked(scored=s, ml_score=0.0, final_score=s.total / rules.MAX_SCORE)
            for s in pool[:top_k]
        ]

    result = PipelineResult(
        session_id=session_id,
        profile=profile,
        n_catalog=len(catalog),
        n_candidates=len(pool),
        used_ml=used_ml,
        shown=shown,
        candidates=pool,
    )
    if log_impressions:
        _log_impressions(result, user_id=user_id, source=source)
    return result


def _log_impressions(result: PipelineResult, user_id: str, source: str) -> None:
    for i, item in enumerate(result.shown, start=1):
        s = item.scored
        ev = Event(
            session_id=result.session_id,
            user_id=user_id,
            timestamp=now_iso(),
            profile=result.profile,
            candidate=_candidate_payload(s),
            rule_score=s.total,
            components=s.components,
            rank_rules=i,
            shown=True,
            clicked=0,
            saved=0,
            purchased=0,
            rated=None,
            source=source,
            sample_weight=SOURCE_WEIGHT.get(source, 1.0),
        )
        append_event(LIVE_LOG, ev)


def record_feedback(
    result: PipelineResult,
    rank: int,
    action: str = "clicked",
    rating: int | None = None,
    user_id: str = "pipeline",
    source: str = "tester",
) -> Event:
    """Record click/save/purchase/rating on a shown product and push it
    into the training JSONL so the next train run picks it up."""
    if rank < 1 or rank > len(result.shown):
        raise IndexError(f"rank {rank} is outside 1..{len(result.shown)}")
    item = result.shown[rank - 1]
    s = item.scored
    flags = {
        "clicked": int(action in {"clicked", "saved", "purchased", "up"} or (rating is not None and rating >= 4)),
        "saved": int(action in {"saved", "purchased"}),
        "purchased": int(action == "purchased"),
        "rated": rating if rating is not None else (5 if action == "up" else (1 if action == "down" else None)),
    }
    if action == "down":
        flags["clicked"] = 0
        flags["saved"] = 0
        flags["purchased"] = 0
        flags["rated"] = 1
    ev = Event(
        session_id=result.session_id,
        user_id=user_id,
        timestamp=now_iso(),
        profile=result.profile,
        candidate=_candidate_payload(s),
        rule_score=s.total,
        components=s.components,
        rank_rules=rank,
        shown=True,
        source=source,
        sample_weight=SOURCE_WEIGHT.get(source, 1.0),
        **flags,
    )
    payload = annotate(ev, source)
    append_event(LIVE_LOG, ev)
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    append_event(TRAINING_LOG, payload)
    return ev


def _print_result(result: PipelineResult) -> None:
    stage = "rules → ML ranker" if result.used_ml else "rules only (no model on disk)"
    print(f"\nPipeline  [{stage}]")
    print(f"  catalog {result.n_catalog}  →  candidates {result.n_candidates}  →  shown {len(result.shown)}")
    print(f"  session {result.session_id}\n")
    for i, item in enumerate(result.shown, start=1):
        s = item.scored
        r = s.row
        why = " · ".join(c.reason for c in s.contributions[:3]) or "rule match"
        ml = f"  ml={item.ml_score:.3f}" if result.used_ml else ""
        print(f"  {i}. [{s.total:3d}]{ml}  {r.brand} — {r.product}")
        print(f"      Shade {r.shade_name}  •  {r.depth}/{r.undertone}  •  {r.finish}/{r.coverage}  •  ${r.price:.2f}")
        print(f"      why: {why}")
        print()


def _interactive_feedback(result: PipelineResult) -> None:
    print("Feedback  (press Enter to skip)")
    print("  Format:  <rank> <clicked|saved|purchased|up|down>  [rating 1-5]")
    print("  Example:  1 saved     or     2 up 5     or     3 down")
    while True:
        raw = input("  > ").strip()
        if not raw:
            break
        parts = raw.split()
        if not parts[0].isdigit():
            print("  Start with the rank number (1–%d)." % len(result.shown))
            continue
        rank = int(parts[0])
        action = parts[1] if len(parts) > 1 else "clicked"
        rating = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        try:
            record_feedback(result, rank, action=action, rating=rating)
        except (IndexError, ValueError) as e:
            print(f"  {e}")
            continue
        print(f"  logged {action} on #{rank} → {LIVE_LOG.name} and {TRAINING_LOG.name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, default=ROOT / "user_profile.demo.json")
    p.add_argument("--csv", type=Path, default=CATALOG_PATH)
    p.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATE_K,
                   help="Rule-filtered pool size (clamped to 20–50)")
    p.add_argument("--top", type=int, default=DEFAULT_TOP_K,
                   help="How many products to show (clamped to 3–5)")
    p.add_argument("--ml-alpha", type=float, default=0.7)
    p.add_argument("--no-ml", action="store_true", help="Skip the ranker even if a model exists")
    p.add_argument("--feedback", action="store_true",
                   help="Prompt for click/save/rating after showing matches")
    p.add_argument("--json", dest="as_json", action="store_true")
    args = p.parse_args(argv)

    if not args.profile.exists():
        print(f"No profile at {args.profile}. Run python3 questionnaire.py first.", file=sys.stderr)
        return 2

    result = recommend(
        load_profile(args.profile),
        catalog=load_catalog(args.csv),
        candidate_k=args.candidates,
        top_k=args.top,
        alpha=args.ml_alpha,
        use_ml=False if args.no_ml else None,
    )
    if args.as_json:
        print(json.dumps({
            "session_id": result.session_id,
            "n_candidates": result.n_candidates,
            "used_ml": result.used_ml,
            "shown": [
                {
                    "rank": i,
                    "rule_score": m.scored.total,
                    "ml_score": m.ml_score,
                    "final_score": m.final_score,
                    "brand": m.scored.row.brand,
                    "product": m.scored.row.product,
                    "shade_name": m.scored.row.shade_name,
                }
                for i, m in enumerate(result.shown, start=1)
            ],
        }, indent=2))
    else:
        _print_result(result)
    if args.feedback and result.shown and sys.stdin.isatty():
        _interactive_feedback(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
