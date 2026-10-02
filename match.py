"""Match a user profile against foundations.csv and return a ranked shortlist.

Stage 1 — rules (`rules.py`) hard-filter the catalog and score survivors:
    total = tone_match + undertone_match + skin_type_match
          + finish_match + preference_match     (max 100)

Stage 2 — optional ML rerank (`--ml`) reorders those survivors using a
model trained on interaction logs. Filtered-out products never come back.

Usage:
    python3 match.py --profile user_profile.demo.json
    python3 match.py --profile user_profile.demo.json --ml --breakdown
    python3 match.py --ml --ml-alpha 0.8 --top 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import rules

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_rows(csv_path: Path) -> list[rules.Row]:
    with csv_path.open() as f:
        return [rules.Row.from_dict(d) for d in csv.DictReader(f)]


def load_profile(profile_path: Path) -> dict:
    with profile_path.open() as f:
        prof = json.load(f)
    return prof.get("resolved", prof)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_price(x: float) -> str:
    return f"${x:.2f}"


def _print_pretty(scored: list[rules.Scored], profile: dict,
                  explain: bool, breakdown: bool,
                  ml_rows: list | None = None) -> None:
    if not scored:
        print("\nNo matches found. Consider loosening a constraint:")
        print(f"  price cap:  {_fmt_price(rules._price_cap(profile))}")
        print(f"  depth:      {profile.get('skin_depth')} -> "
              f"{rules.DEPTH_MAP.get(profile.get('skin_depth', ''), [])}")
        print(f"  undertone:  {profile.get('undertone')} -> "
              f"{rules.UNDERTONE_MAP.get(profile.get('undertone', ''), [])}")
        print(f"  finish:     {rules._finish_maps(profile)}")
        print(f"  coverage:   {rules._coverage_maps(profile)}")
        return

    print(f"\nTop {len(scored)} matches (rule score is out of {rules.MAX_SCORE}):\n")
    for i, s in enumerate(scored, start=1):
        r = s.row
        ml_bit = ""
        if ml_rows is not None:
            m = ml_rows[i - 1]
            ml_bit = f"  ml={m.ml_score:.3f}  final={m.final_score:.3f}"
        print(f"  {i:2d}. [{s.total:3d}] {r.brand} — {r.product}{ml_bit}")
        print(f"       Shade: {r.shade_name} ({r.shade_code})  •  {r.depth} / {r.undertone}")
        print(f"       {r.finish} finish, {r.coverage} coverage  •  {_fmt_price(r.price)}"
              f"{' • SPF ' + str(r.spf) if r.spf else ''}"
              f"{' • fragrance-free' if r.fragrance_free else ''}")

        if breakdown:
            parts = [f"{k}={s.components.get(k, 0)}/{rules.COMPONENT_MAX[k]}"
                     for k in rules.COMPONENT_MAX]
            print(f"       score: " + "  ".join(parts))

        if explain:
            for c in s.contributions:
                sign = "+" if c.points >= 0 else ""
                print(f"          {sign}{c.points:>3} {c.component:<17} {c.reason}")
        print()


def _to_json(scored: list[rules.Scored], ml_rows: list | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, s in enumerate(scored):
        r = s.row
        item = {
            "total_score": s.total,
            "components": s.components,
            "brand": r.brand,
            "product": r.product,
            "shade_name": r.shade_name,
            "shade_code": r.shade_code,
            "depth": r.depth,
            "undertone": r.undertone,
            "finish": r.finish,
            "coverage": r.coverage,
            "price_usd": r.price,
            "spf": r.spf,
            "fragrance_free": r.fragrance_free,
            "contributions": [
                {"component": c.component, "points": c.points, "reason": c.reason}
                for c in s.contributions
            ],
        }
        if ml_rows is not None:
            item["ml_score"] = ml_rows[i].ml_score
            item["final_score"] = ml_rows[i].final_score
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, default=HERE / "user_profile.json",
                   help="Path to user profile JSON")
    p.add_argument("--csv", type=Path, default=HERE / "foundations.csv",
                   help="Path to foundations CSV")
    p.add_argument("--top", type=int, default=10, help="Number of matches to return")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit machine-readable JSON")
    p.add_argument("--explain", action="store_true",
                   help="Show each individual score contribution per match")
    p.add_argument("--breakdown", action="store_true",
                   help="Show per-component score totals per match")
    p.add_argument("--ml", action="store_true",
                   help="Re-rank rule survivors with the trained ML model")
    p.add_argument("--ml-model", type=Path, default=None,
                   help="Path to ranker.joblib (default: ml/artifacts/ranker.joblib)")
    p.add_argument("--ml-alpha", type=float, default=0.7,
                   help="Blend: final = (1-alpha)*rule + alpha*ml  (default 0.7)")
    p.add_argument("--ml-pool", type=int, default=25,
                   help="How many rule-ranked candidates to hand the ML model")
    args = p.parse_args(argv)

    if not args.profile.exists():
        print(f"Error: profile not found at {args.profile}", file=sys.stderr)
        print("Run `python3 questionnaire.py` first.", file=sys.stderr)
        return 2

    profile = load_profile(args.profile)
    rows = load_rows(args.csv)
    pool = max(args.top, args.ml_pool if args.ml else args.top)
    scored = rules.rank(rows, profile, top=pool)
    ml_rows = None

    if args.ml:
        from ml.ranker import load_ranker
        try:
            ranker = load_ranker(args.ml_model)
        except FileNotFoundError:
            print("No trained ranker found. Run:", file=sys.stderr)
            print("  python3 -m ml.generate_logs && python3 -m ml.train", file=sys.stderr)
            return 2
        ml_rows = ranker.rerank(profile, scored, top=args.top, alpha=args.ml_alpha)
        scored = [m.scored for m in ml_rows]
    else:
        scored = scored[:args.top]

    if args.as_json:
        print(json.dumps(_to_json(scored, ml_rows), indent=2))
    else:
        _print_pretty(scored, profile, explain=args.explain, breakdown=args.breakdown,
                      ml_rows=ml_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
