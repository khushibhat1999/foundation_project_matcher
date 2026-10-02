"""Assemble a training set from every label source we have.

Priority (a stronger source replaces a weaker one on the same user×SKU):

    rule  <  synthetic  <  expert  <  tester  <  behavior

That is the bootstrap: rule-score pseudo-labels fill the catalog, then
real ratings and clicks overwrite them as they arrive.

Usage:
    python3 -m ml.bootstrap_labels
    python3 -m ml.bootstrap_labels --no-synthetic --no-rules
    python3 -m ml.train --logs data/labels/training.jsonl --target rated
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.labels import (
    SOURCE_PRIORITY,
    expert_csv_to_events,
    load_jsonl,
    merge_by_priority,
    rule_weak_labels,
    source_counts,
)

LABELS = ROOT / "data" / "labels"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=ROOT / "foundations.csv")
    p.add_argument("--expert", type=Path, default=LABELS / "expert_labels.csv")
    p.add_argument("--tester", type=Path, default=LABELS / "tester.jsonl")
    p.add_argument("--behavior", type=Path, default=ROOT / "data" / "interactions.live.jsonl")
    p.add_argument("--synthetic", type=Path, default=ROOT / "data" / "interactions.jsonl")
    p.add_argument("--out", type=Path, default=LABELS / "training.jsonl")
    p.add_argument("--no-rules", action="store_true", help="Skip rule-score pseudo-labels")
    p.add_argument("--no-synthetic", action="store_true", help="Skip generated users")
    p.add_argument("--rules-top", type=int, default=8,
                   help="Candidates per synthetic-grid user for rule weak labels")
    args = p.parse_args(argv)

    with args.csv.open() as f:
        catalog = [rules.Row.from_dict(d) for d in csv.DictReader(f)]

    buckets: list[tuple[str, list[dict]]] = []

    if not args.no_rules:
        print("Generating rule-score pseudo-labels (weak)…")
        buckets.append(("rule", rule_weak_labels(catalog, top_k=args.rules_top)))

    if not args.no_synthetic:
        syn = load_jsonl(args.synthetic, default_source="synthetic")
        buckets.append(("synthetic", syn))

    expert = expert_csv_to_events(args.expert, catalog) if args.expert.exists() else []
    buckets.append(("expert", expert))

    tester = load_jsonl(args.tester, default_source="tester")
    live = load_jsonl(args.behavior, default_source="tester")
    # Streamlit live file is beta/tester until a production pipeline exists.
    buckets.append(("tester", tester + live))

    print("Source sizes before merge:")
    for name, evs in buckets:
        print(f"  {name:10s} {len(evs):5d}")

    merged = merge_by_priority(evs for _, evs in buckets)
    counts = source_counts(merged)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ev in merged:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(merged)} unique (user, SKU) labels → {args.out}")
    print("After override (stronger source wins):")
    for src in sorted(SOURCE_PRIORITY, key=SOURCE_PRIORITY.get):
        print(f"  {src:10s} {counts.get(src, 0):5d}")
    print("\nAs real behavior arrives, re-run this command; those rows replace "
          "rule / synthetic / expert labels on the same pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
