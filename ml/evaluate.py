"""Evaluate the matcher offline (ranking quality) and as a product (CTR, satisfaction, diversity).

Usage:
    python3 -m ml.evaluate
    python3 -m ml.evaluate --logs data/labels/training.jsonl --live data/interactions.live.jsonl

Offline (session-held-out, rules vs ML):
  Top-k accuracy, Precision@3, MAP, NDCG@3/@5, calibration (ECE, Brier).

Product (impressions + feedback):
  CTR, mean rating, good-match yes/no rate, brand/price diversity,
  and the same sliced by skin type.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.interactions import group_by_session, load_events, relevance_score
from ml.labels import _label_strength, _score_unfiltered
from ml.ranker import load_ranker
from ml.train import _catalog_index, _rebuild_scored, split_sessions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_relevant(ev: dict) -> bool:
    rated = ev.get("rated")
    if rated is not None and int(rated) >= 4:
        return True
    return bool(ev.get("clicked") or ev.get("saved") or ev.get("purchased"))


def _is_good_match(ev: dict) -> bool | None:
    """Yes/no from an explicit rating. None if the user did not rate."""
    rated = ev.get("rated")
    if rated is None:
        return None
    return int(rated) >= 4


def _skin_segment(profile: dict) -> str:
    st = profile.get("skin_type", {})
    selected = st.get("selected", []) if isinstance(st, dict) else (st or [])
    if not selected:
        return "unknown"
    return ",".join(sorted(selected))


def _ndcg_at(rels: list[float], k: int) -> float:
    rels_k = rels[:k]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels_k))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _average_precision(rels: list[int]) -> float:
    hits, running = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            running += hits / i
    return running / hits if hits else 0.0


def rank_session(events: list[dict], scores: list[float]) -> list[dict]:
    order = sorted(range(len(events)), key=lambda i: -scores[i])
    return [events[i] for i in order]


def session_ranking_metrics(ranked: list[dict], k: int = 3) -> dict:
    rel_bin = [1 if _is_relevant(e) else 0 for e in ranked]
    rel_grad = [float(relevance_score(e)) for e in ranked]
    has_pos = any(rel_bin)
    return {
        "top_k_hit": int(has_pos and any(rel_bin[:k])),
        "precision_at_k": (sum(rel_bin[:k]) / k) if k else 0.0,
        "map": _average_precision(rel_bin) if has_pos else None,
        "ndcg": _ndcg_at(rel_grad, k) if has_pos else None,
        "has_positive": has_pos,
    }


def aggregate_ranking(per_session: list[dict], k: int) -> dict:
    usable = [m for m in per_session if m["has_positive"]]
    n = len(usable)
    if not n:
        return {"n_sessions": 0, f"top{k}_accuracy": 0.0, f"precision@{k}": 0.0,
                "MAP": 0.0, f"NDCG@{k}": 0.0}
    maps = [m["map"] for m in usable if m["map"] is not None]
    ndcgs = [m["ndcg"] for m in usable if m["ndcg"] is not None]
    return {
        "n_sessions": n,
        f"top{k}_accuracy": sum(m["top_k_hit"] for m in usable) / n,
        f"precision@{k}": sum(m["precision_at_k"] for m in usable) / n,
        "MAP": float(np.mean(maps)) if maps else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def calibration(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> dict:
    """Expected calibration error + Brier score for compatibility probabilities."""
    p = np.clip(np.asarray(p_hat, dtype=float), 0.0, 1.0)
    y = np.asarray(y_true, dtype=float)
    if len(y) == 0:
        return {"ece": 0.0, "brier": 0.0, "bins": []}
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        frac = float(mask.mean())
        ece += frac * abs(acc - conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "n": int(mask.sum()), "pred": conf, "actual": acc})
    return {"ece": ece, "brier": brier, "bins": bins}


def diversity(shown: list[dict]) -> dict:
    brands = [e.get("candidate", {}).get("brand") for e in shown if e.get("candidate")]
    prices = [float(e.get("candidate", {}).get("price_usd") or 0) for e in shown if e.get("candidate")]
    n = len(shown) or 1
    counts = Counter(brands)
    probs = np.array([c / n for c in counts.values()], dtype=float) if counts else np.array([1.0])
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    return {
        "n_shown": len(shown),
        "unique_brands": len(counts),
        "brand_entropy": entropy,
        "unique_brand_share": len(counts) / n,
        "price_mean": float(np.mean(prices)) if prices else 0.0,
        "price_std": float(np.std(prices)) if len(prices) > 1 else 0.0,
        "price_min": float(min(prices)) if prices else 0.0,
        "price_max": float(max(prices)) if prices else 0.0,
    }


# ---------------------------------------------------------------------------
# Offline ranking: rules vs ML on a session holdout
# ---------------------------------------------------------------------------

def score_events_ml(events: list[dict], index: dict, ranker) -> list[float]:
    out = []
    for ev in events:
        scored = _rebuild_scored(ev, index)
        if scored is None:
            scored = None
            c = ev["candidate"]
            row = index.get((c["brand"], c["product"], c["shade_name"]))
            if row is not None:
                scored = _score_unfiltered(row, ev["profile"])
        if scored is None:
            out.append(0.0)
        else:
            out.append(ranker.predict_proba(ev["profile"], scored))
    return out


def offline_report(events: list[dict], index: dict, ranker, seed: int = 7) -> dict:
    _, test = split_sessions(events, seed=seed)
    if len(test) < 10:
        test = events
    y = np.array([1 if _is_relevant(e) else 0 for e in test], dtype=float)
    rule_scores = np.array([float(e.get("rule_score") or 0) / 100.0 for e in test])
    try:
        ml_scores = np.array(score_events_ml(test, index, ranker), dtype=float)
        have_ml = True
    except Exception:
        ml_scores = rule_scores.copy()
        have_ml = False

    def by_ranker(scores: np.ndarray, k: int) -> dict:
        per = []
        idx_by_sess: dict[str, list[int]] = defaultdict(list)
        for i, ev in enumerate(test):
            idx_by_sess[ev["session_id"]].append(i)
        for idxs in idx_by_sess.values():
            ranked = rank_session([test[i] for i in idxs], [float(scores[i]) for i in idxs])
            per.append(session_ranking_metrics(ranked, k=k))
        return aggregate_ranking(per, k)

    def metrics_for(scores: np.ndarray) -> dict:
        at3 = by_ranker(scores, 3)
        at5 = by_ranker(scores, 5)
        return {
            **at3,
            "top5_accuracy": at5.get("top5_accuracy"),
            "NDCG@5": at5.get("NDCG@5"),
        }

    cal = calibration(y, ml_scores) if have_ml else calibration(y, rule_scores)
    return {
        "n_test_impressions": len(test),
        "n_test_sessions": len(group_by_session(test)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "used_ml": have_ml,
        "rules": metrics_for(rule_scores),
        "ml": metrics_for(ml_scores),
        "calibration": cal,
        "by_skin_type": _offline_by_segment(test, rule_scores, ml_scores, have_ml),
    }


def _offline_by_segment(test, rule_scores, ml_scores, have_ml) -> dict:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, ev in enumerate(test):
        buckets[_skin_segment(ev.get("profile") or {})].append(i)
    out = {}
    for seg, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 8:
            continue
        evs = [test[i] for i in idxs]
        rs = [float(rule_scores[i]) for i in idxs]
        ms = [float(ml_scores[i]) for i in idxs]
        idx_by_sess: dict[str, list[int]] = defaultdict(list)
        for j, ev in enumerate(evs):
            idx_by_sess[ev["session_id"]].append(j)

        def agg(scores):
            per = []
            for local in idx_by_sess.values():
                ranked = rank_session([evs[j] for j in local], [scores[j] for j in local])
                per.append(session_ranking_metrics(ranked, k=3))
            return aggregate_ranking(per, 3)

        out[seg] = {"n": len(idxs), "rules": agg(rs), "ml": agg(ms) if have_ml else agg(rs)}
    return out


# ---------------------------------------------------------------------------
# Product metrics from impression + feedback logs
# ---------------------------------------------------------------------------

def collapse_impressions(events: list[dict]) -> list[dict]:
    """One row per (session, SKU); keep the strongest feedback if several land."""
    chosen: dict[tuple, dict] = {}
    for ev in events:
        c = ev.get("candidate") or {}
        key = (ev.get("session_id"), c.get("brand"), c.get("product"), c.get("shade_name"))
        prev = chosen.get(key)
        if prev is None or _label_strength(ev) >= _label_strength(prev):
            chosen[key] = ev
    return list(chosen.values())


def product_report(events: list[dict]) -> dict:
    rows = collapse_impressions(events)
    n = len(rows) or 1
    n_click = sum(1 for e in rows if e.get("clicked"))
    rated = [int(e["rated"]) for e in rows if e.get("rated") is not None]
    yes = sum(1 for e in rows if _is_good_match(e) is True)
    no = sum(1 for e in rows if _is_good_match(e) is False)
    judged = yes + no

    # Diversity on each session's shown set.
    by_sess = group_by_session(rows)
    divs = [diversity(evs) for evs in by_sess.values() if evs]
    def mean_key(key):
        vals = [d[key] for d in divs] or [0.0]
        return float(np.mean(vals))

    by_skin: dict[str, dict] = {}
    skin_groups: dict[str, list[dict]] = defaultdict(list)
    for e in rows:
        skin_groups[_skin_segment(e.get("profile") or {})].append(e)
    for seg, evs in sorted(skin_groups.items(), key=lambda kv: -len(kv[1])):
        r = [int(x["rated"]) for x in evs if x.get("rated") is not None]
        y = sum(1 for x in evs if _is_good_match(x) is True)
        n_ = sum(1 for x in evs if _is_good_match(x) is False)
        by_skin[seg] = {
            "n": len(evs),
            "ctr": sum(1 for x in evs if x.get("clicked")) / len(evs),
            "mean_rating": float(np.mean(r)) if r else None,
            "good_match_rate": y / (y + n_) if (y + n_) else None,
        }

    return {
        "n_impressions": len(rows),
        "n_sessions": len(by_sess),
        "ctr": n_click / n,
        "n_rated": len(rated),
        "mean_satisfaction": float(np.mean(rated)) if rated else None,
        "good_match_yes": yes,
        "good_match_no": no,
        "good_match_rate": yes / judged if judged else None,
        "diversity": {
            "avg_unique_brands": mean_key("unique_brands"),
            "avg_brand_entropy": mean_key("brand_entropy"),
            "avg_price": mean_key("price_mean"),
            "avg_price_std": mean_key("price_std"),
            "price_span_mean": mean_key("price_max") - mean_key("price_min"),
        },
        "by_skin_type": by_skin,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%"


def _fmt_num(x, nd=3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def print_report(offline: dict, product: dict) -> None:
    print("\n==========  OFFLINE EVALUATION  ==========")
    print(f"  test sessions={offline['n_test_sessions']}  "
          f"impressions={offline['n_test_impressions']}  "
          f"positive rate={offline['positive_rate']:.2f}")
    print(f"  {'':18s}  {'Top-3 acc':>10s}  {'P@3':>8s}  {'MAP':>8s}  {'NDCG@3':>8s}  {'NDCG@5':>8s}")
    for name in ("rules", "ml"):
        m = offline[name]
        print(f"  {name:18s}  {_fmt_pct(m.get('top3_accuracy')):>10s}  "
              f"{_fmt_num(m.get('precision@3')):>8s}  {_fmt_num(m.get('MAP')):>8s}  "
              f"{_fmt_num(m.get('NDCG@3')):>8s}  {_fmt_num(m.get('NDCG@5')):>8s}")
    cal = offline["calibration"]
    print(f"\n  Calibration of ML compatibility scores")
    print(f"    ECE={cal['ece']:.3f}   Brier={cal['brier']:.3f}   (lower is better)")
    if cal["bins"]:
        print(f"    {'bin':>11s}  {'n':>5s}  {'pred':>6s}  {'actual':>6s}")
        for b in cal["bins"]:
            print(f"    {b['lo']:.1f}–{b['hi']:.1f}  {b['n']:5d}  {b['pred']:6.2f}  {b['actual']:6.2f}")

    if offline.get("by_skin_type"):
        print("\n  By skin-type segment (Precision@3)")
        for seg, block in offline["by_skin_type"].items():
            print(f"    {seg:28s}  n={block['n']:4d}  rules P@3={block['rules'].get('precision@3', 0):.3f}"
                  f"  ml P@3={block['ml'].get('precision@3', 0):.3f}")

    print("\n==========  PRODUCT EVALUATION  ==========")
    print(f"  impressions={product['n_impressions']}  sessions={product['n_sessions']}")
    print(f"  CTR (clicked / shown)          {_fmt_pct(product['ctr'])}")
    print(f"  Mean satisfaction (1–5)        {_fmt_num(product['mean_satisfaction'], 2)}  "
          f"(n={product['n_rated']})")
    print(f"  “Good match?” yes / no         {product['good_match_yes']} / {product['good_match_no']}"
          f"   rate={_fmt_pct(product['good_match_rate'])}")
    d = product["diversity"]
    print(f"  Diversity per session")
    print(f"    unique brands (avg)          {_fmt_num(d['avg_unique_brands'], 2)}")
    print(f"    brand entropy (avg bits)     {_fmt_num(d['avg_brand_entropy'], 2)}")
    print(f"    price mean ± std             ${d['avg_price']:.2f} ± {d['avg_price_std']:.2f}")
    print(f"    price span (max−min, avg)    ${d['price_span_mean']:.2f}")
    if product.get("by_skin_type"):
        print("\n  By skin-type segment")
        print(f"    {'segment':28s}  {'n':>5s}  {'CTR':>7s}  {'rating':>7s}  {'good%':>7s}")
        for seg, block in product["by_skin_type"].items():
            print(f"    {seg:28s}  {block['n']:5d}  {_fmt_pct(block['ctr']):>7s}  "
                  f"{_fmt_num(block['mean_rating'], 2):>7s}  {_fmt_pct(block['good_match_rate']):>7s}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, default=ROOT / "data" / "labels" / "training.jsonl")
    p.add_argument("--live", type=Path, default=ROOT / "data" / "interactions.live.jsonl")
    p.add_argument("--tester", type=Path, default=ROOT / "data" / "labels" / "tester.jsonl")
    p.add_argument("--csv", type=Path, default=ROOT / "foundations.csv")
    p.add_argument("--model", type=Path, default=ROOT / "ml" / "artifacts" / "ranker.joblib")
    p.add_argument("--out", type=Path, default=ROOT / "ml" / "artifacts" / "eval_report.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    labeled = load_events(args.logs) if args.logs.exists() else []
    if len(labeled) < 20:
        print("Need labeled sessions. Run python3 -m ml.bootstrap_labels first.", file=sys.stderr)
        return 2

    with args.csv.open() as f:
        catalog = [rules.Row.from_dict(d) for d in csv.DictReader(f)]
    index = _catalog_index(catalog)
    ranker = load_ranker(args.model) if args.model.exists() else None
    if ranker is None:
        print("No ranker.joblib — offline ML metrics will copy the rules ranking.")

    offline = offline_report(labeled, index, ranker, seed=args.seed)

    product_events = list(labeled)
    for extra in (args.live, args.tester):
        if extra.exists():
            product_events.extend(load_events(extra))
    product = product_report(product_events)

    print_report(offline, product)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"offline": offline, "product": product}, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
