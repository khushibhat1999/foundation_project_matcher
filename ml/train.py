"""Train a compatibility / ranking model on interaction logs.

Usage:
    python3 -m ml.train --logs data/interactions.jsonl --model logreg
    python3 -m ml.train --logs data/interactions.jsonl --model xgboost --target clicked
    python3 -m ml.train --logs data/interactions.jsonl --model pairwise --target relevance

The trainer:
  1. Joins each log row back to the live catalog and *re-scores* it
     with the current rules (so features stay in sync if rules change).
  2. Holds out 20% of *sessions* (not rows) for evaluation.
  3. Fits the chosen model.
  4. Reports AUC / AP plus NDCG@5/@10 against a rules-only baseline.
  5. Writes ``ml/artifacts/ranker.joblib``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.features import FEATURE_NAMES, extract_feature_vector
from ml.interactions import TARGETS, binary_label, group_by_session, load_events, relevance_score

try:
    from xgboost import XGBClassifier, XGBRanker
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def _catalog_index(rows: list[rules.Row]) -> dict[tuple[str, str, str], rules.Row]:
    return {(r.brand, r.product, r.shade_name): r for r in rows}


def _rebuild_scored(event: dict, index: dict) -> rules.Scored | None:
    c = event["candidate"]
    row = index.get((c["brand"], c["product"], c["shade_name"]))
    if row is None:
        return None
    scored, _ = rules.evaluate(row, event["profile"])
    if scored is None and event.get("source") in {"expert", "tester"}:
        from ml.labels import _score_unfiltered
        scored = _score_unfiltered(row, event["profile"])
    return scored


def build_matrix(
    events: list[dict],
    index: dict,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[rules.Scored]]:
    X, y, graded, kept, scored_kept = [], [], [], [], []
    for ev in events:
        scored = _rebuild_scored(ev, index)
        if scored is None:
            continue
        X.append(extract_feature_vector(ev["profile"], scored))
        y.append(binary_label(ev, target))
        graded.append(relevance_score(ev))
        kept.append(ev)
        scored_kept.append(scored)
    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(graded, dtype=float),
        kept,
        scored_kept,
    )


def split_sessions(events: list[dict], seed: int = 7, test_frac: float = 0.2
                   ) -> tuple[list[dict], list[dict]]:
    sessions = list(group_by_session(events))
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac))
    test_ids = set(sessions[:n_test])
    train = [e for e in events if e["session_id"] not in test_ids]
    test = [e for e in events if e["session_id"] in test_ids]
    return train, test


def _ndcg_at(rels: list[float], k: int) -> float:
    rels_k = rels[:k]
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels_k))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def session_ndcg(events: list[dict], scores: np.ndarray, k: int) -> float:
    by_sess: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ev, score in zip(events, scores):
        by_sess[ev["session_id"]].append((score, float(relevance_score(ev))))
    vals = []
    for pairs in by_sess.values():
        pairs.sort(key=lambda t: -t[0])
        rels = [r for _, r in pairs]
        if any(rels):
            vals.append(_ndcg_at(rels, k))
    return float(np.mean(vals)) if vals else 0.0


def pairwise_arrays(X: np.ndarray, y: np.ndarray, events: list[dict]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Build RankSVM-style difference pairs within each session."""
    by_sess: dict[str, list[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        by_sess[ev["session_id"]].append(i)
    diffs, labels = [], []
    for idxs in by_sess.values():
        pos = [i for i in idxs if y[i] == 1]
        neg = [i for i in idxs if y[i] == 0]
        if not pos or not neg:
            continue
        for i in pos:
            for j in neg:
                diffs.append(X[i] - X[j])
                labels.append(1)
                diffs.append(X[j] - X[i])
                labels.append(0)
    if not diffs:
        return np.empty((0, X.shape[1])), np.empty((0,))
    return np.asarray(diffs, dtype=float), np.asarray(labels, dtype=int)


def fit_logreg(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=400, class_weight="balanced")),
    ])
    clf.fit(X, y, model__sample_weight=sample_weight)
    return clf


def fit_pairwise(X: np.ndarray, y: np.ndarray, events: list[dict]) -> LogisticRegression:
    Xd, yd = pairwise_arrays(X, y, events)
    if len(yd) < 20:
        raise RuntimeError(
            "Not enough within-session pos/neg pairs for pairwise training. "
            "Generate more sessions or use --model logreg."
        )
    # No intercept: score(x) = w·x is consistent with training on x_i - x_j.
    clf = LogisticRegression(max_iter=400, fit_intercept=False, class_weight="balanced")
    clf.fit(Xd, yd)
    return clf


def fit_xgboost(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed. `pip install xgboost` or use --model logreg.")
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        n_jobs=2,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def fit_xgbranker(X: np.ndarray, graded: np.ndarray, events: list[dict]):
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed.")
    order = sorted(range(len(events)), key=lambda i: events[i]["session_id"])
    Xg = X[order]
    yg = graded[order]
    groups, last, count = [], None, 0
    for i in order:
        sid = events[i]["session_id"]
        if last is None:
            last, count = sid, 1
        elif sid == last:
            count += 1
        else:
            groups.append(count)
            last, count = sid, 1
    if count:
        groups.append(count)
    model = XGBRanker(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="rank:ndcg",
        n_jobs=2,
    )
    model.fit(Xg, yg, group=groups)
    return model


def evaluate_split(name: str, predict_scores, events: list[dict],
                   y: np.ndarray, graded: np.ndarray, X: np.ndarray) -> dict:
    scores = np.asarray(predict_scores(X), dtype=float)
    metrics = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else 0.0,
    }
    if len(np.unique(y)) > 1:
        metrics["auc"] = float(roc_auc_score(y, scores))
        metrics["ap"] = float(average_precision_score(y, scores))
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
    metrics["ndcg@5"] = session_ndcg(events, scores, 5)
    metrics["ndcg@10"] = session_ndcg(events, scores, 10)
    print(f"  {name:18s}  n={metrics['n']:5d}  pos={metrics['pos_rate']:.2f}  "
          f"AUC={metrics['auc']:.3f}  AP={metrics['ap']:.3f}  "
          f"NDCG@5={metrics['ndcg@5']:.3f}  NDCG@10={metrics['ndcg@10']:.3f}")
    return metrics


def _top_coefficients(model, k: int = 12) -> list[tuple[str, float]]:
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps["model"]
        scaler = model.named_steps.get("scaler")
        coef = est.coef_.ravel()
        if scaler is not None:
            # Coefficients are in scaled space; still useful for direction.
            pass
    else:
        coef = est.coef_.ravel()
    pairs = sorted(zip(FEATURE_NAMES, coef), key=lambda t: -abs(t[1]))
    return [(n, float(c)) for n, c in pairs[:k]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_logs = ROOT / "data" / "labels" / "training.jsonl"
    if not default_logs.exists():
        default_logs = ROOT / "data" / "interactions.jsonl"
    p.add_argument("--logs", type=Path, default=default_logs,
                   help="JSONL labels (default: data/labels/training.jsonl if present)")
    p.add_argument("--csv", type=Path, default=ROOT / "foundations.csv")
    p.add_argument("--model", choices=["logreg", "xgboost", "pairwise", "xgbranker"],
                   default="logreg")
    p.add_argument("--target", choices=TARGETS, default="clicked",
                   help="Training label: clicked | saved | purchased | rated | relevance")
    p.add_argument("--out", type=Path, default=ROOT / "ml" / "artifacts" / "ranker.joblib")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    events = load_events(args.logs)
    if len(events) < 50:
        print("Need more labeled rows. Run: python3 -m ml.bootstrap_labels", file=sys.stderr)
        return 2

    from collections import Counter
    src_mix = Counter(e.get("source") or "unknown" for e in events)
    print("Label sources:", dict(src_mix))

    with args.csv.open() as f:
        catalog = [rules.Row.from_dict(d) for d in csv.DictReader(f)]
    index = _catalog_index(catalog)

    train_ev, test_ev = split_sessions(events, seed=args.seed)
    Xtr, ytr, gtr, train_kept, _ = build_matrix(train_ev, index, args.target)
    Xte, yte, gte, test_kept, _ = build_matrix(test_ev, index, args.target)
    wtr = np.asarray([float(e.get("sample_weight") or 1.0) for e in train_kept], dtype=float)

    print(f"Train impressions={len(ytr)} (sessions≈{len({e['session_id'] for e in train_kept})})  "
          f"Test impressions={len(yte)}  target={args.target}  model={args.model}")

    if args.model == "logreg":
        model = fit_logreg(Xtr, ytr, sample_weight=wtr)
        predict = lambda X: model.predict_proba(X)[:, 1]
        model_type = "logreg"
    elif args.model == "xgboost":
        model = fit_xgboost(Xtr, ytr, sample_weight=wtr)
        predict = lambda X: model.predict_proba(X)[:, 1]
        model_type = "xgboost"
    elif args.model == "pairwise":
        model = fit_pairwise(Xtr, ytr, train_kept)
        predict = lambda X: model.decision_function(X)
        model_type = "pairwise_logreg"
    else:
        model = fit_xgbranker(Xtr, gtr, train_kept)
        predict = lambda X: model.predict(X)
        model_type = "xgbranker"

    print("Evaluation (session-held-out):")
    # Rules baseline: use stored rule_score from the log (what the user saw).
    rule_scores = np.asarray([e.get("rule_score", 0) for e in test_kept], dtype=float)
    base = evaluate_split("rules baseline", lambda _X: rule_scores, test_kept, yte, gte, Xte)
    model_metrics = evaluate_split(args.model, predict, test_kept, yte, gte, Xte)

    if args.model in {"logreg", "pairwise"}:
        print("\nTop coefficients (sign = direction toward a positive label):")
        for name, coef in _top_coefficients(model):
            print(f"  {coef:+7.3f}  {name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "model_type": model_type,
        "target": args.target,
        "metrics": {"baseline": base, "model": model_metrics},
    }
    joblib.dump(artifact, args.out)
    print(f"\nSaved {args.model} ranker → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
