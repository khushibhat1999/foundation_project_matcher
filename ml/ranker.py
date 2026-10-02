"""Serve a trained ranker: score rule-approved candidates and reorder them.

``Ranker.rerank`` never adds products the rules filtered out. Optional
``alpha`` blends rule score with model probability:

    final = (1 - alpha) * rule_total/100 + alpha * p_ml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules
from ml.features import FEATURE_NAMES, extract_feature_vector

DEFAULT_MODEL = ROOT / "ml" / "artifacts" / "ranker.joblib"


@dataclass
class Ranked:
    scored: rules.Scored
    ml_score: float
    final_score: float
    features: list[float] = field(default_factory=list)


class Ranker:
    def __init__(self, artifact: dict[str, Any]):
        self.model = artifact["model"]
        self.feature_names: list[str] = artifact["feature_names"]
        self.model_type: str = artifact["model_type"]
        self.target: str = artifact.get("target", "clicked")
        self.metrics: dict = artifact.get("metrics", {})
        if self.feature_names != FEATURE_NAMES:
            missing = set(FEATURE_NAMES) - set(self.feature_names)
            extra = set(self.feature_names) - set(FEATURE_NAMES)
            if missing or extra:
                raise ValueError(
                    f"Feature schema mismatch vs current code. "
                    f"missing={sorted(missing)} extra={sorted(extra)}. Retrain."
                )

    def predict_proba(self, profile: dict, scored: rules.Scored) -> float:
        x = np.asarray([extract_feature_vector(profile, scored)], dtype=float)
        if self.model_type == "pairwise_logreg":
            # Trained on diffs with no intercept; score is w·x.
            raw = float(self.model.decision_function(x)[0])
            return 1.0 / (1.0 + np.exp(-raw))
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x)[0, 1])
        if hasattr(self.model, "predict"):
            return float(self.model.predict(x)[0])
        raise TypeError(f"Don't know how to score model type {type(self.model)}")

    def rerank(
        self,
        profile: dict,
        scored: list[rules.Scored],
        top: int | None = None,
        alpha: float = 0.7,
    ) -> list[Ranked]:
        ranked: list[Ranked] = []
        for s in scored:
            p = self.predict_proba(profile, s)
            final = (1.0 - alpha) * (s.total / rules.MAX_SCORE) + alpha * p
            ranked.append(Ranked(
                scored=s,
                ml_score=p,
                final_score=final,
                features=extract_feature_vector(profile, s),
            ))
        ranked.sort(key=lambda r: (-r.final_score, r.scored.row.price, r.scored.row.brand))
        return ranked[:top] if top is not None else ranked


def load_ranker(path: Path | None = None) -> Ranker:
    path = Path(path) if path else DEFAULT_MODEL
    artifact = joblib.load(path)
    return Ranker(artifact)
