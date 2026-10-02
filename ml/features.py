"""Feature extraction for (user profile, rule-scored candidate) pairs.

Features are deliberately a mix of:
  * user attributes (skin type, undertone, depth, prefs)
  * product attributes (finish, coverage, price, SPF)
  * *cross* features the model can use to learn residual patterns
    the rules only partly capture, e.g. dry × satin, warm × warm shade.

The vector is a fixed-length dict → list so train and serve cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rules

SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive", "acne_prone"]
USER_DEPTHS = ["fair", "light", "medium", "tan", "deep", "very_deep"]
USER_UNDERTONES = ["cool", "warm", "neutral", "olive"]
USER_FINISHES = ["matte", "natural", "satin", "dewy"]
USER_COVERAGES = ["light", "medium", "full"]
PREF_LEVELS = ["required", "preferred", "no_pref"]

PROD_DEPTHS = ["Fair", "Light", "Light-Medium", "Medium", "Medium-Tan", "Tan", "Deep", "Rich"]
PROD_UNDERTONES = ["Cool", "Neutral", "Warm", "Olive", "Neutral-Cool", "Neutral-Warm"]
PROD_FINISHES = ["Matte", "Natural", "Satin", "Radiant", "Dewy", "Luminous"]
PROD_COVERAGES = ["Sheer", "Light", "Light-Medium", "Medium", "Medium-Full", "Full"]

DEWY_FAMILY = {"Dewy", "Radiant", "Luminous"}
WARM_SHADES = {"Warm", "Neutral-Warm"}
COOL_SHADES = {"Cool", "Neutral-Cool"}


def _onehot(value: str | None, vocab: list[str], prefix: str) -> dict[str, float]:
    return {f"{prefix}={v}": 1.0 if value == v else 0.0 for v in vocab}


def _multihot(values: list[str] | None, vocab: list[str], prefix: str) -> dict[str, float]:
    present = set(values or [])
    return {f"{prefix}={v}": 1.0 if v in present else 0.0 for v in vocab}


def _pref_level(profile: dict, key: str) -> str:
    val = profile.get(key, "no_pref")
    if isinstance(val, dict):
        val = val.get("selected", "no_pref")
    return val if val in PREF_LEVELS else "no_pref"


def extract_features(profile: dict, scored: rules.Scored) -> dict[str, float]:
    """Return a name → float feature map for one (user, candidate) pair."""
    row = scored.row
    user_types = rules._user_skin_types(profile)
    user_finishes = []
    finish_block = profile.get("finish")
    if isinstance(finish_block, dict):
        user_finishes = finish_block.get("selected") or []
    elif isinstance(finish_block, list):
        user_finishes = finish_block

    coverage_sel = profile.get("coverage")
    if isinstance(coverage_sel, dict):
        coverage_sel = coverage_sel.get("selected")

    depth = profile.get("skin_depth")
    undertone = profile.get("undertone")
    cap = rules._price_cap(profile)
    price_ratio = row.price / cap if cap and cap < 1e8 else 0.0

    accepted_depths = rules.DEPTH_MAP.get(depth or "", [])
    exact_depth = 1.0 if accepted_depths and row.depth == accepted_depths[0] else 0.0
    exact_ut = 1.0 if undertone and row.undertone.lower() == undertone else 0.0

    feats: dict[str, float] = {}
    feats.update(_multihot(user_types, SKIN_TYPES, "user_skin"))
    feats.update(_onehot(depth, USER_DEPTHS, "user_depth"))
    feats.update(_onehot(undertone, USER_UNDERTONES, "user_undertone"))
    feats.update(_multihot(user_finishes, USER_FINISHES, "user_finish"))
    feats.update(_onehot(coverage_sel, USER_COVERAGES, "user_coverage"))
    feats.update(_onehot(_pref_level(profile, "fragrance_free"), PREF_LEVELS, "user_ff"))
    feats.update(_onehot(_pref_level(profile, "non_comedogenic"), PREF_LEVELS, "user_nc"))
    feats.update(_onehot(profile.get("spf_needed") or "no_pref",
                         ["yes", "yes_30", "no_pref"], "user_spf"))

    feats.update(_onehot(row.depth, PROD_DEPTHS, "prod_depth"))
    feats.update(_onehot(row.undertone, PROD_UNDERTONES, "prod_undertone"))
    feats.update(_onehot(row.finish, PROD_FINISHES, "prod_finish"))
    feats.update(_onehot(row.coverage, PROD_COVERAGES, "prod_coverage"))
    feats["prod_price"] = row.price / 100.0
    feats["prod_price_ratio"] = min(price_ratio, 2.0)
    feats["prod_spf"] = min(row.spf / 50.0, 1.0)
    feats["prod_fragrance_free"] = 1.0 if row.fragrance_free else 0.0
    feats["prod_hydrating"] = 1.0 if rules._contains_any(row.ingredients, rules.HYDRATION_SIGNALS) else 0.0
    feats["prod_longwear"] = 1.0 if rules._contains_any(row.ingredients, rules.LONGWEAR_SIGNALS) else 0.0

    for name, mx in rules.COMPONENT_MAX.items():
        feats[f"rule_{name}"] = scored.components.get(name, 0) / mx
    feats["rule_total"] = scored.total / rules.MAX_SCORE
    feats["exact_depth"] = exact_depth
    feats["exact_undertone"] = exact_ut
    feats["finish_in_pref"] = 1.0 if (row.finish.lower() in {f.lower() for f in user_finishes}
                                      or row.finish in (rules._finish_maps(profile) or [])) else 0.0

    # Cross features — residual patterns the rules only partly encode.
    dry = "dry" in user_types
    oily = "oily" in user_types or "acne_prone" in user_types
    sensitive = "sensitive" in user_types
    feats["x_dry_dewy"] = 1.0 if dry and row.finish in DEWY_FAMILY else 0.0
    feats["x_dry_satin"] = 1.0 if dry and row.finish == "Satin" else 0.0
    feats["x_dry_matte"] = 1.0 if dry and row.finish == "Matte" else 0.0
    feats["x_oily_matte"] = 1.0 if oily and row.finish == "Matte" else 0.0
    feats["x_oily_dewy"] = 1.0 if oily and row.finish in DEWY_FAMILY else 0.0
    feats["x_sensitive_fragrance"] = 1.0 if sensitive and not row.fragrance_free else 0.0
    feats["x_warm_warmshade"] = 1.0 if undertone == "warm" and row.undertone in WARM_SHADES else 0.0
    feats["x_cool_coolshade"] = 1.0 if undertone == "cool" and row.undertone in COOL_SHADES else 0.0
    feats["x_olive_oliveshade"] = 1.0 if undertone == "olive" and row.undertone == "Olive" else 0.0
    feats["x_neutral_neutralshade"] = 1.0 if undertone == "neutral" and "Neutral" in row.undertone else 0.0
    return feats


def feature_names() -> list[str]:
    """Stable feature order derived from a dummy (profile, row) pair."""
    dummy_row = rules.Row(
        brand="x", product="x", shade_name="x", shade_code="x",
        depth="Fair", undertone="Neutral", finish="Natural", coverage="Medium",
        skin_types="All", spf=0, fragrance_free=True, price=20.0, ingredients="",
    )
    dummy_scored = rules.Scored(
        row=dummy_row,
        components={k: 0 for k in rules.COMPONENT_MAX},
    )
    dummy_profile = {
        "skin_type": {"selected": []},
        "undertone": "neutral",
        "skin_depth": "medium",
        "finish": {"selected": ["natural"], "maps_to_dataset": ["Natural"]},
        "coverage": {"selected": "medium", "maps_to_dataset": ["Medium"]},
        "fragrance_free": "no_pref",
        "non_comedogenic": "no_pref",
        "spf_needed": "no_pref",
        "price_range": {"selected": "prestige", "max_usd": 60},
    }
    return list(extract_features(dummy_profile, dummy_scored).keys())


FEATURE_NAMES: list[str] = feature_names()


def extract_feature_vector(profile: dict, scored: rules.Scored) -> list[float]:
    feats = extract_features(profile, scored)
    return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]


def candidate_key(row: rules.Row) -> dict[str, Any]:
    return {
        "brand": row.brand,
        "product": row.product,
        "shade_name": row.shade_name,
        "shade_code": row.shade_code,
    }
