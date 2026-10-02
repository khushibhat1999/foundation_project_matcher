"""Rule-based filtering & scoring engine for the foundation matcher.

This is the first-pass, deterministic ranker. It runs BEFORE any ML model
and its job is:

  1. FILTER: eliminate rows that clearly can't work (wrong depth bucket,
     over budget, missing required SPF, etc).
  2. SCORE: for surviving rows, sum contributions across 5 named
     components so the final score is transparent and auditable:

        total = tone_match + undertone_match + skin_type_match
              + finish_match + preference_match     (max 100)

Everything here is pure data + pure functions: no I/O, no CSV parsing, no
CLI. `match.py` composes these with I/O; `web/app.js` mirrors them.

To add a new rule, either:
  * add a `Filter` to FILTER_RULES  (returns FilterResult), or
  * add a `Scorer` to SCORE_RULES   (returns list[Contribution]).

Never mix filter + score logic in one rule — keep it declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Component weights (max points per component). Total possible = 100.
# ---------------------------------------------------------------------------

COMPONENT_MAX: dict[str, int] = {
    "tone_match":       25,
    "undertone_match":  25,
    "finish_match":     15,
    "skin_type_match":  15,
    "preference_match": 20,
}
MAX_SCORE = sum(COMPONENT_MAX.values())  # 100


# ---------------------------------------------------------------------------
# Mapping tables (user answer  ->  accepted dataset values, best-fit first)
# ---------------------------------------------------------------------------

DEPTH_MAP: dict[str, list[str]] = {
    "fair":      ["Fair"],
    "light":     ["Light", "Light-Medium"],
    "medium":    ["Light-Medium", "Medium", "Medium-Tan"],
    "tan":       ["Medium-Tan", "Tan"],
    "deep":      ["Deep"],
    "very_deep": ["Rich"],
}

UNDERTONE_MAP: dict[str, list[str]] = {
    "cool":    ["Cool", "Neutral-Cool"],
    "warm":    ["Warm", "Neutral-Warm"],
    "neutral": ["Neutral", "Neutral-Cool", "Neutral-Warm"],
    "olive":   ["Olive", "Neutral"],
}

# Which skin-type-suitability substrings satisfy each user skin-type answer.
SKIN_TYPE_KEYWORDS: dict[str, list[str]] = {
    "oily":        ["oily"],
    "dry":         ["dry"],
    "combination": ["combination"],
    "normal":      ["normal"],
    "sensitive":   ["sensitive"],
    "acne_prone":  ["acne", "non-comedogenic"],
}

# Ingredient / finish signals for hero rules.
LONGWEAR_SIGNALS = ("24-hr", "24 hr", "16-hr", "16 hr", "long-wear",
                    "transfer-resistant", "12-hr", "12 hr")
HYDRATION_SIGNALS = ("hyaluronic", "glycerin", "hydrat", "aloe",
                     "squalane", "coconut water", "watermelon")
OIL_CONTROL_SIGNALS = ("oil-control", "oil control", "oil-absorb",
                       "mattif", "matte")
NONCOMEDOGENIC_SIGNALS = ("non-comedogenic",)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Row:
    """A single (product, shade) row from foundations.csv, typed."""
    brand: str
    product: str
    shade_name: str
    shade_code: str
    depth: str
    undertone: str
    finish: str
    coverage: str
    skin_types: str
    spf: int
    fragrance_free: bool
    price: float
    ingredients: str

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Row":
        return cls(
            brand=d["brand"],
            product=d["product_name"],
            shade_name=d["shade_name"],
            shade_code=d["shade_code"],
            depth=d["shade_depth_bucket"],
            undertone=d["undertone_bucket"],
            finish=d["finish"],
            coverage=d["coverage"],
            skin_types=d["skin_type_suitability"],
            spf=int(d["spf"] or 0),
            fragrance_free=(d["fragrance_free"] == "True"),
            price=float(d["price_usd"]),
            ingredients=d["ingredient_notes"],
        )


@dataclass
class Contribution:
    """A single (component, points, reason) score contribution."""
    component: str
    points: int
    reason: str


@dataclass
class FilterResult:
    keep: bool
    reason: str = ""


@dataclass
class Scored:
    """Result of running the rules against one row."""
    row: Row
    components: dict[str, int] = field(default_factory=dict)
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.components.values())


Filter = Callable[[Row, dict], FilterResult]
Scorer = Callable[[Row, dict], list[Contribution]]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _user_skin_types(profile: dict) -> list[str]:
    st = profile.get("skin_type", {})
    if isinstance(st, dict):
        return st.get("selected", []) or []
    if isinstance(st, list):
        return st
    return []


def _finish_maps(profile: dict) -> list[str] | None:
    f = profile.get("finish")
    if isinstance(f, dict):
        return f.get("maps_to_dataset")
    return None


def _coverage_maps(profile: dict) -> list[str] | None:
    c = profile.get("coverage")
    if isinstance(c, dict):
        return c.get("maps_to_dataset")
    return None


def _price_cap(profile: dict) -> float:
    pr = profile.get("price_range")
    if isinstance(pr, dict):
        return float(pr.get("max_usd", 10**9))
    return 10**9


def _min_spf(profile: dict) -> int | None:
    spf = profile.get("spf_needed")
    if spf == "yes":    return 15
    if spf == "yes_30": return 30
    return None


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# FILTER RULES — reject rows that can't work at all.
# Order matters only for the reason emitted; correctness is independent.
# ---------------------------------------------------------------------------

def filter_depth(row: Row, profile: dict) -> FilterResult:
    """Depth bucket must match the user's requested tone."""
    depth = profile.get("skin_depth")
    if not depth:
        return FilterResult(True)
    accepted = DEPTH_MAP.get(depth, [])
    if row.depth not in accepted:
        return FilterResult(False, f"depth {row.depth} outside {accepted}")
    return FilterResult(True)


def filter_undertone(row: Row, profile: dict) -> FilterResult:
    """Undertone must be compatible."""
    ut = profile.get("undertone")
    if not ut:
        return FilterResult(True)
    accepted = UNDERTONE_MAP.get(ut, [])
    if row.undertone not in accepted:
        return FilterResult(False, f"undertone {row.undertone} outside {accepted}")
    return FilterResult(True)


def filter_finish(row: Row, profile: dict) -> FilterResult:
    """Finish must be in user's chosen set (respecting 'no preference')."""
    accepted = _finish_maps(profile)
    if accepted and row.finish not in accepted:
        return FilterResult(False, f"finish {row.finish} not in {accepted}")
    return FilterResult(True)


def filter_coverage(row: Row, profile: dict) -> FilterResult:
    accepted = _coverage_maps(profile)
    if accepted and row.coverage not in accepted:
        return FilterResult(False, f"coverage {row.coverage} not in {accepted}")
    return FilterResult(True)


def filter_price(row: Row, profile: dict) -> FilterResult:
    """Rule: if user wants low budget, filter out expensive products."""
    cap = _price_cap(profile)
    if row.price > cap:
        return FilterResult(False, f"${row.price:.2f} exceeds cap ${cap:.0f}")
    return FilterResult(True)


def filter_fragrance_free(row: Row, profile: dict) -> FilterResult:
    """Only reject if user REQUIRED fragrance-free."""
    if profile.get("fragrance_free") == "required" and not row.fragrance_free:
        return FilterResult(False, "not fragrance-free (required)")
    return FilterResult(True)


def filter_spf_required(row: Row, profile: dict) -> FilterResult:
    min_spf = _min_spf(profile)
    if min_spf is not None and row.spf < min_spf:
        return FilterResult(False, f"SPF {row.spf} below required {min_spf}")
    return FilterResult(True)


FILTER_RULES: list[Filter] = [
    filter_depth,
    filter_undertone,
    filter_finish,
    filter_coverage,
    filter_price,
    filter_fragrance_free,
    filter_spf_required,
]


# ---------------------------------------------------------------------------
# SCORING RULES — grouped by the component they contribute to.
# Each returns a (possibly empty) list of Contributions.
# ---------------------------------------------------------------------------

# ---- tone_match (max 25) --------------------------------------------------

def score_tone(row: Row, profile: dict) -> list[Contribution]:
    depth = profile.get("skin_depth")
    if not depth:
        return []
    accepted = DEPTH_MAP.get(depth, [])
    if row.depth == accepted[0]:
        return [Contribution("tone_match", 25, f"exact depth match ({row.depth})")]
    if row.depth in accepted:
        return [Contribution("tone_match", 10, f"in-range depth ({row.depth})")]
    return []


# ---- undertone_match (max 25) --------------------------------------------

def score_undertone(row: Row, profile: dict) -> list[Contribution]:
    ut = profile.get("undertone")
    if not ut:
        return []
    if row.undertone.lower() == ut:
        return [Contribution("undertone_match", 25, f"exact undertone ({row.undertone})")]
    if row.undertone in UNDERTONE_MAP.get(ut, []):
        return [Contribution("undertone_match", 10, f"compatible undertone ({row.undertone})")]
    return []


# ---- finish_match (max 15) -----------------------------------------------

def score_finish(row: Row, profile: dict) -> list[Contribution]:
    """Base points for preferred finish + hero rules per skin type."""
    contribs: list[Contribution] = []
    accepted = _finish_maps(profile)
    if accepted and row.finish in accepted:
        contribs.append(Contribution("finish_match", 10, f"{row.finish} finish preferred"))

    user_types = _user_skin_types(profile)

    # Rule: If user has oily skin, boost matte and long-wear formulas.
    if "oily" in user_types or "acne_prone" in user_types:
        if row.finish == "Matte":
            contribs.append(Contribution("finish_match", 5, "matte boost for oily/acne-prone"))
        elif _contains_any(row.ingredients, LONGWEAR_SIGNALS):
            contribs.append(Contribution("finish_match", 3, "long-wear boost for oily/acne-prone"))

    # Rule: If user has dry skin, boost hydrating or dewy formulas.
    if "dry" in user_types:
        if row.finish in ("Dewy", "Radiant", "Luminous"):
            contribs.append(Contribution("finish_match", 5, "dewy boost for dry skin"))
        elif _contains_any(row.ingredients, HYDRATION_SIGNALS):
            contribs.append(Contribution("finish_match", 3, "hydrating formula boost for dry skin"))

    return _cap_component(contribs, "finish_match")


# ---- skin_type_match (max 15) --------------------------------------------

def score_skin_type(row: Row, profile: dict) -> list[Contribution]:
    contribs: list[Contribution] = []
    user_types = _user_skin_types(profile)
    if not user_types:
        return contribs

    suit = (row.skin_types or "").lower()
    if suit.strip() == "all":
        contribs.append(Contribution("skin_type_match", 10, "labelled for all skin types"))
    else:
        matched = []
        for t in user_types:
            for kw in SKIN_TYPE_KEYWORDS.get(t, []):
                if kw in suit:
                    matched.append(t)
                    break
        if matched:
            pts = 5 * len(matched)
            contribs.append(Contribution(
                "skin_type_match", pts,
                "suits " + ", ".join(t.replace("_", "-") for t in matched)))

    # Rule: oily/acne + non-comedogenic or oil-control signals in formula.
    if any(t in {"oily", "acne_prone"} for t in user_types):
        combined = row.skin_types + " " + row.ingredients
        if _contains_any(combined, NONCOMEDOGENIC_SIGNALS):
            contribs.append(Contribution("skin_type_match", 5, "non-comedogenic"))
        elif row.finish == "Matte" and _contains_any(combined, OIL_CONTROL_SIGNALS):
            contribs.append(Contribution("skin_type_match", 3, "oil-controlling matte formula"))

    return _cap_component(contribs, "skin_type_match")


# ---- preference_match (max 20) -------------------------------------------

def score_preferences(row: Row, profile: dict) -> list[Contribution]:
    contribs: list[Contribution] = []

    # Fragrance-free preference
    ff_pref = profile.get("fragrance_free")
    if row.fragrance_free and ff_pref in {"required", "preferred"}:
        contribs.append(Contribution("preference_match", 5, "fragrance-free"))

    # Rule: If user is sensitive, down-rank fragranced products.
    if "sensitive" in _user_skin_types(profile) and not row.fragrance_free:
        contribs.append(Contribution("preference_match", -5,
                                     "contains fragrance (sensitive skin penalty)"))

    # SPF preference
    spf_pref = profile.get("spf_needed")
    if spf_pref in {"yes", "yes_30"} and row.spf > 0:
        contribs.append(Contribution("preference_match", 5, f"has SPF {row.spf}"))

    # Non-comedogenic when the user asked for it
    if profile.get("non_comedogenic") in {"required", "preferred"}:
        combined = (row.skin_types + " " + row.ingredients).lower()
        if "non-comedogenic" in combined:
            contribs.append(Contribution("preference_match", 5, "non-comedogenic"))

    # Budget headroom
    cap = _price_cap(profile)
    if row.price <= cap * 0.8:
        contribs.append(Contribution("preference_match", 5,
                                     f"comfortably under budget (${row.price:.2f} ≤ ${cap:.0f})"))

    return _cap_component(contribs, "preference_match", allow_negative=True)


SCORE_RULES: list[Scorer] = [
    score_tone,
    score_undertone,
    score_finish,
    score_skin_type,
    score_preferences,
]


# ---------------------------------------------------------------------------
# Helpers to cap a component at its max
# ---------------------------------------------------------------------------

def _cap_component(contribs: list[Contribution], component: str,
                   allow_negative: bool = False) -> list[Contribution]:
    """Ensure a component's total contribution doesn't exceed COMPONENT_MAX.
    Trims the LAST contribution that would push it over, keeping earlier
    reasons intact for transparency."""
    max_pts = COMPONENT_MAX[component]
    running = 0
    kept: list[Contribution] = []
    for c in contribs:
        remaining = max_pts - running
        if remaining <= 0 and c.points > 0:
            break
        allocated = min(c.points, remaining) if c.points > 0 else c.points
        if allocated == 0:
            continue
        kept.append(Contribution(c.component, allocated, c.reason))
        running += allocated
    lo = -max_pts if allow_negative else 0
    total = _clip(sum(k.points for k in kept), lo, max_pts)
    if total != sum(k.points for k in kept) and kept:
        kept[-1] = Contribution(kept[-1].component,
                                kept[-1].points + (total - sum(k.points for k in kept)),
                                kept[-1].reason)
    return kept


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def evaluate(row: Row, profile: dict) -> tuple[Scored | None, list[str]]:
    """Run all filter + score rules against one row.

    Returns:
        (Scored, []) if the row passes filtering, with a Scored breakdown.
        (None, filter_reasons) if the row was filtered out.
    """
    reasons: list[str] = []
    for f in FILTER_RULES:
        r = f(row, profile)
        if not r.keep:
            reasons.append(r.reason)
            return None, reasons

    scored = Scored(row=row, components={k: 0 for k in COMPONENT_MAX})
    for s in SCORE_RULES:
        for c in s(row, profile):
            scored.components[c.component] = scored.components.get(c.component, 0) + c.points
            scored.contributions.append(c)

    return scored, []


def candidates(rows: Iterable[Row], profile: dict) -> list[Scored]:
    """Return every row that survives hard filters, scored but unsorted.

    This is the candidate set the ML ranker re-orders. Rules decide *who
    is eligible*; the model decides *in what order*.
    """
    results: list[Scored] = []
    for r in rows:
        s, _ = evaluate(r, profile)
        if s is not None:
            results.append(s)
    return results


def rank(rows: Iterable[Row], profile: dict, top: int = 10) -> list[Scored]:
    """Rank rows by total score. Ties break by lower price, then brand."""
    results = candidates(rows, profile)
    results.sort(key=lambda s: (-s.total, s.row.price, s.row.brand, s.row.product))
    return results[:top]
