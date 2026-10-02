"""Streamlit UI for the Foundation Matcher.

Reuses the rule engine in `rules.py` directly (Python-native, no logic port).
Renders the questionnaire from `questionnaire.json` and the dataset from
`foundations.csv`.

Run:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import streamlit as st

import rules
from ml.interactions import Event, append_event, now_iso

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_data
def load_spec() -> dict:
    return json.loads((HERE / "questionnaire.json").read_text())


@st.cache_data
def load_rows() -> list[rules.Row]:
    with (HERE / "foundations.csv").open() as f:
        return [rules.Row.from_dict(d) for d in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_single(q: dict, key_prefix: str = "") -> str | None:
    """Render a single-select question with st.radio."""
    labels = [o["label"] for o in q["options"]]
    ids = [o["id"] for o in q["options"]]

    idx = st.radio(
        q["prompt"] + ("" if q.get("required", True) else "  *(optional)*"),
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}{q['id']}",
        index=None if q.get("required", True) else 0,
        help=q.get("help"),
    )
    return ids[idx] if idx is not None else None


def render_multi(q: dict) -> list[str]:
    """Render a multi-select question, enforcing max_selections."""
    labels = {o["id"]: o["label"] for o in q["options"]}
    picked = st.multiselect(
        q["prompt"],
        options=list(labels.keys()),
        format_func=lambda k: labels[k],
        key=q["id"],
        help=q.get("help"),
        max_selections=q.get("max_selections"),
    )
    return picked


def render_price(q: dict) -> dict | None:
    """Render the single-or-number price question."""
    labels = [o["label"] for o in q["options"]]
    ids = [o["id"] for o in q["options"]]
    idx = st.radio(q["prompt"], options=list(range(len(labels))),
                   format_func=lambda i: labels[i], key=q["id"], help=q.get("help"),
                   index=None)
    if idx is None:
        return None
    chosen = q["options"][idx]
    if chosen["id"] == q.get("number_prompt_if"):
        val = st.number_input(q["number_prompt"], min_value=int(q["number_min"]),
                              max_value=int(q["number_max"]), value=30, step=1,
                              key=q["id"] + "__num")
        return {"id": "custom", "max_usd": float(val)}
    return {"id": chosen["id"], "max_usd": chosen.get("max_usd")}


# ---------------------------------------------------------------------------
# Profile resolution (mirrors questionnaire.py)
# ---------------------------------------------------------------------------

def resolve_profile(spec: dict, answers: dict) -> tuple[dict, list[str]]:
    """Turn raw answers into the resolved shape expected by rules.py."""
    resolved: dict = {}
    missing: list[str] = []
    sections_by_id = {s["id"]: s for s in spec["sections"]}

    for section in spec["sections"]:
        if section.get("conditional_on"):
            continue
        for q in section["questions"]:
            val = answers.get(q["id"])
            if q.get("required", True) and (val is None or (isinstance(val, list) and not val)):
                missing.append(q["prompt"])
                continue
            if val is None:
                continue

            if q["type"] == "multi":
                opts = [o for o in q["options"] if o["id"] in val]
                maps: list[str] = []
                for o in opts:
                    for m in o.get("maps_to_dataset", []):
                        if m not in maps:
                            maps.append(m)
                resolved[q["id"]] = {"selected": val, "maps_to_dataset": maps or None}

            elif q["type"] == "single_or_number":
                resolved[q["id"]] = {"selected": val["id"], "max_usd": val.get("max_usd")}

            else:
                opt = next((o for o in q["options"] if o["id"] == val), None)
                if opt and opt.get("maps_to_dataset"):
                    resolved[q["id"]] = {"selected": val, "maps_to_dataset": opt["maps_to_dataset"]}
                else:
                    resolved[q["id"]] = val

    # Undertone quiz resolution
    if answers.get("undertone") == "unsure":
        quiz = sections_by_id["undertone_quiz"]
        scores = {b: 0 for b in quiz["scoring"]["buckets"]}
        for sq in quiz["questions"]:
            v = answers.get(sq["id"])
            if v is None:
                missing.append(sq["prompt"])
                continue
            opt = next(o for o in sq["options"] if o["id"] == v)
            for b, pts in (opt.get("score") or {}).items():
                scores[b] = scores.get(b, 0) + pts
        max_s = max(scores.values())
        winners = [b for b, s in scores.items() if s == max_s]
        resolved["undertone"] = "neutral" if len(winners) > 1 else winners[0]

    return resolved, missing


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _log_feedback(profile: dict, s: rules.Scored, rank_num: int, action: str) -> None:
    log_path = HERE / "data" / "interactions.live.jsonl"
    append_event(log_path, Event(
        session_id=st.session_state.setdefault("session_id", f"ui-{uuid.uuid4().hex[:10]}"),
        user_id="streamlit",
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
        rank_rules=rank_num,
        clicked=1 if action in {"clicked", "saved", "purchased"} else 0,
        saved=1 if action in {"saved", "purchased"} else 0,
        purchased=1 if action == "purchased" else 0,
        source="tester",
        sample_weight=1.5,
    ))
    st.toast(f"Logged {action} → data/interactions.live.jsonl")


def render_match(s: rules.Scored, rank_num: int, profile: dict | None = None,
                 ml_score: float | None = None) -> None:
    r = s.row
    with st.container(border=True):
        left, mid, right = st.columns([0.5, 5, 1.2])
        with left:
            st.markdown(f"<div style='font-size:1.8rem;font-weight:700;color:#b46a54;text-align:center;'>{rank_num}</div>",
                        unsafe_allow_html=True)
        with mid:
            st.markdown(f"**{r.brand}** — {r.product}")
            st.caption(
                f"Shade **{r.shade_name}**"
                + (f" ({r.shade_code})" if r.shade_code and r.shade_code != r.shade_name else "")
                + f" · {r.depth} / {r.undertone}"
            )
            badges = []
            exact_depth = any(c.reason.startswith("exact depth") for c in s.contributions)
            exact_ut = any(c.reason.startswith("exact undertone") for c in s.contributions)
            if exact_depth and exact_ut:
                badges.append(("Perfect shade fit", "#dfeee4", "#4b7a5a"))
            elif exact_depth or exact_ut:
                badges.append(("Strong shade fit", "#dfeee4", "#4b7a5a"))
            if r.fragrance_free:
                badges.append(("Fragrance-free", "#dfeee4", "#4b7a5a"))
            if r.spf:
                badges.append((f"SPF {r.spf}", "#f2ebe6", "#2b2320"))
            badges.append((r.finish, "#f4e2da", "#8f4d3a"))
            badges.append((f"{r.coverage} coverage", "#f2ebe6", "#2b2320"))

            badge_html = " ".join(
                f"<span style='display:inline-block;padding:.15rem .55rem;border-radius:99px;"
                f"font-size:.72rem;background:{bg};color:{fg};margin-right:.25rem;'>{escape_html(t)}</span>"
                for t, bg, fg in badges
            )
            st.markdown(badge_html, unsafe_allow_html=True)

        with right:
            st.markdown(
                f"<div style='text-align:right;'>"
                f"<div style='font-size:1.3rem;font-weight:600;'>${r.price:.2f}</div>"
                f"<div style='font-size:.85rem;color:#7a6f68;'>score "
                f"<strong style='color:#b46a54;'>{s.total}</strong>/{rules.MAX_SCORE}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Per-component score breakdown
        comp_cols = st.columns(len(rules.COMPONENT_MAX))
        for col, (name, max_pts) in zip(comp_cols, rules.COMPONENT_MAX.items()):
            v = s.components.get(name, 0)
            with col:
                pct = max(0.0, min(1.0, v / max_pts)) if max_pts else 0
                st.progress(pct)
                st.caption(
                    f"**{name.replace('_match', '').replace('_', ' ')}**  \n"
                    f"{v} / {max_pts}"
                )

        with st.expander(f"Why this match — {len(s.contributions)} rules fired"):
            for c in s.contributions:
                sign = "+" if c.points >= 0 else ""
                color = "#4b7a5a" if c.points >= 0 else "#b04040"
                st.markdown(
                    f"<div style='display:grid;grid-template-columns:3.5em 10em 1fr;gap:.5rem;"
                    f"padding:.15rem 0;font-size:.9rem;'>"
                    f"<span style='color:{color};font-weight:600;text-align:right;'>{sign}{c.points}</span>"
                    f"<span style='color:#7a6f68;font-size:.8rem;'>{c.component}</span>"
                    f"<span>{escape_html(c.reason)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if r.ingredients:
                st.caption(f"*Formula:* {r.ingredients}")
        if profile is not None:
            c1, c2, c3 = st.columns(3)
            key = f"{s.row.brand}-{s.row.shade_code}-{rank_num}"
            if c1.button("Clicked", key=f"clk-{key}"):
                _log_feedback(profile, s, rank_num, "clicked")
            if c2.button("Saved", key=f"sav-{key}"):
                _log_feedback(profile, s, rank_num, "saved")
            if c3.button("Purchased", key=f"buy-{key}"):
                _log_feedback(profile, s, rank_num, "purchased")
        if ml_score is not None:
            st.caption(f"ML compatibility {ml_score:.3f}")


def escape_html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Foundation Matcher", page_icon="💄", layout="centered")

    spec = load_spec()
    rows = load_rows()

    st.title("💄 Foundation Matcher")
    st.caption(
        f"Rule-based matcher — filters and ranks {len(rows)} shades "
        f"across {len({r.brand for r in rows})} brands. "
        f"Score is out of {rules.MAX_SCORE}."
    )

    with st.sidebar:
        st.markdown("### How ranking works")
        st.markdown(
            "`total = tone_match + undertone_match + finish_match + "
            "skin_type_match + preference_match`"
        )
        for k, v in rules.COMPONENT_MAX.items():
            st.markdown(f"- **{k}** — max **{v}**")
        st.divider()
        top_n = st.slider("Number of matches", 3, 25, 10, key="_topn")
        model_path = HERE / "ml" / "artifacts" / "ranker.joblib"
        use_ml = False
        if model_path.exists():
            use_ml = st.checkbox("Re-rank with ML model", value=True, key="_use_ml")
            st.caption("Rules still filter. The model only reorders survivors.")
        st.divider()
        st.markdown(
            "Rules live in [`rules.py`](rules.py). Hard filters remove rows "
            "that can't work; scoring rules adjust one of 5 named components. "
            "Feedback buttons write `data/interactions.live.jsonl` for retraining."
        )

    st.markdown("### Your questionnaire")

    # Collect raw answers
    answers: dict = {}
    sections_by_id = {s["id"]: s for s in spec["sections"]}

    for section in spec["sections"]:
        if section.get("conditional_on"):
            continue
        st.markdown(f"#### {section['title']}")
        for q in section["questions"]:
            if q["type"] == "multi":
                answers[q["id"]] = render_multi(q)
            elif q["type"] == "single_or_number":
                answers[q["id"]] = render_price(q)
            else:
                answers[q["id"]] = render_single(q)

            # Conditional undertone mini-quiz
            if q["id"] == "undertone" and answers.get("undertone") == "unsure":
                st.info("Answer these 3 short questions to infer your undertone:")
                quiz = sections_by_id["undertone_quiz"]
                with st.container(border=True):
                    for sq in quiz["questions"]:
                        answers[sq["id"]] = render_single(sq, key_prefix="ut_")

    st.divider()

    col_go, col_dl = st.columns([1, 1])
    with col_go:
        submit = st.button("✨ Show my matches", type="primary", use_container_width=True)
    with col_dl:
        resolved_preview, missing_preview = resolve_profile(spec, answers)
        st.download_button(
            "Download profile (JSON)",
            data=json.dumps({"answers": answers, "resolved": resolved_preview}, indent=2),
            file_name="user_profile.json",
            mime="application/json",
            use_container_width=True,
            disabled=bool(missing_preview),
        )

    if submit:
        resolved, missing = resolve_profile(spec, answers)
        if missing:
            st.error("Please answer these questions first:\n\n- " + "\n- ".join(missing))
            return

        matches = rules.rank(rows, resolved, top=st.session_state.get("_topn", 10))
        ml_scores = [None] * len(matches)
        if st.session_state.get("_use_ml") and (HERE / "ml" / "artifacts" / "ranker.joblib").exists():
            from ml.ranker import load_ranker
            ranked = load_ranker().rerank(resolved, matches, top=len(matches), alpha=0.7)
            matches = [m.scored for m in ranked]
            ml_scores = [m.ml_score for m in ranked]
        st.session_state["last_matches"] = matches
        st.session_state["last_profile"] = resolved
        st.session_state["last_ml_scores"] = ml_scores

    if "last_matches" not in st.session_state:
        return

    matches = st.session_state["last_matches"]
    resolved = st.session_state["last_profile"]
    ml_scores = st.session_state.get("last_ml_scores") or [None] * len(matches)

    st.divider()
    st.markdown(f"### Your matches — {len(matches)} results")

    with st.container(border=True):
        cols = st.columns(4)
        summary_items = [
            ("Skin type", ", ".join(resolved.get("skin_type", {}).get("selected", []) or [])),
            ("Undertone", resolved.get("undertone", "—")),
            ("Depth",     resolved.get("skin_depth", "—")),
            ("Finish",    ", ".join(resolved.get("finish", {}).get("selected", []) or [])),
            ("Coverage",  resolved.get("coverage", {}).get("selected", "—")),
            ("Budget",    f"≤ ${resolved.get('price_range', {}).get('max_usd', '—')}"),
            ("Fragrance-free", resolved.get("fragrance_free", "—")),
            ("SPF",       resolved.get("spf_needed", "no pref")),
        ]
        for i, (k, v) in enumerate(summary_items):
            cols[i % 4].markdown(
                f"<div style='font-size:.7rem;color:#7a6f68;text-transform:uppercase;letter-spacing:.04em;'>{k}</div>"
                f"<div style='color:#8f4d3a;font-weight:500;margin-bottom:.5rem;'>{escape_html(v)}</div>",
                unsafe_allow_html=True,
            )

    if not matches:
        st.warning(
            "**No matches found.** Try raising your budget, broadening finish/coverage, "
            "or making fragrance-free / SPF a preference instead of a requirement."
        )
        return

    for i, m in enumerate(matches, start=1):
        render_match(m, i, profile=resolved, ml_score=ml_scores[i - 1] if i - 1 < len(ml_scores) else None)


if __name__ == "__main__":
    main()
