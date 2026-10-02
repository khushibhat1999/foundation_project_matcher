"""Interactive CLI that walks a user through the Foundation Matcher questionnaire.

Reads the question spec from `questionnaire.json`, prompts the user in the
terminal, validates their input, and writes the result to `user_profile.json`.

Typical usage:

    python3 questionnaire.py                # writes user_profile.json
    python3 questionnaire.py --out me.json  # custom output path
    python3 questionnaire.py --dry-run      # print the resolved profile only

The output profile is designed to plug straight into a downstream matcher
against `foundations.csv` (columns: shade_depth_bucket, undertone_bucket,
finish, coverage, spf, fragrance_free, price_usd, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
SPEC_PATH = HERE / "questionnaire.json"


# ---------------------------------------------------------------------------
# Small CLI helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_help(help_text: str | None) -> None:
    if help_text:
        print(f"  ({help_text})")


def _prompt_single(question: dict) -> dict:
    """Ask a single-select question. Returns the chosen option dict."""
    _print_header(question["prompt"])
    _print_help(question.get("help"))
    options = question["options"]
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt['label']}")

    while True:
        raw = input("\n  Choice (number): ").strip()
        if not raw and not question.get("required", True):
            return {}
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Please enter a valid number from the list.")


def _prompt_multi(question: dict) -> list[dict]:
    """Ask a multi-select question. Returns a list of chosen option dicts."""
    _print_header(question["prompt"])
    _print_help(question.get("help"))
    options = question["options"]
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt['label']}")

    min_n = question.get("min_selections", 1)
    max_n = question.get("max_selections", len(options))

    while True:
        raw = input(
            f"\n  Choices (comma-separated numbers, {min_n}-{max_n} allowed): "
        ).strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            print("  Please pick at least one option.")
            continue
        try:
            idxs = [int(p) for p in parts]
        except ValueError:
            print("  Please enter numbers only, separated by commas.")
            continue
        if any(i < 1 or i > len(options) for i in idxs):
            print(f"  Numbers must be between 1 and {len(options)}.")
            continue
        if len(set(idxs)) < len(idxs):
            print("  Please avoid duplicate selections.")
            continue
        if not (min_n <= len(idxs) <= max_n):
            print(f"  Pick between {min_n} and {max_n} options.")
            continue
        return [options[i - 1] for i in idxs]


def _prompt_single_or_number(question: dict) -> dict:
    """A single-select question where one option may trigger a numeric prompt."""
    chosen = _prompt_single(question)
    if chosen.get("id") == question.get("number_prompt_if"):
        lo = question.get("number_min", 1)
        hi = question.get("number_max", 10**9)
        while True:
            raw = input(f"  {question['number_prompt']} ").strip()
            try:
                val = float(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
            if not (lo <= val <= hi):
                print(f"  Value must be between {lo} and {hi}.")
                continue
            chosen = dict(chosen)
            chosen["max_usd"] = val
            return chosen
    return chosen


# ---------------------------------------------------------------------------
# Undertone helper (only asked when the user picks "unsure")
# ---------------------------------------------------------------------------

def _resolve_undertone_quiz(quiz_section: dict) -> tuple[str, dict]:
    """Ask the quiz questions, tally scores, return (winning_bucket, raw_answers)."""
    scores: dict[str, int] = {b: 0 for b in quiz_section["scoring"]["buckets"]}
    raw: dict[str, str] = {}

    for q in quiz_section["questions"]:
        chosen = _prompt_single(q)
        raw[q["id"]] = chosen["id"]
        for bucket, pts in chosen.get("score", {}).items():
            scores[bucket] = scores.get(bucket, 0) + pts

    # Highest score wins; ties -> Neutral (spec).
    max_score = max(scores.values())
    winners = [b for b, s in scores.items() if s == max_score]
    winner = "neutral" if len(winners) > 1 else winners[0]

    print(f"\n  → Inferred undertone: {winner.upper()}  (scores: {scores})")
    return winner, raw


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_questionnaire(spec: dict) -> dict:
    """Walk the user through every section and return a resolved profile dict."""
    profile: dict[str, Any] = {
        "schema_version": spec.get("version", "unknown"),
        "answers": {},
        "resolved": {},
    }

    sections_by_id = {s["id"]: s for s in spec["sections"]}

    for section in spec["sections"]:
        if section.get("conditional_on"):
            # Handled inline when triggered
            continue

        _print_header(section["title"])

        for q in section["questions"]:
            qtype = q["type"]
            if qtype == "single":
                chosen = _prompt_single(q)
                profile["answers"][q["id"]] = chosen.get("id")

                # Inline sub-questionnaire trigger (undertone quiz)
                if chosen.get("triggers_subquestions"):
                    sub_id = chosen["triggers_subquestions"]
                    sub_section = sections_by_id[sub_id]
                    winner, raw = _resolve_undertone_quiz(sub_section)
                    profile["answers"][sub_id] = raw
                    profile["resolved"][q["id"]] = winner
                else:
                    profile["resolved"][q["id"]] = chosen.get("id")

            elif qtype == "multi":
                chosen_list = _prompt_multi(q)
                profile["answers"][q["id"]] = [c["id"] for c in chosen_list]
                # For finish etc., also stash dataset mappings
                mapped: list[str] = []
                for c in chosen_list:
                    for m in c.get("maps_to_dataset", []):
                        if m not in mapped:
                            mapped.append(m)
                profile["resolved"][q["id"]] = {
                    "selected": [c["id"] for c in chosen_list],
                    "maps_to_dataset": mapped or None,
                }

            elif qtype == "single_or_number":
                chosen = _prompt_single_or_number(q)
                profile["answers"][q["id"]] = chosen.get("id")
                resolved: dict[str, Any] = {"selected": chosen.get("id")}
                if "max_usd" in chosen:
                    resolved["max_usd"] = chosen["max_usd"]
                profile["resolved"][q["id"]] = resolved

            else:
                raise ValueError(f"Unknown question type: {qtype!r}")

    # For single-select questions whose options carry `maps_to_dataset`,
    # attach the mapping too (e.g. coverage).
    for section in spec["sections"]:
        if section.get("conditional_on"):
            continue
        for q in section["questions"]:
            if q["type"] != "single":
                continue
            answer_id = profile["answers"].get(q["id"])
            if not answer_id:
                continue
            opt = next((o for o in q["options"] if o["id"] == answer_id), None)
            if opt and opt.get("maps_to_dataset"):
                profile["resolved"][q["id"]] = {
                    "selected": answer_id,
                    "maps_to_dataset": opt["maps_to_dataset"],
                }

    return profile


def summarize(profile: dict) -> str:
    r = profile["resolved"]
    lines = ["", "── Your profile ────────────────────────────────────────────────"]

    def _fmt(v):
        if isinstance(v, dict):
            return v.get("selected") or v
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return v

    for key in [
        "skin_type", "undertone", "skin_depth",
        "finish", "coverage",
        "fragrance_free", "non_comedogenic", "cruelty_free",
        "price_range", "spf_needed",
    ]:
        if key in r:
            lines.append(f"  {key:18s} {_fmt(r[key])}")
    lines.append("────────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH,
                        help="Path to questionnaire.json (default: alongside script)")
    parser.add_argument("--out", type=Path, default=HERE / "user_profile.json",
                        help="Where to write the resolved profile")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to disk, just print the profile")
    args = parser.parse_args(argv)

    with args.spec.open() as f:
        spec = json.load(f)

    print()
    print("Foundation Matcher — questionnaire")
    print("Answer each question by typing the number of your choice and pressing Enter.")

    try:
        profile = run_questionnaire(spec)
    except (KeyboardInterrupt, EOFError):
        print("\n\nAborted.")
        return 130

    print(summarize(profile))

    if args.dry_run:
        print("\n(dry-run: not writing to disk)")
        return 0

    args.out.write_text(json.dumps(profile, indent=2))
    print(f"\nSaved profile → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
