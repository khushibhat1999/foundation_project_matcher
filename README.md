# Foundation Products Dataset

A curated dataset of beauty foundation products across 41 brands and 62 product
lines, expanded to one row per (product, shade) pair.

- **Rows:** 291
- **Products:** 62
- **Brands:** 41
- **File:** `foundations.csv`
- **Generator:** `build_dataset.py`

Re-run the generator any time to rebuild the CSV:

```bash
python3 build_dataset.py
```

## Schema

| Column                  | Type    | Description |
|-------------------------|---------|-------------|
| `brand`                 | string  | Brand name (e.g. `Fenty Beauty`). |
| `product_name`          | string  | Full product/line name (e.g. `Pro Filt'r Soft Matte Longwear Foundation`). |
| `shade_name`            | string  | Marketed shade name (e.g. `Mont Blanc`, `NW10`, `2N1`). |
| `shade_code`            | string  | Brand-assigned shade code where distinct from the name. May equal the shade name if the brand uses a numeric-only system. |
| `shade_depth_bucket`    | enum    | Normalized depth bucket — see below. |
| `undertone_bucket`      | enum    | Normalized undertone bucket — see below. |
| `finish`                | enum    | Product-level marketed finish — see below. |
| `coverage`              | enum    | Product-level marketed coverage — see below. |
| `skin_type_suitability` | string  | Comma-separated list of skin types the brand markets the product for. |
| `spf`                   | integer | SPF value (0 if not an SPF product). |
| `fragrance_free`        | boolean | Whether the product is marketed as fragrance-free / unfragranced. |
| `price_usd`             | float   | Approximate MSRP in US dollars at time of authoring. |
| `ingredient_notes`      | string  | Short summary of marketed hero ingredients and formula family — NOT the full INCI. |

## Controlled vocabularies

**`shade_depth_bucket`** (fair → rich):
`Fair`, `Light`, `Light-Medium`, `Medium`, `Medium-Tan`, `Tan`, `Deep`, `Rich`.

**`undertone_bucket`**:
`Cool`, `Neutral-Cool`, `Neutral`, `Neutral-Warm`, `Warm`, `Olive`.

**`finish`**:
`Matte`, `Natural`, `Satin`, `Radiant`, `Dewy`, `Luminous`.

**`coverage`**:
`Sheer`, `Light`, `Light-Medium`, `Medium`, `Medium-Full`, `Full`.

## Sampling methodology

Each product line typically ships in 15–60+ shades. To keep the dataset within
the requested 100–300 row range while preserving depth diversity, the generator
samples **4–5 shades per product**, chosen at evenly-spaced indices across the
brand's ordered shade range (i.e. we always include a fair-end shade, a
rich-end shade, and 2–3 in between). Products with 7+ shades are down-sampled
to 4; smaller ranges to 5. Every brand's fair→rich span is therefore
represented, even if not every individual shade is listed.

## Distributions (current build)

- **Depth:** Fair 60, Light 31, Light-Medium 59, Medium 20, Medium-Tan 14,
  Tan 41, Deep 41, Rich 25.
- **Undertone:** Warm 128, Neutral 121, Cool 38, Neutral-Warm 2,
  Neutral-Cool 2.
- **Finish:** Natural 100, Matte 75, Radiant 66, Dewy 23, Satin 19, Luminous 13.
- **SPF-containing rows:** 109 / 291.
- **Fragrance-free rows:** 141 / 291.
- **Price:** min $8.99, max $88.00, mean ≈$43.52.

## Caveats

- **Prices** are approximate MSRP at time of authoring; regional pricing,
  promotions, and reformulation can move these.
- **Shade codes** use each brand's real code convention where known. When a
  brand only uses a shade name (not a separate SKU code), `shade_code` mirrors
  `shade_name` rather than being fabricated.
- **Undertone/depth** are normalized into a shared vocabulary; a shade a brand
  markets as e.g. "Cool Rosy" or "N‑neutral" is mapped to its nearest bucket.
- **`ingredient_notes`** is a short marketing-level summary of hero ingredients
  and formula family (e.g. "hyaluronic acid; niacinamide; mineral SPF") — it
  is **not** a full INCI list and should not be used for allergy or
  regulatory decisions.
- **`fragrance_free`** reflects the brand's positioning ("unfragranced",
  "fragrance-free", "clean/no added fragrance"). Some products may still
  contain trace naturally-derived aromatics.
- **Not every brand's full shade range is enumerated** — see Sampling above.

## User questionnaire

A user-facing intake questionnaire lives alongside the dataset so answers can
be resolved directly into filters against `foundations.csv`.

- `questionnaire.json` — machine-readable spec of every question, option, and
  dataset-column mapping (drives both the CLI and any future web form).
- `questionnaire.py` — interactive CLI runner.
- `user_profile.example.json` — a filled-out sample profile.

Run it:

```bash
python3 questionnaire.py                     # writes user_profile.json
python3 questionnaire.py --dry-run           # print only, don't save
python3 questionnaire.py --out me.json       # custom output path
```

### What it asks

| Axis                    | Type          | Options |
|-------------------------|---------------|---------|
| **Skin type**           | multi (1–3)   | oily · dry · combination · normal · sensitive · acne-prone |
| **Undertone**           | single        | cool · warm · neutral · olive · **unsure** (→ 3-question quiz on vein color, jewelry, sun reaction) |
| **Skin depth**          | single        | fair · light · medium · tan · deep · very deep |
| **Finish preference**   | multi (1–2)   | matte · natural · satin · dewy · no preference |
| **Coverage preference** | single        | light · medium · full · no preference |
| **Fragrance-free**      | single        | required · preferred · no preference |
| **Non-comedogenic**     | single        | required · preferred · no preference |
| **Cruelty-free**        | single        | cruelty-free required · vegan required · preferred · no preference |
| **Price range**         | single + number | drugstore ≤$20 · mid $20–40 · prestige $40–60 · luxury $60+ · custom max · no cap |
| **SPF**                 | single (opt)  | SPF 15+ · SPF 30+ · no preference |

### Output shape

The profile has two blocks:

- **`answers`** — exactly what the user picked (canonical option IDs).
- **`resolved`** — the same answers pre-processed for matching, including:
  - `undertone` resolved to a concrete bucket when the user picked "unsure"
    (score-based inference from the mini quiz),
  - `finish` and `coverage` expanded to the concrete `foundations.csv` values
    they cover (e.g. finish=`dewy` → `["Dewy","Radiant","Luminous"]`,
    coverage=`medium` → `["Light-Medium","Medium","Medium-Full"]`),
  - `price_range` resolved to a numeric `max_usd`.

A matcher can then filter `foundations.csv` directly:

```python
import json, csv
profile = json.load(open("user_profile.json"))["resolved"]

rows = list(csv.DictReader(open("foundations.csv")))
matches = [
    r for r in rows
    if r["shade_depth_bucket"].lower().startswith(profile["skin_depth"])
    and r["undertone_bucket"].lower().startswith(profile["undertone"])
    and r["finish"] in profile["finish"]["maps_to_dataset"]
    and r["coverage"] in profile["coverage"]["maps_to_dataset"]
    and float(r["price_usd"]) <= profile["price_range"]["max_usd"]
]
```

## Matcher — rule-based filter + score engine

This is the first-pass, deterministic ranker (runs before any ML). Its job
is to eliminate obviously bad matches with hard filters, then rank surviving
rows using a transparent, additive score:

```
total = tone_match + undertone_match + finish_match
      + skin_type_match + preference_match       (max 100)
```

The rules live in `rules.py` (Python) and `web/rules.js` (browser port). A
parity test (`node web/test_parity.mjs`) verifies the two produce identical
scores, per-component, on the demo profile.

### Component weights

| Component | Max | Sources |
|-----------|----:|---------|
| `tone_match` | 25 | Exact depth (25) · in-range depth (10) |
| `undertone_match` | 25 | Exact undertone (25) · compatible undertone (10) |
| `finish_match` | 15 | Preferred finish (10) · oily→matte boost (+5) · dry→dewy boost (+5) · long-wear boost for oily (+3) · hydrating boost for dry (+3) |
| `skin_type_match` | 15 | Labelled "All" (+10) · per matched keyword (+5, max 15) · non-comedogenic (+5) · oil-controlling matte (+3) |
| `preference_match` | 20 | Fragrance-free preferred (+5) · **sensitive + fragranced (−5)** · SPF present when preferred (+5) · non-comedogenic (+5) · budget headroom ≤ 80% cap (+5) |

Total possible: **100**. Components are individually capped so no single
category can dominate.

### Hard filter rules (row rejected if any fail)

| Rule | Rejects when |
|------|--------------|
| `filter_depth` | Row's depth bucket outside user's depth map |
| `filter_undertone` | Row's undertone outside user's undertone map |
| `filter_finish` | Row's finish not in user's chosen finish set |
| `filter_coverage` | Row's coverage not in user's coverage set |
| `filter_price` | Price exceeds user's `max_usd` |
| `filter_fragrance_free` | User required fragrance-free and row isn't |
| `filter_spf_required` | User required SPF and row's SPF is below threshold |

### Example rules realized in code

Each of the example rules you'd expect is a real, testable function in `rules.py`:

- *"If user has oily skin, boost matte and long-wear formulas"* → `score_finish` awards +5 for `Matte`, +3 for long-wear ingredient keywords when user is `oily` or `acne_prone`.
- *"If user has dry skin, boost hydrating or dewy formulas"* → `score_finish` awards +5 for `Dewy`/`Radiant`/`Luminous`, +3 for hydrating ingredient keywords when user is `dry`.
- *"If undertone is warm, filter toward warm/golden/olive shades"* → `filter_undertone` accepts only `Warm`/`Neutral-Warm` for warm users; `score_undertone` gives +25 for exact `Warm` match, +10 for the compatible `Neutral-Warm`.
- *"If user is sensitive, down-rank fragrance-heavy products"* → `score_preferences` applies **−5 to `preference_match`** when user is `sensitive` and the product isn't fragrance-free.
- *"If user wants low budget, filter out expensive products"* → `filter_price` rejects anything over `max_usd`; `score_preferences` gives +5 for prices ≤ 80% of cap.

### Usage

```bash
python3 match.py --profile user_profile.demo.json --top 10 --explain     # every rule that fired
python3 match.py --profile user_profile.demo.json --top 10 --breakdown   # per-component score bar
python3 match.py --profile user_profile.demo.json --json                 # machine-readable output
```

### Sample output (`--breakdown`)

```
Top 3 matches (score is out of 100):

   1. [ 80] Rare Beauty — Liquid Touch Weightless Foundation
        Shade: 230W (230W)  •  Light-Medium / Warm
        Natural finish, Light-Medium coverage  •  $31.00 • fragrance-free
        score: tone_match=25/25  undertone_match=25/25  finish_match=10/15  skin_type_match=10/15  preference_match=10/20

   2. [ 80] Fenty Beauty — Pro Filt'r Soft Matte Longwear Foundation
        Shade: 240 (240)  •  Light-Medium / Warm
        Matte finish, Medium-Full coverage  •  $42.00 • fragrance-free
        score: tone_match=25/25  undertone_match=25/25  finish_match=15/15  skin_type_match=5/15  preference_match=10/20

   3. [ 78] Revlon — ColorStay Makeup for Combination/Oily Skin
        Shade: Natural Beige (220)  •  Light-Medium / Warm
        Matte finish, Medium-Full coverage  •  $14.99 • SPF 15
        score: tone_match=25/25  undertone_match=25/25  finish_match=15/15  skin_type_match=8/15  preference_match=5/20
```

Sample output (`--explain`, showing individual rule firings):

```
   2. [ 80] Fenty Beauty — Pro Filt'r Soft Matte Longwear Foundation
       Shade: 240 (240)  •  Light-Medium / Warm
       Matte finish, Medium-Full coverage  •  $42.00 • fragrance-free
          + 25 tone_match        exact depth match (Light-Medium)
          + 25 undertone_match   exact undertone (Warm)
          + 10 finish_match      Matte finish preferred
          +  5 finish_match      matte boost for oily/acne-prone
          +  5 skin_type_match   suits oily
          +  5 preference_match  fragrance-free
          +  5 preference_match  comfortably under budget ($42.00 ≤ $60)
```

The sensitive-skin fragrance penalty in action (dry+sensitive fair-cool user):

```
   3. [ 65] Charlotte Tilbury — Beautiful Skin Foundation
       Shade: 1 Fair Cool (1CF)  •  Fair / Cool
       Radiant finish, Light-Medium coverage  •  $49.00 • SPF 20
          + 25 tone_match        exact depth match (Fair)
          + 25 undertone_match   exact undertone (Cool)
          + 10 finish_match      Radiant finish preferred
          +  5 finish_match      dewy boost for dry skin
          +  5 skin_type_match   suits dry
           -5 preference_match  contains fragrance (sensitive skin penalty)
```

### Extending the rules

To add a new rule, edit `rules.py` (and mirror in `web/rules.js`):

```python
def score_my_rule(row: Row, profile: dict) -> list[Contribution]:
    if some_condition(row, profile):
        return [Contribution("preference_match", 5, "why this matters")]
    return []

SCORE_RULES.append(score_my_rule)
```

Then rerun `node web/test_parity.mjs` to confirm the two engines stay in sync.

## ML ranking layer

Rules decide *who is eligible*. The model decides *in what order*. A
filtered-out product is never resurrected.

```
catalog
  → rules hard-filter + score     (stage 1, deterministic)
  → ML re-rank of survivors       (stage 2, trained on logs)
  → top-K shown to the user
```

### Training targets

Each impression in `data/interactions.jsonl` can carry any of:

| Field | Meaning | Typical use |
|-------|---------|-------------|
| `clicked` | User opened / tapped the card | First model, highest volume |
| `saved` | Favorited / shortlisted | Stronger intent |
| `purchased` | Bought or “I wear this” | Highest-quality, sparsest |
| `rated` | 1–5 match quality | Direct satisfaction |
| `relevance` | Composite: purchase×8 + save×4 + click×2 + rating leftover | Graded label for NDCG |

### First models

| `--model` | When to use |
|-----------|-------------|
| `logreg` | Default. Interpretable coefficients; good at ~hundreds of sessions. |
| `xgboost` | Non-linear leftovers (dry × satin, undertone × shade family). |
| `pairwise` | RankSVM-style. Trains on (clicked, skipped) pairs *inside* a session. |
| `xgbranker` | Learning-to-rank later, once you have more grouped usage logs. |

### Features

User one-hots + product one-hots + rule component scores + **cross
features** the model can use to learn residual patterns:

- `x_dry_satin`, `x_dry_dewy`, `x_dry_matte`
- `x_oily_matte`, `x_oily_dewy`
- `x_warm_warmshade`, `x_cool_coolshade`, `x_olive_oliveshade`
- `x_sensitive_fragrance`
- `exact_depth`, `exact_undertone`, `rule_total`, price ratio

### Getting labels (bootstrap)

At the start there is not enough real feedback. Labels are stacked from
four sources. A **stronger source replaces a weaker one** on the same
(user profile, SKU) pair:

```
rule  <  synthetic  <  expert  <  tester  <  behavior
0.25     0.50         1.50      1.50       2.00     ← sample weights
```

| Source | What it is | Where it lives |
|--------|------------|----------------|
| **rule** | Weak / pseudo-label: the stage-1 rule score is mapped to a 1–5 rating (`≥70` → clicked). Fills the catalog on day one. | generated by `ml.bootstrap_labels` |
| **synthetic** | Generated users from realistic skin-type × undertone × depth combos, with a latent click model. | `data/interactions.jsonl` via `ml.generate_logs` |
| **expert** | Manual 1–5 ratings on a product subset. Edit the CSV; one row = one (user, SKU). | `data/labels/expert_labels.csv` |
| **tester** | Friends / small beta. Streamlit Clicked / Saved / Purchased buttons. | `data/labels/tester.jsonl` and `data/interactions.live.jsonl` |
| **behavior** | Production clicks later. Same JSONL schema; just set `"source": "behavior"`. | your live log |

The practical trick: **start the model on rule-score pseudo-labels**, then
overwrite those rows as experts, testers, and real users label the same
pairs. Retrain; do not change the serve path.

```bash
# 1) optional: refresh synthetic users
python3 -m ml.generate_logs --n-sessions 350

# 2) merge every source (rule + synthetic + expert + tester + live)
python3 -m ml.bootstrap_labels

# 3) train on the merged file (sample-weighted by source)
python3 -m ml.train --logs data/labels/training.jsonl --target rated
python3 match.py --profile user_profile.demo.json --ml --breakdown
```

Add more expert labels by appending rows to `expert_labels.csv`
(`skin_type` and `finish` accept `oily|acne_prone` style lists, `rating`
is 1–5). Testers just use the Streamlit buttons.

At serve time:

```
final = (1 - alpha) * (rule_score / 100) + alpha * p_ml
```

`--ml-alpha 0.7` (default) trusts the model more; `--ml-alpha 0.0` is
rules-only. After a beta session, re-run `python3 -m ml.bootstrap_labels`
so live tester rows replace rule/synthetic labels on those pairs.

## Streamlit app

The fastest way to try the matcher end-to-end. Renders the questionnaire,
computes matches, and displays per-component score bars and rule-firing
breakdowns for each result.

```bash
pip install -r requirements.txt      # installs streamlit
streamlit run app_streamlit.py       # opens http://localhost:8501
```

The Streamlit app imports `rules.py` directly (no logic port) so it stays
byte-for-byte consistent with `match.py` automatically. Features:

- Same conditional undertone mini-quiz that appears when you pick "Unsure".
- **5-column score-breakdown bars** on every match card (tone / undertone / finish / skin-type / preference — each with a mini progress bar out of its max).
- **"Why this match" expander** listing every individual rule that fired, with green `+` bonuses and red `−` penalties.
- Sidebar shows the scoring formula and the max points per component; a slider controls how many matches to display.
- **Download profile (JSON)** button — the same shape `match.py` consumes, so users can go from the UI to a CLI query in one file.

## Web app

`web/` is a self-contained static frontend that renders the questionnaire,
resolves the profile client-side, and computes matches in-browser. No build
step, no npm install.

```bash
# From project root:
python3 -m http.server 8000
# Then open  http://localhost:8000/web/
```

The web app:
- Loads `../questionnaire.json` and renders every question, including the
  conditional undertone mini-quiz that appears when you pick "Unsure".
- Enforces multi-select caps client-side.
- Resolves the profile (undertone scoring, dataset mappings, price parsing)
  with logic that mirrors `questionnaire.py`.
- Loads `../foundations.csv` and runs the same matching algorithm as
  `match.py` via `web/rules.js` (a Node parity test in `web/test_parity.mjs`
  confirms byte-for-byte identical scores AND per-component breakdowns).
- Renders each result card with:
  - "Perfect shade fit" / "Fragrance-free" / "SPF" badges,
  - a 5-cell **score-breakdown bar** showing tone / undertone / finish / skin-type / preference contributions out of their per-component max,
  - a "Why this match" panel listing every individual rule that fired and its point value.
- Lets you download the resolved profile as JSON.

Run the parity test any time you change the matching logic:

```bash
node web/test_parity.mjs
```

## Suggested downstream uses

- Training a foundation shade-matcher (input: depth + undertone + finish + coverage → shortlist of SKUs).
- Filtering by skin-type / SPF / fragrance-free constraints.
- Price-tier analysis (drugstore vs. prestige) across finish/coverage.
- Coverage-vs-finish balance analysis across the market.
- Extending the dataset with per-SKU `cruelty_free` / `vegan` / `non_comedogenic` columns so those constraints can become hard filters.
