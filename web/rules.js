// Rule-based filtering + scoring engine (browser port of ../rules.py).
// Same 5-component model, same weights. Keep in sync!
//
// total = tone_match + undertone_match + finish_match
//       + skin_type_match + preference_match    (max 100)

export const COMPONENT_MAX = {
  tone_match:       25,
  undertone_match:  25,
  finish_match:     15,
  skin_type_match:  15,
  preference_match: 20,
};
export const MAX_SCORE = Object.values(COMPONENT_MAX).reduce((a, b) => a + b, 0);

export const DEPTH_MAP = {
  fair:      ["Fair"],
  light:     ["Light", "Light-Medium"],
  medium:    ["Light-Medium", "Medium", "Medium-Tan"],
  tan:       ["Medium-Tan", "Tan"],
  deep:      ["Deep"],
  very_deep: ["Rich"],
};
export const UNDERTONE_MAP = {
  cool:    ["Cool", "Neutral-Cool"],
  warm:    ["Warm", "Neutral-Warm"],
  neutral: ["Neutral", "Neutral-Cool", "Neutral-Warm"],
  olive:   ["Olive", "Neutral"],
};

const SKIN_TYPE_KEYWORDS = {
  oily:        ["oily"],
  dry:         ["dry"],
  combination: ["combination"],
  normal:      ["normal"],
  sensitive:   ["sensitive"],
  acne_prone:  ["acne", "non-comedogenic"],
};
const LONGWEAR_SIGNALS = ["24-hr", "24 hr", "16-hr", "16 hr", "long-wear",
                          "transfer-resistant", "12-hr", "12 hr"];
const HYDRATION_SIGNALS = ["hyaluronic", "glycerin", "hydrat", "aloe",
                           "squalane", "coconut water", "watermelon"];
const OIL_CONTROL_SIGNALS = ["oil-control", "oil control", "oil-absorb",
                             "mattif", "matte"];

// ---- Helpers -------------------------------------------------------------

const userSkinTypes = (p) => {
  const st = p.skin_type;
  if (st && typeof st === "object") return st.selected || [];
  if (Array.isArray(st)) return st;
  return [];
};
const finishMaps   = (p) => (p.finish   && typeof p.finish   === "object") ? p.finish.maps_to_dataset   : null;
const coverageMaps = (p) => (p.coverage && typeof p.coverage === "object") ? p.coverage.maps_to_dataset : null;
const priceCap     = (p) => (p.price_range && typeof p.price_range === "object")
                            ? Number(p.price_range.max_usd ?? Infinity) : Infinity;
const minSpf       = (p) => p.spf_needed === "yes" ? 15 : p.spf_needed === "yes_30" ? 30 : null;

const containsAny = (text, needles) => {
  const t = String(text || "").toLowerCase();
  return needles.some(n => t.includes(n));
};

// Parse a CSV row (dict from parseCsv) into a typed Row.
export function rowFromDict(d) {
  return {
    brand: d.brand,
    product: d.product_name,
    shade_name: d.shade_name,
    shade_code: d.shade_code,
    depth: d.shade_depth_bucket,
    undertone: d.undertone_bucket,
    finish: d.finish,
    coverage: d.coverage,
    skin_types: d.skin_type_suitability,
    spf: Number(d.spf) || 0,
    fragrance_free: d.fragrance_free === "True",
    price: Number(d.price_usd),
    ingredients: d.ingredient_notes || "",
  };
}

// ---- FILTER RULES --------------------------------------------------------

const FILTER_RULES = [
  {
    id: "depth",
    fn: (row, p) => {
      const depth = p.skin_depth;
      if (!depth) return { keep: true };
      const accepted = DEPTH_MAP[depth] || [];
      return accepted.includes(row.depth)
        ? { keep: true }
        : { keep: false, reason: `depth ${row.depth} outside ${JSON.stringify(accepted)}` };
    },
  },
  {
    id: "undertone",
    fn: (row, p) => {
      const ut = p.undertone;
      if (!ut) return { keep: true };
      const accepted = UNDERTONE_MAP[ut] || [];
      return accepted.includes(row.undertone)
        ? { keep: true }
        : { keep: false, reason: `undertone ${row.undertone} outside ${JSON.stringify(accepted)}` };
    },
  },
  {
    id: "finish",
    fn: (row, p) => {
      const accepted = finishMaps(p);
      if (accepted && !accepted.includes(row.finish))
        return { keep: false, reason: `finish ${row.finish} not in ${JSON.stringify(accepted)}` };
      return { keep: true };
    },
  },
  {
    id: "coverage",
    fn: (row, p) => {
      const accepted = coverageMaps(p);
      if (accepted && !accepted.includes(row.coverage))
        return { keep: false, reason: `coverage ${row.coverage} not in ${JSON.stringify(accepted)}` };
      return { keep: true };
    },
  },
  {
    id: "price",
    fn: (row, p) => {
      const cap = priceCap(p);
      return row.price <= cap
        ? { keep: true }
        : { keep: false, reason: `$${row.price.toFixed(2)} exceeds $${cap}` };
    },
  },
  {
    id: "fragrance_free_required",
    fn: (row, p) => (p.fragrance_free === "required" && !row.fragrance_free)
      ? { keep: false, reason: "not fragrance-free (required)" }
      : { keep: true },
  },
  {
    id: "spf_required",
    fn: (row, p) => {
      const min = minSpf(p);
      return (min != null && row.spf < min)
        ? { keep: false, reason: `SPF ${row.spf} below required ${min}` }
        : { keep: true };
    },
  },
];

// ---- SCORING RULES -------------------------------------------------------

function scoreTone(row, p) {
  const depth = p.skin_depth;
  if (!depth) return [];
  const accepted = DEPTH_MAP[depth] || [];
  if (row.depth === accepted[0])
    return [{ component: "tone_match", points: 25, reason: `exact depth match (${row.depth})` }];
  if (accepted.includes(row.depth))
    return [{ component: "tone_match", points: 10, reason: `in-range depth (${row.depth})` }];
  return [];
}

function scoreUndertone(row, p) {
  const ut = p.undertone;
  if (!ut) return [];
  if (row.undertone.toLowerCase() === ut)
    return [{ component: "undertone_match", points: 25, reason: `exact undertone (${row.undertone})` }];
  if ((UNDERTONE_MAP[ut] || []).includes(row.undertone))
    return [{ component: "undertone_match", points: 10, reason: `compatible undertone (${row.undertone})` }];
  return [];
}

function scoreFinish(row, p) {
  const out = [];
  const accepted = finishMaps(p);
  if (accepted && accepted.includes(row.finish))
    out.push({ component: "finish_match", points: 10, reason: `${row.finish} finish preferred` });

  const types = userSkinTypes(p);

  if (types.includes("oily") || types.includes("acne_prone")) {
    if (row.finish === "Matte")
      out.push({ component: "finish_match", points: 5, reason: "matte boost for oily/acne-prone" });
    else if (containsAny(row.ingredients, LONGWEAR_SIGNALS))
      out.push({ component: "finish_match", points: 3, reason: "long-wear boost for oily/acne-prone" });
  }

  if (types.includes("dry")) {
    if (["Dewy", "Radiant", "Luminous"].includes(row.finish))
      out.push({ component: "finish_match", points: 5, reason: "dewy boost for dry skin" });
    else if (containsAny(row.ingredients, HYDRATION_SIGNALS))
      out.push({ component: "finish_match", points: 3, reason: "hydrating formula boost for dry skin" });
  }

  return capComponent(out, "finish_match", false);
}

function scoreSkinType(row, p) {
  const out = [];
  const types = userSkinTypes(p);
  if (!types.length) return out;

  const suit = (row.skin_types || "").toLowerCase();
  if (suit.trim() === "all") {
    out.push({ component: "skin_type_match", points: 10, reason: "labelled for all skin types" });
  } else {
    const matched = [];
    for (const t of types) {
      for (const kw of (SKIN_TYPE_KEYWORDS[t] || [])) {
        if (suit.includes(kw)) { matched.push(t); break; }
      }
    }
    if (matched.length) {
      out.push({
        component: "skin_type_match",
        points: 5 * matched.length,
        reason: "suits " + matched.map(t => t.replace(/_/g, "-")).join(", "),
      });
    }
  }

  if (types.some(t => t === "oily" || t === "acne_prone")) {
    const combined = (row.skin_types + " " + row.ingredients).toLowerCase();
    if (combined.includes("non-comedogenic"))
      out.push({ component: "skin_type_match", points: 5, reason: "non-comedogenic" });
    else if (row.finish === "Matte" && containsAny(combined, OIL_CONTROL_SIGNALS))
      out.push({ component: "skin_type_match", points: 3, reason: "oil-controlling matte formula" });
  }

  return capComponent(out, "skin_type_match", false);
}

function scorePreferences(row, p) {
  const out = [];

  if (row.fragrance_free && (p.fragrance_free === "required" || p.fragrance_free === "preferred"))
    out.push({ component: "preference_match", points: 5, reason: "fragrance-free" });

  if (userSkinTypes(p).includes("sensitive") && !row.fragrance_free)
    out.push({ component: "preference_match", points: -5,
               reason: "contains fragrance (sensitive skin penalty)" });

  if ((p.spf_needed === "yes" || p.spf_needed === "yes_30") && row.spf > 0)
    out.push({ component: "preference_match", points: 5, reason: `has SPF ${row.spf}` });

  if (p.non_comedogenic === "required" || p.non_comedogenic === "preferred") {
    const combined = (row.skin_types + " " + row.ingredients).toLowerCase();
    if (combined.includes("non-comedogenic"))
      out.push({ component: "preference_match", points: 5, reason: "non-comedogenic" });
  }

  const cap = priceCap(p);
  if (row.price <= cap * 0.8)
    out.push({ component: "preference_match", points: 5,
               reason: `comfortably under budget ($${row.price.toFixed(2)} ≤ $${cap})` });

  return capComponent(out, "preference_match", true);
}

const SCORE_RULES = [scoreTone, scoreUndertone, scoreFinish, scoreSkinType, scorePreferences];

// ---- Component-cap helper ------------------------------------------------

function capComponent(contribs, component, allowNegative) {
  const max = COMPONENT_MAX[component];
  const kept = [];
  let running = 0;
  for (const c of contribs) {
    const remaining = max - running;
    if (remaining <= 0 && c.points > 0) break;
    const allocated = c.points > 0 ? Math.min(c.points, remaining) : c.points;
    if (allocated === 0) continue;
    kept.push({ ...c, points: allocated });
    running += allocated;
  }
  const lo = allowNegative ? -max : 0;
  const total = Math.max(lo, Math.min(max, kept.reduce((s, k) => s + k.points, 0)));
  const sumKept = kept.reduce((s, k) => s + k.points, 0);
  if (total !== sumKept && kept.length) {
    kept[kept.length - 1] = { ...kept[kept.length - 1],
                              points: kept[kept.length - 1].points + (total - sumKept) };
  }
  return kept;
}

// ---- Top-level API -------------------------------------------------------

export function evaluate(row, profile) {
  for (const f of FILTER_RULES) {
    const r = f.fn(row, profile);
    if (!r.keep) return { scored: null, filterReason: r.reason };
  }
  const components = Object.fromEntries(Object.keys(COMPONENT_MAX).map(k => [k, 0]));
  const contributions = [];
  for (const s of SCORE_RULES) {
    for (const c of s(row, profile)) {
      components[c.component] = (components[c.component] || 0) + c.points;
      contributions.push(c);
    }
  }
  const total = Object.values(components).reduce((a, b) => a + b, 0);
  return { scored: { row, components, contributions, total }, filterReason: null };
}

export function rankRows(rows, profile, top = 12) {
  const scored = [];
  for (const r of rows) {
    const { scored: s } = evaluate(r, profile);
    if (s) scored.push(s);
  }
  scored.sort((a, b) =>
    b.total - a.total
    || a.row.price - b.row.price
    || a.row.brand.localeCompare(b.row.brand));
  return scored.slice(0, top);
}
