// Foundation Matcher — client-side app.
// Loads ../questionnaire.json + ../foundations.csv, renders the form,
// resolves the profile locally, and computes ranked matches via rules.js.

import {
  COMPONENT_MAX, MAX_SCORE, rowFromDict, rankRows,
} from "./rules.js";

const PATHS = {
  spec: "../questionnaire.json",
  csv: "../foundations.csv",
};

let spec = null;
let rows = [];

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function boot() {
  try {
    const [specRes, csvRes] = await Promise.all([
      fetch(PATHS.spec),
      fetch(PATHS.csv),
    ]);
    if (!specRes.ok) throw new Error(`Failed to load ${PATHS.spec} (${specRes.status})`);
    if (!csvRes.ok)  throw new Error(`Failed to load ${PATHS.csv} (${csvRes.status})`);
    spec = await specRes.json();
    rows = parseCsv(await csvRes.text()).map(rowFromDict);
    document.getElementById("loading").remove();
    renderForm(spec);
    wireButtons();
  } catch (e) {
    document.getElementById("loading").textContent =
      `Error: ${e.message}. Serve this folder with a local HTTP server (see README).`;
  }
}

// ---------------------------------------------------------------------------
// CSV parsing (handles quoted fields with commas)
// ---------------------------------------------------------------------------

export function parseCsv(text) {
  const lines = [];
  let cur = "", inQuotes = false, row = [];
  const pushCell = () => { row.push(cur); cur = ""; };
  const pushRow = () => { lines.push(row); row = []; };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') { inQuotes = false; }
      else { cur += ch; }
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ",") pushCell();
      else if (ch === "\n") { pushCell(); pushRow(); }
      else if (ch === "\r") { /* skip */ }
      else cur += ch;
    }
  }
  if (cur.length || row.length) { pushCell(); pushRow(); }

  const header = lines.shift();
  return lines
    .filter(r => r.length === header.length)
    .map(r => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

// ---------------------------------------------------------------------------
// Form rendering
// ---------------------------------------------------------------------------

function renderForm(spec) {
  const form = document.getElementById("questionnaire");
  form.hidden = false;

  for (const section of spec.sections) {
    if (section.conditional_on) continue;

    const secHeader = document.createElement("h3");
    secHeader.textContent = section.title;
    secHeader.style.margin = "1.5rem 0 0.5rem";
    secHeader.style.color = "var(--accent-dark)";
    secHeader.style.fontSize = "1rem";
    form.appendChild(secHeader);

    for (const q of section.questions) {
      form.appendChild(renderQuestion(q));
    }
  }
  document.getElementById("form-actions").hidden = false;
}

function renderQuestion(q) {
  const fs = document.createElement("fieldset");
  fs.dataset.qid = q.id;
  fs.dataset.qtype = q.type;

  const legend = document.createElement("legend");
  legend.textContent = q.prompt + (q.required === false ? " (optional)" : "");
  fs.appendChild(legend);

  if (q.help) {
    const help = document.createElement("p");
    help.className = "help";
    help.textContent = q.help;
    fs.appendChild(help);
  }

  const ul = document.createElement("ul");
  ul.className = "options";
  for (const opt of q.options) {
    ul.appendChild(renderOption(q, opt));
  }
  fs.appendChild(ul);

  if (q.type === "single_or_number") {
    const wrap = document.createElement("div");
    wrap.className = "custom-price";
    wrap.hidden = true;
    wrap.dataset.numberFor = q.id;
    wrap.innerHTML = `<label>Max $ <input type="number" min="${q.number_min}" max="${q.number_max}" step="1" placeholder="35" data-qid="${q.id}__num" /></label>`;
    fs.appendChild(wrap);
  }
  return fs;
}

function renderOption(q, opt) {
  const li = document.createElement("li");
  const label = document.createElement("label");
  label.className = "option";

  const input = document.createElement("input");
  input.type = q.type === "multi" ? "checkbox" : "radio";
  input.name = q.id;
  input.value = opt.id;
  input.dataset.optId = opt.id;

  input.addEventListener("change", () => onOptionChange(q, opt, input));

  const span = document.createElement("span");
  span.className = "option-label";
  span.textContent = opt.label;

  label.appendChild(input);
  label.appendChild(span);
  li.appendChild(label);

  if (opt.triggers_subquestions) {
    const sub = document.createElement("div");
    sub.className = "subquestions";
    sub.hidden = true;
    sub.dataset.subFor = q.id;
    const quiz = spec.sections.find(s => s.id === opt.triggers_subquestions);
    if (quiz) {
      for (const sq of quiz.questions) sub.appendChild(renderQuestion(sq));
    }
    li.appendChild(sub);
  }
  return li;
}

function onOptionChange(q, opt, input) {
  if (q.type === "multi" && q.max_selections) {
    const checked = [...document.querySelectorAll(`input[name="${q.id}"]:checked`)];
    if (checked.length > q.max_selections) {
      input.checked = false;
      alert(`Please pick at most ${q.max_selections} options for: ${q.prompt}`);
      return;
    }
  }

  const container = input.closest("fieldset");
  const subs = container.querySelectorAll('[data-sub-for="' + q.id + '"]');
  subs.forEach(sub => {
    sub.hidden = !(input.checked && opt.triggers_subquestions);
  });

  if (q.type === "single_or_number" && q.number_prompt_if) {
    const wrap = container.querySelector(`[data-number-for="${q.id}"]`);
    if (wrap) wrap.hidden = !(opt.id === q.number_prompt_if && input.checked);
  }
}

// ---------------------------------------------------------------------------
// Collect answers + resolve profile
// ---------------------------------------------------------------------------

function collectProfile() {
  const answers = {};
  const resolved = {};
  const missing = [];

  const sectionsById = Object.fromEntries(spec.sections.map(s => [s.id, s]));

  for (const section of spec.sections) {
    if (section.conditional_on) continue;
    for (const q of section.questions) {
      const val = readQuestion(q);
      if (q.required !== false && (val == null || (Array.isArray(val) && !val.length))) {
        missing.push(q.prompt);
        continue;
      }
      answers[q.id] = val;
      Object.assign(resolved, resolveQuestion(q, val));
    }
  }

  if (answers.undertone === "unsure") {
    const quiz = sectionsById.undertone_quiz;
    const scores = Object.fromEntries(quiz.scoring.buckets.map(b => [b, 0]));
    const rawAns = {};
    for (const sq of quiz.questions) {
      const v = readQuestion(sq);
      if (!v) { missing.push(sq.prompt); continue; }
      rawAns[sq.id] = v;
      const opt = sq.options.find(o => o.id === v);
      for (const [b, pts] of Object.entries(opt.score || {})) scores[b] += pts;
    }
    answers.undertone_quiz = rawAns;
    const max = Math.max(...Object.values(scores));
    const winners = Object.entries(scores).filter(([, s]) => s === max).map(([b]) => b);
    resolved.undertone = winners.length > 1 ? "neutral" : winners[0];
  }

  return { profile: { schema_version: spec.version, answers, resolved }, missing };
}

function readQuestion(q) {
  if (q.type === "multi") {
    return [...document.querySelectorAll(`input[name="${q.id}"]:checked`)].map(i => i.value);
  }
  const el = document.querySelector(`input[name="${q.id}"]:checked`);
  if (!el) return null;
  if (q.type === "single_or_number" && el.value === q.number_prompt_if) {
    const num = document.querySelector(`input[data-qid="${q.id}__num"]`);
    if (num && num.value) return { id: el.value, max_usd: Number(num.value) };
    return null;
  }
  return el.value;
}

function resolveQuestion(q, val) {
  const out = {};
  if (q.type === "multi") {
    const opts = val.map(v => q.options.find(o => o.id === v)).filter(Boolean);
    const maps = [];
    for (const o of opts) for (const m of (o.maps_to_dataset || [])) if (!maps.includes(m)) maps.push(m);
    out[q.id] = { selected: val, maps_to_dataset: maps.length ? maps : null };
    return out;
  }
  if (q.type === "single_or_number") {
    if (typeof val === "object") { out[q.id] = { selected: val.id, max_usd: val.max_usd }; return out; }
    const opt = q.options.find(o => o.id === val);
    const r = { selected: val };
    if (opt && "max_usd" in opt) r.max_usd = opt.max_usd;
    out[q.id] = r;
    return out;
  }
  const opt = q.options.find(o => o.id === val);
  if (opt && opt.maps_to_dataset) {
    out[q.id] = { selected: val, maps_to_dataset: opt.maps_to_dataset };
  } else {
    out[q.id] = val;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function renderResults(profile, matches) {
  document.getElementById("questionnaire-section").hidden = true;
  const section = document.getElementById("results-section");
  section.hidden = false;
  document.getElementById("match-count").textContent =
    `${matches.length} matches · scores out of ${MAX_SCORE}`;

  const summary = document.getElementById("profile-summary");
  summary.innerHTML = "";
  const r = profile.resolved;
  const fmt = v => (v && typeof v === "object" ? (v.selected ?? JSON.stringify(v)) : v);
  const entries = [
    ["Skin type",   Array.isArray(r.skin_type?.selected) ? r.skin_type.selected.join(", ") : fmt(r.skin_type)],
    ["Undertone",   r.undertone],
    ["Depth",       r.skin_depth],
    ["Finish",      Array.isArray(r.finish?.selected) ? r.finish.selected.join(", ") : fmt(r.finish)],
    ["Coverage",    fmt(r.coverage)],
    ["Budget",      r.price_range?.max_usd ? `≤ $${r.price_range.max_usd}` : "any"],
    ["Fragrance-free", r.fragrance_free],
    ["SPF",         r.spf_needed || "no pref"],
  ];
  for (const [k, v] of entries) {
    const d = document.createElement("div");
    d.innerHTML = `<span>${k}</span><span>${v ?? "—"}</span>`;
    summary.appendChild(d);
  }

  const cont = document.getElementById("matches");
  cont.innerHTML = "";
  if (!matches.length) {
    cont.innerHTML = `
      <div class="empty">
        <p><strong>No matches found.</strong></p>
        <p>Try raising your budget, broadening finish/coverage, or making fragrance-free / SPF a preference instead of a requirement.</p>
      </div>`;
    return;
  }

  matches.forEach((m, i) => cont.appendChild(matchCard(m, i + 1)));
}

function matchCard(s, rank) {
  const r = s.row;
  const el = document.createElement("article");
  el.className = "match";

  const exactDepth = s.contributions.some(c => c.reason.startsWith("exact depth"));
  const exactUt = s.contributions.some(c => c.reason.startsWith("exact undertone"));

  const badges = [];
  if (exactDepth && exactUt) badges.push(`<span class="badge hero">Perfect shade fit</span>`);
  else if (exactDepth || exactUt) badges.push(`<span class="badge good">Strong shade fit</span>`);
  if (r.fragrance_free) badges.push(`<span class="badge good">Fragrance-free</span>`);
  if (r.spf) badges.push(`<span class="badge">SPF ${r.spf}</span>`);
  badges.push(`<span class="badge">${r.finish}</span>`);
  badges.push(`<span class="badge">${r.coverage} coverage</span>`);

  // Per-component score bar
  const barCells = Object.keys(COMPONENT_MAX).map(k => {
    const v = s.components[k] || 0;
    const max = COMPONENT_MAX[k];
    const pct = Math.max(0, Math.min(100, (v / max) * 100));
    const label = k.replace(/_match$/, "").replace(/_/g, " ");
    return `
      <div class="score-cell" title="${label}: ${v}/${max}">
        <div class="score-cell-bar"><div style="width:${pct}%"></div></div>
        <div class="score-cell-label">${label}</div>
        <div class="score-cell-value">${v}<span class="muted">/${max}</span></div>
      </div>`;
  }).join("");

  const contribsHtml = s.contributions.map(c => {
    const sign = c.points >= 0 ? "+" : "";
    const cls = c.points >= 0 ? "" : " negative";
    return `<li class="contrib${cls}"><span class="pts">${sign}${c.points}</span>
             <span class="comp">${c.component}</span>
             <span class="why">${escape(c.reason)}</span></li>`;
  }).join("");

  el.innerHTML = `
    <div class="rank">${rank}</div>
    <div class="body">
      <h3><span class="brand">${escape(r.brand)}</span> — ${escape(r.product)}</h3>
      <p class="attrs">
        Shade <strong>${escape(r.shade_name)}</strong> ${r.shade_code && r.shade_code !== r.shade_name ? `(${escape(r.shade_code)})` : ""}
        · ${escape(r.depth)} / ${escape(r.undertone)}
      </p>
      <div class="badges">${badges.join("")}</div>
      <div class="score-breakdown">${barCells}</div>
      <details>
        <summary>Why this match — ${s.contributions.length} rules fired</summary>
        <ul class="contribs">${contribsHtml}</ul>
        ${r.ingredients ? `<p class="ingredients"><em>Formula:</em> ${escape(r.ingredients)}</p>` : ""}
      </details>
    </div>
    <div class="side">
      <div class="price">$${r.price.toFixed(2)}</div>
      <div class="score">${s.total}<span class="muted">/${MAX_SCORE}</span></div>
    </div>`;
  return el;
}

function escape(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ---------------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------------

function wireButtons() {
  document.getElementById("submit-btn").addEventListener("click", onSubmit);
  document.getElementById("download-btn").addEventListener("click", onDownload);
  document.getElementById("reset-btn").addEventListener("click", () => location.reload());
  document.getElementById("back-btn").addEventListener("click", () => {
    document.getElementById("results-section").hidden = true;
    document.getElementById("questionnaire-section").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function onSubmit() {
  const err = document.getElementById("form-error");
  const { profile, missing } = collectProfile();
  if (missing.length) {
    err.hidden = false;
    err.innerHTML = `Please answer:<br>• ${missing.map(escape).join("<br>• ")}`;
    return;
  }
  err.hidden = true;
  const matches = rankRows(rows, profile.resolved, 12);
  renderResults(profile, matches);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function onDownload() {
  const { profile, missing } = collectProfile();
  if (missing.length) { alert("Please finish answering first."); return; }
  const blob = new Blob([JSON.stringify(profile, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "user_profile.json";
  a.click();
  URL.revokeObjectURL(url);
}

boot();
