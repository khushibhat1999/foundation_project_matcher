// Parity test: run rules.js (browser code) directly in Node, feed it the
// real CSV + demo profile, and verify the top matches + component scores
// exactly match what match.py produces via rules.py.
//
// Run:   node web/test_parity.mjs   (from project root)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);

// --- Load rules.js as an ES module -------------------------------------
const rulesUrl = "file://" + path.join(HERE, "rules.js");
const { rowFromDict, rankRows, MAX_SCORE, COMPONENT_MAX } = await import(rulesUrl);

// Load and parse CSV (copy of the browser parseCsv, since it lives in app.js)
function parseCsv(text) {
  const lines = [];
  let cur = "", inQuotes = false, row = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') { inQuotes = false; }
      else cur += ch;
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ",") { row.push(cur); cur = ""; }
      else if (ch === "\n") { row.push(cur); lines.push(row); row = []; cur = ""; }
      else if (ch === "\r") { /* skip */ }
      else cur += ch;
    }
  }
  if (cur.length || row.length) { row.push(cur); lines.push(row); }
  const header = lines.shift();
  return lines.filter(r => r.length === header.length)
              .map(r => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

const csvText = fs.readFileSync(path.join(ROOT, "foundations.csv"), "utf8");
const profile = JSON.parse(fs.readFileSync(path.join(ROOT, "user_profile.demo.json"), "utf8"));
const rows = parseCsv(csvText).map(rowFromDict);

const jsMatches = rankRows(rows, profile.resolved, 8);

console.log(`JS matcher — parsed ${rows.length} rows, top ${jsMatches.length}:\n`);
for (let i = 0; i < jsMatches.length; i++) {
  const m = jsMatches[i];
  const r = m.row;
  const parts = Object.keys(COMPONENT_MAX).map(k =>
    `${k.replace("_match", "")}=${m.components[k]}/${COMPONENT_MAX[k]}`).join(" ");
  console.log(`  ${i + 1}. [${m.total}/${MAX_SCORE}] ${r.brand} — ${r.shade_name}`);
  console.log(`     ${parts}`);
}

// --- Get Python matcher output for the exact same profile --------------
const pyRaw = execSync(
  "python3 match.py --profile user_profile.demo.json --top 8 --json",
  { cwd: ROOT, encoding: "utf8" });
const pyMatches = JSON.parse(pyRaw);

let ok = true;
for (let i = 0; i < Math.min(pyMatches.length, jsMatches.length); i++) {
  const py = pyMatches[i];
  const js = jsMatches[i];
  const sameId = py.brand === js.row.brand && py.shade_name === js.row.shade_name;
  const sameScore = py.total_score === js.total;
  const sameComponents = JSON.stringify(py.components) === JSON.stringify(js.components);
  if (!sameId || !sameScore || !sameComponents) {
    ok = false;
    console.log(`\nMISMATCH at rank ${i + 1}:`);
    console.log("  python:", py.brand, py.shade_name, py.total_score, py.components);
    console.log("  js:    ", js.row.brand, js.row.shade_name, js.total, js.components);
  }
}

console.log(`\nParity with match.py (top ${jsMatches.length}, incl. per-component scores): ${ok ? "PASS" : "FAIL"}`);
if (!ok) process.exit(1);
