"use strict";

/* Provenance — Facing Pages UI (design 1b, Classical system).
   Answer on the verso, the live source on the recto like an open book. Wired to
   the real API: POST /api/query, GET /api/documents/{accession}, GET /api/eval.
   The source viewer is the recto's other half, not a modal; clicking the recto
   opens the whole filing (design state 3c). */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const NS = "http://www.w3.org/2000/svg";
const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const fmt = (n, d = 4) => (n == null ? "—" : Number(n).toFixed(d));

// ── tabs ────────────────────────────────────────────────────────────────
document.querySelectorAll(".prov-tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".prov-tab").forEach((x) => x.classList.remove("on"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    t.classList.add("on");
    $("#" + t.dataset.view).classList.add("active");
    if (t.dataset.view === "eval") loadEval();
  });
});

// ── re-ranker toggle (Ask) ──────────────────────────────────────────────
const rerankToggle = $("#rerank");
function rerankOn() { return rerankToggle.getAttribute("aria-checked") === "true"; }
function setRerank(on) {
  rerankToggle.setAttribute("aria-checked", on ? "true" : "false");
  rerankToggle.classList.toggle("off", !on);
  $("#rerank-state").textContent = on ? "on" : "off";
  $("#rerank-state").style.color = on ? "var(--color-accent)" : "var(--color-neutral-500)";
}
rerankToggle.addEventListener("click", () => setRerank(!rerankOn()));
rerankToggle.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); setRerank(!rerankOn()); }
});

// ── Ask ─────────────────────────────────────────────────────────────────
let sources = [];
let currentSource = null;

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#q").value.trim() || $("#q").placeholder;
  const btn = $("#ask-btn");
  btn.disabled = true;
  $("#status").textContent = "retrieving + re-ranking + answering…";
  $("#stepper").classList.remove("hidden");
  $("#answer").innerHTML = "";
  $("#rail").innerHTML = "";
  $("#rail-count").textContent = "";
  $("#plate").innerHTML = "";
  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 8, rerank: rerankOn() }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    renderAsk(data);
    $("#status").textContent = data.reranked
      ? "answer grounded in re-ranked passages"
      : "answer grounded in hybrid passages (no re-rank)";
  } catch (err) {
    $("#status").textContent = "error: " + err.message;
  } finally {
    btn.disabled = false;
    $("#stepper").classList.add("hidden");
  }
});

function renderAsk(data) {
  sources = data.sources || [];
  const answerBox = $("#answer");
  const eyebrow = $("#answer-eyebrow");

  if (data.answer) {
    eyebrow.textContent = "Answer";
    const p = el("p", "prov-answer");
    // turn [n] markers into clickable citation chips that select the recto source
    const parts = data.answer.split(/(\[\d+\])/g);
    parts.forEach((part) => {
      const m = /^\[(\d+)\]$/.exec(part);
      if (m) {
        const a = el("a", "prov-cite");
        a.textContent = part;
        a.dataset.i = m[1];
        a.addEventListener("click", () => selectSource(parseInt(m[1], 10), true));
        p.append(a);
      } else if (part) {
        appendRich(p, part);
      }
    });
    answerBox.append(p);
  } else {
    // answer layer off — retrieval still stands (design state 3b)
    eyebrow.textContent = "Answer unavailable";
    const box = el("div", "prov-note-box");
    const h = el("h4"); h.textContent = "The answer layer is disabled";
    const note = el("p");
    note.textContent = data.note || "No grounded prose is generated, but retrieval and the "
      + "trained re-ranker still run — the cited-quality passages are shown, ordered by re-rank score.";
    box.append(h, note);
    answerBox.append(box);
  }

  // source rail
  const rail = $("#rail");
  const cited = sources.filter((s) => s.cited).length;
  $("#rail-count").textContent = `${cited} of ${sources.length} cited`;
  sources.forEach((s) => {
    const row = el("div", "prov-fsrc" + (s.cited ? "" : " dim"));
    row.dataset.i = s.index;
    const r = el("div", "prov-fsrc-row");
    const num = el("span", "prov-fnum"); num.textContent = s.index;
    const co = el("span", "prov-fco"); co.textContent = s.company;
    const meta = el("span", "prov-fmeta");
    meta.textContent = `${s.form} · ${s.section || "—"}`;
    r.append(num, co, meta);
    if (s.cited) {
      const tag = el("span", "tag tag-outline");
      tag.style.cssText = "margin-left:2px;font-size:9.5px;padding:1px 6px";
      tag.textContent = "cited";
      r.append(tag);
    }
    if (s.rerank_score != null) {
      const sc = el("span", "prov-fscore"); sc.textContent = s.rerank_score.toFixed(3);
      r.append(sc);
    }
    row.append(r);
    row.addEventListener("click", () => selectSource(s.index, false));
    rail.append(row);
  });

  // default recto: first cited source, else the first
  const first = sources.find((s) => s.cited) || sources[0];
  if (first) selectSource(first.index, false);
  else $("#plate").innerHTML = '<p class="prov-plate-text muted">No sources returned.</p>';
}

function selectSource(index, flash) {
  const s = sources.find((x) => x.index === index);
  if (!s) return;
  currentSource = s;
  // recto plate — the chunk text IS the exact cited passage
  $("#recto-eyebrow").textContent = `Source ${s.index} · in original filing`;
  const plate = $("#plate");
  plate.innerHTML = "";
  const co = el("div", "prov-plate-co"); co.textContent = `${s.company} · ${s.form}`;
  const meta = el("div", "prov-plate-meta");
  meta.textContent = `${s.accession} · ${s.section || "—"} · chars [${s.char_start.toLocaleString()}:${s.char_end.toLocaleString()}]`;
  const p = el("p", "prov-plate-text");
  p.append(document.createTextNode("…"));
  const mark = el("span", "prov-mark"); mark.textContent = trim(s.text, 620);
  p.append(mark, document.createTextNode("…"));
  plate.append(co, meta, p);

  document.querySelectorAll("#rail .prov-fsrc").forEach((rw) =>
    rw.classList.toggle("on", parseInt(rw.dataset.i, 10) === index));
  document.querySelectorAll("#answer .prov-cite").forEach((a) =>
    a.classList.toggle("on", parseInt(a.dataset.i, 10) === index));

  if (flash) {
    const row = document.querySelector(`#rail .prov-fsrc[data-i="${index}"]`);
    if (row) { row.classList.remove("prov-flash"); void row.offsetWidth; row.classList.add("prov-flash"); }
  }
}

function trim(t, n) { t = t.replace(/\s+/g, " ").trim(); return t.length > n ? t.slice(0, n).trimEnd() + "…" : t; }

// LLM answers sometimes carry markdown **bold**; render it as <strong>, everything
// else as safe text nodes (never innerHTML — the answer is model-generated).
function appendRich(parent, text) {
  text.split(/(\*\*[^*]+\*\*)/g).forEach((seg) => {
    const b = /^\*\*([^*]+)\*\*$/.exec(seg);
    if (b) { const s = el("strong"); s.textContent = b[1]; parent.append(s); }
    else if (seg) parent.append(document.createTextNode(seg));
  });
}

// ── recto → full filing overlay (design state 3c) ───────────────────────
async function openOverlay() {
  const s = currentSource;
  if (!s) return;
  const ov = $("#overlay");
  $("#ov-title").textContent = `${s.company} · Form ${s.form}`;
  $("#ov-sub").textContent = s.accession;
  $("#ov-crumb").textContent = s.section || "—";
  $("#ov-span").textContent = `cited span · chars ${s.char_start.toLocaleString()}–${s.char_end.toLocaleString()}`;
  const body = $("#ov-body");
  body.textContent = "Loading document…";
  ov.classList.add("open");
  try {
    const resp = await fetch("/api/documents/" + encodeURIComponent(s.accession));
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const doc = await resp.json();
    const ft = doc.full_text;
    body.innerHTML = "";
    body.append(document.createTextNode(ft.slice(0, s.char_start)));
    const mark = el("span", "prov-mark");
    mark.textContent = ft.slice(s.char_start, s.char_end);
    body.append(mark);
    body.append(document.createTextNode(ft.slice(s.char_end)));
    requestAnimationFrame(() => { body.scrollTop = Math.max(0, mark.offsetTop - 120); });
  } catch (err) {
    body.textContent = "Could not load document: " + err.message;
  }
}
$("#recto").addEventListener("click", (e) => {
  if (e.target.closest("#ov-close") || e.target.closest("#overlay")) return;
  openOverlay();
});
$("#ov-close").addEventListener("click", (e) => { e.stopPropagation(); $("#overlay").classList.remove("open"); });
$("#overlay").addEventListener("click", (e) => { if (e.target.id === "overlay") $("#overlay").classList.remove("open"); });

// ── Evaluation spread ───────────────────────────────────────────────────
const EVAL_SYS = [
  { key: "no-rerank", head: "hybrid", sub: "no re-rank" },
  { key: "off-the-shelf", head: "off-the-shelf", sub: "ms-marco CE" },
  { key: "trained", head: "trained", sub: "ours" },
];
const rankText = (r) => (r == null ? "not retrieved" : "#" + r);

let evalLoaded = false;
let rankNodes = [];
let rrOn = true;

async function loadEval() {
  if (evalLoaded) return;
  try {
    const resp = await fetch("/api/eval");
    if (!resp.ok) throw new Error(resp.status === 404 ? "no results yet — run eval/run_eval.py" : "HTTP " + resp.status);
    const data = await resp.json();
    buildEvalArgument(data);
    buildRankLedger(data);
    buildEvalTables(data);
    evalLoaded = true;
  } catch (err) {
    $("#eval-content").innerHTML = "";
    const p = el("p", "muted"); p.textContent = err.message;
    $("#eval-content").append(p);
  }
}

function delta(on, off) { const d = on - off; return (d >= 0 ? "+" : "") + d.toFixed(2) + " vs base"; }

function buildEvalArgument(data) {
  const r = data.results;
  const box = $("#eval-content");
  box.innerHTML = "";

  const arg = el("p", "prov-arg");
  arg.textContent = `Hybrid retrieval casts a wide net, but the passage that actually answers `
    + `the question rarely surfaces first. A cross-encoder trained in-house re-scores the top `
    + `${data.candidate_pool} candidates and pulls the gold passage up the list — lifting every `
    + `ranking metric over the hybrid baseline across a held-out set of ${data.gold_size} questions.`;
  box.append(arg);

  // toggle row
  const row = el("div", "prov-toggle-row");
  const tog = el("span", "prov-toggle"); tog.id = "rr-toggle";
  tog.setAttribute("role", "switch"); tog.setAttribute("aria-checked", "true");
  tog.setAttribute("aria-label", "Trained re-ranker"); tog.tabIndex = 0;
  tog.append(el("span", "knob"));
  const lab = el("span");
  lab.append(document.createTextNode("Trained re-ranker "));
  const st = el("span", "rr-state"); st.id = "rr-state"; st.textContent = "on";
  lab.append(st);
  const side = el("span", "rr-side");
  side.textContent = `held-out · ${data.gold_size} queries`;
  row.append(tog, lab, side);
  box.append(row);

  // metric tiles
  const tiles = el("div", "prov-metrics");
  const specs = [
    { k: "nDCG@10", key: "ndcg@10", id: "ndcg" },
    { k: "Recall@10", key: "recall@10", id: "rec" },
    { k: "MRR", key: "mrr", id: "mrr" },
  ];
  specs.forEach((sp) => {
    const t = el("div", "prov-metric");
    const kk = el("div", "prov-metric-k"); kk.textContent = sp.k;
    const vv = el("div", "prov-metric-v"); vv.id = "m-" + sp.id;
    const dd = el("div", "prov-metric-d"); dd.id = "m-" + sp.id + "-d";
    t.append(kk, vv, dd);
    tiles.append(t);
  });
  box.append(tiles);

  const hr = el("hr", "hr"); hr.style.margin = "26px 0 16px"; box.append(hr);
  const fn = el("p", "prov-fignote");
  fn.textContent = "Read the recto as a rank ledger: each rule is one held-out query, the dot its "
    + "gold passage. With the re-ranker off the dots scatter down the list; on, they gather toward rank 1.";
  box.append(fn);

  // metric application
  window._evalMetrics = {
    on: {
      ndcg: [fmt(r.trained["ndcg@10"], 2), delta(r.trained["ndcg@10"], r["no-rerank"]["ndcg@10"])],
      rec: [fmt(r.trained["recall@10"], 2), delta(r.trained["recall@10"], r["no-rerank"]["recall@10"])],
      mrr: [fmt(r.trained["mrr"], 2), delta(r.trained["mrr"], r["no-rerank"]["mrr"])],
    },
    off: {
      ndcg: [fmt(r["no-rerank"]["ndcg@10"], 2), "baseline"],
      rec: [fmt(r["no-rerank"]["recall@10"], 2), "baseline"],
      mrr: [fmt(r["no-rerank"]["mrr"], 2), "baseline"],
    },
  };
  applyEval(true);
  tog.addEventListener("click", () => { rrOn = !rrOn; applyEval(rrOn); });
  tog.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); rrOn = !rrOn; applyEval(rrOn); }
  });
}

function applyEval(on) {
  const m = on ? window._evalMetrics.on : window._evalMetrics.off;
  const set = (id, v) => {
    const a = $("#m-" + id), b = $("#m-" + id + "-d");
    if (a) a.textContent = v[0];
    if (b) { b.textContent = v[1]; b.classList.toggle("base", !on); }
  };
  set("ndcg", m.ndcg); set("rec", m.rec); set("mrr", m.mrr);
  const tog = $("#rr-toggle");
  if (tog) { tog.classList.toggle("off", !on); tog.setAttribute("aria-checked", on ? "true" : "false"); }
  const st = $("#rr-state");
  if (st) { st.textContent = on ? "on" : "off"; st.style.color = on ? "var(--color-accent)" : "var(--color-neutral-500)"; }
  // slide the ledger dots between baseline and re-ranked rank
  rankNodes.forEach((nd) => {
    const r = on ? nd.rr : nd.base;
    const x = rankX(r);
    nd.dot.setAttribute("cx", x);
    nd.seg.setAttribute("x2", x);
    nd.dot.setAttribute("fill", r == null ? "var(--color-neutral-400)" : (on ? "var(--color-accent)" : "var(--color-neutral-500)"));
    nd.seg.setAttribute("stroke", on ? "var(--color-accent)" : "var(--color-neutral-500)");
  });
}

// ledger geometry (module-level so applyEval can reuse rankX)
const LW = 520, LH = 560, LL = 18, LR = 52, LT = 16, LB = 26;
const LRPLOT = LW - LR - 26;   // rank scale ends here, leaving a gap before "none"
let LPOOL = 50;
function rankX(r) {
  // log scale: the action is in the top ranks. null → "none" lane past the gap.
  if (r == null) return LW - LR - 2;
  return LL + (Math.log(r) / Math.log(LPOOL)) * (LRPLOT - LL);
}

function buildRankLedger(data) {
  const svg = $("#rank-chart");
  svg.innerHTML = "";
  const per = (data.per_question || []).slice();
  LPOOL = data.candidate_pool || 50;
  svg.setAttribute("viewBox", `0 0 ${LW} ${LH}`);
  // sort by baseline rank so the off-state reads as a descending fan
  const rankOr = (v) => (v == null ? LPOOL + 5 : v);
  per.sort((a, b) => rankOr(a.ranks["no-rerank"]) - rankOr(b.ranks["no-rerank"]));
  const n = per.length;
  const rowY = (i) => LT + (i + 0.5) * ((LH - LT - LB) / n);

  // axis ticks (log positions)
  [1, 5, 10, 20, 50].filter((t) => t <= LPOOL).forEach((t) => {
    const x = rankX(t);
    svg.append(mk("line", { x1: x, y1: LT, x2: x, y2: LH - LB, stroke: "var(--color-divider)",
      "stroke-width": t === 1 ? 1.4 : 0.8, "stroke-dasharray": t === 1 ? "" : "2 4" }));
    const tx = mk("text", { x: x, y: LH - 8, "text-anchor": "middle", "font-size": 9,
      fill: "var(--color-neutral-500)", "font-variant-numeric": "tabular-nums" });
    tx.textContent = t; svg.append(tx);
  });
  // "none" lane label
  const nl = mk("text", { x: rankX(null), y: LH - 8, "text-anchor": "middle", "font-size": 8,
    fill: "var(--color-neutral-500)" });
  nl.textContent = "none"; svg.append(nl);

  rankNodes = per.map((q, i) => {
    const y = rowY(i);
    const g = mk("g", { class: "prov-rankrow" });
    g.append(mk("line", { x1: rankX(1), y1: y, x2: rankX(LPOOL), y2: y,
      stroke: "var(--color-divider)", "stroke-width": 0.5 }));
    const seg = mk("line", { x1: rankX(1), y1: y, x2: rankX(1), y2: y,
      stroke: "var(--color-accent)", "stroke-width": 1.3, opacity: 0.5 });
    const dot = mk("circle", { class: "dot", cx: rankX(1), cy: y, r: 3, fill: "var(--color-accent)" });
    g.append(seg, dot);
    // fat transparent hit line for hover/focus
    const hit = mk("line", { class: "prov-hit", x1: rankX(1), y1: y, x2: rankX(LPOOL), y2: y, tabindex: "0" });
    hit.addEventListener("pointermove", (e) => showTip(q, e));
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("focus", (e) => showTip(q, e));
    hit.addEventListener("blur", hideTip);
    g.append(hit);
    svg.append(g);
    return { seg, dot, base: q.ranks["no-rerank"], rr: q.ranks["trained"], y };
  });
  applyEval(rrOn);
}

function showTip(q, evt) {
  const tip = $("#rank-tip");
  tip.innerHTML = "";
  const t = el("div"); t.textContent = q.question; t.style.marginBottom = "6px"; tip.append(t);
  EVAL_SYS.forEach((sy) => {
    const row = el("div");
    const b = el("b"); b.textContent = rankText(q.ranks[sy.key]);
    row.append(b, document.createTextNode(" · " + sy.head));
    tip.append(row);
  });
  const host = $("#eval-spread").getBoundingClientRect();
  const x = (evt.clientX != null ? evt.clientX : host.left + host.width - 200) - host.left;
  const y = (evt.clientY != null ? evt.clientY : host.top + 40) - host.top;
  tip.style.left = Math.max(8, Math.min(x + 14, host.width - 270)) + "px";
  tip.style.top = (y + 14) + "px";
  tip.classList.add("on");
}
function hideTip() { $("#rank-tip").classList.remove("on"); }

function buildEvalTables(data) {
  const det = $("#eval-tables");
  det.innerHTML = "";
  const sum = el("summary");
  sum.textContent = "Table view — aggregate metrics & every question's rank";
  det.append(sum);

  // aggregate three-way table
  const cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "found"];
  const agg = el("table", "eval");
  const ah = el("thead");
  ah.innerHTML = "<tr><th>system</th>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
  const ab = el("tbody");
  for (const [name, a] of Object.entries(data.results)) {
    const tr = el("tr", name === "trained" ? "trained" : "");
    const td0 = el("td"); td0.textContent = name; tr.append(td0);
    cols.forEach((c) => { const td = el("td"); td.textContent = fmt(a[c] ?? 0); tr.append(td); });
    ab.append(tr);
  }
  agg.append(ah, ab);
  det.append(agg);

  // per-question ranks
  if (data.per_question && data.per_question.length) {
    const pq = el("table", "eval");
    pq.style.marginTop = "18px";
    const ph = el("thead");
    ph.innerHTML = "<tr><th>held-out question</th>" + EVAL_SYS.map((s) => `<th>${s.head}</th>`).join("") + "</tr>";
    const pb = el("tbody");
    data.per_question.forEach((q) => {
      const tr = el("tr");
      const qd = el("td", "prov-q-cell"); qd.textContent = q.question; tr.append(qd);
      EVAL_SYS.forEach((s) => { const td = el("td"); td.textContent = rankText(q.ranks[s.key]); tr.append(td); });
      pb.append(tr);
    });
    pq.append(ph, pb);
    det.append(pq);
  }
}
