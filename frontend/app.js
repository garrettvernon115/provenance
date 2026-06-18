"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let lastSources = [];

// ---- tab switching ----
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#" + t.dataset.view).classList.add("active");
    if (t.dataset.view === "eval") loadEval();
  });
});

// ---- ask ----
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#q").value.trim();
  if (!query) return;
  const btn = $("#ask-btn");
  btn.disabled = true;
  $("#status").textContent = "retrieving + re-ranking + answering…";
  $("#answer").classList.add("hidden");
  $("#sources").innerHTML = "";
  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 8, rerank: $("#rerank").checked }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    renderAnswer(data);
    renderSources(data.sources);
    $("#status").textContent = data.reranked ? "answer grounded in re-ranked passages" : "answer grounded in hybrid passages (no re-rank)";
  } catch (err) {
    $("#status").textContent = "error: " + err.message;
  } finally {
    btn.disabled = false;
  }
});

function renderAnswer(data) {
  const box = $("#answer");
  box.classList.remove("hidden");
  if (data.answer) {
    // turn [n] markers into clickable citation chips
    const html = esc(data.answer).replace(/\[(\d+)\]/g,
      (m, n) => `<a class="cite" data-i="${n}">[${n}]</a>`);
    box.innerHTML = html;
    box.querySelectorAll(".cite").forEach((a) =>
      a.addEventListener("click", () => focusSource(parseInt(a.dataset.i, 10))));
  } else {
    box.innerHTML = `<span class="note">${esc(data.note || "No answer generated.")}</span>`;
  }
}

function renderSources(sources) {
  lastSources = sources;
  const wrap = $("#sources");
  wrap.innerHTML = "";
  sources.forEach((s) => {
    const card = el("div", "src" + (s.cited ? " cited" : ""));
    card.id = "src-" + s.index;
    const head = el("div", "src-head");
    head.innerHTML =
      `<span class="src-num">[${s.index}]</span>` +
      `<span class="src-co">${esc(s.company)}</span>` +
      `<span class="src-meta">· ${esc(s.form)} · ${esc(s.section || "—")}</span>` +
      (s.cited ? ` <span class="badge">cited</span>` : "") +
      (s.rerank_score != null ? `<span class="src-score">ce ${s.rerank_score.toFixed(3)}</span>` : "");
    const text = el("div", "src-text");
    text.textContent = trim(s.text, 360);
    const link = el("button", "src-cite-link");
    link.textContent = `view source · ${s.accession} [${s.char_start}:${s.char_end}]`;
    link.addEventListener("click", () => openViewer(s));
    card.append(head, text, link);
    wrap.append(card);
  });
}

function trim(t, n) { t = t.replace(/\s+/g, " ").trim(); return t.length > n ? t.slice(0, n).trimEnd() + "…" : t; }

function focusSource(i) {
  const card = $("#src-" + i);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
}

// ---- source viewer (exact-passage citation) ----
async function openViewer(s) {
  const v = $("#viewer");
  $("#viewer-title").textContent = `${s.company} · ${s.form} · ${s.accession}`;
  $("#viewer-body").textContent = "Loading document…";
  v.classList.remove("hidden");
  try {
    const resp = await fetch("/api/documents/" + encodeURIComponent(s.accession));
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const doc = await resp.json();
    const ft = doc.full_text;
    // render with the cited span highlighted, exactly at char offsets
    const before = ft.slice(Math.max(0, s.char_start - 600), s.char_start);
    const cited = ft.slice(s.char_start, s.char_end);
    const after = ft.slice(s.char_end, s.char_end + 600);
    const body = $("#viewer-body");
    body.innerHTML = (s.char_start > 600 ? "…" : "") + esc(before) +
      "<mark>" + esc(cited) + "</mark>" + esc(after) +
      (s.char_end + 600 < ft.length ? "…" : "");
    const m = body.querySelector("mark");
    if (m) m.scrollIntoView({ block: "center" });
  } catch (err) {
    $("#viewer-body").textContent = "Could not load document: " + err.message;
  }
}
$("#viewer-close").addEventListener("click", () => $("#viewer").classList.add("hidden"));
$("#viewer").addEventListener("click", (e) => { if (e.target.id === "viewer") $("#viewer").classList.add("hidden"); });

// ---- eval dashboard ----
let evalLoaded = false;
async function loadEval() {
  if (evalLoaded) return;
  const box = $("#eval-content");
  try {
    const resp = await fetch("/api/eval");
    if (!resp.ok) throw new Error(resp.status === 404 ? "no results yet — run eval/run_eval.py" : "HTTP " + resp.status);
    const data = await resp.json();
    box.innerHTML = "";
    const cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "found"];
    const table = el("table", "eval");
    const thead = el("thead");
    thead.innerHTML = "<tr><th>system</th>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    const tbody = el("tbody");
    for (const [name, agg] of Object.entries(data.results)) {
      const tr = el("tr", name === "trained" ? "trained" : "");
      tr.innerHTML = `<td>${name}</td>` + cols.map((c) => `<td>${(agg[c] ?? 0).toFixed(4)}</td>`).join("");
      tbody.append(tr);
    }
    table.append(thead, tbody);
    box.append(table);
    const r = data.results;
    if (r["no-rerank"] && r["trained"]) {
      const hl = el("div", "headline");
      hl.innerHTML = `<strong>nDCG@10:</strong> ${r["no-rerank"]["ndcg@10"].toFixed(3)} (no re-rank) → ` +
        `${r["off-the-shelf"]["ndcg@10"].toFixed(3)} (off-the-shelf) → ` +
        `<span style="color:var(--good)">${r["trained"]["ndcg@10"].toFixed(3)} (trained)</span>. ` +
        `Gold set: ${data.gold_size} held-out questions · pool ${data.candidate_pool}.`;
      box.append(hl);
    }
    evalLoaded = true;
  } catch (err) {
    box.innerHTML = `<span class="muted">${esc(err.message)}</span>`;
  }
}
