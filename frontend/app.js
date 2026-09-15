// Relative to the page, so the app works at "/" locally and under any nginx prefix.
const API = "api";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let session        = null;   // { session_id, chunk_bytes }
let files          = new Map();   // file_id -> { name, size, status, profile, row }
let proposal       = null;   // join proposal from /analyse
let datasetProfile = null;   // profile of the modelling table
let targetAnalysis = null;   // result of /target
let currentResults = null;   // full result JSON of the last job
let _lastInsights  = null;   // cached AI insights for embedding in reports
let scoringSchema  = null;   // form schema for the current model
let batchFileId    = null;

const COLORS = {
  cyan: "#4FC6E0",
  navy: "#0D1C31",
  muted: ["#8FA3BF", "#5F7596", "#B7C4D6", "#3E5577", "#A5B3C7", "#6E86A8", "#93A7C4", "#54698A"],
  ink3: "#7A879A",
  line: "#E3E8EF",
  good: "#1F9D6B",
  bad: "#C8453B",
};
const BAND_COLORS = { "A+": "#1F9D6B", A: "#4FC6E0", B: "#8FA3BF", C: "#E9B949", D: "#E07B39", E: "#C8453B" };
const SPLITS = ["train", "validation", "calibration", "test", "oot"];
const SPLIT_LABELS = { train: "Train", validation: "Validation", calibration: "Calibration", test: "Test", oot: "Out-of-Time" };
const PSI_COLORS = { stable: "#1F9D6B", moderate: "#E9B949", significant: "#C8453B" };
const ROLE_LABELS = { loan_dump: "Main table", bureau: "Bureau", bank_statement: "Bank statement", other: "Other" };
const STAGES = [["tuning", "Tuning zoo"], ["blend", "Blend"], ["nuscore", "Nu Score"], ["shap", "SHAP"], ["save", "Save bundle"]];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmt(value, digits = 3) {
  return Number.isFinite(Number(value)) && value !== null ? Number(value).toFixed(digits) : "–";
}

function pct(value, digits = 1) {
  return Number.isFinite(Number(value)) && value !== null ? `${(Number(value) * 100).toFixed(digits)}%` : "–";
}

function fmtBytes(n) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function setStatus(id, text) {
  const el = $(id);
  el.classList.toggle("active", Boolean(text));
  el.querySelector(".text").textContent = text || "";
}

function setError(id, text) {
  const el = $(id);
  el.classList.toggle("active", Boolean(text));
  el.textContent = text || "";
}

async function apiError(res) {
  const body = await res.json().catch(() => ({}));
  const detail = Array.isArray(body.detail)
    ? body.detail.map((d) => d.msg).join("; ")
    : body.detail;
  return new Error(detail || `${res.status} ${res.statusText}`);
}

async function request(method, path, payload) {
  const res = await fetch(`${API}/${path}`, {
    method,
    headers: payload !== undefined ? { "Content-Type": "application/json" } : {},
    body: payload !== undefined ? JSON.stringify(payload) : undefined,
  });
  if (!res.ok) throw await apiError(res);
  return res;
}
const postJSON = (path, payload) => request("POST", path, payload);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Job polling with stage stepper
// ---------------------------------------------------------------------------
function renderStepper(id, stage, done) {
  const el = $(id);
  if (!el) return;
  el.classList.add("active");
  const idx = STAGES.findIndex(([key]) => key === (stage === "fitting" ? "tuning" : stage));
  el.innerHTML = STAGES.map(([key, label], i) => {
    const cls = done || (idx >= 0 && i < idx) ? "done" : (i === idx ? "now" : "");
    return `<span class="st ${cls}"><i></i>${esc(label)}</span>`;
  }).join("");
}

function clearStepper(id) {
  const el = $(id);
  if (el) { el.classList.remove("active"); el.innerHTML = ""; }
}

// Polls a submitted job until it finishes; returns the result JSON.
async function waitForJob(jobId, statusId, stepperId) {
  const started = Date.now();
  for (;;) {
    await sleep(2000);
    const res = await fetch(`${API}/jobs/${jobId}`);
    if (!res.ok) throw await apiError(res);
    const job = await res.json();
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.status === "succeeded") { renderStepper(stepperId, "save", true); break; }
    if (job.status === "failed") throw new Error(job.message || "Training failed.");
    const stage = (job.progress && job.progress.stage) || (job.status === "running" ? "tuning" : null);
    if (stage) renderStepper(stepperId, stage, false);
    const mins = secs >= 60 ? `${Math.floor(secs / 60)}m ${secs % 60}s` : `${secs}s`;
    setStatus(statusId, job.status === "queued"
      ? `Queued: waiting for a free worker… ${mins}`
      : `${job.message || "Training all models"} · ${mins}`);
  }
  const res = await fetch(`${API}/jobs/${jobId}/result`);
  if (!res.ok) throw await apiError(res);
  return res.json();
}

// ---------------------------------------------------------------------------
// Hero: particle field + health check
// ---------------------------------------------------------------------------
(function heroCanvas() {
  const canvas = $("hero-canvas");
  if (!canvas) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ctx = canvas.getContext("2d");
  let w = 0, h = 0, nodes = [], raf = null, visible = true;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = rect.width; h = rect.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.max(30, Math.min(90, Math.floor(w / 16)));
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
      r: 1 + Math.random() * 1.6,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const n of nodes) {
      if (!reduced) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      }
    }
    ctx.lineWidth = 1;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 150 * 150) {
          ctx.strokeStyle = `rgba(79,198,224,${0.16 * (1 - d2 / (150 * 150))})`;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.fillStyle = "rgba(79,198,224,0.75)";
      ctx.shadowColor = "rgba(79,198,224,0.9)"; ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  function loop() {
    if (!visible) { raf = null; return; }
    draw();
    raf = reduced ? null : requestAnimationFrame(loop);
  }

  resize();
  window.addEventListener("resize", () => { resize(); if (!raf) loop(); });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible && !raf) loop();
    }).observe(canvas);
  }
  loop();
})();

(async function health() {
  const el = $("sys-health");
  try {
    const res = await fetch(`${API}/health`);
    const body = await res.json();
    el.textContent = res.ok ? `online · ${body.app || "api"}` : "degraded";
  } catch {
    el.textContent = "offline";
  }
})();

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const name = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  });
});

// ---------------------------------------------------------------------------
// Step 1: multi-file chunked upload
// ---------------------------------------------------------------------------
let _sessionPromise = null;

// Several files can start uploading at once; they must share one session.
function ensureSession() {
  if (session) return Promise.resolve(session);
  if (!_sessionPromise) {
    _sessionPromise = postJSON("sessions")
      .then((res) => res.json())
      .then((s) => { session = s; return s; })
      .finally(() => { _sessionPromise = null; });
  }
  return _sessionPromise;
}

function fileRow(entry) {
  const li = document.createElement("li");
  li.className = "file-row";
  li.innerHTML = `
    <div><span class="name"></span><span class="role" hidden></span></div>
    <div style="display:flex;align-items:center;gap:8px"><span class="meta"></span><button class="remove" title="Remove">×</button></div>
    <div class="bar"><span></span></div>`;
  li.querySelector(".name").textContent = entry.name;
  li.querySelector(".remove").addEventListener("click", () => removeFile(entry));
  $("file-list").appendChild(li);
  return li;
}

function updateRow(entry, text, pctDone) {
  entry.row.querySelector(".meta").textContent = text;
  if (pctDone !== undefined) entry.row.querySelector(".bar > span").style.width = `${pctDone}%`;
  entry.row.classList.toggle("ready", entry.status === "ready");
  entry.row.classList.toggle("failed", entry.status === "failed");
  refreshAnalyseButton();
}

function refreshAnalyseButton() {
  const list = [...files.values()];
  const ready = list.filter((f) => f.status === "ready").length;
  const busy = list.some((f) => f.status === "uploading" || f.status === "converting");
  $("analyse-btn").disabled = ready === 0 || busy;
}

// PUT one chunk with retries; the server rejects out-of-order chunks with 409,
// in which case we re-read how far it got and continue from there.
async function putChunk(fileId, index, blob) {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(`${API}/sessions/${session.session_id}/files/${fileId}/chunks/${index}`, {
        method: "PUT", body: blob, headers: { "Content-Type": "application/octet-stream" },
      });
      if (res.ok) return res.json();
      if (res.status === 409) {
        const info = await (await request("GET", `sessions/${session.session_id}/files/${fileId}`)).json();
        return { ...info, _resume: info.received_chunks };
      }
      if (res.status === 413 || res.status === 400 || res.status === 404) throw await apiError(res);
    } catch (err) {
      if (attempt === 3 || /limit|accepted|not found/i.test(err.message)) throw err;
    }
    await sleep(1500 * (attempt + 1));
  }
  throw new Error("Upload kept failing; please try again.");
}

async function uploadFile(file) {
  const entry = { name: file.name, size: file.size, status: "uploading", file_id: null, profile: null };
  entry.row = fileRow(entry);
  updateRow(entry, "Starting…", 0);
  try {
    await ensureSession();
    const init = await (await postJSON(`sessions/${session.session_id}/files`, {
      filename: file.name, size_bytes: file.size, chunk_bytes: session.chunk_bytes,
    })).json();
    entry.file_id = init.file_id;
    files.set(init.file_id, entry);

    const chunkBytes = init.chunk_bytes;
    for (let index = 0; index < init.total_chunks; index++) {
      const blob = file.slice(index * chunkBytes, Math.min(file.size, (index + 1) * chunkBytes));
      const info = await putChunk(init.file_id, index, blob);
      if (info._resume !== undefined && info._resume !== index + 1) index = info._resume - 1;
      const done = Math.min(file.size, (index + 1) * chunkBytes);
      updateRow(entry, `Uploading ${fmtBytes(done)} of ${fmtBytes(file.size)}`, (done / file.size) * 100);
    }

    entry.status = "converting";
    updateRow(entry, "Reading and profiling…", 100);
    const done = await (await postJSON(`sessions/${session.session_id}/files/${init.file_id}/complete`)).json();
    entry.status = "ready";
    entry.profile = done.profile;
    updateRow(entry, `${done.profile.row_count.toLocaleString()} rows · ${done.profile.column_count} columns · ${fmtBytes(file.size)}`, 100);
    resetDownstream();
  } catch (err) {
    entry.status = "failed";
    updateRow(entry, `Failed: ${err.message}`, 0);
  }
}

async function removeFile(entry) {
  if (entry.file_id && session) {
    files.delete(entry.file_id);
    try { await request("DELETE", `sessions/${session.session_id}/files/${entry.file_id}`); } catch { /* already gone */ }
  }
  entry.row.remove();
  resetDownstream();
  refreshAnalyseButton();
}

function resetDownstream() {
  proposal = null; datasetProfile = null; targetAnalysis = null;
  $("join-card").style.display = "none";
  $("target-card").style.display = "none";
  $("target-analysis").style.display = "none";
}

function addFiles(list) {
  setError("upload-error", "");
  [...list].forEach((file) => {
    if (!/\.(csv|json|jsonl|ndjson|parquet)$/i.test(file.name)) {
      setError("upload-error", `${file.name}: only .csv, .json/.jsonl and .parquet files are accepted.`);
      return;
    }
    uploadFile(file);
  });
  $("upload-title").textContent = "Drop more files here, or click to add";
}

function wireDropZone(areaId, inputId, onFiles) {
  const area = $(areaId);
  const input = $(inputId);
  area.addEventListener("click", () => input.click());
  area.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", (e) => { onFiles(e.target.files); input.value = ""; });
  area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", (e) => {
    e.preventDefault();
    area.classList.remove("dragover");
    onFiles(e.dataTransfer.files);
  });
}
wireDropZone("upload-area", "csv-file-input", addFiles);

// ---------------------------------------------------------------------------
// Step 2: analyse files, match identifiers, join
// ---------------------------------------------------------------------------
async function analyseFiles() {
  $("analyse-btn").disabled = true;
  setError("upload-error", "");
  setStatus("upload-status", "Reading column names and values, matching identifiers across files…");
  try {
    proposal = await (await postJSON(`sessions/${session.session_id}/analyse`)).json();
    renderRoles();
    const ready = [...files.values()].filter((f) => f.status === "ready");
    if (ready.length === 1) {
      // Nothing to join: go straight to the modelling table.
      await confirmJoin();
    } else {
      renderJoin();
    }
  } catch (err) {
    setError("upload-error", `Analysis failed: ${err.message}`);
  } finally {
    setStatus("upload-status", "");
    refreshAnalyseButton();
  }
}
$("analyse-btn").addEventListener("click", analyseFiles);

function renderRoles() {
  for (const [fid, entry] of files) {
    const role = proposal.roles[fid];
    const badge = entry.row.querySelector(".role");
    if (!role) { badge.hidden = true; continue; }
    badge.hidden = false;
    badge.textContent = ROLE_LABELS[role] || role;
    badge.className = `role ${esc(role)}`;
  }
}

function matchClass(rate) { return rate >= 0.8 ? "good" : rate >= 0.5 ? "warn" : "bad"; }

function renderJoin() {
  const card = $("join-card");
  card.style.display = "block";
  $("join-notes").textContent = proposal.notes || "";
  setError("join-error", "");

  const ready = [...files.entries()].filter(([, f]) => f.status === "ready");
  const primarySel = $("primary-file-select");
  primarySel.innerHTML = ready.map(([fid, f]) => `<option value="${esc(fid)}">${esc(f.name)}</option>`).join("");
  primarySel.value = proposal.primary_file_id;

  const rows = $("join-rows");
  rows.innerHTML = "";
  const primaryCols = files.get(proposal.primary_file_id).profile.columns.map((c) => c.name);
  for (const join of proposal.joins) {
    const entry = files.get(join.file_id);
    if (!entry) continue;
    const fileCols = entry.profile.columns.map((c) => c.name);
    const sel = join.selected;
    const row = document.createElement("div");
    row.className = "join-row";
    row.dataset.fileId = join.file_id;
    row.innerHTML = `
      <div class="file">${esc(entry.name)}<span class="role ${esc(join.role)}">${esc(ROLE_LABELS[join.role] || join.role)}</span></div>
      <div><label>Column in main table</label><select class="pcol"></select></div>
      <div class="eq">=</div>
      <div><label>Column in this file</label><select class="fcol"></select></div>
      <div class="match"></div>`;
    const pcol = row.querySelector(".pcol");
    const fcol = row.querySelector(".fcol");
    pcol.innerHTML = primaryCols.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    fcol.innerHTML = fileCols.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    const setMatch = () => {
      const cand = (join.candidates || []).find((c) => c.primary_col === pcol.value && c.file_col === fcol.value);
      const m = row.querySelector(".match");
      if (cand) {
        m.className = `match ${matchClass(cand.match_rate)}`;
        m.textContent = `${(cand.match_rate * 100).toFixed(0)}% of cases matched · ${cand.rows_per_key} row(s) per case`;
      } else {
        m.className = "match warn";
        m.textContent = "No overlap measured for this pair";
      }
    };
    if (sel) { pcol.value = sel.primary_col; fcol.value = sel.file_col; }
    pcol.addEventListener("change", setMatch);
    fcol.addEventListener("change", setMatch);
    setMatch();
    rows.appendChild(row);
  }
  if (!proposal.joins.length) rows.innerHTML = '<p class="sub">Only one file: nothing to join.</p>';
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function confirmJoin() {
  const primary = $("primary-file-select").value || proposal.primary_file_id;
  const joins = [...document.querySelectorAll("#join-rows .join-row")]
    .filter((row) => row.dataset.fileId !== primary)
    .map((row) => ({
      file_id: row.dataset.fileId,
      primary_col: row.querySelector(".pcol").value,
      file_col: row.querySelector(".fcol").value,
    }));
  $("join-btn").disabled = true;
  setError("join-error", "");
  setStatus("join-status", joins.length ? "Joining files into one modelling table…" : "Profiling your table…");
  try {
    const out = await (await postJSON(`sessions/${session.session_id}/join`, { primary_file_id: primary, joins })).json();
    datasetProfile = out.dataset_profile;
    renderTargetStep(out.join);
  } catch (err) {
    setError("join-error", `Join failed: ${err.message}`);
  } finally {
    setStatus("join-status", "");
    $("join-btn").disabled = false;
  }
}
$("join-btn").addEventListener("click", confirmJoin);

// ---------------------------------------------------------------------------
// Step 3: target, checks, IV
// ---------------------------------------------------------------------------
function renderTargetStep(join) {
  const card = $("target-card");
  card.style.display = "block";
  $("target-analysis").style.display = "none";
  targetAnalysis = null;
  setError("target-error", "");
  const stats = join && join.stats ? ` after joining ${join.joins.length} file(s)` : "";
  $("dataset-summary").textContent =
    `${datasetProfile.row_count.toLocaleString()} rows · ${datasetProfile.column_count} columns${stats}. ` +
    "Pick the column that holds the outcome you want to predict.";

  const sel = $("target-col-select");
  const cols = datasetProfile.columns.map((c) => c.name);
  sel.innerHTML = cols.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  if (join && join.suggested_target_col && cols.includes(join.suggested_target_col)) sel.value = join.suggested_target_col;
  $("positive-class-input").value = "";
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyseTarget() {
  const payload = { target_col: $("target-col-select").value };
  const positiveClass = $("positive-class-input").value.trim();
  if (positiveClass) payload.positive_class = positiveClass;
  $("target-btn").disabled = true;
  setError("target-error", "");
  setStatus("target-status", "Computing Information Value and checking for PII, leakage and identifiers…");
  try {
    targetAnalysis = await (await postJSON(`sessions/${session.session_id}/target`, payload)).json();
    renderTargetAnalysis();
  } catch (err) {
    setError("target-error", err.message);
  } finally {
    setStatus("target-status", "");
    $("target-btn").disabled = false;
  }
}
$("target-btn").addEventListener("click", analyseTarget);

function renderTargetAnalysis() {
  const t = targetAnalysis;
  $("target-analysis").style.display = "block";

  const stats = [
    ["Rows", datasetProfile.row_count.toLocaleString()],
    ["Variables", datasetProfile.column_count - 1],
    ["Positive rate", `${(t.positive_rate * 100).toFixed(1)}%`],
    ["Analysed on", `${t.sample_rows.toLocaleString()} rows`],
  ];
  $("target-stats").innerHTML = stats.map(([label, value]) =>
    `<div class="stat"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join("");

  const b = t.binarization;
  const how = {
    already_binary: "The target is already 0/1.",
    positive_class: `Rows where the target equals "${b.positive_class}" count as positive.`,
    threshold: `Rows with a value above ${b.threshold} count as positive.`,
    not_most_common: `${b.positive_class} counts as positive.`,
  }[b.strategy] || "";
  $("binarization-note").textContent = `${how} ${b.note || ""}`.trim();

  const timeSel = $("time-col-select");
  timeSel.innerHTML = '<option value="">None: random splits</option>' +
    t.date_columns.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  timeSel.value = t.time_col || "";
  const splitNote = () => {
    $("split-note").textContent = timeSel.value
      ? `Time-based validation: the latest 10% of rows by ${timeSel.value} form the stress split, the 10% before them the Out-of-Time split. That is how the model will be judged on future data.`
      : "No time column: Out-of-Time and stress splits are random samples, so they measure stability, not true future performance.";
  };
  timeSel.onchange = splitNote;
  splitNote();

  const list = $("flag-list");
  list.innerHTML = "";
  const flagsByCol = new Map();
  for (const f of t.flags) {
    if (!flagsByCol.has(f.column)) flagsByCol.set(f.column, []);
    flagsByCol.get(f.column).push(f);
  }
  for (const [col, flags] of flagsByCol) {
    const li = document.createElement("li");
    const excluded = t.excluded_by_default.includes(col);
    li.innerHTML = `<input type="checkbox" data-col="${esc(col)}" ${excluded ? "" : "checked"} />
      <div><strong>${esc(col)}</strong> ${flags.map((f) => `<span class="kind ${esc(f.kind)}">${esc(f.kind)}</span>`).join(" ")}
      <div class="reason">${flags.map((f) => esc(f.reason)).join("; ")}. ${esc(flags[0].action)}.</div></div>`;
    list.appendChild(li);
  }
  $("flags-section").style.display = flagsByCol.size ? "" : "none";

  renderIvChart(t.iv.slice(0, 20));
  renderNullChart(datasetProfile.columns);
  $("target-analysis").scrollIntoView({ behavior: "smooth", block: "start" });
}

function excludedColumns() {
  return [...document.querySelectorAll("#flag-list input[type=checkbox]")]
    .filter((cb) => !cb.checked)
    .map((cb) => cb.dataset.col);
}

async function trainSession() {
  if (!targetAnalysis) { setError("csv-train-error", "Analyse the target first."); return; }
  const payload = {
    target_col: targetAnalysis.target_col,
    exclude_cols: excludedColumns(),
    time_col: $("time-col-select").value || null,
    tier: $("tier-select").value,
    seed: 42,
  };
  const positiveClass = $("positive-class-input").value.trim();
  if (positiveClass) payload.positive_class = positiveClass;

  $("train-csv-btn").disabled = true;
  setError("csv-train-error", "");
  setStatus("csv-train-status", "Submitting training job…");
  renderStepper("csv-train-stepper", null, false);
  try {
    const job = await (await postJSON(`sessions/${session.session_id}/train`, payload)).json();
    applyResults(await waitForJob(job.job_id, "csv-train-status", "csv-train-stepper"));
    // The server deletes the session's data once results exist; start fresh next time.
    session = null;
  } catch (err) {
    setError("csv-train-error", `Training failed: ${err.message}`);
    clearStepper("csv-train-stepper");
  } finally {
    setStatus("csv-train-status", "");
    $("train-csv-btn").disabled = false;
  }
}
$("train-csv-btn").addEventListener("click", trainSession);

// ---------------------------------------------------------------------------
// Synthetic loan book
// ---------------------------------------------------------------------------
async function runSynthetic() {
  const payload = {
    n_rows:        parseInt($("rows").value, 10),
    n_features:    32 + parseInt($("noise").value || "0", 10),
    default_rate:  parseFloat($("default-rate").value) / 100,
    seed:          parseInt($("seed").value, 10),
    decision_labels: [$("label-positive").value, $("label-negative").value],
    tier:          $("tier-select-synthetic").value,
  };
  $("run-btn").disabled = true;
  setError("synthetic-error", "");
  setStatus("synthetic-status", "Submitting training job…");
  renderStepper("synthetic-stepper", null, false);
  try {
    const job = await (await postJSON("jobs/train", payload)).json();
    applyResults(await waitForJob(job.job_id, "synthetic-status", "synthetic-stepper"));
  } catch (err) {
    setError("synthetic-error", `Training failed: ${err.message}`);
    clearStepper("synthetic-stepper");
  } finally {
    setStatus("synthetic-status", "");
    $("run-btn").disabled = false;
  }
}
$("run-btn").addEventListener("click", runSynthetic);

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
function testMetrics(r) {
  // New engine payload first, legacy shape as fallback.
  if (r.evaluations) return r.evaluations.test.metrics;
  return r.metrics.test;
}

function splitMetric(r, split, key) {
  const m = r.evaluations ? (r.evaluations[split] || {}).metrics : (r.metrics || {})[split];
  return m ? m[key] : undefined;
}
function hasSplit(r, split) { return r.evaluations ? Boolean(r.evaluations[split]) : Boolean((r.metrics || {})[split]); }
const activeSplits = (r) => SPLITS.filter((s) => hasSplit(r, s));

function modelStats(r) {
  const auc = splitMetric(r, "test", "roc_auc");
  const valAuc = hasSplit(r, "validation") ? splitMetric(r, "validation", "roc_auc") : auc;
  return {
    auc, valAuc,
    gini: 2 * auc - 1,
    overfit: r.stability ? r.stability.overfit_gap : splitMetric(r, "train", "roc_auc") - valAuc,
    drift: r.stability ? r.stability.oot_drop : valAuc - splitMetric(r, "oot", "roc_auc"),
    testDrop: r.stability && r.stability.test_drop !== undefined ? r.stability.test_drop : valAuc - auc,
  };
}

function applyResults(data) {
  currentResults = data;
  _lastInsights = null;
  $("insights-panel").style.display = "none";
  $("results").style.display = "block";

  const order = data.leaderboard.map((row) => row.model);
  const byName = Object.fromEntries(data.results.map((r) => [r.name, r]));
  const ranked = order.map((name) => byName[name]).filter((r) => r && r.status !== "failed");
  const failed = data.results.filter((r) => r.status === "failed");
  const best = ranked[0];

  renderChampion(best, ranked, data.champion, data);
  renderWarnings(data);
  renderLeaderboard(ranked, failed);
  renderFeatureTable(data.dataset, best);
  renderDiagnostics(data);
  renderDrift(data.drift);
  renderUnlabelled(data);
  // Plotly sizes charts from their container, so draw after the section is visible.
  renderRocCurves(ranked);
  renderStability(ranked);
  renderMetrics(ranked);
  renderFeatureImportance(best);
  renderConfusion(best);
  renderLift(best);
  renderDataset(data.dataset);

  if (data.nuscore) { $("nuscore-panel").style.display = ""; renderNuScore(data.nuscore); renderCalibration(data.nuscore); }
  else { $("nuscore-panel").style.display = "none"; $("calib-chart").innerHTML = '<p class="sub">Not available for this run.</p>'; }

  if (data.shap && !data.shap.error) { $("shap-panel").style.display = ""; renderShap(data.shap); }
  else {
    $("shap-panel").style.display = data.shap ? "" : "none";
    if (data.shap) { $("shap-sub").textContent = `SHAP could not be computed: ${data.shap.error}`; $("shap-bar").innerHTML = ""; $("shap-bee").innerHTML = ""; $("shap-dep").innerHTML = ""; }
  }

  if (data.model_id) { $("scoring-panel").style.display = ""; loadScoringForm(data.model_id, data.champion ? data.champion.name : best.name); }
  else $("scoring-panel").style.display = "none";

  document.querySelector(".champion").scrollIntoView({ behavior: "smooth", block: "start" });
}

function gapTag(gap, warnAt, badAt) {
  const cls = gap >= badAt ? "bad" : gap >= warnAt ? "warn" : "good";
  const label = gap >= badAt ? "high" : gap >= warnAt ? "watch" : "ok";
  return `<span class="tag ${cls}">${fmt(gap)} · ${label}</span>`;
}

function renderChampion(best, ranked, champion, data) {
  const s = modelStats(best);
  const m = testMetrics(best);
  const diag = (data && data.diagnostics) || {};
  const ootStatus = (data && data.dataset && data.dataset.split && data.dataset.split.oot_status) || "";
  $("champion-name").textContent = best.name;
  const runnerUp = ranked[1];
  $("champion-why").textContent = champion && champion.why
    ? champion.why
    : (runnerUp ? `Highest Test ROC-AUC of ${ranked.length} models, ahead of ${runnerUp.name} by ${fmt(s.auc - splitMetric(runnerUp, "test", "roc_auc"))}.` : "Only one model was trained.");

  const comp = best.composite;
  const box = $("composite");
  if (comp) {
    const labels = { discrimination: "Discrimination", stability: "Stability", calibration: "Calibration", consistency: "Consistency", overfit: "Overfit", simplicity: "Simplicity" };
    box.innerHTML = Object.entries(comp.components).map(([k, v]) => `
      <div class="row"><span>${esc(labels[k] || k)} <span style="opacity:.6">×${(comp.weights[k] * 100).toFixed(0)}%</span></span>
      <div class="track"><i style="width:${Math.max(2, v * 100)}%"></i></div><b>${fmt(v, 2)}</b></div>`).join("") +
      `<div class="total">Composite ${fmt(comp.score, 3)}</div>`;
  } else box.innerHTML = "";

  const ci = diag.auc_bootstrap_95_interval;
  const kpis = [
    { label: "Validation AUC", value: fmt(s.valAuc), delta: "used for selection" },
    { label: "Test AUC", value: fmt(s.auc), delta: ci ? `95% CI ${fmt(ci[0])}–${fmt(ci[1])}` : `Gini ${fmt(s.gini)}` },
    { label: "OOT AUC", value: fmt(splitMetric(best, "oot", "roc_auc")), delta: ootStatus ? (ootStatus.startsWith("true") ? "true out-of-time" : "random hold-out") : "" },
    { label: "KS · Test", value: fmt(m.ks), delta: m.pr_auc !== undefined ? `PR-AUC ${fmt(m.pr_auc)}` : `Gini ${fmt(s.gini)}` },
    { label: "Brier · Test", value: fmt(m.brier), delta: m.ece !== undefined ? `ECE ${fmt(m.ece)}` : "" },
    { label: "Overfit gap", value: fmt(s.overfit), delta: "Train − Validation AUC" },
    { label: "OOT drop", value: fmt(s.drift), delta: best.stability && best.stability.score_psi_oot !== undefined ? `Validation − OOT · score PSI ${fmt(best.stability.score_psi_oot)}` : "Validation − OOT AUC" },
    { label: "Test drop", value: fmt(s.testDrop), delta: "Validation − Test AUC" },
  ];
  $("champion-kpis").innerHTML = kpis.map((k) => `
    <div class="kpi"><div class="label">${esc(k.label)}</div>
    <div class="value">${esc(k.value)}</div><div class="delta">${esc(k.delta)}</div></div>`).join("");
}

function renderWarnings(data) {
  const panel = $("warnings-panel");
  const list = data.warnings || [];
  const policy = data.champion && data.champion.selection_policy;
  if (!list.length && !policy) { panel.style.display = "none"; return; }
  panel.style.display = "";
  $("selection-policy").textContent = policy ? `Selection policy: ${policy}.` : "";
  $("warnings-list").innerHTML = list.map((w) => `<li>${esc(w)}</li>`).join("");
}

function famBadge(family) {
  return family ? `<span class="fam ${esc(family)}">${esc(family)}</span>` : "";
}

function renderLeaderboard(ranked, failed) {
  const rows = ranked.map((r, i) => {
    const s = modelStats(r);
    const t = testMetrics(r);
    const tuned = r.tuning || {};
    return `<tr class="${i === 0 ? "best" : ""} clickable" data-model="${esc(r.name)}">
      <td>${i + 1}</td>
      <td>${esc(r.name)}${famBadge(r.family)}${i === 0 ? '<span class="crown">CHAMPION</span>' : ""}</td>
      <td>${r.composite ? `<span class="bar-cell"><i style="width:${12 + 48 * r.composite.score}px"></i>${fmt(r.composite.score)}</span>` : "–"}</td>
      <td>${fmt(s.valAuc)}</td>
      <td>${fmt(s.auc)}</td>
      <td>${fmt(t.ks)}</td>
      <td>${fmt(splitMetric(r, "oot", "roc_auc"))}</td>
      <td>${fmt(t.pr_auc)}</td>
      <td>${fmt(t.brier)}</td>
      <td>${gapTag(s.overfit, 0.05, 0.1)}</td>
      <td>${gapTag(s.drift, 0.03, 0.06)}</td>
      <td>${tuned.trials ? `${tuned.trials} · ${fmt(tuned.cv_auc)}` : "–"}</td>
      <td>${r.fit_seconds !== undefined ? `${fmt(r.fit_seconds, 1)}s` : "–"}</td>
    </tr>`;
  }).join("");
  const failedRows = failed.map((r) => `<tr class="failed"><td>–</td><td>${esc(r.name)}${famBadge(r.family)}</td>
    <td colspan="11">Failed: ${esc(r.error)}</td></tr>`).join("");
  $("leaderboard").innerHTML = `
    <thead><tr><th>#</th><th>Model</th><th>Composite</th><th>Val AUC</th><th>Test AUC</th><th>KS</th><th>OOT AUC</th>
    <th>PR-AUC</th><th>Brier</th><th>Overfit gap</th><th>OOT drop</th><th>Trials · CV AUC</th><th>Fit</th></tr></thead>
    <tbody>${rows}${failedRows}</tbody>`;

  const byName = Object.fromEntries(ranked.map((r) => [r.name, r]));
  document.querySelectorAll("#leaderboard tr.clickable").forEach((tr) => {
    tr.addEventListener("click", () => toggleDetail(tr, byName[tr.dataset.model]));
  });
}

function toggleDetail(tr, r) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("detail")) { next.remove(); return; }
  document.querySelectorAll("#leaderboard tr.detail").forEach((d) => d.remove());
  const detail = document.createElement("tr");
  detail.className = "detail";
  const params = Object.entries(r.params || {}).map(([k, v]) =>
    `<span class="chip param">${esc(k)}: ${esc(v === null || v === undefined ? "none" : Array.isArray(v) ? v.join(",") : (typeof v === "number" ? +v.toPrecision(4) : v))}</span>`).join(" ");
  const splitsTable = `<table class="mini"><thead><tr><th>Split</th><th>AUC</th><th>KS</th><th>PR-AUC</th><th>Brier</th><th>F1</th><th>n</th></tr></thead><tbody>` +
    activeSplits(r).map((s) => {
      const m = r.evaluations ? r.evaluations[s].metrics : r.metrics[s];
      return `<tr><td>${esc(SPLIT_LABELS[s])}</td><td>${fmt(m.roc_auc)}</td><td>${fmt(m.ks)}</td><td>${fmt(m.pr_auc)}</td><td>${fmt(m.brier)}</td><td>${fmt(m.f1)}</td><td>${m.n !== undefined ? m.n.toLocaleString() : "–"}</td></tr>`;
    }).join("") + "</tbody></table>";
  const sparkId = `spark-${r.name.replace(/\W+/g, "-")}`;
  detail.innerHTML = `<td colspan="13"><div class="detail-grid">
    <div><h4>About</h4><p>${esc(r.description || "")}</p>
      <h4>Best parameters</h4><div class="chips">${params || '<span class="sub">Library defaults</span>'}</div>
      <h4>Tuning</h4><div id="${sparkId}" class="spark"></div></div>
    <div><h4>Per cohort</h4>${splitsTable}${r.importance_method ? `<p class="sub" style="margin-top:8px">Importance: ${esc(r.importance_method)}</p>` : ""}</div></div></td>`;
  tr.after(detail);
  const hist = ((r.tuning && r.tuning.history) || []).filter((h) => h.cv_auc !== null);
  if (hist.length) {
    let best = -Infinity;
    const bestSoFar = hist.map((h) => (best = Math.max(best, h.cv_auc)));
    Plotly.newPlot(sparkId, [
      { x: hist.map((h) => h.trial + 1), y: hist.map((h) => h.cv_auc), mode: "markers", marker: { color: COLORS.muted[0], size: 6 }, name: "trial", hovertemplate: "trial %{x}: %{y:.4f}<extra></extra>" },
      { x: hist.map((h) => h.trial + 1), y: bestSoFar, mode: "lines", line: { color: COLORS.cyan, width: 2 }, name: "best so far", hoverinfo: "skip" },
    ], baseLayout({ margin: { t: 4, l: 44, r: 8, b: 24 }, showlegend: false, height: 90,
      xaxis: { gridcolor: COLORS.line, title: "", dtick: Math.max(1, Math.ceil(hist.length / 10)) }, yaxis: { gridcolor: COLORS.line, tickformat: ".3f" } }), PLOT_CONFIG);
  } else {
    $(sparkId).innerHTML = `<span class="sub">${r.tuning && r.tuning.trials === 0 ? "No tuning (stacked blend)." : "No completed trials."}</span>`;
  }
}

// Model importances are keyed by transformed names ("x__te", "x_level", "missingindicator_x");
// map each back to the raw variable so it can sit next to that variable's IV.
function cleanFeature(name) { return String(name).replace(/^(num|cat|te)__/, "").replace(/__te$/, ""); }

// Glossary lookup: strip encoder prefixes/suffixes and role prefixes, then fall back
// to the longest glossary key the name starts with.
let _glossary = {};
function glossaryFor(name) {
  const clean = cleanFeature(name).replace(/^missingindicator_/, "");
  if (_glossary[clean]) return _glossary[clean];
  const stripped = clean.replace(/^(loan_dump|bureau|bank_statement|other)__/, "").replace(/__(month|year|days_before_ref)$/, "");
  if (_glossary[stripped]) return _glossary[stripped];
  const keys = Object.keys(_glossary).sort((a, b) => b.length - a.length);
  const hit = keys.find((k) => clean === k || clean.startsWith(`${k}_`) || stripped.startsWith(`${k}_`));
  return hit ? _glossary[hit] : null;
}

function importanceByVariable(best, variables) {
  const byVar = new Map();
  const sorted = [...variables].sort((a, b) => b.length - a.length);
  for (const item of best.feature_importance || []) {
    const clean = cleanFeature(item.feature).replace(/^missingindicator_/, "");
    const variable = sorted.find((v) => clean === v || clean.startsWith(`${v}_`)) || clean;
    byVar.set(variable, (byVar.get(variable) || 0) + Math.abs(item.importance));
  }
  return byVar;
}

function renderFeatureTable(dataset, best) {
  const iv = dataset.feature_iv || [];
  _glossary = dataset.feature_glossary || {};
  const table = $("features-table");
  if (!iv.length) { table.innerHTML = ""; return; }
  const importance = importanceByVariable(best, iv.map((r) => r.feature));
  const maxImp = Math.max(...importance.values(), 1e-9);
  const maxIv = Math.max(...iv.map((r) => r.iv), 1e-9);
  const strengthClass = { useless: "", weak: "", medium: "good", strong: "good", suspicious: "warn" };
  const shown = iv.slice(0, 30);
  const rows = shown.map((r, i) => {
    const imp = importance.get(r.feature) || 0;
    const g = glossaryFor(r.feature);
    const gloss = g ? `<b>${esc(g.label)}</b> · high = ${esc(g.high)}; low = ${esc(g.low)}` : "";
    const notes = (r.notes || []).map((n) => `<span class="note-chip">${esc(n)}</span>`).join("");
    return `<tr class="${r.bins ? "clickable" : ""}" data-idx="${i}" title="${r.bins ? "Click for WOE bins" : ""}">
      <td>${i + 1}</td>
      <td>${esc(r.feature)}${r.type ? ` <span class="method">${esc(r.type)}</span>` : ""}${notes ? `<div>${notes}</div>` : ""}</td>
      <td><span class="bar-cell"><i style="width:${12 + 60 * (r.iv / maxIv)}px"></i>${fmt(r.iv)}</span></td>
      <td><span class="tag ${strengthClass[r.strength] || ""}">${esc(r.strength)}</span></td>
      <td><span class="bar-cell"><i style="width:${imp ? 12 + 60 * (imp / maxImp) : 0}px;background:${COLORS.navy}"></i>${imp ? fmt(imp) : "–"}</span></td>
      <td>${r.missing_pct !== undefined ? pct(r.missing_pct / 100) : "–"}</td>
      <td class="gloss" title="${g ? esc(g.why) : ""}">${gloss || '<span class="sub">–</span>'}</td>
    </tr>`;
  }).join("");
  table.innerHTML = `
    <thead><tr><th>#</th><th>Variable</th><th>IV</th><th>Strength</th><th>Champion importance</th><th>Missing</th><th>What it means</th></tr></thead>
    <tbody>${rows}</tbody>`;
  table.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => toggleWoe(tr, shown[Number(tr.dataset.idx)])));
}

function toggleWoe(tr, r) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("woe")) { next.remove(); return; }
  document.querySelectorAll("#features-table tr.woe").forEach((d) => d.remove());
  const bins = r.bins || [];
  const maxAbs = Math.max(...bins.map((b) => Math.abs(b.woe)), 1e-9);
  const g = glossaryFor(r.feature);
  const woeBar = (w) => {
    const half = 55, len = Math.max(2, half * Math.abs(w) / maxAbs);
    const left = w >= 0 ? half : half - len;
    return `<span class="woe-bar"><i style="left:${left}px;width:${len}px;background:${w >= 0 ? COLORS.cyan : COLORS.bad}"></i></span>`;
  };
  const detail = document.createElement("tr");
  detail.className = "woe";
  detail.innerHTML = `<td colspan="7">
    ${g ? `<p class="sub" style="margin:0 0 8px"><b>${esc(g.label)}.</b> ${esc(g.why)}</p>` : ""}
    <table class="woe"><thead><tr><th>Bin</th><th>Count</th><th>Events</th><th>Event rate</th><th>WOE</th><th></th><th>IV contribution</th></tr></thead>
    <tbody>${bins.map((b) => `<tr><td>${esc(b.bin)}</td><td>${b.count.toLocaleString()}</td><td>${b.events.toLocaleString()}</td>
      <td>${pct(b.event_rate)}</td><td>${fmt(b.woe)}</td><td>${woeBar(b.woe)}</td><td>${fmt(b.iv_contribution, 4)}</td></tr>`).join("")}</tbody></table>
    <p class="sub" style="margin:8px 0 0">Positive WOE = fewer events than average (safer); negative = more events. Category labels are withheld.</p>
  </td>`;
  tr.after(detail);
}

// ---------------------------------------------------------------------------
// Diagnostics, drift, rows without an outcome
// ---------------------------------------------------------------------------
function renderDiagnostics(data) {
  const d = data.diagnostics;
  const panel = $("diag-panel");
  if (!d) { panel.style.display = "none"; return; }
  panel.style.display = "";
  const ci = d.auc_bootstrap_95_interval;
  const dup = d.test_rows_matching_training_predictors || 0;
  const tiles = [
    ["Test AUC 95% interval", ci ? `${fmt(ci[0])} – ${fmt(ci[1])}` : "n/a", "row bootstrap"],
    ["Test rows identical to train", `${dup.toLocaleString()}`, d.test_rows ? `of ${d.test_rows.toLocaleString()} (${pct(dup / d.test_rows)})` : ""],
    ["Constant / all-missing", `${(d.constant_or_all_missing_features || []).length}`, "features"],
    ["Correlated pairs |r| ≥ 0.8", `${(d.correlated_pairs || []).length}`, "top 10 listed"],
  ];
  $("diag-tiles").innerHTML = tiles.map(([l, v, s]) => `<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="label">${esc(s)}</div></div>`).join("");
  const sizes = d.cohort_sizes || {}, pos = d.cohort_positives || {};
  const cohortTable = `<h4 style="margin:0 0 6px;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--ink-3)">Cohorts</h4>
    <table class="mini"><thead><tr><th>Cohort</th><th>Rows</th><th>Positives</th><th>Rate</th></tr></thead><tbody>` +
    SPLITS.filter((s) => sizes[s] !== undefined).map((s) => `<tr><td>${esc(SPLIT_LABELS[s])}</td><td>${sizes[s].toLocaleString()}</td><td>${(pos[s] || 0).toLocaleString()}</td><td>${sizes[s] ? pct((pos[s] || 0) / sizes[s]) : "–"}</td></tr>`).join("") + "</tbody></table>";
  const consts = (d.constant_or_all_missing_features || []);
  const constHtml = consts.length ? `<p class="sub" style="margin:10px 0 0">Constant or all-missing: ${consts.map((c) => `<span class="method">${esc(c)}</span>`).join(" ")}</p>` : "";
  const pairs = (d.correlated_pairs || []).slice(0, 10);
  const pairsHtml = pairs.length ? `<h4 style="margin:14px 0 6px;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--ink-3)">Highly correlated pairs</h4>
    <table class="mini diag"><thead><tr><th>Pair</th><th>|r|</th></tr></thead><tbody>${pairs.map((p) => `<tr><td>${esc(p.left)} ~ ${esc(p.right)}</td><td>${fmt(p.abs_correlation)}</td></tr>`).join("")}</tbody></table>` : "";
  $("diag-body").innerHTML = cohortTable + constHtml + pairsHtml + (d.bootstrap_note ? `<p class="sub" style="margin-top:10px">${esc(d.bootstrap_note)}</p>` : "");
}

let _driftCohort = "oot";
function renderDrift(drift) {
  const panel = $("drift-panel");
  if (!drift || !Object.keys(drift).length) { panel.style.display = "none"; return; }
  panel.style.display = "";
  const cohorts = SPLITS.filter((s) => drift[s]);
  if (!cohorts.includes(_driftCohort)) _driftCohort = cohorts[cohorts.length - 1];
  $("drift-tiles").innerHTML = cohorts.map((c) => {
    const s = drift[c].score;
    return `<div class="psi-tile ${esc(s.status)}"><div class="label">${esc(SPLIT_LABELS[c])} · score PSI</div><div class="value">${fmt(s.psi)}</div>
      <div><span class="st ${esc(s.status)}">${esc(s.status)}</span> <span class="label">· ${drift[c].n_significant} significant, ${drift[c].n_moderate} moderate feature(s)</span></div></div>`;
  }).join("");
  $("drift-method").textContent = drift[_driftCohort].method || "";
  const toggle = $("drift-toggle");
  toggle.innerHTML = cohorts.map((c) => `<button data-cohort="${esc(c)}" class="${c === _driftCohort ? "active" : ""}">${esc(SPLIT_LABELS[c])}</button>`).join("");
  toggle.querySelectorAll("button").forEach((btn) => btn.onclick = () => {
    _driftCohort = btn.dataset.cohort;
    toggle.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    renderDriftChart(drift[_driftCohort]);
  });
  renderDriftChart(drift[_driftCohort]);
}

function renderDriftChart(entry) {
  const feats = [...(entry.features || [])].slice(0, 15).reverse();
  if (!feats.length) { $("drift-chart").innerHTML = '<p class="sub">No feature profiles.</p>'; return; }
  const maxX = Math.max(0.3, ...feats.map((f) => f.psi)) * 1.1;
  Plotly.newPlot("drift-chart", [{
    x: feats.map((f) => f.psi), y: feats.map((f) => cleanFeature(f.feature)), type: "bar", orientation: "h",
    marker: { color: feats.map((f) => PSI_COLORS[f.status] || COLORS.muted[0]) },
    customdata: feats.map((f) => f.missing_pct), hovertemplate: "%{y}<br>PSI %{x:.4f}<br>missing %{customdata:.1f}%<extra></extra>",
  }], baseLayout({ margin: { t: 6, l: 200, r: 16, b: 40 }, showlegend: false,
    xaxis: { title: "PSI vs training", range: [0, maxX], gridcolor: COLORS.line },
    shapes: [0.1, 0.25].map((x) => ({ type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper", line: { color: x === 0.1 ? COLORS.warn : COLORS.bad, dash: "dot", width: 1.2 } })),
    annotations: [{ x: 0.1, y: 1, yref: "paper", text: "0.10", showarrow: false, font: { size: 10, color: COLORS.warn }, yanchor: "bottom" },
                  { x: 0.25, y: 1, yref: "paper", text: "0.25", showarrow: false, font: { size: 10, color: COLORS.bad }, yanchor: "bottom" }] }), PLOT_CONFIG);
}

function renderUnlabelled(data) {
  const panel = $("unlabelled-panel");
  const u = data.unlabelled_scoring;
  if (!u || !u.rows) { panel.style.display = "none"; return; }
  panel.style.display = "";
  $("unlabelled-note").textContent = u.note || "";
  $("unlabelled-tiles").innerHTML = [["Rows scored", u.rows.toLocaleString()], ["Mean Nu Score", fmt(u.mean_score, 0)]]
    .map(([l, v]) => `<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join("");
  const order = ["A+", "A", "B", "C", "D", "E"];
  $("unlabelled-bands").innerHTML = order.filter((b) => u.band_counts && u.band_counts[b] !== undefined)
    .map((b) => `<span class="band-pill" style="background:${bandColor(b)}">${esc(b)} · ${u.band_counts[b].toLocaleString()}</span>`).join(" ");
  const btn = $("unlabelled-download-btn");
  btn.onclick = () => downloadBlob(`models/${data.model_id}/score-file/${u.file_id}/download`, `nu-score-unlabelled-${new Date().toISOString().slice(0, 10)}.csv`, "unlabelled-status", "unlabelled-error");
}

async function downloadBlob(path, filename, statusId, errorId) {
  setError(errorId, "");
  setStatus(statusId, "Preparing download…");
  try {
    const res = await fetch(`${API}/${path}`);
    if (!res.ok) throw await apiError(res);
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    setError(errorId, `Download failed: ${err.message}`);
  } finally {
    setStatus(statusId, "");
  }
}

// ---------------------------------------------------------------------------
// Nu Score section
// ---------------------------------------------------------------------------
let _nu = null;
let _bandSplit = "test";

function bandColor(b) { return BAND_COLORS[b] || COLORS.muted[0]; }

function renderNuScore(nu) {
  _nu = nu;
  $("nu-base").textContent = `Base model: ${nu.base_model}`;

  // Scale legend from bands (ordered A+ … E, high score to low).
  const bands = nu.bands;
  $("scale-legend").innerHTML = [...bands].reverse().map((b) => {
    const width = (b.max_score - b.min_score + 1) / 10.01;
    return `<span title="${esc(b.band)} ${b.min_score}–${b.max_score}" style="width:${width}%;background:${bandColor(b.band)}"></span>`;
  }).join("");
  $("scale-ticks").innerHTML = ["0", "250", "500 · even odds", "750", "1000"].map((t) => `<span>${esc(t)}</span>`).join("");

  const adj = nu.ai_adjustment || {};
  const tiles = [
    ["Base model AUC", fmt(adj.auc_base), nu.base_model],
    ["AI-enhanced AUC", fmt(adj.auc_enhanced), adj.auc_enhanced >= adj.auc_base ? `+${fmt(adj.auc_enhanced - adj.auc_base)} over base` : `${fmt(adj.auc_enhanced - adj.auc_base)} vs base`],
    ["AI adjustment", `${fmt(adj.mean_abs_points, 0)} pts`, "mean |change| per case"],
    ["ECE after calibration", fmt((nu.ece_after_calibration || {}).test), `OOT ${fmt((nu.ece_after_calibration || {}).oot)}`],
  ];
  $("nu-ai-tiles").innerHTML = tiles.map(([l, v, d]) =>
    `<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="label">${esc(d)}</div></div>`).join("");

  renderNuHistogram(nu);
  renderBandTable(nu, _bandSplit);
  document.querySelectorAll("#band-split-toggle button").forEach((btn) => {
    btn.onclick = () => {
      _bandSplit = btn.dataset.split;
      document.querySelectorAll("#band-split-toggle button").forEach((b) => b.classList.toggle("active", b === btn));
      renderBandTable(nu, _bandSplit);
    };
  });

  const slider = $("cut-slider");
  const start = nu.bands.length >= 3 ? nu.bands[nu.bands.length - 2].min_score : 500;
  slider.value = Math.round(start / 20) * 20;
  slider.oninput = () => renderCutoff(nu, Number(slider.value));
  renderCutoff(nu, Number(slider.value));
  renderCutoffChart(nu);
}

function renderNuHistogram(nu) {
  const h = nu.histogram;
  const shapes = nu.bands.map((b) => ({
    type: "rect", xref: "x", yref: "paper", x0: b.min_score, x1: b.max_score + 1, y0: 0, y1: 1,
    fillcolor: bandColor(b.band), opacity: 0.10, line: { width: 0 },
  }));
  const annotations = nu.bands.map((b) => ({
    x: (b.min_score + b.max_score) / 2, y: 1, yref: "paper", text: b.band, showarrow: false,
    font: { size: 11, color: bandColor(b.band) }, yanchor: "bottom",
  }));
  Plotly.newPlot("nu-hist", [
    { x: h.edges.map((e) => e + 12.5), y: h.good, type: "bar", name: "Good", marker: { color: COLORS.cyan }, width: 25, hovertemplate: "score %{x}<br>good %{y}<extra></extra>" },
    { x: h.edges.map((e) => e + 12.5), y: h.bad, type: "bar", name: "Bad", marker: { color: COLORS.navy }, width: 25, hovertemplate: "score %{x}<br>bad %{y}<extra></extra>" },
  ], baseLayout({ barmode: "stack", shapes, annotations, margin: { t: 24, l: 50, r: 10, b: 60 },
    xaxis: { title: "Nu Score", range: [0, 1000], gridcolor: COLORS.line }, yaxis: { title: "Cases", gridcolor: COLORS.line } }), PLOT_CONFIG);
}

function renderBandTable(nu, split) {
  const rows = nu.band_table[split] || [];
  $("band-table").innerHTML = `
    <thead><tr><th>Band</th><th>Score range</th><th>Share</th><th>Bad rate</th><th>Lift</th><th>Bads at or below</th><th>PSI contrib.</th></tr></thead>
    <tbody>${rows.map((r) => `<tr>
      <td><span class="band-pill" style="background:${bandColor(r.band)}">${esc(r.band)}</span></td>
      <td>${r.min_score}–${r.max_score}</td>
      <td>${pct(r.share)}</td>
      <td>${pct(r.bad_rate)}</td>
      <td>${r.lift === null ? "–" : `${fmt(r.lift, 2)}×`}</td>
      <td>${pct(r.cumulative_bad_capture)}</td>
      <td>${r.psi_contribution === undefined ? "–" : fmt(r.psi_contribution, 4)}</td>
    </tr>`).join("")}</tbody>`;
}

function renderCutoff(nu, cut) {
  $("cut-value").textContent = cut;
  const row = nu.cutoff_curve.reduce((best, r) => (Math.abs(r.cutoff - cut) < Math.abs(best.cutoff - cut) ? r : best), nu.cutoff_curve[0]);
  const band = nu.bands.find((b) => cut >= b.min_score && cut <= b.max_score);
  $("cut-tiles").innerHTML = [
    ["Approval rate", pct(row.approval_rate)],
    ["Bad rate among approved", pct(row.approved_bad_rate)],
    ["Bads rejected", pct(row.bads_rejected)],
    ["Band at cut-off", band ? band.band : "–"],
  ].map(([l, v]) => `<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join("");
  const chart = $("cut-chart");
  if (chart.data) {
    Plotly.relayout(chart, { shapes: [{ type: "line", x0: cut, x1: cut, y0: 0, y1: 1, yref: "paper", line: { color: COLORS.navy, dash: "dot", width: 1.5 } }] });
  }
}

function renderCutoffChart(nu) {
  const c = nu.cutoff_curve;
  Plotly.newPlot("cut-chart", [
    { x: c.map((r) => r.cutoff), y: c.map((r) => r.approval_rate), mode: "lines", name: "Approval rate", line: { color: COLORS.cyan, width: 3 }, hovertemplate: "cut %{x}<br>approve %{y:.1%}<extra></extra>" },
    { x: c.map((r) => r.cutoff), y: c.map((r) => r.approved_bad_rate), mode: "lines", name: "Bad rate among approved", line: { color: COLORS.bad, width: 2 }, hovertemplate: "cut %{x}<br>bad rate %{y:.1%}<extra></extra>" },
    { x: c.map((r) => r.cutoff), y: c.map((r) => r.bads_rejected), mode: "lines", name: "Bads rejected", line: { color: COLORS.muted[1], width: 2, dash: "dot" }, hovertemplate: "cut %{x}<br>bads rejected %{y:.1%}<extra></extra>" },
  ], baseLayout({ margin: { t: 40, l: 50, r: 10, b: 50 }, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 11 } },
    xaxis: { title: "Approve at or above", range: [0, 1000], gridcolor: COLORS.line }, yaxis: { tickformat: ".0%", range: [0, 1], gridcolor: COLORS.line } }), PLOT_CONFIG);
  renderCutoff(nu, Number($("cut-slider").value));
}

function renderCalibration(nu) {
  const traces = [{ x: [0, 1], y: [0, 1], mode: "lines", name: "Perfect", line: { color: COLORS.ink3, dash: "dash", width: 1 }, hoverinfo: "skip" }];
  const cols = { test: COLORS.cyan, oot: COLORS.navy };
  if (nu.reliability_before) {
    const b = nu.reliability_before;
    traces.push({ x: b.predicted, y: b.observed, mode: "lines+markers", name: "Test before calibration", line: { color: COLORS.muted[0], width: 2, dash: "dot" }, marker: { size: 5 },
      hovertemplate: "predicted %{x:.3f}<br>observed %{y:.3f}<extra>before</extra>" });
  }
  const eff = nu.calibration_effect;
  if (eff) {
    $("calib-sub").textContent = `Fitted on the calibration cohort (${nu.bands_fitted_on || "calibration"}). Test Brier ${fmt(eff.brier_before, 4)} → ${fmt(eff.brier_after, 4)}, ECE ${fmt(eff.ece_before, 4)} → ${fmt(eff.ece_after, 4)}. Dotted = before, solid = after.`;
  }
  for (const [split, curve] of Object.entries(nu.calibration || {})) {
    traces.push({ x: curve.predicted, y: curve.observed, mode: "lines+markers", name: `${SPLIT_LABELS[split] || split} (ECE ${fmt((nu.ece_after_calibration || {})[split])})`,
      line: { color: cols[split] || COLORS.muted[0], width: split === "test" ? 3 : 2 }, marker: { size: curve.count.map((n) => Math.max(5, Math.min(16, Math.sqrt(n) / 2))) },
      hovertemplate: "predicted %{x:.3f}<br>observed %{y:.3f}<extra>" + esc(split) + "</extra>" });
  }
  const maxV = Math.max(0.2, ...traces.slice(1).flatMap((t) => t.x.concat(t.y))) * 1.05;
  Plotly.newPlot("calib-chart", traces, baseLayout({
    xaxis: { title: "Predicted probability", range: [0, maxV], gridcolor: COLORS.line },
    yaxis: { title: "Observed bad rate", range: [0, maxV], gridcolor: COLORS.line },
  }), PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// Explainability
// ---------------------------------------------------------------------------
function renderShap(shap) {
  $("shap-sub").textContent = `SHAP (${shap.kind} explainer) on ${(shap.rows_explained || 0).toLocaleString()} test cases: how much each variable pushes a case towards the outcome.`;
  const imp = [...shap.importance].reverse();
  Plotly.newPlot("shap-bar", [{
    x: imp.map((i) => i.mean_abs_shap), y: imp.map((i) => cleanFeature(i.feature)), type: "bar", orientation: "h",
    marker: { color: COLORS.cyan }, hovertemplate: "%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>",
  }], baseLayout({ margin: { t: 6, l: 190, r: 10, b: 40 }, showlegend: false, xaxis: { title: "mean |SHAP|", gridcolor: COLORS.line } }), PLOT_CONFIG);

  const rows = shap.beeswarm || [];
  const xs = [], ys = [], cs = [], texts = [];
  rows.forEach((row, i) => {
    const yBase = rows.length - 1 - i;
    row.shap.forEach((v, k) => {
      xs.push(v); ys.push(yBase + (Math.random() - 0.5) * 0.55); cs.push(row.value_pct[k]);
      texts.push(`${cleanFeature(row.feature)} = ${row.value[k]}`);
    });
  });
  Plotly.newPlot("shap-bee", [{
    x: xs, y: ys, mode: "markers", type: "scattergl", text: texts,
    marker: { size: 5, opacity: 0.65, color: cs, colorscale: [[0, COLORS.cyan], [1, COLORS.navy]], cmin: 0, cmax: 1, showscale: true,
      colorbar: { title: "value", thickness: 8, tickvals: [0, 1], ticktext: ["low", "high"], len: 0.6 } },
    hovertemplate: "%{text}<br>SHAP %{x:.4f}<extra></extra>",
  }], baseLayout({ margin: { t: 6, l: 190, r: 40, b: 40 }, showlegend: false,
    xaxis: { title: "SHAP value (→ higher risk)", gridcolor: COLORS.line, zeroline: true, zerolinecolor: COLORS.ink3 },
    yaxis: { tickvals: rows.map((_, i) => rows.length - 1 - i), ticktext: rows.map((r) => cleanFeature(r.feature)), gridcolor: COLORS.line } }), PLOT_CONFIG);

  const dep = $("shap-dep");
  dep.innerHTML = (shap.dependence || []).map((d, i) => `<div><h3 style="font-size:0.9rem;margin:0 0 4px">Dependence · ${esc(cleanFeature(d.feature))}</h3><div id="dep-${i}" class="chart short"></div></div>`).join("");
  (shap.dependence || []).forEach((d, i) => {
    Plotly.newPlot(`dep-${i}`, [{ x: d.x, y: d.shap, mode: "markers", type: "scattergl", marker: { size: 5, color: COLORS.cyan, opacity: 0.6 },
      hovertemplate: "value %{x}<br>SHAP %{y:.4f}<extra></extra>" }],
      baseLayout({ margin: { t: 6, l: 50, r: 10, b: 40 }, showlegend: false, xaxis: { title: cleanFeature(d.feature), gridcolor: COLORS.line }, yaxis: { title: "SHAP", gridcolor: COLORS.line, zeroline: true } }), PLOT_CONFIG);
  });
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function baseLayout(extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Host Grotesk, Segoe UI, sans-serif", color: "#44546B", size: 12 },
    margin: { t: 10, l: 56, r: 16, b: 60 },
    // Below the axis title; charts are 380px tall so this stays inside the card.
    legend: { orientation: "h", y: -0.32, yanchor: "top" },
    xaxis: { gridcolor: COLORS.line, zeroline: false },
    yaxis: { gridcolor: COLORS.line, zeroline: false },
    ...extra,
  };
}

// Champion in brand cyan, challengers in muted navy tones.
function modelColor(index) {
  return index === 0 ? COLORS.cyan : COLORS.muted[(index - 1) % COLORS.muted.length];
}

function renderIvChart(rows) {
  const data = [...rows].reverse();
  const colors = data.map((r) => (r.strength === "suspicious" ? "#E9B949" : r.strength === "useless" ? "#B7C4D6" : COLORS.cyan));
  Plotly.newPlot("iv-chart", [{
    x: data.map((r) => r.iv),
    y: data.map((r) => r.feature),
    type: "bar", orientation: "h",
    marker: { color: colors },
    hovertemplate: "%{y}<br>IV %{x:.3f}<extra></extra>",
  }], baseLayout({ margin: { t: 10, l: 240, r: 16, b: 40 }, showlegend: false,
    xaxis: { title: "Information Value", gridcolor: COLORS.line } }), PLOT_CONFIG);
}

function renderNullChart(columns) {
  const data = [...columns].filter((c) => c.null_pct > 0).sort((a, b) => b.null_pct - a.null_pct).slice(0, 20).reverse();
  if (!data.length) {
    $("null-chart").innerHTML = '<p class="sub" style="padding-top:20px">No missing values in any column.</p>';
    return;
  }
  Plotly.newPlot("null-chart", [{
    x: data.map((c) => c.null_pct),
    y: data.map((c) => c.name),
    type: "bar", orientation: "h",
    marker: { color: data.map((c) => (c.null_pct > 30 ? "#C8453B" : COLORS.muted[0])) },
    hovertemplate: "%{y}<br>%{x:.1f}% missing<extra></extra>",
  }], baseLayout({ margin: { t: 10, l: 240, r: 16, b: 40 }, showlegend: false,
    xaxis: { title: "% missing", gridcolor: COLORS.line, range: [0, 100] } }), PLOT_CONFIG);
}

function rocOf(r, split) { return r.evaluations ? r.evaluations[split].roc_curve : r.roc_curve[split]; }

function renderRocCurves(ranked) {
  const traces = ranked.map((r, i) => ({
    x: rocOf(r, "test").fpr,
    y: rocOf(r, "test").tpr,
    mode: "lines",
    name: `${r.name} (${fmt(splitMetric(r, "test", "roc_auc"))})`,
    line: { color: modelColor(i), width: i === 0 ? 3.5 : 1.5 },
    hovertemplate: "FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + esc(r.name) + "</extra>",
  })).reverse();  // draw the champion last so it sits on top
  traces.push({
    x: [0, 1], y: [0, 1], mode: "lines", name: "Random",
    line: { color: COLORS.ink3, dash: "dash", width: 1 }, hoverinfo: "skip", showlegend: false,
  });
  Plotly.newPlot("roc-chart", traces, baseLayout({
    margin: { t: 10, l: 56, r: 16, b: 110 },
    legend: { orientation: "h", y: -0.2, yanchor: "top", font: { size: 10 } },
    xaxis: { title: "False positive rate", gridcolor: COLORS.line, range: [0, 1] },
    yaxis: { title: "True positive rate", gridcolor: COLORS.line, range: [0, 1] },
  }), PLOT_CONFIG);
}

function renderStability(ranked) {
  const traces = ranked.map((r, i) => ({
    x: activeSplits(r).map((s) => SPLIT_LABELS[s]),
    y: activeSplits(r).map((s) => splitMetric(r, s, "roc_auc")),
    mode: "lines+markers",
    name: r.name,
    line: { color: modelColor(i), width: i === 0 ? 3.5 : 1.5 },
    marker: { size: i === 0 ? 9 : 6 },
    hovertemplate: "%{x}: %{y:.3f}<extra>" + esc(r.name) + "</extra>",
  })).reverse();
  Plotly.newPlot("stability-chart", traces, baseLayout({
    margin: { t: 10, l: 56, r: 16, b: 110 },
    legend: { orientation: "h", y: -0.2, yanchor: "top", font: { size: 10 } },
    yaxis: { title: "ROC-AUC", gridcolor: COLORS.line },
  }), PLOT_CONFIG);
}

function renderMetrics(ranked) {
  const metrics = [["roc_auc", "ROC-AUC"], ["ks", "KS"], ["pr_auc", "PR-AUC"], ["f1", "F1"]];
  const shades = [COLORS.navy, COLORS.cyan, "#8FA3BF", "#B7C4D6"];
  const traces = metrics.map(([key, label], i) => ({
    x: ranked.map((r) => r.name),
    y: ranked.map((r) => testMetrics(r)[key]),
    name: label,
    type: "bar",
    marker: { color: shades[i] },
    hovertemplate: "%{x}<br>" + label + " %{y:.3f}<extra></extra>",
  })).filter((t) => t.y.some((v) => v !== undefined));
  Plotly.newPlot("metrics-chart", traces, baseLayout({
    barmode: "group",
    bargap: 0.25,
    margin: { t: 10, l: 56, r: 16, b: 110 },
    xaxis: { tickangle: -25, gridcolor: COLORS.line, tickfont: { size: 10 } },
    yaxis: { range: [0, 1], gridcolor: COLORS.line },
    legend: { orientation: "h", y: -0.5, yanchor: "top" },
  }), PLOT_CONFIG);
}

function renderFeatureImportance(best) {
  const data = (best.feature_importance || []).slice(0, 15).reverse();
  $("feature-chart-sub").textContent = best.importance_method && best.importance_method !== "native"
    ? `Features the champion relies on most (${best.importance_method}; larger = bigger AUC loss when shuffled).`
    : "Features the champion relies on most (model-native importance).";
  if (!data.length) {
    $("feature-chart").innerHTML = '<p class="sub">This model does not expose feature importances.</p>';
    return;
  }
  Plotly.newPlot("feature-chart", [{
    x: data.map((d) => Math.abs(d.importance)),
    y: data.map((d) => cleanFeature(d.feature)),
    type: "bar",
    orientation: "h",
    marker: { color: COLORS.cyan },
    hovertemplate: "%{y}<br>%{x:.4f}<extra></extra>",
  }], baseLayout({ margin: { t: 10, l: 190, r: 16, b: 40 }, showlegend: false }), PLOT_CONFIG);
}

function renderConfusion(best) {
  const matrix = best.evaluations ? best.evaluations.test.confusion : best.confusion.test;
  const threshold = best.evaluations ? best.evaluations.test.metrics.threshold : 0.5;
  $("confusion-sub").textContent = `Counts at the F1-optimal probability threshold of ${fmt(threshold)}.`;
  const labels = ["Negative (0)", "Positive (1)"];
  const total = matrix.flat().reduce((a, b) => a + b, 0) || 1;
  Plotly.newPlot("confusion-chart", [{
    z: matrix,
    x: labels.map((l) => `Predicted ${l}`),
    y: labels.map((l) => `Actual ${l}`),
    type: "heatmap",
    colorscale: [[0, "#F5F7FA"], [0.5, COLORS.cyan], [1, COLORS.navy]],
    showscale: false,
    text: matrix.map((row) => row.map((v) => `${v.toLocaleString()}<br>${((v / total) * 100).toFixed(1)}%`)),
    texttemplate: "%{text}",
    hovertemplate: "%{y}<br>%{x}<br>%{z}<extra></extra>",
  }], baseLayout({
    margin: { t: 10, l: 130, r: 16, b: 60 },
    yaxis: { autorange: "reversed" },
    showlegend: false,
  }), PLOT_CONFIG);
}

function renderLift(best) {
  const lift = best.evaluations ? best.evaluations.test.lift : null;
  if (!lift || !lift.length) { $("lift-chart").innerHTML = '<p class="sub">Not available for this run.</p>'; return; }
  Plotly.newPlot("lift-chart", [
    { x: lift.map((r) => `D${r.decile}`), y: lift.map((r) => r.lift), type: "bar", name: "Lift", marker: { color: COLORS.cyan }, hovertemplate: "%{x}<br>lift %{y:.2f}×<br>bad rate " + "%{customdata:.1%}<extra></extra>", customdata: lift.map((r) => r.bad_rate) },
    { x: lift.map((r) => `D${r.decile}`), y: lift.map((r) => r.cumulative_capture), type: "scatter", mode: "lines+markers", name: "Cumulative bads captured", yaxis: "y2", line: { color: COLORS.navy, width: 2.5 }, hovertemplate: "%{x}<br>captured %{y:.1%}<extra></extra>" },
  ], baseLayout({
    margin: { t: 10, l: 50, r: 50, b: 70 },
    xaxis: { title: "Decile (riskiest → safest)", gridcolor: COLORS.line },
    yaxis: { title: "Lift", gridcolor: COLORS.line, rangemode: "tozero" },
    yaxis2: { title: "Captured", overlaying: "y", side: "right", tickformat: ".0%", range: [0, 1.02], showgrid: false },
    legend: { orientation: "h", y: -0.28, yanchor: "top" },
  }), PLOT_CONFIG);
}

function renderDataset(summary) {
  const cards = [
    ["Rows", summary.n_rows.toLocaleString()],
    ["Features", summary.n_features],
    ["Numeric", summary.feature_types.numeric],
    ["Categorical", summary.feature_types.categorical],
    ["Validation", summary.split_strategy === "time" ? "Time-based cohorts" : "Random cohorts"],
  ];
  if (summary.unlabelled_rows) cards.push(["Without outcome", `${summary.unlabelled_rows.toLocaleString()} rows scored only`]);
  if (summary.tier) cards.push(["Search depth", summary.tier]);
  if (summary.n_models) cards.push(["Models trained", summary.n_models]);
  if (summary.training_seconds) cards.push(["Training time", summary.training_seconds >= 60 ? `${(summary.training_seconds / 60).toFixed(1)} min` : `${summary.training_seconds}s`]);
  if (summary.train_rows_used) cards.push(["Fit on", `${summary.train_rows_used.toLocaleString()} rows`]);
  if (summary.excluded_columns && summary.excluded_columns.length) {
    cards.push(["Left out", summary.excluded_columns.length + " column(s)"]);
  }
  $("dataset-stats").innerHTML = cards.map(([label, value]) =>
    `<div class="stat"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join("");

  const sizes = summary.cohort_sizes || {};
  $("class-balance").innerHTML = Object.entries(summary.class_balance).map(([split, ratios]) => {
    const text = Object.entries(ratios).map(([label, ratio]) => `${label}: ${fmt(ratio * 100, 1)}%`).join(" · ");
    const range = summary.time_ranges && summary.time_ranges[split] ? ` · ${summary.time_ranges[split].join(" → ")}` : "";
    const n = sizes[split] !== undefined ? ` · n=${sizes[split].toLocaleString()}` : "";
    return `<span class="split-badge">${esc(SPLIT_LABELS[split] || split)}${esc(n)} · ${esc(text)}${esc(range)}</span>`;
  }).join("");
  const oot = summary.split && summary.split.oot_status;
  $("dataset-sub").textContent = oot ? `Shape, validation design and class balance per cohort. Out-of-time cohort: ${oot}.` : "Shape, validation design and class balance per cohort.";
  $("stability-sub").textContent = oot ? `ROC-AUC on Train, Validation, Calibration, Test and Out-of-Time (${oot}). Flat lines are stable models.` : "ROC-AUC on Train, Validation, Calibration, Test and Out-of-Time. Flat lines are stable models.";
  const log = summary.normalisation_log || [];
  const details = $("cleanup-details");
  details.style.display = log.length ? "" : "none";
  details.open = false;
  $("cleanup-list").innerHTML = log.map((c) => `<li><b>${esc(c.column)}</b>: ${esc(c.action)}</li>`).join("");
}

// ---------------------------------------------------------------------------
// Scoring new applications
// ---------------------------------------------------------------------------
async function loadScoringForm(modelId, championName) {
  scoringSchema = null;
  batchFileId = null;
  $("score-out").style.display = "none";
  $("batch-out").style.display = "none";
  setError("score-error", "");
  setError("batch-error", "");
  $("scoring-model").textContent = `Model ${championName || ""} · id ${modelId.slice(0, 8)}`;
  $("score-form").innerHTML = '<p class="sub">Loading the input schema…</p>';
  try {
    const schema = await (await request("GET", `models/${modelId}/schema`)).json();
    scoringSchema = { ...schema, model_id: modelId };
    renderScoreForm(schema);
  } catch (err) {
    $("score-form").innerHTML = "";
    setError("score-error", `Scoring form unavailable: ${err.message}`);
  }
}

function renderScoreForm(schema) {
  const form = $("score-form");
  form.innerHTML = schema.fields.map((f, i) => {
    const id = `sf-${i}`;
    if (f.type === "category") {
      const opts = (f.options || []).map((o) => `<option value="${esc(o)}" ${o === f.default ? "selected" : ""}>${esc(o)}</option>`).join("");
      return `<div class="fld" data-name="${esc(f.name)}" data-type="category"><label for="${id}">${esc(f.name)}</label>
        <select id="${id}" class="sel">${opts}<option value="__other__">Other…</option></select>
        <input class="other" type="text" placeholder="type a value" /></div>`;
    }
    const range = f.range ? ` title="typical ${fmt(f.range[0], 2)} – ${fmt(f.range[1], 2)}"` : "";
    return `<div class="fld" data-name="${esc(f.name)}" data-type="number"><label for="${id}">${esc(f.name)}</label>
      <input id="${id}" type="number" step="any" value="${f.default === null || f.default === undefined ? "" : esc(+Number(f.default).toPrecision(6))}"${range} /></div>`;
  }).join("");
  form.querySelectorAll(".sel").forEach((sel) => {
    sel.addEventListener("change", () => {
      const other = sel.parentElement.querySelector(".other");
      other.style.display = sel.value === "__other__" ? "block" : "none";
    });
  });
}

function scoreFormRecord() {
  const record = {};
  document.querySelectorAll("#score-form .fld").forEach((fld) => {
    const name = fld.dataset.name;
    if (fld.dataset.type === "category") {
      const sel = fld.querySelector(".sel");
      record[name] = sel.value === "__other__" ? fld.querySelector(".other").value : sel.value;
    } else {
      const v = fld.querySelector("input").value;
      record[name] = v === "" ? null : Number(v);
    }
  });
  return record;
}

async function scoreApplication() {
  if (!scoringSchema) return;
  $("score-btn").disabled = true;
  setError("score-error", "");
  setStatus("score-status", "Scoring…");
  try {
    const out = await (await postJSON(`models/${scoringSchema.model_id}/score`, { records: [scoreFormRecord()] })).json();
    renderScore(out.results[0]);
  } catch (err) {
    setError("score-error", `Scoring failed: ${err.message}`);
  } finally {
    setStatus("score-status", "");
    $("score-btn").disabled = false;
  }
}
$("score-btn").addEventListener("click", scoreApplication);
$("score-reset-btn").addEventListener("click", () => { if (scoringSchema) renderScoreForm(scoringSchema); });

function renderScore(r) {
  $("score-out").style.display = "";
  drawGauge(r.nu_score, r.band);
  $("gauge-band").innerHTML = `<span class="band-pill" style="background:${bandColor(r.band)};font-size:1rem;padding:4px 14px">Band ${esc(r.band)}</span>`;
  $("score-tiles").innerHTML = [
    ["Nu Score", r.nu_score],
    ["Probability of outcome", pct(r.probability)],
    ["Base model score", r.base_score],
    ["AI adjustment", `${r.ai_adjustment > 0 ? "+" : ""}${r.ai_adjustment} pts`],
  ].map(([l, v]) => `<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join("");
  const reasons = r.reasons || [];
  $("score-reasons").innerHTML = reasons.length
    ? reasons.map((x) => {
        const g = glossaryFor(x.feature);
        const label = g ? `${esc(g.label)} <span class="v">(${esc(cleanFeature(x.feature))})</span>` : esc(cleanFeature(x.feature));
        const value = x.value !== null && x.value !== undefined ? ` <span class="v">= ${esc(typeof x.value === "number" ? +x.value.toPrecision(5) : x.value)}</span>` : "";
        return `<li><span>${label}${value}</span><span class="c">+${fmt(x.contribution, 3)}</span>${g ? `<span class="why">${esc(g.why)}</span>` : ""}</li>`;
      }).join("")
    : '<li><span class="sub">No factor raises the risk above the population baseline.</span></li>';
  $("score-out").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Semicircular gauge: band arcs proportional to their score ranges, needle at the score.
function drawGauge(score, band) {
  const svg = $("gauge");
  const cx = 150, cy = 160, R = 130, r = 100;
  const angle = (s) => Math.PI - (s / 1000) * Math.PI;
  const pt = (rad, a) => [cx + rad * Math.cos(a), cy - rad * Math.sin(a)];
  const arc = (s0, s1, color) => {
    const a0 = angle(s0), a1 = angle(s1);
    const [x0, y0] = pt(R, a0), [x1, y1] = pt(R, a1), [x2, y2] = pt(r, a1), [x3, y3] = pt(r, a0);
    const large = (a0 - a1) > Math.PI ? 1 : 0;
    return `<path d="M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3} Z" fill="${color}" opacity="0.9"/>`;
  };
  const bands = (scoringSchema && scoringSchema.bands) || (_nu && _nu.bands) || [];
  let parts = bands.map((b) => arc(b.min_score, b.max_score + 1, bandColor(b.band))).join("");
  if (!bands.length) parts = arc(0, 1000, COLORS.line);
  // Indicator sits on the arc itself so the number in the middle stays readable.
  const a = angle(score);
  const [ix, iy] = pt((R + r) / 2, a);
  const [tx0, ty0] = pt(r - 8, a), [tx1, ty1] = pt(R + 8, a);
  svg.innerHTML = `${parts}
    <line x1="${tx0}" y1="${ty0}" x2="${tx1}" y2="${ty1}" stroke="${COLORS.navy}" stroke-width="3" stroke-linecap="round"/>
    <circle cx="${ix}" cy="${iy}" r="9" fill="${COLORS.navy}" stroke="#fff" stroke-width="3"/>
    <text x="${cx}" y="${cy - 30}" text-anchor="middle" font-size="44" font-weight="600" fill="${COLORS.navy}" font-family="Host Grotesk, sans-serif">${score}</text>
    <text x="${cx}" y="${cy - 8}" text-anchor="middle" font-size="11" fill="${COLORS.ink3}" letter-spacing="1">NU SCORE</text>
    <text x="${cx - R}" y="${cy + 18}" font-size="11" fill="${COLORS.ink3}">0</text>
    <text x="${cx + R}" y="${cy + 18}" text-anchor="end" font-size="11" fill="${COLORS.ink3}">1000</text>`;
}

// ---- Batch scoring ----
async function batchScore(list) {
  const file = list && list[0];
  if (!file || !scoringSchema) return;
  if (!/\.csv$/i.test(file.name)) { setError("batch-error", "Please upload a .csv file."); return; }
  $("batch-title").textContent = `${file.name} · ${fmtBytes(file.size)}`;
  setError("batch-error", "");
  $("batch-out").style.display = "none";
  setStatus("batch-status", "Uploading, mapping columns and scoring…");
  try {
    const res = await fetch(`${API}/models/${scoringSchema.model_id}/score-file?filename=${encodeURIComponent(file.name)}`, {
      method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" },
    });
    if (!res.ok) throw await apiError(res);
    const out = await res.json();
    batchFileId = out.file_id;
    renderMapping(out);
  } catch (err) {
    setError("batch-error", `Batch scoring failed: ${err.message}`);
  } finally {
    setStatus("batch-status", "");
  }
}
wireDropZone("batch-area", "batch-file-input", batchScore);

function renderMapping(out) {
  $("batch-out").style.display = "";
  const missing = out.missing || [];
  $("batch-summary").textContent = `${(out.rows || 0).toLocaleString()} rows scored. ${missing.length ? `${missing.length} model variable(s) were not found and use typical values: ${missing.join(", ")}.` : "Every model variable was found in your file."}`;
  $("mapping-table").innerHTML = `<thead><tr><th>Model variable</th><th>Your column</th><th>Matched by</th></tr></thead><tbody>` +
    (out.mapping || []).map((m) => `<tr><td>${esc(m.schema_col)}</td><td class="${m.source_col ? "" : "missing"}">${m.source_col ? esc(m.source_col) : "not found → typical value"}</td>
      <td style="text-align:left"><span class="method ${esc(m.method)}">${esc(m.method)}</span></td></tr>`).join("") + "</tbody>";
  const box = $("batch-drift");
  const dr = out.drift;
  if (!dr || !dr.score) { box.innerHTML = ""; return; }
  const top = (dr.features || []).slice(0, 5);
  box.innerHTML = `<div class="psi-tile ${esc(dr.score.status)}" style="display:inline-block;min-width:220px"><div class="label">Score PSI vs training</div>
      <div class="value">${fmt(dr.score.psi)}</div><span class="st ${esc(dr.score.status)}">${esc(dr.score.status)}</span></div>
    ${top.length ? `<p class="sub" style="margin:8px 0 4px">Most drifted inputs in this file:</p><div class="chips">${top.map((f) => `<span class="chip" style="background:${PSI_COLORS[f.status]}22;color:${PSI_COLORS[f.status]}">${esc(cleanFeature(f.feature))} · ${fmt(f.psi)}</span>`).join("")}</div>` : ""}
    <p class="sub">${esc(dr.method || "")}</p>`;
}

$("batch-download-btn").addEventListener("click", async () => {
  if (!batchFileId || !scoringSchema) return;
  setError("batch-error", "");
  setStatus("batch-status", "Preparing download…");
  try {
    const res = await fetch(`${API}/models/${scoringSchema.model_id}/score-file/${batchFileId}/download`);
    if (!res.ok) throw await apiError(res);
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `nu-score-scored-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    setError("batch-error", `Download failed: ${err.message}`);
  } finally {
    setStatus("batch-status", "");
  }
});

// ---------------------------------------------------------------------------
// AI explanation and reports
// ---------------------------------------------------------------------------
function resultsPayload() {
  return {
    leaderboard: currentResults.leaderboard,
    results: currentResults.results,
    dataset_summary: currentResults.dataset,
  };
}

async function explainResults() {
  if (!currentResults) return;
  setError("action-error", "");
  setStatus("action-status", "Generating plain-English insights…");
  try {
    renderInsights(await (await postJSON("explain", resultsPayload())).json());
  } catch (err) {
    setError("action-error", `Explain failed: ${err.message}`);
  } finally {
    setStatus("action-status", "");
  }
}
$("explain-btn").addEventListener("click", explainResults);

function renderInsights(data) {
  _lastInsights = data;
  $("insight-summary").textContent = data.executive_summary || "";
  $("insight-best-model").textContent = data.best_model_analysis || "";
  const toList = (id, items) => { $(id).innerHTML = (items || []).map((t) => `<li>${esc(t)}</li>`).join(""); };
  toList("insight-features", data.feature_insights);
  toList("insight-flags", data.risk_flags);
  toList("insight-recs", data.recommendations);
  const panel = $("insights-panel");
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function downloadReport(format) {
  if (!currentResults) return;
  setError("action-error", "");
  setStatus("action-status", format === "pdf" ? "Building PDF report (about 10 seconds)…" : "Building HTML report…");
  try {
    const res = await postJSON(`report/${format}`, { ...resultsPayload(), ai_insights: _lastInsights });
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `nu-score-report-${new Date().toISOString().slice(0, 10)}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    setError("action-error", `Report generation failed: ${err.message}`);
  } finally {
    setStatus("action-status", "");
  }
}
document.querySelectorAll("[data-report]").forEach((btn) => {
  btn.addEventListener("click", () => downloadReport(btn.dataset.report));
});
