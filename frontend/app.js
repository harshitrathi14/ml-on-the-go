const BASE_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentSessionId = null;
let currentResults   = null;   // { leaderboard, results, dataset }
let _lastInsights    = null;   // cached AI insights for embedding in reports

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((btn, i) => {
    const tabs = ["synthetic", "csv"];
    btn.classList.toggle("active", tabs[i] === name);
  });
  document.getElementById("tab-synthetic").classList.toggle("active", name === "synthetic");
  document.getElementById("tab-csv").classList.toggle("active", name === "csv");
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
const runBtn       = document.getElementById("run-btn");
const loading      = document.getElementById("loading");
const datasetStats = document.getElementById("dataset-stats");
const classBalance = document.getElementById("class-balance");
const leaderboardTable = document.getElementById("leaderboard");

function setLoading(id, active) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("active", active);
}

function number(value, digits = 3) {
  return Number(value).toFixed(digits);
}

// ---------------------------------------------------------------------------
// Upload CSV
// ---------------------------------------------------------------------------
let _selectedFile = null;

function handleFileSelect(event) {
  _selectedFile = event.target.files[0] || null;
  const uploadBtn = document.getElementById("upload-btn");
  if (_selectedFile) {
    const area = document.getElementById("upload-area");
    area.querySelector("p").textContent = _selectedFile.name;
    uploadBtn.disabled = false;
  }
}

// Drag-and-drop
(function () {
  const area = document.getElementById("upload-area");
  area.addEventListener("dragover", (e) => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", (e) => {
    e.preventDefault();
    area.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
      _selectedFile = file;
      area.querySelector("p").textContent = file.name;
      document.getElementById("upload-btn").disabled = false;
    }
  });
})();

async function uploadCSV() {
  if (!_selectedFile) return;

  setLoading("upload-loading", true);
  document.getElementById("upload-btn").disabled = true;
  document.getElementById("ai-analysis-card").style.display = "none";

  try {
    const form = new FormData();
    form.append("file", _selectedFile);

    const res = await fetch(`${BASE_URL}/upload-csv`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    currentSessionId = data.session_id;
    renderAIAnalysis(data.ai_analysis, data.columns);
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  } finally {
    setLoading("upload-loading", false);
    document.getElementById("upload-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Render AI analysis card
// ---------------------------------------------------------------------------
function renderAIAnalysis(analysis, columns) {
  const card = document.getElementById("ai-analysis-card");
  card.style.display = "block";

  // Meta chips
  const chips = document.getElementById("ai-meta-chips");
  chips.innerHTML = "";
  const confClass = { high: "chip-high", medium: "chip-medium", low: "chip-low" };
  chips.innerHTML = `
    <span class="ai-meta-chip chip-info">Problem: ${analysis.problem_type || "unknown"}</span>
    <span class="ai-meta-chip ${confClass[analysis.confidence] || "chip-info"}">Confidence: ${analysis.confidence || "?"}</span>
    <span class="ai-meta-chip chip-info">Strategy: ${(analysis.binarization_strategy || {}).strategy || "unknown"}</span>
  `;

  // Recommendation
  document.getElementById("ai-recommendation").textContent = analysis.recommendation || "";

  // Quality issues
  const issues = analysis.data_quality_issues || [];
  const qSection = document.getElementById("quality-issues-section");
  if (issues.length > 0) {
    qSection.style.display = "";
    const pills = document.getElementById("quality-pills");
    pills.innerHTML = issues.map((iss) => {
      const cls = iss.severity === "high" ? "pill-high" : iss.severity === "medium" ? "pill-medium" : "pill-low";
      return `<span class="quality-pill ${cls}" title="${iss.description}">${iss.column}: ${iss.issue_type}</span>`;
    }).join("");
  } else {
    qSection.style.display = "none";
  }

  // Feature notes
  const notesList = document.getElementById("feature-notes-list");
  notesList.innerHTML = (analysis.feature_notes || []).map((n) => `<li>${n}</li>`).join("");

  // Target column dropdown — pre-select AI suggestion
  const sel = document.getElementById("target-col-select");
  sel.innerHTML = columns.map((c) => `<option value="${c}">${c}</option>`).join("");
  if (analysis.suggested_target_col && columns.includes(analysis.suggested_target_col)) {
    sel.value = analysis.suggested_target_col;
  }

  // Positive class hint
  const bs = analysis.binarization_strategy || {};
  if (bs.positive_class) {
    document.getElementById("positive-class-input").value = bs.positive_class;
  }
}

// ---------------------------------------------------------------------------
// Train on uploaded CSV
// ---------------------------------------------------------------------------
async function trainCSV() {
  if (!currentSessionId) { alert("No CSV uploaded."); return; }

  const targetCol     = document.getElementById("target-col-select").value;
  const positiveClass = document.getElementById("positive-class-input").value.trim() || null;
  const payload = { session_id: currentSessionId, target_col: targetCol, seed: 42 };
  if (positiveClass) payload.positive_class = positiveClass;

  setLoading("csv-train-loading", true);
  document.getElementById("train-csv-btn").disabled = true;

  try {
    const res = await fetch(`${BASE_URL}/train-csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    _applyResults(data);
  } catch (err) {
    alert(`Training failed: ${err.message}`);
  } finally {
    setLoading("csv-train-loading", false);
    document.getElementById("train-csv-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Synthetic training
// ---------------------------------------------------------------------------
function setSyntheticLoading(active) {
  loading.classList.toggle("active", active);
  runBtn.disabled = active;
  runBtn.textContent = active ? "Training..." : "Run Training";
}

async function runTraining() {
  setSyntheticLoading(true);
  try {
    const payload = {
      n_rows:        parseInt(document.getElementById("rows").value, 10),
      n_features:    parseInt(document.getElementById("features").value, 10),
      n_categorical: parseInt(document.getElementById("categories").value, 10),
      seed:          parseInt(document.getElementById("seed").value, 10),
      decision_labels: [
        document.getElementById("label-positive").value,
        document.getElementById("label-negative").value,
      ],
    };
    const res = await fetch(`${BASE_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    _applyResults(await res.json());
  } catch (err) {
    console.error(err);
    alert("Training failed. Check the console or API logs.");
  } finally {
    setSyntheticLoading(false);
  }
}

document.getElementById("run-btn").addEventListener("click", runTraining);

// ---------------------------------------------------------------------------
// Shared result renderer
// ---------------------------------------------------------------------------
function _applyResults(data) {
  currentResults = data;
  renderDataset(data.dataset);
  renderLeaderboard(data.leaderboard);
  renderMetrics(data.results);
  renderRocCurves(data.results);
  const bestModelName = data.leaderboard[0].model;
  const bestModel = data.results.find((item) => item.name === bestModelName);
  if (bestModel) {
    renderFeatureImportance(bestModel);
    renderConfusion(bestModel);
  }
  // Show action buttons
  document.getElementById("explain-btn-row").style.display = "flex";
  // Reset previous session insights
  _lastInsights = null;
  document.getElementById("insights-panel").style.display = "none";
  // Scroll results into view
  document.getElementById("dataset-stats").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Report download
// ---------------------------------------------------------------------------
async function downloadReport(format) {
  if (!currentResults) return;

  const loadingText = document.getElementById("report-loading-text");
  loadingText.textContent = format === "pdf"
    ? "Building PDF report... this takes ~10 seconds."
    : "Building HTML report...";
  setLoading("report-loading", true);

  try {
    const payload = {
      leaderboard: currentResults.leaderboard,
      results: currentResults.results,
      dataset_summary: currentResults.dataset,
      // include AI insights if already fetched
      ai_insights: _lastInsights || null,
    };

    const res = await fetch(`${BASE_URL}/report/${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const blob   = await res.blob();
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement("a");
    const ts     = new Date().toISOString().slice(0, 10);
    a.href       = url;
    a.download   = `ml-report-${ts}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Report generation failed: ${err.message}`);
  } finally {
    setLoading("report-loading", false);
  }
}

// ---------------------------------------------------------------------------
// Explain with AI
// ---------------------------------------------------------------------------
async function explainResults() {
  if (!currentResults) return;

  setLoading("explain-loading", true);

  try {
    const payload = {
      leaderboard: currentResults.leaderboard,
      results: currentResults.results,
      dataset_summary: currentResults.dataset,
    };
    const res = await fetch(`${BASE_URL}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    renderInsights(await res.json());
  } catch (err) {
    alert(`Explain failed: ${err.message}`);
  } finally {
    setLoading("explain-loading", false);
  }
}

function renderInsights(data) {
  _lastInsights = data;  // cache for report embedding
  document.getElementById("insight-summary").textContent    = data.executive_summary    || "";
  document.getElementById("insight-best-model").textContent = data.best_model_analysis  || "";

  const toList = (id, items) => {
    document.getElementById(id).innerHTML = (items || []).map((t) => `<li>${t}</li>`).join("");
  };
  toList("insight-features", data.feature_insights);
  toList("insight-flags",    data.risk_flags);
  toList("insight-recs",     data.recommendations);

  const panel = document.getElementById("insights-panel");
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Chart renderers (unchanged)
// ---------------------------------------------------------------------------
function renderDataset(summary) {
  datasetStats.innerHTML = "";
  const cards = [
    { label: "Total Rows",            value: summary.n_rows },
    { label: "Total Features",        value: summary.n_features },
    { label: "Numeric Features",      value: summary.feature_types.numeric },
    { label: "Categorical Features",  value: summary.feature_types.categorical },
    { label: "Target",                value: summary.target_col },
  ];
  cards.forEach((card) => {
    const div = document.createElement("div");
    div.className = "stat-card";
    div.innerHTML = `<h3>${card.label}</h3><p>${card.value}</p>`;
    datasetStats.appendChild(div);
  });

  classBalance.innerHTML = "";
  Object.entries(summary.class_balance).forEach(([split, ratios]) => {
    const detail = document.createElement("div");
    detail.className = "split-badge";
    const ratioText = Object.entries(ratios)
      .map(([label, ratio]) => `${label}: ${number(ratio * 100, 1)}%`)
      .join(" | ");
    detail.textContent = `${split.toUpperCase()} · ${ratioText}`;
    classBalance.appendChild(detail);
  });
}

function renderLeaderboard(rows) {
  leaderboardTable.innerHTML = "";
  leaderboardTable.insertAdjacentHTML("beforeend",
    `<thead><tr><th>Rank</th><th>Model</th><th>ROC-AUC</th></tr></thead>`);
  const body = document.createElement("tbody");
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${index + 1}</td><td>${row.model}</td><td>${number(row.roc_auc)}</td>`;
    body.appendChild(tr);
  });
  leaderboardTable.appendChild(body);
}

function renderMetrics(results) {
  const models = results.map((item) => item.name);
  const metrics = ["roc_auc", "f1", "precision", "recall"];
  const traces = metrics.map((metric) => ({
    x: models,
    y: results.map((item) => item.metrics.test[metric]),
    name: metric.toUpperCase(),
    type: "bar",
  }));
  Plotly.newPlot("metrics-chart", traces, {
    barmode: "group",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 40, l: 40, r: 20, b: 40 },
    legend: { orientation: "h" },
  });
}

function renderFeatureImportance(bestModel) {
  const data = bestModel.feature_importance || [];
  const trace = {
    x: data.map((item) => item.importance),
    y: data.map((item) => item.feature),
    type: "bar",
    orientation: "h",
    marker: { color: "#ff6f59" },
  };
  Plotly.newPlot("feature-chart", [trace], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 30, l: 120, r: 20, b: 30 },
  });
}

function renderRocCurves(results) {
  const traces = results.map((item) => ({
    x: item.roc_curve.test.fpr,
    y: item.roc_curve.test.tpr,
    mode: "lines",
    name: item.name,
  }));
  Plotly.newPlot("roc-chart", traces, {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: "False Positive Rate" },
    yaxis: { title: "True Positive Rate" },
    margin: { t: 30, l: 50, r: 20, b: 40 },
  });
}

function renderConfusion(bestModel) {
  const matrix = bestModel.confusion.test;
  Plotly.newPlot("confusion-chart", [{
    z: matrix,
    type: "heatmap",
    colorscale: [[0, "#eff6ff"], [0.5, "#7dd3fc"], [1, "#0f172a"]],
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 30, l: 40, r: 20, b: 40 },
  });
}
