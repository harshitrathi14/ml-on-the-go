const API_URL = "http://localhost:8000/train";

const runBtn = document.getElementById("run-btn");
const loading = document.getElementById("loading");
const datasetStats = document.getElementById("dataset-stats");
const classBalance = document.getElementById("class-balance");
const leaderboardTable = document.getElementById("leaderboard");

function setLoading(active) {
  loading.classList.toggle("active", active);
  runBtn.disabled = active;
  runBtn.textContent = active ? "Training..." : "Run Training";
}

function number(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function renderDataset(summary) {
  datasetStats.innerHTML = "";
  const cards = [
    { label: "Total Rows", value: summary.n_rows },
    { label: "Total Features", value: summary.n_features },
    { label: "Numeric Features", value: summary.feature_types.numeric },
    { label: "Categorical Features", value: summary.feature_types.categorical },
    { label: "Target", value: summary.target_col },
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
  leaderboardTable.insertAdjacentHTML(
    "beforeend",
    `<thead><tr><th>Rank</th><th>Model</th><th>ROC-AUC</th></tr></thead>`
  );
  const body = document.createElement("tbody");
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${index + 1}</td><td>${row.model}</td><td>${number(
      row.roc_auc
    )}</td>`;
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
  Plotly.newPlot(
    "confusion-chart",
    [
      {
        z: matrix,
        type: "heatmap",
        colorscale: [
          [0, "#eff6ff"],
          [0.5, "#7dd3fc"],
          [1, "#0f172a"],
        ],
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { t: 30, l: 40, r: 20, b: 40 },
    }
  );
}

async function runTraining() {
  setLoading(true);
  try {
    const payload = {
      n_rows: parseInt(document.getElementById("rows").value, 10),
      n_features: parseInt(document.getElementById("features").value, 10),
      n_categorical: parseInt(document.getElementById("categories").value, 10),
      seed: parseInt(document.getElementById("seed").value, 10),
      decision_labels: [
        document.getElementById("label-positive").value,
        document.getElementById("label-negative").value,
      ],
    };

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
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
  } catch (error) {
    console.error(error);
    alert("Training failed. Check the console or API logs.");
  } finally {
    setLoading(false);
  }
}

runBtn.addEventListener("click", runTraining);
