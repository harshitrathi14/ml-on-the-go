"""
Report generation: self-contained HTML and PDF.

HTML report — all charts embedded as base64 PNG, no external dependencies.
PDF report  — ReportLab-based, professional A4 layout with embedded charts.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive, safe for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Palette (matches the platform UI)
# ---------------------------------------------------------------------------
_AQUA   = "#3dd6c6"
_CORAL  = "#ff6f59"
_GOLD   = "#ffcb3c"
_DEEP   = "#0a3d62"
_PURPLE = "#6c47ff"
_COLORS = [_AQUA, _CORAL, _GOLD, _DEEP, _PURPLE]
_GRAY   = "#64748b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


# ===========================================================================
# Chart helpers
# ===========================================================================

def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _bytes_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _chart_leaderboard(leaderboard: List[dict]) -> plt.Figure:
    models = [r["model"] for r in leaderboard]
    aucs   = [r["roc_auc"] for r in leaderboard]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(models) * 0.55)))
    bars = ax.barh(models[::-1], aucs[::-1],
                   color=[_COLORS[i % len(_COLORS)] for i in range(len(models))],
                   edgecolor="none", height=0.55)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("ROC-AUC", color=_GRAY, fontsize=9)
    ax.set_title("Model Leaderboard", fontweight="bold", pad=10)
    for bar, auc in zip(bars, aucs[::-1]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{auc:.4f}", va="center", fontsize=9, color=_DEEP)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8, colors=_GRAY)
    fig.tight_layout()
    return fig


def _chart_roc(results: List[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, r in enumerate(results):
        roc = (r.get("roc_curve") or {}).get("test", {})
        fpr, tpr = roc.get("fpr", []), roc.get("tpr", [])
        auc = ((r.get("metrics") or {}).get("test") or {}).get("roc_auc", 0)
        if fpr and tpr:
            ax.plot(fpr, tpr, lw=2, color=_COLORS[i % len(_COLORS)],
                    label=f"{r['name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#cbd5e1", label="Random")
    ax.set_xlabel("False Positive Rate", color=_GRAY, fontsize=9)
    ax.set_ylabel("True Positive Rate", color=_GRAY, fontsize=9)
    ax.set_title("ROC Curves — Test Set", fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig


def _chart_metrics(results: List[dict]) -> plt.Figure:
    metric_keys = ["roc_auc", "f1", "precision", "recall"]
    models = [r["name"] for r in results]
    x = np.arange(len(models))
    w = 0.19
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 4.5))
    for i, mk in enumerate(metric_keys):
        vals = [((r.get("metrics") or {}).get("test") or {}).get(mk, 0) for r in results]
        ax.bar(x + i * w, vals, w, label=mk.upper(), color=_COLORS[i], edgecolor="none")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Metrics — Test Set", fontweight="bold", pad=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    return fig


def _chart_feature_importance(best: dict) -> Optional[plt.Figure]:
    fi = (best.get("feature_importance") or [])[:15]
    if not fi:
        return None
    feats = [f["feature"] for f in fi][::-1]
    imps  = [abs(f["importance"]) for f in fi][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, len(feats) * 0.42)))
    ax.barh(feats, imps, color=_CORAL, edgecolor="none", height=0.65)
    ax.set_xlabel("Importance", color=_GRAY, fontsize=9)
    ax.set_title(f"Top Features — {best['name']}", fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def _chart_stability(results: List[dict]) -> plt.Figure:
    splits = ["train", "test", "oot", "etrc"]
    models = [r["name"] for r in results]
    data = np.array([
        [((r.get("metrics") or {}).get(s) or {}).get("roc_auc", 0) for s in splits]
        for r in results
    ])
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(models) * 0.7)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.upper() for s in splits], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("ROC-AUC Stability Across Splits", fontweight="bold", pad=10)
    for i in range(len(models)):
        for j in range(len(splits)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if data[i, j] > 0.8 else _DEEP)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    return fig


def _make_all_charts(leaderboard, results) -> Dict[str, str]:
    """Returns dict name→base64 PNG for all charts."""
    best = _best_model(leaderboard, results)
    charts: Dict[str, str] = {}
    charts["leaderboard"] = _b64(_chart_leaderboard(leaderboard))
    charts["roc"]         = _b64(_chart_roc(results))
    charts["metrics"]     = _b64(_chart_metrics(results))
    charts["stability"]   = _b64(_chart_stability(results))
    fi_fig = _chart_feature_importance(best)
    if fi_fig:
        charts["features"] = _b64(fi_fig)
    return charts


def _best_model(leaderboard: List[dict], results: List[dict]) -> dict:
    name = leaderboard[0]["model"] if leaderboard else ""
    for r in results:
        if r["name"] == name:
            return r
    return results[0] if results else {}


# ===========================================================================
# HTML Report
# ===========================================================================

def generate_html_report(
    leaderboard: List[dict],
    results: List[dict],
    dataset_summary: Optional[dict] = None,
    ai_insights: Optional[dict] = None,
) -> str:
    charts   = _make_all_charts(leaderboard, results)
    best     = _best_model(leaderboard, results)
    ts       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    n_models = len(leaderboard)
    best_auc = leaderboard[0]["roc_auc"] if leaderboard else 0
    best_name = leaderboard[0]["model"] if leaderboard else "N/A"

    ds = dataset_summary or {}
    ai = ai_insights or {}

    # Build leaderboard rows
    lb_rows = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, row in enumerate(leaderboard):
        medal = medals[i] if i < 3 else f"#{i+1}"
        m = results[i]["metrics"]["test"] if i < len(results) else {}
        f1 = m.get("f1", 0) if isinstance(m, dict) else getattr(m, "f1", 0)
        lb_rows += f"""
        <tr>
          <td>{medal}</td>
          <td><strong>{row['model']}</strong></td>
          <td class="num">{row['roc_auc']:.4f}</td>
          <td class="num">{f1:.4f}</td>
        </tr>"""

    # Build per-model metrics table
    splits = ["train", "test", "oot", "etrc"]
    metric_keys = ["roc_auc", "f1", "precision", "recall"]
    model_sections = ""
    for r in results:
        rows_html = ""
        for mk in metric_keys:
            row_html = f"<tr><td>{mk.replace('_', ' ').upper()}</td>"
            for sp in splits:
                m = (r.get("metrics") or {}).get(sp) or {}
                val = m.get(mk, 0) if isinstance(m, dict) else getattr(m, mk, 0)
                cls = "good" if val >= 0.75 else ("warn" if val >= 0.60 else "bad")
                row_html += f'<td class="num {cls}">{val:.4f}</td>'
            rows_html += row_html + "</tr>"
        model_sections += f"""
        <div class="model-section">
          <h3>{r['name']}</h3>
          <table class="metrics-table">
            <thead><tr><th>Metric</th>{''.join(f'<th>{s.upper()}</th>' for s in splits)}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    # AI insights section
    ai_section = ""
    if ai:
        def _list_items(items):
            return "".join(f"<li>{x}</li>" for x in (items or []))

        ai_section = f"""
        <section class="section">
          <h2>AI Insights</h2>
          <div class="ai-card">
            <h3>Executive Summary</h3>
            <p>{ai.get('executive_summary', '')}</p>
          </div>
          <div class="ai-card">
            <h3>Best Model Analysis</h3>
            <p>{ai.get('best_model_analysis', '')}</p>
          </div>
          <div class="ai-grid">
            <div class="ai-card">
              <h3>Feature Insights</h3>
              <ul>{_list_items(ai.get('feature_insights'))}</ul>
            </div>
            <div class="ai-card">
              <h3>Risk Flags</h3>
              <ul class="flags">{_list_items(ai.get('risk_flags'))}</ul>
            </div>
            <div class="ai-card">
              <h3>Recommendations</h3>
              <ul class="recs">{_list_items(ai.get('recommendations'))}</ul>
            </div>
          </div>
        </section>"""

    # Features chart (optional)
    features_section = ""
    if "features" in charts:
        features_section = f"""
        <section class="section">
          <h2>Top Feature Drivers</h2>
          <div class="chart-wrap"><img src="data:image/png;base64,{charts['features']}" alt="Feature Importance" /></div>
        </section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ML Report — {ts}</title>
<style>
  :root {{
    --aqua:#3dd6c6; --coral:#ff6f59; --gold:#ffcb3c; --deep:#0a3d62;
    --purple:#6c47ff; --ink:#0b0f19; --muted:#64748b; --bg:#f8fafc;
    --card:white; --border:#e2e8f0;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:var(--bg);color:var(--ink);font-size:14px;line-height:1.6}}
  a{{color:var(--deep)}}

  /* Cover */
  .cover{{background:linear-gradient(135deg,var(--deep) 0%,#121826 100%);
          color:white;padding:60px 7vw 48px;position:relative;overflow:hidden}}
  .cover::after{{content:'';position:absolute;width:400px;height:400px;right:-100px;top:-100px;
                background:radial-gradient(circle,rgba(61,214,198,.3),transparent 70%);z-index:0}}
  .cover-inner{{position:relative;z-index:1}}
  .cover-badge{{display:inline-block;padding:4px 14px;background:rgba(255,255,255,.12);
                border-radius:999px;font-size:12px;margin-bottom:20px;letter-spacing:.06em}}
  .cover h1{{font-size:clamp(1.8rem,4vw,3rem);font-weight:700;margin-bottom:12px}}
  .cover p{{color:rgba(255,255,255,.75);max-width:600px;margin-bottom:24px}}
  .cover-meta{{display:flex;flex-wrap:wrap;gap:24px}}
  .cover-stat{{background:rgba(255,255,255,.08);border-radius:12px;padding:16px 24px;min-width:140px}}
  .cover-stat .val{{font-size:1.8rem;font-weight:700;color:var(--aqua)}}
  .cover-stat .lbl{{font-size:12px;color:rgba(255,255,255,.6);margin-top:2px}}

  /* Layout */
  main{{max-width:960px;margin:0 auto;padding:40px 5vw 80px}}
  .section{{margin-bottom:48px}}
  .section h2{{font-size:1.2rem;font-weight:700;color:var(--deep);
               border-left:4px solid var(--aqua);padding-left:12px;margin-bottom:20px}}

  /* Leaderboard */
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:10px 12px;background:#f1f5f9;color:var(--muted);
      font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
  td{{padding:10px 12px;border-bottom:1px solid var(--border)}}
  tr:last-child td{{border-bottom:none}}
  .num{{font-variant-numeric:tabular-nums;font-weight:600}}
  .good{{color:#16a34a}} .warn{{color:#d97706}} .bad{{color:#dc2626}}

  /* Per-model metrics */
  .model-section{{background:var(--card);border-radius:16px;padding:20px;
                  border:1px solid var(--border);margin-bottom:20px}}
  .model-section h3{{font-size:1rem;font-weight:700;color:var(--deep);margin-bottom:12px}}
  .metrics-table th{{background:#f8fafc}}

  /* Charts */
  .chart-wrap{{background:var(--card);border-radius:16px;padding:16px;
               border:1px solid var(--border);text-align:center}}
  .chart-wrap img{{max-width:100%;height:auto;border-radius:8px}}
  .chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}}

  /* AI insights */
  .ai-card{{background:var(--card);border-radius:14px;padding:18px;
            border:1px solid var(--border);margin-bottom:14px}}
  .ai-card h3{{font-size:.9rem;font-weight:700;color:var(--deep);margin-bottom:8px}}
  .ai-card p{{color:#374151;line-height:1.7}}
  .ai-card ul{{padding-left:18px;color:#374151}}
  .ai-card ul.flags li{{color:#92400e}}
  .ai-card ul.recs  li{{color:#1e40af}}
  .ai-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}

  /* Print button */
  .print-bar{{position:fixed;bottom:24px;right:24px;z-index:999;display:flex;gap:10px}}
  .print-btn{{background:var(--deep);color:white;border:none;padding:12px 20px;
              border-radius:999px;font-size:14px;font-weight:600;cursor:pointer;
              box-shadow:0 8px 24px rgba(10,61,98,.35);display:flex;align-items:center;gap:8px}}
  .print-btn:hover{{background:#0c4d7a}}
  .print-btn.secondary{{background:var(--purple)}}

  /* Print styles */
  @media print{{
    .print-bar{{display:none!important}}
    body{{background:white;font-size:12px}}
    .cover{{background:var(--deep)!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
    .section{{page-break-inside:avoid}}
    main{{padding:20px}}
  }}
</style>
</head>
<body>

<!-- Cover -->
<div class="cover">
  <div class="cover-inner">
    <div class="cover-badge">ML on the Go · Model Report</div>
    <h1>Training Results</h1>
    <p>Generated {ts} · {n_models} models trained and evaluated across Train / Test / OOT / ETRC splits</p>
    <div class="cover-meta">
      <div class="cover-stat">
        <div class="val">{best_auc:.4f}</div>
        <div class="lbl">Best ROC-AUC</div>
      </div>
      <div class="cover-stat">
        <div class="val">{best_name}</div>
        <div class="lbl">Best Model</div>
      </div>
      <div class="cover-stat">
        <div class="val">{ds.get('n_rows', '–'):,}</div>
        <div class="lbl">Total Rows</div>
      </div>
      <div class="cover-stat">
        <div class="val">{ds.get('n_features', '–')}</div>
        <div class="lbl">Features</div>
      </div>
    </div>
  </div>
</div>

<main>

{ai_section}

<!-- Leaderboard -->
<section class="section">
  <h2>Model Leaderboard</h2>
  <div class="model-section" style="padding:0;overflow:hidden">
    <table>
      <thead><tr><th>Rank</th><th>Model</th><th>ROC-AUC</th><th>F1</th></tr></thead>
      <tbody>{lb_rows}</tbody>
    </table>
  </div>
  <div class="chart-wrap" style="margin-top:16px">
    <img src="data:image/png;base64,{charts['leaderboard']}" alt="Leaderboard chart"/>
  </div>
</section>

<!-- Per-model metrics -->
<section class="section">
  <h2>Model Metrics by Split</h2>
  {model_sections}
</section>

<!-- Charts grid -->
<section class="section">
  <h2>ROC Curves &amp; Metrics</h2>
  <div class="chart-grid">
    <div class="chart-wrap"><img src="data:image/png;base64,{charts['roc']}" alt="ROC Curves"/></div>
    <div class="chart-wrap"><img src="data:image/png;base64,{charts['metrics']}" alt="Metrics"/></div>
  </div>
</section>

{features_section}

<!-- Stability -->
<section class="section">
  <h2>Stability Heatmap</h2>
  <div class="chart-wrap">
    <img src="data:image/png;base64,{charts['stability']}" alt="Stability Heatmap"/>
  </div>
</section>

</main>

<!-- Floating buttons -->
<div class="print-bar">
  <button class="print-btn secondary" onclick="downloadHTML()">⬇ Save HTML</button>
  <button class="print-btn" onclick="window.print()">🖨 Print / PDF</button>
</div>

<script>
function downloadHTML(){{
  const blob=new Blob([document.documentElement.outerHTML],{{type:'text/html'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='ml-report-{datetime.utcnow().strftime("%Y%m%d-%H%M")}.html';
  a.click();
}}
</script>
</body>
</html>
"""


# ===========================================================================
# PDF Report
# ===========================================================================

def generate_pdf_report(
    leaderboard: List[dict],
    results: List[dict],
    dataset_summary: Optional[dict] = None,
    ai_insights: Optional[dict] = None,
) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable, KeepTogether,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        title="ML on the Go — Training Report",
        author="ML on the Go",
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4.4 * cm  # usable width

    # Custom styles
    S = {
        "title": ParagraphStyle("title", parent=styles["Title"],
                                fontSize=26, textColor=_rl(_DEEP), spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", parent=styles["Normal"],
                                   fontSize=12, textColor=_rl(_GRAY), spaceAfter=20),
        "h2": ParagraphStyle("h2", parent=styles["Heading2"],
                              fontSize=14, textColor=_rl(_DEEP),
                              spaceBefore=18, spaceAfter=8,
                              borderPad=0, leftIndent=0,
                              borderLeftColor=_rl(_AQUA), borderLeftWidth=3,
                              borderLeftPadding=8),
        "h3": ParagraphStyle("h3", parent=styles["Heading3"],
                              fontSize=11, textColor=_rl(_DEEP),
                              spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("body", parent=styles["Normal"],
                               fontSize=10, textColor=_rl("#374151"), leading=16),
        "note": ParagraphStyle("note", parent=styles["Normal"],
                               fontSize=9, textColor=_rl(_GRAY),
                               leftIndent=12, spaceBefore=4),
    }

    # --------------- table helpers ---------------
    _TH_BG  = _rl("#f1f5f9")
    _TH_TXT = _rl(_GRAY)
    _LINE   = _rl(_DEEP)
    _GOOD   = _rl("#16a34a")
    _WARN   = _rl("#d97706")
    _BAD    = _rl("#dc2626")

    def _base_table_style():
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), _TH_BG),
            ("TEXTCOLOR",    (0, 0), (-1, 0), _TH_TXT),
            ("FONTSIZE",     (0, 0), (-1, 0), 8),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_rl("white"), _rl("#f8fafc")]),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.4, _rl("#e2e8f0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ])

    def _img(fig: plt.Figure, w_cm: float = 14.0) -> RLImage:
        # Read size BEFORE saving (savefig closes the figure)
        fig_w, fig_h = fig.get_size_inches()
        aspect = fig_h / fig_w
        data = _bytes_png(fig)
        target_w = w_cm * cm
        target_h = target_w * aspect
        img_buf = io.BytesIO(data)
        return RLImage(img_buf, width=target_w, height=target_h)

    # --------------- build story ---------------
    best     = _best_model(leaderboard, results)
    ts       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ds       = dataset_summary or {}
    ai       = ai_insights or {}
    best_auc = leaderboard[0]["roc_auc"] if leaderboard else 0
    best_name = leaderboard[0]["model"] if leaderboard else "N/A"

    story: list = []

    # --- Cover ---
    story += [
        Spacer(1, 1 * cm),
        Paragraph("ML on the Go", ParagraphStyle("brand", parent=styles["Normal"],
            fontSize=11, textColor=_rl(_AQUA), spaceAfter=8)),
        Paragraph("Training Report", S["title"]),
        Paragraph(f"Generated {ts}", S["subtitle"]),
        HRFlowable(width="100%", thickness=1, color=_rl(_AQUA), spaceAfter=20),
    ]

    # Summary stats table
    cover_data = [
        ["Best ROC-AUC", "Best Model", "Total Rows", "Features"],
        [f"{best_auc:.4f}", best_name,
         f"{ds.get('n_rows', '–'):,}" if ds.get("n_rows") else "–",
         str(ds.get("n_features", "–"))],
    ]
    ct = Table(cover_data, colWidths=[W / 4] * 4)
    ct.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _rl(_DEEP)),
        ("TEXTCOLOR",   (0, 0), (-1, 0), _rl("white")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, 1), 16),
        ("TEXTCOLOR",   (0, 1), (-1, 1), _rl(_DEEP)),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, _rl("#e2e8f0")),
        ("BACKGROUND",  (0, 1), (-1, 1), _rl("#f0fdf4")),
    ]))
    story += [ct, Spacer(1, 0.5 * cm)]

    # --- AI insights ---
    if ai:
        story.append(Paragraph("AI Insights", S["h2"]))
        for key, label in [
            ("executive_summary",  "Executive Summary"),
            ("best_model_analysis","Best Model Analysis"),
        ]:
            if ai.get(key):
                story += [
                    Paragraph(label, S["h3"]),
                    Paragraph(ai[key], S["body"]),
                ]
        for key, label in [
            ("feature_insights", "Feature Insights"),
            ("risk_flags",       "Risk Flags"),
            ("recommendations",  "Recommendations"),
        ]:
            items = ai.get(key) or []
            if items:
                story.append(Paragraph(label, S["h3"]))
                for item in items:
                    story.append(Paragraph(f"• {item}", S["note"]))

    # --- Leaderboard ---
    story.append(Paragraph("Model Leaderboard", S["h2"]))
    medals = ["🥇", "🥈", "🥉"]
    lb_data = [["Rank", "Model", "ROC-AUC (Test)", "F1 (Test)"]]
    for i, row in enumerate(leaderboard):
        rank = medals[i] if i < 3 else f"#{i+1}"
        m = (results[i].get("metrics") or {}).get("test") if i < len(results) else {}
        f1 = (m.get("f1", 0) if isinstance(m, dict) else getattr(m, "f1", 0)) if m else 0
        lb_data.append([rank, row["model"], f"{row['roc_auc']:.4f}", f"{f1:.4f}"])
    lb_table = Table(lb_data, colWidths=[W * .12, W * .40, W * .24, W * .24])
    lb_table.setStyle(_base_table_style())
    story += [lb_table, Spacer(1, 0.3 * cm)]

    # Leaderboard chart
    story.append(_img(_chart_leaderboard(leaderboard), 14))

    # --- Per-model metrics ---
    story.append(Paragraph("Model Metrics by Split", S["h2"]))
    splits = ["train", "test", "oot", "etrc"]
    metric_keys = ["ROC-AUC", "F1", "Precision", "Recall"]
    mk_map = {"ROC-AUC": "roc_auc", "F1": "f1", "Precision": "precision", "Recall": "recall"}

    for r in results:
        story.append(Paragraph(r["name"], S["h3"]))
        mdata = [["Metric"] + [s.upper() for s in splits]]
        for mk_label, mk in mk_map.items():
            row_d = [mk_label]
            for sp in splits:
                m = (r.get("metrics") or {}).get(sp) or {}
                val = m.get(mk, 0) if isinstance(m, dict) else getattr(m, mk, 0)
                row_d.append(f"{val:.4f}")
            mdata.append(row_d)
        mt = Table(mdata, colWidths=[W * .26] + [W * .185] * 4)
        mt.setStyle(_base_table_style())
        story.append(mt)

    # --- ROC curves + metrics charts ---
    story += [
        PageBreak(),
        Paragraph("ROC Curves", S["h2"]),
        _img(_chart_roc(results), 14),
        Spacer(1, 0.5 * cm),
        Paragraph("Metrics Comparison", S["h2"]),
        _img(_chart_metrics(results), 16),
    ]

    # --- Feature importance ---
    fi_fig = _chart_feature_importance(best)
    if fi_fig:
        story += [
            Spacer(1, 0.3 * cm),
            Paragraph("Top Feature Drivers", S["h2"]),
            _img(fi_fig, 14),
        ]

    # --- Stability ---
    story += [
        Spacer(1, 0.3 * cm),
        Paragraph("Stability Heatmap", S["h2"]),
        _img(_chart_stability(results), 14),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", thickness=0.5, color=_rl("#e2e8f0"), spaceAfter=8),
        Paragraph(f"Report generated by ML on the Go · {ts}", S["note"]),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rl(hex_or_name: str):
    """Convert hex colour string or named colour to ReportLab Color."""
    from reportlab.lib import colors as rl_colors
    # Named colours supported by ReportLab
    _named = {"white": "#ffffff", "black": "#000000"}
    s = _named.get(hex_or_name.lower(), hex_or_name)
    h = s.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return rl_colors.Color(r / 255, g / 255, b / 255)
