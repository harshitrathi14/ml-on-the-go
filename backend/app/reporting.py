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
# Plain-English feature interpretation (no AI required)
# ===========================================================================

# What the target variable means for each known target column
_TARGET_CONTEXT: Dict[str, Dict] = {
    "high_value_transaction": {
        "goal":    "identify which sales transactions are high-value",
        "positive": "the transaction is above the revenue threshold — a big sale",
        "negative": "the transaction is a smaller / routine purchase",
        "impact":  "Knowing this in advance helps focus sales staff attention, allocate promotions, and understand what drives your biggest revenue moments.",
        "use_case":"e.g., flag a customer at checkout who is likely to make a large purchase, so you can offer upsell at the right moment.",
    },
    "stockout_risk": {
        "goal":    "predict which products are at risk of running out of stock before the next restock",
        "positive": "the SKU is likely to sell out — a stockout event",
        "negative": "stock levels are expected to be safe",
        "impact":  "Catching stockout risk early lets you reorder faster, avoiding lost sales and disappointed customers.",
        "use_case":"e.g., automatically trigger a purchase order for any SKU flagged as high-risk before it hits zero.",
    },
    "churned": {
        "goal":    "predict which customers are likely to cancel or stop engaging",
        "positive": "the customer has churned (left)",
        "negative": "the customer is still active",
        "impact":  "Identifying at-risk customers before they leave gives you time to intervene — a retention offer, a support call, or a personalised promotion.",
        "use_case":"e.g., send a 'We miss you' discount to any customer whose churn probability exceeds 60%.",
    },
}

# Pattern → plain-English description template
# Each entry: (pattern_in_name, short_label, high_means, low_means, business_note)
_FEAT_PATTERNS = [
    ("days_since_last",   "Recency",          "customer hasn't bought recently",  "customer bought very recently",     "Recency is a classic churn/engagement signal — the longer since the last interaction, the more at-risk."),
    ("days_since",        "Days Since",       "more time has passed",             "less time has passed",              "Time-since features capture how fresh or stale a relationship/event is."),
    ("tenure",            "Tenure",           "long-standing customer/supplier",  "new customer/supplier",             "Longer tenure usually signals loyalty and stability; short tenure can mean higher uncertainty."),
    ("avg_transaction",   "Avg Transaction",  "higher average spend per visit",   "lower average spend per visit",     "The average transaction value is a core indicator of customer quality and purchasing power."),
    ("total_spend",       "Total Spend",      "high lifetime spending",           "low lifetime spending",             "Cumulative spend is one of the strongest predictors across most business models."),
    ("spend",             "Spending",         "higher spending activity",         "lower spending activity",           "Spend-related features are among the most predictive in retail and subscription contexts."),
    ("transaction_count", "Transaction Count","buys frequently",                  "rarely transacts",                  "Frequency of purchase reflects engagement and stickiness."),
    ("num_transactions",  "Transaction Count","buys frequently",                  "rarely transacts",                  "Frequency of purchase reflects engagement and stickiness."),
    ("count",             "Activity Count",   "high frequency of activity",       "low frequency of activity",         "How often an event occurs is a strong signal of engagement or risk."),
    ("avg_order",         "Avg Order Size",   "large orders",                     "small orders",                      "Order size relative to baseline predicts basket value and purchasing intent."),
    ("discount",          "Discount Usage",   "heavy discount-seeker",            "buys at full price",                "Customers who always buy on discount can be more price-sensitive and churn when discounts stop."),
    ("return_rate",       "Return Rate",      "frequently returns items",         "rarely returns items",              "High return rates raise fulfilment costs and may signal dissatisfaction."),
    ("support_tickets",   "Support Contacts", "frequently raises issues",         "rarely contacts support",           "Support volume is linked to friction — customers with many tickets are more at risk of churning."),
    ("contract",          "Contract Type",    "longer / more committed contract", "short / month-to-month",            "Contract length is one of the strongest churn signals — month-to-month customers leave most easily."),
    ("monthly_charge",    "Monthly Charge",   "higher monthly bill",              "lower monthly bill",                "Price is a top churn driver — customers with high charges scrutinise value more closely."),
    ("num_products",      "Product Count",    "uses many products/services",      "uses only one product",             "Cross-sell depth (# of products) is a strong loyalty indicator — more products = stickier relationship."),
    ("reorder_point",     "Reorder Point",    "item should reorder sooner",       "item can wait longer",              "Reorder point reflects how quickly an SKU typically sells through relative to lead time."),
    ("lead_time",         "Lead Time",        "slow supplier / long delivery",    "fast supplier / quick delivery",    "Longer supplier lead time increases exposure to stockout — less buffer time to react."),
    ("stock_on_hand",     "Stock on Hand",    "well stocked",                     "low inventory level",               "Current stock level is the most direct signal for stockout risk."),
    ("safety_stock",      "Safety Stock",     "high safety buffer",               "minimal safety buffer",             "Safety stock is the cushion against demand spikes — running thin on it is risky."),
    ("velocity",          "Sales Velocity",   "fast-moving SKU",                  "slow-moving SKU",                   "High-velocity items are most exposed to stockout because they sell out quickly."),
    ("demand",            "Demand",           "high demand",                      "low demand",                        "Demand magnitude directly drives how quickly stock depletes."),
    ("price",             "Price",            "premium-priced item",              "budget-priced item",                "Price affects both demand elasticity and margin exposure."),
    ("quantity",          "Quantity",         "high volume",                      "low volume",                        "Volume features capture scale — how much is being moved or at stake."),
    ("payment_method",    "Payment Method",   "—",                                "—",                                 "Payment method can signal customer intent and risk tolerance."),
    ("age",               "Age",              "older customer / item",            "newer customer / item",             "Age can reflect loyalty (customers) or obsolescence risk (products)."),
    ("hour",              "Time of Day",      "later in the day",                 "earlier in the day",                "Time-of-day captures intraday demand patterns and behavioural rhythms."),
    ("day_of_week",       "Day of Week",      "—",                                "—",                                 "Weekday vs weekend patterns drive demand in almost every retail context."),
    ("month",             "Month",            "later in the year",                "earlier in the year",               "Seasonality (month) captures annual demand cycles."),
    ("is_weekend",        "Weekend Flag",     "happened on a weekend",            "happened on a weekday",             "Weekend transactions often differ in size, category mix, and customer segment."),
    ("is_holiday",        "Holiday Flag",     "occurred during a holiday",        "occurred outside holidays",         "Holiday periods drive demand spikes and can create stockout risk."),
    ("category",          "Product Category", "—",                                "—",                                 "Product category captures systematic differences in demand, margin, and customer behaviour."),
    ("region",            "Region",           "—",                                "—",                                 "Regional variation reflects local market conditions, competition, and demographics."),
    ("channel",           "Sales Channel",    "—",                                "—",                                 "Online vs in-store vs wholesale channels have very different risk and value profiles."),
    ("ratio",             "Ratio",            "higher relative proportion",       "lower relative proportion",         "Ratio features normalise for scale, making comparisons more meaningful."),
    ("score",             "Score",            "higher score",                     "lower score",                       "Scores condense complex signals into a single ranked value."),
    ("pct",               "Percentage",       "higher percentage",                "lower percentage",                  "Percentage features show relative importance independent of absolute scale."),
]


def _clean_feat_name(raw: str) -> str:
    """Strip sklearn ColumnTransformer prefixes."""
    for pfx in ("num__", "cat__", "remainder__"):
        if raw.startswith(pfx):
            return raw[len(pfx):]
    return raw


def _humanize_name(name: str) -> str:
    """snake_case → Title Case readable label."""
    import re
    name = re.sub(r"_+\d+$", "", name)   # strip OHE numeric suffix
    return name.replace("_", " ").strip().title()


def _match_pattern(clean_name: str):
    """Return the first matching _FEAT_PATTERNS entry, or None."""
    lower = clean_name.lower()
    for (pattern, short, high, low, note) in _FEAT_PATTERNS:
        if pattern in lower:
            return short, high, low, note
    return None


def _feat_description(feat_name: str, importance: float,
                      target_col: str, model_name: str) -> Dict[str, str]:
    """
    Return a dict with human-readable keys describing one feature's
    role, direction, and business significance — no AI required.
    """
    clean  = _clean_feat_name(feat_name)
    label  = _humanize_name(clean)
    target = target_col.replace("_", " ").title()
    match  = _match_pattern(clean)

    if match:
        short, high_means, low_means, note = match
        # Determine direction for models with signed importance
        if model_name == "LogisticRegression":
            if importance > 0:
                direction = f"Higher values ({high_means}) → <strong>more likely</strong> to be {target}"
            else:
                direction = f"Higher values ({high_means}) → <strong>less likely</strong> to be {target}"
        else:
            direction = f"A major driver: {high_means} is associated with a different outcome for {target}"
        return {"label": label, "short": short, "direction": direction, "note": note}

    # Fallback: generic
    return {
        "label": label,
        "short": "Feature",
        "direction": f"One of the strongest predictors of {target} in this dataset.",
        "note": "This variable captures a pattern the model found highly discriminating between the two outcome classes.",
    }


def _target_context_html(target_col: str, positive_class: str,
                         dataset_label: str, class_balance: dict,
                         n_rows: int) -> str:
    """Render the 'What This Model Predicts' block with target context."""
    ctx = _TARGET_CONTEXT.get(target_col, {
        "goal":     f"predict {target_col}",
        "positive": f"the outcome is '{positive_class}'",
        "negative": f"the outcome is not '{positive_class}'",
        "impact":   "Understanding what drives this outcome helps make faster, more accurate decisions.",
        "use_case": "",
    })
    pos_pct = class_balance.get(1, class_balance.get("1", 0))
    if isinstance(pos_pct, float) and pos_pct <= 1:
        pos_pct = round(pos_pct * 100, 1)
    neg_pct = round(100 - float(pos_pct), 1)

    use_case_html = f'<p class="uc-note">💡 <em>{ctx["use_case"]}</em></p>' if ctx.get("use_case") else ""

    return f"""
    <div class="target-box">
      <div class="target-header">
        <span class="target-pill">Target: <code>{target_col}</code></span>
        <span class="target-pill pos">Positive class: <code>{positive_class}</code></span>
      </div>
      <h3 class="target-goal">Goal: {ctx['goal'].capitalize()}</h3>
      <div class="target-classes">
        <div class="tc positive">
          <div class="tc-pct">{pos_pct}%</div>
          <div class="tc-label">Positive cases</div>
          <div class="tc-desc">{ctx['positive']}</div>
        </div>
        <div class="tc negative">
          <div class="tc-pct">{neg_pct}%</div>
          <div class="tc-label">Negative cases</div>
          <div class="tc-desc">{ctx['negative']}</div>
        </div>
      </div>
      <p class="target-impact"><strong>Why it matters:</strong> {ctx['impact']}</p>
      {use_case_html}
    </div>"""


def _html_key_drivers(best_result: dict, target_col: str,
                      positive_class: str) -> str:
    """Render a detailed Key Drivers section with business explanations."""
    fi = (best_result.get("feature_importance") or [])[:15]
    if not fi:
        return ""
    model_name = best_result.get("name", "")
    max_imp = max(abs(f["importance"]) for f in fi) or 1

    rows = ""
    for rank, f in enumerate(fi, 1):
        imp = f["importance"]
        abs_imp = abs(imp)
        pct_bar = round(abs_imp / max_imp * 100, 1)
        desc = _feat_description(f["feature"], imp, target_col, model_name)
        dir_sign = "pos-dir" if imp >= 0 else "neg-dir"
        rows += f"""
        <div class="driver-row">
          <div class="driver-rank">#{rank}</div>
          <div class="driver-main">
            <div class="driver-name">{desc['label']}
              <span class="driver-short">{desc['short']}</span>
            </div>
            <div class="driver-bar-wrap">
              <div class="driver-bar {dir_sign}" style="width:{pct_bar}%"></div>
            </div>
            <div class="driver-direction {dir_sign}">{desc['direction']}</div>
            <div class="driver-note">{desc['note']}</div>
          </div>
          <div class="driver-imp">{abs_imp:.4f}</div>
        </div>"""

    return f"""
    <div class="drivers-wrap">
      <p class="section-note">Ranked by importance from the best model
      (<strong>{model_name}</strong>). The importance score shows how
      much each variable contributed to the model's decisions.
      {"Direction arrows show whether higher values push towards or away from the positive outcome." if model_name == "LogisticRegression" else ""}
      </p>
      {rows}
    </div>"""


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

def _html_feature_table(feature_meta: List[dict]) -> str:
    """Render feature metadata as an HTML table for layman readers."""
    if not feature_meta:
        return ""
    rows = ""
    for f in feature_meta:
        col  = f.get("column", "")
        typ  = f.get("type", "")
        null = f.get("null_pct", 0)
        null_color = "warn" if null > 15 else ("bad" if null > 30 else "")
        null_cell = f'<td class="num {null_color}">{null}%</td>'
        if typ == "numeric":
            detail = (f'min {f["min"]:,g} / avg {f["mean"]:,g} / max {f["max"]:,g}'
                      if f.get("mean") is not None else "—")
        else:
            top = f.get("top_val", "—")
            detail = f'top value: <em>{top}</em> · {f.get("unique", "?")} categories'
        rows += f"<tr><td>{col}</td><td><span class='type-pill {typ}'>{typ}</span></td>{null_cell}<td>{detail}</td></tr>"
    return f"""
    <table class="feat-table">
      <thead><tr><th>Variable Name</th><th>Type</th><th>Missing %</th><th>Summary</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


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

    # Pull enriched metadata passed from run_and_report.py
    report_meta   = ds.get("report_meta") or {}
    feature_meta  = ds.get("feature_meta") or []
    dropped_cols  = ds.get("dropped_cols") or {}
    dataset_label = report_meta.get("dataset_label") or ds.get("name") or "ML Dataset"
    target_col    = report_meta.get("target_col") or "target"
    positive_cls  = report_meta.get("positive_class") or "1"
    class_balance = report_meta.get("class_balance") or {}

    # Numeric / categorical counts
    num_features  = sum(1 for f in feature_meta if f.get("type") == "numeric")
    cat_features  = sum(1 for f in feature_meta if f.get("type") == "categorical")
    pos_pct = class_balance.get(1, class_balance.get("1", 0))
    if isinstance(pos_pct, float) and pos_pct <= 1:
        pos_pct = round(pos_pct * 100, 1)

    # ── What Does This Model Predict? ──────────────────────────────────────
    target_ctx_html = _target_context_html(
        target_col, str(positive_cls), dataset_label, class_balance,
        ds.get("n_rows", 0)
    )
    intro_section = f"""
    <section class="section">
      <h2>What Does This Model Predict?</h2>
      {target_ctx_html}
      <div class="stat-row" style="margin-top:16px">
        <div class="mini-stat"><span class="val">{ds.get('n_rows', '–'):,}</span><span class="lbl">Total rows</span></div>
        <div class="mini-stat"><span class="val">{len(feature_meta)}</span><span class="lbl">Variables used</span></div>
        <div class="mini-stat"><span class="val">{num_features}</span><span class="lbl">Numeric variables</span></div>
        <div class="mini-stat"><span class="val">{cat_features}</span><span class="lbl">Categorical variables</span></div>
        <div class="mini-stat"><span class="val">{pos_pct}%</span><span class="lbl">Positive cases</span></div>
        <div class="mini-stat"><span class="val">{len(dropped_cols)}</span><span class="lbl">Columns removed</span></div>
      </div>
    </section>"""

    # ── Variables Used ──────────────────────────────────────────────────────
    feat_table_html = _html_feature_table(feature_meta)
    variables_section = f"""
    <section class="section">
      <h2>Variables Used by the Model ({len(feature_meta)} total)</h2>
      <p class="section-note">These are the inputs the model learned from. Each row shows the variable name,
      its data type, how much data is missing, and a quick summary of its values.</p>
      {feat_table_html}
    </section>""" if feature_meta else ""

    # ── Dropped Variables ───────────────────────────────────────────────────
    dropped_section = ""
    if dropped_cols:
        drop_rows = "".join(
            f"<tr><td><code>{col}</code></td><td>{reason}</td></tr>"
            for col, reason in dropped_cols.items()
        )
        dropped_section = f"""
    <section class="section">
      <h2>Variables Removed Before Training ({len(dropped_cols)})</h2>
      <p class="section-note">These columns were <strong>automatically excluded</strong> because they are
      not useful for predicting the outcome (e.g., timestamps, IDs, constant values).</p>
      <div class="model-section" style="padding:0;overflow:hidden">
        <table><thead><tr><th>Column</th><th>Why Removed</th></tr></thead>
        <tbody>{drop_rows}</tbody></table>
      </div>
    </section>"""

    # ── Metric Glossary ─────────────────────────────────────────────────────
    glossary_section = """
    <section class="section">
      <h2>How to Read These Numbers</h2>
      <div class="glossary-grid">
        <div class="gloss-card">
          <div class="gloss-title">ROC-AUC</div>
          <div class="gloss-body">The model's overall ability to distinguish between positive and negative cases.
          <strong>1.0 = perfect, 0.5 = random guessing.</strong> Above 0.75 is generally considered good;
          above 0.85 is very strong.</div>
        </div>
        <div class="gloss-card">
          <div class="gloss-title">Precision</div>
          <div class="gloss-body">Of all the records the model flagged as positive, what fraction actually were?
          High precision = <strong>few false alarms</strong>. Useful when acting on a flag is costly.</div>
        </div>
        <div class="gloss-card">
          <div class="gloss-title">Recall</div>
          <div class="gloss-body">Of all the true positives, what fraction did the model catch?
          High recall = <strong>few missed cases</strong>. Useful when missing a positive outcome is costly.</div>
        </div>
        <div class="gloss-card">
          <div class="gloss-title">F1 Score</div>
          <div class="gloss-body">A balanced average of Precision and Recall — the
          <strong>harmonic mean</strong>. Helpful when you need a single score that weighs both.
          Closer to 1.0 is better.</div>
        </div>
        <div class="gloss-card">
          <div class="gloss-title">Train / Test / OOT / ETRC</div>
          <div class="gloss-body">The data is split into four parts:
          <strong>Train</strong> (model learns here),
          <strong>Test</strong> (first validation),
          <strong>OOT</strong> (out-of-time — recent data the model never saw),
          <strong>ETRC</strong> (stress-test scenario). A good model scores well on all four.</div>
        </div>
        <div class="gloss-card">
          <div class="gloss-title">Feature Importance</div>
          <div class="gloss-body">Shows which variables had the biggest influence on the model's decisions.
          Higher bars = more influence. Use this to understand <strong>why the model makes its predictions</strong>.</div>
        </div>
      </div>
    </section>"""

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

    # Key drivers section (detailed, no AI needed)
    key_drivers_html = _html_key_drivers(best, target_col, str(positive_cls))
    features_section = f"""
    <section class="section">
      <h2>Key Drivers — What Influences the Prediction</h2>
      {key_drivers_html}
      {"<div class='chart-wrap' style='margin-top:20px'><img src='data:image/png;base64," + charts["features"] + "' alt='Feature Importance'/></div>" if "features" in charts else ""}
    </section>""" if key_drivers_html or "features" in charts else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{dataset_label} — ML Report</title>
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
  code{{background:#f1f5f9;padding:1px 6px;border-radius:4px;font-size:12px}}

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
  main{{max-width:1000px;margin:0 auto;padding:40px 5vw 80px}}
  .section{{margin-bottom:48px}}
  .section h2{{font-size:1.2rem;font-weight:700;color:var(--deep);
               border-left:4px solid var(--aqua);padding-left:12px;margin-bottom:16px}}
  .section-note{{color:var(--muted);font-size:13px;margin-bottom:14px}}

  /* Info card */
  .info-card{{background:var(--card);border-radius:16px;padding:20px 24px;
              border:1px solid var(--border)}}
  .info-card p{{color:#374151;line-height:1.7;margin-bottom:8px}}

  /* Mini stats row */
  .stat-row{{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}}
  .mini-stat{{background:#f1f5f9;border-radius:10px;padding:10px 16px;min-width:110px}}
  .mini-stat .val{{font-size:1.3rem;font-weight:700;color:var(--deep);display:block}}
  .mini-stat .lbl{{font-size:11px;color:var(--muted)}}

  /* Tables */
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:10px 12px;background:#f1f5f9;color:var(--muted);
      font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
  td{{padding:10px 12px;border-bottom:1px solid var(--border)}}
  tr:last-child td{{border-bottom:none}}
  .num{{font-variant-numeric:tabular-nums;font-weight:600}}
  .good{{color:#16a34a}} .warn{{color:#d97706}} .bad{{color:#dc2626}}

  /* Feature table */
  .feat-table{{background:var(--card);border-radius:12px;border:1px solid var(--border);overflow:hidden}}
  .feat-table tbody tr:nth-child(even){{background:#fafafa}}
  .feat-table td:first-child{{font-weight:600;color:var(--deep)}}
  .type-pill{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}}
  .type-pill.numeric{{background:#dbeafe;color:#1e40af}}
  .type-pill.categorical{{background:#fef9c3;color:#854d0e}}

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

  /* Glossary */
  .glossary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
  .gloss-card{{background:var(--card);border-radius:14px;padding:16px 18px;border:1px solid var(--border)}}
  .gloss-title{{font-size:.9rem;font-weight:700;color:var(--deep);margin-bottom:6px}}
  .gloss-body{{font-size:13px;color:#374151;line-height:1.65}}

  /* Target context */
  .target-box{{background:var(--card);border-radius:16px;padding:22px 24px;border:1px solid var(--border)}}
  .target-header{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}}
  .target-pill{{background:#f1f5f9;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;color:var(--muted)}}
  .target-pill.pos{{background:#dcfce7;color:#166534}}
  .target-goal{{font-size:1.05rem;font-weight:700;color:var(--deep);margin-bottom:14px}}
  .target-classes{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}}
  .tc{{border-radius:12px;padding:14px 16px;border:1px solid var(--border)}}
  .tc.positive{{background:#f0fdf4;border-color:#bbf7d0}}
  .tc.negative{{background:#faf5ff;border-color:#e9d5ff}}
  .tc-pct{{font-size:1.5rem;font-weight:700;color:var(--deep)}}
  .tc.positive .tc-pct{{color:#16a34a}}
  .tc.negative .tc-pct{{color:var(--purple)}}
  .tc-label{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin:2px 0 4px}}
  .tc-desc{{font-size:13px;color:#374151}}
  .target-impact{{font-size:13px;color:#374151;line-height:1.7;margin-bottom:8px}}
  .uc-note{{font-size:13px;color:#1e40af;background:#eff6ff;border-radius:8px;padding:8px 12px;border-left:3px solid #93c5fd}}

  /* Key drivers */
  .drivers-wrap{{display:flex;flex-direction:column;gap:0}}
  .driver-row{{display:grid;grid-template-columns:36px 1fr 64px;gap:14px;align-items:start;
               padding:16px 0;border-bottom:1px solid var(--border)}}
  .driver-row:last-child{{border-bottom:none}}
  .driver-rank{{font-size:1.1rem;font-weight:700;color:var(--muted);padding-top:2px;text-align:center}}
  .driver-main{{min-width:0}}
  .driver-name{{font-size:.95rem;font-weight:700;color:var(--deep);margin-bottom:6px}}
  .driver-short{{display:inline-block;background:#f1f5f9;color:var(--muted);font-size:10px;
                 font-weight:600;padding:2px 8px;border-radius:999px;margin-left:8px;
                 vertical-align:middle;text-transform:uppercase;letter-spacing:.05em}}
  .driver-bar-wrap{{height:8px;background:#f1f5f9;border-radius:4px;margin-bottom:8px;overflow:hidden}}
  .driver-bar{{height:100%;border-radius:4px;transition:width .4s ease}}
  .driver-bar.pos-dir{{background:linear-gradient(90deg,var(--aqua),var(--deep))}}
  .driver-bar.neg-dir{{background:linear-gradient(90deg,var(--coral),#991b1b)}}
  .driver-direction{{font-size:13px;margin-bottom:6px}}
  .driver-direction.pos-dir{{color:#0f766e}}
  .driver-direction.neg-dir{{color:#dc2626}}
  .driver-note{{font-size:12px;color:var(--muted);line-height:1.6;font-style:italic}}
  .driver-imp{{font-size:.85rem;font-weight:700;font-variant-numeric:tabular-nums;
               color:var(--muted);text-align:right;padding-top:2px}}

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
    <h1>{dataset_label}</h1>
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
        <div class="val">{len(feature_meta)}</div>
        <div class="lbl">Variables Used</div>
      </div>
    </div>
  </div>
</div>

<main>

{intro_section}

{ai_section}

{glossary_section}

{variables_section}

{dropped_section}

<!-- Leaderboard -->
<section class="section">
  <h2>Model Leaderboard</h2>
  <p class="section-note">All {n_models} models ranked by their ROC-AUC score on the held-out Test set.</p>
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
  <h2>Detailed Model Metrics</h2>
  <p class="section-note">Green = good (≥ 0.75), Amber = acceptable (≥ 0.60), Red = needs improvement.
  A consistent score across Train/Test/OOT/ETRC indicates the model is stable and not over-fitted.</p>
  {model_sections}
</section>

<!-- Charts grid -->
<section class="section">
  <h2>ROC Curves &amp; Metrics Comparison</h2>
  <p class="section-note">The ROC curve shows how well each model separates positives from negatives across
  all decision thresholds. The closer the curve hugs the top-left corner, the better.</p>
  <div class="chart-grid">
    <div class="chart-wrap"><img src="data:image/png;base64,{charts['roc']}" alt="ROC Curves"/></div>
    <div class="chart-wrap"><img src="data:image/png;base64,{charts['metrics']}" alt="Metrics"/></div>
  </div>
</section>

{features_section}

<!-- Stability -->
<section class="section">
  <h2>Stability Across Data Splits</h2>
  <p class="section-note">A well-generalising model should show similar ROC-AUC values across all four splits.
  Large drops from Train → OOT or ETRC may indicate over-fitting or concept drift.</p>
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

    # Enriched metadata
    report_meta   = ds.get("report_meta") or {}
    feature_meta  = ds.get("feature_meta") or []
    dropped_cols  = ds.get("dropped_cols") or {}
    dataset_label = report_meta.get("dataset_label") or "ML Dataset"
    target_col    = report_meta.get("target_col") or "target"
    positive_cls  = report_meta.get("positive_class") or "1"
    class_balance = report_meta.get("class_balance") or {}
    num_features  = sum(1 for f in feature_meta if f.get("type") == "numeric")
    cat_features  = sum(1 for f in feature_meta if f.get("type") == "categorical")
    pos_pct = class_balance.get(1, class_balance.get("1", 0))
    if isinstance(pos_pct, float) and pos_pct <= 1:
        pos_pct = round(pos_pct * 100, 1)

    story: list = []

    # --- Cover ---
    story += [
        Spacer(1, 1 * cm),
        Paragraph("ML on the Go", ParagraphStyle("brand", parent=styles["Normal"],
            fontSize=11, textColor=_rl(_AQUA), spaceAfter=8)),
        Paragraph(dataset_label, S["title"]),
        Paragraph(f"Generated {ts}", S["subtitle"]),
        HRFlowable(width="100%", thickness=1, color=_rl(_AQUA), spaceAfter=20),
    ]

    # Summary stats table
    cover_data = [
        ["Best ROC-AUC", "Best Model", "Total Rows", "Variables Used"],
        [f"{best_auc:.4f}", best_name,
         f"{ds.get('n_rows', '–'):,}" if ds.get("n_rows") else "–",
         str(len(feature_meta)) if feature_meta else str(ds.get("n_features", "–"))],
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

    # --- What Does This Model Predict? ---
    ctx = _TARGET_CONTEXT.get(target_col, {
        "goal":     f"predict {target_col}",
        "positive": f"the outcome is '{positive_cls}'",
        "negative": f"the outcome is not '{positive_cls}'",
        "impact":   "Understanding what drives this outcome helps make faster, more accurate decisions.",
        "use_case": "",
    })
    pos_pct_val = class_balance.get(1, class_balance.get("1", 0))
    if isinstance(pos_pct_val, float) and pos_pct_val <= 1:
        pos_pct_val = round(pos_pct_val * 100, 1)
    neg_pct_val = round(100 - float(pos_pct_val), 1)

    story.append(Paragraph("What Does This Model Predict?", S["h2"]))
    story.append(Paragraph(
        f"<b>Goal:</b> {ctx['goal'].capitalize()}. "
        f"The model was trained on <b>{dataset_label}</b> to predict "
        f"<b>{target_col}</b> (positive class = {positive_cls}).",
        S["body"]
    ))
    story += [Spacer(1, 0.2 * cm)]
    outcome_data = [
        ["Outcome", "What It Means", "Share of Data"],
        [f"Positive ({positive_cls})", ctx["positive"], f"{pos_pct_val}%"],
        [f"Negative", ctx["negative"], f"{neg_pct_val}%"],
    ]
    ot = Table(outcome_data, colWidths=[W*0.2, W*0.6, W*0.2])
    ot.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _rl(_DEEP)),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _rl("white")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (-1, 1), _rl("#f0fdf4")),
        ("BACKGROUND",    (0, 2), (-1, 2), _rl("#faf5ff")),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, _rl("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [ot, Spacer(1, 0.2 * cm)]
    story.append(Paragraph(f"<b>Why it matters:</b> {ctx['impact']}", S["body"]))
    if ctx.get("use_case"):
        story.append(Paragraph(f"<b>Example use:</b> {ctx['use_case']}", S["note"]))
    story += [Spacer(1, 0.3 * cm)]

    overview_data = [
        ["Metric", "Value"],
        ["Total rows", f"{ds.get('n_rows', '–'):,}" if ds.get("n_rows") else "–"],
        ["Variables used", str(len(feature_meta))],
        ["Numeric variables", str(num_features)],
        ["Categorical variables", str(cat_features)],
        ["Positive cases", f"{pos_pct_val}%"],
        ["Variables removed", str(len(dropped_cols))],
    ]
    ov_table = Table(overview_data, colWidths=[W * 0.55, W * 0.45])
    ov_table.setStyle(_base_table_style())
    story += [ov_table, Spacer(1, 0.5 * cm)]

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

    # --- Metric Glossary ---
    story.append(Paragraph("How to Read These Numbers", S["h2"]))
    glossary_items = [
        ("ROC-AUC",
         "Overall ability to separate positive from negative cases. "
         "1.0 = perfect, 0.5 = random guessing. Above 0.75 is good; above 0.85 is very strong."),
        ("Precision",
         "Of all records flagged as positive, what fraction actually were? "
         "High precision = few false alarms."),
        ("Recall",
         "Of all true positives, what fraction did the model catch? "
         "High recall = few missed cases."),
        ("F1 Score",
         "A balanced average of Precision and Recall (harmonic mean). Closer to 1.0 is better."),
        ("Train / Test / OOT / ETRC",
         "Data is split into four parts: Train (model learns here), Test (first validation), "
         "OOT (out-of-time — recent data model never saw), ETRC (stress-test scenario). "
         "A good model scores consistently across all four."),
    ]
    gloss_data = [["Term", "Plain-English Meaning"]]
    for term, meaning in glossary_items:
        gloss_data.append([term, meaning])
    gloss_table = Table(gloss_data, colWidths=[W * 0.28, W * 0.72])
    gloss_table.setStyle(_base_table_style())
    story += [gloss_table, Spacer(1, 0.5 * cm)]

    # --- Variables Used ---
    if feature_meta:
        story.append(Paragraph(f"Variables Used by the Model ({len(feature_meta)} total)", S["h2"]))
        story.append(Paragraph(
            "These are the inputs the model learned from. Each row shows the variable name, "
            "its type, how much data is missing, and a quick summary of its values.",
            S["body"]
        ))
        feat_header = ["Variable", "Type", "Missing %", "Summary"]
        feat_rows = [feat_header]
        for f in feature_meta:
            col  = f.get("column", "")
            typ  = f.get("type", "")
            null = f"{f.get('null_pct', 0)}%"
            if typ == "numeric":
                detail = (f'avg {f["mean"]:g}, range [{f["min"]:g}–{f["max"]:g}]'
                          if f.get("mean") is not None else "—")
            else:
                detail = f'top: {f.get("top_val","—")} ({f.get("unique","?")} categories)'
            feat_rows.append([col, typ, null, detail])
        feat_table = Table(feat_rows, colWidths=[W*0.26, W*0.12, W*0.10, W*0.52])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _rl(_DEEP)),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _rl("white")),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rl("white"), _rl("#f8fafc")]),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, _rl("#e2e8f0")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [Spacer(1, 0.3 * cm), feat_table, Spacer(1, 0.5 * cm)]

    # --- Dropped Variables ---
    if dropped_cols:
        story.append(Paragraph(f"Variables Removed Before Training ({len(dropped_cols)})", S["h2"]))
        story.append(Paragraph(
            "These columns were automatically excluded because they are not useful for predicting "
            "the outcome (e.g., timestamps, IDs, constant values).",
            S["body"]
        ))
        drop_rows = [["Column Name", "Reason Removed"]]
        for col, reason in dropped_cols.items():
            drop_rows.append([col, reason])
        drop_table = Table(drop_rows, colWidths=[W * 0.35, W * 0.65])
        drop_table.setStyle(_base_table_style())
        story += [Spacer(1, 0.3 * cm), drop_table, Spacer(1, 0.5 * cm)]

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

    # --- Key Drivers (detailed, no AI) ---
    fi_list = (best.get("feature_importance") or [])[:15]
    if fi_list:
        story += [
            PageBreak(),
            Paragraph("Key Drivers — What Influences the Prediction", S["h2"]),
            Paragraph(
                f"Ranked by importance from the best model ({best.get('name','')})."
                " Each variable's importance score shows how much it contributed to the model's decisions.",
                S["body"]
            ),
            Spacer(1, 0.3 * cm),
        ]
        max_imp = max(abs(f["importance"]) for f in fi_list) or 1
        driver_data = [["#", "Variable", "Type", "Importance", "What It Tells Us"]]
        for rank, f in enumerate(fi_list, 1):
            desc = _feat_description(f["feature"], f["importance"],
                                     target_col, best.get("name",""))
            bar_pct = round(abs(f["importance"]) / max_imp * 100)
            driver_data.append([
                str(rank),
                desc["label"],
                desc["short"],
                f"{abs(f['importance']):.4f}",
                desc["direction"].replace("<strong>", "").replace("</strong>", ""),
            ])
        dt = Table(driver_data, colWidths=[W*0.05, W*0.22, W*0.12, W*0.10, W*0.51])
        dt.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _rl(_DEEP)),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _rl("white")),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rl("white"), _rl("#f8fafc")]),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, _rl("#e2e8f0")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [dt, Spacer(1, 0.4 * cm)]

        # Business notes for top 5
        story.append(Paragraph("Business Interpretation of Top 5 Drivers", S["h3"]))
        for f in fi_list[:5]:
            desc = _feat_description(f["feature"], f["importance"],
                                     target_col, best.get("name",""))
            story.append(Paragraph(
                f"<b>{desc['label']}:</b> {desc['note']}", S["note"]
            ))
        story += [Spacer(1, 0.3 * cm)]

    # Feature importance chart
    fi_fig = _chart_feature_importance(best)
    if fi_fig:
        story += [
            Paragraph("Feature Importance Chart", S["h3"]),
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
