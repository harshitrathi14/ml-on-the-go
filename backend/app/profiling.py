"""
Column profiling and data checks on Parquet sources.

DuckDB does the heavy lifting (SUMMARIZE runs in seconds on tens of
millions of rows); pandas is only used on bounded samples for date
parsing, PII pattern checks and IV computation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

PROFILE_MAX_COLUMNS = 400
SAMPLE_ROWS = 5_000
IV_SAMPLE_ROWS = 500_000

_NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                  "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL")
_DATE_TYPES = ("DATE", "TIMESTAMP")

# Column-name hints. Values are checked too, so these only need to be broad.
_ID_NAME = re.compile(r"(^|_)(id|ids|loan|lan|account|acct|acc|ref|reference|number|no|num|key|uuid|application|app|customer|cust|contract|agreement)(_|$)", re.I)
_DATE_NAME = re.compile(r"(date|_dt$|^dt_|time|timestamp|month|period|snapshot|as_of|asof|disburs|vintage)", re.I)
_PII_NAME = re.compile(r"(^|_)(name|first_name|last_name|full_name|father|mother|spouse|mobile|phone|contact|email|e_mail|address|addr|street|pan|pan_no|aadhaar|aadhar|uid|passport|voter|dob|date_of_birth|birth|ip_address|lat|long|latitude|longitude|gps)(_|$)", re.I)
_LEAK_NAME = re.compile(r"(post_|_post$|after_|outcome|writeoff|write_off|written_off|charge_off|chargeoff|settled|settlement|recover|npa|npa_flag|bad_flag|default_flag|default_date|closure|closed_on|foreclos|status_final|final_)", re.I)

_PII_VALUE = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I),
    "indian_mobile": re.compile(r"^(\+91[\-\s]?)?[6-9]\d{9}$"),
    "pan": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
    "aadhaar": re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$"),
}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


# ----------------------------------------------------------------- profile


def profile_parquet(path: Path, con: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, Any]:
    con = con or duckdb.connect()
    src = f"'{path.as_posix()}'"
    row_count = con.sql(f"SELECT count(*) FROM {src}").fetchone()[0]
    summary = con.sql(f"SUMMARIZE SELECT * FROM {src}").df()
    if len(summary) > PROFILE_MAX_COLUMNS:
        summary = summary.iloc[:PROFILE_MAX_COLUMNS]

    sample = con.sql(f"SELECT * FROM {src} USING SAMPLE {SAMPLE_ROWS} ROWS").df() if row_count else pd.DataFrame()

    columns: List[Dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        name = row.column_name
        dtype = _kind(str(row.column_type))
        null_pct = _num(row.null_percentage) or 0.0
        distinct = int(_num(row.approx_unique) or 0)
        non_null = max(row_count * (1 - null_pct / 100), 1)
        col: Dict[str, Any] = {
            "name": name,
            "dtype": dtype,
            "sql_type": str(row.column_type),
            "null_pct": round(null_pct, 2),
            "distinct": distinct,
            "unique_ratio": round(min(distinct / non_null, 1.0), 4),
        }
        if dtype == "numeric":
            col.update(min=_num(row.min), max=_num(row.max), mean=_num(row.avg),
                       std=_num(row.std), median=_num(row.q50))
        elif dtype == "date":
            col.update(min=str(row.min), max=str(row.max))
        if name in sample.columns:
            values = sample[name].dropna()
            if dtype == "text":
                top = values.astype(str).value_counts().head(5)
                col["sample_values"] = [str(v)[:40] for v in top.index]
                col["date_like"] = _looks_like_date(values.astype(str), name)
            elif dtype == "numeric":
                col["date_like"] = _looks_like_yyyymmdd(values)
                if col.get("date_like") is False and distinct <= 20:
                    col["sample_values"] = [str(v) for v in sorted(values.unique())[:20]]
        columns.append(col)

    return {
        "row_count": int(row_count),
        "column_count": int(len(summary)),
        "columns": columns,
        "date_columns": [c["name"] for c in columns if c["dtype"] == "date" or c.get("date_like")],
        "id_columns": [c["name"] for c in columns if is_id_like(c, row_count)],
    }


def _kind(sql_type: str) -> str:
    upper = sql_type.upper()
    if upper.startswith(_NUMERIC_TYPES):
        return "numeric"
    if upper.startswith(_DATE_TYPES):
        return "date"
    if upper == "BOOLEAN":
        return "bool"
    return "text"


def _looks_like_date(values: pd.Series, name: str) -> bool:
    if values.empty:
        return False
    sample = values.head(2000)
    # Dates contain digits and a separator; skip plain codes such as "tier1".
    if (sample.str.contains(r"\d", regex=True) & sample.str.contains(r"[-/ :.T]", regex=True)).mean() < 0.9:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=bool(re.search(r"dd[-/]?mm", name, re.I)))
    ok = parsed.notna().mean()
    years_ok = parsed.dt.year.between(1950, 2100).mean() if ok else 0
    return bool(ok >= 0.95 and years_ok >= 0.95)


def _looks_like_yyyymmdd(values: pd.Series) -> bool:
    # Nullable Int64 (what DuckDB returns for joined columns) is not a numpy dtype.
    if values.empty or not pd.api.types.is_integer_dtype(values):
        return False
    sample = values.head(2000)
    if not sample.between(19500101, 21001231).mean() >= 0.98:
        return False
    parsed = pd.to_datetime(sample.astype(str), errors="coerce", format="%Y%m%d")
    return bool(parsed.notna().mean() >= 0.98)


def is_id_like(col: Dict[str, Any], row_count: int) -> bool:
    if row_count < 50 or col["dtype"] not in ("text", "numeric"):
        return False
    if col.get("date_like"):
        return False
    if col["unique_ratio"] >= 0.95 and col["dtype"] == "text":
        return True
    if col["unique_ratio"] >= 0.95 and col["dtype"] == "numeric" and _ID_NAME.search(col["name"]):
        return True
    return False


def key_candidates(profile: Dict[str, Any]) -> List[str]:
    """Columns that could serve as a join key, best first."""
    scored = []
    for col in profile["columns"]:
        if col["dtype"] not in ("text", "numeric") or col.get("date_like"):
            continue
        score = col["unique_ratio"]
        if _ID_NAME.search(col["name"]):
            score += 0.5
        if col["distinct"] < 2 or col["null_pct"] > 50:
            continue
        if score >= 0.4:
            scored.append((score, col["name"]))
    return [name for _, name in sorted(scored, reverse=True)][:12]


# ------------------------------------------------------------------ checks


def pii_check(profile: Dict[str, Any], sample: pd.DataFrame) -> List[Dict[str, Any]]:
    """Columns that look like personal data, by name or by value pattern."""
    flags = []
    for col in profile["columns"]:
        name = col["name"]
        reasons = []
        if _PII_NAME.search(name):
            reasons.append("name suggests personal data")
        if col["dtype"] == "text" and name in sample.columns:
            values = sample[name].dropna().astype(str).str.strip().head(2000)
            if len(values) >= 20:
                for label, pattern in _PII_VALUE.items():
                    if values.str.match(pattern).mean() >= 0.8:
                        reasons.append(f"values look like {label.replace('_', ' ')}")
        if reasons:
            flags.append({"column": name, "kind": "pii", "severity": "high",
                          "reason": "; ".join(reasons),
                          "action": "excluded from features and dropped before training"})
    return flags


def leakage_check(iv_table: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    flags = []
    iv_by = {r["feature"]: r for r in iv_table}
    for col in profile["columns"]:
        name = col["name"]
        iv = iv_by.get(name, {}).get("iv")
        reasons = []
        if iv is not None and iv > 1.0:
            reasons.append(f"IV {iv:.2f} is implausibly high for a genuine predictor")
        if _LEAK_NAME.search(name):
            reasons.append("name suggests it is known only after the outcome")
        if reasons:
            severity = "high" if (iv is not None and iv > 1.0) else "medium"
            flags.append({"column": name, "kind": "leakage", "severity": severity,
                          "reason": "; ".join(reasons),
                          "action": "excluded by default; include only if it is known at decision time"})
    return flags


def id_check(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"column": name, "kind": "identifier", "severity": "low",
             "reason": "nearly unique per row, so it cannot generalise",
             "action": "kept as the record key, not used as a feature"}
            for name in profile["id_columns"]]


# ---------------------------------------------------------------------- IV


def _bin_series(series: pd.Series, max_bins: int) -> Tuple[pd.Series, Dict[Any, str]]:
    """Bin codes plus display labels. Numeric bins show their range; category
    labels are anonymised so raw values never leave the process."""
    codes = pd.Series(-1, index=series.index, dtype=int)
    labels: Dict[Any, str] = {-1: "Missing"}
    observed = series.dropna()
    if observed.empty:
        return codes, labels
    if pd.api.types.is_numeric_dtype(series):
        distinct = observed.nunique()
        if distinct <= 1:
            codes.loc[observed.index] = 0
            labels[0] = "Constant value"
        elif distinct <= max_bins:
            unique = sorted(observed.unique())
            codes.loc[observed.index] = observed.map({v: i for i, v in enumerate(unique)}).astype(int)
            labels.update({i: f"= {v:.5g}" for i, v in enumerate(unique)})
        else:
            cuts, edges = pd.qcut(observed, q=max_bins, labels=False, duplicates="drop", retbins=True)
            codes.loc[observed.index] = cuts.astype(int)
            for i in range(len(edges) - 1):
                labels[i] = f"{'[' if i == 0 else '('}{edges[i]:.5g}, {edges[i + 1]:.5g}]"
    else:
        counts = observed.astype(str).value_counts()
        frequent = counts[counts >= 5].head(20).index.tolist()
        mapping = {v: i for i, v in enumerate(frequent)}
        codes.loc[observed.index] = observed.astype(str).map(mapping).fillna(len(frequent)).astype(int)
        labels.update({i: f"Category {i + 1} (label withheld)" for i in range(len(frequent))})
        labels[len(frequent)] = "Rare / other categories"
    return codes, labels


def compute_iv(df: pd.DataFrame, target: pd.Series, max_bins: int = 10) -> List[Dict[str, Any]]:
    """Information Value per column against a 0/1 target, highest first, with
    the per-bin WOE table (count, events, WOE, IV contribution) and notes."""
    y = pd.Series(target.astype(int).to_numpy(), index=df.index)
    total_bad = max(int(y.sum()), 1)
    total_good = max(int(len(y) - y.sum()), 1)
    out = []
    for name in df.columns:
        series = df[name]
        try:
            codes, labels = _bin_series(series, max_bins)
            table = pd.crosstab(codes, y)
        except (ValueError, TypeError):
            continue
        if table.shape[1] < 2:
            continue
        n_bins = len(table)
        bins, iv = [], 0.0
        for code, row in table.iterrows():
            bad, good = float(row.get(1, 0)), float(row.get(0, 0))
            dist_bad = (bad + 0.5) / (total_bad + 0.5 * n_bins)
            dist_good = (good + 0.5) / (total_good + 0.5 * n_bins)
            woe = float(np.log(dist_good / dist_bad))
            contribution = float((dist_good - dist_bad) * woe)
            iv += contribution
            bins.append({"bin": labels.get(code, str(code)), "count": int(bad + good), "events": int(bad),
                         "event_rate": round(bad / max(bad + good, 1), 4), "woe": round(woe, 4),
                         "iv_contribution": round(contribution, 5)})
        notes = []
        if iv >= 0.5:
            notes.append("Very high IV: check for leakage and confirm on future data")
        if any(min(b["events"], b["count"] - b["events"]) < 5 for b in bins):
            notes.append("Sparse outcome counts in one or more bins")
        out.append({"feature": name, "iv": round(iv, 4), "strength": iv_strength(iv), "bins": bins, "notes": notes,
                    "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
                    "missing_pct": round(100 * float(series.isna().mean()), 2)})
    return sorted(out, key=lambda r: r["iv"], reverse=True)


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


def read_sample(path: Path, n_rows: int, con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    con = con or duckdb.connect()
    return con.sql(f"SELECT * FROM '{path.as_posix()}' USING SAMPLE {int(n_rows)} ROWS").df()
