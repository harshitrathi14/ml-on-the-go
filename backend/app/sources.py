"""
Convert uploaded sources (CSV, bureau JSON, Parquet) to Parquet.

CSV conversion streams through Polars so a multi-GB file never sits in
memory. JSON (bureau responses) is flattened: nested objects become
``a.b`` columns and lists of records (trade lines, enquiries) become
count / sum / max / min aggregates per numeric field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import polars as pl

# Column names are used in SQL and as model features; keep them simple.
_BAD_CHARS = re.compile(r"[^0-9A-Za-z_]+")


def clean_column_name(name: str) -> str:
    cleaned = _BAD_CHARS.sub("_", str(name).strip()).strip("_")
    return cleaned or "col"


def _dedupe(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def convert_to_parquet(raw: Path, kind: str, out: Path) -> None:
    if kind == "csv":
        _csv_to_parquet(raw, out)
    elif kind == "json":
        _json_to_parquet(raw, out)
    elif kind == "parquet":
        lf = pl.scan_parquet(raw)
        lf = lf.rename(dict(zip(lf.collect_schema().names(),
                                _dedupe([clean_column_name(c) for c in lf.collect_schema().names()]))))
        lf.sink_parquet(out)
    else:
        raise ValueError(f"Unsupported source kind: {kind}")
    raw.unlink(missing_ok=True)


def _csv_to_parquet(raw: Path, out: Path) -> None:
    # infer_schema_length: look far enough that a late string in a numeric
    # column does not break the stream; polars promotes the column to string.
    lf = pl.scan_csv(
        raw,
        infer_schema_length=100_000,
        ignore_errors=False,
        try_parse_dates=False,
        null_values=["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"],
    )
    names = lf.collect_schema().names()
    lf = lf.rename(dict(zip(names, _dedupe([clean_column_name(c) for c in names]))))
    try:
        lf.sink_parquet(out)
    except pl.exceptions.ComputeError:
        # Schema inference failed on a later row: fall back to all-string
        # scan, then let downstream profiling recover numeric types.
        lf = pl.scan_csv(raw, infer_schema=False, null_values=["", "NA", "N/A", "null", "NULL", "None"])
        names = lf.collect_schema().names()
        lf = lf.rename(dict(zip(names, _dedupe([clean_column_name(c) for c in names]))))
        lf.sink_parquet(out)


# ----------------------------------------------------------------- JSON


def _iter_records(raw: Path) -> Iterable[Dict[str, Any]]:
    text = raw.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)]
    if text[0] == "{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            # Either a wrapper {"data": [...]} or a map {id: {...}}.
            for value in data.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return value
            if all(isinstance(v, dict) for v in data.values()):
                return [{"_id": k, **v} for k, v in data.items()]
            return [data]
    # NDJSON: one object per line.
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def flatten_record(record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_record(value, f"{name}."))
        elif isinstance(value, list):
            flat.update(_flatten_list(name, value))
        else:
            flat[name] = value
    return flat


def _flatten_list(name: str, items: List[Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {f"{name}.count": len(items)}
    dict_items = [i for i in items if isinstance(i, dict)]
    if not dict_items:
        scalars = [i for i in items if isinstance(i, (int, float)) and not isinstance(i, bool)]
        if scalars:
            flat[f"{name}.sum"] = sum(scalars)
            flat[f"{name}.max"] = max(scalars)
        elif items:
            flat[f"{name}.first"] = items[0] if isinstance(items[0], (str, int, float)) else None
        return flat
    # Aggregate numeric sub-fields across the list of records.
    values: Dict[str, List[float]] = {}
    for item in dict_items:
        for sub, value in flatten_record(item).items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                values.setdefault(sub, []).append(float(value))
            elif isinstance(value, str):
                # Numeric strings are common in bureau payloads.
                try:
                    number = float(value.replace(",", ""))
                except ValueError:
                    continue
                values.setdefault(sub, []).append(number)
    for sub, nums in values.items():
        flat[f"{name}.{sub}.sum"] = sum(nums)
        flat[f"{name}.{sub}.max"] = max(nums)
        flat[f"{name}.{sub}.min"] = min(nums)
    return flat


def _json_to_parquet(raw: Path, out: Path) -> None:
    rows = [flatten_record(r) for r in _iter_records(raw)]
    if not rows:
        raise ValueError("No JSON records found (expected an array of objects or one object per line).")
    df = pd.DataFrame(rows)
    df.columns = _dedupe([clean_column_name(c) for c in df.columns])
    # Mixed-type object columns cannot be written; coerce them to text.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: None if v is None else str(v))
    df.to_parquet(out, index=False)
