"""
Deterministic input normalisation with a reviewable change log.

Users format the same variable differently: "₹1,20,000", "1,200,000",
"25%", "yes"/"no", "NA"/"-" for missing. Text columns that turn out to be
numeric after stripping recognised finance presentation are converted,
and every conversion is reported so the user can review what happened.
Nothing is guessed from locale; only explicit patterns are recognised.

Ported from the codex implementation's input_normalization.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

MISSING = {"", "na", "n/a", "null", "none", "nan", "-", "--", "nil", "#n/a"}
_YES = {"yes", "y", "true", "t"}
_NO = {"no", "n", "false", "f"}
# Western 1,234,567 or Indian 12,34,567 grouping, optional sign/decimals.
_GROUPED = r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?$"


def normalise_frame(frame: pd.DataFrame, skip: Tuple[str, ...] = ()) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = frame.copy()
    changes: List[Dict[str, Any]] = []
    for col in out.columns:
        if col in skip or pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        original = out[col]
        values = original.astype("string").str.strip()
        missing_mask = values.str.lower().isin(MISSING)
        values = values.mask(missing_mask)
        non_null = values.dropna()
        if non_null.empty:
            out[col] = np.nan
            changes.append({"column": col, "action": "All values empty; represented as missing"})
            continue

        lower = non_null.str.lower()
        if lower.isin(_YES | _NO).all():
            out[col] = lower.map(lambda v: 1.0 if v in _YES else 0.0).reindex(values.index).astype(float)
            changes.append({"column": col, "action": "Yes/No text converted to 1/0"})
            continue

        clean = non_null.str.replace(r"^[₹$€£]\s*", "", regex=True).str.replace(r"\s*%$", "", regex=True)
        grouped = clean.str.match(_GROUPED)
        clean = clean.mask(grouped, clean.str.replace(",", "", regex=False))
        numeric = pd.to_numeric(clean, errors="coerce")
        if numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all():
            out[col] = numeric.reindex(values.index).to_numpy(dtype=float, na_value=np.nan)
            action = "Numeric text converted"
            details = []
            if non_null.str.contains(r"^[₹$€£]", regex=True).any():
                details.append("currency symbol removed")
            if bool(grouped.any()):
                details.append("thousand/lakh separators removed")
            if non_null.str.endswith("%").any():
                details.append("percent sign removed (25% becomes 25)")
            if missing_mask.any():
                details.append(f"{int(missing_mask.sum())} empty markers set to missing")
            changes.append({"column": col, "action": action + (": " + ", ".join(details) if details else "")})
        else:
            out[col] = values.astype(object).where(values.notna(), None)
            if missing_mask.any() or not original.fillna("").astype(str).str.strip().equals(values.fillna("").astype(str)):
                changes.append({"column": col, "action": "Whitespace trimmed and empty markers standardised; text kept as categories"})
    return out, changes
