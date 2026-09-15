"""
Unit tests for the engine pieces that must hold regardless of the data:
Nu Score scaling, learned bands, selection never seeing test/OOT,
input normalisation, column harmonisation, chronological cohorts and PSI.

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.data import split_cohorts
from backend.app.engine import drift, evaluate, harmonise, nuscore
from backend.app.engine.normalise import normalise_frame
from backend.app.engine.preprocess import InputSchema


# ------------------------------------------------------------------ Nu Score


def test_score_scale_anchor_and_pdo():
    assert nuscore.to_score(np.array([0.5]))[0] == 500
    # Halving the odds of going bad (safer) adds PDO points.
    p_even, p_safer = 0.5, 1 / 3        # odds bad:good 1:1 -> 1:2
    s_even, s_safer = nuscore.to_score(np.array([p_even, p_safer]))
    assert abs((s_safer - s_even) - nuscore.PDO) <= 1


def test_score_monotone_and_bounded():
    p = np.linspace(0.0001, 0.9999, 500)
    s = nuscore.to_score(p)
    assert np.all(np.diff(s) <= 0)          # more risk -> lower score
    assert s.min() >= 0 and s.max() <= 1000


def test_bands_are_monotone_in_bad_rate_and_have_support():
    rng = np.random.default_rng(0)
    score = rng.integers(200, 900, 5000)
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp((score - 550) / 60))).astype(int)
    bands = nuscore.fit_bands(score, y)
    table = nuscore.band_table(score, y, bands)
    rates = [row["bad_rate"] for row in table]
    assert rates == sorted(rates), "bad rate must rise from A+ to E"
    assert all(row["share"] >= nuscore.MIN_SHARE - 1e-9 for row in table)
    assert bands[0]["max_score"] == 1000 and bands[-1]["min_score"] == 0
    for upper, lower in zip(bands, bands[1:]):
        assert upper["min_score"] == lower["max_score"] + 1     # contiguous


def test_calibrator_is_smooth_and_monotone():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 3000)
    y = (rng.uniform(size=3000) < p ** 1.5).astype(int)
    cal = nuscore.Calibrator().fit(p, y)
    out = cal(np.linspace(0.001, 0.999, 200))
    assert np.all(np.diff(out) >= 0)
    assert 0 < out.min() and out.max() < 1


# ----------------------------------------------------- selection discipline


def _fake_model(name, val_auc, val_brier, test_auc, family="boosting"):
    def ev(auc, brier):
        return {"metrics": {"roc_auc": auc, "brier": brier}}
    return {
        "name": name, "family": family, "fit_seconds": 1.0,
        "evaluations": {"train": ev(val_auc + 0.02, val_brier), "validation": ev(val_auc, val_brier),
                        "calibration": ev(val_auc, val_brier), "test": ev(test_auc, val_brier), "oot": ev(test_auc, val_brier)},
        "stability": {"overfit_gap": 0.02, "test_drop": 0.0, "oot_drop": 0.0, "calibration_drop": 0.0,
                      "score_psi_calibration": 0.0, "score_psi_oot": 0.0},
    }


def test_composite_selection_ignores_test_and_oot():
    a = _fake_model("A", val_auc=0.80, val_brier=0.15, test_auc=0.60)
    b = _fake_model("B", val_auc=0.75, val_brier=0.16, test_auc=0.95)
    scores = {m["name"]: evaluate.composite_score(m, [a, b])["score"] for m in (a, b)}
    assert scores["A"] > scores["B"], "a much better test AUC must not change the winner"
    # Flip test/OOT numbers entirely: the composite must be identical.
    for m in (a, b):
        for cohort in ("test", "oot"):
            m["evaluations"][cohort]["metrics"]["roc_auc"] = 0.5
    again = {m["name"]: evaluate.composite_score(m, [a, b])["score"] for m in (a, b)}
    assert again == scores


# ------------------------------------------------------------ normalisation


def test_normalise_frame_handles_indian_and_western_formats():
    frame = pd.DataFrame({
        "amount": ["₹1,20,000", "1,200,000", "80000", "NA"],
        "rate": ["12.5%", "9%", "-", "10%"],
        "flag": ["yes", "No", "Y", "n"],
        "city": ["Pune ", " Pune", "Delhi", ""],
    })
    out, log = normalise_frame(frame)
    assert out["amount"].tolist()[:3] == [120000.0, 1200000.0, 80000.0] and np.isnan(out["amount"].iloc[3])
    assert out["rate"].tolist()[:2] == [12.5, 9.0]
    assert out["flag"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert out["city"].tolist()[:3] == ["Pune", "Pune", "Delhi"] and out["city"].iloc[3] is None
    assert {entry["column"] for entry in log} == {"amount", "rate", "flag", "city"}


def test_harmonise_maps_renamed_columns_without_llm():
    schema = InputSchema(numeric=["loan_amount", "monthly_income", "bureau_score", "tenure_months"],
                         categorical=["employment_type"], categories={"employment_type": ["salaried"]})
    mapping = harmonise.map_columns(schema, ["Loan Amt (Rs)", "Net Monthly Income (INR)", "CIBIL Score",
                                             "Tenor (months)", "EmploymentType"], ai_client=None)
    assert all(m["source_col"] is not None for m in mapping), mapping
    frame = pd.DataFrame({"Loan Amt (Rs)": ["2.5 lakh"], "Net Monthly Income (INR)": ["45,000"], "CIBIL Score": [720],
                          "Tenor (months)": [24], "EmploymentType": [" Salaried "]})
    applied = harmonise.apply_mapping(frame, schema, mapping)
    assert applied.loc[0, "loan_amount"] == 250000 and applied.loc[0, "monthly_income"] == 45000
    assert applied.loc[0, "employment_type"] == "salaried"


# ------------------------------------------------------------------ cohorts


def test_chronological_cohorts_never_split_a_date():
    rng = np.random.default_rng(2)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 60, 3000), unit="D")
    df = pd.DataFrame({"d": dates, "x": rng.normal(size=3000), "y": rng.integers(0, 2, 3000)})
    cohorts, info = split_cohorts(df, "y", time_col="d")
    assert info["strategy"] == "time" and "true out-of-time" in info["oot_status"]
    seen = {}
    for name, part in cohorts.items():
        for d in part["d"].unique():
            assert seen.setdefault(d, name) == name, "a date appeared in two cohorts"
    order = ["train", "validation", "calibration", "test", "oot"]
    maxes = [cohorts[n]["d"].max() for n in order]
    assert maxes == sorted(maxes)
    assert sum(len(c) for c in cohorts.values()) == len(df)


def test_random_cohorts_flag_oot_as_not_true():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=1000), "y": rng.integers(0, 2, 1000)})
    cohorts, info = split_cohorts(df, "y", time_col=None)
    assert info["strategy"] == "random" and "not a true out-of-time" in info["oot_status"]
    assert set(cohorts) == {"train", "validation", "calibration", "test", "oot"}


# ---------------------------------------------------------------------- PSI


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=20000)
    profile = drift.fit_distribution(ref)
    same = drift.compare(profile, rng.normal(size=20000))
    shifted = drift.compare(profile, rng.normal(loc=1.5, size=20000))
    assert same["psi"] < 0.02 and same["status"] == "stable"
    assert shifted["psi"] > 0.25 and shifted["status"] == "significant"
    cats = drift.fit_distribution(pd.Series(["a", "b", "c"] * 1000))
    assert drift.compare(cats, pd.Series(["a"] * 3000))["status"] == "significant"
