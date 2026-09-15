from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    etrc: pd.DataFrame


@dataclass
class DatasetSummary:
    n_rows: int
    n_features: int
    target_col: str
    class_balance: Dict[str, Dict[str, float]]
    feature_types: Dict[str, int]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


SYNTHETIC_BASE_FEATURES = 32  # realistic loan-book variables below


def generate_synthetic_dataset(
    n_rows: int = 5000,
    n_features: int = 100,
    n_categorical: int = 10,
    seed: int = 42,
    decision_labels: Tuple[str, str] = ("default", "no_default"),
    default_rate: float = 0.18,
) -> pd.DataFrame:
    """A synthetic retail loan book with realistic variables.

    Default risk is driven by bureau score, enquiries, DPD history,
    leverage (LTV, FOIR), bank-statement behaviour and product mix, so the
    trained models rank familiar drivers at the top. ``n_features`` beyond
    the realistic set adds pure-noise ``noise_XX`` columns, useful to show
    that the models ignore them; ``n_categorical`` is kept for API
    compatibility and has no effect.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    age = np.clip(rng.normal(36, 9, n), 21, 65).round()
    monthly_income = rng.lognormal(10.4, 0.5, n).round(-2)
    employment_type = rng.choice(["salaried", "self_employed", "business", "casual"], n, p=[0.45, 0.25, 0.2, 0.1])
    education = rng.choice(["school", "graduate", "post_graduate"], n, p=[0.35, 0.45, 0.2])
    residence_type = rng.choice(["owned", "rented", "family"], n, p=[0.4, 0.4, 0.2])
    city_tier = rng.choice(["tier1", "tier2", "tier3"], n, p=[0.35, 0.4, 0.25])
    region = rng.choice(["north", "south", "east", "west"], n)
    product = rng.choice(["two_wheeler", "used_car", "personal", "msme", "gold"], n, p=[0.3, 0.15, 0.25, 0.15, 0.15])
    channel = rng.choice(["branch", "dsa", "digital", "dealer"], n, p=[0.35, 0.3, 0.2, 0.15])

    loan_amount = rng.lognormal(11.2, 0.7, n).round(-3)
    tenure_months = rng.choice([12, 18, 24, 36, 48, 60], n)
    interest_rate = np.clip(rng.normal(17, 4, n), 8, 36).round(2)
    emi = (loan_amount * (interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-tenure_months))).round()
    existing_emi = (rng.uniform(0, 0.35, n) * monthly_income).round(-2)
    foir = np.clip((emi + existing_emi) / monthly_income, 0.02, 1.5).round(3)
    ltv = rng.uniform(0.3, 0.95, n).round(3)
    down_payment_pct = (1 - ltv).round(3)

    bureau_score = np.clip(rng.normal(690, 60, n), 300, 900).round()
    bureau_vintage_months = rng.integers(0, 180, n)
    enquiries_3m = rng.poisson(1.5, n)
    enquiries_12m = enquiries_3m + rng.poisson(2.5, n)
    live_trade_lines = rng.poisson(3, n)
    total_outstanding = (rng.lognormal(11.5, 1.0, n) * (live_trade_lines > 0)).round(-3)
    credit_utilisation = np.clip(rng.beta(2, 3, n), 0, 1).round(3)
    worst_dpd_24m = rng.choice([0, 0, 0, 0, 30, 60, 90], n)
    months_since_last_dpd = np.where(worst_dpd_24m > 0, rng.integers(1, 24, n), 99)
    unsecured_share = rng.beta(2, 2, n).round(3)

    avg_bank_balance = rng.lognormal(9.6, 0.9, n).round(-2)
    min_bank_balance = (avg_bank_balance * rng.uniform(0.05, 0.6, n)).round(-2)
    salary_credit_regularity = np.clip(rng.beta(5, 2, n), 0, 1).round(3)
    cheque_bounces_6m = rng.poisson(0.35, n)
    cash_withdrawal_share = rng.beta(2, 5, n).round(3)

    logit = (
        -0.014 * (bureau_score - 690)
        + 0.22 * enquiries_3m
        + 0.010 * worst_dpd_24m
        - 0.02 * np.minimum(months_since_last_dpd, 24)
        + 1.8 * (foir - 0.4)
        + 1.2 * (ltv - 0.6)
        + 1.5 * (credit_utilisation - 0.4)
        - 0.35 * np.log1p(avg_bank_balance / 1000)
        + 0.55 * cheque_bounces_6m
        - 0.8 * (salary_credit_regularity - 0.7)
        + 0.3 * (employment_type == "casual") + 0.15 * (employment_type == "self_employed")
        + 0.25 * (product == "personal") - 0.3 * (product == "gold")
        + 0.2 * (channel == "dsa")
        - 0.01 * (age - 36)
        + rng.normal(0, 0.7, n)
    )
    # Shift the intercept (bisection) so the expected default rate matches default_rate.
    lo, hi = -15.0, 15.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() < default_rate:
            lo = mid
        else:
            hi = mid
    target = (rng.uniform(size=n) < _sigmoid(logit + (lo + hi) / 2)).astype(int)

    df = pd.DataFrame({
        "age": age, "monthly_income": monthly_income, "employment_type": employment_type,
        "education": education, "residence_type": residence_type, "city_tier": city_tier,
        "region": region, "product": product, "channel": channel,
        "loan_amount": loan_amount, "tenure_months": tenure_months, "interest_rate": interest_rate,
        "emi": emi, "existing_emi": existing_emi, "foir": foir, "ltv": ltv, "down_payment_pct": down_payment_pct,
        "bureau_score": bureau_score, "bureau_vintage_months": bureau_vintage_months,
        "enquiries_3m": enquiries_3m, "enquiries_12m": enquiries_12m, "live_trade_lines": live_trade_lines,
        "total_outstanding": total_outstanding, "credit_utilisation": credit_utilisation,
        "worst_dpd_24m": worst_dpd_24m, "months_since_last_dpd": months_since_last_dpd,
        "unsecured_share": unsecured_share, "avg_bank_balance": avg_bank_balance,
        "min_bank_balance": min_bank_balance, "salary_credit_regularity": salary_credit_regularity,
        "cheque_bounces_6m": cheque_bounces_6m, "cash_withdrawal_share": cash_withdrawal_share,
    })
    for idx in range(max(n_features - SYNTHETIC_BASE_FEATURES, 0)):
        df[f"noise_{idx + 1:02d}"] = rng.normal(size=n).round(4)

    label_positive, label_negative = decision_labels
    df["decision"] = np.where(target == 1, label_positive, label_negative)
    df["decision_binary"] = target
    return df


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "decision_binary",
    seed: int = 42,
    train_size: float = 0.6,
    test_size: float = 0.2,
    oot_size: float = 0.1,
    etrc_size: float = 0.1,
) -> DatasetSplits:
    if not np.isclose(train_size + test_size + oot_size + etrc_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df[target_col],
    )

    temp_ratio = 1 - train_size
    test_ratio = test_size / temp_ratio
    oot_ratio = oot_size / temp_ratio

    test_df, holdout_df = train_test_split(
        temp_df,
        train_size=test_ratio,
        random_state=seed,
        stratify=temp_df[target_col],
    )

    oot_df, etrc_df = train_test_split(
        holdout_df,
        train_size=oot_ratio / (oot_ratio + etrc_size / temp_ratio),
        random_state=seed,
        stratify=holdout_df[target_col],
    )

    return DatasetSplits(train=train_df, test=test_df, oot=oot_df, etrc=etrc_df)


COHORTS = ("train", "validation", "calibration", "test", "oot")
# Share of rows per cohort. Chronological: cut-offs at the cumulative shares,
# never splitting rows that share an observation date. Random: stratified.
COHORT_SHARES = {"train": 0.55, "validation": 0.12, "calibration": 0.10, "test": 0.10, "oot": 0.13}


def split_cohorts(
    df: pd.DataFrame, target_col: str, time_col: Optional[str] = None, seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Five disjoint cohorts with distinct jobs:

    train        model fitting and tuning
    validation   model selection, thresholds (never the test set)
    calibration  probability calibration and Nu Score bands
    test         final, untouched performance estimate
    oot          latest period by ``time_col`` (true out-of-time); when there is
                 no usable time column it is a random hold-out and is labelled so

    Rows sharing an observation date are never divided across chronological cohorts.
    """
    shares = list(COHORT_SHARES.values())
    bounds = np.cumsum(shares)[:-1]
    info: Dict[str, Any] = {"cohorts": list(COHORTS), "shares": COHORT_SHARES}

    if time_col and time_col in df.columns and df[time_col].notna().all():
        dates = df[time_col]
        unique_dates = dates.nunique()
        cumulative = dates.value_counts().sort_index().cumsum()
        cutoffs = [cumulative.index[min(int(np.searchsorted(cumulative.to_numpy(), len(df) * b)), len(cumulative) - 1)]
                   for b in bounds]
        if unique_dates >= 10 and len(set(cutoffs)) == len(cutoffs):
            groups, previous = [], None
            for boundary in list(cutoffs) + [None]:
                mask = pd.Series(True, index=df.index)
                if previous is not None:
                    mask &= dates > previous
                if boundary is not None:
                    mask &= dates <= boundary
                groups.append(df.loc[mask])
                previous = boundary
            if all(len(g) >= 10 and g[target_col].nunique() == 2 for g in groups):
                cohorts = dict(zip(COHORTS, groups))
                info.update(strategy="time", oot_status="true out-of-time: latest observation dates",
                            time_ranges={n: [str(c[time_col].min())[:10], str(c[time_col].max())[:10]] for n, c in cohorts.items()})
                return cohorts, info
        info["time_split_fallback"] = ("dates too concentrated or too few distinct dates for five chronological "
                                       "cohorts; used random splits instead")

    rest = df
    cohorts = {}
    remaining = 1.0
    for name, share in COHORT_SHARES.items():
        if name == COHORTS[-1]:
            cohorts[name] = rest
            break
        frac = share / remaining
        stratify = rest[target_col] if rest[target_col].nunique() > 1 else None
        part, rest = train_test_split(rest, train_size=frac, random_state=seed, stratify=stratify)
        cohorts[name] = part
        remaining -= share
    info.update(strategy="random", oot_status="not a true out-of-time: random hold-out (no usable time column)")
    return cohorts, info


def split_by_time(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    seed: int = 42,
    oot_size: float = 0.1,
    etrc_size: float = 0.1,
    test_share_of_dev: float = 0.25,
) -> DatasetSplits:
    """Time-ordered splits: the latest rows form the stress (ETRC) split, the
    rows before them the Out-of-Time split, and the earliest 80% are split
    randomly into train (60% of all) and test (20% of all)."""
    ordered = df.sort_values(time_col, kind="stable")
    n = len(ordered)
    n_etrc = int(n * etrc_size)
    n_oot = int(n * oot_size)
    dev = ordered.iloc[: n - n_oot - n_etrc]
    oot_df = ordered.iloc[n - n_oot - n_etrc : n - n_etrc]
    etrc_df = ordered.iloc[n - n_etrc :]
    stratify = dev[target_col] if dev[target_col].nunique() > 1 else None
    train_df, test_df = train_test_split(
        dev, test_size=test_share_of_dev, random_state=seed, stratify=stratify
    )
    return DatasetSplits(train=train_df, test=test_df, oot=oot_df, etrc=etrc_df)


def summarize_dataset(
    splits: DatasetSplits, target_col: str = "decision"
) -> DatasetSummary:
    full_df = pd.concat([splits.train, splits.test, splits.oot, splits.etrc], axis=0)
    feature_cols = [
        col for col in full_df.columns if col not in {target_col, "decision_binary"}
    ]

    feature_types = {
        "numeric": int(full_df[feature_cols].select_dtypes(include=[np.number]).shape[1]),
        "categorical": int(
            full_df[feature_cols].select_dtypes(exclude=[np.number]).shape[1]
        ),
    }

    class_balance: Dict[str, Dict[str, float]] = {}
    for split_name, split_df in {
        "train": splits.train,
        "test": splits.test,
        "oot": splits.oot,
        "etrc": splits.etrc,
    }.items():
        counts = split_df[target_col].value_counts(normalize=True).to_dict()
        class_balance[split_name] = {str(k): float(v) for k, v in counts.items()}

    return DatasetSummary(
        n_rows=int(full_df.shape[0]),
        n_features=len(feature_cols),
        target_col=target_col,
        class_balance=class_balance,
        feature_types=feature_types,
    )
