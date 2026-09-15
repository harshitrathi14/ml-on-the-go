"""
Generate a realistic multi-source test set for Nu Score On the Go:

  loan_dump.csv        one row per loan: loan features, disbursal date, default flag,
                       plus deliberately PII-looking and leaky columns to exercise the checks
  bureau.json          bureau pull per loan (nested JSON with trade lines), keyed by
                       a differently named identifier
  bank_statement.csv   monthly bank-statement variables, several rows per loan

Usage: python input/generate_multisource.py [out_dir] [n_loans]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out_dir: Path, n: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    loan_ids = np.array([f"LN{100000 + i}" for i in range(n)])
    disbursal = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    amount = rng.lognormal(11, 0.6, n).round(-2)
    tenure = rng.choice([12, 18, 24, 36, 48], n)
    rate = rng.normal(18, 3, n).round(2)
    income = rng.lognormal(10.3, 0.5, n).round(-2)
    ltv = rng.uniform(0.3, 0.95, n).round(3)
    product = rng.choice(["two_wheeler", "used_car", "personal", "msme"], n, p=[0.3, 0.2, 0.3, 0.2])
    city_tier = rng.choice(["tier1", "tier2", "tier3"], n)
    pincode = rng.integers(400001, 999999, n)

    bureau_score = np.clip(rng.normal(690, 60, n), 300, 900).round()
    enquiries = rng.poisson(2, n)
    live_tl = rng.poisson(3, n)
    worst_dpd = rng.choice([0, 0, 0, 30, 60, 90], n)
    avg_balance = rng.lognormal(9.5, 0.9, n).round()
    bounces = rng.poisson(0.4, n)

    logit = (-3.2 - 0.012 * (bureau_score - 690) + 0.25 * enquiries + 0.012 * worst_dpd
             + 2.0 * ltv - 0.4 * np.log1p(avg_balance / 1000) + 0.6 * bounces
             + 0.15 * (product == "personal") + rng.normal(0, 0.8, n))
    p = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(size=n) < p).astype(int)

    loans = pd.DataFrame({
        "loan_id": loan_ids,
        "customer_name": [f"Customer {i}" for i in range(n)],          # PII by name
        "mobile": [f"9{rng.integers(100000000, 999999999)}" for _ in range(n)],  # PII by value
        "disbursal_date": disbursal.strftime("%Y-%m-%d"),
        "product": product,
        "loan_amount": amount,
        "tenure_months": tenure,
        "interest_rate": rate,
        "emi": (amount * (rate / 1200) / (1 - (1 + rate / 1200) ** (-tenure))).round(),
        "ltv": ltv,
        "monthly_income": income,
        "city_tier": city_tier,
        "pincode": pincode,
        "writeoff_amount": np.where(default == 1, amount * rng.uniform(0.3, 0.9, n), 0).round(),  # leakage
        "default_90dpd": default,
    })
    loans = loans.sample(frac=1, random_state=seed)
    loans.to_csv(out_dir / "loan_dump.csv", index=False)

    # Bureau: 92% coverage, different key name, nested trade lines.
    bureau = []
    for i in rng.permutation(n)[: int(n * 0.92)]:
        trade_lines = [
            {"type": str(rng.choice(["CC", "PL", "AL", "HL"])), "sanctioned": round(float(rng.lognormal(11, 0.8)), -2),
             "outstanding": round(float(rng.lognormal(10, 0.9)), -2), "dpd": int(rng.choice([0, 0, 0, 30, 60]))}
            for _ in range(int(live_tl[i]) or 1)
        ]
        bureau.append({
            "ref_no": loan_ids[i],
            "pull_date": (disbursal[i] - pd.Timedelta(days=int(rng.integers(1, 10)))).strftime("%Y-%m-%d"),
            "score": {"value": int(bureau_score[i]), "version": "v3"},
            "enquiries": {"last_3m": int(enquiries[i]), "last_12m": int(enquiries[i] + rng.poisson(2))},
            "worst_dpd_24m": int(worst_dpd[i]),
            "trade_lines": trade_lines,
        })
    (out_dir / "bureau.json").write_text(json.dumps(bureau))

    # Bank statement: 6 monthly rows per loan for 85% of loans, key "account_number".
    rows = []
    for i in rng.permutation(n)[: int(n * 0.85)]:
        for m in range(6):
            rows.append({
                "account_number": loan_ids[i],
                "statement_month": (disbursal[i] - pd.DateOffset(months=m + 1)).strftime("%Y-%m"),
                "avg_balance": float((avg_balance[i] * rng.uniform(0.7, 1.3)).round()),
                "min_balance": float((avg_balance[i] * rng.uniform(0.1, 0.5)).round()),
                "credits_count": int(rng.poisson(12)),
                "debits_count": int(rng.poisson(25)),
                "salary_credit": float(income[i] * rng.uniform(0.9, 1.1)),
                "bounces": int(bounces[i] if m == 0 else rng.poisson(0.2)),
            })
    pd.DataFrame(rows).to_csv(out_dir / "bank_statement.csv", index=False)
    print(f"wrote {n} loans, {len(bureau)} bureau records, {len(rows)} bank rows to {out_dir} "
          f"(default rate {default.mean():.1%})")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/multisource")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    main(out, count)
