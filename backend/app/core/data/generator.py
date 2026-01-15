"""
Enhanced Synthetic Financial Dataset Generator.

Generates realistic credit risk and investment decision datasets with:
- 100k+ rows and 100+ features
- Demographic, financial, behavioral, bureau, and time-based variables
- Class imbalance, missing values, correlated features, non-linear effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FinancialDataConfig:
    """Configuration for realistic financial data generation."""

    n_rows: int = 100_000
    seed: int = 42
    default_rate: float = 0.15  # ~15% default rate (class imbalance)
    missing_rate: float = 0.05  # 5% missing values
    noise_factor: float = 0.1

    # Feature counts by category (totaling 1000+ features)
    n_demographic: int = 50
    n_financial: int = 150
    n_behavioral: int = 200
    n_bureau: int = 150
    n_time: int = 100
    n_derived: int = 150
    n_interaction: int = 100
    n_lag: int = 100

    # Decision labels
    decision_labels: Tuple[str, str] = ("default", "no_default")


@dataclass
class DatasetSummary:
    """Summary statistics for generated dataset."""

    n_rows: int
    n_features: int
    target_col: str
    class_balance: Dict[str, Dict[str, float]]
    feature_types: Dict[str, int]
    missing_summary: Dict[str, float] = field(default_factory=dict)


class FinancialDataGenerator:
    """
    Generates realistic credit risk and investment decision datasets.

    Features are grouped into categories:
    - Demographic: age, income, employment, region, education
    - Financial: balances, limits, utilization, debt ratios
    - Behavioral: payment history, delinquencies, usage patterns
    - Bureau: external credit scores, tradelines, inquiries
    - Time-based: account age, vintage, time since events
    - Derived: calculated ratios and composite metrics
    """

    def __init__(self, config: Optional[FinancialDataConfig] = None):
        self.config = config or FinancialDataConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._feature_weights: Dict[str, float] = {}

    def generate(self) -> pd.DataFrame:
        """Generate complete financial dataset with 1000+ features."""
        n = self.config.n_rows

        # Generate core features by category
        demographic_df = self._generate_demographic_features(n)
        financial_df = self._generate_financial_features(n)
        behavioral_df = self._generate_behavioral_features(n)
        bureau_df = self._generate_bureau_features(n)
        time_df = self._generate_time_features(n)

        # Combine core features
        df = pd.concat(
            [demographic_df, financial_df, behavioral_df, bureau_df, time_df],
            axis=1,
        )

        # Generate derived features based on existing ones
        derived_df = self._generate_derived_features(df)
        df = pd.concat([df, derived_df], axis=1)

        # Generate additional scaled and transformed features (to reach 1000+)
        scaled_df = self._generate_scaled_features(df, n)
        df = pd.concat([df, scaled_df], axis=1)

        # Generate interaction features
        interaction_df = self._generate_interaction_features(df, n)
        df = pd.concat([df, interaction_df], axis=1)

        # Generate lag features (simulating time-series behavior)
        lag_df = self._generate_lag_features(df, n)
        df = pd.concat([df, lag_df], axis=1)

        # Generate noise features (to test model robustness)
        noise_df = self._generate_noise_features(n)
        df = pd.concat([df, noise_df], axis=1)

        # Inject correlations between related features
        df = self._inject_correlations(df)

        # Inject missing values with realistic patterns
        df = self._inject_missing_values(df)

        # Compute target variable with non-linear effects
        target = self._compute_target(df)
        df["decision_binary"] = target
        df["decision"] = np.where(
            target == 1,
            self.config.decision_labels[0],
            self.config.decision_labels[1],
        )

        return df

    def _generate_demographic_features(self, n: int) -> pd.DataFrame:
        """Generate demographic features: age, income, employment, region, education."""
        data = {}

        # Age (18-80, skewed toward working age)
        data["age"] = np.clip(
            self.rng.normal(42, 12, n).astype(int), 18, 80
        )

        # Annual income (log-normal distribution, realistic for US)
        data["annual_income"] = np.clip(
            self.rng.lognormal(10.8, 0.8, n), 15000, 500000
        ).astype(int)

        # Monthly income derived
        data["monthly_income"] = data["annual_income"] / 12

        # Employment status (categorical)
        emp_types = ["employed", "self_employed", "unemployed", "retired", "student"]
        emp_probs = [0.65, 0.15, 0.08, 0.10, 0.02]
        data["employment_status"] = self.rng.choice(emp_types, n, p=emp_probs)

        # Employment tenure (months, conditioned on status)
        data["employment_tenure_months"] = np.where(
            np.isin(data["employment_status"], ["employed", "self_employed"]),
            np.clip(self.rng.exponential(48, n), 0, 480).astype(int),
            0,
        )

        # Region (categorical)
        regions = ["northeast", "midwest", "south", "west", "other"]
        region_probs = [0.18, 0.21, 0.38, 0.20, 0.03]
        data["region"] = self.rng.choice(regions, n, p=region_probs)

        # Education level (categorical, ordered)
        edu_levels = ["high_school", "some_college", "bachelors", "masters", "doctorate"]
        edu_probs = [0.28, 0.22, 0.32, 0.14, 0.04]
        data["education_level"] = self.rng.choice(edu_levels, n, p=edu_probs)

        # Home ownership (categorical)
        home_types = ["own", "mortgage", "rent", "other"]
        home_probs = [0.20, 0.35, 0.40, 0.05]
        data["home_ownership"] = self.rng.choice(home_types, n, p=home_probs)

        # Number of dependents
        data["num_dependents"] = self.rng.poisson(1.2, n)
        data["num_dependents"] = np.clip(data["num_dependents"], 0, 8)

        # Marital status
        marital = ["single", "married", "divorced", "widowed"]
        marital_probs = [0.32, 0.48, 0.15, 0.05]
        data["marital_status"] = self.rng.choice(marital, n, p=marital_probs)

        return pd.DataFrame(data)

    def _generate_financial_features(self, n: int) -> pd.DataFrame:
        """Generate financial features: balances, limits, utilization, ratios."""
        data = {}

        # Total credit limit
        data["total_credit_limit"] = np.clip(
            self.rng.lognormal(9.5, 1.0, n), 1000, 200000
        ).astype(int)

        # Credit card balance
        data["credit_card_balance"] = np.clip(
            self.rng.exponential(5000, n), 0, data["total_credit_limit"]
        ).astype(int)

        # Credit utilization ratio
        data["credit_utilization"] = np.clip(
            data["credit_card_balance"] / np.maximum(data["total_credit_limit"], 1),
            0, 1,
        )

        # Revolving balance
        data["revolving_balance"] = np.clip(
            self.rng.lognormal(8.0, 1.2, n), 0, 100000
        ).astype(int)

        # Installment loan balance
        data["installment_balance"] = np.clip(
            self.rng.lognormal(9.0, 1.5, n), 0, 300000
        ).astype(int)

        # Mortgage balance
        data["mortgage_balance"] = np.where(
            self.rng.random(n) < 0.4,
            np.clip(self.rng.lognormal(12.0, 0.8, n), 50000, 800000).astype(int),
            0,
        )

        # Total debt
        data["total_debt"] = (
            data["credit_card_balance"]
            + data["revolving_balance"]
            + data["installment_balance"]
            + data["mortgage_balance"]
        )

        # Monthly debt payment (estimated)
        data["monthly_debt_payment"] = np.clip(
            data["total_debt"] * 0.02, 0, 20000
        ).astype(int)

        # Available credit
        data["available_credit"] = np.maximum(
            data["total_credit_limit"] - data["credit_card_balance"], 0
        )

        # Loan amount (current application)
        data["loan_amount_requested"] = np.clip(
            self.rng.lognormal(9.0, 1.0, n), 1000, 100000
        ).astype(int)

        # Loan term (months)
        terms = [12, 24, 36, 48, 60, 72, 84]
        data["loan_term_months"] = self.rng.choice(terms, n)

        # Interest rate
        data["interest_rate"] = np.clip(
            self.rng.normal(12, 5, n), 3, 30
        )

        # Loan purpose (categorical)
        purposes = [
            "debt_consolidation", "home_improvement", "major_purchase",
            "medical", "car", "vacation", "wedding", "other"
        ]
        purpose_probs = [0.35, 0.15, 0.12, 0.08, 0.12, 0.05, 0.03, 0.10]
        data["loan_purpose"] = self.rng.choice(purposes, n, p=purpose_probs)

        # Number of open accounts
        data["num_open_accounts"] = np.clip(
            self.rng.poisson(8, n), 1, 30
        )

        # Number of credit cards
        data["num_credit_cards"] = np.clip(
            self.rng.poisson(3, n), 0, 15
        )

        # Number of installment loans
        data["num_installment_loans"] = np.clip(
            self.rng.poisson(2, n), 0, 10
        )

        # Monthly minimum payment
        data["min_monthly_payment"] = np.clip(
            data["credit_card_balance"] * 0.02, 25, 5000
        ).astype(int)

        # Checking account balance
        data["checking_balance"] = np.clip(
            self.rng.lognormal(7.5, 1.5, n), 0, 100000
        ).astype(int)

        # Savings account balance
        data["savings_balance"] = np.clip(
            self.rng.lognormal(8.0, 2.0, n), 0, 500000
        ).astype(int)

        # Total liquid assets
        data["total_liquid_assets"] = data["checking_balance"] + data["savings_balance"]

        # Net worth estimate
        data["estimated_net_worth"] = (
            data["total_liquid_assets"] - data["total_debt"]
        )

        # Number of recent credit applications
        data["recent_credit_applications"] = self.rng.poisson(1.5, n)

        # Secured vs unsecured debt ratio
        total_secured = data["mortgage_balance"] + data["installment_balance"] * 0.3
        total_unsecured = data["credit_card_balance"] + data["revolving_balance"]
        data["secured_debt_ratio"] = np.clip(
            total_secured / np.maximum(data["total_debt"], 1), 0, 1
        )

        return pd.DataFrame(data)

    def _generate_behavioral_features(self, n: int) -> pd.DataFrame:
        """Generate behavioral features: payment history, delinquencies, patterns."""
        data = {}

        # Days past due (DPD) counts
        data["dpd_30_count_12m"] = np.clip(
            self.rng.poisson(0.3, n), 0, 12
        )
        data["dpd_60_count_12m"] = np.clip(
            self.rng.poisson(0.1, n), 0, 6
        )
        data["dpd_90_count_12m"] = np.clip(
            self.rng.poisson(0.05, n), 0, 4
        )
        data["dpd_120_plus_count_12m"] = np.clip(
            self.rng.poisson(0.02, n), 0, 3
        )

        # Historical DPD
        data["dpd_30_count_ever"] = data["dpd_30_count_12m"] + np.clip(
            self.rng.poisson(0.5, n), 0, 20
        )
        data["dpd_60_count_ever"] = data["dpd_60_count_12m"] + np.clip(
            self.rng.poisson(0.2, n), 0, 10
        )
        data["dpd_90_count_ever"] = data["dpd_90_count_12m"] + np.clip(
            self.rng.poisson(0.1, n), 0, 6
        )

        # Current delinquency status
        data["currently_delinquent"] = (self.rng.random(n) < 0.08).astype(int)
        data["current_dpd_days"] = np.where(
            data["currently_delinquent"] == 1,
            self.rng.choice([30, 60, 90, 120, 150, 180], n),
            0,
        )

        # Payment history score (0-100)
        base_score = 85 - data["dpd_30_count_ever"] * 3 - data["dpd_60_count_ever"] * 5
        data["payment_history_score"] = np.clip(
            base_score + self.rng.normal(0, 5, n), 0, 100
        ).astype(int)

        # On-time payment rate (last 24 months)
        data["ontime_payment_rate_24m"] = np.clip(
            1 - (data["dpd_30_count_12m"] * 2 + data["dpd_60_count_12m"]) / 24,
            0.5, 1.0,
        )

        # Missed payments in last 6 months
        data["missed_payments_6m"] = np.clip(
            self.rng.poisson(0.2, n), 0, 6
        )

        # Average days to payment
        data["avg_days_to_payment"] = np.clip(
            self.rng.normal(15, 8, n), 0, 60
        ).astype(int)

        # Payment amount consistency (std dev of payment amounts / mean)
        data["payment_consistency_score"] = np.clip(
            self.rng.beta(5, 2, n), 0.3, 1.0
        )

        # Number of NSF (insufficient funds) events
        data["nsf_count_12m"] = np.clip(
            self.rng.poisson(0.3, n), 0, 10
        )

        # Overlimit count
        data["overlimit_count_12m"] = np.clip(
            self.rng.poisson(0.2, n), 0, 8
        )

        # Cash advance usage
        data["cash_advance_count_12m"] = np.clip(
            self.rng.poisson(0.5, n), 0, 15
        )
        data["cash_advance_ratio"] = np.clip(
            self.rng.beta(1, 10, n), 0, 0.5
        )

        # Balance transfer activity
        data["balance_transfer_count_12m"] = np.clip(
            self.rng.poisson(0.3, n), 0, 5
        )

        # Account closures
        data["accounts_closed_12m"] = np.clip(
            self.rng.poisson(0.2, n), 0, 5
        )

        # New accounts opened
        data["new_accounts_12m"] = np.clip(
            self.rng.poisson(1.0, n), 0, 8
        )

        # Average monthly spend
        data["avg_monthly_spend"] = np.clip(
            self.rng.lognormal(7.5, 1.0, n), 100, 50000
        ).astype(int)

        # Spend trend (increasing = 1, stable = 0, decreasing = -1)
        data["spend_trend_6m"] = self.rng.choice([-1, 0, 1], n, p=[0.25, 0.50, 0.25])

        # Utilization trend
        data["utilization_trend_6m"] = np.clip(
            self.rng.normal(0, 0.1, n), -0.3, 0.3
        )

        # Payment method diversity
        data["payment_method_count"] = np.clip(
            self.rng.poisson(2, n), 1, 6
        )

        # Autopay enrollment
        data["has_autopay"] = (self.rng.random(n) < 0.45).astype(int)

        # Digital engagement score
        data["digital_engagement_score"] = np.clip(
            self.rng.beta(3, 2, n) * 100, 0, 100
        ).astype(int)

        # Customer tenure (months)
        data["customer_tenure_months"] = np.clip(
            self.rng.exponential(60, n), 1, 360
        ).astype(int)

        # Number of products held
        data["num_products"] = np.clip(
            self.rng.poisson(2.5, n), 1, 10
        )

        return pd.DataFrame(data)

    def _generate_bureau_features(self, n: int) -> pd.DataFrame:
        """Generate bureau/external credit features."""
        data = {}

        # External credit score (FICO-like, 300-850)
        data["external_credit_score"] = np.clip(
            self.rng.normal(680, 80, n), 300, 850
        ).astype(int)

        # Score band (categorical)
        conditions = [
            data["external_credit_score"] < 580,
            data["external_credit_score"] < 670,
            data["external_credit_score"] < 740,
            data["external_credit_score"] < 800,
        ]
        choices = ["poor", "fair", "good", "very_good"]
        data["score_band"] = np.select(conditions, choices, default="excellent")

        # Number of tradelines
        data["num_tradelines"] = np.clip(
            self.rng.poisson(12, n), 1, 50
        )

        # Age of oldest tradeline (months)
        data["oldest_tradeline_months"] = np.clip(
            self.rng.exponential(120, n), 6, 600
        ).astype(int)

        # Age of newest tradeline (months)
        data["newest_tradeline_months"] = np.clip(
            self.rng.exponential(12, n), 0, data["oldest_tradeline_months"]
        ).astype(int)

        # Average age of tradelines
        data["avg_tradeline_age_months"] = (
            (data["oldest_tradeline_months"] + data["newest_tradeline_months"]) / 2
        ).astype(int)

        # Number of inquiries (last 6 months)
        data["inquiries_6m"] = np.clip(
            self.rng.poisson(1.5, n), 0, 15
        )

        # Number of inquiries (last 12 months)
        data["inquiries_12m"] = data["inquiries_6m"] + np.clip(
            self.rng.poisson(1.0, n), 0, 10
        )

        # Public records
        data["public_records_count"] = np.clip(
            self.rng.poisson(0.05, n), 0, 3
        )

        # Bankruptcies
        data["bankruptcies_count"] = (self.rng.random(n) < 0.02).astype(int)

        # Collections
        data["collections_count"] = np.clip(
            self.rng.poisson(0.1, n), 0, 5
        )
        data["collections_amount"] = np.where(
            data["collections_count"] > 0,
            np.clip(self.rng.lognormal(7, 1, n), 100, 50000).astype(int),
            0,
        )

        # Total credit exposure
        data["total_credit_exposure"] = np.clip(
            self.rng.lognormal(10.5, 1.2, n), 1000, 1000000
        ).astype(int)

        # Derogatory marks
        data["derogatory_count"] = np.clip(
            self.rng.poisson(0.3, n), 0, 10
        )

        # Months since last derogatory
        data["months_since_derogatory"] = np.where(
            data["derogatory_count"] > 0,
            np.clip(self.rng.exponential(24, n), 1, 120).astype(int),
            999,  # No derogatory = 999 months
        )

        # Satisfactory accounts ratio
        data["satisfactory_accounts_ratio"] = np.clip(
            1 - data["derogatory_count"] / np.maximum(data["num_tradelines"], 1),
            0, 1,
        )

        # Revolving credit utilization (bureau reported)
        data["bureau_revolving_utilization"] = np.clip(
            self.rng.beta(2, 5, n), 0, 1
        )

        # Installment credit utilization
        data["bureau_installment_utilization"] = np.clip(
            self.rng.beta(3, 4, n), 0, 1
        )

        # Credit mix score (0-100)
        data["credit_mix_score"] = np.clip(
            self.rng.normal(60, 20, n), 0, 100
        ).astype(int)

        return pd.DataFrame(data)

    def _generate_time_features(self, n: int) -> pd.DataFrame:
        """Generate time-based features."""
        data = {}

        # Reference date for calculations
        ref_date = datetime(2024, 1, 15)

        # Account vintage (months since origination)
        data["account_vintage_months"] = np.clip(
            self.rng.exponential(36, n), 1, 240
        ).astype(int)

        # Origination date
        origination_days = data["account_vintage_months"] * 30
        data["origination_date"] = [
            (ref_date - timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in origination_days
        ]

        # Observation date (current)
        data["observation_date"] = ref_date.strftime("%Y-%m-%d")

        # Time since last payment (days)
        data["days_since_last_payment"] = np.clip(
            self.rng.exponential(20, n), 0, 180
        ).astype(int)

        # Time since last delinquency (months)
        data["months_since_delinq"] = np.clip(
            self.rng.exponential(36, n), 0, 120
        ).astype(int)

        # Time since last credit inquiry (days)
        data["days_since_last_inquiry"] = np.clip(
            self.rng.exponential(60, n), 0, 365
        ).astype(int)

        # Time on books (months) - customer relationship duration
        data["time_on_books_months"] = np.maximum(
            data["account_vintage_months"],
            np.clip(self.rng.exponential(48, n), 1, 300).astype(int),
        )

        # First credit date (months ago)
        data["first_credit_months_ago"] = np.clip(
            self.rng.exponential(120, n), 12, 480
        ).astype(int)

        # Season/quarter of application (for seasonality effects)
        data["application_quarter"] = self.rng.choice([1, 2, 3, 4], n)

        return pd.DataFrame(data)

    def _generate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate derived/calculated features from existing features."""
        data = {}

        # Debt-to-income ratio
        annual_income = df.get("annual_income", pd.Series([50000] * len(df)))
        monthly_income = annual_income / 12
        total_debt = df.get("total_debt", pd.Series([0] * len(df)))
        data["dti_ratio"] = np.clip(total_debt / np.maximum(annual_income, 1), 0, 5)

        # Monthly DTI
        monthly_payment = df.get("monthly_debt_payment", pd.Series([0] * len(df)))
        data["monthly_dti"] = np.clip(
            monthly_payment / np.maximum(monthly_income, 1), 0, 2
        )

        # Payment to income ratio
        data["payment_to_income_ratio"] = np.clip(
            monthly_payment / np.maximum(monthly_income, 1), 0, 1
        )

        # Credit utilization squared (non-linear effect)
        utilization = df.get("credit_utilization", pd.Series([0.3] * len(df)))
        data["utilization_squared"] = utilization ** 2

        # Delinquency intensity (weighted DPD)
        dpd_30 = df.get("dpd_30_count_12m", pd.Series([0] * len(df)))
        dpd_60 = df.get("dpd_60_count_12m", pd.Series([0] * len(df)))
        dpd_90 = df.get("dpd_90_count_12m", pd.Series([0] * len(df)))
        data["delinquency_intensity"] = dpd_30 + dpd_60 * 2 + dpd_90 * 4

        # Risk score composite
        credit_score = df.get("external_credit_score", pd.Series([680] * len(df)))
        payment_score = df.get("payment_history_score", pd.Series([80] * len(df)))
        data["composite_risk_score"] = (
            credit_score * 0.4 + payment_score * 6 * 0.3 + (1 - utilization) * 850 * 0.3
        )

        # Velocity: new accounts / tenure
        new_accounts = df.get("new_accounts_12m", pd.Series([1] * len(df)))
        tenure = df.get("customer_tenure_months", pd.Series([12] * len(df)))
        data["account_velocity"] = new_accounts / np.maximum(tenure / 12, 1)

        # Liquidity ratio
        liquid = df.get("total_liquid_assets", pd.Series([5000] * len(df)))
        data["liquidity_ratio"] = np.clip(
            liquid / np.maximum(total_debt, 1), 0, 10
        )

        # Credit age quality score
        oldest = df.get("oldest_tradeline_months", pd.Series([60] * len(df)))
        data["credit_age_score"] = np.clip(oldest / 12, 0, 30)

        # Inquiry intensity
        inquiries = df.get("inquiries_6m", pd.Series([1] * len(df)))
        data["inquiry_intensity"] = np.clip(inquiries / 6, 0, 3)

        return pd.DataFrame(data)

    def _inject_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic correlations between features."""
        df = df.copy()

        # Higher income correlates with higher credit limit
        if "annual_income" in df.columns and "total_credit_limit" in df.columns:
            income_effect = (df["annual_income"] - df["annual_income"].mean()) / df["annual_income"].std()
            df["total_credit_limit"] = np.clip(
                df["total_credit_limit"] + income_effect * 10000,
                1000, 200000,
            ).astype(int)

        # Higher credit score correlates with lower utilization
        if "external_credit_score" in df.columns and "credit_utilization" in df.columns:
            score_effect = (df["external_credit_score"] - 680) / 100
            df["credit_utilization"] = np.clip(
                df["credit_utilization"] - score_effect * 0.1,
                0, 1,
            )

        # Age correlates with credit history length
        if "age" in df.columns and "first_credit_months_ago" in df.columns:
            df["first_credit_months_ago"] = np.clip(
                (df["age"] - 18) * 12 * 0.7 + self.rng.normal(0, 24, len(df)),
                12, 480,
            ).astype(int)

        return df

    def _inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject missing values with realistic patterns (MCAR, MAR)."""
        df = df.copy()
        n = len(df)

        # MCAR (Missing Completely at Random) - random 2% for some numeric cols
        mcar_cols = [
            "savings_balance", "checking_balance", "avg_monthly_spend",
            "digital_engagement_score",
        ]
        for col in mcar_cols:
            if col in df.columns:
                mask = self.rng.random(n) < 0.02
                df.loc[mask, col] = np.nan

        # MAR (Missing at Random) - missing based on other features
        # Income missing more often for unemployed
        if "annual_income" in df.columns and "employment_status" in df.columns:
            unemployed_mask = df["employment_status"] == "unemployed"
            random_mask = self.rng.random(n) < 0.15
            df.loc[unemployed_mask & random_mask, "annual_income"] = np.nan

        # Bureau features missing for thin file customers
        if "num_tradelines" in df.columns:
            thin_file = df["num_tradelines"] < 3
            for col in ["oldest_tradeline_months", "credit_mix_score"]:
                if col in df.columns:
                    random_mask = self.rng.random(n) < 0.3
                    df.loc[thin_file & random_mask, col] = np.nan

        return df

    def _compute_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute binary target variable using non-linear relationships.

        Target represents default/no-default or invest/no-invest decision.
        """
        n = len(df)

        # Base probability from key risk factors
        log_odds = np.zeros(n)

        # Credit score effect (non-linear)
        if "external_credit_score" in df.columns:
            score = df["external_credit_score"].fillna(680).values
            log_odds -= (score - 600) / 50  # Higher score = lower risk

        # Utilization effect (non-linear, high utilization is risky)
        if "credit_utilization" in df.columns:
            util = df["credit_utilization"].fillna(0.3).values
            log_odds += util * 2 + (util > 0.8) * 1.5  # Jump at 80%

        # DTI effect
        if "dti_ratio" in df.columns:
            dti = df["dti_ratio"].fillna(0.3).values
            log_odds += dti * 0.8

        # Delinquency effect (strong predictor)
        if "dpd_30_count_12m" in df.columns:
            dpd = df["dpd_30_count_12m"].fillna(0).values
            log_odds += dpd * 0.6

        if "dpd_90_count_12m" in df.columns:
            dpd90 = df["dpd_90_count_12m"].fillna(0).values
            log_odds += dpd90 * 1.2

        # Payment history effect
        if "payment_history_score" in df.columns:
            pmt_score = df["payment_history_score"].fillna(80).values
            log_odds -= (pmt_score - 50) / 25

        # Income effect (protective)
        if "annual_income" in df.columns:
            income = df["annual_income"].fillna(50000).values
            log_odds -= np.log1p(income) / 5

        # Tenure effect (longer tenure = lower risk)
        if "customer_tenure_months" in df.columns:
            tenure = df["customer_tenure_months"].fillna(24).values
            log_odds -= np.log1p(tenure) / 4

        # Add noise
        noise = self.rng.normal(0, self.config.noise_factor, n)
        log_odds += noise

        # Convert to probability
        probability = 1 / (1 + np.exp(-log_odds))

        # Set threshold to achieve target default rate
        threshold = np.quantile(probability, 1 - self.config.default_rate)
        target = (probability > threshold).astype(int)

        return target

    def _generate_scaled_features(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate scaled and transformed versions of existing features to reach 1000+."""
        data = {}

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Generate multiple transformations for each numeric feature
        for i, col in enumerate(numeric_cols[:100]):  # Limit to first 100 for performance
            base_vals = df[col].fillna(df[col].median()).values

            # Log transform
            data[f"{col}_log"] = np.log1p(np.abs(base_vals))

            # Square root transform
            data[f"{col}_sqrt"] = np.sqrt(np.abs(base_vals))

            # Z-score normalized
            mean_val = np.mean(base_vals)
            std_val = np.std(base_vals) + 1e-8
            data[f"{col}_zscore"] = (base_vals - mean_val) / std_val

            # Percentile rank
            data[f"{col}_pctl"] = (np.argsort(np.argsort(base_vals)) / n)

            # Binned versions (deciles)
            data[f"{col}_decile"] = pd.qcut(
                base_vals, q=10, labels=False, duplicates="drop"
            )

        # Rolling statistics features (simulated)
        key_features = [
            "credit_utilization", "annual_income", "total_debt",
            "external_credit_score", "payment_history_score"
        ]
        for col in key_features:
            if col in df.columns:
                base = df[col].fillna(df[col].median()).values
                # Simulated rolling averages at different windows
                for window in [3, 6, 12]:
                    noise = self.rng.normal(0, 0.05, n)
                    data[f"{col}_avg_{window}m"] = base * (1 + noise)
                    data[f"{col}_max_{window}m"] = base * (1 + np.abs(noise))
                    data[f"{col}_min_{window}m"] = base * (1 - np.abs(noise))
                    data[f"{col}_std_{window}m"] = np.abs(base * noise * 0.2)

        return pd.DataFrame(data)

    def _generate_interaction_features(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate interaction features between key variables."""
        data = {}

        # Key feature pairs for interactions
        interaction_pairs = [
            ("annual_income", "total_debt"),
            ("credit_utilization", "external_credit_score"),
            ("age", "customer_tenure_months"),
            ("num_tradelines", "inquiries_12m"),
            ("payment_history_score", "dpd_30_count_12m"),
            ("total_credit_limit", "credit_card_balance"),
            ("monthly_income", "monthly_debt_payment"),
            ("savings_balance", "checking_balance"),
            ("num_open_accounts", "num_credit_cards"),
            ("account_vintage_months", "first_credit_months_ago"),
        ]

        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                v1 = df[col1].fillna(0).values.astype(float)
                v2 = df[col2].fillna(0).values.astype(float)

                # Multiplicative interaction
                data[f"{col1}_x_{col2}"] = v1 * v2

                # Ratio interaction
                data[f"{col1}_div_{col2}"] = v1 / np.maximum(v2, 1e-8)

                # Sum
                data[f"{col1}_plus_{col2}"] = v1 + v2

                # Difference
                data[f"{col1}_minus_{col2}"] = v1 - v2

                # Max
                data[f"{col1}_max_{col2}"] = np.maximum(v1, v2)

                # Min
                data[f"{col1}_min_{col2}"] = np.minimum(v1, v2)

        # Three-way interactions for key risk factors
        if all(c in df.columns for c in ["credit_utilization", "dti_ratio", "external_credit_score"]):
            util = df["credit_utilization"].fillna(0.3).values
            dti = df["dti_ratio"].fillna(0.3).values
            score = df["external_credit_score"].fillna(680).values

            data["risk_triple_product"] = util * dti * (850 - score) / 850
            data["risk_weighted_sum"] = util * 0.3 + dti * 0.3 + (850 - score) / 850 * 0.4

        # Polynomial features for key variables
        poly_cols = ["credit_utilization", "dti_ratio", "inquiries_6m"]
        for col in poly_cols:
            if col in df.columns:
                v = df[col].fillna(0).values
                data[f"{col}_squared"] = v ** 2
                data[f"{col}_cubed"] = v ** 3

        return pd.DataFrame(data)

    def _generate_lag_features(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate lag features simulating time-series behavior."""
        data = {}

        # Key features to create lag versions for
        lag_features = [
            "credit_utilization", "total_debt", "credit_card_balance",
            "payment_history_score", "dpd_30_count_12m", "avg_monthly_spend",
            "external_credit_score", "num_open_accounts", "inquiries_6m"
        ]

        lag_periods = [1, 3, 6, 12]  # months

        for col in lag_features:
            if col in df.columns:
                base = df[col].fillna(df[col].median()).values

                for lag in lag_periods:
                    # Simulated lag with random walk
                    drift = self.rng.normal(0, 0.02 * lag, n)
                    data[f"{col}_lag_{lag}m"] = base * (1 + drift)

                    # Change vs lag
                    data[f"{col}_chg_{lag}m"] = base * drift

                    # Percent change
                    data[f"{col}_pct_chg_{lag}m"] = drift

        # Trend features
        for col in ["credit_utilization", "total_debt", "external_credit_score"]:
            if col in df.columns:
                base = df[col].fillna(df[col].median()).values
                # Simulated trend direction
                data[f"{col}_trend_3m"] = self.rng.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3])
                data[f"{col}_trend_6m"] = self.rng.choice([-1, 0, 1], n, p=[0.25, 0.5, 0.25])
                data[f"{col}_momentum"] = self.rng.normal(0, 0.1, n)

        return pd.DataFrame(data)

    def _generate_noise_features(self, n: int) -> pd.DataFrame:
        """Generate noise features to test model robustness and feature selection."""
        data = {}

        # Pure random noise features (should have no predictive power)
        for i in range(100):
            # Normal distribution noise
            data[f"noise_normal_{i:03d}"] = self.rng.normal(0, 1, n)

        for i in range(50):
            # Uniform distribution noise
            data[f"noise_uniform_{i:03d}"] = self.rng.uniform(0, 1, n)

        for i in range(30):
            # Categorical noise
            data[f"noise_cat_{i:03d}"] = self.rng.choice(
                ["A", "B", "C", "D", "E"], n
            )

        for i in range(20):
            # Binary noise
            data[f"noise_binary_{i:03d}"] = self.rng.choice([0, 1], n)

        # Correlated noise (noise that correlates with other noise but not target)
        base_noise = self.rng.normal(0, 1, n)
        for i in range(50):
            correlation = self.rng.uniform(0.3, 0.9)
            noise = self.rng.normal(0, 1, n)
            data[f"noise_corr_{i:03d}"] = (
                correlation * base_noise + np.sqrt(1 - correlation**2) * noise
            )

        return pd.DataFrame(data)

    def get_feature_metadata(self) -> Dict[str, Dict]:
        """Return metadata about generated features."""
        return {
            "demographic": {
                "features": [
                    "age", "annual_income", "monthly_income", "employment_status",
                    "employment_tenure_months", "region", "education_level",
                    "home_ownership", "num_dependents", "marital_status",
                ],
                "description": "Customer demographic information",
            },
            "financial": {
                "features": [
                    "total_credit_limit", "credit_card_balance", "credit_utilization",
                    "revolving_balance", "installment_balance", "mortgage_balance",
                    "total_debt", "monthly_debt_payment", "available_credit",
                    "loan_amount_requested", "loan_term_months", "interest_rate",
                    "loan_purpose", "num_open_accounts", "num_credit_cards",
                    "num_installment_loans", "min_monthly_payment", "checking_balance",
                    "savings_balance", "total_liquid_assets", "estimated_net_worth",
                    "recent_credit_applications", "secured_debt_ratio",
                ],
                "description": "Financial accounts and debt information",
            },
            "behavioral": {
                "features": [
                    "dpd_30_count_12m", "dpd_60_count_12m", "dpd_90_count_12m",
                    "dpd_120_plus_count_12m", "dpd_30_count_ever", "dpd_60_count_ever",
                    "dpd_90_count_ever", "currently_delinquent", "current_dpd_days",
                    "payment_history_score", "ontime_payment_rate_24m",
                    "missed_payments_6m", "avg_days_to_payment",
                    "payment_consistency_score", "nsf_count_12m", "overlimit_count_12m",
                    "cash_advance_count_12m", "cash_advance_ratio",
                    "balance_transfer_count_12m", "accounts_closed_12m",
                    "new_accounts_12m", "avg_monthly_spend", "spend_trend_6m",
                    "utilization_trend_6m", "payment_method_count", "has_autopay",
                    "digital_engagement_score", "customer_tenure_months", "num_products",
                ],
                "description": "Payment and usage behavior patterns",
            },
            "bureau": {
                "features": [
                    "external_credit_score", "score_band", "num_tradelines",
                    "oldest_tradeline_months", "newest_tradeline_months",
                    "avg_tradeline_age_months", "inquiries_6m", "inquiries_12m",
                    "public_records_count", "bankruptcies_count", "collections_count",
                    "collections_amount", "total_credit_exposure", "derogatory_count",
                    "months_since_derogatory", "satisfactory_accounts_ratio",
                    "bureau_revolving_utilization", "bureau_installment_utilization",
                    "credit_mix_score",
                ],
                "description": "External credit bureau data",
            },
            "time": {
                "features": [
                    "account_vintage_months", "origination_date", "observation_date",
                    "days_since_last_payment", "months_since_delinq",
                    "days_since_last_inquiry", "time_on_books_months",
                    "first_credit_months_ago", "application_quarter",
                ],
                "description": "Time-based and vintage features",
            },
            "derived": {
                "features": [
                    "dti_ratio", "monthly_dti", "payment_to_income_ratio",
                    "utilization_squared", "delinquency_intensity",
                    "composite_risk_score", "account_velocity", "liquidity_ratio",
                    "credit_age_score", "inquiry_intensity",
                ],
                "description": "Calculated and composite features",
            },
        }
