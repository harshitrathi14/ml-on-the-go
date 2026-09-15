"""
Plain-English glossary for variables: what a high / low value means and
why a credit or business analyst cares. Credit-risk patterns first, then
the retail/general patterns shared with the report generator.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..reporting import _FEAT_PATTERNS as _RETAIL_PATTERNS

# (name fragment, label, high means, low means, why it matters)
_CREDIT_PATTERNS: List[Tuple[str, str, str, str, str]] = [
    ("bureau_score", "Bureau score", "strong repayment track record", "thin or damaged credit history",
     "The bureau score condenses past repayment behaviour; it is usually the single strongest driver of default risk."),
    ("cibil", "Bureau score", "strong repayment track record", "thin or damaged credit history",
     "The bureau score condenses past repayment behaviour; it is usually the single strongest driver of default risk."),
    ("enquir", "Credit enquiries", "shopping for credit at several lenders", "no recent credit seeking",
     "Many enquiries in a short window signal credit hunger or being declined elsewhere."),
    ("worst_dpd", "Worst days past due", "has been seriously late before", "always paid on time",
     "Past delinquency is the best predictor of future delinquency."),
    ("dpd", "Days past due", "payments are running late", "payments are current",
     "Delinquency depth (30/60/90 DPD) tracks how far a loan has slipped."),
    ("months_since_last_dpd", "Time since last delinquency", "clean for a long time", "delinquent recently",
     "Recency of the last missed payment matters more than how long ago the account opened."),
    ("utilis", "Credit utilisation", "revolving limits nearly maxed out", "plenty of unused limit",
     "High utilisation means the borrower is stretched and has little headroom for shocks."),
    ("utiliz", "Credit utilisation", "revolving limits nearly maxed out", "plenty of unused limit",
     "High utilisation means the borrower is stretched and has little headroom for shocks."),
    ("foir", "Fixed obligation to income", "most income already committed to EMIs", "comfortable surplus after obligations",
     "FOIR is the affordability test: above ~50–60% leaves no buffer for income dips."),
    ("ltv", "Loan to value", "little equity in the asset", "large down payment / margin",
     "Higher LTV raises loss given default and lowers the borrower's incentive to keep paying."),
    ("down_payment", "Down payment", "borrower has real skin in the game", "financed almost entirely",
     "A larger down payment signals savings discipline and reduces loss on repossession."),
    ("emi", "EMI", "large monthly instalment", "small monthly instalment",
     "The instalment relative to income drives affordability and stress in bad months."),
    ("existing_emi", "Existing EMIs", "already servicing other loans", "no other loan obligations",
     "Existing obligations compete with the new loan for the same cash flow."),
    ("bounce", "Cheque / EMI bounces", "payments have been dishonoured", "no bounces",
     "Bounces are a direct, near-term signal of cash-flow trouble."),
    ("salary_credit", "Salary credit regularity", "salary lands reliably every month", "irregular or missing salary credits",
     "Regular salary credits confirm stable income; gaps often precede default."),
    ("avg_bank_balance", "Average bank balance", "healthy cash buffer", "lives close to zero",
     "Average balance is a behavioural income proxy independent of declared income."),
    ("min_bank_balance", "Minimum bank balance", "never dips near zero", "regularly runs the account down",
     "The monthly low point shows how thin the buffer gets before payday."),
    ("balance", "Bank balance", "healthy cash buffer", "little cash on hand",
     "Bank balances reveal actual liquidity, not just declared income."),
    ("cash_withdrawal", "Cash withdrawal share", "mostly cash-based spending", "mostly digital spending",
     "Heavy cash withdrawal can hide undisclosed obligations or informal income."),
    ("monthly_income", "Monthly income", "higher earning capacity", "lower earning capacity",
     "Income sets the repayment capacity; it matters most relative to obligations (FOIR)."),
    ("income", "Income", "higher earning capacity", "lower earning capacity",
     "Income sets the repayment capacity; it matters most relative to obligations."),
    ("loan_amount", "Loan amount", "large exposure", "small ticket",
     "Ticket size relative to income and asset value drives both risk and loss severity."),
    ("tenure", "Tenure", "long repayment period", "short repayment period",
     "Longer tenures lower the EMI but extend the window in which life events can derail repayment."),
    ("interest_rate", "Interest rate", "priced as higher risk", "priced as prime",
     "The rate already embeds the lender's risk view; very high rates also raise the EMI burden."),
    ("vintage", "Bureau vintage", "long credit history", "new to credit",
     "A longer history makes the bureau score more reliable; new-to-credit borrowers are harder to assess."),
    ("trade_line", "Trade lines", "many open credit accounts", "few or no accounts",
     "The number of open accounts shows how much credit the borrower is juggling."),
    ("outstanding", "Total outstanding", "large existing debt", "little existing debt",
     "Existing debt load competes with the new loan for repayment capacity."),
    ("unsecured", "Unsecured share", "mostly unsecured borrowing", "mostly secured borrowing",
     "Unsecured-heavy profiles tend to be riskier and more rate-sensitive."),
    ("overdue", "Overdue amount", "money is past due now", "nothing overdue",
     "Current overdue amounts are the most immediate distress signal."),
    ("writeoff", "Write-off", "previous written-off loans", "no write-offs",
     "A past write-off is a severe negative; if it refers to this loan it is leakage."),
    ("settle", "Settlement", "previous settled-for-less accounts", "no settlements",
     "Settlements indicate the borrower could not repay in full before."),
    ("employment", "Employment type", "—", "—",
     "Salaried, self-employed and casual earners have very different income stability."),
    ("residence", "Residence type", "—", "—",
     "Owned residence tends to indicate stability; rented or family accommodation less so."),
    ("city_tier", "City tier", "—", "—",
     "Geography proxies for income stability, job markets and collection reach."),
    ("pincode", "Pincode", "—", "—",
     "Location captures local economic conditions and collection feasibility."),
    ("product", "Product", "—", "—",
     "Product type carries its own risk profile (e.g. gold loans vs unsecured personal loans)."),
    ("channel", "Sourcing channel", "—", "—",
     "DSA, dealer, branch and digital channels differ in application quality."),
]


def _clean(name: str) -> str:
    for prefix in ("num__", "cat__", "te__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Joined sources carry a role prefix: "bureau__score_value".
    for role in ("loan_dump__", "bureau__", "bank_statement__", "other__"):
        if name.startswith(role):
            name = name[len(role):]
    return name


def humanise(name: str) -> str:
    name = re.sub(r"__(te|month|year|days_before_ref)$", " (\\1)", _clean(name))
    return name.replace("_", " ").replace(".", " › ").strip().capitalize()


def describe(name: str) -> Dict[str, str]:
    lower = _clean(name).lower()
    for pattern, label, high, low, note in _CREDIT_PATTERNS + list(_RETAIL_PATTERNS):
        if pattern in lower:
            return {"feature": name, "label": label, "high": high, "low": low, "why": note}
    return {"feature": name, "label": humanise(name), "high": "higher values", "low": "lower values",
            "why": "No glossary entry; interpret from the SHAP direction and the IV bins."}


def build(features: List[str]) -> Dict[str, Dict[str, str]]:
    return {f: describe(f) for f in features}
