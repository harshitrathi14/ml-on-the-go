"""
Sample dataset generator for ML on the Go.

Produces three realistic CSV files in this folder:
  1. small_shop_sales_2yr.csv      — 5,840 rows × 76 cols
  2. online_seller_inventory_1yr.csv — 5,200 rows × 38 cols
  3. customer_churn_1yr.csv          — 4,000 rows × 31 cols

Run:
    python input/generate_samples.py
"""

import os
import textwrap
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))


# ── helpers ────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(HERE, name)
    df.to_csv(path, index=False)
    print(f"  ✓  {name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

def add_nulls(df: pd.DataFrame, cols: list, rate: float = 0.04) -> pd.DataFrame:
    """Scatter realistic missing values into selected columns."""
    for col in cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df

def date_range(start: str, n: int) -> list:
    d = date.fromisoformat(start)
    return [d + timedelta(days=i) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# DATASET 1 — Small Retail Shop Sales  (2 years, transaction level)
# Target: high_value_transaction  (1 if sale amount > category median)
# ══════════════════════════════════════════════════════════════════════════

def gen_shop_sales() -> pd.DataFrame:
    N = 5840   # ~8 transactions / day × 730 days

    START = date(2022, 7, 1)
    all_dates = date_range("2022-07-01", 730)

    # ── date scaffold ──────────────────────────────────────────────────────
    row_dates  = rng.choice(all_dates, size=N)
    row_dates  = sorted(row_dates)

    dates      = pd.to_datetime(row_dates)
    dow        = dates.dayofweek            # 0=Mon
    month      = dates.month
    quarter    = dates.quarter
    woy        = dates.isocalendar().week.astype(int)
    is_weekend = (dow >= 5).astype(int)
    is_holiday = ((month == 10) | (month == 11) |
                  ((month == 1) & (dates.day <= 3)) |
                  (dates.day == 25)).astype(int)   # rough proxy

    season_map = {12:"winter",1:"winter",2:"winter",
                  3:"spring",4:"spring",5:"spring",
                  6:"summer",7:"summer",8:"summer",
                  9:"autumn",10:"autumn",11:"autumn"}
    season     = pd.array([season_map[m] for m in month])

    hour       = rng.integers(9, 21, N)
    days_since = np.array([(d - date(2022, 7, 1)).days for d in row_dates])

    # ── categories & base prices ────────────────────────────────────────────
    categories  = ["food","clothing","electronics","stationery",
                   "home","beauty","toys"]
    cat_weights = [0.30, 0.20, 0.12, 0.12, 0.10, 0.10, 0.06]
    cat_base    = {"food":350,"clothing":900,"electronics":2800,
                   "stationery":250,"home":1100,"beauty":600,"toys":700}

    primary_cat = rng.choice(categories, size=N, p=cat_weights)
    cat_base_v  = np.array([cat_base[c] for c in primary_cat])

    # ── customer types ──────────────────────────────────────────────────────
    cust_types = rng.choice(["new","returning","loyal","vip"],
                             size=N, p=[0.30, 0.35, 0.25, 0.10])
    cust_mult  = {"new":0.85,"returning":1.0,"loyal":1.15,"vip":1.35}
    cm         = np.array([cust_mult[c] for c in cust_types])

    cust_age   = rng.choice(["teen","young_adult","adult","senior"],
                             size=N, p=[0.08, 0.30, 0.45, 0.17])
    cust_gender= rng.choice(["M","F","unknown"],
                             size=N, p=[0.44, 0.50, 0.06])

    tenure_map = {"new":0,"returning":180,"loyal":540,"vip":1200}
    tenure_days= np.array([rng.integers(0, tenure_map[c]+1) for c in cust_types])

    ltv_map    = {"new":0,"returning":4000,"loyal":15000,"vip":50000}
    lifetime_v = np.array([rng.integers(0, ltv_map[c]+1)
                            if ltv_map[c] > 0 else 0 for c in cust_types])

    # ── transaction numerics ────────────────────────────────────────────────
    num_items        = rng.integers(1, 18, N)
    discount_pct     = np.where(
        rng.random(N) < 0.35,
        rng.uniform(5, 35, N), 0.0)
    coupon_applied   = (discount_pct > 0).astype(int)
    promo_code       = (rng.random(N) < 0.20).astype(int)

    total_amount_raw = (cat_base_v * cm
                        * (1 + num_items * 0.07)
                        * (1 - discount_pct / 100)
                        * np.where(is_weekend, 1.12, 1.0)
                        * np.where(is_holiday, 1.25, 1.0)
                        * (1 + rng.normal(0, 0.18, N)))
    total_amount     = np.clip(total_amount_raw, 50, 25000).round(2)

    avg_item_price   = (total_amount / num_items).round(2)
    max_item_price   = (avg_item_price * rng.uniform(1.0, 2.5, N)).round(2)
    tax_amount       = (total_amount * 0.18).round(2)
    net_amount       = (total_amount - (total_amount * discount_pct / 100)).round(2)

    loyalty_points_used   = np.where(cust_types != "new",
                                     rng.integers(0, 500, N), 0)
    loyalty_points_earned = (total_amount / 100).astype(int)

    payment_method = rng.choice(["cash","card","upi","wallet"],
                                 size=N, p=[0.22, 0.30, 0.38, 0.10])
    txn_type       = rng.choice(["sale","return","exchange"],
                                 size=N, p=[0.92, 0.05, 0.03])
    receipt_issued = (rng.random(N) < 0.88).astype(int)

    # ── product breakdown ────────────────────────────────────────────────────
    food_cnt       = np.where(primary_cat=="food",  rng.integers(1,8,N), rng.integers(0,4,N))
    clothing_cnt   = np.where(primary_cat=="clothing", rng.integers(1,5,N), rng.integers(0,3,N))
    elec_cnt       = np.where(primary_cat=="electronics", rng.integers(1,4,N), rng.integers(0,2,N))
    stat_cnt       = np.where(primary_cat=="stationery", rng.integers(1,6,N), rng.integers(0,3,N))
    home_cnt       = np.where(primary_cat=="home", rng.integers(1,5,N), rng.integers(0,2,N))
    beauty_cnt     = np.where(primary_cat=="beauty", rng.integers(1,5,N), rng.integers(0,3,N))
    toys_cnt       = np.where(primary_cat=="toys", rng.integers(1,4,N), rng.integers(0,2,N))

    num_cats       = (food_cnt>0).astype(int) + (clothing_cnt>0).astype(int) + \
                     (elec_cnt>0).astype(int) + (stat_cnt>0).astype(int) + \
                     (home_cnt>0).astype(int) + (beauty_cnt>0).astype(int) + \
                     (toys_cnt>0).astype(int)
    num_cats       = np.clip(num_cats, 1, 5)

    perishable_pct  = np.clip((food_cnt / num_items * 100).round(1), 0, 100)
    branded_pct     = rng.uniform(20, 80, N).round(1)
    private_lbl_pct = (100 - branded_pct - rng.uniform(5, 20, N)).clip(0).round(1)
    high_margin_cnt = rng.integers(0, 5, N)
    low_margin_cnt  = rng.integers(0, 8, N)
    items_on_promo  = (num_items * rng.uniform(0, 0.4, N)).astype(int)

    # ── staff / operations ──────────────────────────────────────────────────
    staff_on_duty        = rng.integers(2, 9, N)
    cashier_exp_yrs      = rng.uniform(0, 10, N).round(1)
    queue_length         = rng.integers(0, 20, N)
    time_in_store_min    = rng.integers(5, 65, N)
    checkout_time_sec    = rng.integers(30, 300, N)
    stock_replenished    = (rng.random(N) < 0.20).astype(int)
    store_footfall       = (200 + rng.normal(0, 40, N)
                             + is_weekend * 80
                             + is_holiday * 120).clip(50, 600).astype(int)
    promoter_present     = (rng.random(N) < 0.15).astype(int)

    # ── external / environment ───────────────────────────────────────────────
    weather       = rng.choice(["sunny","cloudy","rainy","stormy"],
                                size=N, p=[0.45, 0.30, 0.20, 0.05])
    temperature   = np.where(
        pd.Series(season).isin(["summer"]), rng.uniform(30, 42, N),
        np.where(pd.Series(season).isin(["winter"]), rng.uniform(10, 20, N),
                 rng.uniform(20, 32, N))).round(1)
    local_event      = (rng.random(N) < 0.08).astype(int)
    competitor_sale  = (rng.random(N) < 0.12).astype(int)
    payday_week      = ((woy % 4 == 0) | (woy % 4 == 1)).astype(int)
    festival_period  = (is_holiday | (month == 10) | (month == 4)).astype(int)
    school_term      = ((month >= 6) & (month <= 11)).astype(int)
    transport_disrup = (rng.random(N) < 0.04).astype(int)
    construction     = (rng.random(N) < 0.06).astype(int)
    social_promo     = (rng.random(N) < 0.18).astype(int)

    # ── financial context ────────────────────────────────────────────────────
    daily_target         = 30000 + rng.normal(0, 5000, N)
    daily_rev_so_far     = rng.uniform(0.1, 0.9, N) * daily_target
    daily_txn_so_far     = rng.integers(1, 50, N)
    daily_tgt_progress   = (daily_rev_so_far / daily_target * 100).round(1)
    operating_cost_daily = rng.uniform(4000, 8000, N).round(0)
    gross_margin_pct     = np.where(primary_cat == "electronics",
                                    rng.uniform(8, 15, N),
                                    rng.uniform(18, 45, N)).round(1)
    inventory_value      = rng.uniform(200000, 800000, N).round(0)
    cash_in_register     = rng.uniform(5000, 80000, N).round(0)

    # ── customer behavioural scores ──────────────────────────────────────────
    purchase_frequency   = np.where(cust_types=="new", rng.uniform(0,1,N),
                           np.where(cust_types=="returning", rng.uniform(1,4,N),
                           np.where(cust_types=="loyal", rng.uniform(4,10,N),
                                    rng.uniform(8,20,N)))).round(1)
    avg_basket_hist      = (lifetime_v / (purchase_frequency * 12 + 1)).round(2)
    distance_km          = rng.uniform(0, 10, N).round(2)
    referral_src         = rng.choice(["walk_in","friend","social","search","ad"],
                                       size=N, p=[0.35,0.20,0.22,0.15,0.08])
    cust_segment         = rng.choice(["budget","regular","premium"],
                                       size=N, p=[0.38, 0.42, 0.20])

    # ── TARGET ────────────────────────────────────────────────────────────────
    # high_value_transaction: 1 if amount is in top ~40% for that category
    # Score based on realistic drivers
    score = (
          0.35 * (total_amount - total_amount.mean()) / total_amount.std()
        + 0.15 * (np.isin(cust_types, ["loyal","vip"])).astype(float)
        + 0.10 * (num_items - num_items.mean()) / num_items.std()
        + 0.08 * payday_week
        + 0.08 * festival_period
        + 0.06 * is_weekend
        + 0.05 * (social_promo | promoter_present).astype(float)
        + 0.05 * (weather == "sunny").astype(float)
        - 0.04 * (weather == "rainy").astype(float)
        - 0.04 * competitor_sale
        + rng.normal(0, 0.3, N)
    )
    high_value_txn = (sigmoid(score) > 0.50).astype(int)

    # ── assemble ─────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        # date / time
        "date":                     dates.strftime("%Y-%m-%d"),
        "day_of_week":              dow.values,
        "month":                    month.values,
        "quarter":                  quarter.values,
        "week_of_year":             woy,
        "hour_of_day":              hour,
        "is_weekend":               is_weekend,
        "is_holiday":               is_holiday,
        "season":                   season,
        "days_since_start":         days_since,
        # transaction
        "num_items":                num_items,
        "total_amount":             total_amount,
        "avg_item_price":           avg_item_price,
        "max_item_price":           max_item_price,
        "discount_pct":             discount_pct.round(1),
        "coupon_applied":           coupon_applied,
        "promo_code":               promo_code,
        "loyalty_points_used":      loyalty_points_used,
        "loyalty_points_earned":    loyalty_points_earned,
        "tax_amount":               tax_amount,
        "net_amount":               net_amount,
        "payment_method":           payment_method,
        "transaction_type":         txn_type,
        "receipt_issued":           receipt_issued,
        # customer
        "customer_type":            cust_types,
        "customer_age_group":       cust_age,
        "customer_gender":          cust_gender,
        "customer_tenure_days":     tenure_days,
        "purchase_frequency_pm":    purchase_frequency,
        "avg_basket_size_hist":     avg_basket_hist,
        "customer_lifetime_value":  lifetime_v,
        "distance_from_shop_km":    distance_km,
        "referral_source":          referral_src,
        "customer_segment":         cust_segment,
        # product breakdown
        "primary_category":         primary_cat,
        "num_categories_purchased": num_cats,
        "food_items_count":         food_cnt,
        "clothing_items_count":     clothing_cnt,
        "electronics_items_count":  elec_cnt,
        "stationery_items_count":   stat_cnt,
        "home_items_count":         home_cnt,
        "beauty_items_count":       beauty_cnt,
        "toys_items_count":         toys_cnt,
        "perishable_pct":           perishable_pct,
        "branded_items_pct":        branded_pct,
        "private_label_pct":        private_lbl_pct,
        "high_margin_items_count":  high_margin_cnt,
        "low_margin_items_count":   low_margin_cnt,
        "items_on_promotion":       items_on_promo,
        # staff / operations
        "staff_on_duty":            staff_on_duty,
        "cashier_experience_yrs":   cashier_exp_yrs,
        "queue_length_at_entry":    queue_length,
        "time_in_store_minutes":    time_in_store_min,
        "checkout_time_seconds":    checkout_time_sec,
        "stock_replenished_today":  stock_replenished,
        "store_footfall_today":     store_footfall,
        "promoter_present":         promoter_present,
        # external
        "weather_condition":        weather,
        "temperature_celsius":      temperature,
        "local_event_nearby":       local_event,
        "competitor_sale_ongoing":  competitor_sale,
        "payday_week":              payday_week,
        "festival_period":          festival_period,
        "school_term_active":       school_term,
        "transport_disruption":     transport_disrup,
        "nearby_construction":      construction,
        "social_media_promo_active":social_promo,
        # financial context
        "daily_revenue_so_far":     daily_rev_so_far.round(2),
        "daily_txn_count_so_far":   daily_txn_so_far,
        "daily_target_progress_pct":daily_tgt_progress,
        "shop_operating_cost_daily":operating_cost_daily,
        "gross_margin_pct":         gross_margin_pct,
        "inventory_value_today":    inventory_value,
        "cash_in_register":         cash_in_register,
        # target
        "high_value_transaction":   high_value_txn,
    })

    # Sprinkle realistic missing values
    df = add_nulls(df, ["temperature_celsius","cashier_experience_yrs",
                        "avg_basket_size_hist","distance_from_shop_km",
                        "customer_lifetime_value","queue_length_at_entry"], rate=0.03)
    return df


# ══════════════════════════════════════════════════════════════════════════
# DATASET 2 — Online Seller Inventory  (1 year, weekly per-SKU snapshots)
# Target: stockout_risk  (1 if stock will fall below safety stock within 2 wks)
# ══════════════════════════════════════════════════════════════════════════

def gen_inventory() -> pd.DataFrame:
    N_SKUS  = 100
    N_WEEKS = 52
    N       = N_SKUS * N_WEEKS   # 5,200

    sku_ids = [f"SKU-{i:04d}" for i in range(1, N_SKUS + 1)]
    all_dates = [date(2023, 1, 2) + timedelta(weeks=w) for w in range(N_WEEKS)]

    # Repeat each SKU × 52 weeks
    sku_col    = np.tile(sku_ids, N_WEEKS)
    dates_col  = np.repeat(all_dates, N_SKUS)
    dates_ser  = pd.to_datetime(dates_col)

    week_of_year = dates_ser.isocalendar().week.astype(int).values
    month        = dates_ser.month.values
    quarter      = dates_ser.quarter.values

    # ── per-SKU static attributes (broadcast) ─────────────────────────────
    categories   = rng.choice(
        ["electronics","clothing","books","home","beauty","sports","food","toys"],
        size=N_SKUS, p=[0.15,0.20,0.10,0.15,0.12,0.10,0.10,0.08])
    brand_tier   = rng.choice(["budget","mid","premium"],
                               size=N_SKUS, p=[0.35, 0.45, 0.20])
    product_age  = rng.integers(30, 1200, N_SKUS)  # days since listing
    supplier_id  = rng.choice([f"SUP-{i:02d}" for i in range(1, 16)], size=N_SKUS)
    lead_time_sk = rng.integers(3, 25, N_SKUS)     # days
    reliability  = rng.uniform(60, 99, N_SKUS).round(1)
    purchase_px  = np.where(
        np.isin(categories, ["electronics"]), rng.uniform(800, 8000, N_SKUS),
        np.where(np.isin(categories, ["clothing","sports"]), rng.uniform(200, 1500, N_SKUS),
                 rng.uniform(80, 800, N_SKUS))).round(2)
    margin_mult  = {"budget":1.30,"mid":1.55,"premium":1.95}
    sell_price   = np.array([purchase_px[i] * margin_mult[brand_tier[i]]
                              for i in range(N_SKUS)]).round(2)
    gross_margin = ((sell_price - purchase_px) / sell_price * 100).round(1)
    wh_zone      = rng.choice(["A","B","C"], size=N_SKUS, p=[0.33, 0.34, 0.33])
    storage_cost = rng.uniform(0.05, 0.80, N_SKUS).round(3)  # per unit/day

    # broadcast static columns
    cat_col     = np.tile(categories, N_WEEKS)
    brand_col   = np.tile(brand_tier, N_WEEKS)
    age_col     = np.tile(product_age, N_WEEKS) + np.repeat(np.arange(N_WEEKS)*7, N_SKUS)
    sup_col     = np.tile(supplier_id, N_WEEKS)
    lead_col    = np.tile(lead_time_sk, N_WEEKS)
    rely_col    = np.tile(reliability, N_WEEKS)
    ppx_col     = np.tile(purchase_px, N_WEEKS)
    spx_col     = np.tile(sell_price, N_WEEKS)
    gm_col      = np.tile(gross_margin, N_WEEKS)
    wh_col      = np.tile(wh_zone, N_WEEKS)
    sc_col      = np.tile(storage_cost, N_WEEKS)

    # ── seasonal demand pattern per SKU ───────────────────────────────────
    base_demand = np.where(
        np.isin(categories, ["electronics"]), rng.uniform(5, 30, N_SKUS),
        np.where(np.isin(categories, ["food","beauty"]), rng.uniform(20, 80, N_SKUS),
                 rng.uniform(8, 45, N_SKUS)))

    # Weekly demand: base × seasonal index × noise
    week_idx_arr = np.tile(np.arange(N_WEEKS), N_SKUS).reshape(N_SKUS, N_WEEKS)
    seasonal_idx = (1.0
                    + 0.30 * np.sin(2 * np.pi * week_idx_arr / 52)
                    + 0.15 * np.sin(4 * np.pi * week_idx_arr / 52))

    # Holiday spikes (weeks 0-2 = New Year/Jan, weeks 48-51 = Dec)
    for hw in [0, 1, 2, 48, 49, 50, 51]:
        if hw < N_WEEKS:
            mult = rng.uniform(1.1, 1.5, N_SKUS) if hw < 3 else rng.uniform(1.3, 2.0, N_SKUS)
            seasonal_idx[:, hw] *= mult

    demand_base = (base_demand.reshape(-1, 1) * seasonal_idx
                   * (1 + rng.normal(0, 0.15, (N_SKUS, N_WEEKS))))
    demand_base = np.clip(demand_base, 0, None)

    demand_flat    = demand_base.T.flatten()   # shape (N,) week-major
    demand_4wk     = np.array([
        demand_flat[max(0, i-4):i].mean() if i > 0 else demand_flat[0]
        for i in range(N)
    ])
    demand_12wk    = np.array([
        demand_flat[max(0, i-12):i].mean() if i > 0 else demand_flat[0]
        for i in range(N)
    ])
    demand_vol     = (np.array([
        demand_flat[max(0, i-8):i].std() / (demand_flat[max(0,i-8):i].mean() + 1e-6)
        if i > 0 else 0.1
        for i in range(N)
    ])).round(3)

    # ── inventory position ─────────────────────────────────────────────────
    safety_stock   = (demand_4wk * (lead_col / 7) * rng.uniform(0.8, 1.5, N)).round(0)
    reorder_pt     = (safety_stock + demand_4wk * (lead_col / 7)).round(0)
    eoq            = (np.sqrt(2 * demand_4wk * 52 * ppx_col * 0.03 / (sc_col + 0.01)) + 1).round(0)

    # Units in stock: fluctuating around a mean that tracks demand
    stock_base     = (reorder_pt * rng.uniform(0.6, 3.0, N)).round(0)
    units_in_stock = np.clip(stock_base, 0, None)

    days_remaining = (units_in_stock / (demand_flat / 7 + 0.1)).round(1)
    days_remaining = np.clip(days_remaining, 0, 365)

    last_repl      = rng.integers(1, 60, N)
    pending_units  = (rng.random(N) < 0.35) * (eoq * rng.uniform(0.5, 1.0, N)).round(0)

    # ── pricing / competitive ───────────────────────────────────────────────
    discount_pct   = np.where(rng.random(N) < 0.30,
                              rng.uniform(5, 35, N), 0.0).round(1)
    promo_active   = (discount_pct > 0).astype(int)
    comp_ratio     = rng.uniform(0.80, 1.25, N).round(3)    # our price / comp price
    return_rate    = rng.uniform(1, 18, N).round(1)

    # ── product visibility ──────────────────────────────────────────────────
    search_rank    = rng.integers(1, 200, N)
    ctr            = np.clip(rng.normal(3.5, 1.5, N), 0.1, 15).round(2)
    cvr            = np.clip(rng.normal(4.0, 1.8, N), 0.2, 20).round(2)
    rating         = np.clip(rng.normal(4.1, 0.5, N), 1.0, 5.0).round(1)
    num_reviews    = np.clip(rng.integers(0, 1200, N), 0, None)

    # ── fulfillment ─────────────────────────────────────────────────────────
    fulfill_time   = np.clip(rng.normal(2.5, 1.0, N), 0.5, 10).round(1)
    overstock_units= np.clip(units_in_stock - reorder_pt * 2, 0, None).round(0)

    # ── seasonal index (broadcast back for output) ─────────────────────────
    seas_flat      = seasonal_idx.T.flatten().round(3)

    # ── demand trend label ──────────────────────────────────────────────────
    trend_val      = demand_flat - demand_12wk
    trend_lbl      = np.where(trend_val > demand_12wk * 0.10, "increasing",
                     np.where(trend_val < -demand_12wk * 0.10, "decreasing","stable"))

    # ── TARGET ────────────────────────────────────────────────────────────────
    # stockout_risk: product likely to stock out within 2 weeks
    score = (
        -0.40 * (units_in_stock - reorder_pt) / (reorder_pt + 1)
        +0.20 * (demand_vol)
        +0.15 * (demand_4wk - demand_12wk) / (demand_12wk + 1)
        +0.12 * ((days_remaining < (lead_col + 7)).astype(float))
        -0.08 * (pending_units > 0).astype(float)
        +0.05 * ((lead_col - rely_col / 10))
        + rng.normal(0, 0.35, N)
    )
    stockout_risk = (sigmoid(score) > 0.48).astype(int)

    # ── assemble ─────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "sku_id":                     sku_col,
        "snapshot_date":              dates_ser.strftime("%Y-%m-%d"),
        "week_of_year":               week_of_year,
        "month":                      month,
        "quarter":                    quarter,
        "category":                   cat_col,
        "brand_tier":                 brand_col,
        "product_age_days":           age_col,
        "units_in_stock":             units_in_stock.astype(int),
        "units_sold_last_week":       demand_flat.round(1),
        "units_sold_last_4wks":       demand_4wk.round(1),
        "units_sold_last_12wks":      demand_12wk.round(1),
        "demand_trend":               trend_lbl,
        "demand_volatility_cv":       demand_vol,
        "seasonal_index":             seas_flat,
        "reorder_point":              reorder_pt.astype(int),
        "safety_stock":               safety_stock.astype(int),
        "economic_order_qty":         eoq.astype(int),
        "days_of_stock_remaining":    days_remaining,
        "last_replenishment_days_ago":last_repl,
        "pending_order_units":        pending_units.astype(int),
        "supplier_id":                sup_col,
        "supplier_lead_time_days":    lead_col,
        "supplier_reliability_score": rely_col,
        "purchase_price":             ppx_col,
        "selling_price":              spx_col,
        "gross_margin_pct":           gm_col,
        "discount_pct":               discount_pct,
        "promotion_active":           promo_active,
        "competitor_price_ratio":     comp_ratio,
        "return_rate_pct":            return_rate,
        "customer_rating":            rating,
        "num_reviews":                num_reviews,
        "search_rank":                search_rank,
        "click_through_rate_pct":     ctr,
        "conversion_rate_pct":        cvr,
        "warehouse_zone":             wh_col,
        "storage_cost_per_unit_day":  sc_col,
        "fulfillment_time_days":      fulfill_time,
        "overstock_units":            overstock_units.astype(int),
        # target
        "stockout_risk":              stockout_risk,
    })

    df = add_nulls(df, ["customer_rating","num_reviews","competitor_price_ratio",
                        "fulfillment_time_days","supplier_reliability_score"], rate=0.04)
    return df


# ══════════════════════════════════════════════════════════════════════════
# DATASET 3 — Customer Churn (subscription / loyalty program, 1 year)
# Target: churned  (1 = cancelled / lapsed within the observation window)
# ══════════════════════════════════════════════════════════════════════════

def gen_churn() -> pd.DataFrame:
    N = 4000

    # ── demographics ──────────────────────────────────────────────────────
    age             = rng.integers(18, 72, N)
    gender          = rng.choice(["M","F","Other"], size=N, p=[0.46,0.50,0.04])
    city_tier       = rng.choice(["tier1","tier2","tier3"], size=N, p=[0.40,0.35,0.25])
    education       = rng.choice(["school","graduate","postgraduate"],
                                  size=N, p=[0.20, 0.55, 0.25])
    occupation      = rng.choice(["salaried","self_employed","student","homemaker","retired"],
                                  size=N, p=[0.42,0.22,0.15,0.12,0.09])
    income_band     = rng.choice(["low","medium","high","very_high"],
                                  size=N, p=[0.25, 0.40, 0.25, 0.10])

    # ── subscription / account ────────────────────────────────────────────
    plan_type       = rng.choice(["basic","standard","premium","enterprise"],
                                  size=N, p=[0.30, 0.38, 0.22, 0.10])
    plan_mult       = {"basic":1.0,"standard":1.4,"premium":2.0,"enterprise":3.5}
    monthly_charge  = np.array([rng.uniform(200, 400) * plan_mult[p] for p in plan_type]).round(2)
    tenure_months   = rng.integers(1, 72, N)
    contract_type   = rng.choice(["monthly","annual","biannual"],
                                  size=N, p=[0.45, 0.40, 0.15])
    auto_renew      = (rng.random(N) < 0.60).astype(int)
    payment_method  = rng.choice(["credit_card","debit_card","net_banking","upi","wallet"],
                                  size=N, p=[0.25, 0.30, 0.15, 0.22, 0.08])

    # ── usage / engagement ────────────────────────────────────────────────
    logins_per_month        = np.clip(rng.poisson(12, N), 0, 60)
    avg_session_min         = np.clip(rng.normal(18, 8, N), 1, 90).round(1)
    features_used_count     = rng.integers(1, 20, N)
    mobile_app_usage_pct    = rng.uniform(0, 100, N).round(1)
    support_tickets_raised  = rng.integers(0, 12, N)
    support_tickets_resolved= (support_tickets_raised
                                * rng.uniform(0.7, 1.0, N)).astype(int)
    nps_score               = rng.integers(0, 11, N)
    last_login_days_ago     = rng.integers(0, 180, N)
    referrals_made          = rng.integers(0, 8, N)

    # ── payment behaviour ─────────────────────────────────────────────────
    late_payments_count     = rng.integers(0, 6, N)
    payment_failures        = rng.integers(0, 4, N)
    total_spend_12m         = (monthly_charge * tenure_months.clip(0, 12)
                                * rng.uniform(0.8, 1.0, N)).round(2)
    discount_used_count     = rng.integers(0, 5, N)
    upgrade_downgrade_count = rng.integers(0, 3, N)

    # ── product satisfaction ──────────────────────────────────────────────
    product_rating          = np.clip(rng.normal(3.8, 0.9, N), 1, 5).round(1)
    onboarding_completed    = (rng.random(N) < 0.72).astype(int)
    tutorial_completed      = (rng.random(N) < 0.55).astype(int)
    days_to_first_value     = rng.integers(1, 45, N)   # time to "aha moment"
    competitor_product_used = (rng.random(N) < 0.28).astype(int)
    promo_sensitivity       = rng.choice(["low","medium","high"],
                                          size=N, p=[0.35, 0.40, 0.25])

    # ── TARGET ────────────────────────────────────────────────────────────────
    # churned: customer cancelled/lapsed within the observation window
    score = (
        -0.25 * (logins_per_month - logins_per_month.mean()) / logins_per_month.std()
        -0.20 * (tenure_months - tenure_months.mean()) / tenure_months.std()
        +0.15 * (last_login_days_ago - last_login_days_ago.mean()) / last_login_days_ago.std()
        +0.10 * (support_tickets_raised - support_tickets_resolved)
        +0.08 * late_payments_count
        -0.08 * (nps_score - 5.0) / 5.0
        -0.07 * auto_renew.astype(float)
        +0.06 * competitor_product_used.astype(float)
        +0.05 * payment_failures
        -0.05 * onboarding_completed.astype(float)
        + rng.normal(0, 0.45, N)
    )
    churned = (sigmoid(score) > 0.48).astype(int)

    df = pd.DataFrame({
        # demographics
        "age":                      age,
        "gender":                   gender,
        "city_tier":                city_tier,
        "education":                education,
        "occupation":               occupation,
        "income_band":              income_band,
        # subscription
        "plan_type":                plan_type,
        "monthly_charge":           monthly_charge,
        "tenure_months":            tenure_months,
        "contract_type":            contract_type,
        "auto_renew":               auto_renew,
        "payment_method":           payment_method,
        # usage
        "logins_per_month":         logins_per_month,
        "avg_session_minutes":      avg_session_min,
        "features_used_count":      features_used_count,
        "mobile_app_usage_pct":     mobile_app_usage_pct,
        "support_tickets_raised":   support_tickets_raised,
        "support_tickets_resolved": support_tickets_resolved,
        "nps_score":                nps_score,
        "last_login_days_ago":      last_login_days_ago,
        "referrals_made":           referrals_made,
        # payments
        "late_payments_count":      late_payments_count,
        "payment_failures":         payment_failures,
        "total_spend_12m":          total_spend_12m,
        "discount_used_count":      discount_used_count,
        "upgrade_downgrade_count":  upgrade_downgrade_count,
        # product
        "product_rating":           product_rating,
        "onboarding_completed":     onboarding_completed,
        "tutorial_completed":       tutorial_completed,
        "days_to_first_value":      days_to_first_value,
        "competitor_product_used":  competitor_product_used,
        "promo_sensitivity":        promo_sensitivity,
        # target
        "churned":                  churned,
    })

    df = add_nulls(df, ["product_rating","nps_score","income_band",
                        "days_to_first_value","avg_session_minutes"], rate=0.04)
    return df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\nGenerating sample datasets…\n")

    ds1 = gen_shop_sales()
    save(ds1, "small_shop_sales_2yr.csv")

    ds2 = gen_inventory()
    save(ds2, "online_seller_inventory_1yr.csv")

    ds3 = gen_churn()
    save(ds3, "customer_churn_1yr.csv")

    print(textwrap.dedent("""
    ─────────────────────────────────────────────────────────
     Dataset guide
    ─────────────────────────────────────────────────────────
     1. small_shop_sales_2yr.csv
        ▸ 5,840 rows · 76 columns
        ▸ 730 days of retail transactions (Jul 2022 – Jun 2024)
        ▸ Target: high_value_transaction (binary)
        ▸ Suggested target column: high_value_transaction
        ▸ Positive class: 1

     2. online_seller_inventory_1yr.csv
        ▸ 5,200 rows · 41 columns
        ▸ Weekly SKU-level inventory snapshots (100 products × 52 weeks)
        ▸ Target: stockout_risk (binary)
        ▸ Suggested target column: stockout_risk
        ▸ Positive class: 1

     3. customer_churn_1yr.csv
        ▸ 4,000 rows · 33 columns
        ▸ Subscription / loyalty programme customer records
        ▸ Target: churned (binary)
        ▸ Suggested target column: churned
        ▸ Positive class: 1

     Upload any of these via the "Upload CSV" tab in the app.
    ─────────────────────────────────────────────────────────
    """))


if __name__ == "__main__":
    main()
