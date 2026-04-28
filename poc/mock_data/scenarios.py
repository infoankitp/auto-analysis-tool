"""
Anomaly scenario injection.

Three realistic retail failure modes are injected into TODAY's data:
  1. Staff Shortage  — STORE_003 / Apparel  / NE region
  2. Weather Event   — STORE_007 / Electronics / MW region
  3. Stock-Out       — STORE_012 / Beverages  / SE region

Each function modifies the DataFrames in-place and returns them.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from poc.config import TODAY, ANOMALY_SCENARIOS, CATEGORY_BASELINE_REVENUE, STORE_REVENUE_MULTIPLIER


def inject_staff_shortage(
    sales_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STORE_003 Apparel: severe staff walkout — sales crash, reviews turn toxic.

    Signal pattern:
      - Sales today: 25% of baseline (down ~75%)
      - Reviews last 3 days: rating 1.8-2.4, sentiment -0.55 to -0.72,
        keywords mention 'understaffed', 'no staff', 'walked out'
    """
    cfg = ANOMALY_SCENARIOS["staff_shortage"]
    store_id, category = cfg["store_id"], cfg["category"]
    today_ts = pd.Timestamp(TODAY)

    # --- Sales drop ---
    base   = CATEGORY_BASELINE_REVENUE[category] * STORE_REVENUE_MULTIPLIER["high"]
    target = base * cfg["sales_factor"]
    mask   = (
        (sales_df["store_id"] == store_id) &
        (sales_df["category"] == category) &
        (sales_df["date"] == today_ts)
    )
    sales_df.loc[mask, "net_sales"]   = round(target + np.random.normal(0, target * 0.05), 2)
    sales_df.loc[mask, "units_sold"]  = max(0, int(sales_df.loc[mask, "net_sales"].values[0] / 46))
    sales_df.loc[mask, "basket_size"] = round(target / max(1, sales_df.loc[mask, "units_sold"].values[0]), 2)

    # --- Negative reviews for last 3 days ---
    negative_reviews = [
        "understaffed,no staff available,walked out,long queue",
        "terrible wait time,understaffed,poor service,no managers",
        "no staff,checkout nightmare,understaffed store,will not return",
    ]
    for i, offset in enumerate([2, 1, 0]):
        review_date = today_ts - pd.Timedelta(days=offset)
        rmask = (reviews_df["store_id"] == store_id) & (reviews_df["date"] == review_date)
        if rmask.any():
            rating    = round(np.random.uniform(1.7, 2.4), 1)
            sentiment = round(np.random.uniform(-0.72, -0.50), 3)
            reviews_df.loc[rmask, "rating"]          = rating
            reviews_df.loc[rmask, "sentiment_score"] = sentiment
            reviews_df.loc[rmask, "review_count"]    = np.random.randint(18, 28)
            reviews_df.loc[rmask, "keywords"]        = negative_reviews[i]

    return sales_df, reviews_df


def inject_weather_event(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STORE_007 Electronics: heavy storm in MW region cuts foot traffic by ~65%.

    Signal pattern:
      - Sales today: 35% of baseline
      - Weather today (MW): storm, precipitation 38mm (vs normal 2mm)
      - Severity code: 'storm' — above rain threshold but not blizzard,
        so context_guard does NOT suppress the anomaly alert
    """
    cfg = ANOMALY_SCENARIOS["weather_event"]
    store_id, category, region = cfg["store_id"], cfg["category"], cfg["region"]
    today_ts = pd.Timestamp(TODAY)

    # --- Sales drop ---
    base   = CATEGORY_BASELINE_REVENUE[category] * STORE_REVENUE_MULTIPLIER["high"]
    target = base * cfg["sales_factor"]
    mask   = (
        (sales_df["store_id"] == store_id) &
        (sales_df["category"] == category) &
        (sales_df["date"] == today_ts)
    )
    sales_df.loc[mask, "net_sales"]   = round(target + np.random.normal(0, target * 0.04), 2)
    sales_df.loc[mask, "units_sold"]  = max(0, int(sales_df.loc[mask, "net_sales"].values[0] / 47))
    sales_df.loc[mask, "basket_size"] = round(target / max(1, sales_df.loc[mask, "units_sold"].values[0]), 2)

    # --- Severe weather in MW today ---
    wmask = (weather_df["region"] == region) & (weather_df["date"] == today_ts)
    weather_df.loc[wmask, "precipitation_mm"] = 38.0
    weather_df.loc[wmask, "temp_c"]           = 3.2
    weather_df.loc[wmask, "severity_code"]    = "storm"   # storm but not blizzard/hurricane

    return sales_df, weather_df


def inject_stock_outage(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    STORE_012 Beverages: supplier failure — shelves bare, sales near zero.

    Signal pattern:
      - Sales today: 10% of baseline (a few residual purchases from backstock)
      - Stock level: 0.03 (essentially zero)
      - Supplier flag: 'disrupted'
      - Reviews mention 'out of stock', 'empty shelves'
    """
    cfg = ANOMALY_SCENARIOS["stock_outage"]
    store_id, category = cfg["store_id"], cfg["category"]
    today_ts = pd.Timestamp(TODAY)

    # --- Sales near-zero ---
    base   = CATEGORY_BASELINE_REVENUE[category] * STORE_REVENUE_MULTIPLIER["medium"]
    target = base * cfg["sales_factor"]
    mask   = (
        (sales_df["store_id"] == store_id) &
        (sales_df["category"] == category) &
        (sales_df["date"] == today_ts)
    )
    sales_df.loc[mask, "net_sales"]   = round(target, 2)
    sales_df.loc[mask, "units_sold"]  = max(0, int(target / 42))
    sales_df.loc[mask, "basket_size"] = round(target / max(1, sales_df.loc[mask, "units_sold"].values[0]), 2)

    # --- Stock-out ---
    smask = (
        (stock_df["store_id"] == store_id) &
        (stock_df["category"] == category) &
        (stock_df["date"] == today_ts)
    )
    stock_df.loc[smask, "stock_level"]   = 0.03
    stock_df.loc[smask, "supplier_flag"] = "disrupted"

    # --- Reviews mentioning out-of-stock ---
    rmask = (reviews_df["store_id"] == store_id) & (reviews_df["date"] == today_ts)
    if rmask.any():
        reviews_df.loc[rmask, "rating"]          = 2.3
        reviews_df.loc[rmask, "sentiment_score"] = -0.45
        reviews_df.loc[rmask, "review_count"]    = 9
        reviews_df.loc[rmask, "keywords"]        = "out of stock,empty shelves,no beverages,supplier issue"

    return sales_df, stock_df, reviews_df


def inject_all_scenarios(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply all three anomaly scenarios to the generated baseline data."""
    sales_df, reviews_df             = inject_staff_shortage(sales_df, reviews_df)
    sales_df, weather_df             = inject_weather_event(sales_df, weather_df)
    sales_df, stock_df, reviews_df   = inject_stock_outage(sales_df, stock_df, reviews_df)
    return sales_df, weather_df, reviews_df, stock_df
