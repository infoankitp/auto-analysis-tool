"""
Mock data generators for all retail signals.

Each generator returns a clean pandas DataFrame. Data spans HISTORY_START to TODAY.
The final day (TODAY) will have anomalies injected by scenarios.py.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

from poc.config import (
    STORES, CATEGORIES, REGIONS,
    CATEGORY_BASELINE_REVENUE, STORE_REVENUE_MULTIPLIER,
    WEEKLY_SEASONAL_PATTERN, HISTORY_START, TODAY,
)


def _date_range(start: date, end: date) -> list[date]:
    delta = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(delta)]


def generate_store_metadata() -> pd.DataFrame:
    """Static store attributes: region, cluster type, revenue tier."""
    rows = []
    for i, s in enumerate(STORES):
        rows.append({
            "store_id":    s["store_id"],
            "region":      s["region"],
            "cluster_type": s["cluster_type"],
            "revenue_tier": s["revenue_tier"],
            "cluster_rank": i + 1,
        })
    return pd.DataFrame(rows)


def generate_sales_data(start: date = HISTORY_START, end: date = TODAY, seed: int = 42) -> pd.DataFrame:
    """
    Daily net_sales, units_sold, basket_size for every store × category.
    Includes weekly seasonality + Gaussian noise; no anomalies here.
    """
    rng = np.random.default_rng(seed)
    records = []
    dates = _date_range(start, end)

    for store in STORES:
        store_id  = store["store_id"]
        region    = store["region"]
        tier      = store["revenue_tier"]
        tier_mult = STORE_REVENUE_MULTIPLIER[tier]

        for category in CATEGORIES:
            base    = CATEGORY_BASELINE_REVENUE[category] * tier_mult
            std_dev = base * 0.12   # 12% noise

            for dt in dates:
                seasonal   = WEEKLY_SEASONAL_PATTERN[dt.weekday()]
                net_sales  = max(0.0, base * seasonal + rng.normal(0, std_dev))
                avg_basket = max(10.0, 45.0 + rng.normal(0, 5))
                units      = max(0, int(net_sales / avg_basket))
                basket     = net_sales / max(1, units)

                records.append({
                    "date":        dt,
                    "store_id":    store_id,
                    "category":    category,
                    "region":      region,
                    "channel":     "instore",
                    "net_sales":   round(net_sales, 2),
                    "units_sold":  units,
                    "basket_size": round(basket, 2),
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_weather_data(start: date = HISTORY_START, end: date = TODAY, seed: int = 99) -> pd.DataFrame:
    """
    Daily weather per region: temperature, precipitation, severity code.
    Normal weather — anomalous weather injected by scenarios.py.
    """
    rng = np.random.default_rng(seed)
    records = []

    for dt in _date_range(start, end):
        for region in REGIONS:
            temp_c     = rng.normal(15, 8)
            precip     = max(0.0, rng.normal(2, 3))
            if precip > 15:
                severity = "heavy_rain"
            elif precip > 8:
                severity = "rain"
            else:
                severity = "normal"

            records.append({
                "date":             dt,
                "region":           region,
                "temp_c":           round(temp_c, 1),
                "precipitation_mm": round(precip, 1),
                "severity_code":    severity,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_reviews_data(start: date = HISTORY_START, end: date = TODAY, seed: int = 77) -> pd.DataFrame:
    """
    Daily aggregated review stats per store: avg rating, sentiment, keyword themes.
    Baseline reviews are positive — negative reviews injected by scenarios.py.
    """
    rng = np.random.default_rng(seed)
    records = []

    positive_keywords = [
        "great service,friendly staff,clean store",
        "fast checkout,helpful team,good selection",
        "well stocked,professional staff,easy to find items",
        "quick service,great layout,knowledgeable staff",
    ]

    for store in STORES:
        store_id = store["store_id"]
        for dt in _date_range(start, end):
            rating       = float(np.clip(rng.normal(4.2, 0.3), 1.0, 5.0))
            sentiment    = float(np.clip(rng.normal(0.32, 0.12), -1.0, 1.0))
            review_count = max(0, int(rng.normal(12, 3)))
            keywords     = positive_keywords[rng.integers(0, len(positive_keywords))]

            records.append({
                "date":           dt,
                "store_id":       store_id,
                "rating":         round(rating, 1),
                "sentiment_score": round(sentiment, 3),
                "review_count":   review_count,
                "keywords":       keywords,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_holiday_calendar(start: date = HISTORY_START, end: date = TODAY) -> pd.DataFrame:
    """
    Known retail holidays that could suppress anomaly detection.
    Returns a sparse table (only dates with actual holidays).
    """
    all_holidays = [
        {"month": 5, "day": 27, "holiday_name": "Memorial Day",         "holiday_type": "federal", "impact_score": 0.45},
        {"month": 7, "day":  4, "holiday_name": "Fourth of July",       "holiday_type": "federal", "impact_score": 0.50},
        {"month": 9, "day":  2, "holiday_name": "Labor Day",            "holiday_type": "federal", "impact_score": 0.40},
        {"month": 11,"day": 28, "holiday_name": "Thanksgiving",         "holiday_type": "federal", "impact_score": 0.75},
        {"month": 11,"day": 29, "holiday_name": "Black Friday",         "holiday_type": "retail",  "impact_score": 0.95},
        {"month": 12,"day": 24, "holiday_name": "Christmas Eve",        "holiday_type": "retail",  "impact_score": 0.90},
        {"month": 12,"day": 25, "holiday_name": "Christmas Day",        "holiday_type": "federal", "impact_score": 0.85},
        {"month": 1, "day":  1, "holiday_name": "New Year's Day",       "holiday_type": "federal", "impact_score": 0.60},
    ]

    records = []
    for dt in _date_range(start, end):
        for h in all_holidays:
            if dt.month == h["month"] and dt.day == h["day"]:
                records.append({"date": dt, **{k: v for k, v in h.items() if k not in ("month", "day")}})

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["date", "holiday_name", "holiday_type", "impact_score"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def generate_stock_data(start: date = HISTORY_START, end: date = TODAY, seed: int = 55) -> pd.DataFrame:
    """
    Daily stock levels per store × category (0.0 = empty, 1.0 = full).
    Normal stock is maintained at 0.6-0.9; out-of-stock injected by scenarios.py.
    """
    rng = np.random.default_rng(seed)
    records = []

    for store in STORES:
        store_id = store["store_id"]
        for category in CATEGORIES:
            for dt in _date_range(start, end):
                stock_level = float(np.clip(rng.normal(0.75, 0.08), 0.2, 1.0))
                records.append({
                    "date":          dt,
                    "store_id":      store_id,
                    "category":      category,
                    "stock_level":   round(stock_level, 3),
                    "reorder_point": 0.20,
                    "supplier_flag": "normal",
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
