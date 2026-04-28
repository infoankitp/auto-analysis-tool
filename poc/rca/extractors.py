"""
Causal dataset extractors for RCA.

Each extractor pulls a ±CAUSAL_WINDOW_DAYS slice of one signal type
and returns a structured summary dict.  In production these query
ClickHouse; here they query the in-memory DataFrames.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from poc.config import CAUSAL_WINDOW_DAYS


def _window_mask(df: pd.DataFrame, centre_date: pd.Timestamp, days: int) -> pd.Series:
    start = centre_date - pd.Timedelta(days=days)
    end   = centre_date
    return (df["date"] >= start) & (df["date"] <= end)


class SalesExtractor:
    """Pull ±window_days of net_sales for the anomalous store × category."""

    def extract(
        self,
        sales_df:    pd.DataFrame,
        store_id:    str,
        category:    str,
        anchor_date: pd.Timestamp,
        window_days: int = CAUSAL_WINDOW_DAYS,
    ) -> dict:
        mask = (
            (sales_df["store_id"] == store_id) &
            (sales_df["category"] == category) &
            _window_mask(sales_df, anchor_date, window_days)
        )
        sub = sales_df[mask].sort_values("date")

        if sub.empty:
            return {"status": "no_data"}

        today_row    = sub[sub["date"] == anchor_date]
        history_rows = sub[sub["date"] < anchor_date]

        hist_mean = float(history_rows["net_sales"].mean()) if not history_rows.empty else 0.0
        hist_std  = float(history_rows["net_sales"].std())  if not history_rows.empty else 0.0
        today_val = float(today_row["net_sales"].iloc[0])   if not today_row.empty else 0.0

        daily = [
            {
                "date":       str(r["date"].date()),
                "net_sales":  round(float(r["net_sales"]), 0),
                "units_sold": int(r["units_sold"]),
            }
            for _, r in sub.iterrows()
        ]

        return {
            "store_id":           store_id,
            "category":           category,
            "window_days":        window_days,
            "today_net_sales":    round(today_val, 0),
            "history_mean_28d":   round(hist_mean, 0),
            "history_std_28d":    round(hist_std, 0),
            "pct_drop_vs_mean":   round((today_val - hist_mean) / max(hist_mean, 1) * 100, 1),
            "consecutive_low_days": int(
                sub["net_sales"].lt(hist_mean * 0.7).iloc[::-1].cumprod().sum()
            ),
            "daily_series":       daily[-7:],   # last 7 days for prompt
        }


class WeatherExtractor:
    """Pull weather for the store's region around the anomaly date."""

    def extract(
        self,
        weather_df:  pd.DataFrame,
        region:      str,
        anchor_date: pd.Timestamp,
        window_days: int = CAUSAL_WINDOW_DAYS,
    ) -> dict:
        mask = (weather_df["region"] == region) & _window_mask(weather_df, anchor_date, window_days)
        sub  = weather_df[mask].sort_values("date")

        if sub.empty:
            return {"status": "no_data"}

        today_row = sub[sub["date"] == anchor_date]
        hist_rows = sub[sub["date"] < anchor_date]

        today_sev   = today_row["severity_code"].iloc[0]   if not today_row.empty else "normal"
        today_prec  = float(today_row["precipitation_mm"].iloc[0]) if not today_row.empty else 0.0
        avg_prec    = float(hist_rows["precipitation_mm"].mean()) if not hist_rows.empty else 0.0
        max_prec    = float(hist_rows["precipitation_mm"].max())  if not hist_rows.empty else 0.0

        recent = [
            {
                "date":             str(r["date"].date()),
                "severity_code":    r["severity_code"],
                "precipitation_mm": r["precipitation_mm"],
                "temp_c":           r["temp_c"],
            }
            for _, r in sub.tail(5).iterrows()
        ]

        return {
            "region":                  region,
            "today_severity":          today_sev,
            "today_precipitation_mm":  round(today_prec, 1),
            "avg_precipitation_14d":   round(avg_prec, 1),
            "max_precipitation_14d":   round(max_prec, 1),
            "weather_anomaly_flagged": today_sev not in ("normal", "rain"),
            "recent_5_days":           recent,
        }


class HolidayExtractor:
    """Check whether any high-impact holidays fall within the causal window."""

    def extract(
        self,
        holiday_df:  pd.DataFrame,
        anchor_date: pd.Timestamp,
        window_days: int = CAUSAL_WINDOW_DAYS,
    ) -> dict:
        if holiday_df.empty:
            return {"holidays_in_window": [], "max_impact": 0.0}

        mask = _window_mask(holiday_df, anchor_date, window_days)
        sub  = holiday_df[mask].sort_values("date")

        holidays = [
            {
                "date":         str(r["date"].date()),
                "name":         r["holiday_name"],
                "type":         r["holiday_type"],
                "impact_score": r["impact_score"],
            }
            for _, r in sub.iterrows()
        ]

        max_impact = float(sub["impact_score"].max()) if not sub.empty else 0.0

        return {
            "holidays_in_window": holidays,
            "max_impact":         round(max_impact, 2),
            "any_high_impact":    max_impact >= 0.7,
        }


class ReviewExtractor:
    """Pull review sentiment trajectory for a store in the causal window."""

    def extract(
        self,
        reviews_df:  pd.DataFrame,
        store_id:    str,
        anchor_date: pd.Timestamp,
        window_days: int = CAUSAL_WINDOW_DAYS,
    ) -> dict:
        mask = (reviews_df["store_id"] == store_id) & _window_mask(reviews_df, anchor_date, window_days)
        sub  = reviews_df[mask].sort_values("date")

        if sub.empty:
            return {"status": "no_data"}

        today_row = sub[sub["date"] == anchor_date]
        hist_rows = sub[sub["date"] < anchor_date]

        today_rating    = float(today_row["rating"].iloc[0])          if not today_row.empty else None
        today_sentiment = float(today_row["sentiment_score"].iloc[0]) if not today_row.empty else None
        today_keywords  = today_row["keywords"].iloc[0]               if not today_row.empty else ""

        avg_hist_rating    = float(hist_rows["rating"].mean())          if not hist_rows.empty else 0.0
        avg_hist_sentiment = float(hist_rows["sentiment_score"].mean()) if not hist_rows.empty else 0.0

        recent = [
            {
                "date":           str(r["date"].date()),
                "rating":         r["rating"],
                "sentiment_score": r["sentiment_score"],
                "keywords":       r["keywords"],
            }
            for _, r in sub.tail(5).iterrows()
        ]

        # Detect a sharp rating decline
        rating_decline = None
        if today_rating is not None and avg_hist_rating > 0:
            rating_decline = round(today_rating - avg_hist_rating, 2)

        return {
            "store_id":              store_id,
            "today_rating":          today_rating,
            "today_sentiment":       today_sentiment,
            "today_keywords":        today_keywords,
            "avg_hist_rating_14d":   round(avg_hist_rating, 2),
            "avg_hist_sentiment_14d": round(avg_hist_sentiment, 3),
            "rating_delta":          rating_decline,
            "sentiment_declined":    (today_sentiment is not None and today_sentiment < avg_hist_sentiment - 0.3),
            "negative_keywords":     [k for k in today_keywords.split(",") if k.strip()],
            "recent_5_days":         recent,
        }


class StockExtractor:
    """Pull stock level data for the anomalous store × category."""

    def extract(
        self,
        stock_df:    pd.DataFrame,
        store_id:    str,
        category:    str,
        anchor_date: pd.Timestamp,
        window_days: int = CAUSAL_WINDOW_DAYS,
    ) -> dict:
        mask = (
            (stock_df["store_id"] == store_id) &
            (stock_df["category"] == category) &
            _window_mask(stock_df, anchor_date, window_days)
        )
        sub = stock_df[mask].sort_values("date")

        if sub.empty:
            return {"status": "no_data"}

        today_row    = sub[sub["date"] == anchor_date]
        today_stock  = float(today_row["stock_level"].iloc[0])   if not today_row.empty else None
        today_flag   = today_row["supplier_flag"].iloc[0]         if not today_row.empty else "normal"
        hist_rows    = sub[sub["date"] < anchor_date]
        avg_stock    = float(hist_rows["stock_level"].mean())     if not hist_rows.empty else 0.75

        return {
            "store_id":            store_id,
            "category":            category,
            "today_stock_level":   today_stock,
            "avg_stock_level_14d": round(avg_stock, 3),
            "stock_below_reorder": (today_stock is not None and today_stock < 0.20),
            "effectively_empty":   (today_stock is not None and today_stock < 0.05),
            "supplier_flag":       today_flag,
            "supplier_disrupted":  today_flag == "disrupted",
        }
