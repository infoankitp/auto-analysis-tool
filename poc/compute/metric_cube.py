"""
Metric cube — pre-computed rolling baselines for every store × category × metric.

In production this lives in ClickHouse and is refreshed nightly by a Spark job.
Here it is an in-memory pandas structure that the detection engine queries.

Key outputs per segment-metric row:
  baseline_28d  — 28-day rolling mean (excluding the current day)
  std_28d       — 28-day rolling std
  baseline_7d   — 7-day rolling mean (for short-window detectors)
  std_7d        — 7-day rolling std
  yoy_delta     — year-on-year delta (uses 0 when < 365 days available)
"""

import pandas as pd
import numpy as np
from datetime import date

from poc.config import STORES, REVENUE_WEIGHTS


# Tier assignment based on cluster_rank (mirrors active_segments.sql)
def _assign_tier(cluster_rank: int) -> str:
    if cluster_rank <= 2:
        return "tier1"
    elif cluster_rank <= 5:
        return "tier2"
    return "tier3"


class MetricCube:
    """
    Builds and serves a DataFrame-backed metric cube.

    Usage:
        cube = MetricCube()
        cube.build(sales_df, store_metadata_df)
        series, baseline, std = cube.get_segment_series("STORE_003|Apparel", "net_sales")
        segments = cube.get_active_segments()
    """

    def __init__(self):
        self._cube:     pd.DataFrame | None = None
        self._meta:     pd.DataFrame | None = None
        self._is_built: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, sales_df: pd.DataFrame, store_meta_df: pd.DataFrame) -> None:
        """
        Compute 28-day and 7-day rolling baselines for net_sales.
        Must be called before any query method.
        """
        self._meta = store_meta_df.set_index("store_id")

        df = (
            sales_df[["date", "store_id", "category", "net_sales", "units_sold", "basket_size"]]
            .copy()
            .sort_values(["store_id", "category", "date"])
        )

        # Melt to long format: one row per (date, store_id, category, metric_name)
        melted = df.melt(
            id_vars=["date", "store_id", "category"],
            value_vars=["net_sales", "units_sold", "basket_size"],
            var_name="metric_name",
            value_name="value",
        )

        # Rolling baselines — same-day-of-week approach removes seasonal bias.
        # Comparing Tuesday to Tuesday (last 4) gives a noise-only std,
        # making the z-score a true signal-to-noise ratio.
        def _dow_rolling_stats(dow_group: pd.DataFrame) -> pd.DataFrame:
            dow_group = dow_group.sort_values("date")
            dow_group["baseline_28d"] = dow_group["value"].shift(1).rolling(4, min_periods=2).mean()
            dow_group["std_28d"]      = dow_group["value"].shift(1).rolling(4, min_periods=2).std()
            return dow_group

        def _rolling_stats(g: pd.DataFrame) -> pd.DataFrame:
            g   = g.sort_values("date").copy()
            g["_dow"] = pd.to_datetime(g["date"]).dt.dayofweek

            # Same-weekday rolling (last 4 weeks) — deseasonalized baseline
            g = (
                g.groupby("_dow", group_keys=False)
                 .apply(_dow_rolling_stats)
                 .sort_values("date")
            )
            g["std_28d"]  = g["std_28d"].fillna(1.0).clip(lower=1.0)

            # 7-day rolling (for context; retains seasonality — used by STL/IF, not z-score)
            g["baseline_7d"] = g["value"].shift(1).rolling(7, min_periods=3).mean()
            g["std_7d"]      = g["value"].shift(1).rolling(7, min_periods=3).std().fillna(1.0)
            g["yoy_delta"]   = g["value"] - g["value"].shift(364)
            g                = g.drop("_dow", axis=1)
            return g

        cube = (
            melted
            .groupby(["store_id", "category", "metric_name"], group_keys=False)
            .apply(_rolling_stats)
            .reset_index(drop=True)
        )

        # Attach revenue weight and region from store metadata
        cube["segment_key"]    = cube["store_id"] + "|" + cube["category"]
        cube["revenue_weight"] = cube["store_id"].map(
            lambda sid: REVENUE_WEIGHTS.get(
                self._meta.loc[sid, "revenue_tier"] if sid in self._meta.index else "medium",
                0.65,
            )
        )
        cube["region"] = cube["store_id"].map(
            lambda sid: self._meta.loc[sid, "region"] if sid in self._meta.index else ""
        )

        self._cube    = cube
        self._is_built = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_segment_series(
        self,
        store_id:    str,
        category:    str,
        metric_name: str = "net_sales",
        up_to_date:  date | None = None,
    ) -> tuple[pd.Series, float, float]:
        """
        Returns (time-series values, baseline_28d for last row, std_28d for last row).
        up_to_date restricts the series to that date (inclusive).
        """
        self._check_built()
        mask = (
            (self._cube["store_id"]    == store_id) &
            (self._cube["category"]    == category) &
            (self._cube["metric_name"] == metric_name)
        )
        sub = self._cube[mask].sort_values("date")
        if up_to_date is not None:
            sub = sub[sub["date"] <= pd.Timestamp(up_to_date)]

        series    = sub["value"].values
        baseline  = float(sub["baseline_28d"].iloc[-1]) if len(sub) else 0.0
        std       = float(sub["std_28d"].iloc[-1])      if len(sub) else 1.0
        return pd.Series(series), baseline, std

    def get_today_value(self, store_id: str, category: str, metric_name: str, today: date) -> float:
        self._check_built()
        mask = (
            (self._cube["store_id"]    == store_id) &
            (self._cube["category"]    == category) &
            (self._cube["metric_name"] == metric_name) &
            (self._cube["date"]        == pd.Timestamp(today))
        )
        rows = self._cube[mask]
        return float(rows["value"].iloc[0]) if len(rows) else 0.0

    def get_active_segments(self) -> list[dict]:
        """
        Returns all unique (store_id, category, tier, revenue_weight) combinations
        that have enough history to run detection.
        """
        self._check_built()
        seen = {}
        for _, row in self._cube[self._cube["metric_name"] == "net_sales"].iterrows():
            key = (row["store_id"], row["category"])
            if key not in seen:
                store_id = row["store_id"]
                rank     = (
                    int(self._meta.loc[store_id, "cluster_rank"])
                    if store_id in self._meta.index else 99
                )
                seen[key] = {
                    "store_id":       store_id,
                    "category":       row["category"],
                    "segment_key":    row["segment_key"],
                    "tier":           _assign_tier(rank),
                    "revenue_weight": row["revenue_weight"],
                    "region":         row.get("region", ""),
                }
        return list(seen.values())

    def _check_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("MetricCube.build() must be called before querying.")
