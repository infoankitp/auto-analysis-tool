"""
Anomaly detectors — each implements the BaseDetector interface.

Three detectors are available:
  ZScoreDetector       — fast, cheap, always runs
  STLDetector          — seasonal decomposition; needs ≥ 14 data points
  IsolationForestDetector — multivariate; needs ≥ 20 data points

Each detector.detect() returns a DetectorResult dataclass.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DetectorResult:
    model:      str
    is_anomaly: bool
    sigma:      float   # standardized deviation; higher = more anomalous
    direction:  str     # "down" | "up" | "none"
    detail:     str     # human-readable explanation


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class BaseDetector:
    name: str = "base"

    def detect(self, series: pd.Series, baseline: float, std: float) -> DetectorResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Z-score detector
# ---------------------------------------------------------------------------
class ZScoreDetector(BaseDetector):
    """
    Classic z-score on the most recent value against the 28-day rolling baseline.
    Cheap, always-on, works with as few as 7 data points.
    """
    name = "zscore"

    def __init__(self, threshold_sigma: float = 3.0):
        self.threshold_sigma = threshold_sigma

    def detect(self, series: pd.Series, baseline: float, std: float) -> DetectorResult:
        if len(series) == 0:
            return DetectorResult("zscore", False, 0.0, "none", "Empty series")

        current = float(series.iloc[-1])
        safe_std = max(std, 1e-6)
        z = (current - baseline) / safe_std

        is_anomaly = abs(z) >= self.threshold_sigma
        direction  = "down" if z < 0 else ("up" if z > 0 else "none")
        detail     = (
            f"current={current:,.0f}, baseline={baseline:,.0f}, "
            f"std={safe_std:,.0f}, z={z:.2f} ({'anomaly' if is_anomaly else 'normal'})"
        )
        return DetectorResult("zscore", is_anomaly, abs(z), direction, detail)


# ---------------------------------------------------------------------------
# STL decomposition detector
# ---------------------------------------------------------------------------
class STLDetector(BaseDetector):
    """
    STL (Seasonal-Trend decomposition using LOESS) extracts the trend and
    seasonal components; the residual is checked for outliers.

    Requires statsmodels and ≥ 2 complete weekly cycles (14+ points).
    """
    name = "stl"

    def __init__(self, threshold_sigma: float = 2.5, period: int = 7):
        self.threshold_sigma = threshold_sigma
        self.period          = period

    def detect(self, series: pd.Series, baseline: float, std: float) -> DetectorResult:
        from statsmodels.tsa.seasonal import STL

        min_length = self.period * 2
        if len(series) < min_length:
            return DetectorResult(
                "stl", False, 0.0, "none",
                f"Insufficient data: {len(series)} < {min_length} points required"
            )

        values = series.values.astype(float)

        try:
            result    = STL(values, period=self.period, robust=True).fit()
            residuals = result.resid
        except Exception as exc:
            return DetectorResult("stl", False, 0.0, "none", f"STL fit failed: {exc}")

        hist_std  = float(np.std(residuals[:-1])) + 1e-9
        sigma     = abs(residuals[-1]) / hist_std
        current   = float(series.iloc[-1])
        direction = "down" if residuals[-1] < 0 else "up"

        is_anomaly = sigma >= self.threshold_sigma
        detail     = (
            f"STL residual={residuals[-1]:,.0f}, hist_std={hist_std:,.0f}, "
            f"sigma={sigma:.2f}, trend={result.trend[-1]:,.0f}"
        )
        return DetectorResult("stl", is_anomaly, float(sigma), direction, detail)


# ---------------------------------------------------------------------------
# Isolation Forest detector
# ---------------------------------------------------------------------------
class IsolationForestDetector(BaseDetector):
    """
    Multivariate anomaly detector using sklearn's Isolation Forest.

    Features: [value, day_of_week_sin, day_of_week_cos, ratio_to_baseline]
    Fits on all historical points (excluding today) and scores today.

    Requires ≥ 20 historical points.
    """
    name = "isolation_forest"

    def __init__(self, threshold_sigma: float = 3.0, contamination: float = 0.05):
        self.threshold_sigma = threshold_sigma
        self.contamination   = contamination

    def detect(self, series: pd.Series, baseline: float, std: float) -> DetectorResult:
        from sklearn.ensemble import IsolationForest

        if len(series) < 20:
            return DetectorResult(
                "isolation_forest", False, 0.0, "none",
                f"Insufficient data: {len(series)} < 20 points required"
            )

        values   = series.values.astype(float)
        safe_bl  = max(baseline, 1.0)
        n        = len(values)

        # Feature matrix: one row per day
        dow_idx = np.arange(n)
        X = np.column_stack([
            values,
            np.sin(2 * np.pi * dow_idx / 7),
            np.cos(2 * np.pi * dow_idx / 7),
            values / safe_bl,
        ])

        try:
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            clf.fit(X[:-1])                        # train on history
            score = clf.score_samples(X[-1:])      # score today
        except Exception as exc:
            return DetectorResult("isolation_forest", False, 0.0, "none", f"IF fit failed: {exc}")

        # score_samples returns negative avg path length; more negative = more anomalous
        # Map to sigma-like scale: normal ≈ 0.0, anomaly < -0.5
        anomaly_score = -float(score[0])           # flip sign: higher = more anomalous
        sigma         = max(0.0, (anomaly_score - 0.45) / 0.05)  # rough calibration

        current   = float(series.iloc[-1])
        direction = "down" if current < baseline else "up"
        is_anomaly = anomaly_score > 0.60          # empirical threshold for IF score

        detail = (
            f"IF score={anomaly_score:.3f} ({'anomaly' if is_anomaly else 'normal'}), "
            f"current={current:,.0f}, baseline={baseline:,.0f}"
        )
        return DetectorResult("isolation_forest", is_anomaly, sigma, direction, detail)
