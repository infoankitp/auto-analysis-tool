"""
Detection engine — orchestrates the full detect-vote-score pipeline for every
active store × category segment.

Flow per segment:
  1. Look up tier from metric cube
  2. Pull historical series + baseline/std from metric cube
  3. Run each detector specified by the tier profile
  4. Apply context guard (holiday / weather suppression) if tier enables it
  5. Ensemble vote across detector results
  6. Compute impact score
  7. If anomaly → emit AnomalyCandidate

Returns a list of AnomalyCandidate dicts ready for the AlertManager.
"""

import uuid
from datetime import date

import pandas as pd

from poc.config import (
    DETECTOR_PROFILES, ALERT_PRIORITY_THRESHOLDS,
)
from poc.compute.metric_cube import MetricCube
from poc.detection.detectors import (
    ZScoreDetector, STLDetector, IsolationForestDetector, DetectorResult,
)
from poc.detection.context_guard import ContextGuard
from poc.detection.ensemble import EnsembleVoter, ImpactScorer


DETECTOR_REGISTRY = {
    "zscore":           ZScoreDetector,
    "stl":              STLDetector,
    "isolation_forest": IsolationForestDetector,
}


def _priority(impact_score: float) -> str:
    for level in ("P1", "P2", "P3"):
        if impact_score >= ALERT_PRIORITY_THRESHOLDS[level]:
            return level
    return "P3"


class DetectionEngine:
    """
    Runs the full detection pipeline on a given date for all active segments.

    Parameters
    ----------
    cube        : MetricCube   — built and ready
    weather_df  : pd.DataFrame — needed by ContextGuard
    holiday_df  : pd.DataFrame — needed by ContextGuard
    grain       : str          — 'daily' for this POC
    """

    def __init__(
        self,
        cube:       MetricCube,
        weather_df: pd.DataFrame,
        holiday_df: pd.DataFrame,
        grain:      str = "daily",
    ):
        self.cube         = cube
        self.guard        = ContextGuard(weather_df, holiday_df)
        self.voter        = EnsembleVoter()
        self.scorer       = ImpactScorer()
        self.grain        = grain

    def run(self, detection_date: date) -> list[dict]:
        """
        Run detection for all segments on detection_date.
        Returns list of anomaly dicts (only truly anomalous segments included).
        """
        anomalies    = []
        segments     = self.cube.get_active_segments()
        today_ts     = pd.Timestamp(detection_date)
        metric       = "net_sales"

        for seg in segments:
            store_id       = seg["store_id"]
            category       = seg["category"]
            tier           = seg["tier"]
            revenue_weight = seg["revenue_weight"]
            profile        = DETECTOR_PROFILES[tier]

            series, baseline, std = self.cube.get_segment_series(
                store_id, category, metric, up_to_date=detection_date
            )

            if len(series) < profile["min_periods"]:
                continue    # not enough history — skip silently

            # --- Run detectors ---
            results: list[DetectorResult] = []
            for model_name in profile["models"]:
                cls    = DETECTOR_REGISTRY[model_name]
                inst   = cls(threshold_sigma=profile["threshold_sigma"])
                result = inst.detect(series, baseline, std)
                results.append(result)

            # --- Context guard ---
            suppression_reason = None
            if profile.get("context_guard"):
                results, suppression_reason = self.guard.apply(results, seg.get("region", ""), today_ts)

            # --- Ensemble vote ---
            vote = self.voter.vote(results)
            if not vote["is_anomaly"]:
                continue

            # --- Impact score ---
            impact_score = self.scorer.compute(
                sigma          = vote["sigma"],
                revenue_weight = revenue_weight,
                grain          = self.grain,
            )

            anomalies.append({
                "alert_id":           f"alt_{detection_date.strftime('%Y%m%d')}_{str(uuid.uuid4())[:6]}",
                "store_id":           store_id,
                "category":           category,
                "segment_key":        f"{seg['segment_key']}|grain={self.grain}",
                "metric_name":        metric,
                "sigma":              round(vote["sigma"], 2),
                "impact_score":       impact_score,
                "priority":           _priority(impact_score),
                "revenue_weight":     revenue_weight,
                "tier":               tier,
                "vote_weight":        round(vote["vote_weight"], 3),
                "contributing":       vote.get("contributing", []),
                "direction":          vote.get("direction", "down"),
                "suppressed_by":      suppression_reason,
                "triggered_at":       today_ts.isoformat(),
                "detector_details":   vote.get("detector_details", {}),
                "current_value":      round(float(series.iloc[-1]), 2),
                "baseline_28d":       round(baseline, 2),
                "pct_deviation":      round((float(series.iloc[-1]) - baseline) / max(baseline, 1) * 100, 1),
                "region":             seg.get("region", ""),
            })

        # Sort by impact score descending
        anomalies.sort(key=lambda x: x["impact_score"], reverse=True)
        return anomalies
