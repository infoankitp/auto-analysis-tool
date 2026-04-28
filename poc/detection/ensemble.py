"""
Ensemble voter and impact scorer.

The voter combines per-detector results into a single anomaly decision
using weighted majority voting (weights from config.MODEL_WEIGHTS).

The scorer converts the voted result into a 0-100 impact score that
drives alert prioritisation (P1 / P2 / P3).
"""

import math
from poc.config import MODEL_WEIGHTS, GRAIN_MULTIPLIERS, IMPACT_SCALE_FACTOR
from poc.detection.detectors import DetectorResult


class EnsembleVoter:
    """
    Weighted majority vote: anomaly when weighted_vote >= 0.5.
    Models not in MODEL_WEIGHTS fall back to equal weighting.
    """

    def vote(self, results: list[DetectorResult]) -> dict:
        if not results:
            return {"is_anomaly": False, "sigma": 0.0, "vote_weight": 0.0, "contributing": []}

        total_weight   = 0.0
        weighted_votes = 0.0

        for r in results:
            w              = MODEL_WEIGHTS.get(r.model, 0.25)
            total_weight  += w
            weighted_votes += w * float(r.is_anomaly)

        vote_ratio = weighted_votes / max(total_weight, 1e-9)

        if vote_ratio < 0.5:
            return {
                "is_anomaly":   False,
                "sigma":        max(r.sigma for r in results),
                "vote_weight":  vote_ratio,
                "contributing": [],
                "direction":    "none",
            }

        return {
            "is_anomaly":   True,
            "sigma":        max(r.sigma for r in results if r.is_anomaly),
            "vote_weight":  vote_ratio,
            "contributing": [r.model for r in results if r.is_anomaly],
            "direction":    next((r.direction for r in results if r.is_anomaly), "none"),
            "detector_details": {r.model: r.detail for r in results},
        }


class ImpactScorer:
    """
    impact_score = min(100, σ × revenue_weight × novelty_factor × grain_multiplier × SCALE)

    novelty_factor decays for repeat anomalies; starts at 1.0.
    """

    def compute(
        self,
        sigma:          float,
        revenue_weight: float,
        grain:          str   = "daily",
        repeat_count:   int   = 0,
    ) -> float:
        novelty_factor   = max(0.10, 1.0 / (1 + math.log1p(repeat_count)))
        grain_multiplier = GRAIN_MULTIPLIERS.get(grain, 1.0)
        raw              = sigma * revenue_weight * novelty_factor * grain_multiplier * IMPACT_SCALE_FACTOR
        return round(min(100.0, raw), 2)
