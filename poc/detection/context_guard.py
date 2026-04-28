"""
Context guard — suppresses anomaly detections that are explained by
known external events (major holidays, extreme weather).

Applied AFTER individual detectors but BEFORE ensemble voting.
Suppressed results still flow through the pipeline with is_anomaly=False
and a suppressed_by tag so they are visible in dashboards.
"""

import pandas as pd
from poc.config import (
    HOLIDAY_SUPPRESS_THRESHOLD,
    WEATHER_SUPPRESS_SEVERITY_CODES,
)
from poc.detection.detectors import DetectorResult


class ContextGuard:
    """
    Checks holiday and weather context for a given segment and date.
    If a known high-impact event is present, all detector results are
    flipped to is_anomaly=False with a suppression reason.
    """

    def __init__(self, weather_df: pd.DataFrame, holiday_df: pd.DataFrame):
        self._weather  = weather_df
        self._holidays = holiday_df

    def apply(
        self,
        results:  list[DetectorResult],
        region:   str,
        check_date: pd.Timestamp,
    ) -> tuple[list[DetectorResult], str | None]:
        """
        Returns (results, suppression_reason).
        suppression_reason is None if no suppression occurred.
        """
        reason = self._check_holiday(check_date)
        if reason is None:
            reason = self._check_weather(region, check_date)

        if reason is None:
            return results, None

        suppressed = [
            DetectorResult(
                model      = r.model,
                is_anomaly = False,
                sigma      = r.sigma,
                direction  = r.direction,
                detail     = f"[SUPPRESSED — {reason}] {r.detail}",
            )
            for r in results
        ]
        return suppressed, reason

    def _check_holiday(self, check_date: pd.Timestamp) -> str | None:
        if self._holidays.empty:
            return None
        mask = self._holidays["date"] == check_date
        row  = self._holidays[mask]
        if not row.empty:
            impact = float(row["impact_score"].iloc[0])
            if impact >= HOLIDAY_SUPPRESS_THRESHOLD:
                return f"holiday:{row['holiday_name'].iloc[0]} (impact={impact:.2f})"
        return None

    def _check_weather(self, region: str, check_date: pd.Timestamp) -> str | None:
        if self._weather.empty:
            return None
        mask = (self._weather["region"] == region) & (self._weather["date"] == check_date)
        row  = self._weather[mask]
        if not row.empty:
            severity = row["severity_code"].iloc[0]
            if severity in WEATHER_SUPPRESS_SEVERITY_CODES:
                return f"weather:{severity}"
        return None
