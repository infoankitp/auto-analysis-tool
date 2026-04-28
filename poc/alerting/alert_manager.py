"""
Alert manager — deduplication, rate limiting, and final alert emission.

In production:
  - Dedup uses a Bloom filter + Redis TTL (1-hour window)
  - Rate limit uses Redis incr/expire per segment per hour
  - Alerts are published to the anomaly_alerts Kafka topic

Here we use:
  - A set() for fingerprint dedup
  - A simple counter dict for rate limiting
  - An in-memory list as the "alerts" topic
"""

import hashlib
from datetime import datetime
from poc.config import MAX_ALERTS_PER_SEGMENT_PER_HOUR


class AlertManager:
    """
    Processes anomaly candidates from the detection engine and produces
    a clean, deduplicated alert list.
    """

    def __init__(self):
        self._seen_fingerprints: set[str]       = set()
        self._rate_counters:     dict[str, int] = {}
        self._alerts:            list[dict]     = []

    def process(self, anomaly_candidates: list[dict]) -> list[dict]:
        """
        Filter candidates through dedup and rate limiting.
        Returns the accepted alerts.
        """
        accepted = []
        for candidate in anomaly_candidates:
            fp = self._fingerprint(candidate)

            if self._is_duplicate(fp):
                candidate["_skip_reason"] = "dedup"
                continue

            if self._is_rate_limited(candidate["segment_key"]):
                candidate["_skip_reason"] = "rate_limited"
                continue

            self._seen_fingerprints.add(fp)
            self._increment_rate(candidate["segment_key"])

            alert = {**candidate, "fingerprint": fp, "emitted_at": datetime.utcnow().isoformat()}
            self._alerts.append(alert)
            accepted.append(alert)

        return accepted

    def get_all_alerts(self) -> list[dict]:
        return list(self._alerts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(candidate: dict) -> str:
        key = f"{candidate['segment_key']}|{candidate['metric_name']}|{candidate['triggered_at'][:10]}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, fingerprint: str) -> bool:
        return fingerprint in self._seen_fingerprints

    def _is_rate_limited(self, segment_key: str) -> bool:
        return self._rate_counters.get(segment_key, 0) >= MAX_ALERTS_PER_SEGMENT_PER_HOUR

    def _increment_rate(self, segment_key: str) -> None:
        self._rate_counters[segment_key] = self._rate_counters.get(segment_key, 0) + 1
