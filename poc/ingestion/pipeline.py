"""
Ingestion pipeline — simulates Kafka + schema normalization without real brokers.

In production this would be a Kafka consumer that reads Avro-encoded events from
the unified_events topic and writes normalized rows to ClickHouse.

Here we use an in-memory EventBus (a list acting as the queue) and a normalizer
that converts raw DataFrames into the UnifiedEvent envelope format.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Unified event envelope (mirrors the Avro schema in CLAUDE.md)
# ---------------------------------------------------------------------------
@dataclass
class UnifiedEvent:
    event_id:    str
    event_type:  str   # SALE | WEATHER | REVIEW | HOLIDAY | STORE_META
    store_id:    str
    region:      str
    category:    str | None
    channel:     str | None
    metric_name: str
    value:       float
    occurred_at: datetime
    ingested_at: datetime
    metadata:    dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory event bus (stands in for Kafka topic)
# ---------------------------------------------------------------------------
class EventBus:
    def __init__(self):
        self._queue: list[UnifiedEvent] = []

    def publish(self, event: UnifiedEvent) -> None:
        self._queue.append(event)

    def consume_all(self) -> list[UnifiedEvent]:
        events, self._queue = self._queue, []
        return events

    def __len__(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Normalizer: DataFrame row → UnifiedEvent
# ---------------------------------------------------------------------------
class EventNormalizer:

    @staticmethod
    def from_sale_row(row: pd.Series) -> list[UnifiedEvent]:
        """One sale row expands to three metric events: net_sales, units_sold, basket_size."""
        base = dict(
            event_type  = "SALE",
            store_id    = row["store_id"],
            region      = row["region"],
            category    = row["category"],
            channel     = row.get("channel", "instore"),
            occurred_at = pd.Timestamp(row["date"]).to_pydatetime(),
            ingested_at = datetime.utcnow(),
        )
        return [
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="net_sales",   value=row["net_sales"],   **base),
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="units_sold",  value=row["units_sold"],  **base),
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="basket_size", value=row["basket_size"], **base),
        ]

    @staticmethod
    def from_weather_row(row: pd.Series) -> list[UnifiedEvent]:
        base = dict(
            event_type  = "WEATHER",
            store_id    = "",
            region      = row["region"],
            category    = None,
            channel     = None,
            occurred_at = pd.Timestamp(row["date"]).to_pydatetime(),
            ingested_at = datetime.utcnow(),
            metadata    = {"severity_code": row["severity_code"]},
        )
        return [
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="temp_c",           value=row["temp_c"],           **base),
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="precipitation_mm",  value=row["precipitation_mm"], **base),
        ]

    @staticmethod
    def from_review_row(row: pd.Series) -> list[UnifiedEvent]:
        base = dict(
            event_type  = "REVIEW",
            store_id    = row["store_id"],
            region      = "",
            category    = None,
            channel     = None,
            occurred_at = pd.Timestamp(row["date"]).to_pydatetime(),
            ingested_at = datetime.utcnow(),
            metadata    = {"keywords": row["keywords"]},
        )
        return [
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="rating",          value=row["rating"],          **base),
            UnifiedEvent(event_id=str(uuid.uuid4()), metric_name="sentiment_score",  value=row["sentiment_score"], **base),
        ]


# ---------------------------------------------------------------------------
# Ingestion pipeline — ties bus + normalizer together
# ---------------------------------------------------------------------------
class IngestPipeline:
    def __init__(self):
        self.bus        = EventBus()
        self.normalizer = EventNormalizer()
        self.stats      = {"sales": 0, "weather": 0, "reviews": 0}

    def ingest_sales(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            for event in self.normalizer.from_sale_row(row):
                self.bus.publish(event)
        self.stats["sales"] += len(df)

    def ingest_weather(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            for event in self.normalizer.from_weather_row(row):
                self.bus.publish(event)
        self.stats["weather"] += len(df)

    def ingest_reviews(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            for event in self.normalizer.from_review_row(row):
                self.bus.publish(event)
        self.stats["reviews"] += len(df)

    def ingest_all(
        self,
        sales_df:   pd.DataFrame,
        weather_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
    ) -> dict[str, int]:
        """Ingest all signal types and return row counts."""
        self.ingest_sales(sales_df)
        self.ingest_weather(weather_df)
        self.ingest_reviews(reviews_df)
        total = sum(self.stats.values())
        return {**self.stats, "total_rows": total, "total_events": len(self.bus)}

    def drain(self) -> list[UnifiedEvent]:
        return self.bus.consume_all()
