"""
Central configuration for the Retail Anomaly Detection POC.
All tuneable constants live here — no magic numbers scattered through the codebase.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
TODAY         = date.today()
HISTORY_DAYS  = 60
HISTORY_START = TODAY - timedelta(days=HISTORY_DAYS)
HISTORY_END   = TODAY - timedelta(days=1)

# ---------------------------------------------------------------------------
# Stores & Dimensions
# ---------------------------------------------------------------------------
STORES = [
    {"store_id": "STORE_001", "region": "NE", "cluster_type": "urban",    "revenue_tier": "high"},
    {"store_id": "STORE_002", "region": "MW", "cluster_type": "suburban", "revenue_tier": "high"},
    {"store_id": "STORE_003", "region": "NE", "cluster_type": "urban",    "revenue_tier": "high"},   # scenario: staff shortage
    {"store_id": "STORE_004", "region": "SE", "cluster_type": "rural",    "revenue_tier": "medium"},
    {"store_id": "STORE_005", "region": "MW", "cluster_type": "urban",    "revenue_tier": "medium"},
    {"store_id": "STORE_007", "region": "MW", "cluster_type": "suburban", "revenue_tier": "high"},   # scenario: weather event
    {"store_id": "STORE_012", "region": "SE", "cluster_type": "suburban", "revenue_tier": "medium"}, # scenario: stock-out
]

CATEGORIES = ["Apparel", "Electronics", "Beverages", "Home", "Sports"]
REGIONS    = ["NE", "MW", "SE", "SW", "W"]

# ---------------------------------------------------------------------------
# Revenue weights — used in impact score formula
# ---------------------------------------------------------------------------
REVENUE_WEIGHTS           = {"high": 0.94, "medium": 0.65, "low": 0.30}
STORE_REVENUE_MULTIPLIER  = {"high": 1.0,  "medium": 0.60, "low": 0.30}

# Base daily revenue per category (before store multiplier and seasonality)
CATEGORY_BASELINE_REVENUE = {
    "Apparel":     150_000,
    "Electronics": 200_000,
    "Beverages":    80_000,
    "Home":        120_000,
    "Sports":       60_000,
}

# Mon-Sun seasonal multipliers
WEEKLY_SEASONAL_PATTERN = [0.80, 0.85, 0.90, 0.95, 1.20, 1.40, 0.70]

# ---------------------------------------------------------------------------
# Anomaly scenarios injected into today's data
# ---------------------------------------------------------------------------
ANOMALY_SCENARIOS = {
    "staff_shortage": {
        "store_id":     "STORE_003",
        "category":     "Apparel",
        "region":       "NE",
        "description":  "Severe staff shortage — long queues, poor service, walkouts",
        "sales_factor": 0.25,   # drop to 25% of expected baseline
    },
    "weather_event": {
        "store_id":     "STORE_007",
        "category":     "Electronics",
        "region":       "MW",
        "description":  "Heavy storm sharply reduced in-store foot traffic",
        "sales_factor": 0.35,
    },
    "stock_outage": {
        "store_id":     "STORE_012",
        "category":     "Beverages",
        "region":       "SE",
        "description":  "Supplier failure — entire Beverages aisle out of stock",
        "sales_factor": 0.10,
    },
}

# ---------------------------------------------------------------------------
# Detector profiles by tier
# ---------------------------------------------------------------------------
DETECTOR_PROFILES = {
    "tier1": {
        "models":          ["stl", "zscore", "isolation_forest"],
        "threshold_sigma": 2.5,
        "min_periods":     28,
        "context_guard":   True,
    },
    "tier2": {
        "models":          ["stl", "zscore", "isolation_forest"],
        "threshold_sigma": 3.0,
        "min_periods":     14,
        "context_guard":   True,
    },
    "tier3": {
        "models":          ["zscore"],
        "threshold_sigma": 3.5,
        "min_periods":     7,
        "context_guard":   False,
    },
}

MODEL_WEIGHTS = {
    "stl":              0.40,
    "zscore":           0.30,
    "isolation_forest": 0.30,
}

GRAIN_MULTIPLIERS = {
    "hourly":  0.6,
    "daily":   1.0,
    "weekly":  1.3,
    "monthly": 1.5,
}

# Scales raw sigma×weight product into a 0-100 impact range
IMPACT_SCALE_FACTOR = 15.0

# ---------------------------------------------------------------------------
# Alert thresholds & rate limits
# ---------------------------------------------------------------------------
ALERT_PRIORITY_THRESHOLDS      = {"P1": 75, "P2": 40, "P3": 0}
MAX_ALERTS_PER_SEGMENT_PER_HOUR = 3
DEDUP_WINDOW_SECS              = 3600

# ---------------------------------------------------------------------------
# Context guard — suppress detections caused by known external events
# ---------------------------------------------------------------------------
HOLIDAY_SUPPRESS_THRESHOLD      = 0.7
WEATHER_SUPPRESS_SEVERITY_CODES = {"blizzard", "hurricane", "ice_storm"}

# ---------------------------------------------------------------------------
# RCA
# ---------------------------------------------------------------------------
RCA_TRIGGER_PRIORITIES = {"P1", "P2"}
CAUSAL_WINDOW_DAYS     = 14

# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------
CLAUDE_MODEL      = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 2048
