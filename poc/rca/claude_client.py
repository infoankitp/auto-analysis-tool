"""
Claude API client for RCA.

Calls the Anthropic Messages API with the assembled prompt and returns
a parsed dict with 'hypotheses', 'actions', and 'data_gaps'.

Features:
  - Prompt caching on the system prompt (static content)
  - Graceful fallback to DEMO_RESPONSE when ANTHROPIC_API_KEY is absent
  - Structured JSON output enforced by explicit instruction in the prompt
"""

import json
import os

from poc.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
from poc.rca.prompt_builder import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Demo / fallback response (used when no API key is configured)
# ---------------------------------------------------------------------------
def _demo_response(alert: dict) -> dict:
    store_id  = alert.get("store_id", "?")
    category  = alert.get("category", "?")
    deviation = alert.get("pct_deviation", 0)
    scenario  = _infer_scenario(alert)
    return DEMO_RESPONSES.get(scenario, DEMO_RESPONSES["default"])


def _infer_scenario(alert: dict) -> str:
    store_id = alert.get("store_id", "")
    if store_id == "STORE_003":
        return "staff_shortage"
    elif store_id == "STORE_007":
        return "weather_event"
    elif store_id == "STORE_012":
        return "stock_outage"
    return "default"


DEMO_RESPONSES = {
    "staff_shortage": {
        "hypotheses": [
            {
                "rank": 1,
                "cause": "Severe staff shortage causing service failure and customer walkouts",
                "likelihood": 0.91,
                "evidence": [
                    "Staff review keywords: 'understaffed', 'no staff', 'walked out', 'long queue' appear in 100% of recent reviews",
                    "Average store rating crashed from 4.2 → 2.0 over the last 3 days — a 2.2-point decline",
                    "Sentiment score dropped from +0.32 → -0.68 (negative territory for 3 consecutive days)",
                    "Sales dropped 75% with no corresponding weather event or holiday",
                    "Units sold declined proportionally, ruling out a pricing/basket-size issue"
                ],
                "confidence": "high"
            },
            {
                "rank": 2,
                "cause": "Management or scheduling system failure leading to under-staffing",
                "likelihood": 0.06,
                "evidence": [
                    "Pattern is consistent with a scheduling error rather than voluntary walkout",
                    "No region-wide signal — impact isolated to STORE_003"
                ],
                "confidence": "low"
            },
            {
                "rank": 3,
                "cause": "Localised competitor event drawing away foot traffic",
                "likelihood": 0.03,
                "evidence": [
                    "No corroborating competitor data available",
                    "Review keywords focus on internal service issues, not external draw"
                ],
                "confidence": "low"
            }
        ],
        "actions": [
            {
                "action": "Deploy emergency staffing from nearby STORE_001 and STORE_002 immediately",
                "owner_team": "Regional Store Operations",
                "urgency": "P1",
                "due_hours": 2
            },
            {
                "action": "Activate shift-fill protocol — contact part-time staff pool for same-day coverage",
                "owner_team": "Store Manager / HR",
                "urgency": "P1",
                "due_hours": 3
            },
            {
                "action": "Root cause debrief with store manager to prevent recurrence — review scheduling system",
                "owner_team": "District Manager",
                "urgency": "P2",
                "due_hours": 24
            }
        ],
        "data_gaps": [
            "Scheduled vs actual headcount for today (HR system)",
            "Clocking-in records to confirm actual staff on floor",
            "Whether a sudden resignation or sick-out event was logged",
            "Foot traffic sensor data to distinguish walkins vs walkaways"
        ]
    },

    "weather_event": {
        "hypotheses": [
            {
                "rank": 1,
                "cause": "Heavy storm (38mm precipitation) sharply reduced in-store foot traffic in the MW region",
                "likelihood": 0.88,
                "evidence": [
                    "Weather severity upgraded to 'storm' today vs 14-day average of 2mm precipitation",
                    "38mm precipitation today is 19x the 14-day average — an extreme weather day",
                    "Sales drop of 65% aligns with expected storm-day traffic reduction patterns",
                    "Review sentiment unchanged — no service complaints — rules out operational issues",
                    "Stock levels normal — rules out supply-side cause"
                ],
                "confidence": "high"
            },
            {
                "rank": 2,
                "cause": "Regional internet/power outage affecting POS systems during storm",
                "likelihood": 0.08,
                "evidence": [
                    "Storm events frequently cause power disruptions in MW suburban locations",
                    "Would explain near-zero transactions even if some customers arrived"
                ],
                "confidence": "low"
            },
            {
                "rank": 3,
                "cause": "Competing online channel cannibalisation on a high-precipitation day",
                "likelihood": 0.04,
                "evidence": [
                    "Electronics category has high online substitutability",
                    "Online channel data not available to confirm"
                ],
                "confidence": "low"
            }
        ],
        "actions": [
            {
                "action": "Monitor online channel sales to confirm storm-driven channel shift vs total demand loss",
                "owner_team": "Digital Commerce Analytics",
                "urgency": "P1",
                "due_hours": 2
            },
            {
                "action": "Alert inventory team: expect demand surge in the 2-3 days post-storm (pent-up demand)",
                "owner_team": "Supply Chain / Replenishment",
                "urgency": "P2",
                "due_hours": 8
            },
            {
                "action": "No staffing or stock changes needed — monitor closely through the next 48 hours",
                "owner_team": "Regional Operations",
                "urgency": "P3",
                "due_hours": 48
            }
        ],
        "data_gaps": [
            "Online channel sales for MW region on the same day",
            "Foot traffic sensor data from store entrance",
            "Power/POS system uptime log for STORE_007",
            "Competitor store closures during the storm"
        ]
    },

    "stock_outage": {
        "hypotheses": [
            {
                "rank": 1,
                "cause": "Complete supplier failure leaving Beverages shelves empty (stock_level = 0.03)",
                "likelihood": 0.93,
                "evidence": [
                    "Stock level at 0.03 — effectively zero (reorder threshold is 0.20)",
                    "Supplier flag set to 'disrupted' — confirms supply chain issue, not demand failure",
                    "Units sold = 1-3 units (residual backstock only) vs 40+ on a normal day",
                    "Sales dropped 90% with no weather event, no review deterioration",
                    "Review keyword 'out of stock', 'empty shelves' confirms customer-visible stock-out"
                ],
                "confidence": "high"
            },
            {
                "rank": 2,
                "cause": "Receiving/logistics failure — stock may be at warehouse but not on shelves",
                "likelihood": 0.05,
                "evidence": [
                    "Supplier disruption flag could indicate delayed delivery vs true stock-out",
                    "Distinction matters for urgency of supplier escalation vs restocking from nearby DC"
                ],
                "confidence": "low"
            },
            {
                "rank": 3,
                "cause": "Product recall or regulatory hold on Beverages SKUs",
                "likelihood": 0.02,
                "evidence": [
                    "Stock removal without supplier disruption flag would indicate this",
                    "No regulatory recall data available to confirm or deny"
                ],
                "confidence": "low"
            }
        ],
        "actions": [
            {
                "action": "Immediately contact primary Beverages supplier to determine ETA for restocking",
                "owner_team": "Supply Chain / Procurement",
                "urgency": "P1",
                "due_hours": 2
            },
            {
                "action": "Arrange emergency transfer of Beverages stock from STORE_004 or STORE_009 (SE region)",
                "owner_team": "Regional Distribution / Store Ops",
                "urgency": "P1",
                "due_hours": 4
            },
            {
                "action": "Update customer-facing systems (app, store website) to reflect Beverages unavailability",
                "owner_team": "Digital Commerce / Customer Experience",
                "urgency": "P2",
                "due_hours": 6
            }
        ],
        "data_gaps": [
            "Warehouse inventory levels for Beverages SKUs at SE distribution centre",
            "Supplier communication log — when was the disruption first reported?",
            "Whether same supplier disruption affects other SE region stores",
            "Check if a product recall notice was issued for any Beverages SKUs"
        ]
    },

    "default": {
        "hypotheses": [
            {
                "rank": 1,
                "cause": "Operational or demand anomaly requiring further investigation",
                "likelihood": 0.6,
                "evidence": ["Significant statistical deviation from 28-day baseline detected"],
                "confidence": "low"
            }
        ],
        "actions": [
            {
                "action": "Investigate store operations and check all signal sources",
                "owner_team": "Store Operations",
                "urgency": "P2",
                "due_hours": 8
            }
        ],
        "data_gaps": ["Additional operational data needed to determine root cause"]
    }
}


# ---------------------------------------------------------------------------
# Live Claude API client
# ---------------------------------------------------------------------------
def call_claude(prompt: str, alert: dict) -> dict:
    """
    Call Claude API for RCA.  Falls back to DEMO_RESPONSES if no API key.

    Returns dict with keys: hypotheses, actions, data_gaps.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("    [INFO] ANTHROPIC_API_KEY not set — using demo RCA response.")
        return _demo_response(alert)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = CLAUDE_MAX_TOKENS,
            system     = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages = [{"role": "user", "content": prompt}],
        )

        raw   = response.content[0].text
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)

    except json.JSONDecodeError as e:
        print(f"    [WARN] Claude returned non-JSON — using demo response. Error: {e}")
        return _demo_response(alert)
    except Exception as e:
        print(f"    [WARN] Claude API error ({e}) — using demo response.")
        return _demo_response(alert)
