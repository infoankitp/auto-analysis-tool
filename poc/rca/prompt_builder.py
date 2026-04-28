"""
Prompt builder — assembles the structured LLM prompt for RCA.

The prompt is deterministic: given the same alert + dataset, it produces
the same prompt text.  Claude is instructed to return only valid JSON.
"""

import json
from poc.config import CAUSAL_WINDOW_DAYS

SYSTEM_PROMPT = """You are a senior retail analytics expert conducting Root Cause Analysis (RCA) for a large omnichannel retail chain.

Your job is to analyze anomaly signals and determine the most likely root cause of a sales drop, ranking hypotheses by likelihood and providing concrete evidence from the data signals provided.

You must respond with ONLY a valid JSON object — no preamble, no explanation outside the JSON, no markdown fences.

Consider these common retail root causes and their evidence patterns:
- Staff shortage: poor reviews, keywords like 'understaffed', 'long wait', sudden rating drop, sentiment decline
- Weather/external: severe precipitation or weather severity codes, region-wide impact
- Stock unavailability: stock_level near 0, supplier_flag = 'disrupted', units_sold near 0
- Pricing/competitive: basket_size changes, unusual units-to-revenue ratio
- System/operational: isolated store impact with no external signal
"""

PROMPT_TEMPLATE = """\
ANOMALY ALERT DETAILS:
======================
Store:            {store_id}
Category:         {category}
Region:           {region}
Metric:           net_sales
Current Value:    ${current_value:,.0f}
28-day Baseline:  ${baseline_28d:,.0f}
% Deviation:      {pct_deviation:.1f}%
Sigma (severity): {sigma:.2f}σ
Impact Score:     {impact_score:.1f} / 100
Priority:         {priority}
Detection Time:   {triggered_at}
Models Flagged:   {contributing}

CAUSAL SIGNALS (last {window_days} days):
==========================================

[SALES TREND]
{sales_summary}

[WEATHER CONTEXT]
{weather_summary}

[STAFF REVIEWS]
{review_summary}

[STOCK LEVELS]
{stock_summary}

[HOLIDAY CALENDAR]
{holiday_summary}

INSTRUCTIONS:
=============
1. Identify the top 3 most plausible root causes, ranked by likelihood (0.0-1.0 where 1.0 = certain)
2. For each cause, cite SPECIFIC data points from the signals above as evidence
3. Flag any data gaps that would improve your confidence
4. Recommend 1-3 immediate actions with owning team and urgency level

Return ONLY a JSON object with this exact structure:
{{
  "hypotheses": [
    {{
      "rank":       1,
      "cause":      "Short cause title",
      "likelihood": 0.87,
      "evidence":   ["specific evidence point 1", "specific evidence point 2"],
      "confidence": "high"
    }}
  ],
  "actions": [
    {{
      "action":     "What to do immediately",
      "owner_team": "Team name",
      "urgency":    "P1",
      "due_hours":  4
    }}
  ],
  "data_gaps": ["Gap 1 that would improve confidence", "Gap 2"]
}}
"""


def _format_sales(s: dict) -> str:
    if s.get("status") == "no_data":
        return "  No sales data available."
    lines = [
        f"  Today:        ${s['today_net_sales']:,.0f}",
        f"  14-day mean:  ${s['history_mean_28d']:,.0f}  (±${s['history_std_28d']:,.0f})",
        f"  Drop vs mean: {s['pct_drop_vs_mean']:.1f}%",
        f"  Consec. low days: {s['consecutive_low_days']}",
        "  Last 7 days:",
    ]
    for d in s.get("daily_series", []):
        lines.append(f"    {d['date']}: ${d['net_sales']:,.0f}  ({d['units_sold']} units)")
    return "\n".join(lines)


def _format_weather(w: dict) -> str:
    if w.get("status") == "no_data":
        return "  No weather data available."
    lines = [
        f"  Today severity:      {w['today_severity']}",
        f"  Today precipitation: {w['today_precipitation_mm']} mm",
        f"  14-day avg precip:   {w['avg_precipitation_14d']} mm",
        f"  14-day max precip:   {w['max_precipitation_14d']} mm",
        f"  Weather anomaly:     {'YES' if w['weather_anomaly_flagged'] else 'No'}",
        "  Recent 5 days:",
    ]
    for d in w.get("recent_5_days", []):
        lines.append(f"    {d['date']}: {d['severity_code']}, {d['precipitation_mm']}mm, {d['temp_c']}°C")
    return "\n".join(lines)


def _format_reviews(r: dict) -> str:
    if r.get("status") == "no_data":
        return "  No review data available."
    lines = [
        f"  Today rating:     {r['today_rating']} / 5.0",
        f"  Today sentiment:  {r['today_sentiment']:.3f}  (range -1 to +1)",
        f"  14-day avg rating:    {r['avg_hist_rating_14d']}",
        f"  14-day avg sentiment: {r['avg_hist_sentiment_14d']:.3f}",
        f"  Rating delta:     {r['rating_delta']:+.2f}",
        f"  Keywords today:   {', '.join(r['negative_keywords'])}",
        "  Recent 5 days:",
    ]
    for d in r.get("recent_5_days", []):
        lines.append(
            f"    {d['date']}: rating={d['rating']}, sentiment={d['sentiment_score']:.3f}  [{d['keywords']}]"
        )
    return "\n".join(lines)


def _format_stock(s: dict) -> str:
    if s.get("status") == "no_data":
        return "  No stock data available."
    return "\n".join([
        f"  Today stock level:   {s['today_stock_level']:.3f}  (0=empty, 1=full)",
        f"  14-day avg stock:    {s['avg_stock_level_14d']:.3f}",
        f"  Below reorder point: {'YES' if s['stock_below_reorder'] else 'No'}",
        f"  Effectively empty:   {'YES — shelves bare' if s['effectively_empty'] else 'No'}",
        f"  Supplier flag:       {s['supplier_flag']}",
    ])


def _format_holidays(h: dict) -> str:
    if not h.get("holidays_in_window"):
        return "  No major holidays in the 14-day window."
    lines = [f"  Max holiday impact in window: {h['max_impact']:.2f}"]
    for ho in h["holidays_in_window"]:
        lines.append(f"    {ho['date']}: {ho['name']} (impact={ho['impact_score']})")
    return "\n".join(lines)


def build_prompt(alert: dict, dataset: dict) -> str:
    return PROMPT_TEMPLATE.format(
        store_id       = alert["store_id"],
        category       = alert["category"],
        region         = alert.get("region", "N/A"),
        current_value  = alert["current_value"],
        baseline_28d   = alert["baseline_28d"],
        pct_deviation  = alert["pct_deviation"],
        sigma          = alert["sigma"],
        impact_score   = alert["impact_score"],
        priority       = alert["priority"],
        triggered_at   = alert["triggered_at"],
        contributing   = ", ".join(alert.get("contributing", [])),
        window_days    = CAUSAL_WINDOW_DAYS,
        sales_summary  = _format_sales(dataset.get("sales", {})),
        weather_summary = _format_weather(dataset.get("weather", {})),
        review_summary = _format_reviews(dataset.get("reviews", {})),
        stock_summary  = _format_stock(dataset.get("stock", {})),
        holiday_summary = _format_holidays(dataset.get("holidays", {})),
    )
