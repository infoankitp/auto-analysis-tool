"""
End-to-end orchestrator — ties every layer together for the POC run.

Steps:
  1. Generate mock data (all signals, 60-day history + today)
  2. Inject anomaly scenarios into today's data
  3. Run ingestion pipeline
  4. Build metric cube with rolling baselines
  5. Run detection engine on all segments
  6. Route through alert manager (dedup + rate limiting)
  7. For each P1/P2 alert → run RCA (extract → prompt → Claude)
  8. Return structured results

Returns:
    {
        "alerts":      [alert dict, ...],
        "rca_results": [(alert, rca_response), ...],
        "metadata":    {timing, counts}
    }
"""

import time
import pandas as pd
from datetime import date

from poc.config import TODAY, HISTORY_START, RCA_TRIGGER_PRIORITIES, CAUSAL_WINDOW_DAYS
from poc.mock_data.generators import (
    generate_store_metadata, generate_sales_data,
    generate_weather_data, generate_reviews_data,
    generate_holiday_calendar, generate_stock_data,
)
from poc.mock_data.scenarios import inject_all_scenarios
from poc.ingestion.pipeline import IngestPipeline
from poc.compute.metric_cube import MetricCube
from poc.detection.engine import DetectionEngine
from poc.alerting.alert_manager import AlertManager
from poc.rca.extractors import (
    SalesExtractor, WeatherExtractor, HolidayExtractor,
    ReviewExtractor, StockExtractor,
)
from poc.rca.prompt_builder import build_prompt
from poc.rca.claude_client import call_claude


def run_rca_for_alert(
    alert:      dict,
    sales_df:   pd.DataFrame,
    weather_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    stock_df:   pd.DataFrame,
) -> dict:
    """Extract causal datasets and call Claude for one alert."""
    anchor = pd.Timestamp(TODAY)
    store_id = alert["store_id"]
    category = alert["category"]
    region   = alert.get("region", "")

    dataset = {
        "sales":    SalesExtractor().extract(sales_df,   store_id, category, anchor),
        "weather":  WeatherExtractor().extract(weather_df, region,  anchor),
        "reviews":  ReviewExtractor().extract(reviews_df, store_id, anchor),
        "holidays": HolidayExtractor().extract(holiday_df, anchor),
        "stock":    StockExtractor().extract(stock_df,   store_id, category, anchor),
    }

    prompt = build_prompt(alert, dataset)
    rca    = call_claude(prompt, alert)

    return {"prompt": prompt, "dataset": dataset, "rca": rca}


def run_poc(verbose: bool = True) -> dict:
    """
    Full end-to-end POC run.

    Parameters
    ----------
    verbose : if True, print progress to stdout

    Returns
    -------
    dict with keys: alerts, rca_results, raw_data, metadata
    """
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # Step 1 — Generate mock data                                         #
    # ------------------------------------------------------------------ #
    log("\n══════════════════════════════════════════════════════")
    log("  RETAIL ANOMALY DETECTION & RCA — POC RUN")
    log(f"  Detection date: {TODAY}")
    log("══════════════════════════════════════════════════════\n")
    log("▶  [1/7] Generating mock data ...")

    store_meta  = generate_store_metadata()
    raw_sales   = generate_sales_data(HISTORY_START, TODAY)
    raw_weather = generate_weather_data(HISTORY_START, TODAY)
    raw_reviews = generate_reviews_data(HISTORY_START, TODAY)
    raw_holidays = generate_holiday_calendar(HISTORY_START, TODAY)
    raw_stock   = generate_stock_data(HISTORY_START, TODAY)

    log(f"     Stores: {len(store_meta)}  |  "
        f"Sales rows: {len(raw_sales):,}  |  "
        f"Weather rows: {len(raw_weather):,}  |  "
        f"Review rows: {len(raw_reviews):,}")

    # ------------------------------------------------------------------ #
    # Step 2 — Inject anomaly scenarios                                   #
    # ------------------------------------------------------------------ #
    log("▶  [2/7] Injecting 3 anomaly scenarios into today's data ...")

    sales_df, weather_df, reviews_df, stock_df = inject_all_scenarios(
        raw_sales, raw_weather, raw_reviews, raw_stock
    )

    log("     ✦ Scenario 1: STORE_003 / Apparel   — staff shortage (-75% sales)")
    log("     ✦ Scenario 2: STORE_007 / Electronics — storm event (-65% sales)")
    log("     ✦ Scenario 3: STORE_012 / Beverages  — stock-out    (-90% sales)")

    # ------------------------------------------------------------------ #
    # Step 3 — Ingestion pipeline                                         #
    # ------------------------------------------------------------------ #
    log("▶  [3/7] Running ingestion pipeline ...")

    pipeline = IngestPipeline()
    stats    = pipeline.ingest_all(sales_df, weather_df, reviews_df)

    log(f"     Ingested: {stats['sales']:,} sale rows, "
        f"{stats['weather']:,} weather rows, "
        f"{stats['reviews']:,} review rows "
        f"→ {stats['total_events']:,} unified events")

    # ------------------------------------------------------------------ #
    # Step 4 — Metric cube                                                #
    # ------------------------------------------------------------------ #
    log("▶  [4/7] Building metric cube (28-day rolling baselines) ...")

    cube = MetricCube()
    cube.build(sales_df, store_meta)

    segments = cube.get_active_segments()
    log(f"     Active segments: {len(segments)}  (store × category combinations)")

    # ------------------------------------------------------------------ #
    # Step 5 — Detection engine                                           #
    # ------------------------------------------------------------------ #
    log("▶  [5/7] Running ensemble detection engine ...")

    engine    = DetectionEngine(cube, weather_df, raw_holidays)
    anomalies = engine.run(TODAY)

    log(f"     Anomalies detected: {len(anomalies)}")

    # ------------------------------------------------------------------ #
    # Step 6 — Alert manager                                              #
    # ------------------------------------------------------------------ #
    log("▶  [6/7] Processing alerts (dedup + rate limiting) ...")

    alert_mgr = AlertManager()
    alerts    = alert_mgr.process(anomalies)

    p1 = [a for a in alerts if a["priority"] == "P1"]
    p2 = [a for a in alerts if a["priority"] == "P2"]
    p3 = [a for a in alerts if a["priority"] == "P3"]
    log(f"     Final alerts: {len(alerts)} total  "
        f"(P1={len(p1)}, P2={len(p2)}, P3={len(p3)})")

    if alerts:
        log("\n  ┌─ ALERT SUMMARY ────────────────────────────────────────────────┐")
        for a in alerts:
            log(f"  │  {a['priority']}  {a['store_id']:10s}  {a['category']:12s}  "
                f"σ={a['sigma']:.1f}  impact={a['impact_score']:5.1f}  "
                f"Δ={a['pct_deviation']:+.0f}%  [{', '.join(a['contributing'])}]")
        log("  └────────────────────────────────────────────────────────────────┘")

    # ------------------------------------------------------------------ #
    # Step 7 — RCA workflow                                               #
    # ------------------------------------------------------------------ #
    log("\n▶  [7/7] Running RCA for P1/P2 alerts ...")

    rca_results = []
    rca_alerts  = [a for a in alerts if a["priority"] in RCA_TRIGGER_PRIORITIES]

    if not rca_alerts:
        log("     No P1/P2 alerts — no RCA triggered.")
    else:
        for alert in rca_alerts:
            log(f"\n  ┌─ RCA: {alert['alert_id']} ──────────────────────────────────────────")
            log(f"  │  {alert['store_id']} / {alert['category']} / {alert['region']}")
            log(f"  │  Priority: {alert['priority']}  |  Impact: {alert['impact_score']}  |  σ={alert['sigma']}")
            log(f"  │  Extracting causal datasets + calling Claude ...")

            result = run_rca_for_alert(
                alert, sales_df, weather_df, reviews_df, raw_holidays, stock_df
            )

            rca  = result["rca"]
            hyps = rca.get("hypotheses", [])
            acts = rca.get("actions", [])

            log(f"  │")
            log(f"  │  TOP ROOT CAUSE HYPOTHESIS:")
            if hyps:
                h1 = hyps[0]
                log(f"  │    [{h1['rank']}] {h1['cause']}")
                log(f"  │        Likelihood: {h1['likelihood']:.0%}  |  Confidence: {h1['confidence']}")
                for ev in h1.get("evidence", []):
                    log(f"  │        • {ev}")

            log(f"  │")
            log(f"  │  RECOMMENDED ACTIONS:")
            for act in acts:
                log(f"  │    [{act['urgency']}] {act['action']}")
                log(f"  │         → {act['owner_team']}  (due in {act['due_hours']}h)")

            gaps = rca.get("data_gaps", [])
            if gaps:
                log(f"  │")
                log(f"  │  DATA GAPS: {', '.join(gaps[:2])}")

            log(f"  └──────────────────────────────────────────────────────────────────")

            rca_results.append({"alert": alert, **result})

    elapsed = time.time() - t0
    log(f"\n✅ POC complete in {elapsed:.1f}s")
    log(f"   {len(alerts)} alerts raised | {len(rca_results)} RCA workflows completed\n")

    return {
        "alerts":      alerts,
        "rca_results": rca_results,
        "raw_data": {
            "sales_df":   sales_df,
            "weather_df": weather_df,
            "reviews_df": reviews_df,
            "stock_df":   stock_df,
            "holiday_df": raw_holidays,
            "store_meta": store_meta,
        },
        "metric_cube": cube,
        "metadata": {
            "detection_date":    str(TODAY),
            "segments_checked":  len(segments),
            "anomalies_raw":     len(anomalies),
            "alerts_emitted":    len(alerts),
            "rca_count":         len(rca_results),
            "elapsed_seconds":   round(elapsed, 2),
        },
    }
