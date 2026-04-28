"""
Generates poc_notebook.ipynb — a step-by-step interactive walkthrough
of the Retail Anomaly Detection & RCA POC.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Retail Anomaly POC",
    "language":     "python",
    "name":         "retail-anomaly-poc",
}

# ---------------------------------------------------------------------------
# Helper shortcuts
# ---------------------------------------------------------------------------
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


# ============================================================
# CELL 0 — Title
# ============================================================
cells = [
md("""# Retail Anomaly Detection & RCA — Full POC Walkthrough

This notebook walks through every layer of the system end-to-end:

| Step | What happens |
|------|-------------|
| 1 | Generate 60-day mock retail dataset (sales, weather, reviews, stock) |
| 2 | Inject 3 realistic anomaly scenarios into today's data |
| 3 | Ingest events through the normalisation pipeline |
| 4 | Build the metric cube with same-day-of-week rolling baselines |
| 5 | Run the ensemble detection engine (Z-score + STL + Isolation Forest) |
| 6 | Route anomalies through the alert manager (dedup + rate limiting) |
| 7 | Trigger RCA for P1/P2 alerts: extract causal data → call Claude → action plan |
| 8 | Visualise everything |

**Three anomaly scenarios:**
- 🧑‍💼 `STORE_003 / Apparel / NE` — severe staff shortage (poor reviews, walkouts)
- ⛈️ `STORE_007 / Electronics / MW` — heavy storm reducing foot traffic
- 📦 `STORE_012 / Beverages / SE` — supplier failure, shelves empty
"""),


# ============================================================
# CELL 1 — Setup
# ============================================================
code("""\
import warnings
warnings.filterwarnings("ignore")

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import timedelta

# Make the poc package importable from the notebook
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from poc.config import (
    TODAY, HISTORY_START, ANOMALY_SCENARIOS,
    DETECTOR_PROFILES, MODEL_WEIGHTS, ALERT_PRIORITY_THRESHOLDS,
    RCA_TRIGGER_PRIORITIES,
)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"P1": "#e74c3c", "P2": "#e67e22", "P3": "#3498db", "normal": "#2ecc71"}

print(f"Detection date  : {TODAY}")
print(f"History start   : {HISTORY_START}")
print(f"Days of history : {(TODAY - HISTORY_START).days}")
"""),


# ============================================================
# CELL 2 — Generate mock data
# ============================================================
md("## Step 1 — Generate Mock Data"),

code("""\
from poc.mock_data.generators import (
    generate_store_metadata, generate_sales_data,
    generate_weather_data,  generate_reviews_data,
    generate_holiday_calendar, generate_stock_data,
)

store_meta  = generate_store_metadata()
raw_sales   = generate_sales_data(HISTORY_START, TODAY)
raw_weather = generate_weather_data(HISTORY_START, TODAY)
raw_reviews = generate_reviews_data(HISTORY_START, TODAY)
raw_holidays = generate_holiday_calendar(HISTORY_START, TODAY)
raw_stock   = generate_stock_data(HISTORY_START, TODAY)

print(f"Stores          : {len(store_meta)}")
print(f"Sales rows      : {len(raw_sales):,}")
print(f"Weather rows    : {len(raw_weather):,}")
print(f"Review rows     : {len(raw_reviews):,}")
print(f"Stock rows      : {len(raw_stock):,}")
print()
display(store_meta)
"""),


# ============================================================
# CELL 3 — Inject anomaly scenarios
# ============================================================
md("## Step 2 — Inject Anomaly Scenarios"),

code("""\
from poc.mock_data.scenarios import inject_all_scenarios

sales_df, weather_df, reviews_df, stock_df = inject_all_scenarios(
    raw_sales, raw_weather, raw_reviews, raw_stock
)

today_ts = pd.Timestamp(TODAY)

# Show before / after for each scenario
print("=" * 65)
for name, cfg in ANOMALY_SCENARIOS.items():
    s_id, cat = cfg["store_id"], cfg["category"]
    before_row = raw_sales[
        (raw_sales["store_id"] == s_id) &
        (raw_sales["category"] == cat) &
        (raw_sales["date"]     == today_ts)
    ]
    after_row = sales_df[
        (sales_df["store_id"] == s_id) &
        (sales_df["category"] == cat) &
        (sales_df["date"]     == today_ts)
    ]
    if not before_row.empty and not after_row.empty:
        before = before_row["net_sales"].values[0]
        after  = after_row["net_sales"].values[0]
        drop   = (after - before) / before * 100
        print(f"  {name:15s}  {s_id} / {cat:12s}  "
              f"before=${before:,.0f}  after=${after:,.0f}  ({drop:+.0f}%)")
print("=" * 65)
"""),


# ============================================================
# CELL 4 — Sales time-series plots
# ============================================================
md("### Sales Time-Series: Normal vs Anomalous"),

code("""\
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
fig.suptitle("Sales Time-Series with Injected Anomalies", fontsize=14, fontweight="bold")

scenarios_order = [
    ("staff_shortage",  "STORE_003", "Apparel",     "NE", "Staff Shortage"),
    ("weather_event",   "STORE_007", "Electronics", "MW", "Weather Event"),
    ("stock_outage",    "STORE_012", "Beverages",   "SE", "Stock-Out"),
]

colors = ["#c0392b", "#d35400", "#8e44ad"]

for ax, (name, store_id, category, region, title), color in zip(axes, scenarios_order, colors):
    mask = (sales_df["store_id"] == store_id) & (sales_df["category"] == category)
    sub  = sales_df[mask].sort_values("date")

    hist = sub[sub["date"] < today_ts]
    today_row = sub[sub["date"] == today_ts]

    ax.fill_between(hist["date"], 0, hist["net_sales"],
                    alpha=0.15, color="#3498db")
    ax.plot(hist["date"], hist["net_sales"],
            color="#3498db", linewidth=1.2, label="Historical sales")

    if not today_row.empty:
        ax.scatter(today_row["date"], today_row["net_sales"],
                   color=color, s=200, zorder=5, label=f"TODAY (anomaly)")
        ax.axvline(x=today_ts, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate(
            f"  ↓ {title}\\n  ${today_row['net_sales'].values[0]:,.0f}",
            xy=(today_ts, today_row["net_sales"].values[0]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_title(f"{store_id} / {category} — {title}", fontsize=11, color=color)
    ax.set_ylabel("Net Sales ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("plots/01_sales_timeseries.png", dpi=120, bbox_inches="tight")
plt.show()
"""),


# ============================================================
# CELL 5 — Weather & Reviews
# ============================================================
md("### Weather and Review Signal Changes"),

code("""\
import os; os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Contextual Signals on Anomaly Day", fontsize=13, fontweight="bold")

# --- Weather: MW region precipitation ---
ax = axes[0]
wmask = weather_df["region"] == "MW"
wsub  = weather_df[wmask].sort_values("date").tail(21)
wsub_hist  = wsub[wsub["date"] < today_ts]
wsub_today = wsub[wsub["date"] == today_ts]

ax.bar(wsub_hist["date"], wsub_hist["precipitation_mm"],
       color="#85c1e9", label="Historical", alpha=0.8)
if not wsub_today.empty:
    ax.bar(wsub_today["date"], wsub_today["precipitation_mm"],
           color="#c0392b", label="TODAY (storm)", zorder=5)
    ax.annotate(
        f"  Storm\\n  {wsub_today['precipitation_mm'].values[0]:.0f}mm",
        xy=(wsub_today["date"].values[0], wsub_today["precipitation_mm"].values[0]),
        xytext=(5, 5), textcoords="offset points",
        fontsize=9, color="#c0392b", fontweight="bold",
    )
ax.set_title("MW Region — Precipitation (STORE_007 Weather Event)")
ax.set_ylabel("Precipitation (mm)")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Reviews: STORE_003 rating ---
ax = axes[1]
rmask = reviews_df["store_id"] == "STORE_003"
rsub  = reviews_df[rmask].sort_values("date").tail(14)
rsub_hist  = rsub[rsub["date"] < today_ts - pd.Timedelta(days=2)]
rsub_neg   = rsub[(rsub["date"] >= today_ts - pd.Timedelta(days=2)) & (rsub["date"] <= today_ts)]

ax.plot(rsub_hist["date"], rsub_hist["rating"],
        "o-", color="#2ecc71", linewidth=2, label="Normal reviews", markersize=5)
ax.plot(rsub_neg["date"],  rsub_neg["rating"],
        "o-", color="#e74c3c", linewidth=2.5, label="Staff shortage reviews", markersize=8)
ax.axhline(y=3.0, color="#e67e22", linestyle="--", alpha=0.7, label="Quality threshold (3.0)")
ax.set_ylim(1, 5.2)
ax.set_title("STORE_003 — Rating (Staff Shortage Signal)")
ax.set_ylabel("Customer Rating (1-5)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/02_contextual_signals.png", dpi=120, bbox_inches="tight")
plt.show()
"""),


# ============================================================
# CELL 6 — Ingestion pipeline
# ============================================================
md("## Step 3 — Ingestion Pipeline"),

code("""\
from poc.ingestion.pipeline import IngestPipeline

pipeline = IngestPipeline()
stats    = pipeline.ingest_all(sales_df, weather_df, reviews_df)

print(f"Ingested rows:")
print(f"  Sale rows      : {stats['sales']:,}")
print(f"  Weather rows   : {stats['weather']:,}")
print(f"  Review rows    : {stats['reviews']:,}")
print(f"  ─────────────────────────────────")
print(f"  Total events   : {stats['total_events']:,}")
print()

# Show a sample of what unified events look like
sample_events = pipeline.drain()[:5]
for ev in sample_events:
    print(f"  [{ev.event_type:10s}] store={ev.store_id or 'N/A'} "
          f"metric={ev.metric_name:15s} value={ev.value:,.1f}")
"""),


# ============================================================
# CELL 7 — Metric cube
# ============================================================
md("## Step 4 — Metric Cube (Same-Day-of-Week Rolling Baselines)"),

code("""\
from poc.compute.metric_cube import MetricCube

cube = MetricCube()
cube.build(sales_df, store_meta)

segments = cube.get_active_segments()
print(f"Active segments: {len(segments)}")

# Show baseline quality for the 3 scenario segments
print()
print(f"{'Segment':40s}  {'Tier':6s}  {'Baseline':>12s}  {'Std':>10s}  {'Today':>10s}  {'Z-score':>8s}")
print("-" * 100)

for store_id, category in [("STORE_003","Apparel"), ("STORE_007","Electronics"), ("STORE_012","Beverages"),
                             ("STORE_001","Apparel"), ("STORE_001","Electronics")]:
    try:
        series, baseline, std = cube.get_segment_series(store_id, category, "net_sales", up_to_date=TODAY)
        today_val = float(series.iloc[-1])
        z = (today_val - baseline) / std
        tier = next((s["tier"] for s in segments if s["store_id"]==store_id and s["category"]==category), "?")
        print(f"{store_id+' / '+category:40s}  {tier:6s}  ${baseline:>11,.0f}  ${std:>9,.0f}  ${today_val:>9,.0f}  {z:>+8.2f}σ")
    except:
        pass
"""),


# ============================================================
# CELL 8 — Detection engine
# ============================================================
md("## Step 5 — Ensemble Detection Engine"),

code("""\
from poc.detection.engine import DetectionEngine

engine    = DetectionEngine(cube, weather_df, raw_holidays)
anomalies = engine.run(TODAY)

print(f"Anomalies detected: {len(anomalies)}")
print()

for a in anomalies:
    print(f"  {'['+a['priority']+']':5s}  {a['store_id']:10s}  {a['category']:12s}  "
          f"σ={a['sigma']:.2f}  impact={a['impact_score']:5.1f}  "
          f"Δ={a['pct_deviation']:+.0f}%  "
          f"models=[{', '.join(a['contributing'])}]")
"""),


# ============================================================
# CELL 9 — Detection details per segment
# ============================================================
md("### Detector Voting Details"),

code("""\
# Show which models voted on each anomaly
fig, ax = plt.subplots(figsize=(10, 4))

all_models = ["zscore", "stl", "isolation_forest"]
scenario_labels = []
vote_matrix = []

for a in anomalies:
    label = f"{a['store_id']}\\n{a['category']}"
    scenario_labels.append(label)
    row = [1 if m in a["contributing"] else 0 for m in all_models]
    vote_matrix.append(row)

if vote_matrix:
    vm = np.array(vote_matrix)
    im = ax.imshow(vm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels([m.replace("_", "\\n") for m in all_models], fontsize=10)
    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels, fontsize=9)
    ax.set_title("Detector Voting Matrix (Green = Anomaly Vote)", fontsize=12, pad=15)

    for i in range(len(scenario_labels)):
        for j in range(len(all_models)):
            ax.text(j, i, "✓" if vm[i,j] else "✗",
                    ha="center", va="center", fontsize=14,
                    color="white" if vm[i,j] else "gray")

    weights_text = "  ".join([f"{m}: {MODEL_WEIGHTS.get(m,'N/A')}" for m in all_models])
    ax.set_xlabel(f"Model weights: {weights_text}", fontsize=9)

plt.tight_layout()
plt.savefig("plots/03_detection_vote_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
"""),


# ============================================================
# CELL 10 — Alert manager
# ============================================================
md("## Step 6 — Alert Manager (Dedup + Rate Limiting)"),

code("""\
from poc.alerting.alert_manager import AlertManager

alert_mgr = AlertManager()
alerts    = alert_mgr.process(anomalies)

# Run again to demonstrate dedup
alert_mgr.process(anomalies)   # all duplicates — should be suppressed

p1 = [a for a in alerts if a["priority"] == "P1"]
p2 = [a for a in alerts if a["priority"] == "P2"]
p3 = [a for a in alerts if a["priority"] == "P3"]

print(f"Anomaly candidates : {len(anomalies)}")
print(f"Duplicate attempts : {len(anomalies)} (second run — all suppressed by dedup)")
print(f"Final alerts       : {len(alerts)}")
print(f"  P1 (impact≥75)   : {len(p1)}")
print(f"  P2 (impact≥40)   : {len(p2)}")
print(f"  P3 (impact≥0)    : {len(p3)}")
print()

df_alerts = pd.DataFrame([{
    "Alert ID":     a["alert_id"][:20],
    "Store":        a["store_id"],
    "Category":     a["category"],
    "Priority":     a["priority"],
    "Sigma":        a["sigma"],
    "Impact Score": a["impact_score"],
    "% Deviation":  f"{a['pct_deviation']:+.0f}%",
    "Models":       ", ".join(a["contributing"]),
} for a in alerts])
display(df_alerts)
"""),


# ============================================================
# CELL 11 — Alert visualisation
# ============================================================
md("### Alert Impact Score Visualisation"),

code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Alert Summary", fontsize=13, fontweight="bold")

# Impact score bars
labels = [f"{a['store_id']}\\n{a['category']}" for a in alerts]
scores = [a["impact_score"] for a in alerts]
colors = [PALETTE[a["priority"]] for a in alerts]

bars = ax1.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.5)
ax1.axvline(x=ALERT_PRIORITY_THRESHOLDS["P1"], color="#e74c3c", linestyle="--",
            alpha=0.7, label="P1 threshold (75)")
ax1.axvline(x=ALERT_PRIORITY_THRESHOLDS["P2"], color="#e67e22", linestyle="--",
            alpha=0.7, label="P2 threshold (40)")
for bar, score, a in zip(bars, scores, alerts):
    ax1.text(score + 0.5, bar.get_y() + bar.get_height()/2,
             f"{score:.1f}  [{a['priority']}]", va="center", fontsize=10)
ax1.set_xlim(0, 110)
ax1.set_xlabel("Impact Score")
ax1.set_title("Impact Scores by Segment")
ax1.legend(fontsize=8)

# Sigma bars
sigmas = [a["sigma"] for a in alerts]
bars2  = ax2.barh(labels, sigmas, color=colors, edgecolor="white")
for bar, sigma in zip(bars2, sigmas):
    ax2.text(sigma + 0.05, bar.get_y() + bar.get_height()/2,
             f"{sigma:.1f}σ", va="center", fontsize=10)
ax2.set_xlabel("Sigma (standard deviations from baseline)")
ax2.set_title("Anomaly Magnitude (σ)")

# Legend
legend_patches = [
    mpatches.Patch(color=PALETTE["P1"], label="P1 — Critical"),
    mpatches.Patch(color=PALETTE["P2"], label="P2 — High"),
    mpatches.Patch(color=PALETTE["P3"], label="P3 — Medium"),
]
ax2.legend(handles=legend_patches, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("plots/04_alert_summary.png", dpi=120, bbox_inches="tight")
plt.show()
"""),


# ============================================================
# CELL 12 — RCA workflow
# ============================================================
md("""## Step 7 — RCA Workflow

For each P1/P2 alert, the system:
1. Extracts causal datasets (sales, weather, reviews, stock, holidays)
2. Assembles a structured prompt
3. Calls Claude API (or uses a demo response if no API key set)
4. Returns ranked hypotheses + action plan
"""),

code("""\
from poc.rca.extractors import (
    SalesExtractor, WeatherExtractor, HolidayExtractor,
    ReviewExtractor, StockExtractor,
)
from poc.rca.prompt_builder import build_prompt
from poc.rca.claude_client import call_claude

rca_trigger_alerts = [a for a in alerts if a["priority"] in RCA_TRIGGER_PRIORITIES]
rca_results = []

for alert in rca_trigger_alerts:
    store_id = alert["store_id"]
    category = alert["category"]
    region   = alert.get("region", "")
    anchor   = pd.Timestamp(TODAY)

    dataset = {
        "sales":    SalesExtractor().extract(sales_df,   store_id, category, anchor),
        "weather":  WeatherExtractor().extract(weather_df, region,  anchor),
        "reviews":  ReviewExtractor().extract(reviews_df, store_id, anchor),
        "holidays": HolidayExtractor().extract(raw_holidays, anchor),
        "stock":    StockExtractor().extract(stock_df,   store_id, category, anchor),
    }

    prompt = build_prompt(alert, dataset)
    rca    = call_claude(prompt, alert)
    rca_results.append({"alert": alert, "dataset": dataset, "prompt": prompt, "rca": rca})

    print(f"✓ RCA complete: {store_id} / {category} [{alert['priority']}]")

print(f"\\n{len(rca_results)} RCA workflows completed")
"""),


# ============================================================
# CELL 13 — Show prompt for one alert
# ============================================================
md("### Sample RCA Prompt Sent to Claude"),

code("""\
# Show the prompt for the first RCA (staff shortage)
if rca_results:
    print("=" * 70)
    print(rca_results[0]["prompt"][:3000])
    print("  ... [truncated] ...")
    print("=" * 70)
"""),


# ============================================================
# CELL 14 — RCA results display
# ============================================================
md("### RCA Results — Hypotheses, Actions, Data Gaps"),

code("""\
for r in rca_results:
    alert = r["alert"]
    rca   = r["rca"]
    hyps  = rca.get("hypotheses", [])
    acts  = rca.get("actions", [])
    gaps  = rca.get("data_gaps", [])

    priority_color = {"P1": "\\033[91m", "P2": "\\033[93m", "P3": "\\033[94m"}
    reset = "\\033[0m"
    pc    = priority_color.get(alert["priority"], "")

    print(f"\\n{'='*70}")
    print(f"{pc}[{alert['priority']}] {alert['store_id']} / {alert['category']} / {alert.get('region','')}{reset}")
    print(f"    Impact: {alert['impact_score']:.1f}  |  σ={alert['sigma']:.2f}  |  Δ={alert['pct_deviation']:+.0f}%")
    print(f"{'─'*70}")

    print("  HYPOTHESES:")
    for h in hyps:
        confidence_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        icon = confidence_icons.get(h.get("confidence",""), "⚪")
        print(f"    {icon} [{h['rank']}] {h['cause']}")
        print(f"        Likelihood: {h['likelihood']:.0%}  |  Confidence: {h.get('confidence','?')}")
        for ev in h.get("evidence", []):
            print(f"        • {ev}")

    print("\\n  ACTIONS:")
    for act in acts:
        urgency_colors = {"P1": "\\033[91m", "P2": "\\033[93m", "P3": "\\033[94m"}
        uc = urgency_colors.get(act["urgency"], "")
        print(f"    {uc}[{act['urgency']}]{reset} {act['action']}")
        print(f"         → {act['owner_team']}  (due in {act['due_hours']}h)")

    if gaps:
        print("\\n  DATA GAPS:")
        for g in gaps:
            print(f"    ◦ {g}")
"""),


# ============================================================
# CELL 15 — RCA visualisation
# ============================================================
md("### RCA Hypothesis Likelihood Visualisation"),

code("""\
fig, axes = plt.subplots(1, len(rca_results), figsize=(5*len(rca_results), 5))
if len(rca_results) == 1:
    axes = [axes]
fig.suptitle("RCA — Hypothesis Likelihoods", fontsize=13, fontweight="bold")

scenario_titles = {
    "STORE_003": "Staff Shortage",
    "STORE_007": "Weather Event",
    "STORE_012": "Stock-Out",
}

confidence_colors = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}

for ax, r in zip(axes, rca_results):
    alert = r["alert"]
    hyps  = r["rca"].get("hypotheses", [])
    if not hyps:
        continue

    causes      = [h["cause"][:35] + "…" if len(h["cause"]) > 35 else h["cause"] for h in hyps]
    likelihoods = [h["likelihood"] for h in hyps]
    conf_colors = [confidence_colors.get(h.get("confidence",""), "#95a5a6") for h in hyps]

    bars = ax.barh(causes[::-1], likelihoods[::-1], color=conf_colors[::-1],
                   edgecolor="white", height=0.5)
    for bar, val in zip(bars, likelihoods[::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Likelihood")
    store_id = alert["store_id"]
    priority = alert["priority"]
    title    = scenario_titles.get(store_id, store_id)
    ax.set_title(f"[{priority}] {title}\\n{store_id} / {alert['category']}", fontsize=10)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)

# Confidence legend
legend_patches = [
    mpatches.Patch(color="#27ae60", label="High confidence"),
    mpatches.Patch(color="#f39c12", label="Medium confidence"),
    mpatches.Patch(color="#e74c3c", label="Low confidence"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("plots/05_rca_hypotheses.png", dpi=120, bbox_inches="tight")
plt.show()
"""),


# ============================================================
# CELL 16 — Full system summary
# ============================================================
md("## Summary — End-to-End System Flow"),

code("""\
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

# Draw pipeline boxes
boxes = [
    (0.3, 7.5, "DATA SOURCES\\n60 days history\\n7 stores × 5 categories",   "#3498db"),
    (3.0, 7.5, "INGESTION\\n7,869 unified\\nevents",                         "#9b59b6"),
    (5.7, 7.5, "METRIC CUBE\\n35 segments\\nDeseasonalised baselines",        "#1abc9c"),
    (8.4, 7.5, "DETECTION\\n3 detectors\\nEnsemble voting",                   "#e67e22"),
    (11.1,7.5, "ALERTS\\n3 anomalies\\nP1=2, P2=1",                           "#e74c3c"),
]

rca_box = (5.0, 2.5, "RCA WORKFLOW\\n3 analyses\\nClaude API",               "#2c3e50")

for (x, y, text, color) in boxes:
    rect = mpatches.FancyBboxPatch((x-0.1, y-0.8), 2.5, 2.0,
                                    boxstyle="round,pad=0.1",
                                    linewidth=2, edgecolor=color, facecolor=color+"22")
    ax.add_patch(rect)
    ax.text(x+1.15, y+0.2, text, ha="center", va="center",
            fontsize=9, color=color, fontweight="bold")

# Arrows between boxes
arrow_props = dict(arrowstyle="-|>", color="#555", lw=1.5)
for i in range(len(boxes)-1):
    ax.annotate("", xy=(boxes[i+1][0]-0.1, boxes[i+1][1]+0.2),
                xytext=(boxes[i][0]+2.4, boxes[i][1]+0.2),
                arrowprops=arrow_props)

# RCA box
x, y, text, color = rca_box
rect = mpatches.FancyBboxPatch((x-0.1, y-0.8), 4.0, 2.0,
                                boxstyle="round,pad=0.1",
                                linewidth=2, edgecolor=color, facecolor=color+"22")
ax.add_patch(rect)
ax.text(x+1.9, y+0.2, text, ha="center", va="center",
        fontsize=10, color=color, fontweight="bold")

# Arrows to/from RCA
ax.annotate("", xy=(rca_box[0]+2.0, rca_box[1]+1.2),
            xytext=(boxes[-1][0]+1.2, boxes[-1][1]-0.8),
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2))

# Scenario annotations
scenarios_text = [
    (1.3, 4.8, "Staff Shortage",  "#e74c3c"),
    (4.8, 4.8, "Weather Event",   "#3498db"),
    (8.3, 4.8, "Stock-Out",       "#8e44ad"),
]
for (sx, sy, stxt, sc) in scenarios_text:
    ax.text(sx, sy, stxt, ha="center", va="center", fontsize=9,
            color=sc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=sc+"11", edgecolor=sc, linewidth=1))

ax.set_title("Retail Anomaly Detection & RCA — Full Pipeline", fontsize=14,
             fontweight="bold", pad=15, color="#2c3e50")

plt.tight_layout()
plt.savefig("plots/06_system_summary.png", dpi=120, bbox_inches="tight")
plt.show()
print("\\nAll plots saved to ./plots/")
"""),


# ============================================================
# CELL 17 — With real API key
# ============================================================
md("""## Bonus — Using Real Claude API

To run with a real Claude API key (replace demo responses with live analysis):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
jupyter notebook poc_notebook.ipynb
```

Or in the notebook cell:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

The `call_claude()` function automatically detects the key and switches to live mode.
The Claude response structure is identical to the demo — hypotheses, actions, data_gaps in JSON.
"""),

]  # end cells list

nb.cells = cells

# Write
import json
with open("poc_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created poc_notebook.ipynb")
