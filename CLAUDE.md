# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Environment

All Python work runs inside the project virtualenv:

```bash
# First-time setup
bash setup.sh

# Activate (required before any python/jupyter commands)
source .venv/bin/activate
```

## Running the POC

```bash
# Full pipeline run (demo RCA — no API key required)
python main.py

# Full pipeline run with live Claude RCA
ANTHROPIC_API_KEY=sk-ant-... python main.py

# Interactive notebook
jupyter notebook poc_notebook.ipynb   # kernel: "Retail Anomaly POC"

# Re-generate the notebook from source
python create_notebook.py
```

## Architecture

The system is a self-contained POC in the `poc/` package. `poc/orchestrator.py:run_poc()` is the single entry point that wires every layer together. `main.py` calls it.

### Data flow

```
mock_data/generators.py   →  raw DataFrames (60-day history + today)
mock_data/scenarios.py    →  inject 3 anomalies into today's rows
ingestion/pipeline.py     →  normalise to UnifiedEvent (in-memory EventBus)
compute/metric_cube.py    →  build rolling baselines → MetricCube
detection/engine.py       →  DetectionEngine.run(date) → anomaly candidates
alerting/alert_manager.py →  dedup + rate limit → final alerts
rca/                      →  extract signals → build prompt → call Claude
```

### Key architectural decisions to understand

**Same-day-of-week baseline** (`compute/metric_cube.py`): The MetricCube uses the mean of the last 4 same-weekday values as `baseline_28d`, NOT a simple 28-day rolling mean. This is critical — a naive 28-day rolling std captures weekly seasonal variation (Mon vs Sat sales differ ~2×), which would inflate the std and make z-scores too small to cross detection thresholds. The DOW-matched baseline gives a noise-only std, producing accurate z-scores.

**Tiered detector profiles** (`poc/config.py:DETECTOR_PROFILES`): Each segment is assigned tier1/2/3 based on `cluster_rank`. Tier drives which models run (`stl`, `zscore`, `isolation_forest`), the sigma threshold, and whether the context guard is active. `detection/engine.py` reads the tier via `MetricCube.get_active_segments()`.

**Ensemble voting** (`detection/ensemble.py`): `EnsembleVoter` takes a list of `DetectorResult` objects and returns an anomaly decision only when `weighted_vote ≥ 0.5`. Weights are in `config.MODEL_WEIGHTS`. Impact score = `min(100, σ × revenue_weight × novelty_factor × grain_multiplier × 15.0)`.

**Context guard** (`detection/context_guard.py`): Runs after detectors, before the vote. Flips `is_anomaly=False` when a holiday with `impact_score ≥ 0.7` or a weather severity in `{"blizzard", "hurricane", "ice_storm"}` is present. Note: `"storm"` does NOT suppress — this is intentional so the weather scenario still produces an alert whose RCA then identifies weather as the cause.

**Claude client** (`rca/claude_client.py`): Falls back to scenario-specific hardcoded responses in `DEMO_RESPONSES` when `ANTHROPIC_API_KEY` is unset. The live path uses the Anthropic SDK with `cache_control: ephemeral` on the system prompt. The model is `claude-sonnet-4-6` (set in `config.CLAUDE_MODEL`).

### Adding a new anomaly scenario

1. Add an entry to `ANOMALY_SCENARIOS` in `poc/config.py`
2. Write an `inject_<scenario>()` function in `poc/mock_data/scenarios.py` and call it from `inject_all_scenarios()`
3. Add a matching entry in `DEMO_RESPONSES` in `poc/rca/claude_client.py` (keyed on `store_id`) so the demo fallback produces a meaningful response

### Adding a new detector

1. Subclass `BaseDetector` in `poc/detection/detectors.py`; implement `detect(series, baseline, std) → DetectorResult`
2. Register it in `DETECTOR_REGISTRY` in `poc/detection/engine.py`
3. Add its weight to `MODEL_WEIGHTS` in `poc/config.py`
4. Reference it by name in the relevant tier's `"models"` list in `DETECTOR_PROFILES`

## All tuneable constants

Everything lives in `poc/config.py` — thresholds, weights, scenario definitions, model name, window sizes. No other file should contain numeric constants.

## Design document

The original system design (production architecture with Kafka, ClickHouse, Ray, Prefect) is preserved in the `design-docs/` directory as PNG diagrams. The POC replaces external infrastructure with in-memory equivalents but mirrors the same interfaces.
