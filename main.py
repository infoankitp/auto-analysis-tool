"""
Entry point for the Retail Anomaly Detection & RCA POC.

Usage:
    python main.py                  # full run with demo RCA
    ANTHROPIC_API_KEY=sk-ant-...  python main.py   # live Claude RCA
"""

from poc.orchestrator import run_poc

if __name__ == "__main__":
    run_poc(verbose=True)
