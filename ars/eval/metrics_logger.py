"""
Lightweight local run history -- appends the summary from run_evals.py to
eval_history.json so you can plot faithfulness trend over time as you
change the writer prompt, model, or retrieval strategy.

This is deliberately NOT MLflow yet. Swapping this for real MLflow
tracking is Phase 6 -- don't build that infra until this simple version
has actually produced a few data points worth tracking.

Usage:
    python -m eval.metrics_logger          # logs the latest eval_results.json
    python -m eval.metrics_logger --plot    # also prints a tiny ASCII trend
"""

import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "eval_results.json"
HISTORY_PATH = EVAL_DIR / "eval_history.json"


def log_latest_run() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found -- run `python -m eval.run_evals` first."
        )

    latest = json.loads(RESULTS_PATH.read_text())
    entry = {
        "run_at": latest["run_at"],
        "average_faithfulness": latest["average_faithfulness"],
        "num_questions": latest["num_questions"],
        "num_scored": latest["num_scored"],
        "passed": latest["passed"],
    }

    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

    print(f"Logged run: {entry['average_faithfulness']} average faithfulness "
          f"({'PASS' if entry['passed'] else 'FAIL'})")
    print(f"History now has {len(history)} run(s) -- see {HISTORY_PATH}")
    return entry


def print_trend():
    if not HISTORY_PATH.exists():
        print("No history yet.")
        return
    history = json.loads(HISTORY_PATH.read_text())
    print("\nFaithfulness trend:")
    for i, entry in enumerate(history, 1):
        bar = "#" * int(entry["average_faithfulness"] * 40)
        print(f"  run {i:>2} [{entry['run_at'][:10]}] {entry['average_faithfulness']:.3f} {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Print an ASCII trend of all logged runs")
    args = parser.parse_args()

    log_latest_run()
    if args.plot:
        print_trend()


if __name__ == "__main__":
    main()
