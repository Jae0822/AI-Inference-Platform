"""
scripts/check_regression.py — Performance regression gate.

Reads the committed baseline.json and a current benchmark result,
compares key metrics, and exits with code 1 if any metric degrades
beyond its threshold.

Usage:
    python scripts/check_regression.py --current results/locust_raw.csv

The script:
  1. Reads baseline.json (committed to repo — the performance contract)
  2. Reads current benchmark results from locust_raw.csv
  3. Computes p50/p99 for TTFT and end-to-end latency, p50 for TPS
  4. Compares each metric against baseline + threshold
  5. Prints a report
  6. Exits 0 if all metrics pass, 1 if any metric regresses

Exit codes:
  0: All metrics within threshold — CI passes
  1: One or more metrics regressed — CI fails
  2: Missing input files or invalid data — CI fails with error

Why exit codes instead of exceptions?
  CI systems (GitHub Actions) interpret the process exit code.
  Exit 0 = success (green CI). Exit non-zero = failure (red CI).
  Raising an exception would also cause exit 1, but the error message
  would be a Python traceback rather than a human-readable regression report.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent
BASELINE_PATH = REPO_ROOT / "baseline.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline() -> dict:
    """Load the committed performance baseline."""
    if not BASELINE_PATH.exists():
        print(f"ERROR: baseline.json not found at {BASELINE_PATH}")
        sys.exit(2)

    with open(BASELINE_PATH) as f:
        return json.load(f)


def load_current_results(csv_path: str) -> dict:
    """
    Load current benchmark results from locust_raw.csv.

    Returns dict with p50/p99 for ttft and total latency, p50 for tps.
    Returns None if file is missing or has insufficient data.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: Current results file not found: {csv_path}")
        sys.exit(2)

    ttfts   = []
    totals  = []
    tps_vals = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ttft = float(row["ttft_ms"])
                total = float(row["total_ms"])
                tps  = float(row["tps"])
                if ttft > 0:
                    ttfts.append(ttft)
                totals.append(total)
                tps_vals.append(tps)
            except (KeyError, ValueError):
                continue

    if len(ttfts) < 5:
        print(f"ERROR: Insufficient data in {csv_path} ({len(ttfts)} valid rows, need >= 5)")
        sys.exit(2)

    def pct(lst, p):
        lst = sorted(lst)
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    return {
        "ttft_p50_ms":  round(pct(ttfts,   50), 1),
        "ttft_p99_ms":  round(pct(ttfts,   99), 1),
        "total_p50_ms": round(pct(totals,  50), 1),
        "total_p99_ms": round(pct(totals,  99), 1),
        "tps_p50":      round(pct(tps_vals, 50), 1),
        "n_requests":   len(totals),
    }


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def check_regressions(baseline: dict, current: dict) -> list[dict]:
    """
    Compare current metrics against baseline + thresholds.

    Returns a list of regression dicts, one per failing metric.
    Each dict contains: metric, baseline_val, current_val, threshold_pct,
    actual_change_pct, status ("PASS" or "FAIL").
    """
    thresholds = baseline.get("regression_thresholds", {})
    results = []

    metrics_to_check = [
        ("ttft_p99_ms",  "higher is worse"),
        ("total_p99_ms", "higher is worse"),
        ("tps_p50",      "lower is worse"),
    ]

    for metric, direction in metrics_to_check:
        baseline_val  = baseline.get(metric)
        current_val   = current.get(metric)
        threshold_pct = thresholds.get(metric, 15)

        if baseline_val is None or current_val is None:
            continue

        if baseline_val == 0:
            continue

        # Compute % change (positive = got worse for latency metrics)
        if "tps" in metric:
            # For throughput: negative change = regression
            change_pct = ((current_val - baseline_val) / baseline_val) * 100
            regressed = change_pct < -threshold_pct
        else:
            # For latency: positive change = regression (higher = worse)
            change_pct = ((current_val - baseline_val) / baseline_val) * 100
            regressed = change_pct > threshold_pct

        results.append({
            "metric":           metric,
            "baseline_val":     baseline_val,
            "current_val":      current_val,
            "threshold_pct":    threshold_pct,
            "change_pct":       round(change_pct, 1),
            "direction":        direction,
            "status":           "FAIL" if regressed else "PASS",
        })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(baseline: dict, current: dict, results: list[dict]) -> bool:
    """
    Print a human-readable regression report.
    Returns True if all checks passed, False if any failed.
    """
    print()
    print("=" * 65)
    print("  Performance Regression Report")
    print("=" * 65)
    print(f"  Baseline model  : {baseline.get('_model', 'unknown')}")
    print(f"  Requests tested : {current.get('n_requests', '?')}")
    print()
    print(f"  {'Metric':<22} {'Baseline':>10} {'Current':>10} "
          f"{'Change':>8} {'Threshold':>10} {'Status':>6}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")

    all_passed = True
    for r in results:
        status_icon = "✓" if r["status"] == "PASS" else "✗"
        change_str = f"{r['change_pct']:+.1f}%"
        threshold_str = f"±{r['threshold_pct']}%"
        print(
            f"  {r['metric']:<22} "
            f"{r['baseline_val']:>10} "
            f"{r['current_val']:>10} "
            f"{change_str:>8} "
            f"{threshold_str:>10} "
            f"{status_icon:>6} {r['status']}"
        )
        if r["status"] == "FAIL":
            all_passed = False

    print()
    if all_passed:
        print("  RESULT: All metrics within threshold. CI PASSES.")
    else:
        failed = [r["metric"] for r in results if r["status"] == "FAIL"]
        print(f"  RESULT: REGRESSION DETECTED in: {', '.join(failed)}")
        print("  CI FAILS. Review changes that may have affected performance.")
    print("=" * 65)
    print()

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check for performance regressions against baseline.json"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current benchmark CSV (locust_raw.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override regression threshold % for all metrics",
    )
    args = parser.parse_args()

    baseline = load_baseline()
    current  = load_current_results(args.current)

    # Apply threshold override if provided
    if args.threshold is not None:
        for key in baseline.get("regression_thresholds", {}):
            baseline["regression_thresholds"][key] = args.threshold

    results    = check_regressions(baseline, current)
    all_passed = print_report(baseline, current, results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
