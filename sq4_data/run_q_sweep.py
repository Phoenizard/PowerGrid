"""
Experiment 4A: q-sweep — κ_c time series for varying rewiring probability.

For q ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}:
  1. Preload ensemble at that q
  2. Run 56-point κ_c time series (no edge modification)
  3. Save per-q CSV

After all q values: produce summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kappa_pipeline import (
    preload_ensemble, run_timeseries,
    FAST_CONFIG, PROD_CONFIG,
)

Q_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 4A: q-sweep")
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260209)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--q_only", type=float, default=None,
                        help="Run only a single q value (for testing)")
    return parser.parse_args()


def summarize_q_sweep(results_dir: str, q_values: list[float], output_path: str):
    """Produce summary CSV from per-q result files."""
    rows = []
    for q in q_values:
        csv_path = os.path.join(results_dir, f"kappa_ts_q{q:.2f}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARN: missing {csv_path}, skipping q={q}")
            continue

        df = pd.read_csv(csv_path)
        noon = df[df["hour"] == 12]
        dawn = df[df["hour"] == 6]

        kc_noon_mean = noon["kappa_c_mean"].mean()
        kc_noon_std = noon["kappa_c_mean"].std()
        kc_dawn_mean = dawn["kappa_c_mean"].mean()
        kc_dawn_std = dawn["kappa_c_mean"].std()
        ratio = kc_noon_mean / kc_dawn_mean if kc_dawn_mean > 0 else float("inf")

        rows.append({
            "q": q,
            "kc_noon_mean": kc_noon_mean,
            "kc_noon_std": kc_noon_std,
            "kc_dawn_mean": kc_dawn_mean,
            "kc_dawn_std": kc_dawn_std,
            "peak_valley_ratio": ratio,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    print(f"\nSummary saved: {output_path}")
    print(summary.to_string(index=False))
    return summary


def main():
    args = parse_args()
    sim_config = PROD_CONFIG if args.production else FAST_CONFIG
    config_name = "PRODUCTION" if args.production else "FAST"

    results_dir = os.path.join(SCRIPT_DIR, "results", "exp4A")
    os.makedirs(results_dir, exist_ok=True)

    q_values = [args.q_only] if args.q_only is not None else Q_VALUES

    print(f"=== Experiment 4A: q-sweep ===")
    print(f"  Config: {config_name}")
    print(f"  n_ensemble: {args.n_ensemble}")
    print(f"  q values: {q_values}")
    print(f"  Seed: {args.seed}")
    sys.stdout.flush()

    t_total_start = time.time()

    for q in q_values:
        print(f"\n--- q = {q:.2f} ---")
        output_path = os.path.join(results_dir, f"kappa_ts_q{q:.2f}.csv")

        # Check if already complete
        if not args.clean and os.path.exists(output_path):
            df = pd.read_csv(output_path)
            if len(df) >= 56:
                print(f"  Already complete ({len(df)} rows), skipping.")
                continue

        print(f"  Pre-loading ensemble (q={q:.2f})...")
        sys.stdout.flush()
        ensemble = preload_ensemble(
            n_ensemble=args.n_ensemble,
            season=args.season,
            seed=args.seed,
            q=q,
        )

        print(f"  Running timeseries...")
        sys.stdout.flush()
        run_timeseries(
            ensemble=ensemble,
            sim_config=sim_config,
            output_path=output_path,
            network_modifier=None,
            clean=args.clean,
        )

    # Summary
    summary_path = os.path.join(results_dir, "sq4a_q_sweep.csv")
    summarize_q_sweep(results_dir, q_values if args.q_only is None else Q_VALUES, summary_path)

    t_total = time.time() - t_total_start
    print(f"\n=== Exp 4A complete in {t_total/60:.1f} min ===")


if __name__ == "__main__":
    main()
