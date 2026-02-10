"""
Experiment 4B Step 1: Strategy comparison — κ_c time series for 4 edge addition
strategies + baseline, all with m=4, q=0.1.

Strategies: baseline (no edges), random, max_power, score, pcc_direct.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kappa_pipeline import (
    preload_ensemble, run_timeseries,
    FAST_CONFIG, PROD_CONFIG, EnsembleInstance,
)
from edge_strategies import (
    STRATEGIES, compute_node_power_stats, verify_edge_addition,
)

STRATEGY_NAMES = ["baseline", "random", "max_power", "score", "pcc_direct"]
DEFAULT_M = 4


def make_modifier(strategy_name: str, m: int):
    """Create a NetworkModifier for the given strategy and m."""
    if strategy_name == "baseline":
        return None

    select_fn = STRATEGIES[strategy_name]

    def modifier(A_base: csr_matrix, instance: EnsembleInstance) -> csr_matrix:
        P_max_n, P_sign_n = compute_node_power_stats(
            instance.cons_interps, instance.pv_interps
        )
        edge_rng = default_rng(instance.net_seed + 1_000_000)
        A_lil = A_base.tolil()
        A_original_nnz = A_base.nnz
        A_modified = select_fn(A_lil, m, P_max_n, P_sign_n, edge_rng)
        verify_edge_addition(A_base, A_modified, (A_modified.nnz - A_original_nnz) // 2)
        return A_modified

    return modifier


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 4B Step 1: Strategy comparison")
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260209)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--strategy_only", type=str, default=None,
                        help="Run only one strategy (for testing)")
    return parser.parse_args()


def summarize_strategies(results_dir: str, strategies: list[str], m: int, output_path: str):
    """Produce summary CSV from per-strategy results."""
    # Load baseline for improvement calculation
    baseline_noon = None
    rows = []

    for strat in strategies:
        csv_path = os.path.join(results_dir, f"kappa_ts_{strat}_m{m}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARN: missing {csv_path}, skipping {strat}")
            continue

        df = pd.read_csv(csv_path)
        noon = df[df["hour"] == 12]
        dawn = df[df["hour"] == 6]

        kc_noon_mean = noon["kappa_c_mean"].mean()
        kc_noon_std = noon["kappa_c_mean"].std()
        kc_dawn_mean = dawn["kappa_c_mean"].mean()
        kc_dawn_std = dawn["kappa_c_mean"].std()
        ratio = kc_noon_mean / kc_dawn_mean if kc_dawn_mean > 0 else float("inf")

        if strat == "baseline":
            baseline_noon = kc_noon_mean

        improvement = 0.0
        if baseline_noon is not None and baseline_noon > 0 and strat != "baseline":
            improvement = (kc_noon_mean - baseline_noon) / baseline_noon * 100

        rows.append({
            "strategy": strat,
            "m": m,
            "kc_noon_mean": kc_noon_mean,
            "kc_noon_std": kc_noon_std,
            "kc_dawn_mean": kc_dawn_mean,
            "kc_dawn_std": kc_dawn_std,
            "peak_valley_ratio": ratio,
            "improvement_pct": improvement,
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

    results_dir = os.path.join(SCRIPT_DIR, "results", "exp4B_s1")
    os.makedirs(results_dir, exist_ok=True)

    strategies = [args.strategy_only] if args.strategy_only else STRATEGY_NAMES

    print(f"=== Experiment 4B Step 1: Strategy Comparison ===")
    print(f"  Config: {config_name}")
    print(f"  n_ensemble: {args.n_ensemble}")
    print(f"  m (edges): {args.m}")
    print(f"  Strategies: {strategies}")
    print(f"  Seed: {args.seed}")
    sys.stdout.flush()

    # Preload ensemble once (all strategies use same q=0.1 ensemble)
    print("\nPre-loading ensemble (q=0.1)...")
    sys.stdout.flush()
    ensemble = preload_ensemble(
        n_ensemble=args.n_ensemble,
        season=args.season,
        seed=args.seed,
        q=0.1,
    )

    t_total_start = time.time()

    for strat in strategies:
        print(f"\n--- Strategy: {strat} (m={args.m}) ---")
        output_path = os.path.join(results_dir, f"kappa_ts_{strat}_m{args.m}.csv")

        # Check if already complete
        if not args.clean and os.path.exists(output_path):
            df = pd.read_csv(output_path)
            if len(df) >= 56:
                print(f"  Already complete ({len(df)} rows), skipping.")
                continue

        modifier = make_modifier(strat, args.m)
        run_timeseries(
            ensemble=ensemble,
            sim_config=sim_config,
            output_path=output_path,
            network_modifier=modifier,
            clean=args.clean,
        )

    # Summary
    summary_path = os.path.join(results_dir, "sq4b_step1_summary.csv")
    summarize_strategies(
        results_dir,
        strategies if not args.strategy_only else STRATEGY_NAMES,
        args.m,
        summary_path,
    )

    t_total = time.time() - t_total_start
    print(f"\n=== Exp 4B-S1 complete in {t_total/60:.1f} min ===")


if __name__ == "__main__":
    main()
