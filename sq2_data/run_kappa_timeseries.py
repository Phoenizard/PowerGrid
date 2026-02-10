"""
Experiment 3B: Compute kappa_c at 56 representative time points.

For each ensemble instance:
  1. Generate WS+PCC network (50 nodes: 49 houses + 1 PCC)
  2. Build random LCL + PV interpolators (validated SQ2-A pipeline)
  3. Compute P_max from 264-step time grid (consistent with SQ2-A)
  4. For each of 56 time points (8/day x 7 days):
     evaluate P(t) via interpolators, compute kappa_c via bisection

Output: mean/std of normalized kappa_c across ensemble -> CSV with checkpoint-resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from model import compute_kappa_c  # noqa: E402
from data_loader import (  # noqa: E402
    build_microgrid_interpolators,
    evaluate_power_vector,
    compute_pmax_from_interpolators,
)
from network import generate_network_with_pcc  # noqa: E402

# --- Paths (match SQ2-A / run_trajectory.py) ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_HOURLY_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "EXPORT HourlyData - Customer Endpoints.csv",
)

N_HOUSES = 49       # GridResilience: 49 houses + 1 PCC = 50 nodes
PENETRATION = 49    # fullpen: all 49 houses have PV
GAMMA = 1.0

# 8 time points per day: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
HOURS_PER_DAY = [0, 3, 6, 9, 12, 15, 18, 21]
N_DAYS = 7

FAST_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 50.0),  # PCC absorbs all excess PV → |P_PCC|~62, needs κ~15-20
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 1.0,
}

PROD_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 50.0),  # PCC absorbs all excess PV → |P_PCC|~62, needs κ~15-20
    "bisection_steps": 20,
    "t_integrate": 100,
    "conv_tol": 1e-3,
    "max_step": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3B: kappa_c time series over one summer week"
    )
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260209)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Use fast config (dev)")
    mode.add_argument("--production", action="store_true", help="Use production config")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing results before running")

    return parser.parse_args()


def get_time_points() -> list[tuple[int, int, float]]:
    """
    Generate the 56 representative time points.

    Returns list of (day, hour, t_seconds) tuples.
    t_seconds is directly usable by interpolators: day * 86400 + hour * 3600.
    """
    points = []
    for day in range(N_DAYS):
        for hour in HOURS_PER_DAY:
            t_sec = day * 86400 + hour * 3600
            points.append((day, hour, t_sec))
    return points


def ensure_output_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day", "hour", "kappa_c_mean", "kappa_c_std"])


def load_completed_points(path: str) -> set[tuple[int, int]]:
    if not os.path.exists(path):
        return set()
    completed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((int(row["day"]), int(row["hour"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    args = parse_args()

    if args.fast:
        sim_config = FAST_CONFIG
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2_B")
        print("Using FAST simulation config")
    elif args.production:
        sim_config = PROD_CONFIG
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2_B")
        print("Using PRODUCTION simulation config")
    else:
        sim_config = FAST_CONFIG
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2_B")
        print("Using FAST simulation config (default)")

    if args.output_dir is None:
        args.output_dir = default_dir

    output_path = os.path.join(args.output_dir, f"kappa_c_timeseries_{args.season}.csv")

    if args.clean and os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Cleaned: {output_path}")

    ensure_output_csv(output_path)

    completed = load_completed_points(output_path)
    time_points = get_time_points()
    remaining = [(d, h, t) for d, h, t in time_points if (d, h) not in completed]

    print(f"Experiment 3B: kappa_c time series")
    print(f"  Season: {args.season}")
    print(f"  N_HOUSES: {N_HOUSES} (+1 PCC = {N_HOUSES + 1} nodes)")
    print(f"  Penetration: {PENETRATION}/{N_HOUSES}")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Time points: {len(time_points)} total, {len(completed)} done, {len(remaining)} remaining")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------
    # Pre-load ensemble: interpolators + network + P_max
    # ---------------------------------------------------------------
    print("\nPre-loading ensemble data...")
    sys.stdout.flush()

    # Each element: (cons_interps, pv_interps, A_csr, P_max) or None
    ensemble = []
    n_loaded = 0
    t_load_start = time.time()

    for i in range(args.n_ensemble):
        instance_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))

        try:
            t0 = time.time()
            cons_interps, pv_interps = build_microgrid_interpolators(
                lcl_dir=LCL_DIR,
                pv_hourly_path=PV_HOURLY_PATH,
                season=args.season,
                n_houses=N_HOUSES,
                penetration=PENETRATION,
                seed=instance_seed,
            )

            # Compute P_max from the 264-step grid (consistent with SQ2-A)
            P_max_i = compute_pmax_from_interpolators(cons_interps, pv_interps)

            # Verify power balance on a sample time point
            P_test = evaluate_power_vector(cons_interps, pv_interps, 86400.0)
            balance = abs(np.sum(P_test))
            if balance > 1e-10:
                print(f"  WARN instance {i+1}: power balance = {balance:.2e}")

            A = generate_network_with_pcc(
                n_houses=N_HOUSES, k=4, q=0.1, n_pcc_links=4, seed=net_seed
            )

            ensemble.append((cons_interps, pv_interps, A, P_max_i))
            n_loaded += 1
            elapsed = time.time() - t0
            print(f"  Loading instance {i+1}/{args.n_ensemble} ({elapsed:.1f}s, P_max={P_max_i:.4f})")
            sys.stdout.flush()

        except Exception as exc:
            print(f"  WARN: Failed to load instance {i+1}: {exc}")
            traceback.print_exc()
            ensemble.append(None)

    t_load = time.time() - t_load_start
    print(f"Pre-loading done: {n_loaded}/{args.n_ensemble} instances in {t_load:.1f}s\n")
    sys.stdout.flush()

    if n_loaded == 0:
        print("ERROR: No ensemble instances loaded. Aborting.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Bisection loop over 56 time points
    # ---------------------------------------------------------------
    n_done = 0
    t_bisect_start = time.time()

    for day, hour, t_sec in time_points:
        if (day, hour) in completed:
            n_done += 1
            continue

        t_point_start = time.time()
        kappa_values = []

        for i in range(args.n_ensemble):
            if ensemble[i] is None:
                continue
            cons_interps, pv_interps, A, P_max = ensemble[i]

            if P_max < 1e-12:
                continue

            try:
                P_t = evaluate_power_vector(cons_interps, pv_interps, t_sec)

                # Verify power balance
                balance = abs(np.sum(P_t))
                if balance > 1e-10:
                    print(f"  WARN: power balance = {balance:.2e} at day={day} h={hour} inst={i}")

                kc = compute_kappa_c(A, P_t, config_params=sim_config)
                kc_normalized = kc / P_max
                kappa_values.append(kc_normalized)

            except Exception as exc:
                print(f"  WARN instance {i+1} failed at day={day} h={hour}: {exc}")
                continue

        if not kappa_values:
            print(f"  ERROR: No valid samples for day={day} hour={hour:02d}")
            sys.stdout.flush()
            continue

        kv = np.array(kappa_values)
        mean_kc = float(np.mean(kv))
        std_kc = float(np.std(kv))
        n_done += 1
        elapsed = time.time() - t_point_start

        append_row(output_path, [day, hour, f"{mean_kc:.6f}", f"{std_kc:.6f}"])

        # ETA estimate
        total_elapsed = time.time() - t_bisect_start
        remaining_pts = len(time_points) - n_done
        if n_done > len(completed):
            pts_computed = n_done - len(completed)
            avg_per_pt = total_elapsed / pts_computed
            eta_s = avg_per_pt * remaining_pts
            eta_str = f"ETA {eta_s/60:.1f}min"
        else:
            eta_str = ""

        print(
            f"  [{n_done}/{len(time_points)}] Day {day} {hour:02d}:00 -- "
            f"kc_mean={mean_kc:.4f} +/- {std_kc:.4f} "
            f"({len(kappa_values)} valid, {elapsed:.1f}s) {eta_str}"
        )
        sys.stdout.flush()

    total_time = time.time() - t_bisect_start
    print(f"\nExperiment 3B complete in {total_time/60:.1f} min.")
    print(f"Results: {output_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
