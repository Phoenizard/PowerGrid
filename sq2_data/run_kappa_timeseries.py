"""
Experiment 3B: Compute kappa_c at 56 representative time points.

For each ensemble instance:
  1. Generate WS+PCC network (51 nodes)
  2. Load random LCL + PV data for a summer week
  3. Compute net power P(t)
  4. For each of 56 time points (8/day x 7 days):
     compute kappa_c using the full 51-node power vector (including PCC)

Output: mean/std of kappa_c across ensemble → CSV with checkpoint-resume.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from model import compute_kappa_c  # noqa: E402
from data_loader import load_lcl_households, load_pv_generation, compute_net_power
from network import generate_network_with_pcc

# --- Paths ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT TenMinData",
    "EXPORT TenMinData - Customer Endpoints.csv",
)

N_HOUSES = 50
GAMMA = 1.0

# 8 time points per day: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
HOURS_PER_DAY = [0, 3, 6, 9, 12, 15, 18, 21]
N_DAYS = 7
STEPS_PER_HOUR = 2  # 30-min resolution → 2 steps per hour

FAST_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 3.0),
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 1.0,
}

PROD_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 3.0),
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


def get_time_indices() -> list[tuple[int, int, int]]:
    """
    Generate the 56 representative time indices.

    Returns list of (day, hour, timestep_index) tuples.
    day: 0-6, hour: one of HOURS_PER_DAY, timestep_index: position in 336-step array.
    """
    indices = []
    for day in range(N_DAYS):
        for hour in HOURS_PER_DAY:
            t_idx = day * 48 + hour * STEPS_PER_HOUR
            indices.append((day, hour, t_idx))
    return indices


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
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2_fast")
        print("Using FAST simulation config")
    elif args.production:
        sim_config = PROD_CONFIG
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2")
        print("Using PRODUCTION simulation config")
    else:
        sim_config = FAST_CONFIG
        default_dir = os.path.join(SCRIPT_DIR, "results_sq2_fast")
        print("Using FAST simulation config (default)")

    if args.output_dir is None:
        args.output_dir = default_dir

    output_path = os.path.join(args.output_dir, f"kappa_c_timeseries_{args.season}.csv")

    if args.clean and os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Cleaned: {output_path}")

    ensure_output_csv(output_path)

    completed = load_completed_points(output_path)
    time_points = get_time_indices()

    print(f"Experiment 3B: kappa_c time series")
    print(f"  Season: {args.season}")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Time points: {len(time_points)} ({len(completed)} already completed)")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    rng = np.random.default_rng(args.seed)

    # Pre-load ALL ensemble data (power vectors + networks) once
    # This avoids re-reading CSVs for every (time_point, instance) pair
    print("\nPre-loading ensemble data...")
    sys.stdout.flush()
    ensemble_P = []      # list of (51, 336) arrays
    ensemble_A = []      # list of csr_matrix (51, 51)
    ensemble_Pmax = []   # list of float: per-instance P_max (house nodes only)
    n_loaded = 0

    for i in range(args.n_ensemble):
        lcl_seed = int(rng.integers(0, 2**31))
        pv_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))

        try:
            consumption, _ = load_lcl_households(
                LCL_DIR, season=args.season, n_households=N_HOUSES, seed=lcl_seed
            )
            generation, _ = load_pv_generation(
                PV_PATH, season=args.season, n_panels=N_HOUSES, seed=pv_seed
            )
            P = compute_net_power(consumption, generation)
            A = generate_network_with_pcc(
                n_houses=N_HOUSES, k=4, q=0.1, n_pcc_links=4, seed=net_seed
            )
            P_max_i = float(np.max(np.abs(P[:N_HOUSES, :])))
            ensemble_P.append(P)
            ensemble_A.append(A)
            ensemble_Pmax.append(P_max_i)
            n_loaded += 1
            print(f"  Loaded instance {i+1}/{args.n_ensemble}")
            sys.stdout.flush()
        except Exception as exc:
            print(f"  WARN: Failed to load instance {i+1}: {exc}")
            traceback.print_exc()
            ensemble_P.append(None)
            ensemble_A.append(None)
            ensemble_Pmax.append(None)

    print(f"Pre-loading done: {n_loaded}/{args.n_ensemble} instances loaded\n")
    sys.stdout.flush()

    for day, hour, t_idx in time_points:
        if (day, hour) in completed:
            print(f"Skip day={day} hour={hour:02d}: already completed")
            sys.stdout.flush()
            continue

        print(f"Processing day={day} hour={hour:02d}:00 (t_idx={t_idx})...")
        sys.stdout.flush()
        kappa_values = []

        for i in range(args.n_ensemble):
            P = ensemble_P[i]
            A = ensemble_A[i]
            P_max = ensemble_Pmax[i]
            if P is None or A is None or P_max is None:
                continue

            if P_max < 1e-12:
                continue

            try:
                P_t = P[:, t_idx]

                kc = compute_kappa_c(A, P_t, config_params=sim_config)
                kc_normalized = kc / P_max
                kappa_values.append(kc_normalized)

            except Exception as exc:
                print(f"  WARN instance {i+1} failed: {exc}")
                continue

        if not kappa_values:
            print(f"  ERROR: No valid samples for day={day} hour={hour:02d}")
            sys.stdout.flush()
            continue

        kv = np.array(kappa_values)
        mean_kc = float(np.mean(kv))
        std_kc = float(np.std(kv))
        n_small = int(np.sum(kv < 0.01))

        append_row(output_path, [day, hour, f"{mean_kc:.6f}", f"{std_kc:.6f}"])
        print(
            f"  Saved day={day} hour={hour:02d}: "
            f"kappa_c_mean={mean_kc:.6f}, kappa_c_std={std_kc:.6f} "
            f"(n={len(kappa_values)}, min={float(kv.min()):.6f}, "
            f"max={float(kv.max()):.6f}, n_small={n_small})"
        )
        sys.stdout.flush()

    print(f"\nExperiment 3B complete. Results: {output_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
