"""
Experiment 3A: Track microgrid simplex trajectory over one summer week.

For each ensemble instance:
  1. Load random LCL + PV data for a summer week (336 half-hour steps)
  2. Compute net power P(t) for 50 houses + 1 PCC
  3. Compute simplex coordinates (eta+, eta-, eta_p) at each timestep
     using only the 50 house nodes (excluding PCC)
  4. Store trajectory (336, 3)

Output: mean/std of trajectories across ensemble → CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

from data_loader import load_lcl_households, load_pv_generation, compute_net_power
from simplex import compute_simplex_trajectory

# --- Paths ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT TenMinData",
    "EXPORT TenMinData - Customer Endpoints.csv",
)

N_HOUSES = 50
STEPS_PER_WEEK = 336  # 48 * 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3A: Simplex trajectory over one summer week"
    )
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260209)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "results_sq2")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"trajectory_{args.season}.csv")

    print(f"Experiment 3A: Simplex trajectory")
    print(f"  Season: {args.season}")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Output: {output_path}")

    rng = np.random.default_rng(args.seed)

    # Collect trajectories: (n_ensemble, 336, 3)
    all_trajectories: list[np.ndarray] = []
    time_index_ref = None

    for i in range(1, args.n_ensemble + 1):
        try:
            lcl_seed = int(rng.integers(0, 2**31))
            pv_seed = int(rng.integers(0, 2**31))

            consumption, lcl_time = load_lcl_households(
                LCL_DIR, season=args.season, n_households=N_HOUSES, seed=lcl_seed
            )
            generation, pv_time = load_pv_generation(
                PV_PATH, season=args.season, n_panels=N_HOUSES, seed=pv_seed
            )

            # Compute net power (51 nodes x 336 steps)
            P = compute_net_power(consumption, generation)

            # Verify power balance
            balance = np.abs(P.sum(axis=0))
            assert np.all(balance < 1e-10), f"Power balance violation: max={balance.max()}"

            # Compute simplex trajectory (house nodes only)
            traj = compute_simplex_trajectory(P, n_houses=N_HOUSES)
            all_trajectories.append(traj)

            if time_index_ref is None:
                time_index_ref = lcl_time

            print(
                f"  Instance {i}/{args.n_ensemble}: "
                f"eta+=[{traj[:,0].min():.4f},{traj[:,0].max():.4f}], "
                f"eta-=[{traj[:,1].min():.4f},{traj[:,1].max():.4f}], "
                f"eta_p=[{traj[:,2].min():.4f},{traj[:,2].max():.4f}]"
            )

        except Exception as exc:
            print(f"  WARN instance {i}/{args.n_ensemble} failed: {exc}")
            traceback.print_exc()
            continue

    if not all_trajectories:
        raise RuntimeError("No valid ensemble instances")

    # Stack and compute statistics
    stacked = np.array(all_trajectories)  # (n_valid, 336, 3)
    mean_traj = stacked.mean(axis=0)  # (336, 3)
    std_traj = stacked.std(axis=0)  # (336, 3)

    # Build time info: hours within the week
    hours = np.arange(STEPS_PER_WEEK) * 0.5  # 0, 0.5, 1.0, ..., 167.5

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestep", "hour",
            "eta_plus_mean", "eta_plus_std",
            "eta_minus_mean", "eta_minus_std",
            "eta_p_mean", "eta_p_std",
        ])
        for t in range(STEPS_PER_WEEK):
            writer.writerow([
                t,
                f"{hours[t]:.1f}",
                f"{mean_traj[t, 0]:.6f}",
                f"{std_traj[t, 0]:.6f}",
                f"{mean_traj[t, 1]:.6f}",
                f"{std_traj[t, 1]:.6f}",
                f"{mean_traj[t, 2]:.6f}",
                f"{std_traj[t, 2]:.6f}",
            ])

    print(f"\nResults saved to {output_path}")
    print(f"Valid ensemble instances: {len(all_trajectories)}/{args.n_ensemble}")
    print(f"Shape: {STEPS_PER_WEEK} rows x 8 columns")


if __name__ == "__main__":
    main()
