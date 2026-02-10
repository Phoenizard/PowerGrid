"""
Experiment 3A: Track microgrid simplex trajectory over one summer week.

Replicates GridResilience/scripts/powerexperiments.py exactly:
  - 49 houses + 1 PCC = 50 nodes
  - interp1d continuous sampling at np.linspace(0, 604800-1800, 336)[48:]
  - 264 timesteps (skip first 24h)
  - Full PV penetration (49/49) by default
  - Continuous simplex formula on full Pvec including PCC

For each ensemble instance:
  1. Build random microgrid (LCL consumption + PV generation)
  2. Compute net power P(t) at 264 time points
  3. Compute continuous simplex densities (sigma_s, sigma_d, sigma_p)
  4. Store trajectory (264, 3)

Output: mean/std of trajectories across ensemble → CSV.
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

from data_loader import build_microgrid, N_TIMESTEPS
from simplex import compute_simplex_trajectory

# --- Paths ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_HOURLY_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "EXPORT HourlyData - Customer Endpoints.csv",
)

N_HOUSES = 49       # GridResilience: n-1 = 49 houses
PENETRATION = 49    # fullpen: all 49 houses have PV


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3A: Simplex trajectory (GridResilience-matched)"
    )
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--penetration", type=int, default=PENETRATION,
                        help="Number of houses with PV (24=halfpen, 49=fullpen)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260209)
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing results before running")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "results_sq2_A")

    os.makedirs(args.output_dir, exist_ok=True)

    pen_label = "fullpen" if args.penetration >= N_HOUSES else f"pen{args.penetration}"
    output_path = os.path.join(
        args.output_dir, f"trajectory_{args.season}_{pen_label}.csv"
    )

    if args.clean:
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"  Cleaned: {output_path}")

    print(f"Experiment 3A: Simplex trajectory (GridResilience-matched)")
    print(f"  Season: {args.season}")
    print(f"  Houses: {N_HOUSES}, Penetration: {args.penetration}/{N_HOUSES}")
    print(f"  Total nodes: {N_HOUSES + 1} (houses + PCC)")
    print(f"  Timesteps: {N_TIMESTEPS} (skip first 24h)")
    print(f"  Ensemble size: {args.n_ensemble}")
    print(f"  Output: {output_path}")

    rng = np.random.default_rng(args.seed)

    # Collect trajectories: (n_ensemble, N_TIMESTEPS, 3)
    all_trajectories: list[np.ndarray] = []

    for i in range(1, args.n_ensemble + 1):
        try:
            instance_seed = int(rng.integers(0, 2**31))

            P, t_seconds = build_microgrid(
                lcl_dir=LCL_DIR,
                pv_hourly_path=PV_HOURLY_PATH,
                season=args.season,
                n_houses=N_HOUSES,
                penetration=args.penetration,
                seed=instance_seed,
            )

            # Diagnostic: print data magnitudes for first instance
            if i == 1:
                print(f"\n=== Diagnostic (instance 1) ===")
                print(f"  P shape: {P.shape} (nodes x timesteps)")
                print(f"  P houses mean: {P[:N_HOUSES].mean():.6f}")
                print(f"  P PCC mean: {P[N_HOUSES].mean():.6f}")
                print(f"  P houses max: {P[:N_HOUSES].max():.4f}")
                print(f"  P houses min: {P[:N_HOUSES].min():.4f}")
                print(f"  P PCC max: {P[N_HOUSES].max():.4f}")
                print(f"  P PCC min: {P[N_HOUSES].min():.4f}")
                print()

            # Verify power balance
            balance = np.abs(P.sum(axis=0))
            assert np.all(balance < 1e-10), f"Power balance violation: max={balance.max()}"

            # Compute simplex trajectory (all nodes incl. PCC)
            traj = compute_simplex_trajectory(P)
            all_trajectories.append(traj)

            print(
                f"  Instance {i}/{args.n_ensemble}: "
                f"sigma_s=[{traj[:,0].min():.4f},{traj[:,0].max():.4f}], "
                f"sigma_d=[{traj[:,1].min():.4f},{traj[:,1].max():.4f}], "
                f"sigma_p=[{traj[:,2].min():.4f},{traj[:,2].max():.4f}]"
            )

        except Exception as exc:
            print(f"  WARN instance {i}/{args.n_ensemble} failed: {exc}")
            traceback.print_exc()
            continue

    if not all_trajectories:
        raise RuntimeError("No valid ensemble instances")

    # Stack and compute statistics
    stacked = np.array(all_trajectories)  # (n_valid, N_TIMESTEPS, 3)
    mean_traj = stacked.mean(axis=0)  # (N_TIMESTEPS, 3)
    std_traj = stacked.std(axis=0)  # (N_TIMESTEPS, 3)

    # Build time info: hours within the week (starting from hour 24)
    hours = (t_seconds / 3600.0)  # convert seconds to hours

    # Save per-instance trajectories as .npz (for Fig.4 plotting)
    npz_path = output_path.replace(".csv", ".npz")
    np.savez_compressed(npz_path, trajectories=stacked, t_seconds=t_seconds)
    print(f"Per-instance data saved to {npz_path}")

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestep", "hour",
            "eta_plus_mean", "eta_plus_std",
            "eta_minus_mean", "eta_minus_std",
            "eta_p_mean", "eta_p_std",
        ])
        for t in range(N_TIMESTEPS):
            writer.writerow([
                t,
                f"{hours[t]:.2f}",
                f"{mean_traj[t, 0]:.6f}",
                f"{std_traj[t, 0]:.6f}",
                f"{mean_traj[t, 1]:.6f}",
                f"{std_traj[t, 1]:.6f}",
                f"{mean_traj[t, 2]:.6f}",
                f"{std_traj[t, 2]:.6f}",
            ])

    print(f"\nResults saved to {output_path}")
    print(f"Valid ensemble instances: {len(all_trajectories)}/{args.n_ensemble}")
    print(f"Shape: {N_TIMESTEPS} rows x 8 columns")

    # Print summary statistics
    print(f"\n=== Ensemble summary ===")
    print(f"  sigma_s: min={mean_traj[:,0].min():.4f}  max={mean_traj[:,0].max():.4f}  mean={mean_traj[:,0].mean():.4f}")
    print(f"  sigma_d: min={mean_traj[:,1].min():.4f}  max={mean_traj[:,1].max():.4f}  mean={mean_traj[:,1].mean():.4f}")
    print(f"  sigma_p: min={mean_traj[:,2].min():.4f}  max={mean_traj[:,2].max():.4f}  mean={mean_traj[:,2].mean():.4f}")


if __name__ == "__main__":
    main()
