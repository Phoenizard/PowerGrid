"""
Run Experiment 2A: effect of power heterogeneity on critical coupling.

Features:
- argparse options: --n_ensemble, --output
- per-instance progress logs
- checkpoint append to CSV after each sigma_ratio point
- resume support by skipping completed sigma_ratio points
- per-instance error tolerance
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

from model import generate_network, compute_kappa_c_normalized  # noqa: E402
from power_allocation import assign_power_heterogeneous  # noqa: E402


N = 50
N_PLUS = 25
N_MINUS = 25
K = 4
Q = 0.1
GAMMA = 1.0
P_MAX = 1.0
SIGMA_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

SIM_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 3.0),
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 5.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment 2A")
    parser.add_argument("--n_ensemble", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results", "results_2A.csv"),
    )
    return parser.parse_args()


def ensure_output_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sigma_ratio",
                    "kappa_c_mean_gen",
                    "kappa_c_std_gen",
                    "kappa_c_mean_con",
                    "kappa_c_std_con",
                ]
            )


def load_completed_sigma(path: str) -> set[float]:
    if not os.path.exists(path):
        return set()

    completed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(float(row["sigma_ratio"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(path: str, row: list[float]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def map_ordered_power_to_nodes(p_ordered: np.ndarray, generator_indices: np.ndarray, consumer_indices: np.ndarray) -> np.ndarray:
    p = np.zeros(N, dtype=float)
    p[generator_indices] = p_ordered[:N_PLUS]
    p[consumer_indices] = p_ordered[N_PLUS:]
    return p


def run_for_sigma(sigma_ratio: float, n_ensemble: int, rng: np.random.Generator):
    kcs_gen = []
    kcs_con = []

    for i in range(1, n_ensemble + 1):
        try:
            net_seed = int(rng.integers(0, 2**31 - 1))
            a = generate_network(n=N, k=K, q=Q, seed=net_seed)

            indices = rng.permutation(N)
            generator_indices = indices[:N_PLUS]
            consumer_indices = indices[N_PLUS: N_PLUS + N_MINUS]

            p_gen_ordered = assign_power_heterogeneous(
                N_PLUS,
                N_MINUS,
                P_MAX,
                sigma_ratio,
                side="gen",
                rng=rng,
            )
            p_con_ordered = assign_power_heterogeneous(
                N_PLUS,
                N_MINUS,
                P_MAX,
                sigma_ratio,
                side="con",
                rng=rng,
            )

            p_gen = map_ordered_power_to_nodes(p_gen_ordered, generator_indices, consumer_indices)
            p_con = map_ordered_power_to_nodes(p_con_ordered, generator_indices, consumer_indices)

            kc_gen = compute_kappa_c_normalized(a, p_gen, P_MAX, config_params=SIM_CONFIG)
            kc_con = compute_kappa_c_normalized(a, p_con, P_MAX, config_params=SIM_CONFIG)

            print(
                f"[sigma_ratio={sigma_ratio}] instance {i}/{n_ensemble}, "
                f"kappa_c_gen={kc_gen:.6f}, kappa_c_con={kc_con:.6f}"
            )

            kcs_gen.append(float(kc_gen))
            kcs_con.append(float(kc_con))
        except Exception as exc:
            print(
                f"WARN [sigma_ratio={sigma_ratio}] instance {i}/{n_ensemble} failed: {exc}"
            )
            traceback.print_exc()
            continue

    if not kcs_gen or not kcs_con:
        raise RuntimeError(f"No valid samples for sigma_ratio={sigma_ratio}")

    return (
        float(np.mean(kcs_gen)),
        float(np.std(kcs_gen)),
        float(np.mean(kcs_con)),
        float(np.std(kcs_con)),
    )


def main():
    args = parse_args()
    ensure_output_csv(args.output)

    completed = load_completed_sigma(args.output)
    rng = np.random.default_rng(20260208)

    print(f"Output CSV: {args.output}")
    print(f"Completed sigma points: {sorted(completed)}")

    for sigma_ratio in SIGMA_RATIOS:
        if float(sigma_ratio) in completed:
            print(f"Skip sigma_ratio={sigma_ratio}: already completed")
            continue

        print(f"Start sigma_ratio={sigma_ratio}")
        mean_gen, std_gen, mean_con, std_con = run_for_sigma(sigma_ratio, args.n_ensemble, rng)

        append_row(args.output, [sigma_ratio, mean_gen, std_gen, mean_con, std_con])
        print(
            f"Saved sigma_ratio={sigma_ratio}: "
            f"gen(mean={mean_gen:.6f}, std={std_gen:.6f}), "
            f"con(mean={mean_con:.6f}, std={std_con:.6f})"
        )

    print("Experiment 2A complete")


if __name__ == "__main__":
    main()
