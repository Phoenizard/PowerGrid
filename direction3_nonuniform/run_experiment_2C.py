"""
Run Experiment 2C: centralized vs distributed generation.

Features:
- argparse options: --n_ensemble, --output
- per-instance progress logs
- checkpoint append to CSV after each r point
- resume support by skipping completed r points
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
from power_allocation import assign_power_centralized  # noqa: E402


N = 50
N_PLUS = 25
N_MINUS = 25
K = 4
Q = 0.1
GAMMA = 1.0
P_MAX = 1.0
R_VALUES = [0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

SIM_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 3.0),
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 5.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment 2C")
    parser.add_argument("--n_ensemble", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results", "results_2C.csv"),
    )
    return parser.parse_args()


def ensure_output_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["r", "kappa_c_mean", "kappa_c_std"])


def load_completed_r(path: str) -> set[float]:
    if not os.path.exists(path):
        return set()

    completed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(float(row["r"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(path: str, row: list[float]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_node_power_for_centralized(r: float, rng: np.random.Generator) -> np.ndarray:
    ordered = assign_power_centralized(N_PLUS, N_MINUS, P_MAX, r)

    # Randomize which generator is the large station.
    gen_powers = ordered[:N_PLUS].copy()
    con_powers = ordered[N_PLUS:].copy()
    big_pos = int(rng.integers(0, N_PLUS))
    gen_powers[[0, big_pos]] = gen_powers[[big_pos, 0]]

    indices = rng.permutation(N)
    generator_indices = indices[:N_PLUS]
    consumer_indices = indices[N_PLUS: N_PLUS + N_MINUS]

    p = np.zeros(N, dtype=float)
    p[generator_indices] = gen_powers
    p[consumer_indices] = con_powers
    return p


def run_for_r(r: float, n_ensemble: int, rng: np.random.Generator):
    kcs = []

    for i in range(1, n_ensemble + 1):
        try:
            net_seed = int(rng.integers(0, 2**31 - 1))
            a = generate_network(n=N, k=K, q=Q, seed=net_seed)
            p = build_node_power_for_centralized(r, rng)
            kc = compute_kappa_c_normalized(a, p, P_MAX, config_params=SIM_CONFIG)

            print(f"[r={r}] instance {i}/{n_ensemble}, kappa_c={kc:.6f}")
            kcs.append(float(kc))
        except Exception as exc:
            print(f"WARN [r={r}] instance {i}/{n_ensemble} failed: {exc}")
            traceback.print_exc()
            continue

    if not kcs:
        raise RuntimeError(f"No valid samples for r={r}")

    return float(np.mean(kcs)), float(np.std(kcs))


def main():
    args = parse_args()
    ensure_output_csv(args.output)

    completed = load_completed_r(args.output)
    rng = np.random.default_rng(20260208)

    print(f"Output CSV: {args.output}")
    print(f"Completed r points: {sorted(completed)}")

    for r in R_VALUES:
        if float(r) in completed:
            print(f"Skip r={r}: already completed")
            continue

        print(f"Start r={r}")
        mean_k, std_k = run_for_r(r, args.n_ensemble, rng)
        append_row(args.output, [r, mean_k, std_k])
        print(f"Saved r={r}: mean={mean_k:.6f}, std={std_k:.6f}")

    print("Experiment 2C complete")


if __name__ == "__main__":
    main()
