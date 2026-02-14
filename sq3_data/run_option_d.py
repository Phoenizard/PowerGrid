"""
SQ3 Option D — Validate Smith's indirect rho transfer on non-PCC networks.

100 WS(n=50, K_bar=4, q=0.1) networks with equal gen/con power P_i = ±1/25.
No PCC node, no edge addition.
Expect: smooth sigmoid, bisection convergence >90%, rho ~log-normal.

Outputs:
  results/option_d/sq3_option_d_rho_distribution.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")

for _d in (PAPER_DIR, SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from cascade_engine import (
    compute_dc_fmax,
    find_alpha_c,
    sweep_alpha,
)

# --- Config ---
N_INSTANCES = 100  # WS network instances
N_NODES = 50
K_BAR = 4
Q_REWIRE = 0.1
SEED = 20260214
N_ALPHA_POINTS = 50
ALPHA_RANGE = (0.1, 30.0)
BISECTION_TOL = 1e-3

# Output paths
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "option_d")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_ws_network(n, k, q, seed):
    """Create a WS small-world network and return CSR adjacency."""
    G = nx.watts_strogatz_graph(n, k, q, seed=seed)
    A = nx.adjacency_matrix(G).astype(float)
    return A.tocsr()


def make_balanced_power(n, rng):
    """Create balanced power: 25 sources (+1/25) and 25 sinks (-1/25), shuffled."""
    P = np.zeros(n)
    P[:n // 2] = 1.0 / (n // 2)
    P[n // 2:] = -1.0 / (n // 2)
    rng.shuffle(P)
    return P


def run_option_d():
    """Run Option D validation experiment."""
    print("=" * 60)
    print("  SQ3 OPTION D — NON-PCC NETWORK VALIDATION")
    print(f"  n_instances={N_INSTANCES}, n={N_NODES}, K={K_BAR}, q={Q_REWIRE}")
    print("=" * 60)

    t_start = time.time()
    rng = np.random.default_rng(SEED)

    output_path = os.path.join(RESULTS_DIR, "sq3_option_d_rho_distribution.csv")
    sweep_path = os.path.join(RESULTS_DIR, "sq3_option_d_sweep.csv")

    rows = []
    sweep_rows = []
    n_converged = 0
    alphas_rel = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], N_ALPHA_POINTS)

    for i in range(N_INSTANCES):
        net_seed = int(rng.integers(0, 2**31))
        A = make_ws_network(N_NODES, K_BAR, Q_REWIRE, seed=net_seed)
        P = make_balanced_power(N_NODES, rng)

        # Power balance check
        balance = abs(np.sum(P))
        if balance > 1e-10:
            print(f"  WARN inst {i}: power balance = {balance:.2e}")

        # DC f_max
        _, f_max_dc, _ = compute_dc_fmax(A, P)
        if f_max_dc < 1e-12:
            print(f"  WARN inst {i}: f_max_dc ≈ 0, skipping")
            rows.append({
                "instance_id": i,
                "alpha_star": 0.0,
                "alpha_c": np.nan,
                "rho": np.nan,
                "bisection_converged": False,
            })
            continue

        # Alpha sweep
        for alpha_ratio in alphas_rel:
            alpha_abs = alpha_ratio * f_max_dc
            from cascade_engine import run_cascade_dc
            result = run_cascade_dc(A, P, alpha_abs, f_max_dc)
            sweep_rows.append({
                "instance_id": i,
                "alpha_over_alpha_star": alpha_ratio,
                "S": result.S,
                "T_rounds": result.cascade_depth,
            })

        # Bisection
        alpha_c, S_ac = find_alpha_c(A, P, f_max=f_max_dc)
        converged = abs(S_ac - 0.5) < 0.2
        rho = alpha_c / f_max_dc if converged else np.nan
        if converged:
            n_converged += 1

        rows.append({
            "instance_id": i,
            "alpha_star": f_max_dc,
            "alpha_c": alpha_c,
            "rho": rho,
            "bisection_converged": converged,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{N_INSTANCES}] f_max={f_max_dc:.4f}, "
                f"alpha_c={alpha_c:.4f}, rho={rho:.4f}, conv={converged}"
            )
            sys.stdout.flush()

    # Save rho distribution CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_id", "alpha_star", "alpha_c", "rho", "bisection_converged",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Rho CSV: {output_path} ({len(rows)} rows)")

    # Save sweep CSV
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_id", "alpha_over_alpha_star", "S", "T_rounds",
        ])
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"  Sweep CSV: {sweep_path} ({len(sweep_rows)} rows)")

    # Summary
    rhos_valid = [r["rho"] for r in rows if not np.isnan(r["rho"])]
    t_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("  OPTION D SUMMARY")
    print("=" * 60)
    print(f"  Convergence rate: {n_converged}/{N_INSTANCES} ({100*n_converged/N_INSTANCES:.0f}%)")
    if rhos_valid:
        print(f"  rho: mean={np.mean(rhos_valid):.4f}, std={np.std(rhos_valid):.4f}")
        print(f"  rho: median={np.median(rhos_valid):.4f}, "
              f"range=[{np.min(rhos_valid):.4f}, {np.max(rhos_valid):.4f}]")
    print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    run_option_d()
