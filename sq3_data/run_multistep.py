"""
SQ3 Multi-Timestep Cascade Experiment (Priority 1).

Parameters:
  Timesteps: {0, 21600, 32400, 43200, 64800} (00:00, 06:00, 09:00, 12:00, 18:00)
  m configs: {0, 4, 8} pcc_direct; 4_random
  n_ensemble: configurable (default 10 for sanity, 50 for production)
  Seed: 20260214
  Alpha/alpha* sweep: 50 points in [0.1, 2.5]
  DC cascade only (swing deferred)

Outputs:
  results/multistep/sq3_multistep_sweep.csv
  results/multistep/sq3_multistep_summary.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from scipy.sparse import lil_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
SQ4_DIR = os.path.join(PROJECT_ROOT, "sq4_data")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")

for _d in (SQ2_DIR, SQ4_DIR, PAPER_DIR, SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from cascade_engine import (
    compute_dc_fmax,
    run_cascade_dc_tracked,
    find_alpha_c,
)
from kappa_pipeline import preload_ensemble, EnsembleInstance
from edge_strategies import (
    select_edges_pcc_direct,
    select_edges_random,
    compute_node_power_stats,
)
from data_loader import evaluate_power_vector

# --- Config ---
N_ENSEMBLE = 10  # Set to 50 for production
SEED = 20260214
TIMESTEPS = {
    "00:00": 0,
    "06:00": 21600,
    "09:00": 32400,
    "12:00": 43200,
    "18:00": 64800,
}
M_CONFIGS = [
    (0, "pcc_direct"),
    (4, "pcc_direct"),
    (8, "pcc_direct"),
    (4, "random"),
]
N_ALPHA_POINTS = 50
ALPHA_RANGE = (0.1, 2.5)
BISECTION_TOL = 1e-3
SEASON = "summer"
PCC_NODE = 49  # node index for PCC

# Output paths
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "multistep")
os.makedirs(RESULTS_DIR, exist_ok=True)


def add_edges_to_instance(
    inst: EnsembleInstance, m: int, strategy: str
) -> tuple:
    """Add m edges using the given strategy. Returns A_modified."""
    if m == 0:
        return inst.A_base

    P_max_node, P_sign_node = compute_node_power_stats(
        inst.cons_interps, inst.pv_interps
    )
    rng_edge = np.random.default_rng(inst.net_seed + 1_000_000)
    A_lil = lil_matrix(inst.A_base)

    if strategy == "pcc_direct":
        return select_edges_pcc_direct(A_lil, m, P_max_node, P_sign_node, rng_edge)
    elif strategy == "random":
        return select_edges_random(A_lil, m, P_max_node, P_sign_node, rng_edge)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def classify_edges(surviving_edges: set, pcc_node: int = PCC_NODE):
    """Classify surviving edges into PCC and non-PCC."""
    n_pcc = 0
    n_nonpcc = 0
    for (i, j) in surviving_edges:
        if i == pcc_node or j == pcc_node:
            n_pcc += 1
        else:
            n_nonpcc += 1
    return n_pcc, n_nonpcc


def run_multistep():
    """Run the full multi-timestep cascade experiment."""
    print("=" * 60)
    print("  SQ3 MULTI-TIMESTEP CASCADE EXPERIMENT")
    print(f"  n_ensemble={N_ENSEMBLE}, seed={SEED}")
    print(f"  timesteps={list(TIMESTEPS.keys())}")
    print(f"  m_configs={M_CONFIGS}")
    print("=" * 60)

    t_start = time.time()

    # ---- Step 1: Preload ensemble ----
    print("\n[1/3] Pre-loading ensemble...")
    ensemble = preload_ensemble(
        n_ensemble=N_ENSEMBLE,
        season=SEASON,
        seed=SEED,
    )
    valid_instances = [(i, inst) for i, inst in enumerate(ensemble) if inst is not None]
    n_valid = len(valid_instances)
    print(f"  {n_valid}/{N_ENSEMBLE} valid instances loaded")
    t_preload = time.time() - t_start
    print(f"  Preload time: {t_preload:.1f}s\n")

    # ---- Step 2: Sweep + bisection ----
    print("[2/3] Running alpha sweeps and bisection...")

    sweep_path = os.path.join(RESULTS_DIR, "sq3_multistep_sweep.csv")
    summary_path = os.path.join(RESULTS_DIR, "sq3_multistep_summary.csv")

    sweep_rows = []
    summary_rows = []

    alphas_rel = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], N_ALPHA_POINTS)

    for t_label, t_sec in TIMESTEPS.items():
        print(f"\n  === Timestep {t_label} (t={t_sec}s) ===")

        for m, strategy in M_CONFIGS:
            config_label = f"m={m}_{strategy}"
            print(f"\n    --- {config_label} ---")
            t_config_start = time.time()

            alpha_stars = []
            alpha_cs = []
            rhos = []
            bisection_converged_list = []
            pcc_isolation_at_low = []
            S_at_alpha_star_list = []

            for idx, (i, inst) in enumerate(valid_instances):
                # Add edges
                A_mod = add_edges_to_instance(inst, m, strategy)

                # Evaluate P at timestep, normalize
                P_raw = evaluate_power_vector(inst.cons_interps, inst.pv_interps, t_sec)
                P_norm = P_raw / inst.P_max

                # Power balance check
                balance = abs(np.sum(P_norm))
                if balance > 1e-6:
                    print(f"      WARN inst {i}: power balance = {balance:.2e}")

                # DC f_max
                _, f_max_dc, _ = compute_dc_fmax(A_mod, P_norm)
                alpha_star = f_max_dc
                alpha_stars.append(alpha_star)

                if f_max_dc < 1e-12:
                    print(f"      WARN inst {i}: f_max_dc ≈ 0, skipping")
                    continue

                # Alpha sweep (50 points)
                for a_idx, alpha_ratio in enumerate(alphas_rel):
                    result, surviving_edges = run_cascade_dc_tracked(
                        A_mod, P_norm, alpha_ratio, f_max_dc
                    )
                    n_pcc_surv, n_nonpcc_surv = classify_edges(surviving_edges)
                    pcc_isolated = 1 if n_pcc_surv == 0 else 0

                    sweep_rows.append({
                        "timestep": t_label,
                        "m": m,
                        "strategy": strategy,
                        "instance_id": i,
                        "alpha_over_alpha_star": alpha_ratio,
                        "S": result.S,
                        "T_rounds": result.cascade_depth,
                        "n_pcc_survived": n_pcc_surv,
                        "n_nonpcc_survived": n_nonpcc_surv,
                        "pcc_isolated": pcc_isolated,
                        "alpha_star": alpha_star,
                    })

                    # Track S at alpha/alpha*=1.0
                    if a_idx == 0 or abs(alpha_ratio - 1.0) < abs(alphas_rel[0] - 1.0):
                        pass  # handled below

                # Find S closest to alpha/alpha* = 1.0
                closest_idx = np.argmin(np.abs(alphas_rel - 1.0))
                res_at_star, _ = run_cascade_dc_tracked(
                    A_mod, P_norm, alphas_rel[closest_idx], f_max_dc
                )
                S_at_alpha_star_list.append(res_at_star.S)

                # Bisection
                alpha_c, S_ac = find_alpha_c(A_mod, P_norm, f_max=f_max_dc)
                bisection_converged = abs(S_ac - 0.5) < 0.2
                bisection_converged_list.append(bisection_converged)

                if bisection_converged:
                    rho = alpha_c
                    alpha_cs.append(alpha_c)
                    rhos.append(rho)
                else:
                    alpha_cs.append(np.nan)
                    rhos.append(np.nan)

                # PCC isolation at low alpha (alpha/alpha* = 0.1)
                alpha_low = 0.1
                res_low, edges_low = run_cascade_dc_tracked(
                    A_mod, P_norm, alpha_low, f_max_dc
                )
                n_pcc_low, _ = classify_edges(edges_low)
                pcc_isolation_at_low.append(1 if n_pcc_low == 0 else 0)

                print(
                    f"      [{idx+1}/{n_valid}] inst {i}: "
                    f"f_max={f_max_dc:.4f}, alpha_c={alpha_c:.4f}, "
                    f"conv={bisection_converged}"
                )
                sys.stdout.flush()

            # Summary for this (timestep, m, strategy)
            alpha_stars_arr = np.array(alpha_stars)
            alpha_cs_valid = [a for a in alpha_cs if not np.isnan(a)]
            rhos_valid = [r for r in rhos if not np.isnan(r)]

            summary_rows.append({
                "timestep": t_label,
                "m": m,
                "strategy": strategy,
                "alpha_star_mean": np.mean(alpha_stars_arr) if alpha_stars_arr.size else np.nan,
                "alpha_star_std": np.std(alpha_stars_arr) if alpha_stars_arr.size else np.nan,
                "alpha_c_mean": np.mean(alpha_cs_valid) if alpha_cs_valid else np.nan,
                "alpha_c_std": np.std(alpha_cs_valid) if alpha_cs_valid else np.nan,
                "rho_mean": np.mean(rhos_valid) if rhos_valid else np.nan,
                "rho_std": np.std(rhos_valid) if rhos_valid else np.nan,
                "bisection_converged_frac": (
                    np.mean(bisection_converged_list) if bisection_converged_list else np.nan
                ),
                "pcc_isolation_frac_at_low_alpha": (
                    np.mean(pcc_isolation_at_low) if pcc_isolation_at_low else np.nan
                ),
                "S_mean_at_alpha_star": (
                    np.mean(S_at_alpha_star_list) if S_at_alpha_star_list else np.nan
                ),
            })

            t_cfg = time.time() - t_config_start
            print(f"    {config_label} done in {t_cfg:.1f}s")

    # ---- Step 3: Save CSVs ----
    print("\n[3/3] Saving results...")

    # Sweep CSV
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestep", "m", "strategy", "instance_id",
            "alpha_over_alpha_star", "S", "T_rounds",
            "n_pcc_survived", "n_nonpcc_survived", "pcc_isolated", "alpha_star",
        ])
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"  Sweep CSV: {sweep_path} ({len(sweep_rows)} rows)")

    # Summary CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestep", "m", "strategy",
            "alpha_star_mean", "alpha_star_std",
            "alpha_c_mean", "alpha_c_std",
            "rho_mean", "rho_std",
            "bisection_converged_frac",
            "pcc_isolation_frac_at_low_alpha",
            "S_mean_at_alpha_star",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Summary CSV: {summary_path} ({len(summary_rows)} rows)")

    # ---- Final summary ----
    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  MULTI-TIMESTEP SUMMARY")
    print("=" * 60)
    for row in summary_rows:
        print(
            f"  {row['timestep']} m={row['m']}_{row['strategy']}: "
            f"alpha*={row['alpha_star_mean']:.4f}±{row['alpha_star_std']:.4f}, "
            f"alpha_c={row['alpha_c_mean']:.4f}±{row['alpha_c_std']:.4f}, "
            f"rho={row['rho_mean']:.4f}±{row['rho_std']:.4f}, "
            f"conv={row['bisection_converged_frac']:.2f}"
        )
    print(f"\n  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    run_multistep()
