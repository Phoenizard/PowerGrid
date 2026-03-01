"""
SQ3 Phase 1 Pilot — Cascade resilience vs edge addition (m).

Parameters:
  n_ensemble = 10, seed = 20260211
  m values = {0, 4, 8}
  kappa = 10.0 (normalized space)
  gamma = 1.0
  Time point = noon (T=43200s)
  Strategy = pcc_direct
  Alpha sweep: 20 points, alpha/alpha* in [0.1, 2.5]
  Alpha_c bisection: tol=1e-3
  Cascade model: DC (primary) + swing (3 selected alpha points per m)

Outputs:
  results/pilot/sq3_pilot_summary.csv
  results/pilot/sq3_pilot_sigmoid.csv
  figures/fig_sq3_pilot_sigmoid.png
  results/pilot/sq3_pilot_swing_validation.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    find_steady_state,
    compute_edge_flows,
    compute_dc_fmax,
    sweep_alpha,
    find_alpha_c,
    run_cascade_swing,
    cascade_pipeline_dc,
)
from kappa_pipeline import preload_ensemble, EnsembleInstance
from edge_strategies import select_edges_pcc_direct, compute_node_power_stats
from data_loader import evaluate_power_vector

# --- Config ---
N_ENSEMBLE = 10
SEED = 20260211
M_VALUES = [0, 4, 8]
KAPPA = 10.0
GAMMA = 1.0
T_NOON = 43200.0  # noon on day 0
N_ALPHA_POINTS = 20
ALPHA_RANGE = (0.1, 2.5)
BISECTION_TOL = 1e-3
SEASON = "summer"

# Swing validation alphas relative to alpha_c
SWING_ALPHA_FACTORS = [0.5, 1.0, 1.5]

# Output paths
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "pilot")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def add_edges_to_instance(
    inst: EnsembleInstance, m: int
) -> tuple:
    """Add m PCC-direct edges. Returns (A_modified, P_max_node, P_sign_node)."""
    if m == 0:
        return inst.A_base, None, None

    P_max_node, P_sign_node = compute_node_power_stats(
        inst.cons_interps, inst.pv_interps
    )
    rng_edge = np.random.default_rng(inst.net_seed + 1_000_000)
    A_lil = lil_matrix(inst.A_base)
    A_mod = select_edges_pcc_direct(A_lil, m, P_max_node, P_sign_node, rng_edge)
    return A_mod, P_max_node, P_sign_node


def run_pilot():
    """Run the full SQ3 pilot experiment."""
    print("=" * 60)
    print("  SQ3 PILOT EXPERIMENT")
    print(f"  n_ensemble={N_ENSEMBLE}, m={M_VALUES}, kappa={KAPPA}")
    print("=" * 60)

    t_start = time.time()

    # ---- Step 1: Preload ensemble ----
    print("\n[1/4] Pre-loading ensemble...")
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

    # ---- Step 2: DC cascade for each (instance, m) ----
    print("[2/4] Running DC cascade pipeline...")

    # Storage: per m → list of per-instance results
    all_results: dict[int, list[dict]] = {m: [] for m in M_VALUES}
    # For sigmoid: per m → (alphas, S_values_per_instance)
    sigmoid_data: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in M_VALUES}

    for m in M_VALUES:
        print(f"\n  --- m = {m} ---")
        t_m_start = time.time()

        for idx, (i, inst) in enumerate(valid_instances):
            t_inst = time.time()

            # Add edges
            A_mod, _, _ = add_edges_to_instance(inst, m)

            # Evaluate P at noon, normalize
            P_raw = evaluate_power_vector(inst.cons_interps, inst.pv_interps, T_NOON)
            P_norm = P_raw / inst.P_max

            # Power balance check
            balance = abs(np.sum(P_norm))
            if balance > 1e-8:
                print(f"    WARN inst {i}: power balance = {balance:.2e}")

            # Full DC pipeline
            result = cascade_pipeline_dc(
                A_mod, P_norm, KAPPA, GAMMA,
                n_alpha_points=N_ALPHA_POINTS,
                alpha_range=ALPHA_RANGE,
                bisection_tol=BISECTION_TOL,
            )

            all_results[m].append(result)
            sigmoid_data[m].append((result["alphas"], result["S_values"]))

            elapsed = time.time() - t_inst
            print(
                f"    [{idx+1}/{n_valid}] inst {i}: "
                f"conv={result['converged']}, "
                f"f_max_dc={result['f_max_dc']:.4f}, "
                f"f_max_swing={result['f_max_swing']:.4f}, "
                f"alpha_c={result['alpha_c']:.4f} ({elapsed:.2f}s)"
            )
            sys.stdout.flush()

        t_m = time.time() - t_m_start
        print(f"  m={m} done in {t_m:.1f}s")

    # ---- Step 3: Swing validation ----
    print("\n[3/4] Swing validation at selected alpha points...")

    swing_rows = []

    for m in M_VALUES:
        print(f"\n  --- m = {m} (swing) ---")
        for idx, (i, inst) in enumerate(valid_instances):
            A_mod, _, _ = add_edges_to_instance(inst, m)
            P_raw = evaluate_power_vector(inst.cons_interps, inst.pv_interps, T_NOON)
            P_norm = P_raw / inst.P_max

            dc_result = all_results[m][idx]
            alpha_c_dc = dc_result["alpha_c"]
            f_max_swing = dc_result["f_max_swing"]

            if f_max_swing < 1e-12 or np.isnan(alpha_c_dc):
                continue

            theta = dc_result["theta"]
            omega = dc_result["omega"]

            for factor in SWING_ALPHA_FACTORS:
                alpha_test = factor * alpha_c_dc
                alpha_test = max(alpha_test, 1e-6)

                t0 = time.time()
                sw_result = run_cascade_swing(
                    A_mod, P_norm, KAPPA, alpha_test, f_max_swing,
                    gamma=GAMMA, theta0=theta, omega0=omega,
                )
                t_sw = time.time() - t0

                swing_rows.append({
                    "m": m,
                    "instance": i,
                    "alpha_factor": factor,
                    "alpha": alpha_test,
                    "S_swing": sw_result.S,
                    "n_overload_sw": sw_result.n_overload,
                    "n_desync_sw": sw_result.n_desync,
                    "depth_sw": sw_result.cascade_depth,
                    "time_s": t_sw,
                })

            print(f"    inst {i}: swing validation done")
        sys.stdout.flush()

    # ---- Step 4: Save results and plot ----
    print("\n[4/4] Saving results...")

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "sq3_pilot_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n_valid", "alpha_c_mean", "alpha_c_std",
                          "f_max_dc_mean", "f_max_dc_std",
                          "f_max_swing_mean", "f_max_swing_std",
                          "converged_frac"])

        for m in M_VALUES:
            results_m = all_results[m]
            alpha_cs = [r["alpha_c"] for r in results_m if not np.isnan(r["alpha_c"])]
            f_maxs_dc = [r["f_max_dc"] for r in results_m]
            f_maxs_sw = [r["f_max_swing"] for r in results_m]
            n_conv = sum(1 for r in results_m if r["converged"])

            writer.writerow([
                m,
                len(results_m),
                f"{np.mean(alpha_cs):.6f}" if alpha_cs else "NaN",
                f"{np.std(alpha_cs):.6f}" if alpha_cs else "NaN",
                f"{np.mean(f_maxs_dc):.6f}",
                f"{np.std(f_maxs_dc):.6f}",
                f"{np.mean(f_maxs_sw):.6f}",
                f"{np.std(f_maxs_sw):.6f}",
                f"{n_conv/len(results_m):.2f}",
            ])
    print(f"  Summary: {summary_path}")

    # Sigmoid CSV (per-m: alpha, S_mean, S_std)
    sigmoid_path = os.path.join(RESULTS_DIR, "sq3_pilot_sigmoid.csv")
    with open(sigmoid_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "alpha_over_alpha_star", "S_mean", "S_std"])

        for m in M_VALUES:
            if not sigmoid_data[m]:
                continue
            # Stack S values across instances
            all_alphas = sigmoid_data[m][0][0]  # same alpha grid for all
            S_matrix = np.array([s for _, s in sigmoid_data[m]])
            S_mean = np.mean(S_matrix, axis=0)
            S_std = np.std(S_matrix, axis=0)

            for j in range(len(all_alphas)):
                writer.writerow([
                    m,
                    f"{all_alphas[j]:.6f}",
                    f"{S_mean[j]:.6f}",
                    f"{S_std[j]:.6f}",
                ])
    print(f"  Sigmoid: {sigmoid_path}")

    # Swing validation CSV
    swing_path = os.path.join(RESULTS_DIR, "sq3_pilot_swing_validation.csv")
    with open(swing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "instance", "alpha_factor", "alpha",
                          "S_swing", "n_overload_sw", "n_desync_sw",
                          "depth_sw", "time_s"])
        for row in swing_rows:
            writer.writerow([
                row["m"], row["instance"],
                f"{row['alpha_factor']:.2f}",
                f"{row['alpha']:.6f}",
                f"{row['S_swing']:.6f}",
                row["n_overload_sw"], row["n_desync_sw"],
                row["depth_sw"],
                f"{row['time_s']:.3f}",
            ])
    print(f"  Swing validation: {swing_path}")

    # ---- Plot: S vs alpha/alpha* ----
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}

    for m in M_VALUES:
        if not sigmoid_data[m]:
            continue
        all_alphas = sigmoid_data[m][0][0]
        S_matrix = np.array([s for _, s in sigmoid_data[m]])
        S_mean = np.mean(S_matrix, axis=0)
        S_std = np.std(S_matrix, axis=0)

        ax.plot(all_alphas, S_mean, '-o', color=colors[m], label=f"m={m}", markersize=4)
        ax.fill_between(all_alphas, S_mean - S_std, S_mean + S_std,
                         alpha=0.2, color=colors[m])

    ax.set_xlabel(r"Overload threshold $\alpha / \alpha^*$", fontsize=12)
    ax.set_ylabel(r"Surviving fraction $S$", fontsize=12)
    ax.set_title("SQ3 Pilot: Cascade resilience vs edge addition (DC model)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(FIGURES_DIR, "fig_sq3_pilot_sigmoid.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sigmoid plot: {fig_path}")

    # ---- Final summary ----
    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  PILOT SUMMARY")
    print("=" * 60)

    for m in M_VALUES:
        results_m = all_results[m]
        alpha_cs = [r["alpha_c"] for r in results_m if not np.isnan(r["alpha_c"])]
        f_maxs_dc = [r["f_max_dc"] for r in results_m]
        f_maxs_sw = [r["f_max_swing"] for r in results_m]
        n_conv = sum(1 for r in results_m if r["converged"])

        if alpha_cs:
            print(
                f"  m={m}: alpha_c = {np.mean(alpha_cs):.4f} +/- {np.std(alpha_cs):.4f}, "
                f"f_max_dc = {np.mean(f_maxs_dc):.4f}, "
                f"f_max_swing = {np.mean(f_maxs_sw):.4f}, "
                f"converged = {n_conv}/{len(results_m)}"
            )
        else:
            print(f"  m={m}: NO VALID alpha_c")

    print(f"\n  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    run_pilot()
