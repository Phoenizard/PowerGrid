"""
Experiment SQ5-A: Damping Sensitivity Analysis (gamma sweep).

Tests whether SQ4's topology optimization results (m=4, max_power strategy)
are robust under varying damping gamma in [0.1, 5.0].

Physical motivation: inverter-based renewables have lower effective damping
than synchronous generators; as grids transition to renewables, effective
gamma decreases.

Usage:
  python run_sq5a.py --n_ensemble 10          # sanity check
  python run_sq5a.py --n_ensemble 50          # production
  python run_sq5a.py --n_ensemble 10 --clean  # fresh start
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQ4_DIR = os.path.join(PROJECT_ROOT, "sq4_data")

for _d in (SQ4_DIR,):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from kappa_pipeline import (  # noqa: E402
    preload_ensemble,
    EnsembleInstance,
)
from edge_strategies import (  # noqa: E402
    STRATEGIES,
    compute_node_power_stats,
    verify_edge_addition,
)

SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")
for _d in (SQ2_DIR, PAPER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import evaluate_power_vector  # noqa: E402
from model import compute_kappa_c  # noqa: E402

# --- Experiment parameters ---
# Include gamma=1.0 explicitly for regression check
GAMMA_VALUES = np.sort(np.unique(np.append(np.linspace(0.1, 5.0, 15), 1.0)))
M_VALUES = [0, 4]
SEED = 20260209
STRATEGY = "max_power"

# Wednesday noon — consistent reference point
T_NOON = 3 * 86400 + 12 * 3600  # 302400 seconds


def make_config(gamma: float) -> dict:
    """
    Build bisection config for a given damping parameter.

    Integration time scales with 1/gamma (slower damping needs longer
    integration to detect convergence), capped at [20, 500].
    Upper kappa bound widened to 200.0 to avoid ceiling artifacts at low gamma.
    """
    return {
        "gamma": gamma,
        "kappa_range": (0.001, 200.0),
        "bisection_steps": 10,
        "t_integrate": min(500, max(20, int(100 / gamma))),
        "conv_tol": 5e-3,
        "max_step": 1.0,
    }


def ensure_output_csv(path: str):
    """Create output CSV with header if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gamma", "m", "kc_mean", "kc_std", "n_valid"])


def load_completed(path: str) -> set[tuple[str, str]]:
    """Load already-computed (gamma, m) pairs from checkpoint CSV."""
    if not os.path.exists(path):
        return set()
    completed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((row["gamma"], row["m"]))
            except (KeyError, TypeError):
                continue
    return completed


def append_row(path: str, row: list):
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def make_modifier(strategy_name: str, m: int):
    """Create a NetworkModifier for the given strategy and m (reused from SQ4)."""
    if m == 0:
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
        actual_m = (A_modified.nnz - A_original_nnz) // 2
        verify_edge_addition(A_base, A_modified, actual_m)
        return A_modified

    return modifier


def run_gamma_sweep(
    ensemble: list[EnsembleInstance | None],
    output_path: str,
    clean: bool = False,
):
    """
    Sweep gamma values at noon, for m=0 and m=4 (max_power).

    Checkpoint/resume: saves per-(gamma, m) rows as they complete.
    """
    if clean and os.path.exists(output_path):
        os.remove(output_path)

    ensure_output_csv(output_path)
    completed = load_completed(output_path)

    n_ensemble = len(ensemble)

    # Pre-compute modified adjacency matrices for m=4 (gamma-independent)
    print("Pre-computing modified networks (m=4, max_power)...")
    sys.stdout.flush()
    modifier = make_modifier(STRATEGY, 4)
    A_modified: list[csr_matrix | None] = []
    for inst in ensemble:
        if inst is None:
            A_modified.append(None)
        else:
            A_mod = modifier(inst.A_base.copy(), inst)
            A_modified.append(A_mod)
    print(f"  Done. {sum(1 for a in A_modified if a is not None)} networks modified.\n")

    # Total work items
    total_items = len(GAMMA_VALUES) * len(M_VALUES)
    n_skipped = 0
    n_done = 0
    t_start = time.time()

    for gamma in GAMMA_VALUES:
        config = make_config(gamma)
        gamma_str = f"{gamma:.4f}"

        for m in M_VALUES:
            m_str = str(m)

            if (gamma_str, m_str) in completed:
                n_skipped += 1
                n_done += 1
                continue

            t0 = time.time()
            kappa_values = []

            for i in range(n_ensemble):
                inst = ensemble[i]
                if inst is None:
                    continue
                if inst.P_max < 1e-12:
                    continue

                # Select adjacency matrix
                if m == 0:
                    A = inst.A_base
                else:
                    A = A_modified[i]
                    if A is None:
                        continue

                try:
                    P_t = evaluate_power_vector(
                        inst.cons_interps, inst.pv_interps, T_NOON
                    )
                    balance = abs(np.sum(P_t))
                    if balance > 1e-6:
                        print(
                            f"  WARN: power balance = {balance:.2e} "
                            f"gamma={gamma:.2f} m={m} inst={i}"
                        )

                    kc = compute_kappa_c(A, P_t, config_params=config)
                    kc_normalized = kc / inst.P_max
                    kappa_values.append(kc_normalized)

                except Exception as exc:
                    print(f"  WARN: inst {i+1} failed (gamma={gamma:.2f}, m={m}): {exc}")
                    continue

            if not kappa_values:
                print(f"  ERROR: No valid samples for gamma={gamma:.2f}, m={m}")
                sys.stdout.flush()
                n_done += 1
                continue

            kv = np.array(kappa_values)
            mean_kc = float(np.mean(kv))
            std_kc = float(np.std(kv))
            n_valid = len(kappa_values)
            elapsed = time.time() - t0
            n_done += 1

            append_row(output_path, [gamma_str, m_str, f"{mean_kc:.6f}",
                                      f"{std_kc:.6f}", n_valid])

            # ETA
            items_computed = n_done - n_skipped
            if items_computed > 0:
                avg_per_item = (time.time() - t_start) / items_computed
                remaining = total_items - n_done
                eta_s = avg_per_item * remaining
                eta_str = f"ETA {eta_s/60:.1f}min"
            else:
                eta_str = ""

            print(
                f"  [{n_done}/{total_items}] gamma={gamma:.2f} m={m} -- "
                f"kc={mean_kc:.4f} +/- {std_kc:.4f} "
                f"({n_valid} valid, {elapsed:.1f}s) {eta_str}"
            )
            sys.stdout.flush()

    total_time = time.time() - t_start
    print(f"\nGamma sweep complete in {total_time/60:.1f} min.")
    if n_skipped > 0:
        print(f"  ({n_skipped} items resumed from checkpoint)")
    sys.stdout.flush()


def regression_check(output_path: str) -> bool:
    """
    Check gamma=1.0 results against SQ4 baselines.

    SQ4 baselines (n=50, 7-noon avg):
      m=0:  kc(noon) = 6.4286 +/- 0.7228
      m=4:  kc(noon) = 3.2357 +/- 0.3894

    We allow 15% tolerance (single-noon vs 7-noon avg, n=10 vs n=50).
    """
    if not os.path.exists(output_path):
        print("REGRESSION: No output file found.")
        return False

    df = pd.read_csv(output_path)

    # Find gamma closest to 1.0
    gamma_1 = df.loc[(df["gamma"] - 1.0).abs().idxmin(), "gamma"]
    if abs(gamma_1 - 1.0) > 0.05:
        print(f"REGRESSION: No gamma ~1.0 found (closest = {gamma_1})")
        return False

    baselines = {0: 6.4286, 4: 3.2357}
    all_pass = True

    print("\n=== Regression Check (gamma=1.0 vs SQ4 baselines) ===")
    for m, baseline in baselines.items():
        row = df[(df["gamma"] == gamma_1) & (df["m"] == m)]
        if row.empty:
            print(f"  m={m}: MISSING")
            all_pass = False
            continue

        kc = row["kc_mean"].values[0]
        deviation = abs(kc - baseline) / baseline * 100
        status = "PASS" if deviation < 15 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  m={m}: kc={kc:.4f} (baseline={baseline:.4f}, dev={deviation:.1f}%) [{status}]")

    return all_pass


def plot_results(output_path: str, figures_dir: str):
    """Generate 3 figures per the plan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_path)

    # Pivot for easy access
    m0 = df[df["m"] == 0].sort_values("gamma").reset_index(drop=True)
    m4 = df[df["m"] == 4].sort_values("gamma").reset_index(drop=True)

    os.makedirs(figures_dir, exist_ok=True)

    # ---- Fig 5A-1: kc(noon) vs gamma ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m0["gamma"], m0["kc_mean"], "b-o", label="m = 0 (baseline)", markersize=4)
    ax.fill_between(
        m0["gamma"],
        m0["kc_mean"] - m0["kc_std"],
        m0["kc_mean"] + m0["kc_std"],
        alpha=0.2, color="blue",
    )
    ax.plot(m4["gamma"], m4["kc_mean"], "r--s", label="m = 4 (max_power)", markersize=4)
    ax.fill_between(
        m4["gamma"],
        m4["kc_mean"] - m4["kc_std"],
        m4["kc_mean"] + m4["kc_std"],
        alpha=0.2, color="red",
    )
    ax.set_xlabel(r"Damping parameter $\gamma$", fontsize=12)
    ax.set_ylabel(r"Critical coupling $\bar{\kappa}_c$ (noon)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_1 = os.path.join(figures_dir, "fig_5a1_kc_vs_gamma.png")
    fig.savefig(path_1, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path_1}")

    # ---- Fig 5A-2: Absolute reduction delta_kc vs gamma ----
    # Merge on gamma
    merged = pd.merge(m0, m4, on="gamma", suffixes=("_m0", "_m4"))
    delta_kc = merged["kc_mean_m0"] - merged["kc_mean_m4"]
    delta_err = np.sqrt(merged["kc_std_m0"]**2 + merged["kc_std_m4"]**2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        merged["gamma"], delta_kc, yerr=delta_err,
        fmt="g-^", capsize=3, markersize=5, label=r"$\Delta\bar{\kappa}_c$"
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Damping parameter $\gamma$", fontsize=12)
    ax.set_ylabel(r"$\Delta\bar{\kappa}_c = \bar{\kappa}_c(m{=}0) - \bar{\kappa}_c(m{=}4)$", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_2 = os.path.join(figures_dir, "fig_5a2_delta_kc_vs_gamma.png")
    fig.savefig(path_2, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path_2}")

    # ---- Fig 5A-3: Relative reduction (%) vs gamma ----
    rel_reduction = delta_kc / merged["kc_mean_m0"] * 100
    rel_err = delta_err / merged["kc_mean_m0"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        merged["gamma"], rel_reduction, yerr=rel_err,
        fmt="m-D", capsize=3, markersize=5,
        label=r"$\Delta\bar{\kappa}_c / \bar{\kappa}_c(m{=}0) \times 100\%$"
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Damping parameter $\gamma$", fontsize=12)
    ax.set_ylabel("Relative reduction (%)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_3 = os.path.join(figures_dir, "fig_5a3_relative_reduction.png")
    fig.savefig(path_3, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path_3}")


def parse_args():
    parser = argparse.ArgumentParser(description="SQ5-A: Damping sensitivity (gamma sweep)")
    parser.add_argument("--n_ensemble", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--season", type=str, default="summer")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing results and start fresh")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip computation, only generate plots")
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")
    figures_dir = os.path.join(SCRIPT_DIR, "figures")
    output_path = os.path.join(results_dir, "sq5a_gamma_sweep.csv")

    print("=" * 60)
    print("  SQ5-A: Damping Sensitivity Analysis (gamma sweep)")
    print("=" * 60)
    print(f"  n_ensemble: {args.n_ensemble}")
    print(f"  gamma range: [{GAMMA_VALUES[0]:.2f}, {GAMMA_VALUES[-1]:.2f}] ({len(GAMMA_VALUES)} values)")
    print(f"  m values: {M_VALUES}")
    print(f"  strategy: {STRATEGY}")
    print(f"  time point: Wednesday noon (t={T_NOON}s)")
    print(f"  seed: {args.seed}")
    print(f"  output: {output_path}")
    print()
    sys.stdout.flush()

    if args.plot_only:
        print("Plot-only mode.\n")
        plot_results(output_path, figures_dir)
        return

    # Pre-load ensemble ONCE
    print("Pre-loading ensemble...")
    sys.stdout.flush()
    t_load = time.time()
    ensemble = preload_ensemble(
        n_ensemble=args.n_ensemble,
        season=args.season,
        seed=args.seed,
        q=0.1,
    )
    print(f"Ensemble loaded in {(time.time() - t_load)/60:.1f} min.\n")

    # Run gamma sweep
    run_gamma_sweep(ensemble, output_path, clean=args.clean)

    # Regression check
    reg_ok = regression_check(output_path)
    if not reg_ok:
        print("\nWARNING: Regression check FAILED. Review results before proceeding.")

    # Plots
    print("\nGenerating plots...")
    plot_results(output_path, figures_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  SQ5-A COMPLETE")
    print("=" * 60)
    df = pd.read_csv(output_path)
    print(f"  Rows: {len(df)} (expected {len(GAMMA_VALUES) * len(M_VALUES)})")
    print(f"  NaN count: {df.isnull().sum().sum()}")
    print(f"  Regression: {'PASS' if reg_ok else 'FAIL'}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
