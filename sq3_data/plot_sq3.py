"""
SQ3 Plotting — Generate all figures from CSV results.

Figures:
  SQ3-1: S vs alpha/alpha* multi-panel (5 rows=timesteps, 4 lines=m configs)
  SQ3-2: Edge survival by type (PCC vs non-PCC) grouped bar at alpha/alpha*=0.5
  SQ3-3: Option D sigmoid vs PCC staircase + rho histogram
  SQ3-4: Cascade depth T vs alpha/alpha* at noon
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colors for m configs
COLORS = {
    "m=0_pcc_direct": "#1f77b4",
    "m=4_pcc_direct": "#ff7f0e",
    "m=8_pcc_direct": "#2ca02c",
    "m=4_random": "#d62728",
}
LABELS = {
    "m=0_pcc_direct": "m=0 (baseline)",
    "m=4_pcc_direct": "m=4 (PCC direct)",
    "m=8_pcc_direct": "m=8 (PCC direct)",
    "m=4_random": "m=4 (random)",
}

TIMESTEP_ORDER = ["00:00", "06:00", "09:00", "12:00", "18:00"]


def plot_sq3_1():
    """SQ3-1: S vs alpha/alpha* multi-panel, 5 rows (timesteps), 4 lines (m configs)."""
    sweep_path = os.path.join(RESULTS_DIR, "multistep", "sq3_multistep_sweep.csv")
    if not os.path.exists(sweep_path):
        print(f"  SKIP SQ3-1: {sweep_path} not found")
        return

    df = pd.read_csv(sweep_path)

    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True, sharey=True)

    for row_idx, t_label in enumerate(TIMESTEP_ORDER):
        ax = axes[row_idx]
        df_t = df[df["timestep"] == t_label]

        for (m, strategy), group in df_t.groupby(["m", "strategy"]):
            config_key = f"m={m}_{strategy}"
            color = COLORS.get(config_key, "gray")
            label = LABELS.get(config_key, config_key)

            # Compute mean ± std across instances
            agg = group.groupby("alpha_over_alpha_star")["S"].agg(["mean", "std"]).reset_index()

            ax.plot(agg["alpha_over_alpha_star"], agg["mean"],
                    '-o', color=color, label=label, markersize=2, linewidth=1.5)
            ax.fill_between(agg["alpha_over_alpha_star"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            alpha=0.15, color=color)

        ax.set_ylabel(r"$S$", fontsize=11)
        ax.set_title(f"t = {t_label}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        if row_idx == 0:
            ax.legend(fontsize=9, loc='upper left')

    axes[-1].set_xlabel(r"$\alpha / \alpha^*$", fontsize=12)
    fig.suptitle("SQ3-1: Cascade Resilience Across Times of Day", fontsize=14, y=0.995)
    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "fig_sq3_1_sigmoid_multipanel.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  SQ3-1: {fig_path}")


def plot_sq3_2():
    """SQ3-2: Edge survival by type (PCC vs non-PCC) grouped bar at alpha/alpha*≈0.5."""
    sweep_path = os.path.join(RESULTS_DIR, "multistep", "sq3_multistep_sweep.csv")
    if not os.path.exists(sweep_path):
        print(f"  SKIP SQ3-2: {sweep_path} not found")
        return

    df = pd.read_csv(sweep_path)

    # Find alpha/alpha* closest to 0.5
    alpha_vals = df["alpha_over_alpha_star"].unique()
    target_alpha = alpha_vals[np.argmin(np.abs(alpha_vals - 0.5))]
    df_at = df[df["alpha_over_alpha_star"] == target_alpha]

    fig, axes = plt.subplots(1, len(TIMESTEP_ORDER), figsize=(18, 5), sharey=True)

    for col_idx, t_label in enumerate(TIMESTEP_ORDER):
        ax = axes[col_idx]
        df_t = df_at[df_at["timestep"] == t_label]

        configs = []
        pcc_means = []
        nonpcc_means = []

        for (m, strategy), group in df_t.groupby(["m", "strategy"]):
            config_key = f"m={m}_{strategy}"
            configs.append(LABELS.get(config_key, config_key))
            pcc_means.append(group["n_pcc_survived"].mean())
            nonpcc_means.append(group["n_nonpcc_survived"].mean())

        x = np.arange(len(configs))
        width = 0.35

        ax.bar(x - width/2, pcc_means, width, label="PCC edges", color="#e74c3c", alpha=0.8)
        ax.bar(x + width/2, nonpcc_means, width, label="Non-PCC edges", color="#3498db", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax.set_title(f"t = {t_label}", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Surviving edges (mean)", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle(f"SQ3-2: Edge Survival by Type at α/α*≈{target_alpha:.2f}", fontsize=13, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "fig_sq3_2_edge_survival.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  SQ3-2: {fig_path}")


def plot_sq3_3():
    """SQ3-3: Option D non-PCC sigmoid vs PCC staircase + rho histogram."""
    optd_sweep = os.path.join(RESULTS_DIR, "option_d", "sq3_option_d_sweep.csv")
    optd_rho = os.path.join(RESULTS_DIR, "option_d", "sq3_option_d_rho_distribution.csv")
    pcc_sweep = os.path.join(RESULTS_DIR, "multistep", "sq3_multistep_sweep.csv")

    has_optd = os.path.exists(optd_sweep) and os.path.exists(optd_rho)
    has_pcc = os.path.exists(pcc_sweep)

    if not has_optd:
        print(f"  SKIP SQ3-3: Option D files not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Option D sigmoid (mean ± std across 100 instances)
    ax = axes[0]
    df_optd = pd.read_csv(optd_sweep)
    agg = df_optd.groupby("alpha_over_alpha_star")["S"].agg(["mean", "std"]).reset_index()
    ax.plot(agg["alpha_over_alpha_star"], agg["mean"], '-', color="#2ecc71", linewidth=2,
            label="WS(50,4,0.1) — no PCC")
    ax.fill_between(agg["alpha_over_alpha_star"],
                     agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                     alpha=0.2, color="#2ecc71")

    # Overlay PCC m=0 at noon if available
    if has_pcc:
        df_pcc = pd.read_csv(pcc_sweep)
        df_noon = df_pcc[(df_pcc["timestep"] == "12:00") & (df_pcc["m"] == 0)]
        if not df_noon.empty:
            agg_pcc = df_noon.groupby("alpha_over_alpha_star")["S"].agg(["mean", "std"]).reset_index()
            ax.plot(agg_pcc["alpha_over_alpha_star"], agg_pcc["mean"], '-', color="#e74c3c",
                    linewidth=2, label="PCC network m=0 (noon)")
            ax.fill_between(agg_pcc["alpha_over_alpha_star"],
                             agg_pcc["mean"] - agg_pcc["std"],
                             agg_pcc["mean"] + agg_pcc["std"],
                             alpha=0.2, color="#e74c3c")

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"$\alpha / \alpha^*$", fontsize=11)
    ax.set_ylabel(r"$S$", fontsize=11)
    ax.set_title("(a) Sigmoid Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: rho histogram
    ax = axes[1]
    df_rho = pd.read_csv(optd_rho)
    rhos = df_rho["rho"].dropna().values

    if len(rhos) > 0:
        ax.hist(rhos, bins=20, color="#2ecc71", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(rhos), color='red', linestyle='--',
                   label=f"mean={np.mean(rhos):.3f}")
        ax.axvline(np.median(rhos), color='blue', linestyle='--',
                   label=f"median={np.median(rhos):.3f}")
    ax.set_xlabel(r"$\rho = \alpha_c / \alpha^*$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(r"(b) $\rho$ Distribution (Option D)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: convergence summary
    ax = axes[2]
    n_conv = df_rho["bisection_converged"].sum()
    n_total = len(df_rho)
    ax.bar(["Converged", "Not converged"],
           [n_conv, n_total - n_conv],
           color=["#2ecc71", "#e74c3c"], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"(c) Bisection Convergence ({100*n_conv/n_total:.0f}%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("SQ3-3: Option D — Non-PCC Network Validation", fontsize=14, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "fig_sq3_3_option_d.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  SQ3-3: {fig_path}")


def plot_sq3_4():
    """SQ3-4: Cascade depth T vs alpha/alpha* at noon."""
    sweep_path = os.path.join(RESULTS_DIR, "multistep", "sq3_multistep_sweep.csv")
    if not os.path.exists(sweep_path):
        print(f"  SKIP SQ3-4: {sweep_path} not found")
        return

    df = pd.read_csv(sweep_path)
    df_noon = df[df["timestep"] == "12:00"]

    if df_noon.empty:
        print("  SKIP SQ3-4: no noon data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for (m, strategy), group in df_noon.groupby(["m", "strategy"]):
        config_key = f"m={m}_{strategy}"
        color = COLORS.get(config_key, "gray")
        label = LABELS.get(config_key, config_key)

        agg = group.groupby("alpha_over_alpha_star")["T_rounds"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["alpha_over_alpha_star"], agg["mean"],
                '-o', color=color, label=label, markersize=3, linewidth=1.5)
        ax.fill_between(agg["alpha_over_alpha_star"],
                         agg["mean"] - agg["std"],
                         agg["mean"] + agg["std"],
                         alpha=0.15, color=color)

    ax.set_xlabel(r"$\alpha / \alpha^*$", fontsize=12)
    ax.set_ylabel(r"Cascade depth $T$", fontsize=12)
    ax.set_title("SQ3-4: Cascade Depth at Noon (t=12:00)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(FIGURES_DIR, "fig_sq3_4_cascade_depth.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  SQ3-4: {fig_path}")


def main():
    print("=" * 60)
    print("  SQ3 PLOT GENERATION")
    print("=" * 60)

    plot_sq3_1()
    plot_sq3_2()
    plot_sq3_3()
    plot_sq3_4()

    print("\n  All plots generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
