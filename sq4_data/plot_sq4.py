"""
SQ4 figure generation — 6 publication-quality figures.

Exp 4A (q-sweep):
  fig_4a1_kc_vs_q.png         — κ_c(noon) vs q with ±1 SD error bars
  fig_4a2_ratio_vs_q.png      — peak/valley ratio vs q

Exp 4B Step 1 (strategy comparison):
  fig_4b1_strategy_bars.png   — bar chart: baseline + 4 strategies, κ_c(noon) ± SD
  fig_4b2_strategy_timeseries.png — full 7-day κ_c(t) overlaid

Exp 4B Step 2 (m-sweep):
  fig_4b3_kc_vs_m.png         — κ_c(noon) vs m with ±1 SD
  fig_4b4_ratio_vs_m.png      — peak/valley ratio vs m
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")


def _setup_style():
    """Set up publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })


def plot_4a1_kc_vs_q(summary_path: str, out_dir: str):
    """Fig 4a1: κ_c(noon) vs rewiring probability q."""
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots()
    ax.errorbar(
        df["q"], df["kc_noon_mean"], yerr=df["kc_noon_std"],
        fmt="o-", capsize=4, color="tab:blue", label="Noon (12:00)",
    )
    ax.errorbar(
        df["q"], df["kc_dawn_mean"], yerr=df["kc_dawn_std"],
        fmt="s--", capsize=4, color="tab:orange", label="Dawn (06:00)",
    )
    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel(r"$\bar{\kappa}_c / P_{\max}$")
    ax.set_title("Critical coupling vs rewiring probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig_4a1_kc_vs_q.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4a2_ratio_vs_q(summary_path: str, out_dir: str):
    """Fig 4a2: peak/valley ratio vs q."""
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots()
    ax.plot(df["q"], df["peak_valley_ratio"], "o-", color="tab:red")
    ax.set_xlabel("Rewiring probability $q$")
    ax.set_ylabel("Peak / Valley ratio")
    ax.set_title("Diurnal vulnerability ratio vs rewiring probability")
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig_4a2_ratio_vs_q.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4b1_strategy_bars(summary_path: str, out_dir: str):
    """Fig 4b1: bar chart comparing strategies at κ_c(noon)."""
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots()
    x = np.arange(len(df))
    colors = ["gray", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    bars = ax.bar(x, df["kc_noon_mean"], yerr=df["kc_noon_std"],
                  capsize=4, color=colors[:len(df)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy"], rotation=30, ha="right")
    ax.set_ylabel(r"$\bar{\kappa}_c / P_{\max}$ (noon)")
    ax.set_title("Strategy comparison: critical coupling at noon (m=4)")
    ax.grid(True, alpha=0.3, axis="y")
    out = os.path.join(out_dir, "fig_4b1_strategy_bars.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4b2_strategy_timeseries(results_dir: str, m: int, out_dir: str):
    """Fig 4b2: full 7-day κ_c(t) overlaid for all strategies."""
    strategies = ["baseline", "random", "max_power", "score", "pcc_direct"]
    colors = ["gray", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    labels = ["Baseline", "Random", "Max power", "Score", "PCC direct"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for strat, color, label in zip(strategies, colors, labels):
        csv_path = os.path.join(results_dir, f"kappa_ts_{strat}_m{m}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        t_hours = df["day"] * 24 + df["hour"]
        ax.plot(t_hours, df["kappa_c_mean"], "o-", color=color, label=label,
                markersize=4, alpha=0.8)
        ax.fill_between(
            t_hours,
            df["kappa_c_mean"] - df["kappa_c_std"],
            df["kappa_c_mean"] + df["kappa_c_std"],
            color=color, alpha=0.1,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$\bar{\kappa}_c / P_{\max}$")
    ax.set_title(f"7-day critical coupling time series (m={m} added edges)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Day markers
    for d in range(1, 7):
        ax.axvline(d * 24, color="gray", linestyle=":", linewidth=0.5)

    out = os.path.join(out_dir, "fig_4b2_strategy_timeseries.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4b3_kc_vs_m(summary_path: str, out_dir: str):
    """Fig 4b3: κ_c(noon) vs number of added edges m."""
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots()
    ax.errorbar(
        df["m"], df["kc_noon_mean"], yerr=df["kc_noon_std"],
        fmt="o-", capsize=4, color="tab:blue", label="Noon (12:00)",
    )
    ax.errorbar(
        df["m"], df["kc_dawn_mean"], yerr=df["kc_dawn_std"],
        fmt="s--", capsize=4, color="tab:orange", label="Dawn (06:00)",
    )
    ax.set_xlabel("Number of added edges $m$")
    ax.set_ylabel(r"$\bar{\kappa}_c / P_{\max}$")
    strategy = df["strategy"].iloc[0] if "strategy" in df.columns else "best"
    ax.set_title(f"Critical coupling vs added edges ({strategy})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig_4b3_kc_vs_m.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4b4_ratio_vs_m(summary_path: str, out_dir: str):
    """Fig 4b4: peak/valley ratio vs m."""
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots()
    ax.plot(df["m"], df["peak_valley_ratio"], "o-", color="tab:red")
    ax.set_xlabel("Number of added edges $m$")
    ax.set_ylabel("Peak / Valley ratio")
    strategy = df["strategy"].iloc[0] if "strategy" in df.columns else "best"
    ax.set_title(f"Diurnal vulnerability ratio vs added edges ({strategy})")
    ax.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "fig_4b4_ratio_vs_m.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate SQ4 figures")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "4a", "4b1", "4b2"],
                        help="Which experiment figures to generate")
    parser.add_argument("--m", type=int, default=4,
                        help="m value for strategy comparison plots")
    args = parser.parse_args()

    _setup_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.exp in ("all", "4a"):
        summary_4a = os.path.join(RESULTS_DIR, "exp4A", "sq4a_q_sweep.csv")
        if os.path.exists(summary_4a):
            print("Generating Exp 4A figures...")
            plot_4a1_kc_vs_q(summary_4a, FIGURES_DIR)
            plot_4a2_ratio_vs_q(summary_4a, FIGURES_DIR)
        else:
            print(f"  WARN: {summary_4a} not found, skipping 4A figures")

    if args.exp in ("all", "4b1"):
        summary_4b1 = os.path.join(RESULTS_DIR, "exp4B_s1", "sq4b_step1_summary.csv")
        results_4b1 = os.path.join(RESULTS_DIR, "exp4B_s1")
        if os.path.exists(summary_4b1):
            print("Generating Exp 4B-S1 figures...")
            plot_4b1_strategy_bars(summary_4b1, FIGURES_DIR)
            plot_4b2_strategy_timeseries(results_4b1, args.m, FIGURES_DIR)
        else:
            print(f"  WARN: {summary_4b1} not found, skipping 4B-S1 figures")

    if args.exp in ("all", "4b2"):
        summary_4b2 = os.path.join(RESULTS_DIR, "exp4B_s2", "sq4b_step2_m_sweep.csv")
        if os.path.exists(summary_4b2):
            print("Generating Exp 4B-S2 figures...")
            plot_4b3_kc_vs_m(summary_4b2, FIGURES_DIR)
            plot_4b4_ratio_vs_m(summary_4b2, FIGURES_DIR)
        else:
            print(f"  WARN: {summary_4b2} not found, skipping 4B-S2 figures")

    print("\nDone.")


if __name__ == "__main__":
    main()
