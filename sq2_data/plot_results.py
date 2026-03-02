"""
Visualization for SQ2 experiments.

Fig 3A: Ternary simplex trajectory (using python-ternary).
Fig 3B: kappa_c time series over one week.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 300

STRATEGY_COLORS = {
    "baseline":   "#888888",
    "random":     "#1f77b4",   # tab blue
    "max_power":  "#ff7f0e",   # tab orange
    "score":      "#d62728",   # tab red
    "pcc_direct": "#2ca02c",   # tab green
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SQ2 results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing result CSVs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures",
    )
    parser.add_argument(
        "--season", type=str, default="summer",
    )
    return parser.parse_args()


def plot_trajectory(results_dir: str, output_dir: str, season: str):
    """
    Fig 3A: Ternary simplex trajectory.

    Mean trajectory as a time-colored line on the simplex.
    Annotate midnight (00:00) and midday (12:00) positions.
    """
    csv_path = os.path.join(results_dir, f"trajectory_{season}.csv")
    if not os.path.exists(csv_path):
        print(f"Trajectory CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    eta_plus = df["eta_plus_mean"].values
    eta_minus = df["eta_minus_mean"].values
    eta_p = df["eta_p_mean"].values
    hours = df["hour"].values

    # Scale to ternary coordinates (python-ternary uses scale parameter)
    scale = 1.0
    points = list(zip(eta_minus, eta_p, eta_plus))

    fig, tax = ternary.figure(scale=scale)
    fig.set_size_inches(7, 6)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=0.1, color="gray", linewidth=0.3, alpha=0.5)

    # Color by time
    n_points = len(points)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Plot trajectory segments with color gradient
    for i in range(n_points - 1):
        tax.line(
            points[i], points[i + 1],
            color=colors[i], linewidth=1.2, alpha=0.8,
        )

    # Scatter all points with time coloring
    for i, pt in enumerate(points):
        tax.scatter([pt], marker="o", s=3, color=[colors[i]], zorder=5)

    # Mark midnight (blue square) and midday (red diamond) positions
    # Only add text labels for D1 to avoid overlap
    for t_idx, h in enumerate(hours):
        if h % 24.0 == 0.0:  # midnight
            day = int(h // 24)
            tax.scatter(
                [points[t_idx]], marker="s", s=25, color="blue", zorder=10,
            )
            if day == 0:  # Only label D1 midnight
                tax.annotate(
                    "Night 00:00", points[t_idx],
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points",
                )
        elif h % 24.0 == 12.0:  # midday
            day = int(h // 24)
            tax.scatter(
                [points[t_idx]], marker="D", s=25, color="red", zorder=10,
            )
            if day == 0:  # Only label D1 midday
                tax.annotate(
                    "Day 12:00", points[t_idx],
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points",
                )

    # Labels
    tax.left_axis_label("Generators ($\\eta^+$)", fontsize=11, offset=0.16)
    tax.right_axis_label("Consumers ($\\eta^-$)", fontsize=11, offset=0.16)
    tax.bottom_axis_label("Passive ($\\eta_p$)", fontsize=11, offset=0.06)

    tax.ticks(axis="lbr", linewidth=0.5, multiple=0.2, fontsize=8, offset=0.02,
              tick_formats="%.1f")
    tax.clear_matplotlib_ticks()
    tax.get_axes().set_aspect("equal")

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=0, vmax=hours[-1]),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), fraction=0.03, pad=0.08)
    cbar.set_label("Hours from Monday 00:00", fontsize=10)

    fig_path = os.path.join(output_dir, f"fig3A_trajectory_{season}.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_kappa_timeseries(results_dir: str, output_dir: str, season: str):
    """
    Fig 3B: kappa_c / P_max time series over one week.

    Mean line + shaded +-1 SD band.
    """
    csv_path = os.path.join(results_dir, f"kappa_c_timeseries_{season}.csv")
    if not os.path.exists(csv_path):
        print(f"Kappa timeseries CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    days = df["day"].values.astype(float)
    hours_in_day = df["hour"].values.astype(float)
    kc_mean = df["kappa_c_mean"].values.astype(float)
    kc_std = df["kappa_c_std"].values.astype(float)

    # x-axis: continuous time in days
    x = days + hours_in_day / 24.0

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.plot(x, kc_mean, "-", color="#1f77b4", linewidth=1.5, label="$\\bar{\\kappa}_c / P_{\\max}$")
    ax.fill_between(
        x,
        np.maximum(kc_mean - kc_std, 0),
        kc_mean + kc_std,
        alpha=0.25,
        color="#1f77b4",
        label="$\\pm 1$ SD",
    )

    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("$\\bar{\\kappa}_c / P_{\\max}$", fontsize=12)

    # x ticks at day boundaries
    ax.set_xticks(range(N_DAYS := 7))
    ax.set_xticklabels([f"Day {d+1}" for d in range(N_DAYS)], fontsize=10)
    ax.set_xlim(-0.1, 6.95)

    # Add vertical dashed lines at day boundaries
    for d in range(1, N_DAYS):
        ax.axvline(d, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f"fig3B_kappa_timeseries_{season}.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    args = parse_args()

    dir_a = os.path.join(SCRIPT_DIR, "results_sq2_A")
    dir_b = os.path.join(SCRIPT_DIR, "results_sq2_B")

    if args.results_dir is not None:
        # Explicit dir: use it for both plots
        dir_a = dir_b = args.results_dir

    if args.output_dir is None:
        args.output_dir = os.path.join(dir_b, "figures")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Results dir A (trajectory): {dir_a}")
    print(f"Results dir B (kappa):      {dir_b}")
    print(f"Output dir: {args.output_dir}")

    plot_trajectory(dir_a, args.output_dir, args.season)
    plot_kappa_timeseries(dir_b, args.output_dir, args.season)

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
