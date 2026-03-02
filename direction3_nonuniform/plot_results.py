"""
Plot Experiment 2A and 2C results.
Run: python plot_results.py
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _load_csv_named(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if arr.size == 0:
        raise ValueError(f"CSV is empty: {path}")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def plot_2a(input_csv: str, output_png: str):
    data = _load_csv_named(input_csv)

    order = np.argsort(data["sigma_ratio"])
    x = data["sigma_ratio"][order]
    y_gen = data["kappa_c_mean_gen"][order]
    s_gen = data["kappa_c_std_gen"][order]
    y_con = data["kappa_c_mean_con"][order]
    s_con = data["kappa_c_std_con"][order]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    ax.plot(x, y_gen, color="#1f77b4", linewidth=2.0, label="2A-gen")
    ax.fill_between(x, y_gen - s_gen, y_gen + s_gen, color="#1f77b4", alpha=0.20)

    ax.plot(x, y_con, color="#ff7f0e", linewidth=2.0, label="2A-con")
    ax.fill_between(x, y_con - s_con, y_con + s_con, color="#ff7f0e", alpha=0.20)

    ax.set_xlabel("sigma / P_bar")
    ax.set_ylabel("kappa_c / P_max")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def plot_2c(input_csv: str, output_png: str):
    data = _load_csv_named(input_csv)

    order = np.argsort(data["r"])
    x = data["r"][order]
    y = data["kappa_c_mean"][order]
    s = data["kappa_c_std"][order]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    ax.plot(x, y, color="#d62728", linewidth=2.0, label="2C")
    ax.fill_between(x, y - s, y + s, color="#d62728", alpha=0.20)

    ax.set_xlabel("r")
    ax.set_ylabel("kappa_c / P_max")
    ax.grid(alpha=0.25)

    y_min = float(np.min(y - s))
    y_max = float(np.max(y + s))
    y_span = max(1e-6, y_max - y_min)
    y_annot = y_min + 0.06 * y_span

    ax.text(0.06, y_annot, "<- distributed", fontsize=9)
    ax.text(0.80, y_annot, "centralized ->", fontsize=9)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Experiment 2A and 2C results")
    parser.add_argument(
        "--input_2a",
        default=os.path.join(DEFAULT_RESULTS_DIR, "results_2A.csv"),
        type=str,
    )
    parser.add_argument(
        "--input_2c",
        default=os.path.join(DEFAULT_RESULTS_DIR, "results_2C.csv"),
        type=str,
    )
    parser.add_argument(
        "--output_2a",
        default=os.path.join(DEFAULT_RESULTS_DIR, "fig_2A.png"),
        type=str,
    )
    parser.add_argument(
        "--output_2c",
        default=os.path.join(DEFAULT_RESULTS_DIR, "fig_2C.png"),
        type=str,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_2a(args.input_2a, args.output_2a)
    plot_2c(args.input_2c, args.output_2c)

    print(f"Saved {args.output_2a}")
    print(f"Saved {args.output_2c}")


if __name__ == "__main__":
    main()
