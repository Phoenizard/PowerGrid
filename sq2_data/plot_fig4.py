"""
Generate Fig.4E-style simplex trajectory plot.

Replicates GridResilience's TrajectoryPlotter style:
  - python-ternary with scale = n - 2 = 48
  - Coordinate remap: sigma*n -> (nd-1, ne, ns-1)
  - Individual ensemble: thin semi-transparent lines (alpha=0.02)
  - Mean trajectory: thick bold line (alpha=0.8)

Uses GR precomputed pkl data (fast) + our CSV mean trajectory.
"""

from __future__ import annotations

import os
import sys
import types
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ternary

# --- Paths ---
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GRIDRES_ROOT = PROJECT_ROOT / "GridResilience"

RESULTS_DIR = SCRIPT_DIR / "results_sq2"
OUTPUT_DIR = RESULTS_DIR / "figures"
GR_FULLPEN_DIR = GRIDRES_ROOT / "trajdata" / "7" / "fullpen_nobat"

OUR_CSV = RESULTS_DIR / "trajectory_summer_fullpen.csv"

# --- Style ---
N = 50
SCALE = N - 2  # 48
MARTA_RED = "#c24c51"
MARTA_GRAY = "#404040"


def _setup_dill():
    """Stub powerreader so dill can unpickle GR Trajectory objects."""
    stub = types.ModuleType("powerreader")
    def _css(Pvec):
        ls = np.max(Pvec); n = len(Pvec); ld = np.abs(np.min(Pvec))
        s = [x for x in Pvec if x > 0.0]; d = [x for x in Pvec if x < 0.0]
        return np.sum(s)/(n*ls), np.sum(np.abs(d))/(n*ld), 0.0
    stub.continuoussourcesinkcounter = _css
    sys.modules["powerreader"] = stub
    sys.path.insert(0, str(GRIDRES_ROOT / "scripts"))
    import dill
    return dill


def sigma_to_ternary(sigmas: np.ndarray) -> list[tuple]:
    """sigma (T,3) -> GR ternary coords: (nd-1, ne, ns-1)."""
    ns = sigmas[:, 0] * N
    nd = sigmas[:, 1] * N
    ne = sigmas[:, 2] * N
    return list(zip(nd - 1, ne, ns - 1))


def load_gr_trajectories(pkl_dir, dill_mod, n_max=50):
    """Load GR pkl files -> list of (T,3) sigma arrays."""
    trajs = []
    for i in range(n_max):
        p = pkl_dir / f"{i}.pkl"
        if not p.exists():
            break
        with open(p, "rb") as f:
            try:
                t = dill_mod.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                t = dill_mod.load(f, encoding="latin1")
        trajs.append(np.array(t.sigmas))
    return trajs


def plot_fig4e(ax, trajectories, title, mean_override=None):
    """Plot one Fig.4E-style simplex panel.

    Parameters
    ----------
    ax : matplotlib Axes
    trajectories : list of (T,3) sigma arrays (individual ensemble members)
    title : str
    mean_override : optional (T,3) array to use as mean instead of computing
    """
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=SCALE)
    tax.boundary(linewidth=0.5)
    tax.gridlines(linewidth=0.3, multiple=6, color="k", alpha=0.4)

    # Individual trajectories
    for i, traj in enumerate(trajectories):
        pts = sigma_to_ternary(traj)
        color = MARTA_GRAY if i == 0 else MARTA_RED
        tax.plot(pts, linewidth=0.9, color=color, alpha=0.02)

    # Mean trajectory
    if mean_override is not None:
        mean_sigmas = mean_override
    else:
        mean_sigmas = np.array(trajectories).mean(axis=0)
    mean_pts = sigma_to_ternary(mean_sigmas)
    tax.plot(mean_pts, linewidth=2.8, color=MARTA_RED, alpha=0.8)

    tax.set_title(title, fontsize=11, pad=12)
    tax.get_axes().axis("off")
    tax.clear_matplotlib_ticks()

    # Side labels matching paper Fig.4:
    #   GR remap (nd-1, ne, ns-1):
    #     a=nd-1 -> right vertex (n⁻, demand)
    #     b=ne   -> top vertex   (n_p, passive)
    #     c=ns-1 -> left vertex  (n⁺, sources)
    #   Left side = n⁺, Right side = n_p, Bottom side = n⁻
    tax.left_axis_label("$n^+$", fontsize=12, offset=0.16)
    tax.right_axis_label("$n_p$", fontsize=12, offset=0.16)
    tax.bottom_axis_label("$n^-$", fontsize=12, offset=0.06)

    return tax


def plot_mean_only(ax, mean_sigmas, title):
    """Plot a single mean trajectory on simplex (no ensemble background)."""
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=SCALE)
    tax.boundary(linewidth=0.5)
    tax.gridlines(linewidth=0.3, multiple=6, color="k", alpha=0.4)

    pts = sigma_to_ternary(mean_sigmas)
    tax.plot(pts, linewidth=2.8, color=MARTA_RED, alpha=0.8)

    tax.set_title(title, fontsize=11, pad=12)
    tax.get_axes().axis("off")
    tax.clear_matplotlib_ticks()

    tax.left_axis_label("$n^+$", fontsize=12, offset=0.16)
    tax.right_axis_label("$n_p$", fontsize=12, offset=0.16)
    tax.bottom_axis_label("$n^-$", fontsize=12, offset=0.06)
    return tax


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load our mean trajectory from CSV ---
    print("Loading our mean trajectory from CSV...")
    data = np.loadtxt(OUR_CSV, delimiter=",", skiprows=1)
    our_mean = np.column_stack([data[:, 2], data[:, 4], data[:, 6]])
    print(f"  Shape: {our_mean.shape}")

    # === Our Fig.4E: mean trajectory only ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_mean_only(ax, our_mean,
                   "Summer: 100% PV penetration, no batteries")
    fig.tight_layout()
    out = OUTPUT_DIR / "fig4E_simplex.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
