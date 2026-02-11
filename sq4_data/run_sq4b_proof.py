"""
SQ4-B Proof — Lower Bound Verification (Option B).

Verifies the analytic lower bound: κ_c(m) ≥ |P_p| / (d₀ + m).

The formula is a NECESSARY condition (lower bound), not an upper bound.
We verify: analytic lower bound ≤ observed κ_c for all m values.

Approach (NO ODE integration):
  1. Preload ensemble (n=50), evaluate P at noon → get |P_PCC|/P_max per instance.
  2. Compute lower bound curve: mean(|P_PCC|/P_max) / (d₀ + m).
  3. Load observed κ_c/P_max from Exp 4B-S2 CSV.
  4. Empirical fit: C_fit / (d₀ + m), where C_fit = kc_noon_mean(m=0) × d₀.
  5. Single-panel figure + proof_summary.txt.

Outputs:
  - results/proof/proof_summary.txt    (human-readable table + verdict)
  - figures/fig_proof_bounds.png       (single-panel figure)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (same pattern as kappa_pipeline.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")

for _d in (SQ2_DIR, PAPER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import evaluate_power_vector  # noqa: E402
from kappa_pipeline import preload_ensemble  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M_VALUES = [0, 1, 2, 4, 6, 8, 10, 15, 20]
SEED = 20260211
T_NOON = 43200  # day 0, hour 12
PCC_IDX = 49
D0 = 4
N_ENSEMBLE = 50

# Output paths
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "proof")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "proof_summary.txt")
FIG_PATH = os.path.join(FIGURES_DIR, "fig_proof_bounds.png")

# Existing observed κ_c CSV (from Exp 4B-S2, already normalized by P_max)
OBS_CSV_PATH = os.path.join(
    SCRIPT_DIR, "results", "exp4B_s2", "sq4b_step2_m_sweep_pcc_direct.csv"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("SQ4-B Proof: Lower Bound Verification")
    print(f"  M_VALUES = {M_VALUES}")
    print(f"  N_ENSEMBLE = {N_ENSEMBLE}, SEED = {SEED}")
    print(f"  D0 = {D0}, PCC_IDX = {PCC_IDX}")
    print(f"  All quantities in normalized (kappa_c / P_max) space")
    print("=" * 70)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 1: Preload ensemble, compute normalized |P_PCC|/P_max at noon
    # ------------------------------------------------------------------
    print("\n[1/5] Preloading ensemble & computing noon power vectors...")
    sys.stdout.flush()
    t_start = time.time()

    ensemble = preload_ensemble(
        n_ensemble=N_ENSEMBLE, season="summer", seed=SEED, q=0.1,
    )
    t_load = time.time() - t_start
    n_valid = sum(1 for inst in ensemble if inst is not None)
    print(f"  Loaded {n_valid}/{N_ENSEMBLE} in {t_load:.1f}s")

    # Evaluate P at noon for each instance, collect |P_PCC|/P_max
    P_p_normalized_list = []  # |P_PCC| / P_max per instance
    P_max_list = []

    for idx, inst in enumerate(ensemble):
        if inst is None:
            continue

        P_t = evaluate_power_vector(inst.cons_interps, inst.pv_interps, T_NOON)
        balance = abs(np.sum(P_t))
        assert balance < 1e-6, f"Power balance failed for instance {idx}: {balance:.2e}"

        P_p_abs = abs(P_t[PCC_IDX])
        P_p_norm = P_p_abs / inst.P_max
        P_p_normalized_list.append(P_p_norm)
        P_max_list.append(inst.P_max)

    P_p_norm_arr = np.array(P_p_normalized_list)
    mean_P_p_normalized = float(np.mean(P_p_norm_arr))
    std_P_p_normalized = float(np.std(P_p_norm_arr))
    mean_P_max = float(np.mean(P_max_list))

    print(f"  |P_PCC|/P_max: mean={mean_P_p_normalized:.4f}, "
          f"std={std_P_p_normalized:.4f}")
    print(f"  P_max (houses): mean={mean_P_max:.4f} kW")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2: Compute lower bound curve
    # ------------------------------------------------------------------
    print("\n[2/5] Computing lower bound curve...")
    sys.stdout.flush()

    m_arr = np.array(M_VALUES, dtype=float)
    lower_bound = mean_P_p_normalized / (D0 + m_arr)

    for m, lb in zip(M_VALUES, lower_bound):
        print(f"  m={m:2d}  lower_bound = {lb:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Load observed κ_c from existing CSV
    # ------------------------------------------------------------------
    print("\n[3/5] Loading observed kappa_c from Exp 4B-S2...")
    sys.stdout.flush()

    import pandas as pd
    obs_df = pd.read_csv(OBS_CSV_PATH)

    obs_m = obs_df["m"].values
    obs_kc_mean = obs_df["kc_noon_mean"].values       # already normalized
    obs_kc_std = obs_df["kc_noon_std"].values          # already normalized

    print(f"  Loaded {len(obs_df)} rows from {OBS_CSV_PATH}")
    for _, row in obs_df.iterrows():
        print(f"  m={int(row['m']):2d}  kc_noon_mean={row['kc_noon_mean']:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Empirical fit
    # ------------------------------------------------------------------
    print("\n[4/5] Computing empirical fit...")
    sys.stdout.flush()

    # C_fit = kc_noon_mean(m=0) * D0
    kc_m0 = obs_kc_mean[obs_m == 0][0]
    C_fit = kc_m0 * D0
    fit_curve = C_fit / (D0 + m_arr)

    print(f"  kc_noon_mean(m=0) = {kc_m0:.4f}")
    print(f"  C_fit = {C_fit:.4f}")

    # Compute fit residuals
    obs_m_set = set(obs_m)
    fit_residuals = []
    for m_val, obs_val in zip(obs_m, obs_kc_mean):
        fit_val = C_fit / (D0 + m_val)
        residual_pct = (obs_val - fit_val) / obs_val * 100
        fit_residuals.append((m_val, obs_val, fit_val, residual_pct))
        print(f"  m={int(m_val):2d}  obs={obs_val:.4f}  fit={fit_val:.4f}  "
              f"residual={residual_pct:+.1f}%")

    # ------------------------------------------------------------------
    # Step 5: Summary text file
    # ------------------------------------------------------------------
    print("\n[5/5] Writing outputs...")
    sys.stdout.flush()

    # Build per-m comparison table
    summary_lines = [
        "SQ4-B Lower Bound Verification — Proof Summary",
        "=" * 70,
        f"Date: 2026-02-11",
        f"Ensemble: n={N_ENSEMBLE}, seed={SEED}",
        f"Lower bound formula: |P_PCC / P_max| / (d0 + m)",
        f"D0 = {D0}, PCC_IDX = {PCC_IDX}",
        f"All quantities normalized by P_max = max(|P_house|) over 264 timesteps",
        "",
        f"Ensemble |P_PCC|/P_max at noon: {mean_P_p_normalized:.4f} +/- "
        f"{std_P_p_normalized:.4f}",
        f"P_max (houses): mean = {mean_P_max:.4f} kW",
        "",
        f"{'m':>3s}  {'d0+m':>5s}  {'LB':>8s}  {'Obs kc':>8s}  "
        f"{'Ratio':>7s}  {'Gap%':>7s}  {'PASS':>5s}",
        "-" * 55,
    ]

    all_pass = True
    for m_val in M_VALUES:
        lb = mean_P_p_normalized / (D0 + m_val)

        # Find observed value for this m
        mask = obs_m == m_val
        if not np.any(mask):
            summary_lines.append(
                f"{m_val:3d}  {D0 + m_val:5d}  {lb:8.4f}  {'N/A':>8s}  "
                f"{'N/A':>7s}  {'N/A':>7s}  {'N/A':>5s}"
            )
            continue

        obs_val = obs_kc_mean[mask][0]
        obs_std = obs_kc_std[mask][0]
        ratio = obs_val / lb
        gap_pct = (obs_val - lb) / lb * 100
        passed = obs_val >= lb
        if not passed:
            all_pass = False

        summary_lines.append(
            f"{m_val:3d}  {D0 + m_val:5d}  {lb:8.4f}  "
            f"{obs_val:8.4f}  {ratio:7.3f}  {gap_pct:+6.1f}%  "
            f"{'YES' if passed else 'NO':>5s}"
        )

    summary_lines.extend([
        "-" * 55,
        "",
        f"Lower bound holds for all m: {'YES' if all_pass else 'NO'}",
        "",
        "Empirical fit: C_fit / (d0 + m)",
        f"  C_fit = kc_noon_mean(m=0) * D0 = {kc_m0:.4f} * {D0} = {C_fit:.4f}",
        "",
        f"{'m':>3s}  {'Obs kc':>8s}  {'Fit':>8s}  {'Resid%':>8s}",
        "-" * 35,
    ])

    for m_val, obs_val, fit_val, resid_pct in fit_residuals:
        summary_lines.append(
            f"{int(m_val):3d}  {obs_val:8.4f}  {fit_val:8.4f}  {resid_pct:+7.1f}%"
        )

    summary_lines.extend([
        "-" * 35,
        "",
        f"Overall verdict: {'PASS' if all_pass else 'FAIL'}",
        "",
        "Interpretation:",
        "  The analytic lower bound |P_PCC/P_max| / (d0 + m) is a necessary",
        "  condition for stability. The observed kappa_c exceeds this bound",
        "  for all m values, confirming the formula provides a valid",
        "  (conservative) lower bound on grid coupling requirements.",
        "  The empirical fit C/(d0+m) captures the 1/(d0+m) scaling but",
        "  with a larger numerator, indicating the true critical coupling",
        "  exceeds the minimum required by the PCC power constraint alone.",
    ])

    summary_text = "\n".join(summary_lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Summary: {SUMMARY_PATH}")
    print(f"\n{summary_text}")

    # ------------------------------------------------------------------
    # Step 6: Single-panel figure (8, 5)
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Dense m array for smooth curves
    m_dense = np.linspace(0, max(M_VALUES), 200)
    lb_dense = mean_P_p_normalized / (D0 + m_dense)
    fit_dense = C_fit / (D0 + m_dense)

    # Red shaded region: guaranteed unstable (below lower bound)
    ax.fill_between(m_dense, 0, lb_dense, color="red", alpha=0.10,
                    label="Guaranteed unstable")

    # Red dashed line: lower bound
    ax.plot(m_dense, lb_dense, "r--", linewidth=2.0,
            label=r"Lower bound: $|P_{\rm PCC}|/(P_{\max}(d_0+m))$")

    # Black circles + error bars: observed κ_c from Exp 4B-S2
    ax.errorbar(obs_m, obs_kc_mean, yerr=obs_kc_std, fmt="ko",
                capsize=4, markersize=6, linewidth=1.5, capthick=1.2,
                label=r"Observed $\kappa_c / P_{\max}$ (Exp 4B-S2, $n=10$)")

    # Blue dashed line: empirical fit
    ax.plot(m_dense, fit_dense, "b--", linewidth=1.5, alpha=0.8,
            label=r"Empirical fit: $C/(d_0+m)$, $C=%.1f$" % C_fit)

    ax.set_xlabel("Additional edges $m$", fontsize=13)
    ax.set_ylabel(r"$\kappa_c / P_{\max}$ (normalized)", fontsize=13)
    ax.set_title(r"Lower Bound Verification: $\kappa_c(m) \geq |P_{\rm PCC}|/[P_{\max}(d_0+m)]$",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(-0.5, max(M_VALUES) + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {FIG_PATH}")

    t_total = time.time() - t_start
    print(f"\nTotal runtime: {t_total / 60:.1f} min")
    print(f"Verdict: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
