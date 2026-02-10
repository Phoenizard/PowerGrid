# CLAUDE.md — Power Grid Dynamics (MATH3060)

## Project Overview

MATH3060 Group Project. Reproducing and extending Smith et al. (2022) *Science Advances* 8, eabj6734.
- **Swing Equation** model on Watts-Strogatz microgrids
- **Core question**: How do renewable temporal patterns affect grid resilience, and can topology design compensate?
- **Deadline**: Report Mar 9, Presentation Mar 16

## Team

| Role | Agent | Responsibilities |
|------|-------|-----------------|
| PI & Decision Maker | Phoenizard | Strategy, supervisor liaison, final approval |
| Research Lead | Claude Chat | Experiment design, storyline, report structure |
| Engineering | Claude Code (you) | Code implementation, debugging, execution |
| Data & Literature | GPT 5.2 | Data inspection, exploratory analysis |

## Sub-Questions & Progress

| SQ | Description | Status |
|----|-------------|--------|
| SQ1 | Decentralization vs κ_c + non-uniform power | ✅ COMPLETE (n=200, Exp 2A/2C) |
| SQ2-A | Simplex trajectory (η⁺, η⁻, ηₚ) | ✅ COMPLETE — bug fixed, data validated |
| SQ2-B | κ_c time series (56 points × ensemble) | 🔲 NEXT |
| SQ3 | Cascading failures | 🔲 Optional |
| SQ4 | Topology optimization (q + edge addition) | 🔲 Spec complete, pending |

## Current Task: SQ2-B (κ_c Time Series)

Compute κ_c at 8 representative times/day × 7 days = 56 time points.
- Bisection parameters: same as SQ1
- Network: WS(n=50, K̄=4, q=0.1) + 1 PCC = 51 nodes
- Output: `sq2_data/results_sq2/kappa_c_timeseries_summer_100pct.csv`
- Plot: `sq2_data/results_sq2/fig_3B_kappa_timeseries.png`

## Reference Materials (MUST READ before coding)

1. **SQ2 Experiment Spec** (Notion): `🔬 SQ2 Experiment Spec — Data-Driven Simplex Trajectories`
2. **Smith et al. official code**: https://doi.org/10.5281/zenodo.5702877
   - Key file: `scripts/powerclasses.py` — `continuoussourcesinkcounter()` (lines 292-301)
   - PCC IS included in Pvec for density calculation
3. **Data pipeline doc**: `sq2_data/DATA_PIPELINE_README.md`
4. **SQ1 code** (for bisection reference): `direction3/` folder
5. **Investigation log**: `sq2_data/INVESTIGATION_LOG.md`

## Data Pipeline Summary

- **LCL consumption**: `data/LCL/Small LCL Data/` — raw KWH/hh values, NO unit conversion
- **PV generation**: `data/PV/.../EXPORT TenMinData - Customer Endpoints.csv` — 6 substations, resample to 30min
- **PV assignment**: Each house randomly samples one substation PV profile (with replacement)
- **PCC**: P_PCC(t) = −Σ P_i(t), connects to 3-4 random network nodes
- **Node classification threshold**: 0.01 × max(|P(t)|)
- **Continuous density**: Smith Eq. 3-4, PCC included in max/min calculations

### Known Bug (FIXED)
PV data magnitude was causing PCC to dominate density denominator. Fix applied — see INVESTIGATION_LOG.md and DATA_PIPELINE_README.md for details.

## ⚠️ Workflow Rules

### Ensemble Escalation Protocol
For ANY experiment involving ensemble runs:
1. **First**: n_ensemble = 10 (sanity check — verify outputs, no NaN, correct shape)
2. **Then**: n_ensemble = 50 (check convergence, compare with n=10)
3. **Finally**: n_ensemble = 200 (production run)

### Checkpoint & Report Rule
**STOP and report to Phoenizard (PI) before proceeding to the next experiment or phase.**
Specifically:
- After completing n_ensemble=10 sanity check → report results
- After n_ensemble=50 → report and await approval for production run
- After each experiment (e.g., 3A → 3B, SQ2 → SQ4) → report and await next instructions
- If encountering unexpected results or errors → report immediately, do NOT silently retry

### Report Format
When reporting, include:
- What was run (experiment ID, n_ensemble, parameters)
- Key metrics (e.g., σ ranges, κ_c values, runtime)
- Pass/fail on verification checklist
- Any anomalies or concerns
- Proposed next step (await approval)

### Output Requirements
- All plots must use **English labels**
- CSV files: no NaN, correct dimensions
- Power balance check: |Σ P_k| < 1e-10 at every timestep
- Git commit after each successful experiment

## Verification Checklist (SQ2)

### Exp 3A (Simplex Trajectory) — ✅ PASSED
- [x] σ_s + σ_d + σ_p ≈ 1.0 (error < 1e-10)
- [x] Trajectory NOT compressed at simplex bottom
- [x] Daytime → source corner, Nighttime → sink-passive edge
- [x] σ_s, σ_d both vary meaningfully (not stuck near 0)
- [x] Power balance: |Σ P_k| < 1e-10
- [x] Dawn/dusk → transition region

### Exp 3B (κ_c Time Series) — 🔲 PENDING
- [ ] 56 rows × 4 columns (time, kappa_c_mean, kappa_c_std, n_ensemble)
- [ ] κ_c shows diurnal oscillation
- [ ] Daytime κ_c ≠ nighttime κ_c (expect meaningful variation)
- [ ] No NaN or inf values
- [ ] Runtime reasonable (~1.5 hours for n=50)

## File Structure

```
sq2_data/
├── data_loader.py          # LCL + PV data loading
├── run_trajectory.py       # Exp 3A — simplex trajectory
├── run_kappa_timeseries.py # Exp 3B — κ_c vs time
├── plot_results.py         # Visualization
├── DATA_PIPELINE_README.md # Data pipeline documentation
├── INVESTIGATION_LOG.md    # Debug history
└── results_sq2/
    ├── trajectory_summer_fullpen.csv        ✅
    ├── fig_3A_simplex_trajectory.png        🔲 (need production quality)
    ├── kappa_c_timeseries_summer_100pct.csv 🔲
    └── fig_3B_kappa_timeseries.png          🔲
```
