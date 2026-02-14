# Claude Code Briefing — Power Grid Dynamics Project

**Last updated**: 2026-02-10
**Role assignment**: This document is maintained by the Research Lead (Claude Chat). Claude Code should read this before any coding session.

**Team**: Phoenizard (PI) | Claude Chat (Research Lead) | Claude Code (Engineering) | GPT 5.2 (Data & Literature)

> Note: Data inspection and exploratory analysis are handled by GPT 5.2. Claude Code receives cleaned data specs and focuses on model implementation and experiment execution.

---

## Project Overview

MATH3060 group project analyzing renewable energy impact on microgrid stability using Swing Equation models from Smith et al. (2022, Science Advances).

**Core research question**: How can microgrids anchor themselves in the optimal region of the resilience landscape through topology design, given the tension between temporal clustering and spatial distribution of renewable energy?

---

## Repository Structure

```
/Users/shay/Workplace/PowerGrid/
├── codebase/
│   ├── PowerGrid/              # OUR CODE (GitHub: Phoenizard/PowerGrid)
│   │   ├── paper_reproduction/ # Phase 1: Fig.1 reproduction (COMPLETE)
│   │   │   ├── model.py       # Core engine: swing eq, bisection, network gen
│   │   │   ├── config.py      # Parameters (production & dev modes)
│   │   │   ├── run_sweep.py   # Parameter sweep framework
│   │   │   └── plot_*.py      # Visualization scripts
│   │   ├── direction3_nonuniform/ # Phase 2: Non-uniform power (COMPLETE, n_ensemble=50)
│   │   └── results/
│   └── reference_code/
│       └── GridResilience/     # Smith et al. official code (Julia, READ-ONLY reference)
├── data/
│   ├── LCL/                    # London household electricity (168 CSVs, 30-min resolution, ~8GB)
│   └── PV/                     # London PV generation (NEEDS UNZIPPING: "PV Data - csv files only.zip")
├── doc/                        
├── fig/
└── CLAUDE_CODE_BRIEFING.md     # THIS FILE
```

---

## Key Technical Details

### Core Model (model.py)

- **Swing equation**: 2nd order ODE system, converted to 1st order for solve_ivp
- **Network**: Watts-Strogatz via networkx, stored as scipy CSR sparse matrix
- **κ_c computation**: Bisection search (20 steps, range [0.001, 3.0])
- **Convergence criterion**: max(|ω_i|) < conv_tol (frequency decay to zero)
- **Runtime**: ~1.8s per instance (n=50, single bisection)
- **Numba JIT**: Available for ODE RHS acceleration

### Production Parameters (config.py)

| Parameter | Value |
|-----------|-------|
| N | 50 |
| K (mean degree) | 4 |
| GAMMA | 1.0 |
| P_MAX | 1.0 |
| KAPPA_RANGE | (0.001, 3.0) |
| BISECTION_STEPS | 20 |
| CONV_TOL | 1e-3 |
| T_INTEGRATE | 100 |
| ENSEMBLE_SIZE_FINAL | 200 |

---

## Current Task Queue (from Research Lead)

### IMMEDIATE — Phase 2 Production Run
- [ ] Rerun experiments 2A and 2C with `n_ensemble=200` (production parameters)
- [ ] Verify direction3 code files are complete and committed to git

### IMMEDIATE — Data Preparation  
- [ ] Unzip PV data: `cd data/PV && unzip "PV Data - csv files only.zip" -d PV_csv`
- [ ] Write a data inspection script (`data_inspect.py`) that reports:
  - LCL: number of unique households, date range, missing data %
  - PV: number of panels, date range, resolution, missing data %
  - Output: summary to `data/data_summary.md`

### PHASE 3 — SQ2: Data-Driven Simplex Trajectories
*(Spec to follow after data inspection)*

### PHASE 4 — SQ4: Network Topology Design
*(Spec to follow — see experiment design below)*

---

## Experiment Designs (Research Lead Specifications)

### Already Complete: Experiments 2A & 2C (Direction 3)
See `Phase2_Research_Report.md` for details. Need production rerun at n_ensemble=200.

### SQ1: κ_c Simplex Reproduction
**Status**: COMPLETE (Phase 1). Fig.1C and Fig.1D reproduced.

### SQ2: Data-Driven Simplex Trajectories (PENDING — needs data inspection first)

**Goal**: Map real London household + PV data onto configuration simplex trajectories.

**Key computation**: For each time snapshot t:
- Each node i has net power P_i(t) = g_i(t) - c_i(t)
- Classify nodes: generator (P_i > 0), consumer (P_i < 0), passive (P_i ≈ 0)
- Compute continuous densities (η+, η-, ηp) per Smith et al. Eqs. 3-4
- Track trajectory over days/weeks/seasons

**Technical requirements**:
- Sample 50 households randomly from LCL dataset
- Assign PV generation to subset (50% or 100% uptake scenarios)
- Compute κ_c at representative time snapshots (e.g., every 30 min for 1 week)
- Plot trajectories on simplex + κ_c time series

### SQ4: Network Topology Design (NEW)

**Experiment 4A — q Parameter Scan**
- q ∈ {0.0, 0.2, 0.4, 0.6}
- For each q: compute κ̄_c with (n+=25, n-=25, np=0), n_ensemble=200
- If SQ2 is ready: also compute κ_c daily variation curves for each q
- Output: CSV + comparison plot

**Experiment 4B — Strategic Edge Addition (CORE INNOVATION)**
- Baseline: WS(n=50, K̄=4, q=0.1)
- Edge budget: m ∈ {2, 5, 10, 15, 20}
- Three strategies:
  1. **Random**: Uniformly random non-adjacent node pairs
  2. **High-power priority**: Connect pairs with largest |P_i| + |P_j| (big generators to big consumers)
  3. **Low-degree priority**: Connect pairs with lowest degree(i) + degree(j)
- For each (m, strategy): n_ensemble=200 → compute κ̄_c
- Output: CSV with columns [m, strategy, kappa_c_mean, kappa_c_std] + comparison plot

**Implementation notes for 4B**:
```python
def add_strategic_edges(A_csr, P, m, strategy='random', rng=None):
    """
    Add m edges to network A according to strategy.
    
    Parameters:
    -----------
    A_csr : adjacency matrix (will be modified)
    P : power vector (needed for 'high_power' strategy)
    m : number of edges to add
    strategy : 'random' | 'high_power' | 'low_degree'
    
    Returns:
    --------
    A_new : modified adjacency matrix with m additional edges
    """
```

**Experiment 4C — Topology × Power Heterogeneity Interaction**
- Take best strategy + best m from 4B
- Rerun 2A (σ sweep) and 2C (r sweep) under optimized topology
- Compare: original WS vs optimized topology
- Output: overlaid comparison plots

---

## Communication Protocol

1. After completing each task, update this file's task queue (mark [x] for done)
2. If encountering issues, add a section "## Issues for Research Lead" at the bottom
3. Research Lead will update experiment specs and task queue as project progresses

---

## Issues for Research Lead

### [RESOLVED] SQ2 Simplex Calculation — Reverted to Continuous Formula + PCC Fix

**History**: The original implementation used the continuous density formula (`continuoussourcesinkcounter`) but excluded PCC nodes, producing compressed trajectories. This was incorrectly "fixed" by switching to discrete counting (`sourcesinkcounter`).

**Correct approach**: Fig.4-style trajectories use the **continuous density formula** on the **full power vector including PCC**. This matches GridResilience exactly:
- `powerclasses.py:get_power_vec()` returns houses + PCC
- `powerclasses.py:add_trajectory_point()` passes full Pvec to `continuoussourcesinkcounter()`

**Two bugs fixed**:
1. **Formula**: Reverted from discrete counting back to continuous density (`continuoussourcesinkcounter`)
2. **PCC omission**: Now includes PCC node in Pvec (51 nodes, not 50)

**Diagnostic results** (see `sq2_data/INVESTIGATION_LOG.md`):
- Night: PCC is the sole source → sigma_s > 0 (physically correct: grid feeds community)
- Day: PCC absorbs surplus → becomes largest sink, suppressing sigma_d via denominator normalization
- Clear day/night oscillation preserved
