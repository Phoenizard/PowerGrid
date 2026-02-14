# SQ3 Experiment Log — Multi-Timestep Cascade Resilience

## Overview

SQ3 investigates cascade resilience of PCC-augmented power grid networks using Smith et al.'s DC power flow cascade model. The pilot (n=10, noon only) revealed that PCC flow concentration causes S vs alpha/alpha* to exhibit staircase behavior rather than Smith's smooth sigmoid. This experiment extends across multiple times-of-day to reveal the staircase-sigmoid regime transition.

## Code Modifications

### `cascade_engine.py` — Added `run_cascade_dc_tracked()`

A new public function alongside the existing `run_cascade_dc()`. This variant tracks which edges survive the cascade, enabling PCC vs non-PCC classification by the caller.

Key implementation details:
- `_run_cascade_dc_tracked_inner()` carries a `global_nodes` array through recursion
- When recursing into fragments, local edge indices are mapped back to global node indices
- Returns `(CascadeResult, set[tuple[int,int]])` where edges are `(i,j)` with `i < j`
- The original `run_cascade_dc()` is unchanged

### Edge classification (in `run_multistep.py`)

For each surviving edge `(i,j)`:
- PCC edge: `i == 49` or `j == 49`
- `pcc_isolated`: True when no PCC edges survive

## Known Issues

### Bisection convergence on staircase curves

`find_alpha_c()` uses Smith's adaptive bisection (linear advance + bisection). When S vs alpha exhibits staircase behavior (discrete jumps), S may never smoothly cross 0.5, causing the bisection to converge to a jump point rather than a true critical threshold.

**Mitigation**: The `bisection_converged` flag is computed as:
```python
bisection_converged = abs(S_at_alpha_c - 0.5) < 0.2
```
This is recorded in both sweep and summary CSVs. Downstream analysis should use `bisection_converged_frac` to assess result reliability.

### Alpha parameter semantics

- **Engine level** (`cascade_engine.py`): `alpha` is an ABSOLUTE overload threshold. Edge `(i,j)` fails when `|f_ij| > alpha * f_max_initial`.
- **CSV output**: `alpha_over_alpha_star = alpha / f_max_dc` (dimensionless, relative to initial max flow).
- **Bisection**: `find_alpha_c()` operates in absolute alpha space (starts at 0.01, step 0.3, ceiling 20.0). Returns absolute alpha_c.
- **rho**: `rho = alpha_c / f_max_dc = alpha_c / alpha_star` (dimensionless).

## Experiments

### Priority 1: Multi-Timestep Cascade (`run_multistep.py`)

| Parameter | Value |
|-----------|-------|
| Timesteps | 00:00, 06:00, 09:00, 12:00, 18:00 |
| m configs | (0, pcc_direct), (4, pcc_direct), (8, pcc_direct), (4, random) |
| n_ensemble | 10 (sanity) / 50 (production) |
| Seed | 20260214 |
| Alpha sweep | 50 points in [0.1, 2.5] × alpha_star |
| Cascade model | DC only |

### Priority 2: Option D (`run_option_d.py`)

| Parameter | Value |
|-----------|-------|
| Network | WS(50, 4, 0.1) — no PCC node |
| Power | 25 sources (+1/25) + 25 sinks (-1/25), shuffled |
| n_instances | 100 |
| Seed | 20260214 |
| Expected | Smooth sigmoid, convergence >90%, rho ~ log-normal |

## Pilot Results Reference (n=10, noon only)

| m | alpha_c (mean) | f_max_dc (mean) | rho (mean) |
|---|---------------|-----------------|------------|
| 0 | 0.135 | 5.85 | ~0.023 |
| 4 | 0.191 | 3.32 | ~0.058 |
| 8 | 0.232 | 2.44 | ~0.095 |

Key findings:
- f_max decreases with m (more PCC edges distribute flow)
- PCC flow concentration ratio ~11.2x at noon
- No desync events at kappa=10 in swing model

## Deferred Work

- **Swing validation**: Deferred to post-DC analysis. DC model is primary.
- **Production run**: n=50 after n=10 sanity check passes verification checklist.
