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

### Alpha parameter semantics (CORRECTED 2026-02-14)

- **Engine level** (`cascade_engine.py`): `alpha` is a dimensionless multiplier on f_max. Edge `(i,j)` fails when `|f_ij| > alpha * f_max_initial`.
- **CSV output**: `alpha_over_alpha_star` is the dimensionless alpha value passed directly to the engine.
- **Bisection**: `find_alpha_c()` operates in the same dimensionless alpha space. Returns alpha_c as a multiplier on f_max.
- **rho**: `rho = alpha_c` (alpha_c IS Smith's rho — no further division needed).

See `NORMALIZATION_AUDIT.md` for full analysis of the two bugs that were corrected.

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

## Normalization Audit (2026-02-14)

Two bugs were discovered and fixed. See `NORMALIZATION_AUDIT.md` for detailed analysis.

### Bug 1 (Critical): Sweep alpha double-multiplication

`run_multistep.py` and `run_option_d.py` pre-multiplied alpha by f_max before passing to the cascade engine, which internally multiplies by f_max again. This made the effective threshold `alpha * f_max^2` instead of `alpha * f_max`. All sweep CSV data was regenerated.

Bisection (`find_alpha_c`) was NOT affected — it constructs alpha values internally.

### Bug 2: rho double-division

`rho = alpha_c / f_max_dc` was wrong because alpha_c is already a dimensionless multiplier on f_max (i.e., alpha_c IS rho). The division produced values that were too small by a factor of f_max for PCC networks (where f_max > 1) and too large for non-PCC networks (where f_max < 1).

### Corrected results (n=10 sanity check)

**Option D** (non-PCC, n=100):
- rho: mean=0.841, std=0.145, range=[0.558, 1.000] (was ~15 before fix)
- Convergence: 77/100 (77%)

**Multistep** key results:

| Time | m | rho (mean) | conv |
|------|---|-----------|------|
| 00:00 | 0 | 0.170 | 10% |
| 06:00 | 0 | 0.520 | 90% |
| 09:00 | 0 | 0.186 | 80% |
| 12:00 | 0 | 0.200 | 90% |
| 12:00 | 4 | 0.287 | 90% |
| 12:00 | 8 | 0.368 | 70% |
| 18:00 | 0 | 0.274 | 80% |

Key physics:
- rho increases with m (more PCC edges = higher cascade resilience)
- Daytime rho ~ 0.15-0.20 (low due to PCC flow concentration)
- Nighttime convergence is poor due to staircase S(alpha) regime
- f_max still decreases with m as expected

### Files changed

| File | Changes |
|------|---------|
| `run_multistep.py` | 4 locations: removed alpha pre-multiplication, fixed rho |
| `run_option_d.py` | 3 locations: removed alpha pre-multiplication, fixed rho, reverted ALPHA_RANGE to (0.1, 2.5) |
| `NORMALIZATION_AUDIT.md` | Created — full audit documentation |

## Deferred Work

- **Swing validation**: Deferred to post-DC analysis. DC model is primary.
- **Production run**: n=50 after n=10 sanity check passes verification checklist.
