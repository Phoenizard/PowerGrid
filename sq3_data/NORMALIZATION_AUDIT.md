# SQ3 Normalization Audit

Date: 2026-02-14

## Motivation

Option D showed rho ~ 15 (mean) while Smith's Fig. 3 shows rho ~ O(0.1-0.5). This audit traces the alpha/rho normalization chain through Smith's code and ours to identify the discrepancy.

## Q1: How does Smith normalize flows before comparing to alpha?

**Smith's Julia code** (`networkalgorithms.jl:211`):
```julia
F = edgecurrents(theta, E1, X1) ./ fmaxtemp   # F in [0, 1]
if abs(F[i]) > alpha                            # alpha is dimensionless
```

**Our `cascade_engine.py`**:
```python
threshold = alpha * f_max_initial    # edge fails if |f_ij| > threshold
```

These are **mathematically equivalent**. Smith normalizes flows then compares to alpha; we keep flows raw and scale the threshold by f_max. In both cases, alpha is a dimensionless multiplier on f_max.

## Q2: What is alpha_c and how does it relate to rho?

`find_alpha_c()` passes alpha values directly to `run_cascade_dc(A, P, alpha, f_max)`. The engine computes `threshold = alpha * f_max`. Therefore **alpha_c is already dimensionless** (it's the multiplier on f_max at the critical transition).

Smith's rho = alpha_c (the critical multiplier). **No further division by f_max is needed.**

## Bug 1 (Critical): Double-multiplication in sweep alpha

### Location
- `run_multistep.py` lines 184-186 (sweep), 211-213 (alpha* query), 231 (low-alpha PCC check)
- `run_option_d.py` lines 111-113 (sweep)

### Description
The sweep code pre-multiplied alpha_ratio by f_max before passing to the engine:
```python
# WRONG (was):
alpha_abs = alpha_ratio * f_max_dc          # e.g., 0.5 * 6.0 = 3.0
run_cascade_dc(A, P, alpha_abs, f_max_dc)   # threshold = 3.0 * 6.0 = 18.0
```

Inside the engine, `threshold = alpha * f_max_initial`, so the effective threshold became `alpha_ratio * f_max^2` instead of `alpha_ratio * f_max`.

### Fix
Pass alpha_ratio directly — the engine handles the f_max multiplication:
```python
# CORRECT:
run_cascade_dc(A, P, alpha_ratio, f_max_dc)  # threshold = 0.5 * 6.0 = 3.0
```

### Impact
All sweep CSV data was wrong. Thresholds were f_max times too high, meaning cascades were much harder to trigger than intended. This shifted the sigmoid rightward by a factor of f_max.

### Why bisection was NOT affected
`find_alpha_c()` constructs its own alpha values internally (starting at 0.01, step 0.3) and passes them directly to the engine. No pre-multiplication occurs. The returned alpha_c values are correct.

## Bug 2: rho double-division

### Location
- `run_multistep.py` line 223
- `run_option_d.py` line 124

### Description
Since alpha_c is already dimensionless (the critical multiplier on f_max), dividing by f_max again yields a number that's too small by a factor of f_max:
```python
# WRONG (was):
rho = alpha_c / f_max_dc    # double-divides

# CORRECT:
rho = alpha_c               # alpha_c IS rho
```

### Concrete example
For Option D with f_max = 0.08, alpha_c = 0.135:
- Old: rho = 0.135 / 0.08 = 1.69 (too high because f_max is small)
- Correct: rho = 0.135

For PCC networks with f_max = 5.85, alpha_c = 0.135:
- Old: rho = 0.135 / 5.85 = 0.023 (too low because f_max is large)
- Correct: rho = 0.135

The direction of the error depended on whether f_max > 1 or f_max < 1, which is why Option D (small f_max) showed inflated rho while PCC networks showed deflated rho.

## Q3: Trigger mechanism

Smith's `fracture!()` removes ALL overloaded edges simultaneously per round, which matches our `_run_cascade_dc_inner`. No fix needed.

## Q4: ALPHA_RANGE for Option D

With the double-multiplication bug fixed, alpha values are dimensionless multipliers. Smith's range [0.1, 2.5] is appropriate. The previous ALPHA_RANGE = (0.1, 30.0) was a workaround for the bug. Reverted to (0.1, 2.5).

## Summary of Changes

| File | Line(s) | Change |
|------|---------|--------|
| `run_multistep.py` | 183-186 | Remove `alpha_abs = alpha_ratio * f_max_dc`; pass `alpha_ratio` directly |
| `run_multistep.py` | 210-213 | Remove `alpha_abs_at_star`; pass `alphas_rel[closest_idx]` directly |
| `run_multistep.py` | 222 | `rho = alpha_c` (was `alpha_c / f_max_dc`) |
| `run_multistep.py` | 230 | `alpha_low = 0.1` (was `0.1 * f_max_dc`) |
| `run_option_d.py` | 110-113 | Remove `alpha_abs = alpha_ratio * f_max_dc`; pass `alpha_ratio` directly |
| `run_option_d.py` | 123 | `rho = alpha_c` (was `alpha_c / f_max_dc`) |
| `run_option_d.py` | 44 | `ALPHA_RANGE = (0.1, 2.5)` (was `(0.1, 30.0)`) |

## Expected Corrected Results

- **Option D**: rho ~ 0.1-0.5 (was ~15), convergence should increase to >90%
- **PCC networks (noon, m=0)**: rho ~ 0.1-0.2 (was ~0.023)
- Sigmoid sweep should show correct transition in [0.1, 2.5] range
