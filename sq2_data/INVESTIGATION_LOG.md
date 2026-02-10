# SQ2 Investigation Log: Continuous Simplex Formula + PCC Inclusion

**Date**: 2026-02-10
**Context**: Reverting from discrete counting back to continuous density formula, and fixing PCC omission bug.

---

## Background

The previous fix (discrete counting with threshold `tol=0.1`) was incorrect. Smith et al. Fig.4 style trajectories use the **continuous density formula** (`continuoussourcesinkcounter`), not discrete node counting (`sourcesinkcounter`).

Additionally, GridResilience's `continuoussourcesinkcounter()` is called on the **full Pvec including PCC** (see `powerclasses.py:add_trajectory_point()`), but our code was excluding PCC.

---

## Diagnostic: Continuous Formula With/Without PCC

Ran `_diag_continuous.py` with seed=20260209, first ensemble instance.

### Midnight (00:00) — All consumption, no PV

```
--- WITHOUT PCC (n=50, houses only) ---
  max(P)  = +0.000000
  min(P)  = -1.393000
  #sources = 0,  sum(sources) = 0.000000
  #sinks   = 49,  sum(|sinks|) = 7.356531
  sigma_s = 0.000000
  sigma_d = 0.105621
  sigma_p = 0.894379

--- WITH PCC (n=51, houses + PCC) ---
  PCC value = +7.356531
  max(P)  = +7.356531  ← PCC becomes largest source!
  #sources = 1,  sum(sources) = 7.356531
  #sinks   = 49,  sum(|sinks|) = 7.356531
  sigma_s = 0.019608
  sigma_d = 0.103550
  sigma_p = 0.876842
```

**Key insight**: At night, PCC is the only source (supplying all demand). Without PCC, sigma_s=0 exactly. With PCC, sigma_s > 0 and sigma_d decreases slightly due to n increasing from 50→51.

### Morning (09:00) — High PV generation

```
--- WITHOUT PCC (n=50, houses only) ---
  max(P)  = +3.123667
  min(P)  = -1.043333
  sigma_s = 0.352101
  sigma_d = 0.040399
  sigma_p = 0.607500

--- WITH PCC (n=51, houses + PCC) ---
  PCC value = -52.884803  ← PCC absorbs massive excess!
  max(P)  = +3.123667
  min(P)  = -52.884803  ← PCC becomes largest sink by far!
  sigma_s = 0.345197
  sigma_d = 0.020389  ← halved! (denominator n*|min| explodes)
  sigma_p = 0.634414
```

**Key insight**: During daytime, PCC absorbs net surplus. |PCC| >> |any house|, so `largest_sink` jumps from ~1 to ~53. This dramatically reduces sigma_d (denominator explodes). sigma_s decreases slightly from n change.

### Midday (12:00) — Peak PV

```
--- WITHOUT PCC ---
  sigma_s = 0.353919,  sigma_d = 0.038709,  sigma_p = 0.607372

--- WITH PCC ---
  PCC value = -48.819136
  sigma_s = 0.346979,  sigma_d = 0.019955,  sigma_p = 0.633066
```

Similar pattern to 09:00.

---

## Analysis

### PCC's role in the continuous formula

1. **Night**: PCC is the sole source (grid imports). Including it gives sigma_s > 0, which is physically correct — the grid IS supplying power.

2. **Day**: PCC is the largest sink by an order of magnitude (absorbing surplus PV). This **suppresses sigma_d** because the denominator `n * |min(P)|` becomes huge while the numerator (sum of individual house sinks) stays small.

3. **Physical interpretation**: The continuous formula with PCC captures the grid-connected nature of the microgrid — at night the grid feeds the community (sigma_s > 0), and during the day the few remaining consumers are "small" relative to the grid's absorption capacity.

### Impact on trajectories

| Metric | Without PCC | With PCC | Effect |
|--------|-------------|----------|--------|
| sigma_s range | 0.00 – 0.35 | 0.02 – 0.35 | Night floor rises |
| sigma_d range | 0.04 – 0.11 | 0.02 – 0.10 | Daytime suppressed |
| sigma_p range | 0.61 – 0.89 | 0.63 – 0.88 | Slightly higher |

The trajectory should still show clear day/night oscillation but with a more compressed sigma_d range during daytime (PCC normalization effect).

---

## Decision

Reverted to continuous density formula (`continuoussourcesinkcounter`) with full Pvec (including PCC), matching GridResilience exactly. This is the correct formula for Fig.4-style trajectories per Research Lead's directive.
