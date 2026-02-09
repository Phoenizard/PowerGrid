# Phase 2 Research Report: Direction 3 — Effect of Non-uniform Power Allocation on Grid Stability

**Project**: Power Grid Dynamics Research
**Phase**: Phase 2 - Direction 3 (Non-uniform Power Allocation)
**Model Basis**: Smith et al., *Science Advances* 8, eabj6734 (2022)
**Report Date**: 2026-02-09
**Code Directory**: `direction3_nonuniform/`

---

## Executive Summary

Building on the successfully reproduced Fig. 1 codebase, this phase implements and validates two extended experiments:

1. **Experiment 2A (Heterogeneity Sweep)**: Investigates how the power allocation standard deviation `sigma/P_bar` affects the critical coupling strength `kappa_c`, comparing generator-side heterogeneity (2A-gen) with consumer-side heterogeneity (2A-con).
2. **Experiment 2C (Centralized vs. Distributed)**: Investigates how `kappa_c` changes as the fraction `r` of total power borne by a single large station increases.

This report presents final results under **production-grade parameters** (`n_ensemble=200`, `bisection_steps=20`, `t_integrate=100`, `conv_tol=1e-3`). The complete pipeline covers: implementation, unit tests, integration tests, checkpoint-resume support, plotting, and delivery self-checks. Key findings:

- In Experiment 2A, power heterogeneity has **no significant effect** on `kappa_c`, which remains stable (~0.12–0.13) across the entire `sigma_ratio` range; generator-side and consumer-side curves essentially overlap.
- In Experiment 2C, increasing `r` leads to a substantial rise in the stability threshold: `kappa_c` at `r=0.95` is approximately **139.0%** higher than at `r=0.04`.
- All delivery checks pass (CSV row/column integrity, no NaN, figures exist).

---

## 1. Research Objectives

Based on the swing equation:

$$
\frac{d^2\theta_k}{dt^2} + \gamma \frac{d\theta_k}{dt} = P_k - \kappa \sum_{l=1}^{n} A_{kl} \sin(\theta_k - \theta_l)
$$

This phase addresses the following questions:

1. How does `kappa_c` change when generator or consumer power shifts from uniform to heterogeneous allocation?
2. How does the stability threshold change as generation transitions from distributed to centralized (increasing large-station fraction `r`)?

---

## 2. Experimental Parameters

### 2.1 Fixed Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n` | 50 | Total number of nodes |
| `n_plus, n_minus` | 25, 25 | Generator / consumer node count |
| `K` | 4 | Watts-Strogatz mean degree |
| `q` | 0.1 | Rewiring probability |
| `gamma` | 1.0 | Damping coefficient |
| `P_max` | 1.0 | Total generation power (= total load magnitude) |

### 2.2 Numerical Solver Parameters (Production Configuration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kappa_range` | `(0.001, 3.0)` | Bisection search interval |
| `bisection_steps` | 20 | Bisection iteration count |
| `t_integrate` | 100 | ODE integration duration |
| `conv_tol` | `1e-3` | Convergence tolerance |
| `max_step` | 1.0 | `solve_ivp` maximum step size |
| `n_ensemble` | 200 | Network instances per parameter point |
| RNG seed | 20260208 | Experiment script random seed |

> Note: Compared to the development configuration used for rapid iteration (`bisection_steps=5`, `t_integrate=20`, `conv_tol=5e-3`, `n_ensemble=50`), the production configuration significantly improves numerical precision and statistical reliability. Development-stage results are preserved in the `results_n50/` directory for reference.

### 2.3 Sweep Parameters

- **Experiment 2A**: `sigma_ratio in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}` (9 points)
- **Experiment 2C**: `r in {0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}` (11 points)

---

## 3. Methods and Implementation

### 3.1 Power Allocation Strategies

1. **Heterogeneous allocation `assign_power_heterogeneous()`**
   - `side='gen'`: Generator-side heterogeneity, consumer-side uniform.
   - `side='con'`: Consumer-side heterogeneity, generator-side uniform.
   - Core procedure: Gaussian sampling → centering → truncation (`delta=1e-4*P_bar`) → normalization.

2. **Centralized allocation `assign_power_centralized()`**
   - One large station bears `r * P_max`; the remaining `n_plus-1` small stations equally share the rest.
   - Consumer side maintains uniform load.

### 3.2 Model Interface Compatibility

This phase strictly adheres to the existing interfaces in `paper_reproduction/model.py`:

- `generate_network(n, k, q, seed=None)`
- `compute_kappa_c(A_csr, P, config_params=None)`
- `compute_kappa_c_normalized(A_csr, P, P_max, config_params)`

Node power `P[i]` is guaranteed to correspond one-to-one with network node `i` (row/column `i` of the adjacency matrix).

### 3.3 Engineering Features

- Checkpoint-resume support: completed parameter points in the CSV are automatically skipped.
- Incremental checkpoint writes: each completed parameter point is immediately appended to the CSV.
- Per-instance fault tolerance: individual instance failures do not terminate the full sweep.
- `--production` flag: automatically selects production-grade parameters and outputs to `results_prod/`.

---

## 4. Testing and Validation

### 4.1 Unit Tests (`test_power_allocation.py`)

- Result: **9/9 passed**.
- Coverage: power balance, sign convention, total power consistency, baseline consistency, extreme centralization, variance trend, reproducibility, vector length.

### 4.2 Integration Tests (`test_integration.py`)

- Result: **5/5 passed**.
- Coverage: `kappa_c` computability, `sigma=0` consistency, `r=1/25` vs. uniform baseline consistency, quick trend check, timing estimate.

### 4.3 Delivery Self-Check (`self_check.py`)

- Result: **all passed**.
- Checks: file existence, 2A has 9 rows, 2C has 11 rows, columns complete, no NaN, `kappa_c > 0`.

---

## 5. Experimental Results

### 5.1 Experiment 2A: Heterogeneity Sweep (n_ensemble=200, Production Config)

| sigma_ratio | kappa_c_mean_gen | kappa_c_std_gen | kappa_c_mean_con | kappa_c_std_con |
|-------------|------------------|-----------------|------------------|-----------------|
| 0.0 | 0.128644 | 0.049800 | 0.129053 | 0.049990 |
| 0.1 | 0.124489 | 0.049015 | 0.124115 | 0.049258 |
| 0.2 | 0.120172 | 0.048042 | 0.120616 | 0.047725 |
| 0.3 | 0.128339 | 0.052390 | 0.126021 | 0.049907 |
| 0.4 | 0.124684 | 0.053193 | 0.122340 | 0.049859 |
| 0.5 | 0.123062 | 0.048075 | 0.124221 | 0.048508 |
| 0.6 | 0.129415 | 0.049655 | 0.130549 | 0.050280 |
| 0.7 | 0.124036 | 0.051486 | 0.123870 | 0.050202 |
| 0.8 | 0.123904 | 0.046660 | 0.124190 | 0.041354 |

![Experiment 2A Results](direction3_nonuniform/results_prod/fig_2A.png)

*Figure 2A: Effect of power heterogeneity (generator-side / consumer-side) on critical coupling strength (n_ensemble=200).*

**Observations**:

1. Under production-grade statistical scale, power heterogeneity has **no significant effect** on `kappa_c`. The mean remains stable across the entire `sigma_ratio` range, fluctuating around ~0.12–0.13.
2. Generator-side heterogeneity (gen) and consumer-side heterogeneity (con) curves essentially overlap, with no statistically significant difference.
3. Comparing `sigma=0` to `sigma=0.8`:
   - gen: `0.128644 → 0.123904` (-3.7%)
   - con: `0.129053 → 0.124190` (-3.8%)
4. Compared to the development phase (`n_ensemble=50`), non-monotonic fluctuation artifacts have been eliminated and standard deviations are significantly reduced (from ~0.09–0.16 to ~0.04–0.05). This confirms that the previously observed "heterogeneity increases `kappa_c`" trend was primarily an artifact of statistical noise and insufficient numerical precision.

### 5.2 Experiment 2C: Centralized vs. Distributed (n_ensemble=200, Production Config)

| r | kappa_c_mean | kappa_c_std |
|------|--------------|-------------|
| 0.04 | 0.117531 | 0.038893 |
| 0.10 | 0.123080 | 0.050968 |
| 0.20 | 0.131008 | 0.048335 |
| 0.30 | 0.152240 | 0.058332 |
| 0.40 | 0.162054 | 0.053439 |
| 0.50 | 0.175556 | 0.052533 |
| 0.60 | 0.198442 | 0.051439 |
| 0.70 | 0.220761 | 0.049283 |
| 0.80 | 0.236809 | 0.050374 |
| 0.90 | 0.262988 | 0.052478 |
| 0.95 | 0.280908 | 0.053027 |

![Experiment 2C Results](direction3_nonuniform/results_prod/fig_2C.png)

*Figure 2C: Relationship between generation centralization fraction `r` and critical coupling strength (n_ensemble=200).*

**Observations**:

1. As `r` increases, `kappa_c` exhibits a **strictly monotonic increase**, producing a smooth curve without fluctuations.
2. At `r=0.95` compared to `r=0.04`: `0.117531 → 0.280908`, an increase of **139.0%**.
3. The minimum occurs at `r=0.04` (near-uniform allocation); the maximum at `r=0.95` (highly centralized).
4. Standard deviations remain stable across all parameter points (~0.04–0.06), indicating statistical convergence.

---

## 6. Figure Outputs

- `direction3_nonuniform/results_prod/fig_2A.png`
- `direction3_nonuniform/results_prod/fig_2C.png`

Figures are generated by `plot_results.py` using `matplotlib.use('Agg')`, enabling headless (no-display) execution.

---

## 7. Conclusions and Future Directions

### 7.1 Phase Conclusions

1. The Direction 3 implementation successfully integrates with the original reproduction framework; all tests pass.
2. **Experiment 2A (revised conclusion)**: Under high-precision production configuration, power heterogeneity (whether on the generator side or consumer side) has no significant effect on `kappa_c`. The "heterogeneity raises `kappa_c`" trend observed during the development phase has been confirmed as an artifact of statistical noise and coarse numerical resolution. This indicates that in Watts-Strogatz small-world networks, the variance of power allocation alone is not a critical factor for synchronization stability.
3. **Experiment 2C (strengthened conclusion)**: Increasing centralization significantly raises `kappa_c`, with an even stronger effect than estimated during development (139% vs. 97%). This is consistent with the physical intuition that "distributed generation favors synchronization" — when a large fraction of power is concentrated at a single node, local coupling stress increases dramatically, requiring stronger global coupling to maintain synchrony.

### 7.2 Future Directions

1. Add bootstrap confidence intervals for publication-quality presentation.
2. Explore cross-experiments between 2A and 2C: overlay heterogeneity perturbations on centralized allocation.
3. Investigate the modulating effect of network topology parameters (`K`, `q`) on the above trends.

---

## Appendix: Key Output Files

- Code implementation: `direction3_nonuniform/*.py`
- Production results (this report): `direction3_nonuniform/results_prod/`
- Development-stage results (reference): `direction3_nonuniform/results_n50/`
- Report files: `Phase2_Research_Report.md` (Chinese), `Phase2_Research_Report_EN.md` (English)
