# SQ2 Data Pipeline Reference

Quick-reference for reusing the SQ2 (Exp 3A) data loading pipeline in Exp 3B and beyond.

---

## 1. Data Source Paths

| Dataset | Path (relative to project root) |
|---------|-------------------------------|
| LCL consumption | `data/LCL/LCL-June2015v2_*.csv` (168 files) |
| PV generation | `data/PV/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv` |
| LCL parquet cache | `data/LCL_parquet/*.parquet` (auto-detected, 10x faster) |
| PV parquet cache | same directory as CSV, file `pv_hourly_customer_endpoints.parquet` |

---

## 2. Loading Function Interfaces

### `data_loader.py`

```python
build_microgrid(
    lcl_dir: str,
    pv_hourly_path: str,
    season: str = "summer",      # "summer"|"winter"|"spring"|"autumn"
    n_houses: int = 49,
    penetration: int = 49,       # 24 = halfpen, 49 = fullpen
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]
# Returns (P, t_seconds)
#   P:         shape (n_houses + 1, 264)  — rows = nodes, cols = timesteps
#   t_seconds: shape (264,)               — time axis in seconds

make_consumption_interpolator(
    lcl_dir: str,
    target_month: int,           # 1=Jan … 12=Dec
    rng: RandomState,
) -> interp1d | None

make_pv_interpolator(
    pv_hourly_path: str,
    target_month: int,
    rng: RandomState,
) -> interp1d | None

N_TIMESTEPS = 264  # exported constant
```

### `simplex.py`

```python
compute_simplex_coordinates(Pvec: np.ndarray) -> tuple[float, float, float]
# Pvec: shape (n_nodes,) — full vector including PCC
# Returns (sigma_s, sigma_d, sigma_p)

compute_simplex_trajectory(P: np.ndarray) -> np.ndarray
# P: shape (n_nodes, T)
# Returns shape (T, 3) — columns (sigma_s, sigma_d, sigma_p)
```

### `network.py`

```python
generate_network_with_pcc(
    n_houses: int = 50,          # NOTE: default is 50, pass 49 explicitly
    k: int = 4,
    q: float = 0.1,
    n_pcc_links: int = 4,
    seed: int | None = None,
) -> csr_matrix
# Returns adjacency matrix, shape (n_houses + 1, n_houses + 1)
```

---

## 3. PV / LCL Unit Conventions

- **LCL column**: `"KWH/hh (per half hour) "` (trailing space in CSV header)
  - Raw kWh per half-hour; **no x2 conversion** to kW (matches GridResilience)
- **PV columns**: `P_GEN_MAX`, `P_GEN_MIN` from hourly CSV
  - Used as `P_GEN = (P_GEN_MAX + P_GEN_MIN) / 2` (kW)
- Both datasets go through: filter by month -> mean-week profile -> `interp1d(bounds_error=False, fill_value="extrapolate")`
- **Net power per house**: `P[i] = PV(t) - consumption(t)` (or `-consumption(t)` if no PV)
- Units are mixed (kWh/hh vs kW) but internally consistent with GridResilience

---

## 4. PCC Handling

- PCC = last node, index `n_houses` (default: index 49)
- Power: `P[PCC, t] = -sum(P[0:n_houses, t])` — enforces zero net power at every timestep
- **Must be included** in simplex calculations — pass the full `(n_houses+1, T)` array
- Network: Watts-Strogatz small-world graph for houses; PCC linked to 4 random houses

---

## 5. Time Grid

```python
_T_RAW  = np.linspace(0, 604800 - 1800, 336)[:-24]   # 312 steps (drops last 24)
_T_WEEK = _T_RAW[48:]                                  # 264 steps (skip first 24h)
```

- 604800 s = 1 week; 1800 s = 30 min
- 336 raw steps, drop last 24 -> 312, skip first 48 -> **264 timesteps**
- Season -> month mapping: summer=7, winter=1, spring=4, autumn=10

---

## 6. Bug Fix History

Bugs fixed during SQ2 development (commits `1ed0bea`, `7c0398d`):

| Bug | What was wrong | Fix |
|-----|---------------|-----|
| PCC omitted from simplex | Only house nodes passed to simplex | Use full Pvec (houses + PCC) |
| Discrete formula | Used threshold-based node counting | Continuous density: `sum / (n * max)` |
| LCL x2 conversion | Multiplied kWh/hh by 2.0 | Removed; keep raw values to match GR |
| 10-min PV data | Used `P_GEN` column with `.resample("30min")` | Use hourly `(P_GEN_MAX + P_GEN_MIN) / 2` |
| Time grid mismatch | Used all 336 steps | `linspace(..., 336)[:-24][48:]` = 264 |
| Node count | 50 houses + 1 PCC = 51 | 49 houses + 1 PCC = 50 |
| Calendar-week sampling | Picked specific week with discrete 30-min bins | Mean-week profile per month + interp1d |

---

## 7. Quick-Start for Exp 3B

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sq2_data"))

from data_loader import build_microgrid, N_TIMESTEPS
from simplex import compute_simplex_trajectory

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "EXPORT HourlyData - Customer Endpoints.csv",
)

# Build one microgrid instance
P, t_seconds = build_microgrid(
    lcl_dir=LCL_DIR,
    pv_hourly_path=PV_PATH,
    season="summer",
    n_houses=49,
    penetration=49,
    seed=42,
)
# P.shape == (50, 264), t_seconds.shape == (264,)

# Compute simplex trajectory
traj = compute_simplex_trajectory(P)
# traj.shape == (264, 3) — columns: sigma_s, sigma_d, sigma_p
```
