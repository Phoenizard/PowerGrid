# Code Listing — Power Grid Dynamics (MATH3060)

> Reproducing and extending Smith et al. (2022) *Science Advances* 8, eabj6734.
> Swing-equation model on Watts–Strogatz microgrids with empirical LCL/PV load profiles.

---

## 1. Simplex Coordinate Computation (`simplex.py`)

```python
import numpy as np

def compute_simplex_coordinates(Pvec: np.ndarray) -> tuple[float, float, float]:
    n = len(Pvec)
    largest_source = np.max(Pvec)
    largest_sink = np.abs(np.min(Pvec))

    source_terms = Pvec[Pvec > 0.0]
    sink_terms = Pvec[Pvec < 0.0]

    if largest_source <= 0.0:
        sigma_s = 0.0
    else:
        sigma_s = np.sum(source_terms) / (n * largest_source)

    if largest_sink <= 0.0:
        sigma_d = 0.0
    else:
        sigma_d = np.sum(np.abs(sink_terms)) / (n * largest_sink)

    sigma_p = 1.0 - sigma_s - sigma_d
    return sigma_s, sigma_d, sigma_p


def compute_simplex_trajectory(P: np.ndarray) -> np.ndarray:
    T = P.shape[1]
    trajectory = np.zeros((T, 3))
    for t in range(T):
        Pvec = P[:, t]
        sigma_s, sigma_d, sigma_p = compute_simplex_coordinates(Pvec)
        trajectory[t] = [sigma_s, sigma_d, sigma_p]
    return trajectory
```

---

## 2. Network Generation (`network.py`)

```python
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def generate_network_with_pcc(
    n_houses: int = 50,
    k: int = 4,
    q: float = 0.1,
    n_pcc_links: int = 4,
    seed: int | None = None,
) -> csr_matrix:
    rng = np.random.default_rng(seed)
    ws_seed = int(rng.integers(0, 2**31)) if seed is not None else None
    G = nx.watts_strogatz_graph(n_houses, k, q, seed=ws_seed)

    A_ws = nx.adjacency_matrix(G).astype(np.float64)

    n_total = n_houses + 1
    A = lil_matrix((n_total, n_total), dtype=np.float64)
    A[:n_houses, :n_houses] = A_ws

    pcc_idx = n_houses
    pcc_neighbors = rng.choice(n_houses, size=n_pcc_links, replace=False)
    for j in pcc_neighbors:
        A[pcc_idx, j] = 1.0
        A[j, pcc_idx] = 1.0

    return A.tocsr()
```

---

## 3. Data Pipeline (`data_loader.py`)

```python
import glob
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy.interpolate import interp1d

_LCL_KWH_COL = "KWH/hh (per half hour) "

_T_RAW = np.linspace(0, 604800 - 1800, 336)[:-24]
_T_WEEK = _T_RAW[48:]
N_TIMESTEPS = len(_T_WEEK)


def _month_for_season(season: str) -> int:
    mapping = {"summer": 7, "winter": 1, "spring": 4, "autumn": 10}
    if season not in mapping:
        raise ValueError(f"Unsupported season: {season}")
    return mapping[season]


def _build_mean_week_profile(
    series_values: np.ndarray,
    series_datetimes: np.ndarray,
    target_month: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    dts = pd.to_datetime(series_datetimes)
    mask = dts.month == target_month
    if mask.sum() == 0:
        return None

    month_dts = dts[mask]
    month_vals = series_values[mask]

    dows = month_dts.dayofweek
    if len(set(dows)) < 7:
        return None

    artificial_dts = []
    for dt_val in month_dts:
        dow = dt_val.dayofweek
        artificial = dt_val.replace(year=1970, month=1, day=dow + 1)
        artificial_dts.append(artificial)
    artificial_dts = pd.DatetimeIndex(artificial_dts)

    df = pd.DataFrame({"val": month_vals, "dow": dows}, index=artificial_dts)
    df = df.sort_index()

    dates_out, means_out = [], []
    for dow in range(7):
        day_data = df[df["dow"] == dow]
        for ts in day_data.index.unique():
            vals = day_data.loc[ts, "val"]
            if hasattr(vals, "__len__"):
                means_out.append(np.mean(vals))
            else:
                means_out.append(float(vals))
            dates_out.append(ts)

    pairs = sorted(zip(dates_out, means_out), key=lambda x: x[0])
    dates_sorted = [p[0] for p in pairs]
    means_sorted = [p[1] for p in pairs]

    epoch = pd.Timestamp("1970-01-01")
    secs = np.array([(d - epoch).total_seconds() for d in dates_sorted])
    vals = np.array(means_sorted)
    return secs, vals


@lru_cache(maxsize=32)
def _read_lcl(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath, usecols=["LCLid", "DateTime", _LCL_KWH_COL])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df[_LCL_KWH_COL] = pd.to_numeric(df[_LCL_KWH_COL], errors="coerce")
    df = df.dropna(subset=[_LCL_KWH_COL])
    return df


@lru_cache(maxsize=1)
def _read_pv_hourly(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath,
                     usecols=["Substation", "datetime", "P_GEN_MAX", "P_GEN_MIN"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["P_GEN_MAX"] = pd.to_numeric(df["P_GEN_MAX"], errors="coerce")
    df["P_GEN_MIN"] = pd.to_numeric(df["P_GEN_MIN"], errors="coerce")
    df = df.dropna(subset=["P_GEN_MAX", "P_GEN_MIN"])
    return df


def make_consumption_interpolator(
    lcl_dir: str,
    target_month: int,
    rng: RandomState,
) -> interp1d | None:
    pattern = os.path.join(lcl_dir, "LCL-June2015v2_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No LCL files found: {pattern}")

    fpath = csv_files[rng.randint(0, len(csv_files))]
    df = _read_lcl(fpath)

    house_ids = list(df["LCLid"].unique())
    if not house_ids:
        return None
    house_id = house_ids[rng.randint(0, len(house_ids))]

    hh = df[df["LCLid"] == house_id]
    result = _build_mean_week_profile(
        hh[_LCL_KWH_COL].values, hh["DateTime"].values, target_month
    )
    if result is None or len(result[0]) < 2:
        return None

    secs, vals = result
    return interp1d(secs, vals, bounds_error=False, fill_value="extrapolate")


def make_pv_interpolator(
    pv_hourly_path: str,
    target_month: int,
    rng: RandomState,
) -> interp1d | None:
    df = _read_pv_hourly(pv_hourly_path)

    substations = list(df["Substation"].unique())
    if not substations:
        return None
    sub = substations[rng.randint(0, len(substations))]

    sub_df = df[df["Substation"] == sub]
    p_gen = (sub_df["P_GEN_MAX"].values + sub_df["P_GEN_MIN"].values) / 2.0

    result = _build_mean_week_profile(
        p_gen, sub_df["datetime"].values, target_month
    )
    if result is None or len(result[0]) < 2:
        return None

    secs, vals = result
    return interp1d(secs, vals, bounds_error=False, fill_value="extrapolate")


def build_microgrid(
    lcl_dir: str,
    pv_hourly_path: str,
    season: str = "summer",
    n_houses: int = 49,
    penetration: int = 49,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = RandomState(seed)
    target_month = _month_for_season(season)

    consumption_interps = []
    attempts = 0
    while len(consumption_interps) < n_houses:
        attempts += 1
        if attempts > n_houses * 20:
            raise RuntimeError("Could not build enough consumption profiles")
        interp = make_consumption_interpolator(lcl_dir, target_month, rng)
        if interp is not None:
            consumption_interps.append(interp)

    pv_interps = []
    attempts = 0
    while len(pv_interps) < penetration:
        attempts += 1
        if attempts > penetration * 20:
            raise RuntimeError("Could not build enough PV profiles")
        interp = make_pv_interpolator(pv_hourly_path, target_month, rng)
        if interp is not None:
            pv_interps.append(interp)

    n_total = n_houses + 1
    P = np.zeros((n_total, N_TIMESTEPS))

    for t_idx, t_sec in enumerate(_T_WEEK):
        for i in range(n_houses):
            consumption = consumption_interps[i](t_sec)
            if i < penetration:
                P[i, t_idx] = pv_interps[i](t_sec) - consumption
            else:
                P[i, t_idx] = -consumption
        P[n_houses, t_idx] = -np.sum(P[:n_houses, t_idx])

    return P, _T_WEEK.copy()


def evaluate_power_vector(
    consumption_interps,
    pv_interps,
    t_seconds: float,
) -> np.ndarray:
    n_houses = len(consumption_interps)
    penetration = len(pv_interps)
    P = np.zeros(n_houses + 1)

    for i in range(n_houses):
        consumption = float(consumption_interps[i](t_seconds))
        if i < penetration:
            P[i] = float(pv_interps[i](t_seconds)) - consumption
        else:
            P[i] = -consumption

    P[n_houses] = -np.sum(P[:n_houses])
    return P
```

---

## 4. Swing Equation & Critical Coupling (`model.py`)

```python
import numpy as np
from scipy.integrate import solve_ivp


def swing_equation_rhs(t, state, P, kappa, A_data, A_indices, A_indptr, n, gamma):
    theta = state[:n]
    omega = state[n:]

    dtheta_dt = omega
    domega_dt = np.zeros(n)

    for i in range(n):
        coupling_sum = 0.0
        for idx in range(A_indptr[i], A_indptr[i + 1]):
            j = A_indices[idx]
            coupling_sum += A_data[idx] * np.sin(theta[i] - theta[j])
        domega_dt[i] = P[i] - gamma * omega[i] - kappa * coupling_sum

    result = np.empty(2 * n)
    result[:n] = dtheta_dt
    result[n:] = domega_dt
    return result


def swing_equation_wrapper(t, state, P, kappa, A_csr, gamma):
    n = len(P)
    return swing_equation_rhs(
        t, state, P, kappa,
        A_csr.data, A_csr.indices, A_csr.indptr, n, gamma
    )


def check_convergence(state, n, tol=1e-4):
    omega = state[n:]
    return np.max(np.abs(omega)) < tol


def compute_kappa_c(A_csr, P, config_params=None):
    if config_params is None:
        config_params = {}

    n = len(P)
    gamma = config_params.get('gamma', 1.0)
    kappa_min, kappa_max = config_params.get('kappa_range', (0.001, 50.0))
    bisection_steps = config_params.get('bisection_steps', 20)
    t_integrate = config_params.get('t_integrate', 100)
    conv_tol = config_params.get('conv_tol', 1e-3)
    max_step = config_params.get('max_step', 1.0)

    rng = np.random.default_rng()
    theta0 = rng.uniform(-0.1, 0.1, n)
    omega0 = np.zeros(n)
    state0 = np.concatenate([theta0, omega0])

    def is_stable(kappa):
        sol = solve_ivp(
            swing_equation_wrapper,
            [0, t_integrate],
            state0,
            args=(P, kappa, A_csr, gamma),
            method='RK45',
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol.success:
            return check_convergence(sol.y[:, -1], n, conv_tol)
        return False

    for _ in range(bisection_steps):
        kappa_mid = (kappa_min + kappa_max) / 2
        if is_stable(kappa_mid):
            kappa_max = kappa_mid
        else:
            kappa_min = kappa_mid

    return (kappa_min + kappa_max) / 2
```

---

## 5. Experiment 3A — Simplex Trajectory (`run_trajectory.py`)

```python
import numpy as np
from data_loader import build_microgrid, N_TIMESTEPS
from simplex import compute_simplex_trajectory

N_HOUSES = 49
PENETRATION = 49


def run_experiment_3a(
    lcl_dir, pv_hourly_path,
    season="summer", n_ensemble=50, seed=20260209,
):
    rng = np.random.default_rng(seed)
    all_trajectories = []

    for i in range(n_ensemble):
        instance_seed = int(rng.integers(0, 2**31))

        P, t_seconds = build_microgrid(
            lcl_dir=lcl_dir,
            pv_hourly_path=pv_hourly_path,
            season=season,
            n_houses=N_HOUSES,
            penetration=PENETRATION,
            seed=instance_seed,
        )

        assert np.all(np.abs(P.sum(axis=0)) < 1e-10), "Power balance violated"

        traj = compute_simplex_trajectory(P)
        all_trajectories.append(traj)

    stacked = np.array(all_trajectories)
    mean_traj = stacked.mean(axis=0)
    std_traj = stacked.std(axis=0)
    hours = t_seconds / 3600.0

    return mean_traj, std_traj, hours
```

---

## 6. Experiment 3B — κ_c Time Series (`run_kappa_timeseries.py`)

```python
import numpy as np
from data_loader import (
    build_microgrid_interpolators,
    evaluate_power_vector,
    compute_pmax_from_interpolators,
)
from network import generate_network_with_pcc
from model import compute_kappa_c

N_HOUSES = 49
PENETRATION = 49
HOURS_PER_DAY = [0, 3, 6, 9, 12, 15, 18, 21]
N_DAYS = 7

PROD_CONFIG = {
    "gamma": 1.0,
    "kappa_range": (0.001, 50.0),
    "bisection_steps": 20,
    "t_integrate": 100,
    "conv_tol": 1e-3,
    "max_step": 1.0,
}


def get_time_points():
    points = []
    for day in range(N_DAYS):
        for hour in HOURS_PER_DAY:
            t_sec = day * 86400 + hour * 3600
            points.append((day, hour, t_sec))
    return points


def run_experiment_3b(
    lcl_dir, pv_hourly_path,
    season="summer", n_ensemble=50, seed=20260209,
    sim_config=None,
):
    if sim_config is None:
        sim_config = PROD_CONFIG

    rng = np.random.default_rng(seed)
    time_points = get_time_points()

    ensemble = []
    for i in range(n_ensemble):
        instance_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))

        cons_interps, pv_interps = build_microgrid_interpolators(
            lcl_dir=lcl_dir,
            pv_hourly_path=pv_hourly_path,
            season=season,
            n_houses=N_HOUSES,
            penetration=PENETRATION,
            seed=instance_seed,
        )
        P_max = compute_pmax_from_interpolators(cons_interps, pv_interps)
        A = generate_network_with_pcc(
            n_houses=N_HOUSES, k=4, q=0.1, n_pcc_links=4, seed=net_seed
        )
        ensemble.append((cons_interps, pv_interps, A, P_max))

    results = []
    for day, hour, t_sec in time_points:
        kappa_values = []
        for cons_interps, pv_interps, A, P_max in ensemble:
            if P_max < 1e-12:
                continue
            P_t = evaluate_power_vector(cons_interps, pv_interps, t_sec)
            kc = compute_kappa_c(A, P_t, config_params=sim_config)
            kappa_values.append(kc / P_max)

        kv = np.array(kappa_values)
        results.append({
            "day": day,
            "hour": hour,
            "kappa_c_mean": float(np.mean(kv)),
            "kappa_c_std": float(np.std(kv)),
            "n_ensemble": len(kv),
        })

    return results
```
