"""
Data loading and preprocessing for SQ2 experiments.

Replicates GridResilience's data pipeline exactly:
  - LCL consumption: random CSV file → random house → mean-week profile
    for target month → interp1d interpolator (time in seconds)
  - PV generation: hourly CSV → (P_GEN_MAX + P_GEN_MIN)/2 → random
    substation → mean-week profile → interp1d interpolator
  - Net power = PV_interpolator(t) - consumption_interpolator(t)
  - 49 houses + 1 PCC = 50 nodes (matching GridResilience n=50)
  - Time sampling: np.linspace(0, 604800-1800, 336) seconds
  - Skip first 48 steps → tweek_sample = t[48:] → 264 timesteps

Reference: GridResilience/scripts/powerreader.py, powerclasses.py,
           powerexperiments.py
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy.interpolate import interp1d

# LCL column name (note trailing space in original CSV)
_LCL_KWH_COL = "KWH/hh (per half hour) "

# GridResilience time grid:
# t = np.linspace(0, 604800-1800, 336)[:-24]  → 312 steps (drops last 24)
# tweek_sample = t[48:]  → skip first 48 steps (24h) → 264 steps
_T_RAW = np.linspace(0, 604800 - 1800, 336)[:-24]  # 312 elements
_T_WEEK = _T_RAW[48:]  # 264 timesteps (skip first day)
N_TIMESTEPS = len(_T_WEEK)  # 264


def _month_for_season(season: str) -> int:
    """Return the representative month for GridResilience experiments."""
    mapping = {"summer": 7, "winter": 1, "spring": 4, "autumn": 10}
    if season not in mapping:
        raise ValueError(f"Unsupported season: {season}")
    return mapping[season]


def _build_mean_week_profile(
    series_values: np.ndarray,
    series_datetimes: np.ndarray,
    target_month: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build a mean-week profile from time series data for a given month.

    Replicates GridResilience's make_random_week_profiles() logic:
      1. Filter for target month
      2. Assign artificial dates: replace year=1970, month=1, day=day_of_week+1
      3. For each day-of-week, average all same-timestamp readings
      4. Sort by time, convert to seconds axis

    Returns (seconds_array, values_array) or None if insufficient data.
    """
    dts = pd.to_datetime(series_datetimes)
    mask = dts.month == target_month
    if mask.sum() == 0:
        return None

    month_dts = dts[mask]
    month_vals = series_values[mask]

    dows = month_dts.dayofweek
    if len(set(dows)) < 7:
        return None

    # Build artificial timestamps: replace to 1970-01-{dow+1}
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


# ---------------------------------------------------------------------------
# Cached data readers — Parquet preferred (10x faster), CSV fallback
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _read_lcl(fpath: str) -> pd.DataFrame:
    """Read an LCL data file. Prefers .parquet sibling if it exists."""
    # Check for parquet version: LCL_parquet/ directory alongside LCL/
    parquet_dir = os.path.join(os.path.dirname(fpath), "..", "LCL_parquet")
    basename = os.path.splitext(os.path.basename(fpath))[0]
    parquet_path = os.path.join(parquet_dir, f"{basename}.parquet")

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    # Fallback to CSV
    df = pd.read_csv(fpath, usecols=["LCLid", "DateTime", _LCL_KWH_COL])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df[_LCL_KWH_COL] = pd.to_numeric(df[_LCL_KWH_COL], errors="coerce")
    df = df.dropna(subset=[_LCL_KWH_COL])
    return df


@lru_cache(maxsize=1)
def _read_pv_hourly(fpath: str) -> pd.DataFrame:
    """Read PV hourly data. Prefers .parquet sibling if it exists."""
    parquet_path = os.path.join(
        os.path.dirname(fpath), "pv_hourly_customer_endpoints.parquet"
    )
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    # Fallback to CSV
    df = pd.read_csv(fpath,
                     usecols=["Substation", "datetime", "P_GEN_MAX", "P_GEN_MIN"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["P_GEN_MAX"] = pd.to_numeric(df["P_GEN_MAX"], errors="coerce")
    df["P_GEN_MIN"] = pd.to_numeric(df["P_GEN_MIN"], errors="coerce")
    df = df.dropna(subset=["P_GEN_MAX", "P_GEN_MIN"])
    return df


# ---------------------------------------------------------------------------
# Interpolator factories
# ---------------------------------------------------------------------------

def make_consumption_interpolator(
    lcl_dir: str,
    target_month: int,
    rng: RandomState,
) -> interp1d | None:
    """
    Create a consumption interpolator for one random household.

    Replicates GridResilience make_random_week_profiles():
      1. Pick a random LCL CSV file
      2. Pick a random household from that file
      3. Build mean-week profile for target month
      4. Return interp1d(seconds, power_values)
    """
    pattern = os.path.join(lcl_dir, "LCL-June2015v2_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No LCL files found: {pattern}")

    file_idx = rng.randint(0, len(csv_files))
    fpath = csv_files[file_idx]

    df = _read_lcl(fpath)

    house_ids = list(df["LCLid"].unique())
    if not house_ids:
        return None
    house_idx = rng.randint(0, len(house_ids))
    house_id = house_ids[house_idx]

    hh = df[df["LCLid"] == house_id]

    result = _build_mean_week_profile(
        hh[_LCL_KWH_COL].values,
        hh["DateTime"].values,
        target_month,
    )
    if result is None:
        return None

    secs, vals = result
    if len(secs) < 2:
        return None

    return interp1d(secs, vals, bounds_error=False, fill_value="extrapolate")


def make_pv_interpolator(
    pv_hourly_path: str,
    target_month: int,
    rng: RandomState,
) -> interp1d | None:
    """
    Create a PV generation interpolator for one random substation.

    Replicates GridResilience make_random_week_profiles_PV():
      1. Read hourly PV data
      2. Compute P_GEN = (P_GEN_MAX + P_GEN_MIN) / 2
      3. Pick a random substation
      4. Build mean-week profile for target month
      5. Return interp1d(seconds, generation_values)
    """
    df = _read_pv_hourly(pv_hourly_path)

    substations = list(df["Substation"].unique())
    if not substations:
        return None
    sub_idx = rng.randint(0, len(substations))
    sub = substations[sub_idx]

    sub_df = df[df["Substation"] == sub]
    p_gen = (sub_df["P_GEN_MAX"].values + sub_df["P_GEN_MIN"].values) / 2.0

    result = _build_mean_week_profile(
        p_gen,
        sub_df["datetime"].values,
        target_month,
    )
    if result is None:
        return None

    secs, vals = result
    if len(secs) < 2:
        return None

    return interp1d(secs, vals, bounds_error=False, fill_value="extrapolate")


def build_microgrid(
    lcl_dir: str,
    pv_hourly_path: str,
    season: str = "summer",
    n_houses: int = 49,
    penetration: int = 49,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a microgrid and compute net power at each timestep.

    Replicates GridResilience MicroGrid exactly:
      - n_houses consumption interpolators (all houses consume)
      - penetration PV interpolators (first `penetration` houses have PV)
      - Net power per house: PV(t) - consumption(t), or -consumption(t) if no PV
      - PCC = -sum(house powers)
      - Total nodes = n_houses + 1

    Returns
    -------
    P : ndarray, shape (n_houses + 1, N_TIMESTEPS)
    t_seconds : ndarray, shape (N_TIMESTEPS,)
    """
    rng = RandomState(seed)
    target_month = _month_for_season(season)

    # Build consumption interpolators for all houses
    consumption_interps: list[interp1d] = []
    attempts = 0
    while len(consumption_interps) < n_houses:
        attempts += 1
        if attempts > n_houses * 20:
            raise RuntimeError(
                f"Could not build {n_houses} consumption profiles "
                f"(got {len(consumption_interps)} after {attempts} attempts)"
            )
        interp = make_consumption_interpolator(lcl_dir, target_month, rng)
        if interp is not None:
            consumption_interps.append(interp)

    # Build PV interpolators for first `penetration` houses
    pv_interps: list[interp1d] = []
    attempts = 0
    while len(pv_interps) < penetration:
        attempts += 1
        if attempts > penetration * 20:
            raise RuntimeError(
                f"Could not build {penetration} PV profiles "
                f"(got {len(pv_interps)} after {attempts} attempts)"
            )
        interp = make_pv_interpolator(pv_hourly_path, target_month, rng)
        if interp is not None:
            pv_interps.append(interp)

    # Compute power vectors at each timestep
    n_total = n_houses + 1
    P = np.zeros((n_total, N_TIMESTEPS))

    for t_idx, t_sec in enumerate(_T_WEEK):
        for i in range(n_houses):
            consumption = consumption_interps[i](t_sec)
            if i < penetration:
                generation = pv_interps[i](t_sec)
                P[i, t_idx] = generation - consumption
            else:
                P[i, t_idx] = -consumption
        P[n_houses, t_idx] = -np.sum(P[:n_houses, t_idx])

    return P, _T_WEEK.copy()


def build_microgrid_interpolators(
    lcl_dir: str,
    pv_hourly_path: str,
    season: str = "summer",
    n_houses: int = 49,
    penetration: int = 49,
    seed: int | None = None,
) -> tuple[list[interp1d], list[interp1d]]:
    """
    Build consumption + PV interpolators without evaluating at a fixed grid.

    Same random sampling logic as build_microgrid(), but returns the raw
    interpolator objects so callers can evaluate at arbitrary times.

    Returns
    -------
    consumption_interps : list of interp1d, length n_houses
    pv_interps : list of interp1d, length penetration
    """
    rng = RandomState(seed)
    target_month = _month_for_season(season)

    consumption_interps: list[interp1d] = []
    attempts = 0
    while len(consumption_interps) < n_houses:
        attempts += 1
        if attempts > n_houses * 20:
            raise RuntimeError(
                f"Could not build {n_houses} consumption profiles "
                f"(got {len(consumption_interps)} after {attempts} attempts)"
            )
        interp = make_consumption_interpolator(lcl_dir, target_month, rng)
        if interp is not None:
            consumption_interps.append(interp)

    pv_interps: list[interp1d] = []
    attempts = 0
    while len(pv_interps) < penetration:
        attempts += 1
        if attempts > penetration * 20:
            raise RuntimeError(
                f"Could not build {penetration} PV profiles "
                f"(got {len(pv_interps)} after {attempts} attempts)"
            )
        interp = make_pv_interpolator(pv_hourly_path, target_month, rng)
        if interp is not None:
            pv_interps.append(interp)

    return consumption_interps, pv_interps


def evaluate_power_vector(
    consumption_interps: list[interp1d],
    pv_interps: list[interp1d],
    t_seconds: float,
) -> np.ndarray:
    """
    Evaluate net power P at a single time point.

    Parameters
    ----------
    consumption_interps : list of interp1d, length n_houses
    pv_interps : list of interp1d, length penetration (≤ n_houses)
    t_seconds : float, time in seconds within [0, 604800]

    Returns
    -------
    P : ndarray, shape (n_houses + 1,)
        Net power for each house node + PCC.
    """
    n_houses = len(consumption_interps)
    penetration = len(pv_interps)
    n_total = n_houses + 1
    P = np.zeros(n_total)

    for i in range(n_houses):
        consumption = float(consumption_interps[i](t_seconds))
        if i < penetration:
            generation = float(pv_interps[i](t_seconds))
            P[i] = generation - consumption
        else:
            P[i] = -consumption

    # PCC balances the grid
    P[n_houses] = -np.sum(P[:n_houses])
    return P


def compute_pmax_from_interpolators(
    consumption_interps: Sequence[interp1d],
    pv_interps: Sequence[interp1d],
) -> float:
    """
    Compute P_max = max(|P_house|) over the 264-step time grid.

    Consistent with SQ2-A: evaluates at _T_WEEK (same grid as build_microgrid),
    uses only house nodes (excludes PCC).
    """
    n_houses = len(consumption_interps)
    penetration = len(pv_interps)
    P_house = np.zeros((n_houses, N_TIMESTEPS))

    for t_idx, t_sec in enumerate(_T_WEEK):
        for i in range(n_houses):
            consumption = float(consumption_interps[i](t_sec))
            if i < penetration:
                generation = float(pv_interps[i](t_sec))
                P_house[i, t_idx] = generation - consumption
            else:
                P_house[i, t_idx] = -consumption

    return float(np.max(np.abs(P_house)))
