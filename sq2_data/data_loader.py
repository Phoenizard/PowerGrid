"""
Data loading and preprocessing for SQ2 experiments.

Loads London LCL household electricity consumption and PV generation data,
aligns them by season (summer = Jun-Aug), and computes net power vectors
for a 51-node microgrid (50 houses + 1 PCC).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# LCL column name (note trailing space in original CSV)
_LCL_KWH_COL = "KWH/hh (per half hour) "
_SUMMER_MONTHS = {6, 7, 8}
_STEPS_PER_WEEK = 48 * 7  # 336 half-hour steps


def _season_months(season: str) -> set[int]:
    if season == "summer":
        return _SUMMER_MONTHS
    raise ValueError(f"Unsupported season: {season}")


def load_lcl_households(
    data_dir: str,
    season: str = "summer",
    n_households: int = 50,
    seed: int | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load LCL household consumption data for one week in the given season.

    Parameters
    ----------
    data_dir : str
        Path to directory containing LCL-June2015v2_*.csv files.
    season : str
        Season to extract ("summer").
    n_households : int
        Number of households to sample.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    consumption : ndarray, shape (n_households, 336)
        Power consumption in kW (KWH/hh * 2).
    time_index : pd.DatetimeIndex, length 336
        Half-hourly timestamps for the selected week.
    """
    rng = np.random.default_rng(seed)
    months = _season_months(season)

    # Discover all LCL CSV files
    pattern = os.path.join(data_dir, "LCL-June2015v2_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No LCL files found matching {pattern}")

    # Phase 1: collect all (household_id, file_path) pairs that have summer data
    # We read files one at a time to avoid memory issues
    household_file_map: dict[str, str] = {}  # LCLid -> file_path (first file found)

    for fpath in csv_files:
        df = pd.read_csv(fpath, usecols=["LCLid", "DateTime", _LCL_KWH_COL])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        summer_mask = df["DateTime"].dt.month.isin(months)
        summer_ids = df.loc[summer_mask, "LCLid"].unique()
        for hid in summer_ids:
            if hid not in household_file_map:
                household_file_map[hid] = fpath

        if len(household_file_map) >= n_households * 3:
            # Enough candidates collected
            break

    all_hids = list(household_file_map.keys())
    if len(all_hids) < n_households:
        raise ValueError(
            f"Only found {len(all_hids)} households with summer data, "
            f"need {n_households}"
        )

    # Sample households
    chosen_idx = rng.choice(len(all_hids), size=n_households, replace=False)
    chosen_hids = [all_hids[i] for i in chosen_idx]

    # Phase 2: load data for chosen households, pick a common summer week
    # Group by file to minimize re-reads
    file_to_hids: dict[str, list[str]] = {}
    for hid in chosen_hids:
        fpath = household_file_map[hid]
        file_to_hids.setdefault(fpath, []).append(hid)

    household_series: dict[str, pd.Series] = {}

    for fpath, hids in file_to_hids.items():
        df = pd.read_csv(fpath, usecols=["LCLid", "DateTime", _LCL_KWH_COL])
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df[df["DateTime"].dt.month.isin(months)]

        # Replace "Null" strings with NaN
        df[_LCL_KWH_COL] = pd.to_numeric(df[_LCL_KWH_COL], errors="coerce")

        for hid in hids:
            hh_data = df[df["LCLid"] == hid].copy()
            hh_data = hh_data.drop_duplicates(subset="DateTime").sort_values("DateTime")
            hh_data = hh_data.set_index("DateTime")[_LCL_KWH_COL]
            # Interpolate NaN values
            hh_data = hh_data.interpolate(method="linear").ffill().bfill()
            household_series[hid] = hh_data

    # Find a common week where most households have complete data
    # Use the year with most data (typically 2013 for summer)
    ref_series = household_series[chosen_hids[0]]
    years = ref_series.index.year.unique()
    # Prefer 2013 (most LCL summer data)
    target_year = 2013 if 2013 in years else int(years[0])

    # Find all possible Monday starts in summer of target_year
    summer_start = pd.Timestamp(f"{target_year}-06-01")
    summer_end = pd.Timestamp(f"{target_year}-08-31 23:30:00")

    # Generate candidate week starts (every Monday in summer)
    candidates = pd.date_range(summer_start, summer_end - pd.Timedelta(days=7), freq="W-MON")
    if len(candidates) == 0:
        # Fallback: any day
        candidates = pd.date_range(summer_start, summer_end - pd.Timedelta(days=7), freq="D")

    # Pick a random week start
    week_start_idx = rng.integers(0, len(candidates))
    week_start = candidates[week_start_idx]
    week_end = week_start + pd.Timedelta(days=7) - pd.Timedelta(minutes=30)

    time_index = pd.date_range(week_start, week_end, freq="30min")
    assert len(time_index) == _STEPS_PER_WEEK, (
        f"Expected {_STEPS_PER_WEEK} steps, got {len(time_index)}"
    )

    # Extract each household's data for this week
    consumption = np.zeros((n_households, _STEPS_PER_WEEK))
    for i, hid in enumerate(chosen_hids):
        series = household_series[hid]
        # Reindex to the target week, interpolate gaps
        week_data = series.reindex(time_index)
        week_data = week_data.interpolate(method="linear").ffill().bfill()
        # If still NaN (household has no data this week), use the household's mean
        if week_data.isna().any():
            week_data = week_data.fillna(series.mean())
        # Convert KWH/hh to kW: multiply by 2 (0.5h interval → kW)
        consumption[i, :] = week_data.values * 2.0

    return consumption, time_index


def load_pv_generation(
    data_path: str,
    season: str = "summer",
    n_panels: int = 50,
    seed: int | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load PV generation data for one week in the given season.

    Parameters
    ----------
    data_path : str
        Path to the PV CSV file.
    season : str
        Season to extract ("summer").
    n_panels : int
        Number of PV panels (households) to generate data for.
    seed : int or None
        Random seed.

    Returns
    -------
    generation : ndarray, shape (n_panels, 336)
        PV generation in kW (clipped to >= 0).
    time_index : pd.DatetimeIndex, length 336
        Half-hourly timestamps for the selected week.
    """
    rng = np.random.default_rng(seed)
    months = _season_months(season)

    df = pd.read_csv(data_path, usecols=["Substation", "datetime", "P_GEN"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"].dt.month.isin(months)]

    # Clip negative P_GEN to 0
    df["P_GEN"] = df["P_GEN"].clip(lower=0.0)

    substations = df["Substation"].unique()

    # Find a week with good coverage across substations
    # Use 2014 (only PV year)
    summer_start = pd.Timestamp("2014-06-10")  # PV data starts June 10
    summer_end = pd.Timestamp("2014-08-31 23:50:00")

    # Pick a random Monday in summer
    candidates = pd.date_range(summer_start, summer_end - pd.Timedelta(days=7), freq="W-MON")
    if len(candidates) == 0:
        candidates = pd.date_range(summer_start, summer_end - pd.Timedelta(days=7), freq="D")

    week_start_idx = rng.integers(0, len(candidates))
    week_start = candidates[week_start_idx]
    week_end = week_start + pd.Timedelta(days=7) - pd.Timedelta(minutes=30)

    time_index_30min = pd.date_range(week_start, week_end, freq="30min")
    assert len(time_index_30min) == _STEPS_PER_WEEK

    # Extract and resample each substation's data for the week
    substation_profiles: list[np.ndarray] = []

    for sub in substations:
        sub_data = df[df["Substation"] == sub].copy()
        sub_data = sub_data.drop_duplicates(subset="datetime").sort_values("datetime")
        sub_data = sub_data.set_index("datetime")["P_GEN"]

        # Resample 10-min to 30-min by taking mean
        week_mask = (sub_data.index >= week_start) & (
            sub_data.index <= week_end + pd.Timedelta(minutes=20)
        )
        week_10min = sub_data[week_mask]

        if len(week_10min) < 10:
            continue

        # Resample to 30-min
        week_30min = week_10min.resample("30min").mean()
        week_30min = week_30min.reindex(time_index_30min)
        week_30min = week_30min.interpolate(method="linear").ffill().bfill().fillna(0.0)
        substation_profiles.append(week_30min.values)

    if not substation_profiles:
        raise ValueError("No PV substation data available for the selected week")

    # Sample n_panels profiles with replacement from available substations
    profiles_array = np.array(substation_profiles)  # (n_subs, 336)
    chosen = rng.choice(len(profiles_array), size=n_panels, replace=True)
    generation = profiles_array[chosen]  # (n_panels, 336)

    # Clip to non-negative (safety)
    generation = np.clip(generation, 0.0, None)

    return generation, time_index_30min


def compute_net_power(
    consumption: np.ndarray,
    generation: np.ndarray,
) -> np.ndarray:
    """
    Compute net power for each node including PCC.

    P_i(t) = g_i(t) - c_i(t)  for houses (i = 0..49)
    P_PCC(t) = -sum(P_i(t))   for PCC node (i = 50)

    Parameters
    ----------
    consumption : ndarray, shape (n_houses, T)
    generation : ndarray, shape (n_houses, T)

    Returns
    -------
    P : ndarray, shape (n_houses + 1, T)
        Net power for 50 houses + 1 PCC node.
    """
    n_houses, T = consumption.shape
    assert generation.shape == (n_houses, T)

    P_houses = generation - consumption  # (n_houses, T)
    P_pcc = -P_houses.sum(axis=0, keepdims=True)  # (1, T)
    P = np.vstack([P_houses, P_pcc])  # (n_houses + 1, T)

    return P
