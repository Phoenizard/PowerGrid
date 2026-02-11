"""
Shared infrastructure for SQ4 experiments.

Extracted from sq2_data/run_kappa_timeseries.py — provides ensemble preloading
and 56-point κ_c time series computation with optional network modification.

RNG invariant: The main rng draws (instance_seed, net_seed) in the same order
as SQ2-B. Edge selection uses a separate rng derived from net_seed + 1_000_000.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.random import default_rng
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")

for _d in (SQ2_DIR, PAPER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from model import compute_kappa_c  # noqa: E402
from data_loader import (  # noqa: E402
    build_microgrid_interpolators,
    evaluate_power_vector,
    compute_pmax_from_interpolators,
)
from network import generate_network_with_pcc  # noqa: E402

# --- Paths (same as SQ2) ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
PV_HOURLY_PATH = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "EXPORT HourlyData - Customer Endpoints.csv",
)

# --- Constants ---
N_HOUSES = 49
PENETRATION = 49
GAMMA = 1.0
HOURS_PER_DAY = [0, 3, 6, 9, 12, 15, 18, 21]
N_DAYS = 7

FAST_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 50.0),
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 1.0,
}

PROD_CONFIG = {
    "gamma": GAMMA,
    "kappa_range": (0.001, 50.0),
    "bisection_steps": 20,
    "t_integrate": 100,
    "conv_tol": 1e-3,
    "max_step": 1.0,
}


@dataclass
class EnsembleInstance:
    """One ensemble member: interpolators + base network + P_max + seeds."""
    cons_interps: list
    pv_interps: list
    A_base: csr_matrix
    P_max: float
    instance_seed: int
    net_seed: int


# Type alias for network modifier function
NetworkModifier = Callable[[csr_matrix, EnsembleInstance], csr_matrix]


def get_time_points() -> list[tuple[int, int, float]]:
    """Generate 56 representative time points: (day, hour, t_seconds)."""
    points = []
    for day in range(N_DAYS):
        for hour in HOURS_PER_DAY:
            t_sec = day * 86400 + hour * 3600
            points.append((day, hour, t_sec))
    return points


def ensure_output_csv(path: str):
    """Create output CSV with header if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day", "hour", "kappa_c_mean", "kappa_c_std"])


def load_completed_points(path: str) -> set[tuple[int, int]]:
    """Load already-computed (day, hour) pairs from a checkpoint CSV."""
    if not os.path.exists(path):
        return set()
    completed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((int(row["day"]), int(row["hour"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def append_row(path: str, row: list):
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def preload_ensemble(
    n_ensemble: int,
    season: str,
    seed: int,
    q: float = 0.1,
    n_houses: int = N_HOUSES,
    penetration: int = PENETRATION,
) -> list[EnsembleInstance | None]:
    """
    Pre-load ensemble: interpolators + network + P_max for each instance.

    RNG flow matches SQ2-B exactly:
      rng = default_rng(seed)
      for each instance:
        instance_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))

    Parameters
    ----------
    n_ensemble : int
        Number of ensemble members.
    season : str
        Season name ("summer", "winter", etc.).
    seed : int
        Master RNG seed.
    q : float
        WS rewiring probability (parameterized for SQ4 q-sweep).
    n_houses : int
        Number of house nodes.
    penetration : int
        Number of houses with PV.

    Returns
    -------
    list of EnsembleInstance or None (if loading failed)
    """
    rng = default_rng(seed)
    ensemble: list[EnsembleInstance | None] = []
    n_loaded = 0
    t_start = time.time()

    for i in range(n_ensemble):
        instance_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))

        try:
            t0 = time.time()
            cons_interps, pv_interps = build_microgrid_interpolators(
                lcl_dir=LCL_DIR,
                pv_hourly_path=PV_HOURLY_PATH,
                season=season,
                n_houses=n_houses,
                penetration=penetration,
                seed=instance_seed,
            )

            P_max_i = compute_pmax_from_interpolators(cons_interps, pv_interps)

            # Verify power balance
            P_test = evaluate_power_vector(cons_interps, pv_interps, 86400.0)
            balance = abs(np.sum(P_test))
            if balance > 1e-10:
                print(f"  WARN instance {i+1}: power balance = {balance:.2e}")

            A = generate_network_with_pcc(
                n_houses=n_houses, k=4, q=q, n_pcc_links=4, seed=net_seed
            )

            inst = EnsembleInstance(
                cons_interps=cons_interps,
                pv_interps=pv_interps,
                A_base=A,
                P_max=P_max_i,
                instance_seed=instance_seed,
                net_seed=net_seed,
            )
            ensemble.append(inst)
            n_loaded += 1
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_ensemble}] loaded ({elapsed:.1f}s, P_max={P_max_i:.4f})")
            sys.stdout.flush()

        except Exception as exc:
            print(f"  WARN: Failed to load instance {i+1}: {exc}")
            traceback.print_exc()
            ensemble.append(None)

    t_total = time.time() - t_start
    print(f"Pre-loading done: {n_loaded}/{n_ensemble} in {t_total:.1f}s\n")
    sys.stdout.flush()

    if n_loaded == 0:
        raise RuntimeError("No ensemble instances loaded. Aborting.")

    return ensemble


def run_timeseries(
    ensemble: list[EnsembleInstance | None],
    sim_config: dict,
    output_path: str,
    network_modifier: NetworkModifier | None = None,
    clean: bool = False,
) -> list[dict]:
    """
    Run 56-point κ_c time series with optional network modification.

    Parameters
    ----------
    ensemble : list of EnsembleInstance or None
    sim_config : dict
        Bisection config (FAST_CONFIG or PROD_CONFIG).
    output_path : str
        Path to output CSV.
    network_modifier : callable or None
        If provided, called as modifier(A_base.copy(), instance) -> csr_matrix.
        Applied per-instance before bisection.
    clean : bool
        If True, delete existing output before starting.

    Returns
    -------
    list of dicts with keys: day, hour, kc_mean, kc_std
    """
    if clean and os.path.exists(output_path):
        os.remove(output_path)

    ensure_output_csv(output_path)
    completed = load_completed_points(output_path)
    time_points = get_time_points()
    remaining = [(d, h, t) for d, h, t in time_points if (d, h) not in completed]

    n_ensemble = len(ensemble)
    print(f"  Time points: {len(time_points)} total, {len(completed)} done, {len(remaining)} remaining")
    sys.stdout.flush()

    # Pre-compute modified adjacency matrices if modifier provided
    A_per_instance: list[csr_matrix | None] = []
    if network_modifier is not None:
        print("  Pre-computing modified networks...")
        for inst in ensemble:
            if inst is None:
                A_per_instance.append(None)
            else:
                A_mod = network_modifier(inst.A_base.copy(), inst)
                assert isinstance(A_mod, csr_matrix), f"modifier must return csr_matrix, got {type(A_mod)}"
                A_per_instance.append(A_mod)
    else:
        A_per_instance = [inst.A_base if inst is not None else None for inst in ensemble]

    results = []
    n_done = 0
    t_bisect_start = time.time()

    for day, hour, t_sec in time_points:
        if (day, hour) in completed:
            n_done += 1
            continue

        t_point_start = time.time()
        kappa_values = []

        for i in range(n_ensemble):
            inst = ensemble[i]
            A = A_per_instance[i]
            if inst is None or A is None:
                continue
            if inst.P_max < 1e-12:
                continue

            try:
                P_t = evaluate_power_vector(inst.cons_interps, inst.pv_interps, t_sec)
                balance = abs(np.sum(P_t))
                if balance > 1e-6:
                    print(f"  WARN: power balance = {balance:.2e} at day={day} h={hour} inst={i}")

                kc = compute_kappa_c(A, P_t, config_params=sim_config)
                kc_normalized = kc / inst.P_max
                kappa_values.append(kc_normalized)

            except Exception as exc:
                print(f"  WARN instance {i+1} failed at day={day} h={hour}: {exc}")
                continue

        if not kappa_values:
            print(f"  ERROR: No valid samples for day={day} hour={hour:02d}")
            sys.stdout.flush()
            continue

        kv = np.array(kappa_values)
        mean_kc = float(np.mean(kv))
        std_kc = float(np.std(kv))
        n_done += 1
        elapsed = time.time() - t_point_start

        append_row(output_path, [day, hour, f"{mean_kc:.6f}", f"{std_kc:.6f}"])
        results.append({"day": day, "hour": hour, "kc_mean": mean_kc, "kc_std": std_kc})

        # ETA
        total_elapsed = time.time() - t_bisect_start
        remaining_pts = len(time_points) - n_done
        if n_done > len(completed):
            pts_computed = n_done - len(completed)
            avg_per_pt = total_elapsed / pts_computed
            eta_s = avg_per_pt * remaining_pts
            eta_str = f"ETA {eta_s/60:.1f}min"
        else:
            eta_str = ""

        print(
            f"  [{n_done}/{len(time_points)}] Day {day} {hour:02d}:00 -- "
            f"kc_mean={mean_kc:.4f} +/- {std_kc:.4f} "
            f"({len(kappa_values)} valid, {elapsed:.1f}s) {eta_str}"
        )
        sys.stdout.flush()

    total_time = time.time() - t_bisect_start
    print(f"  Timeseries complete in {total_time/60:.1f} min.")
    sys.stdout.flush()

    return results
