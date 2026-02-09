"""
Delivery self-check for Direction 3 outputs.
Run: python self_check.py
"""

from __future__ import annotations

import argparse
import os
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _load_csv_named(path: str):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def _has_no_nan(arr, fields):
    for f in fields:
        if np.isnan(arr[f]).any():
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Self-check for outputs")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir

    f2a = os.path.join(results_dir, "results_2A.csv")
    f2c = os.path.join(results_dir, "results_2C.csv")
    g2a = os.path.join(results_dir, "fig_2A.png")
    g2c = os.path.join(results_dir, "fig_2C.png")

    checks = []

    for f in [f2a, f2c, g2a, g2c]:
        checks.append((f"file exists: {f}", os.path.exists(f)))

    if os.path.exists(f2a):
        d2a = _load_csv_named(f2a)
        required_cols_2a = {
            "sigma_ratio",
            "kappa_c_mean_gen",
            "kappa_c_std_gen",
            "kappa_c_mean_con",
            "kappa_c_std_con",
        }
        checks.append(("2A: 9 rows", len(d2a) == 9))
        checks.append(("2A: columns complete", required_cols_2a.issubset(set(d2a.dtype.names))))
        checks.append(("2A: no NaN", _has_no_nan(d2a, required_cols_2a)))
        checks.append(("2A: kappa_c > 0", np.all(d2a["kappa_c_mean_gen"] > 0) and np.all(d2a["kappa_c_mean_con"] > 0)))

    if os.path.exists(f2c):
        d2c = _load_csv_named(f2c)
        required_cols_2c = {"r", "kappa_c_mean", "kappa_c_std"}
        checks.append(("2C: 11 rows", len(d2c) == 11))
        checks.append(("2C: columns complete", required_cols_2c.issubset(set(d2c.dtype.names))))
        checks.append(("2C: no NaN", _has_no_nan(d2c, required_cols_2c)))

    print("\n===== delivery self-check =====")
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    print("\nALL PASS" if all_ok else "HAS FAILURES")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
