"""
One-time script: Convert LCL CSV + PV hourly CSV to Parquet format.

Parquet is ~10x faster to read than CSV (no date string parsing overhead).
Run once, then data_loader will use Parquet files automatically.
"""
import glob
import os
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- LCL CSVs → Parquet ---
LCL_DIR = os.path.join(PROJECT_ROOT, "data", "LCL")
LCL_PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "LCL_parquet")
os.makedirs(LCL_PARQUET_DIR, exist_ok=True)

LCL_KWH_COL = "KWH/hh (per half hour) "

csv_files = sorted(glob.glob(os.path.join(LCL_DIR, "LCL-June2015v2_*.csv")))
print(f"Converting {len(csv_files)} LCL CSV files to Parquet...")

t0 = time.time()
for i, fpath in enumerate(csv_files):
    basename = os.path.splitext(os.path.basename(fpath))[0]
    out_path = os.path.join(LCL_PARQUET_DIR, f"{basename}.parquet")

    if os.path.exists(out_path):
        continue  # skip already converted

    df = pd.read_csv(fpath, usecols=["LCLid", "DateTime", LCL_KWH_COL])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df[LCL_KWH_COL] = pd.to_numeric(df[LCL_KWH_COL], errors="coerce")
    df = df.dropna(subset=[LCL_KWH_COL])
    df.to_parquet(out_path, index=False)

    if (i + 1) % 20 == 0 or i == 0:
        print(f"  [{i+1}/{len(csv_files)}] {basename} -> parquet")

t1 = time.time()
print(f"LCL done in {t1-t0:.1f}s, output: {LCL_PARQUET_DIR}")

# --- PV hourly CSV → Parquet ---
PV_CSV = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "EXPORT HourlyData - Customer Endpoints.csv",
)
PV_PARQUET = os.path.join(
    PROJECT_ROOT, "data", "PV",
    "2014-11-28 Cleansed and Processed", "EXPORT HourlyData",
    "pv_hourly_customer_endpoints.parquet",
)

print(f"\nConverting PV hourly CSV to Parquet...")
t0 = time.time()
if not os.path.exists(PV_PARQUET):
    df = pd.read_csv(PV_CSV,
                     usecols=["Substation", "datetime", "P_GEN_MAX", "P_GEN_MIN"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["P_GEN_MAX"] = pd.to_numeric(df["P_GEN_MAX"], errors="coerce")
    df["P_GEN_MIN"] = pd.to_numeric(df["P_GEN_MIN"], errors="coerce")
    df = df.dropna(subset=["P_GEN_MAX", "P_GEN_MIN"])
    df.to_parquet(PV_PARQUET, index=False)
    t1 = time.time()
    print(f"PV done in {t1-t0:.1f}s, output: {PV_PARQUET}")
else:
    print("  Already exists, skipping.")

print("\nAll done. Update data_loader.py paths to use Parquet files.")
