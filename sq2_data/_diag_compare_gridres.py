"""
Diagnostic: Load GridResilience precomputed trajectory pickles and compare
sigma ranges with our trajectory_summer.csv results.

Compares:
  - July fullpen_nobat (100% PV, no battery) → closest to our scenario
  - July halfpen_nobat (50% PV, no battery) → Fig.4G scenario

Each pkl contains a Trajectory object with:
  traj.sigmas  →  list of (sigma_s, sigma_d, sigma_p) tuples
  traj.n       →  50

Requires: dill
"""

import sys
import os
import pathlib
import numpy as np

# We need to unpickle Trajectory objects from GridResilience.
# The pickle references powerclasses.Trajectory, which imports powerreader,
# which imports dask (heavy/missing dep). Instead of installing all deps,
# create a minimal stub for powerreader and put scripts dir on sys.path.
import types

GRIDRES_ROOT = pathlib.Path(__file__).resolve().parent.parent / "GridResilience"
SCRIPTS_DIR = str(GRIDRES_ROOT / "scripts")

# Create a minimal powerreader stub so powerclasses can import it
powerreader_stub = types.ModuleType("powerreader")
# Add the one function powerclasses actually calls
def continuoussourcesinkcounter(Pvec):
    largestsource = np.max(Pvec)
    n = len(Pvec)
    largestsink = np.abs(np.min(Pvec))
    sourceterms = [x for x in Pvec if x > 0.0]
    sinkterms = [x for x in Pvec if x < 0.0]
    sigma_s = np.sum(sourceterms) / (n * largestsource)
    sigma_d = np.sum(np.abs(sinkterms)) / (n * largestsink)
    sigma_p = 1.0 - sigma_s - sigma_d
    return sigma_s, sigma_d, sigma_p

powerreader_stub.continuoussourcesinkcounter = continuoussourcesinkcounter
sys.modules["powerreader"] = powerreader_stub

# Now add scripts dir so powerclasses module can be found
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    import dill
except ImportError:
    print("ERROR: dill not installed. Run: pip install dill")
    sys.exit(1)

# -- Paths ------------------------------------------------------------------
TRAJDATA = GRIDRES_ROOT / "trajdata"
RESULTS = pathlib.Path(__file__).resolve().parent / "results_sq2"

FULLPEN_DIR = TRAJDATA / "7" / "fullpen_nobat"
HALFPEN_DIR = TRAJDATA / "7" / "halfpen_nobat"

# -- Helpers ----------------------------------------------------------------

def load_trajectory(pkl_path):
    """Load a Trajectory object from a dill pickle file.

    These pickles were likely created with Python 2, so we need
    encoding='latin1' for cross-version compatibility.
    """
    with open(pkl_path, "rb") as f:
        try:
            traj = dill.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            traj = dill.load(f, encoding="latin1")
    return traj


def sigma_stats(sigmas):
    """Compute min/max/mean for each sigma component."""
    arr = np.array(sigmas)  # shape (T, 3)
    labels = ["sigma_s", "sigma_d", "sigma_p"]
    stats = {}
    for i, lab in enumerate(labels):
        col = arr[:, i]
        stats[lab] = {
            "min": np.min(col),
            "max": np.max(col),
            "mean": np.mean(col),
        }
    return stats


def print_stats(name, stats, n_timesteps):
    """Pretty-print sigma statistics."""
    print(f"\n{'='*60}")
    print(f"  {name}  ({n_timesteps} timesteps)")
    print(f"{'='*60}")
    for lab in ["sigma_s", "sigma_d", "sigma_p"]:
        s = stats[lab]
        print(f"  {lab:10s}  min={s['min']:.6f}  max={s['max']:.6f}  mean={s['mean']:.6f}")


# -- Load GridResilience data -----------------------------------------------

print("Loading GridResilience trajectory pickles...")
print(f"  fullpen dir: {FULLPEN_DIR}")
print(f"  halfpen dir: {HALFPEN_DIR}")

# Load a few ensemble members for statistics
N_ENSEMBLE = 5  # load first 5 for a quick check

# -- Fullpen (100% PV penetration) ------------------------------------------
print(f"\n--- Loading fullpen_nobat (0..{N_ENSEMBLE-1}) ---")
fullpen_all_sigmas = []
for i in range(N_ENSEMBLE):
    pkl_path = FULLPEN_DIR / f"{i}.pkl"
    traj = load_trajectory(pkl_path)
    print(f"  [{i}] n={traj.n}, len(sigmas)={len(traj.sigmas)}, "
          f"len(maxpowers)={len(traj.maxpowers)}")
    fullpen_all_sigmas.extend(traj.sigmas)

# Keep first member for detailed analysis
traj_full_0 = load_trajectory(FULLPEN_DIR / "0.pkl")
stats_full_0 = sigma_stats(traj_full_0.sigmas)
print_stats("GridRes fullpen_nobat [member 0]", stats_full_0, len(traj_full_0.sigmas))

# Ensemble-wide stats
stats_full_all = sigma_stats(fullpen_all_sigmas)
print_stats(f"GridRes fullpen_nobat [ensemble 0..{N_ENSEMBLE-1}]",
            stats_full_all, len(fullpen_all_sigmas))

# -- Halfpen (50% PV penetration) -------------------------------------------
print(f"\n--- Loading halfpen_nobat (0..{N_ENSEMBLE-1}) ---")
halfpen_all_sigmas = []
for i in range(N_ENSEMBLE):
    pkl_path = HALFPEN_DIR / f"{i}.pkl"
    traj = load_trajectory(pkl_path)
    print(f"  [{i}] n={traj.n}, len(sigmas)={len(traj.sigmas)}, "
          f"len(maxpowers)={len(traj.maxpowers)}")
    halfpen_all_sigmas.extend(traj.sigmas)

traj_half_0 = load_trajectory(HALFPEN_DIR / "0.pkl")
stats_half_0 = sigma_stats(traj_half_0.sigmas)
print_stats("GridRes halfpen_nobat [member 0]", stats_half_0, len(traj_half_0.sigmas))

stats_half_all = sigma_stats(halfpen_all_sigmas)
print_stats(f"GridRes halfpen_nobat [ensemble 0..{N_ENSEMBLE-1}]",
            stats_half_all, len(halfpen_all_sigmas))

# -- Our data ---------------------------------------------------------------
print("\n--- Loading our trajectory_summer.csv ---")
traj_csv = RESULTS / "trajectory_summer.csv"
if traj_csv.exists():
    data = np.loadtxt(traj_csv, delimiter=",", skiprows=1)
    # columns: timestep, hour, eta_plus_mean, eta_plus_std,
    #          eta_minus_mean, eta_minus_std, eta_p_mean, eta_p_std
    our_sigmas = list(zip(data[:, 2], data[:, 4], data[:, 6]))  # mean values
    stats_ours = sigma_stats(our_sigmas)
    print_stats("Our trajectory_summer.csv (ensemble means)", stats_ours, len(our_sigmas))
else:
    print(f"  WARNING: {traj_csv} not found!")
    stats_ours = None

# -- Head-to-head comparison ------------------------------------------------
print("\n" + "=" * 60)
print("  COMPARISON TABLE")
print("=" * 60)
header = f"{'':20s} {'sigma_s':>24s} {'sigma_d':>24s} {'sigma_p':>24s}"
subhdr = f"{'':20s} {'min':>8s}{'max':>8s}{'mean':>8s} {'min':>8s}{'max':>8s}{'mean':>8s} {'min':>8s}{'max':>8s}{'mean':>8s}"
print(header)
print(subhdr)

def fmt_row(name, stats):
    parts = [f"{name:20s}"]
    for lab in ["sigma_s", "sigma_d", "sigma_p"]:
        s = stats[lab]
        parts.append(f" {s['min']:7.4f} {s['max']:7.4f} {s['mean']:7.4f}")
    print("".join(parts))

fmt_row("GR fullpen [0]", stats_full_0)
fmt_row(f"GR fullpen [0..{N_ENSEMBLE-1}]", stats_full_all)
fmt_row("GR halfpen [0]", stats_half_0)
fmt_row(f"GR halfpen [0..{N_ENSEMBLE-1}]", stats_half_all)
if stats_ours:
    fmt_row("Ours (mean)", stats_ours)

# -- Key diagnostic: is fullpen also compressed? ----------------------------
print("\n" + "=" * 60)
print("  DIAGNOSIS")
print("=" * 60)
full_sd_max = stats_full_all["sigma_d"]["max"]
half_sd_max = stats_half_all["sigma_d"]["max"]
full_sp_min = stats_full_all["sigma_p"]["min"]
half_sp_min = stats_half_all["sigma_p"]["min"]

print(f"  fullpen sigma_d max  = {full_sd_max:.4f}")
print(f"  halfpen sigma_d max  = {half_sd_max:.4f}")
print(f"  fullpen sigma_p min  = {full_sp_min:.4f}")
print(f"  halfpen sigma_p min  = {half_sp_min:.4f}")

if full_sd_max < 0.15 and half_sd_max > 0.15:
    print("\n  >> CONFIRMED: fullpen trajectory IS compressed (low sigma_d),")
    print("     while halfpen has wide excursions.")
    print("     This is the EXPECTED physical result of 100% PV penetration.")
    print("     NOT a bug in our implementation.")
elif full_sd_max > 0.15:
    print("\n  >> UNEXPECTED: fullpen has large sigma_d swings.")
    print("     Our implementation may have a data processing issue.")
else:
    print("\n  >> BOTH scenarios show compressed sigma_d. Unusual.")

# -- Detailed timestep comparison for member 0 ------------------------------
print("\n" + "=" * 60)
print("  TIMESTEP DETAIL: fullpen member 0 (first 20 steps)")
print("=" * 60)
print(f"  {'step':>4s}  {'sigma_s':>8s}  {'sigma_d':>8s}  {'sigma_p':>8s}")
for i, (ss, sd, sp) in enumerate(traj_full_0.sigmas[:20]):
    print(f"  {i:4d}  {ss:8.5f}  {sd:8.5f}  {sp:8.5f}")

print(f"\n  TIMESTEP DETAIL: halfpen member 0 (first 20 steps)")
print(f"  {'step':>4s}  {'sigma_s':>8s}  {'sigma_d':>8s}  {'sigma_p':>8s}")
for i, (ss, sd, sp) in enumerate(traj_half_0.sigmas[:20]):
    print(f"  {i:4d}  {ss:8.5f}  {sd:8.5f}  {sp:8.5f}")

print("\nDone.")
