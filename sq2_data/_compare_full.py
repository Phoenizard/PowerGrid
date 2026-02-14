"""
Full comparison: our 50-ensemble vs GridResilience 50-ensemble (fullpen, July).
Loads all 50 pkl files and our CSV, prints side-by-side statistics.
"""
import sys, os, pathlib
import numpy as np

# Stub for powerreader so dill can unpickle Trajectory objects
import types
powerreader_stub = types.ModuleType("powerreader")
def _css(Pvec):
    ls = np.max(Pvec); n = len(Pvec); ld = np.abs(np.min(Pvec))
    s = [x for x in Pvec if x > 0.0]; d = [x for x in Pvec if x < 0.0]
    return np.sum(s)/(n*ls), np.sum(np.abs(d))/(n*ld), 0.0
powerreader_stub.continuoussourcesinkcounter = _css
sys.modules["powerreader"] = powerreader_stub

GRIDRES_ROOT = pathlib.Path(__file__).resolve().parent.parent / "GridResilience"
sys.path.insert(0, str(GRIDRES_ROOT / "scripts"))
import dill

RESULTS_CSV = pathlib.Path(__file__).resolve().parent / "results_sq2" / "trajectory_summer_fullpen.csv"
FULLPEN_DIR = GRIDRES_ROOT / "trajdata" / "7" / "fullpen_nobat"
HALFPEN_DIR = GRIDRES_ROOT / "trajdata" / "7" / "halfpen_nobat"

def load_pkl(path):
    with open(path, "rb") as f:
        try: return dill.load(f)
        except UnicodeDecodeError:
            f.seek(0); return dill.load(f, encoding="latin1")

def stats(sigmas):
    a = np.array(sigmas)
    return {lab: {"min": a[:,i].min(), "max": a[:,i].max(), "mean": a[:,i].mean()}
            for i, lab in enumerate(["sigma_s","sigma_d","sigma_p"])}

def fmt(name, s):
    parts = [f"{name:28s}"]
    for lab in ["sigma_s","sigma_d","sigma_p"]:
        v = s[lab]
        parts.append(f" {v['min']:7.4f} {v['max']:7.4f} {v['mean']:7.4f}")
    print("".join(parts))

# --- Load GridResilience (all 50) ---
print("Loading GridResilience fullpen (50 members)...")
gr_full_all = []
gr_full_per_instance = []
for i in range(50):
    t = load_pkl(FULLPEN_DIR / f"{i}.pkl")
    gr_full_all.extend(t.sigmas)
    gr_full_per_instance.append(np.array(t.sigmas))

print("Loading GridResilience halfpen (50 members)...")
gr_half_all = []
for i in range(50):
    t = load_pkl(HALFPEN_DIR / f"{i}.pkl")
    gr_half_all.extend(t.sigmas)

# --- Load ours ---
print("Loading our trajectory_summer_fullpen.csv...")
data = np.loadtxt(RESULTS_CSV, delimiter=",", skiprows=1)
our_mean_sigmas = list(zip(data[:,2], data[:,4], data[:,6]))

# --- Per-instance sigma_d max distribution ---
gr_sd_maxes = [inst[:,1].max() for inst in gr_full_per_instance]

# --- Print comparison ---
print()
print("=" * 88)
print("  FULL 50-ENSEMBLE COMPARISON (Summer, fullpen, no battery)")
print("=" * 88)
hdr = f"{'':28s} {'sigma_s':>24s} {'sigma_d':>24s} {'sigma_p':>24s}"
sub = f"{'':28s} {'min':>8s}{'max':>8s}{'mean':>8s} {'min':>8s}{'max':>8s}{'mean':>8s} {'min':>8s}{'max':>8s}{'mean':>8s}"
print(hdr); print(sub)
fmt("GR fullpen [all 50]", stats(gr_full_all))
fmt("GR halfpen [all 50]", stats(gr_half_all))
fmt("Ours fullpen [mean of 50]", stats(our_mean_sigmas))

# --- Per-instance sigma_d max comparison ---
print()
print("=" * 88)
print("  PER-INSTANCE sigma_d max DISTRIBUTION")
print("=" * 88)
print(f"  GR fullpen: mean={np.mean(gr_sd_maxes):.4f}  "
      f"std={np.std(gr_sd_maxes):.4f}  "
      f"min={np.min(gr_sd_maxes):.4f}  max={np.max(gr_sd_maxes):.4f}")
print(f"  Ours (from CSV, ensemble-mean max): {data[:,4].max():.4f}")

# Quantiles
print(f"\n  GR fullpen sigma_d max quantiles:")
for q in [10, 25, 50, 75, 90]:
    print(f"    {q}th percentile: {np.percentile(gr_sd_maxes, q):.4f}")

# --- Timestep-by-timestep comparison (mean trajectories) ---
# Compute GR mean trajectory
gr_mean = np.zeros((264, 3))
for inst in gr_full_per_instance:
    gr_mean += inst
gr_mean /= 50.0

print()
print("=" * 88)
print("  MEAN TRAJECTORY COMPARISON (first 24 steps = hours 24-36)")
print("=" * 88)
print(f"  {'step':>4s}  {'GR_ss':>7s} {'Our_ss':>7s}  {'GR_sd':>7s} {'Our_sd':>7s}  {'GR_sp':>7s} {'Our_sp':>7s}")
for i in range(24):
    g = gr_mean[i]
    o = (data[i,2], data[i,4], data[i,6])
    print(f"  {i:4d}  {g[0]:7.4f} {o[0]:7.4f}  {g[1]:7.4f} {o[1]:7.4f}  {g[2]:7.4f} {o[2]:7.4f}")

# --- Overall correlation ---
print()
print("=" * 88)
print("  CORRELATION (mean trajectory, 264 timesteps)")
print("=" * 88)
for i, lab in enumerate(["sigma_s","sigma_d","sigma_p"]):
    gr_col = gr_mean[:, i]
    our_col = data[:, 2 + i*2]
    corr = np.corrcoef(gr_col, our_col)[0,1]
    rmse = np.sqrt(np.mean((gr_col - our_col)**2))
    print(f"  {lab}: r={corr:.4f}  RMSE={rmse:.6f}")

print("\nDone.")
