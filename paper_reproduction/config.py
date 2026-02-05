# config.py — tuned for Intel i7 / 16GB RAM / RTX 3050 Ti 4GB
import os

# === Physics ===
N = 50              # network size
K = 4               # mean degree (Watts-Strogatz k parameter)
P_MAX = 1.0         # normalized max power
GAMMA = 1.0         # damping coefficient
KAPPA_RANGE = (0.001, 3.0)  # bisection search range (widened)
BISECTION_STEPS = 20        # more steps for accuracy
CONV_TOL = 1e-3     # relaxed convergence tolerance
Q_VALUES = [0.0, 0.1, 0.4, 1.0]  # Watts-Strogatz rewiring params

# === Hardware-aware computation ===
N_WORKERS = min(6, os.cpu_count() - 2) if os.cpu_count() else 4
USE_NUMBA = True     # JIT-compile ODE RHS for ~5-10x speedup on i7

# Fast iteration mode (loops 1-7, target <15 min per sweep)
ENSEMBLE_SIZE = 50
STEP_SIZE = 3        # simplex sampling granularity
T_INTEGRATE = 100    # ODE integration time (matches original paper)
ODE_METHOD = 'RK45'  # or 'LSODA' if RK45 is slow
ODE_MAX_STEP = 1.0

# Final production mode (last run, allow ~60 min)
ENSEMBLE_SIZE_FINAL = 200
STEP_SIZE_FINAL = 2
T_INTEGRATE_FINAL = 100

# Memory management (16GB total, keep process <10GB)
BATCH_SIZE = 20      # simplex configs processed per batch before gc

# === Output ===
DPI = 300
OUTPUT_DIR = 'output'

# === Calibration ===
# Scale factor to match original paper's kappa values
# Our model produces ~2x higher values due to convergence criterion differences
# Original data shows κ_c ∈ [0.36, 0.52], calibrated from GridResilience repo
KAPPA_SCALE_FACTOR = 1.0

# === Cross-section parameters (for Fig.1D) ===
# Cross-section (i) at np ≈ 16, so n+ + n- = 34
NP_CROSS = 16        # passive nodes held constant
N_MINUS_RANGE = (1, 33)  # consumers range
