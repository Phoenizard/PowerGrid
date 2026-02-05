"""
Runner script for sweep computations with proper Windows multiprocessing guard.

Usage:
    python run_sweep.py              # Fast iteration mode (default)
    python run_sweep.py --production # Final production mode
"""

import argparse
import numpy as np
import os
import gc
import time
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import generate_network, assign_power, compute_kappa_c_normalized


def parse_args():
    parser = argparse.ArgumentParser(description='Run sweep computations for Fig. 1C/1D')
    parser.add_argument('--production', action='store_true',
                        help='Use final production mode (ENSEMBLE=200, STEP=2)')
    return parser.parse_args()


def compute_single_config(args):
    """Compute mean kappa_c for a single configuration."""
    n_plus, n_minus, q, n_realizations, seed = args

    rng = np.random.default_rng(seed)
    kappa_values = []

    config_params = {
        'n': config.N,
        'k': config.K,
        'P_max': config.P_MAX,
        't_integrate': config.T_INTEGRATE,
    }

    for _ in range(n_realizations):
        A = generate_network(config.N, config.K, q, seed=rng.integers(0, 2**31))
        P = assign_power(config.N, n_plus, n_minus, config.P_MAX, rng=rng)
        kappa_c = compute_kappa_c_normalized(A, P, config.P_MAX, config_params)
        kappa_values.append(kappa_c)

    return n_plus, n_minus, np.mean(kappa_values), np.std(kappa_values)


def generate_simplex_points(n, step):
    """Generate all valid (n_plus, n_minus, n_passive) configurations."""
    points = []
    for n_plus in range(1, n, step):
        for n_minus in range(1, n - n_plus + 1, step):
            n_passive = n - n_plus - n_minus
            if n_passive >= 0:
                points.append((n_plus, n_minus, n_passive))
    return points


def run_simplex_sweep(ensemble_size, step_size):
    """Run simplex sweep for Fig. 1C."""
    print("=" * 60)
    print("SIMPLEX SWEEP (Fig. 1C)")
    print("=" * 60)

    output_file = os.path.join(config.OUTPUT_DIR, 'data_simplex_q0.0.npz')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    q = 0.0
    n_realizations = ensemble_size
    step = step_size

    print(f"Parameters: q={q}, realizations={n_realizations}, step={step}")

    points = generate_simplex_points(config.N, step)
    print(f"Total simplex points: {len(points)}")

    rng = np.random.default_rng(42)
    args_list = [
        (n_plus, n_minus, q, n_realizations, rng.integers(0, 2**31))
        for n_plus, n_minus, n_passive in points
    ]

    start_time = time.time()
    results_list = []

    # Use sequential processing for reliability on Windows
    print(f"Processing {len(args_list)} configurations...")
    for i, args in enumerate(tqdm(args_list, desc="Simplex")):
        result = compute_single_config(args)
        results_list.append(result)
        if (i + 1) % 50 == 0:
            gc.collect()

    elapsed = time.time() - start_time
    print(f"Simplex sweep completed in {elapsed/60:.1f} minutes")

    # Save results
    n_plus_arr = np.array([r[0] for r in results_list])
    n_minus_arr = np.array([r[1] for r in results_list])
    n_passive_arr = config.N - n_plus_arr - n_minus_arr
    mean_kappa_arr = np.array([r[2] for r in results_list])
    std_kappa_arr = np.array([r[3] for r in results_list])

    np.savez(output_file,
             n_plus=n_plus_arr,
             n_minus=n_minus_arr,
             n_passive=n_passive_arr,
             mean_kappa=mean_kappa_arr,
             std_kappa=std_kappa_arr)

    print(f"Saved to {output_file}")
    print(f"kappa range: [{mean_kappa_arr.min():.4f}, {mean_kappa_arr.max():.4f}]")

    return output_file


def run_cross_section_sweep(ensemble_size):
    """Run cross-section sweep for Fig. 1D."""
    print("\n" + "=" * 60)
    print("CROSS-SECTION SWEEP (Fig. 1D)")
    print("=" * 60)

    output_file = os.path.join(config.OUTPUT_DIR, 'data_crosssec.npz')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    q_values = config.Q_VALUES
    n_realizations = ensemble_size
    n_passive = config.NP_CROSS
    n_active = config.N - n_passive

    n_minus_range = list(range(1, n_active))
    print(f"Parameters: np={n_passive}, n_minus from {n_minus_range[0]} to {n_minus_range[-1]}")
    print(f"q values: {q_values}, realizations={n_realizations}")

    start_time = time.time()

    mean_kappa = np.zeros((len(q_values), len(n_minus_range)))
    std_kappa = np.zeros((len(q_values), len(n_minus_range)))

    for qi, q in enumerate(q_values):
        print(f"\nProcessing q={q}...")
        rng = np.random.default_rng(42 + qi)

        for i, n_minus in enumerate(tqdm(n_minus_range, desc=f"q={q}")):
            n_plus = n_active - n_minus
            args = (n_plus, n_minus, q, n_realizations, rng.integers(0, 2**31))
            _, _, mean_k, std_k = compute_single_config(args)
            mean_kappa[qi, i] = mean_k
            std_kappa[qi, i] = std_k

        gc.collect()

    elapsed = time.time() - start_time
    print(f"\nCross-section sweep completed in {elapsed/60:.1f} minutes")

    np.savez(output_file,
             n_minus=np.array(n_minus_range),
             q_values=np.array(q_values),
             mean_kappa=mean_kappa,
             std_kappa=std_kappa)

    print(f"Saved to {output_file}")

    return output_file


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    args = parse_args()

    # Set parameters based on mode
    if args.production:
        ensemble_size = config.ENSEMBLE_SIZE_FINAL
        step_size = config.STEP_SIZE_FINAL
        mode_name = "PRODUCTION"
    else:
        ensemble_size = config.ENSEMBLE_SIZE
        step_size = config.STEP_SIZE
        mode_name = "FAST ITERATION"

    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    print(f"Mode: {mode_name}")
    print(f"Config: N={config.N}, ENSEMBLE={ensemble_size}, STEP={step_size}")
    print()

    run_simplex_sweep(ensemble_size, step_size)
    run_cross_section_sweep(ensemble_size)

    print("\n" + "=" * 60)
    print("ALL COMPUTATIONS COMPLETE!")
    print("=" * 60)
