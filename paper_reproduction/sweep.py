"""
Sweep computations for simplex heatmap and cross-section plots.
Saves results to .npz files for caching.
"""

import numpy as np
import os
import gc
import time
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
from tqdm import tqdm

import config
from model import (generate_network, assign_power, compute_kappa_c_normalized)


def compute_single_config(args):
    """
    Compute mean kappa_c for a single (n_plus, n_minus, q) configuration.
    Used for parallel processing.
    """
    n_plus, n_minus, q, n_realizations, seed = args

    n = config.N
    k = config.K
    P_max = config.P_MAX

    rng = np.random.default_rng(seed)
    kappa_values = []

    config_params = {
        'n': n,
        'k': k,
        'P_max': P_max,
        't_integrate': config.T_INTEGRATE,
    }

    for _ in range(n_realizations):
        # Generate network
        A = generate_network(n, k, q, seed=rng.integers(0, 2**31))

        # Assign power
        P = assign_power(n, n_plus, n_minus, P_max, rng=rng)

        # Compute kappa_c
        kappa_c = compute_kappa_c_normalized(A, P, P_max, config_params)
        kappa_values.append(kappa_c)

    return n_plus, n_minus, np.mean(kappa_values), np.std(kappa_values)


def generate_simplex_points(n, step):
    """
    Generate all valid (n_plus, n_minus, n_passive) configurations on the simplex.
    Constraint: n_plus + n_minus + n_passive = n
    n_plus >= 1, n_minus >= 1, n_passive >= 0
    """
    points = []
    for n_plus in range(1, n, step):
        for n_minus in range(1, n - n_plus + 1, step):
            n_passive = n - n_plus - n_minus
            if n_passive >= 0:
                points.append((n_plus, n_minus, n_passive))
    return points


def sweep_simplex(q=0.0, n_realizations=None, step=None, output_file=None, use_cache=True):
    """
    Sweep over the simplex for a given q value.

    Returns:
    --------
    results : dict with keys 'n_plus', 'n_minus', 'n_passive', 'mean_kappa', 'std_kappa'
    """
    if n_realizations is None:
        n_realizations = config.ENSEMBLE_SIZE
    if step is None:
        step = config.STEP_SIZE
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, f'data_simplex_q{q}.npz')

    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached simplex data from {output_file}")
        data = np.load(output_file)
        return {
            'n_plus': data['n_plus'],
            'n_minus': data['n_minus'],
            'n_passive': data['n_passive'],
            'mean_kappa': data['mean_kappa'],
            'std_kappa': data['std_kappa']
        }

    print(f"Computing simplex sweep: q={q}, realizations={n_realizations}, step={step}")

    # Generate all points
    points = generate_simplex_points(config.N, step)
    print(f"Total simplex points: {len(points)}")

    # Prepare arguments for parallel processing
    rng = np.random.default_rng(42)
    args_list = [
        (n_plus, n_minus, q, n_realizations, rng.integers(0, 2**31))
        for n_plus, n_minus, n_passive in points
    ]

    # Run computation
    start_time = time.time()

    results_list = []
    n_workers = config.N_WORKERS

    print(f"Using {n_workers} workers...")

    # Process in batches to manage memory
    batch_size = config.BATCH_SIZE
    for batch_start in tqdm(range(0, len(args_list), batch_size), desc="Batches"):
        batch = args_list[batch_start:batch_start + batch_size]

        with Pool(processes=n_workers) as pool:
            batch_results = pool.map(compute_single_config, batch)

        results_list.extend(batch_results)
        gc.collect()

    elapsed = time.time() - start_time
    print(f"Simplex sweep completed in {elapsed/60:.1f} minutes")

    # Organize results
    n_plus_arr = np.array([r[0] for r in results_list])
    n_minus_arr = np.array([r[1] for r in results_list])
    n_passive_arr = config.N - n_plus_arr - n_minus_arr
    mean_kappa_arr = np.array([r[2] for r in results_list])
    std_kappa_arr = np.array([r[3] for r in results_list])

    # Save to cache
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    np.savez(output_file,
             n_plus=n_plus_arr,
             n_minus=n_minus_arr,
             n_passive=n_passive_arr,
             mean_kappa=mean_kappa_arr,
             std_kappa=std_kappa_arr)
    print(f"Saved to {output_file}")

    return {
        'n_plus': n_plus_arr,
        'n_minus': n_minus_arr,
        'n_passive': n_passive_arr,
        'mean_kappa': mean_kappa_arr,
        'std_kappa': std_kappa_arr
    }


def sweep_cross_section(q_values=None, n_realizations=None, output_file=None, use_cache=True):
    """
    Sweep along the cross-section (i) for multiple q values.
    Cross-section: n_passive = NP_CROSS, n_plus + n_minus = N - NP_CROSS

    Returns:
    --------
    results : dict with keys 'n_minus', 'q_values', 'mean_kappa', 'std_kappa'
    """
    if q_values is None:
        q_values = config.Q_VALUES
    if n_realizations is None:
        n_realizations = config.ENSEMBLE_SIZE
    if output_file is None:
        output_file = os.path.join(config.OUTPUT_DIR, 'data_crosssec.npz')

    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached cross-section data from {output_file}")
        data = np.load(output_file)
        return {
            'n_minus': data['n_minus'],
            'q_values': data['q_values'],
            'mean_kappa': data['mean_kappa'],
            'std_kappa': data['std_kappa']
        }

    print(f"Computing cross-section sweep: q_values={q_values}, realizations={n_realizations}")

    n_passive = config.NP_CROSS
    n_active = config.N - n_passive  # n_plus + n_minus = 34

    # n_minus from 1 to n_active - 1 (so n_plus >= 1)
    n_minus_range = list(range(1, n_active))
    print(f"Cross-section: np={n_passive}, n_minus from {n_minus_range[0]} to {n_minus_range[-1]}")

    start_time = time.time()

    # Results storage
    mean_kappa = np.zeros((len(q_values), len(n_minus_range)))
    std_kappa = np.zeros((len(q_values), len(n_minus_range)))

    n_workers = config.N_WORKERS

    for qi, q in enumerate(q_values):
        print(f"\nProcessing q={q}...")

        # Prepare arguments
        rng = np.random.default_rng(42 + qi)
        args_list = [
            (n_active - n_minus, n_minus, q, n_realizations, rng.integers(0, 2**31))
            for n_minus in n_minus_range
        ]

        # Run computation
        with Pool(processes=n_workers) as pool:
            results_list = list(tqdm(
                pool.imap(compute_single_config, args_list),
                total=len(args_list),
                desc=f"q={q}"
            ))

        for i, (n_plus, n_minus, mean_k, std_k) in enumerate(results_list):
            mean_kappa[qi, i] = mean_k
            std_kappa[qi, i] = std_k

        gc.collect()

    elapsed = time.time() - start_time
    print(f"\nCross-section sweep completed in {elapsed/60:.1f} minutes")

    # Save to cache
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    np.savez(output_file,
             n_minus=np.array(n_minus_range),
             q_values=np.array(q_values),
             mean_kappa=mean_kappa,
             std_kappa=std_kappa)
    print(f"Saved to {output_file}")

    return {
        'n_minus': np.array(n_minus_range),
        'q_values': np.array(q_values),
        'mean_kappa': mean_kappa,
        'std_kappa': std_kappa
    }


if __name__ == '__main__':
    # Windows multiprocessing guard
    set_start_method('spawn', force=True)

    import sys

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Parse command line args
    run_simplex = '--simplex' in sys.argv or len(sys.argv) == 1
    run_crosssec = '--crosssec' in sys.argv or len(sys.argv) == 1
    force_recompute = '--force' in sys.argv

    if run_simplex:
        print("="*60)
        print("SIMPLEX SWEEP (Fig. 1C)")
        print("="*60)
        sweep_simplex(q=0.0, use_cache=not force_recompute)

    if run_crosssec:
        print("\n" + "="*60)
        print("CROSS-SECTION SWEEP (Fig. 1D)")
        print("="*60)
        sweep_cross_section(use_cache=not force_recompute)

    print("\nAll computations complete!")
