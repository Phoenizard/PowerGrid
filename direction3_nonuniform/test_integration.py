"""
Integration tests for compatibility with existing paper_reproduction model.
Run: python test_integration.py

Phase 1 interface notes (from existing code):
1) assign_power signature:
   assign_power(n, n_plus, n_minus, P_max, rng=None) -> np.ndarray shape (n,)
   where generator and consumer node positions are assigned by random permutation.

2) compute_kappa_c signature:
   compute_kappa_c(A_csr, P, config_params=None) -> float
   (no direct gamma argument; gamma comes from config_params['gamma']).

3) network generation:
   generate_network(n, k, q, seed=None) -> scipy sparse adjacency matrix
   via networkx.watts_strogatz_graph.

4) P-to-network mapping:
   P[i] maps to node i and adjacency row/column i in A.

5) run_sweep calling pattern:
   A = generate_network(...)
   P = assign_power(...)
   kappa = compute_kappa_c_normalized(A, P, P_max, config_params)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

import model as model_module  # noqa: E402
from model import generate_network, compute_kappa_c  # noqa: E402
from power_allocation import assign_power_heterogeneous, assign_power_centralized  # noqa: E402

TEST_CONFIG = {
    "gamma": 1.0,
    "kappa_range": (0.001, 3.0),
    "bisection_steps": 5,
    "t_integrate": 20,
    "conv_tol": 5e-3,
    "max_step": 5.0,
}


def _compute_kappa_seeded(A, p, seed=123):
    """Force deterministic initial condition inside compute_kappa_c for test repeatability."""
    original_default_rng = model_module.np.random.default_rng

    def seeded_default_rng(local_seed=None):
        if local_seed is None:
            return original_default_rng(seed)
        return original_default_rng(local_seed)

    model_module.np.random.default_rng = seeded_default_rng
    try:
        return compute_kappa_c(A, p, config_params=TEST_CONFIG)
    finally:
        model_module.np.random.default_rng = original_default_rng


def test_kappa_c_computable():
    a = generate_network(n=50, k=4, q=0.1, seed=100)
    p = assign_power_heterogeneous(25, 25, 1.0, 0.3, "gen")
    kc = _compute_kappa_seeded(a, p, seed=11)
    assert isinstance(kc, (int, float)) and 0 < kc < 10, f"kc={kc}"
    print(f"PASS test_kappa_c_computable (kc={kc:.4f})")


def test_baseline_consistency():
    a = generate_network(n=50, k=4, q=0.1, seed=123)
    p0 = assign_power_heterogeneous(25, 25, 1.0, 0, "gen")
    kc0 = _compute_kappa_seeded(a, p0, seed=22)
    p1 = assign_power_heterogeneous(25, 25, 1.0, 0, "gen")
    kc1 = _compute_kappa_seeded(a, p1, seed=22)
    assert abs(kc0 - kc1) < 1e-10, f"kc0={kc0:.12f} != kc1={kc1:.12f}"
    print(f"PASS test_baseline_consistency (kc={kc0:.4f})")


def test_centralized_r_baseline():
    a = generate_network(n=50, k=4, q=0.1, seed=456)
    kc_u = _compute_kappa_seeded(a, assign_power_heterogeneous(25, 25, 1.0, 0, "gen"), seed=33)
    kc_c = _compute_kappa_seeded(a, assign_power_centralized(25, 25, 1.0, 1 / 25), seed=33)
    assert abs(kc_u - kc_c) < 1e-10, f"kc_uniform={kc_u:.12f} != kc_cent={kc_c:.12f}"
    print(f"PASS test_centralized_r_baseline (kc={kc_u:.4f})")


def test_monotonicity_quick():
    sigmas = [0, 0.3, 0.8]
    means = []
    rng = np.random.default_rng(999)

    for s in sigmas:
        kcs = []
        for i in range(5):
            a = generate_network(50, 4, 0.1, seed=int(rng.integers(0, 2**31 - 1)))
            p = assign_power_heterogeneous(25, 25, 1.0, s, "gen")
            kcs.append(_compute_kappa_seeded(a, p, seed=1000 + i))
        means.append(float(np.mean(kcs)))
        print(f"  sigma_ratio={s:.1f}: mean kc={means[-1]:.4f}")

    assert means[-1] > means[0] * 0.95, f"kc not increasing enough: {means}"
    print("PASS test_monotonicity_quick")


def test_timing():
    a = generate_network(n=50, k=4, q=0.1, seed=2024)
    p = assign_power_heterogeneous(25, 25, 1.0, 0.3, "gen")

    t0 = time.time()
    for i in range(5):
        _compute_kappa_seeded(a, p, seed=200 + i)
    per_call = (time.time() - t0) / 5.0

    total_calls_for_200 = (9 * 2 + 11) * 200
    total_hours = total_calls_for_200 * per_call / 3600.0
    print(f"  per_call={per_call:.2f}s | estimated total={total_hours:.2f}h")
    if total_hours > 4.0:
        print("  WARN estimated runtime > 4h; use --n_ensemble 100 for production")
    print("PASS test_timing")


if __name__ == "__main__":
    print("=" * 60)
    print("Running integration tests")
    print("=" * 60)

    tests = [
        test_kappa_c_computable,
        test_baseline_consistency,
        test_centralized_r_baseline,
        test_monotonicity_quick,
        test_timing,
    ]

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
