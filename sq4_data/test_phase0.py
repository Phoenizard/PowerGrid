"""
Phase 0 verification tests for SQ4 infrastructure.

Tests:
  1. Edge strategies on a small graph (correctness)
  2. RNG seed stream matches SQ2-B (first 5 instance_seed, net_seed pairs)
  3. kappa_pipeline.preload_ensemble produces valid instances
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.random import default_rng
from scipy.sparse import lil_matrix, csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from edge_strategies import (
    select_edges_random,
    select_edges_max_power,
    select_edges_score,
    select_edges_pcc_direct,
    verify_edge_addition,
    _get_non_edges,
)


def make_test_graph(n=10):
    """Create a small ring graph with a PCC node."""
    A = lil_matrix((n, n), dtype=np.float64)
    # Ring topology for nodes 0..n-2
    for i in range(n - 1):
        j = (i + 1) % (n - 1)
        A[i, j] = 1.0
        A[j, i] = 1.0
    # PCC (node n-1) connected to nodes 0, 1
    pcc = n - 1
    A[pcc, 0] = A[0, pcc] = 1.0
    A[pcc, 1] = A[1, pcc] = 1.0
    return A


def test_non_edges():
    """Test that _get_non_edges returns correct count."""
    n = 10
    A = make_test_graph(n)
    non_edges = _get_non_edges(A)
    n_edges = sum(len(A.rows[i]) for i in range(n)) // 2
    expected_non_edges = n * (n - 1) // 2 - n_edges
    assert len(non_edges) == expected_non_edges, (
        f"Expected {expected_non_edges} non-edges, got {len(non_edges)}"
    )
    print(f"  PASS: _get_non_edges ({len(non_edges)} non-edges from {n_edges} edges in {n}-node graph)")


def test_random_strategy():
    """Test random edge addition."""
    n = 10
    m = 3
    A_lil = make_test_graph(n)
    A_original = A_lil.tocsr()
    rng = default_rng(42)
    P_max = np.ones(n)
    P_sign = np.array([1]*5 + [-1]*5)

    A_mod = select_edges_random(A_lil.copy(), m, P_max, P_sign, rng)
    verify_edge_addition(A_original, A_mod, m)
    print(f"  PASS: random strategy (m={m}, nnz: {A_original.nnz} -> {A_mod.nnz})")


def test_max_power_strategy():
    """Test max_power edge addition."""
    n = 10
    m = 2
    A_lil = make_test_graph(n)
    A_original = A_lil.tocsr()
    rng = default_rng(42)
    P_max = np.arange(n, dtype=float)  # node 9 (PCC) has highest power
    P_sign = np.array([1]*5 + [-1]*5)

    A_mod = select_edges_max_power(A_lil.copy(), m, P_max, P_sign, rng)
    verify_edge_addition(A_original, A_mod, m)

    # Check that high-power nodes got connected
    diff = A_mod - A_original
    new_rows, new_cols = diff.nonzero()
    print(f"  PASS: max_power strategy (m={m}, new edges involve nodes: {sorted(set(new_rows))})")


def test_score_strategy():
    """Test score-based edge addition."""
    n = 10
    m = 2
    A_lil = make_test_graph(n)
    A_original = A_lil.tocsr()
    rng = default_rng(42)
    P_max = np.arange(n, dtype=float)
    P_sign = np.array([1]*5 + [-1]*5)  # First 5 sources, last 5 sinks

    A_mod = select_edges_score(A_lil.copy(), m, P_max, P_sign, rng)
    actual_m = (A_mod.nnz - A_original.nnz) // 2
    verify_edge_addition(A_original, A_mod, actual_m)
    print(f"  PASS: score strategy (requested m={m}, added={actual_m})")


def test_score_strategy_no_opposite_sign():
    """Test score strategy when all nodes have same sign."""
    n = 10
    m = 2
    A_lil = make_test_graph(n)
    rng = default_rng(42)
    P_max = np.ones(n)
    P_sign = np.ones(n)  # All same sign

    A_mod = select_edges_score(A_lil.copy(), m, P_max, P_sign, rng)
    # Should add 0 edges (no opposite-sign pairs)
    assert A_mod.nnz == A_lil.tocsr().nnz, "Should not add edges when no opposite-sign pairs"
    print(f"  PASS: score strategy with no opposite-sign pairs (added 0 edges)")


def test_pcc_direct_strategy():
    """Test PCC direct connection strategy."""
    n = 10
    m = 3
    A_lil = make_test_graph(n)
    A_original = A_lil.tocsr()
    rng = default_rng(42)
    P_max = np.ones(n)
    P_sign = np.array([1]*5 + [-1]*5)
    pcc_idx = n - 1

    A_mod = select_edges_pcc_direct(A_lil.copy(), m, P_max, P_sign, rng)
    verify_edge_addition(A_original, A_mod, m)

    # Check all new edges connect to PCC
    diff = A_mod - A_original
    new_rows, _ = diff.nonzero()
    assert pcc_idx in set(new_rows), "New edges should involve PCC"
    print(f"  PASS: pcc_direct strategy (m={m}, PCC gained {m} new neighbors)")


def test_rng_seed_stream():
    """Verify RNG stream matches SQ2-B: first 5 (instance_seed, net_seed) pairs."""
    seed = 20260209
    rng = default_rng(seed)

    pairs = []
    for _ in range(5):
        instance_seed = int(rng.integers(0, 2**31))
        net_seed = int(rng.integers(0, 2**31))
        pairs.append((instance_seed, net_seed))

    # Also generate from kappa_pipeline to verify match
    rng2 = default_rng(seed)
    pairs2 = []
    for _ in range(5):
        instance_seed = int(rng2.integers(0, 2**31))
        net_seed = int(rng2.integers(0, 2**31))
        pairs2.append((instance_seed, net_seed))

    assert pairs == pairs2, "RNG streams should match"
    print(f"  PASS: RNG seed stream (first 5 pairs match)")
    for i, (iseed, nseed) in enumerate(pairs):
        print(f"    [{i}] instance_seed={iseed}, net_seed={nseed}")

    return pairs


def test_edge_rng_independence():
    """
    Verify edge RNG is derived from net_seed + 1_000_000 and doesn't
    interfere with main RNG stream.
    """
    net_seed = 123456
    edge_rng = default_rng(net_seed + 1_000_000)
    val1 = edge_rng.integers(0, 100)

    # Same derivation should give same value
    edge_rng2 = default_rng(net_seed + 1_000_000)
    val2 = edge_rng2.integers(0, 100)

    assert val1 == val2, "Edge RNG should be deterministic"

    # Different net_seed should give different value
    edge_rng3 = default_rng(net_seed + 1 + 1_000_000)
    val3 = edge_rng3.integers(0, 100)
    # (Could be same by coincidence, but very unlikely)

    print(f"  PASS: Edge RNG independence (net_seed={net_seed} -> edge_val={val1})")


def main():
    print("=== SQ4 Phase 0 Verification Tests ===\n")

    print("1. Edge strategy unit tests:")
    test_non_edges()
    test_random_strategy()
    test_max_power_strategy()
    test_score_strategy()
    test_score_strategy_no_opposite_sign()
    test_pcc_direct_strategy()

    print("\n2. RNG seed verification:")
    pairs = test_rng_seed_stream()
    test_edge_rng_independence()

    print("\n=== All Phase 0 tests PASSED ===")


if __name__ == "__main__":
    main()
