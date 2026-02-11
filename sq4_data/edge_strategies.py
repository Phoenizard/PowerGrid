"""
Edge addition strategies for SQ4 topology optimization.

Four strategies for adding m edges to an existing WS+PCC network:
  1. random       — uniform random unconnected pairs
  2. max_power    — top-m by sum of endpoint |P_max|
  3. score        — top-m by |P|/(d*(d+1)), opposite-sign only
  4. pcc_direct   — PCC (node 49) to m random non-neighbors

Each strategy returns a csr_matrix (the modified adjacency).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from numpy.random import Generator
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components

# GridResilience time grid (same as data_loader._T_WEEK)
_T_RAW = np.linspace(0, 604800 - 1800, 336)[:-24]
_T_WEEK = _T_RAW[48:]  # 264 timesteps


def compute_node_power_stats(
    cons_interps: Sequence[interp1d],
    pv_interps: Sequence[interp1d],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-node power statistics over the 264-step time grid.

    Returns
    -------
    P_max_per_node : ndarray, shape (n_total,)
        max(|P_i(t)|) over all timesteps for each node (houses + PCC).
    P_sign_per_node : ndarray, shape (n_total,)
        sign(mean(P_i(t))) over all timesteps. +1 = net source, -1 = net sink.
    """
    n_houses = len(cons_interps)
    penetration = len(pv_interps)
    n_total = n_houses + 1
    n_t = len(_T_WEEK)

    # Build (n_total, n_t) power matrix
    P = np.zeros((n_total, n_t))
    for t_idx, t_sec in enumerate(_T_WEEK):
        for i in range(n_houses):
            consumption = float(cons_interps[i](t_sec))
            if i < penetration:
                generation = float(pv_interps[i](t_sec))
                P[i, t_idx] = generation - consumption
            else:
                P[i, t_idx] = -consumption
        P[n_houses, t_idx] = -np.sum(P[:n_houses, t_idx])

    P_max_per_node = np.max(np.abs(P), axis=1)
    P_mean = np.mean(P, axis=1)
    P_sign_per_node = np.sign(P_mean)

    return P_max_per_node, P_sign_per_node


def _get_non_edges(A_lil: lil_matrix) -> list[tuple[int, int]]:
    """Get all non-edges (i < j) excluding self-loops."""
    n = A_lil.shape[0]
    non_edges = []
    for i in range(n):
        neighbors_i = set(A_lil.rows[i])
        for j in range(i + 1, n):
            if j not in neighbors_i:
                non_edges.append((i, j))
    return non_edges


def _add_edges(A_lil: lil_matrix, edge_list: list[tuple[int, int]]) -> csr_matrix:
    """Add edges (symmetric) and convert to csr."""
    for i, j in edge_list:
        A_lil[i, j] = 1.0
        A_lil[j, i] = 1.0
    return A_lil.tocsr()


def select_edges_random(
    A_lil: lil_matrix,
    m: int,
    P_max_per_node: np.ndarray,
    P_sign_per_node: np.ndarray,
    rng: Generator,
) -> csr_matrix:
    """Strategy 1: Add m random edges among non-connected pairs."""
    non_edges = _get_non_edges(A_lil)
    if len(non_edges) < m:
        warnings.warn(f"Only {len(non_edges)} non-edges available, adding all")
        m = len(non_edges)
    chosen_idx = rng.choice(len(non_edges), size=m, replace=False)
    chosen = [non_edges[idx] for idx in chosen_idx]
    return _add_edges(A_lil, chosen)


def select_edges_max_power(
    A_lil: lil_matrix,
    m: int,
    P_max_per_node: np.ndarray,
    P_sign_per_node: np.ndarray,
    rng: Generator,
) -> csr_matrix:
    """Strategy 2: Add m edges connecting highest-power non-connected pairs."""
    non_edges = _get_non_edges(A_lil)
    if len(non_edges) < m:
        warnings.warn(f"Only {len(non_edges)} non-edges available, adding all")
        m = len(non_edges)
    scores = [P_max_per_node[i] + P_max_per_node[j] for i, j in non_edges]
    top_idx = np.argsort(scores)[-m:][::-1]
    chosen = [non_edges[idx] for idx in top_idx]
    return _add_edges(A_lil, chosen)


def select_edges_score(
    A_lil: lil_matrix,
    m: int,
    P_max_per_node: np.ndarray,
    P_sign_per_node: np.ndarray,
    rng: Generator,
) -> csr_matrix:
    """
    Strategy 3: Score-based — |P_i|/(d_i*(d_i+1)) + |P_j|/(d_j*(d_j+1)),
    opposite-sign only. Targets high-power, low-degree nodes of opposite type.
    """
    n = A_lil.shape[0]
    degrees = np.array([len(A_lil.rows[i]) for i in range(n)])

    non_edges = _get_non_edges(A_lil)
    # Filter to opposite-sign pairs
    candidates = [(i, j) for i, j in non_edges
                   if P_sign_per_node[i] * P_sign_per_node[j] < 0]

    if len(candidates) < m:
        warnings.warn(
            f"Score strategy: only {len(candidates)} opposite-sign non-edges "
            f"(need {m}). Using all available."
        )
        m = len(candidates)

    if m == 0:
        return A_lil.tocsr()

    scores = []
    for i, j in candidates:
        di = degrees[i]
        dj = degrees[j]
        s_i = P_max_per_node[i] / (di * (di + 1)) if di > 0 else P_max_per_node[i]
        s_j = P_max_per_node[j] / (dj * (dj + 1)) if dj > 0 else P_max_per_node[j]
        scores.append(s_i + s_j)

    top_idx = np.argsort(scores)[-m:][::-1]
    chosen = [candidates[idx] for idx in top_idx]
    return _add_edges(A_lil, chosen)


def select_edges_pcc_direct(
    A_lil: lil_matrix,
    m: int,
    P_max_per_node: np.ndarray,
    P_sign_per_node: np.ndarray,
    rng: Generator,
) -> csr_matrix:
    """Strategy 4: Connect PCC (last node) to m random non-neighbors."""
    n = A_lil.shape[0]
    pcc_idx = n - 1
    pcc_neighbors = set(A_lil.rows[pcc_idx])
    non_neighbors = [j for j in range(n - 1) if j not in pcc_neighbors]

    if len(non_neighbors) < m:
        warnings.warn(f"PCC has only {len(non_neighbors)} non-neighbors, adding all")
        m = len(non_neighbors)

    chosen_idx = rng.choice(len(non_neighbors), size=m, replace=False)
    chosen = [(pcc_idx, non_neighbors[idx]) for idx in chosen_idx]
    return _add_edges(A_lil, chosen)


# Registry for strategy lookup
STRATEGIES = {
    "random": select_edges_random,
    "max_power": select_edges_max_power,
    "score": select_edges_score,
    "pcc_direct": select_edges_pcc_direct,
}


def verify_edge_addition(A_original: csr_matrix, A_modified: csr_matrix, m: int):
    """Verify edge addition correctness. Raises AssertionError on failure."""
    # Correct number of new edges (symmetric → 2*m new nonzeros)
    expected_nnz = A_original.nnz + 2 * m
    actual_nnz = A_modified.nnz
    assert actual_nnz == expected_nnz, (
        f"nnz mismatch: expected {expected_nnz}, got {actual_nnz} "
        f"(original={A_original.nnz}, m={m})"
    )

    # Symmetry
    diff = A_modified - A_modified.T
    assert diff.nnz == 0, f"Asymmetric: {diff.nnz} differing entries"

    # No self-loops
    diag_sum = A_modified.diagonal().sum()
    assert diag_sum == 0, f"Self-loops detected: diagonal sum = {diag_sum}"

    # Connectivity
    n_comp, _ = connected_components(A_modified, directed=False)
    assert n_comp == 1, f"Graph has {n_comp} connected components"

    # No multi-edges: all values should be 0 or 1
    assert A_modified.max() <= 1.0, f"Multi-edge detected: max value = {A_modified.max()}"
