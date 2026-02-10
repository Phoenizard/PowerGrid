"""
Network generation for SQ2 microgrid model.

Generates a Watts-Strogatz small-world network for 50 houses,
then attaches a PCC (Point of Common Coupling) node as node 50.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def generate_network_with_pcc(
    n_houses: int = 50,
    k: int = 4,
    q: float = 0.1,
    n_pcc_links: int = 4,
    seed: int | None = None,
) -> csr_matrix:
    """
    Generate a Watts-Strogatz network with an additional PCC node.

    The PCC node (index n_houses) is connected to n_pcc_links randomly
    chosen house nodes.

    Parameters
    ----------
    n_houses : int
        Number of house nodes (default 50).
    k : int
        WS nearest-neighbor parameter.
    q : float
        WS rewiring probability.
    n_pcc_links : int
        Number of house nodes connected to PCC.
    seed : int or None
        Random seed.

    Returns
    -------
    A : csr_matrix, shape (n_houses + 1, n_houses + 1)
        Adjacency matrix (symmetric, binary).
    """
    rng = np.random.default_rng(seed)

    # Generate WS graph for house nodes
    ws_seed = int(rng.integers(0, 2**31)) if seed is not None else None
    G = nx.watts_strogatz_graph(n_houses, k, q, seed=ws_seed)

    # Get adjacency as (n_houses, n_houses)
    A_ws = nx.adjacency_matrix(G).astype(np.float64)

    # Extend to (n_houses+1, n_houses+1) with PCC node
    n_total = n_houses + 1
    A = lil_matrix((n_total, n_total), dtype=np.float64)
    A[:n_houses, :n_houses] = A_ws

    # Connect PCC to random house nodes
    pcc_idx = n_houses  # last node
    pcc_neighbors = rng.choice(n_houses, size=n_pcc_links, replace=False)
    for j in pcc_neighbors:
        A[pcc_idx, j] = 1.0
        A[j, pcc_idx] = 1.0

    return A.tocsr()
