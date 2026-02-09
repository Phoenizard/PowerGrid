"""
Simplex coordinate computation for SQ2 experiments.

Computes continuous source/sink/passive fractions (eta+, eta-, eta_p)
from node power values. Uses ONLY house nodes (excluding PCC).

Reference: GridResilience/scripts/powerreader.py:continuoussourcesinkcounter()
"""

from __future__ import annotations

import numpy as np


def compute_simplex_coordinates(
    P_houses: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute simplex coordinates (eta+, eta-, eta_p) from house-node powers.

    eta+ = (1 / (n * max(P))) * sum_{x in P+} x
    eta- = (1 / (n * |min(P)|)) * sum_{x in P-} |x|
    eta_p = 1 - eta+ - eta-

    Parameters
    ----------
    P_houses : ndarray, shape (n_houses,)
        Power values for house nodes only (NOT including PCC).

    Returns
    -------
    eta_plus : float
        Generator fraction.
    eta_minus : float
        Consumer fraction.
    eta_p : float
        Passive fraction.
    """
    n = len(P_houses)

    largest_source = np.max(P_houses)
    largest_sink = np.abs(np.min(P_houses))

    # Handle edge cases: all zero, all positive, all negative
    if largest_source <= 0.0 and largest_sink <= 0.0:
        # All zeros → fully passive
        return 0.0, 0.0, 1.0

    source_terms = P_houses[P_houses > 0.0]
    sink_terms = P_houses[P_houses < 0.0]

    if largest_source > 0.0 and len(source_terms) > 0:
        eta_plus = np.sum(source_terms) / (n * largest_source)
    else:
        eta_plus = 0.0

    if largest_sink > 0.0 and len(sink_terms) > 0:
        eta_minus = np.sum(np.abs(sink_terms)) / (n * largest_sink)
    else:
        eta_minus = 0.0

    eta_p = 1.0 - eta_plus - eta_minus

    return eta_plus, eta_minus, eta_p


def compute_simplex_trajectory(
    P: np.ndarray,
    n_houses: int = 50,
) -> np.ndarray:
    """
    Compute simplex coordinates for each timestep.

    Parameters
    ----------
    P : ndarray, shape (n_nodes, T)
        Full power vector (including PCC at index n_houses).
    n_houses : int
        Number of house nodes (PCC excluded from simplex computation).

    Returns
    -------
    trajectory : ndarray, shape (T, 3)
        Columns: (eta_plus, eta_minus, eta_p) per timestep.
    """
    T = P.shape[1]
    trajectory = np.zeros((T, 3))

    for t in range(T):
        P_houses = P[:n_houses, t]
        eta_plus, eta_minus, eta_p = compute_simplex_coordinates(P_houses)
        trajectory[t] = [eta_plus, eta_minus, eta_p]

    return trajectory
