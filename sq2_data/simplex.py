"""
Simplex coordinate computation for SQ2 experiments.

Computes continuous source/sink/passive densities (sigma_s, sigma_d, sigma_p)
from the full power vector (including PCC node) using max-normalization.

Reference: GridResilience/scripts/powerreader.py:continuoussourcesinkcounter()
           (lines 136-145)

Note: The full Pvec (houses + PCC) is used, matching GridResilience behaviour
where get_power_vec() includes PCC and add_trajectory_point() passes the
full vector to continuoussourcesinkcounter().
"""

from __future__ import annotations

import numpy as np


def compute_simplex_coordinates(Pvec: np.ndarray) -> tuple[float, float, float]:
    """
    Compute continuous simplex coordinates (sigma_s, sigma_d, sigma_p).

    Strict replica of GridResilience continuoussourcesinkcounter():
        sigma_s = sum(sources) / (n * max(P))
        sigma_d = sum(|sinks|) / (n * |min(P)|)
        sigma_p = 1 - sigma_s - sigma_d

    Parameters
    ----------
    Pvec : ndarray, shape (n,)
        Power values for ALL nodes (houses + PCC).

    Returns
    -------
    sigma_s : float
        Source density.
    sigma_d : float
        Sink (demand) density.
    sigma_p : float
        Passive density.
    """
    n = len(Pvec)
    largest_source = np.max(Pvec)
    largest_sink = np.abs(np.min(Pvec))

    source_terms = Pvec[Pvec > 0.0]
    sink_terms = Pvec[Pvec < 0.0]

    # Edge-case: all nodes non-positive → no sources
    if largest_source <= 0.0:
        sigma_s = 0.0
    else:
        sigma_s = np.sum(source_terms) / (n * largest_source)

    # Edge-case: all nodes non-negative → no sinks
    if largest_sink <= 0.0:
        sigma_d = 0.0
    else:
        sigma_d = np.sum(np.abs(sink_terms)) / (n * largest_sink)

    sigma_p = 1.0 - sigma_s - sigma_d

    return sigma_s, sigma_d, sigma_p


def compute_simplex_trajectory(P: np.ndarray) -> np.ndarray:
    """
    Compute simplex coordinates for each timestep.

    Uses ALL nodes (houses + PCC) per GridResilience convention.

    Parameters
    ----------
    P : ndarray, shape (n_nodes, T)
        Full power vector (houses + PCC).

    Returns
    -------
    trajectory : ndarray, shape (T, 3)
        Columns: (sigma_s, sigma_d, sigma_p) per timestep.
    """
    T = P.shape[1]
    trajectory = np.zeros((T, 3))

    for t in range(T):
        Pvec = P[:, t]
        sigma_s, sigma_d, sigma_p = compute_simplex_coordinates(Pvec)
        trajectory[t] = [sigma_s, sigma_d, sigma_p]

    return trajectory
