"""
Direction 3 power allocation utilities.

All returned power vectors P satisfy:
1) abs(sum(P)) < 1e-10
2) generator entries are positive
3) consumer entries are negative
4) len(P) == n_plus + n_minus
"""

from __future__ import annotations

import numpy as np


def _get_normal_sampler(rng: np.random.Generator | None):
    if rng is None:
        return np.random.normal
    return rng.normal


def assign_power_heterogeneous(
    n_plus: int,
    n_minus: int,
    P_max: float,
    sigma_ratio: float,
    side: str = "gen",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Heterogeneous power allocation for generators or consumers.

    Returns an ordered vector:
    - P[0:n_plus] are generators (>0)
    - P[n_plus:n_plus+n_minus] are consumers (<0)
    """
    if n_plus <= 0 or n_minus <= 0:
        raise ValueError("n_plus and n_minus must be positive")
    if not (0.0 <= sigma_ratio < 1.0):
        raise ValueError(f"sigma_ratio must be in [0, 1), got {sigma_ratio}")
    if side not in {"gen", "con"}:
        raise ValueError(f"side must be 'gen' or 'con', got '{side}'")

    normal = _get_normal_sampler(rng)

    p_bar_gen = P_max / n_plus
    p_bar_con = P_max / n_minus
    delta_gen = 1e-4 * p_bar_gen
    delta_con = 1e-4 * p_bar_con

    if side == "gen":
        if sigma_ratio == 0.0:
            p_gen = np.full(n_plus, p_bar_gen)
        else:
            sigma = sigma_ratio * p_bar_gen
            eps = normal(0.0, sigma, n_plus)
            eps -= eps.mean()
            p_gen = p_bar_gen + eps
            p_gen = np.maximum(p_gen, delta_gen)
            p_gen *= P_max / p_gen.sum()
        p_con = np.full(n_minus, -p_bar_con)
    else:
        p_gen = np.full(n_plus, p_bar_gen)
        if sigma_ratio == 0.0:
            p_con = np.full(n_minus, -p_bar_con)
        else:
            sigma = sigma_ratio * p_bar_con
            eps = normal(0.0, sigma, n_minus)
            eps -= eps.mean()
            consumption = p_bar_con + eps
            consumption = np.maximum(consumption, delta_con)
            consumption *= P_max / consumption.sum()
            p_con = -consumption

    p = np.concatenate([p_gen, p_con])

    imbalance = float(abs(p.sum()))
    if imbalance >= 1e-10:
        raise AssertionError(f"Power imbalance too large: {imbalance:.3e}")
    if not np.all(p[:n_plus] > 0):
        raise AssertionError("Generator powers must be positive")
    if not np.all(p[n_plus:] < 0):
        raise AssertionError("Consumer powers must be negative")

    return p


def assign_power_centralized(
    n_plus: int,
    n_minus: int,
    P_max: float,
    r: float,
) -> np.ndarray:
    """
    Centralized generation allocation.

    One large station carries ratio r of total generation,
    remaining n_plus-1 stations share 1-r equally.

    Returns ordered vector where P[0] is the big station.
    """
    if n_plus <= 1:
        raise ValueError("n_plus must be > 1 for centralized allocation")
    if n_minus <= 0:
        raise ValueError("n_minus must be positive")
    if not (0.0 < r <= 1.0):
        raise ValueError(f"r must be in (0, 1], got {r}")

    p_big = r * P_max
    p_small = (1.0 - r) * P_max / (n_plus - 1)
    p_gen = np.full(n_plus, p_small)
    p_gen[0] = p_big

    p_con = np.full(n_minus, -P_max / n_minus)
    p = np.concatenate([p_gen, p_con])

    imbalance = float(abs(p.sum()))
    if imbalance >= 1e-10:
        raise AssertionError(f"Power imbalance too large: {imbalance:.3e}")
    if not np.all(p[:n_plus] > 0):
        raise AssertionError("Generator powers must be positive")
    if not np.all(p[n_plus:] < 0):
        raise AssertionError("Consumer powers must be negative")

    return p
