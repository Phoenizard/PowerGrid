"""
Cascade failure engine for SQ3 — DC power flow and swing equation models.

Two cascade models following Smith et al. (2022):
  1. DC cascade: Linear power flow, overload-only removal, fast (~1ms)
  2. Swing cascade: Nonlinear ODE, overload + desync, slow (~5-30s)

Key parameters (from GridResilience/src/):
  - synctol = 3.0 rad/s (L2 norm of ω for desync)
  - steady-state tol = 1e-6
  - ODE rtol=atol=1e-8, max integration 500s
  - α bisection: start 0.01, stepsize 0.3, converge |step|<1e-4
  - Flow normalization: |f_ij|/f_max_initial (dimensionless α)
  - Power rebalancing: homogeneous half-split among sources/sinks
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")
SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
SQ4_DIR = os.path.join(PROJECT_ROOT, "sq4_data")

for _d in (PAPER_DIR, SQ2_DIR, SQ4_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from model import swing_equation_wrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """Result of a single cascade run."""
    S: float              # surviving edge fraction (0..1)
    n_overload: int       # edges removed by overload
    n_desync: int         # edges remaining when desync triggered
    cascade_depth: int    # recursion depth reached


# ---------------------------------------------------------------------------
# Function 1: find_steady_state
# ---------------------------------------------------------------------------

def find_steady_state(
    A_csr: csr_matrix,
    P: np.ndarray,
    kappa: float,
    gamma: float = 1.0,
    tol: float = 1e-6,
    max_time: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Integrate swing ODE from small perturbation until steady state.

    Returns (theta, omega, converged).
    Convergence: ||P - kappa*A*sin(Dtheta)||_2 < tol  (Smith's residual criterion)
    """
    n = len(P)
    rng = np.random.default_rng(42)
    theta0 = rng.uniform(-0.01, 0.01, n)
    omega0 = np.zeros(n)
    state0 = np.concatenate([theta0, omega0])

    # Integrate in chunks to check convergence periodically
    chunk = 50.0
    t_current = 0.0
    state = state0.copy()

    while t_current < max_time:
        t_end = min(t_current + chunk, max_time)
        sol = solve_ivp(
            swing_equation_wrapper,
            [t_current, t_end],
            state,
            args=(P, kappa, A_csr, gamma),
            method='RK45',
            max_step=1.0,
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            break
        state = sol.y[:, -1]
        t_current = t_end

        # Check residual convergence (Smith criterion)
        theta = state[:n]
        residual = _compute_residual(A_csr, theta, P, kappa)
        if np.linalg.norm(residual) < tol:
            return theta, state[n:], True

    theta = state[:n]
    omega = state[n:]
    return theta, omega, False


def _compute_residual(
    A_csr: csr_matrix,
    theta: np.ndarray,
    P: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Compute P - kappa * A * sin(Dtheta) for each node."""
    n = len(P)
    coupling = np.zeros(n)
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            coupling[i] += A_csr.data[idx] * np.sin(theta[i] - theta[j])
    return P - kappa * coupling


# ---------------------------------------------------------------------------
# Function 2: compute_edge_flows
# ---------------------------------------------------------------------------

def compute_edge_flows(
    A_csr: csr_matrix,
    theta: np.ndarray,
    kappa: float,
) -> tuple[dict, float, tuple | None]:
    """
    Compute edge flows f_ij = kappa * sin(theta_i - theta_j).

    Returns (flows_dict, f_max, trigger_edge).
    trigger_edge = (i, j) with maximum |f_ij|, or None if no edges.
    """
    n = A_csr.shape[0]
    flows: dict[tuple[int, int], float] = {}
    f_max = 0.0
    trigger_edge = None

    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            if i < j:  # each edge once
                f_ij = kappa * np.sin(theta[i] - theta[j])
                flows[(i, j)] = f_ij
                if abs(f_ij) > f_max:
                    f_max = abs(f_ij)
                    trigger_edge = (i, j)

    return flows, f_max, trigger_edge


def compute_dc_fmax(
    A_csr: csr_matrix,
    P: np.ndarray,
) -> tuple[dict, float, tuple | None]:
    """
    Compute f_max from DC power flow (f_ij = theta_i - theta_j).

    This is the correct normalization for DC cascade, unlike compute_edge_flows
    which uses swing flows f_ij = kappa * sin(theta_i - theta_j).

    Returns (flows_dict, f_max_dc, trigger_edge).
    """
    flows = _dc_power_flow(A_csr, P)
    f_max = 0.0
    trigger_edge = None
    for (i, j), f in flows.items():
        if abs(f) > f_max:
            f_max = abs(f)
            trigger_edge = (i, j)
    return flows, f_max, trigger_edge


# ---------------------------------------------------------------------------
# Function 3: rebalance_power
# ---------------------------------------------------------------------------

def rebalance_power(P_fragment: np.ndarray, tol: float = 1e-5) -> np.ndarray | None:
    """
    Smith's homogeneous half-split rebalancing.

    Distribute surplus/deficit equally among sources and sinks.
    Returns None if fragment has no sources or no sinks (dead fragment).
    """
    source_mask = P_fragment > tol
    sink_mask = P_fragment < -tol

    n_sources = np.sum(source_mask)
    n_sinks = np.sum(sink_mask)

    if n_sources == 0 or n_sinks == 0:
        return None

    imbalance = np.sum(P_fragment)
    delta = imbalance / 2.0

    P_out = P_fragment.copy()
    P_out[source_mask] -= delta / n_sources
    P_out[sink_mask] -= delta / n_sinks

    return P_out


# ---------------------------------------------------------------------------
# Function 4: DC cascade
# ---------------------------------------------------------------------------

def _dc_power_flow(A_csr: csr_matrix, P: np.ndarray) -> dict[tuple[int, int], float]:
    """
    DC power flow: L*theta = P, f_ij = theta_i - theta_j.

    Uses Laplacian pseudoinverse with node 0 as ground (theta_0=0).
    """
    n = A_csr.shape[0]
    if n <= 1:
        return {}

    # Build Laplacian
    L = np.zeros((n, n))
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            L[i, j] = -A_csr.data[idx]
            L[i, i] += A_csr.data[idx]

    # Ground node 0: solve reduced system L_red * theta_red = P_red
    L_red = L[1:, 1:]
    P_red = P[1:]

    try:
        theta_red = np.linalg.solve(L_red, P_red)
    except np.linalg.LinAlgError:
        # Singular — shouldn't happen for connected graph
        return {}

    theta = np.zeros(n)
    theta[1:] = theta_red

    # Compute edge flows
    flows: dict[tuple[int, int], float] = {}
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            if i < j:
                flows[(i, j)] = theta[i] - theta[j]

    return flows


def _run_cascade_dc_inner(
    A_csr: csr_matrix,
    P: np.ndarray,
    alpha: float,
    f_max_initial: float,
    depth: int,
) -> tuple[int, int, int]:
    """
    Internal recursive DC cascade. Returns (surviving_edges, overloaded_edges, max_depth).

    Counts absolute edge numbers, not fractions, to avoid accumulation errors.
    """
    n = A_csr.shape[0]
    if n <= 1 or A_csr.nnz == 0:
        return 0, 0, depth

    # DC power flow
    flows = _dc_power_flow(A_csr, P)
    if not flows:
        return 0, 0, depth

    # Find overloaded edges
    threshold = alpha * f_max_initial
    overloaded = [(i, j) for (i, j), f in flows.items() if abs(f) > threshold]

    if not overloaded:
        # Stable — all edges in this fragment survive
        return len(flows), 0, depth

    # Remove overloaded edges
    A_lil = lil_matrix(A_csr)
    for i, j in overloaded:
        A_lil[i, j] = 0
        A_lil[j, i] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    # Find connected components
    n_comp, labels = connected_components(A_new, directed=False)

    # Recurse on each component
    total_surviving = 0
    total_overloaded = len(overloaded)
    max_depth = depth

    for comp_id in range(n_comp):
        nodes = np.where(labels == comp_id)[0]
        if len(nodes) <= 1:
            continue

        # Extract subgraph
        A_sub = A_new[np.ix_(nodes, nodes)].tocsr()
        P_sub = P[nodes]

        # Rebalance
        P_sub_bal = rebalance_power(P_sub)
        if P_sub_bal is None:
            # Only-sources or only-sinks fragment cannot operate → edges fail.
            # Exception: all-passive fragment (all |P_i| < tol) → no flow, edges trivially survive.
            if np.all(np.abs(P_sub) < 1e-5):
                total_surviving += A_sub.nnz // 2
            # else: dead fragment → edges do NOT survive
            continue

        surv, ovrl, d = _run_cascade_dc_inner(
            A_sub, P_sub_bal, alpha, f_max_initial,
            depth=depth + 1,
        )
        total_surviving += surv
        total_overloaded += ovrl
        max_depth = max(max_depth, d)

    return total_surviving, total_overloaded, max_depth


def _run_cascade_dc_tracked_inner(
    A_csr: csr_matrix,
    P: np.ndarray,
    alpha: float,
    f_max_initial: float,
    depth: int,
    global_nodes: np.ndarray,
) -> tuple[int, int, int, set]:
    """
    Internal recursive DC cascade with edge tracking.
    Returns (surviving_edges, overloaded_edges, max_depth, surviving_edge_set).
    surviving_edge_set contains (i,j) tuples in GLOBAL node indices.
    """
    n = A_csr.shape[0]
    if n <= 1 or A_csr.nnz == 0:
        return 0, 0, depth, set()

    flows = _dc_power_flow(A_csr, P)
    if not flows:
        return 0, 0, depth, set()

    threshold = alpha * f_max_initial
    overloaded = [(i, j) for (i, j), f in flows.items() if abs(f) > threshold]

    if not overloaded:
        # Stable — all edges survive; map local indices to global
        surviving = set()
        for (i, j) in flows:
            gi, gj = global_nodes[i], global_nodes[j]
            surviving.add((min(gi, gj), max(gi, gj)))
        return len(flows), 0, depth, surviving

    # Remove overloaded edges
    A_lil = lil_matrix(A_csr)
    for i, j in overloaded:
        A_lil[i, j] = 0
        A_lil[j, i] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    n_comp, labels = connected_components(A_new, directed=False)

    total_surviving = 0
    total_overloaded = len(overloaded)
    max_depth = depth
    all_surviving_edges: set = set()

    for comp_id in range(n_comp):
        nodes = np.where(labels == comp_id)[0]
        if len(nodes) <= 1:
            continue

        A_sub = A_new[np.ix_(nodes, nodes)].tocsr()
        P_sub = P[nodes]
        sub_global = global_nodes[nodes]

        P_sub_bal = rebalance_power(P_sub)
        if P_sub_bal is None:
            if np.all(np.abs(P_sub) < 1e-5):
                n_edges = A_sub.nnz // 2
                total_surviving += n_edges
                # Add these surviving edges with global indices
                for li in range(A_sub.shape[0]):
                    for idx in range(A_sub.indptr[li], A_sub.indptr[li + 1]):
                        lj = A_sub.indices[idx]
                        if li < lj:
                            gi, gj = sub_global[li], sub_global[lj]
                            all_surviving_edges.add((min(gi, gj), max(gi, gj)))
            continue

        surv, ovrl, d, sub_edges = _run_cascade_dc_tracked_inner(
            A_sub, P_sub_bal, alpha, f_max_initial,
            depth=depth + 1,
            global_nodes=sub_global,
        )
        total_surviving += surv
        total_overloaded += ovrl
        max_depth = max(max_depth, d)
        all_surviving_edges.update(sub_edges)

    return total_surviving, total_overloaded, max_depth, all_surviving_edges


def run_cascade_dc_tracked(
    A_csr: csr_matrix,
    P: np.ndarray,
    alpha: float,
    f_max_initial: float,
) -> tuple[CascadeResult, set]:
    """
    Like run_cascade_dc but also returns the set of surviving edges
    as (i,j) tuples with i < j in global node indices.
    """
    total_edges = A_csr.nnz // 2
    if total_edges == 0:
        return CascadeResult(S=0.0, n_overload=0, n_desync=0, cascade_depth=0), set()

    global_nodes = np.arange(A_csr.shape[0])
    surviving, overloaded, max_depth, edge_set = _run_cascade_dc_tracked_inner(
        A_csr, P, alpha, f_max_initial, depth=0, global_nodes=global_nodes,
    )

    S = surviving / total_edges
    result = CascadeResult(
        S=S,
        n_overload=overloaded,
        n_desync=0,
        cascade_depth=max_depth,
    )
    return result, edge_set


def run_cascade_dc(
    A_csr: csr_matrix,
    P: np.ndarray,
    alpha: float,
    f_max_initial: float,
) -> CascadeResult:
    """
    DC power flow cascade (Smith's fracture!()).

    Parameters
    ----------
    A_csr : adjacency matrix (symmetric, unit weights)
    P : power vector (must sum to ~0)
    alpha : overload threshold (dimensionless, relative to f_max_initial)
    f_max_initial : maximum flow magnitude at initial steady state
    """
    total_edges = A_csr.nnz // 2
    if total_edges == 0:
        return CascadeResult(S=0.0, n_overload=0, n_desync=0, cascade_depth=0)

    surviving, overloaded, max_depth = _run_cascade_dc_inner(
        A_csr, P, alpha, f_max_initial, depth=0
    )

    S = surviving / total_edges
    return CascadeResult(
        S=S,
        n_overload=overloaded,
        n_desync=0,
        cascade_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Function 5: Swing cascade
# ---------------------------------------------------------------------------

def run_cascade_swing(
    A_csr: csr_matrix,
    P: np.ndarray,
    kappa: float,
    alpha: float,
    f_max_initial: float,
    gamma: float = 1.0,
    synctol: float = 3.0,
    theta0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
    max_time: float = 100.0,
    dt_check: float = 1.0,
) -> CascadeResult:
    """
    Swing equation cascade (Smith's swingfracturewithbreakdown!()).

    Integrate ODE in dt_check segments, checking overload and desync between.
    """
    n = A_csr.shape[0]
    total_edges_initial = A_csr.nnz // 2

    if n <= 1 or A_csr.nnz == 0:
        return CascadeResult(S=0.0, n_overload=0, n_desync=0, cascade_depth=0)

    if theta0 is None:
        theta0 = np.zeros(n)
    if omega0 is None:
        omega0 = np.zeros(n)

    state = np.concatenate([theta0.copy(), omega0.copy()])
    A_current = A_csr.copy()
    n_overload = 0
    n_desync = 0
    cascade_depth = 0
    t_current = 0.0

    while t_current < max_time:
        t_end = min(t_current + dt_check, max_time)

        # Integrate one segment
        sol = solve_ivp(
            swing_equation_wrapper,
            [t_current, t_end],
            state,
            args=(P, kappa, A_current, gamma),
            method='RK45',
            max_step=0.5,
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            break

        state = sol.y[:, -1]
        t_current = t_end
        theta = state[:n]
        omega = state[n:]

        # Check desync
        omega_norm = np.linalg.norm(omega)
        if omega_norm > synctol:
            remaining_edges = A_current.nnz // 2
            n_desync = remaining_edges
            S = 0.0  # desync = total failure
            return CascadeResult(
                S=S,
                n_overload=n_overload,
                n_desync=n_desync,
                cascade_depth=cascade_depth,
            )

        # Check overload
        flows, _, _ = compute_edge_flows(A_current, theta, kappa)
        threshold = alpha * f_max_initial
        overloaded = [(i, j) for (i, j), f in flows.items() if abs(f) > threshold]

        if overloaded:
            A_lil = lil_matrix(A_current)
            for i, j in overloaded:
                A_lil[i, j] = 0
                A_lil[j, i] = 0
            A_current = A_lil.tocsr()
            A_current.eliminate_zeros()
            n_overload += len(overloaded)
            cascade_depth += 1

            if A_current.nnz == 0:
                return CascadeResult(
                    S=0.0,
                    n_overload=n_overload,
                    n_desync=0,
                    cascade_depth=cascade_depth,
                )

            # Check connectivity
            n_comp, labels = connected_components(A_current, directed=False)

            if n_comp > 1:
                # For swing cascade, pick largest component and continue
                # (simpler than recursing on each fragment)
                comp_sizes = np.bincount(labels)
                largest = np.argmax(comp_sizes)
                nodes = np.where(labels == largest)[0]

                # Extract subgraph for largest component
                A_sub = A_current[np.ix_(nodes, nodes)].tocsr()
                P_sub = rebalance_power(P[nodes])
                if P_sub is None:
                    remaining = A_current.nnz // 2
                    S = remaining / total_edges_initial if total_edges_initial > 0 else 0.0
                    return CascadeResult(
                        S=S,
                        n_overload=n_overload,
                        n_desync=0,
                        cascade_depth=cascade_depth,
                    )

                # Remap state to subgraph
                n_new = len(nodes)
                theta_sub = theta[nodes]
                omega_sub = omega[nodes]
                state = np.concatenate([theta_sub, omega_sub])
                A_current = A_sub
                P = P_sub
                n = n_new
                continue

        # Check convergence (residual criterion)
        residual = _compute_residual(A_current, state[:n], P, kappa)
        if np.linalg.norm(residual) < 1e-6:
            surviving = A_current.nnz // 2
            S = surviving / total_edges_initial if total_edges_initial > 0 else 1.0
            return CascadeResult(
                S=S,
                n_overload=n_overload,
                n_desync=0,
                cascade_depth=cascade_depth,
            )

    # Timed out — report current state
    surviving = A_current.nnz // 2
    S = surviving / total_edges_initial if total_edges_initial > 0 else 0.0
    return CascadeResult(
        S=S,
        n_overload=n_overload,
        n_desync=0,
        cascade_depth=cascade_depth,
    )


# ---------------------------------------------------------------------------
# Function 6: alpha sweep
# ---------------------------------------------------------------------------

def sweep_alpha(
    A_csr: csr_matrix,
    P: np.ndarray,
    kappa: float,
    f_max: float,
    n_points: int = 20,
    alpha_range: tuple[float, float] = (0.1, 2.5),
    use_swing: bool = False,
    theta0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run cascade at n_points alpha values, return (alpha_array, S_array).

    alpha_range is relative to f_max (i.e., alpha_range=(0.1, 2.5) means
    test alpha from 0.1*f_max to 2.5*f_max in normalized space).
    We pass alpha directly since f_max is already factored in.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    S_values = np.zeros(n_points)

    for idx, alpha in enumerate(alphas):
        if use_swing:
            result = run_cascade_swing(
                A_csr, P, kappa, alpha, f_max,
                theta0=theta0, omega0=omega0,
            )
        else:
            result = run_cascade_dc(A_csr, P, alpha, f_max)
        S_values[idx] = result.S

    return alphas, S_values


# ---------------------------------------------------------------------------
# Function 7: find alpha_c via bisection
# ---------------------------------------------------------------------------

def find_alpha_c(
    A_csr: csr_matrix,
    P: np.ndarray,
    kappa: float = 0.0,
    f_max: float = 1.0,
    use_swing: bool = False,
    tol: float = 1e-3,
    theta0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Smith's adaptive bisection for alpha_c (critical overload threshold).

    Phase 1: Linear advance from alpha=0.01 with step=0.3 until S crosses 0.5.
    Phase 2: Standard bisection on [alpha_low, alpha_high] bracket.

    Returns (alpha_c, S_at_alpha_c).
    """
    def _cascade(a):
        if use_swing:
            return run_cascade_swing(
                A_csr, P, kappa, a, f_max,
                theta0=theta0, omega0=omega0,
            )
        else:
            return run_cascade_dc(A_csr, P, a, f_max)

    # Phase 1: Linear advance to find bracket [alpha_low, alpha_high]
    # where S crosses 0.5 (S <= 0.5 at alpha_low, S > 0.5 at alpha_high)
    step = 0.3
    alpha_low = None
    alpha_high = None
    alpha = 0.01

    while alpha < 20.0:
        result = _cascade(alpha)
        if result.S > 0.5:
            alpha_high = alpha
            alpha_low = max(alpha - step, 1e-6)
            break
        alpha += step

    if alpha_high is None:
        # Never crossed — cascade always fails (S <= 0.5 everywhere)
        return alpha, 0.0

    # Phase 2: Standard bisection on [alpha_low, alpha_high]
    last_S = result.S
    for _ in range(50):
        if alpha_high - alpha_low < tol:
            break
        mid = (alpha_low + alpha_high) / 2.0
        result = _cascade(mid)
        last_S = result.S
        if result.S > 0.5:
            alpha_high = mid
        else:
            alpha_low = mid

    alpha_c = (alpha_low + alpha_high) / 2.0
    return alpha_c, last_S


# ---------------------------------------------------------------------------
# Convenience: full pipeline for one instance
# ---------------------------------------------------------------------------

def cascade_pipeline_dc(
    A_csr: csr_matrix,
    P: np.ndarray,
    kappa: float,
    gamma: float = 1.0,
    n_alpha_points: int = 20,
    alpha_range: tuple[float, float] = (0.1, 2.5),
    bisection_tol: float = 1e-3,
) -> dict:
    """
    Full DC cascade pipeline for one instance:
    1. Find steady state (swing ODE — for swing validation)
    2. Compute DC f_max (for DC cascade) and swing f_max (for reference)
    3. Alpha sweep using DC f_max
    4. Alpha_c bisection using DC f_max

    Returns dict with keys: theta, omega, converged, f_max_dc, f_max_swing,
                            alphas, S_values, alpha_c, S_at_alpha_c
    """
    # Step 1: steady state (swing ODE)
    theta, omega, converged = find_steady_state(A_csr, P, kappa, gamma)

    # Step 2a: DC f_max (correct normalization for DC cascade)
    dc_flows, f_max_dc, dc_trigger = compute_dc_fmax(A_csr, P)

    # Step 2b: Swing f_max (for reference / swing validation)
    swing_flows, f_max_swing, sw_trigger = compute_edge_flows(A_csr, theta, kappa)

    if f_max_dc < 1e-12:
        return {
            "theta": theta, "omega": omega, "converged": converged,
            "f_max_dc": 0.0, "f_max_swing": f_max_swing,
            "f_max": 0.0,  # backward compat
            "alphas": np.array([]), "S_values": np.array([]),
            "alpha_c": np.nan, "S_at_alpha_c": np.nan,
        }

    # Step 3: alpha sweep (DC cascade uses DC f_max)
    alphas, S_values = sweep_alpha(
        A_csr, P, kappa, f_max_dc,
        n_points=n_alpha_points,
        alpha_range=alpha_range,
        use_swing=False,
    )

    # Step 4: alpha_c bisection (DC cascade uses DC f_max)
    alpha_c, S_ac = find_alpha_c(
        A_csr, P, kappa, f_max_dc,
        use_swing=False,
        tol=bisection_tol,
    )

    return {
        "theta": theta,
        "omega": omega,
        "converged": converged,
        "f_max_dc": f_max_dc,
        "f_max_swing": f_max_swing,
        "f_max": f_max_dc,  # backward compat — DC is now primary
        "trigger_edge": dc_trigger,
        "alphas": alphas,
        "S_values": S_values,
        "alpha_c": alpha_c,
        "S_at_alpha_c": S_ac,
    }
