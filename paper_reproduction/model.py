"""
Core physics model for power grid stability analysis.
Implements swing equation and critical coupling computation.
"""

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import config

# Try to import numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

@njit
def swing_equation_rhs(t, state, P, kappa, A_data, A_indices, A_indptr, n, gamma):
    """
    Right-hand side of the swing equation in first-order form.
    state = [theta_0, ..., theta_{n-1}, omega_0, ..., omega_{n-1}]

    d(theta_i)/dt = omega_i
    d(omega_i)/dt = P_i - gamma * omega_i - kappa * sum_j A_ij * sin(theta_i - theta_j)
    """
    theta = state[:n]
    omega = state[n:]

    dtheta_dt = omega
    domega_dt = np.zeros(n)

    for i in range(n):
        coupling_sum = 0.0
        # CSR matrix iteration
        for idx in range(A_indptr[i], A_indptr[i+1]):
            j = A_indices[idx]
            coupling_sum += A_data[idx] * np.sin(theta[i] - theta[j])
        domega_dt[i] = P[i] - gamma * omega[i] - kappa * coupling_sum

    result = np.empty(2*n)
    result[:n] = dtheta_dt
    result[n:] = domega_dt
    return result


def swing_equation_wrapper(t, state, P, kappa, A_csr, gamma):
    """Wrapper for scipy.integrate.solve_ivp"""
    n = len(P)
    return swing_equation_rhs(t, state, P, kappa,
                              A_csr.data, A_csr.indices, A_csr.indptr,
                              n, gamma)


def generate_network(n, k, q, seed=None):
    """
    Generate a Watts-Strogatz network.

    Parameters:
    -----------
    n : int - number of nodes
    k : int - each node is connected to k nearest neighbors in ring topology
    q : float - probability of rewiring each edge
    seed : int - random seed for reproducibility

    Returns:
    --------
    A : scipy.sparse.csr_matrix - adjacency matrix
    """
    from scipy.sparse import csr_matrix

    if seed is not None:
        G = nx.watts_strogatz_graph(n, k, q, seed=int(seed))
    else:
        G = nx.watts_strogatz_graph(n, k, q)

    A = nx.adjacency_matrix(G).astype(np.float64)
    return A


def assign_power(n, n_plus, n_minus, P_max, rng=None):
    """
    Assign power values to nodes.

    Parameters:
    -----------
    n : int - total number of nodes
    n_plus : int - number of generators (positive power)
    n_minus : int - number of consumers (negative power)
    P_max : float - maximum total power
    rng : numpy random generator

    Returns:
    --------
    P : ndarray - power values for each node
    """
    if rng is None:
        rng = np.random.default_rng()

    n_passive = n - n_plus - n_minus

    # Create power array
    P = np.zeros(n)

    # Randomly assign roles to nodes
    indices = rng.permutation(n)

    generator_indices = indices[:n_plus]
    consumer_indices = indices[n_plus:n_plus + n_minus]
    # passive_indices = indices[n_plus + n_minus:]  # P=0 by default

    # Assign power values
    if n_plus > 0:
        P[generator_indices] = P_max / n_plus
    if n_minus > 0:
        P[consumer_indices] = -P_max / n_minus

    return P


def check_convergence(state, n, tol=1e-4):
    """
    Check if the system has converged to a stable fixed point.
    Convergence criterion: max|omega_i| < tol
    """
    omega = state[n:]
    return np.max(np.abs(omega)) < tol


def check_power_balance(state, P, kappa, A_csr, tol=1e-5):
    """
    Check if power balance is achieved (steady-state condition).
    Matches the original paper's convergence criterion.
    Residual = P - κ * Σ_j A_ij * sin(θ_i - θ_j)
    """
    n = len(P)
    theta = state[:n]

    # Compute coupling term for each node
    coupling = np.zeros(n)
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i+1]):
            j = A_csr.indices[idx]
            coupling[i] += A_csr.data[idx] * np.sin(theta[i] - theta[j])

    residual = P - kappa * coupling
    return np.linalg.norm(residual) < tol


def compute_kappa_c(A_csr, P, config_params=None):
    """
    Compute critical coupling kappa_c using bisection.

    Parameters:
    -----------
    A_csr : scipy.sparse.csr_matrix - adjacency matrix
    P : ndarray - power values
    config_params : dict - optional parameters override

    Returns:
    --------
    kappa_c : float - critical coupling value
    """
    if config_params is None:
        config_params = {}

    n = len(P)
    gamma = config_params.get('gamma', config.GAMMA)
    kappa_min, kappa_max = config_params.get('kappa_range', config.KAPPA_RANGE)
    bisection_steps = config_params.get('bisection_steps', config.BISECTION_STEPS)
    t_integrate = config_params.get('t_integrate', config.T_INTEGRATE)
    conv_tol = config_params.get('conv_tol', config.CONV_TOL)
    max_step = config_params.get('max_step', config.ODE_MAX_STEP)

    # Initial conditions: small random perturbation
    rng = np.random.default_rng()
    theta0 = rng.uniform(-0.1, 0.1, n)
    omega0 = np.zeros(n)
    state0 = np.concatenate([theta0, omega0])

    def is_stable(kappa):
        """Check if system is stable at given kappa"""
        sol = solve_ivp(
            swing_equation_wrapper,
            [0, t_integrate],
            state0,
            args=(P, kappa, A_csr, gamma),
            method='RK45',
            max_step=max_step,
            rtol=1e-6,
            atol=1e-8
        )
        if sol.success:
            return check_convergence(sol.y[:, -1], n, conv_tol)
        return False

    # Bisection search
    for _ in range(bisection_steps):
        kappa_mid = (kappa_min + kappa_max) / 2
        if is_stable(kappa_mid):
            kappa_max = kappa_mid
        else:
            kappa_min = kappa_mid

    # Return the midpoint as the estimate
    kappa_c = (kappa_min + kappa_max) / 2
    return kappa_c


def compute_kappa_c_normalized(A_csr, P, P_max=1.0, config_params=None):
    """
    Compute normalized critical coupling kappa_c_bar = kappa_c / P_max
    """
    kappa_c = compute_kappa_c(A_csr, P, config_params)
    return kappa_c / P_max


def compute_ensemble_kappa_c(n_plus, n_minus, q, n_realizations, config_params=None, rng=None):
    """
    Compute mean and std of kappa_c over an ensemble of network realizations.

    Parameters:
    -----------
    n_plus : int - number of generators
    n_minus : int - number of consumers
    q : float - Watts-Strogatz rewiring probability
    n_realizations : int - number of realizations
    config_params : dict - optional parameters
    rng : numpy random generator

    Returns:
    --------
    mean_kappa_c : float
    std_kappa_c : float
    """
    if config_params is None:
        config_params = {}
    if rng is None:
        rng = np.random.default_rng()

    n = config_params.get('n', config.N)
    k = config_params.get('k', config.K)
    P_max = config_params.get('P_max', config.P_MAX)

    kappa_values = []

    for i in range(n_realizations):
        # Generate network
        A = generate_network(n, k, q, seed=rng.integers(0, 2**31))

        # Assign power
        P = assign_power(n, n_plus, n_minus, P_max, rng=rng)

        # Compute kappa_c
        kappa_c = compute_kappa_c_normalized(A, P, P_max, config_params)
        kappa_values.append(kappa_c)

    return np.mean(kappa_values), np.std(kappa_values)


if __name__ == '__main__':
    # Quick test
    print("Testing model.py...")
    A = generate_network(50, 4, 0.0, seed=42)
    P = assign_power(50, 17, 17, 1.0, rng=np.random.default_rng(42))
    kappa_c = compute_kappa_c_normalized(A, P, 1.0)
    print(f"Test kappa_c = {kappa_c:.4f}")
