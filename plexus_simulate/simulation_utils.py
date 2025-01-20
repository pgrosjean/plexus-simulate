from typing import Dict, Union
import numpy as np
from scipy.signal import convolve
import networkx as nx

def calcium_dynamics(S: np.ndarray,
                     cal_params: Union[Dict[str, Union[float, int]], None] = None,
                     over_samp: int = 1,
                     ext_mult: int = 1):
    """
    Simulate calcium dynamics and fluorescence activity from spike data.
    Adapted from the code provided by the authors of the paper:
    Neural anatomy and optical microscopy (NAOMi) simulation for evaluating calcium imaging methods. J Neurosci Methods 358, 109173 (2021).
    Adapted from MATLAB code found at: https://bitbucket.org/adamshch/naomi_sim/src/master/

    Parameters
    ----------
    S : ndarray
        A K x nt array of spiking activity for K neurons over nt time steps.
    cal_params : dict, optional
        Parameters for calcium dynamics simulation:
        - ext_rate: Extrusion rate (default=1800)
        - ca_bind: Calcium binding constant (default=110)
        - ca_rest: Resting-state calcium concentration (default=50e-9)
        - ind_con: Indicator concentration (default=200e-6)
        - ca_dis: Calcium dissociation constant (default=290e-9)
        - ca_sat: Calcium saturation parameter (default=1)
        - sat_type: Saturation type ('double' or 'single', default='double')
        - dt: Sampling rate (default=1/25)
        - ca_amp: Calcium transient amplitude (default=0.09)
        - t_on: Rising time constant (default=0.1)
        - t_off: Falling time constant (default=1.5)
        - a_bind: Binding rate (default=3.5)
        - a_ubind: Unbinding rate (default=7)
    over_samp : int, optional
        Oversampling factor (default=1).
    ext_mult : float, optional
        Multiplier for extrusion rate (default=1).

    Returns
    -------
    CB : ndarray
        Bound calcium concentrations (K x nt array).
    C : ndarray
        Total calcium concentrations (K x nt array).
    F : ndarray
        Fluorescence activity (K x nt array).
    """
    # Default parameters
    if cal_params is None:
        cal_params = {
            "ext_rate": 1800,
            "ca_bind": 110,
            "ca_rest": 50e-9,
            "ind_con": 200e-6,
            "ca_dis": 290e-9,
            "ca_sat": 0.5,
            "sat_type": "double",
            "dt": 1/25,
            "ca_amp": 100000,
            "t_on": 0.1,
            "t_off": 1.8,
            "a_bind": 3.5,
            "a_ubind": 7,
        }

    ext_rate = ext_mult * cal_params["ext_rate"]
    ca_bind = cal_params["ca_bind"]
    ca_rest = cal_params["ca_rest"]
    ind_con = cal_params["ind_con"]
    ca_dis = cal_params["ca_dis"]
    ca_sat = cal_params["ca_sat"]
    sat_type = cal_params["sat_type"]
    dt = cal_params["dt"]
    ca_amp = cal_params["ca_amp"]
    t_on = cal_params["t_on"]
    t_off = cal_params["t_off"]
    a_bind = cal_params["a_bind"]
    a_ubind = cal_params["a_ubind"]

    if over_samp > 1:
        S = np.repeat(S, over_samp, axis=1)

    K, nt = S.shape
    C = np.zeros((K, nt), dtype=np.float32)
    CB = np.zeros_like(C)

    # Initialize calcium and bound calcium
    C[:, 0] = np.maximum(ca_rest, S[:, 0])

    if sat_type == "double":
        CB1 = np.zeros_like(C)
        CB2 = np.zeros_like(C)
        a = [a_bind, a_bind]
        b = [a_ubind, a_ubind]

        for t in range(1, nt):
            # Update total calcium
            C[:, t] = (
                C[:, t - 1]
                + dt * (b[0] * CB1[:, t - 1] + b[1] * CB2[:, t - 1])
                + (-dt * ext_rate * (C[:, t - 1] - CB1[:, t - 1] - CB2[:, t - 1] - ca_rest)
                   + S[:, t]) / (1 + ca_bind + (ind_con * ca_dis) / (C[:, t - 1] + ca_dis) ** 2)
            )
            # Apply saturation
            if 0 <= ca_sat < 1:
                C[:, t] = np.minimum(C[:, t], ca_dis * ca_sat / (1 - ca_sat))

            # Update bound calcium
            CB1[:, t] = (
                CB1[:, t - 1]
                + dt
                * (
                    -b[0] * CB1[:, t - 1]
                    + a[0]
                    * (C[:, t - 1] - CB1[:, t - 1] - CB2[:, t - 1])
                    * (ind_con - CB1[:, t - 1] - CB2[:, t - 1])
                )
            )
            CB2[:, t] = (
                CB2[:, t - 1]
                + dt
                * (
                    -b[1] * CB2[:, t - 1]
                    + a[1]
                    * (C[:, t - 1] - CB1[:, t - 1] - CB2[:, t - 1])
                    * (ind_con - CB1[:, t - 1] - CB2[:, t - 1])
                )
            )
        CB = CB1 + CB2

    elif sat_type == "single":
        for t in range(1, nt):
            # Update calcium
            C[:, t] = (
                C[:, t - 1]
                + dt * a_ubind * CB[:, t - 1]
                + (-dt * ext_rate * (C[:, t - 1] - CB[:, t - 1] - ca_rest) + S[:, t])
                / (1 + ca_bind + (ind_con * ca_dis) / (C[:, t - 1] + ca_dis) ** 2)
            )
            # Apply saturation
            if 0 <= ca_sat < 1:
                C[:, t] = np.minimum(C[:, t], ca_dis * ca_sat / (1 - ca_sat))
            # Update bound calcium
            CB[:, t] = CB[:, t - 1] + dt * (
                -a_ubind * CB[:, t - 1]
                + a_bind
                * (C[:, t - 1] - CB[:, t - 1])
                * (ind_con - CB[:, t - 1])
            )

    # Convolve with double-exponential kernel for fluorescence
    kernel_time = np.arange(0, 10 * t_off, dt)
    kernel = (
        ca_amp * (np.exp(-kernel_time / t_off) - np.exp(-kernel_time / t_on))
    )
    F = np.array([convolve(C[k, :] - ca_rest, kernel, mode="full")[:nt] for k in range(K)])

    return CB, C, F


def simulate_hawkes_discrete(n_nodes: int = 10,
                             k: int = 4,
                             p: float = 0.1,
                             T: int = 100,
                             delta_t: float = 0.1,
                             mu: float = 0.2,
                             alpha_base: float = 0.2,
                             beta: float = 0.7,
                             gamma: float = 0.4,
                             t_feedback_beta=0.2,
                             t_feedback_gamma=0.5,
                             lambda_max: float = 10.0,
                             mu_std: int = 0.1,
                             seed: int = 14):
    """
    Simulate a multivariate Hawkes process in discrete time with negative feedback,
    parallelized using numpy's vectorization, and applying a lambda_max cap before feedback.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the network.
    k: int
        Each node is connected to k nearest neighbors.
    p: float
        Probability of rewiring edges in the Watts-Strogatz network.
    T: int
        Total simulation time.
    delta_t: float
        Time step size.
    mu: float
        Baseline intensity for each node.
    alpha_base: float
        Baseline synaptic strength (excitation factor).
    beta: float
        Exponential decay rate of event influence.
    gamma: float
        Strength of the negative feedback.
    t_feedback_beta: float
        Time window for the exponential decay of the feedback for beta.
    t_feedback_gamma: float
        Time window for the feedback for gamma.
    lambda_max: float
        Maximum allowable intensity for each node.
    mu_std: float
        Standard deviation of the baseline intensity.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    events_by_bin: np.ndarray
        Array of event counts per node per time bin.
    lambdas_over_time: np.ndarray
        Array of lambda values for each node at each time step.
    network_graph: nx.Graph
        NetworkX graph object representing the underlying network.
    """
    np.random.seed(seed)

    # Generate Watts-Strogatz network
    network_graph = nx.watts_strogatz_graph(n_nodes, k, p)
    adj_matrix = nx.to_numpy_array(network_graph)
    adj_matrix = adj_matrix + np.eye(n_nodes)  # Add self-loops

    # Discretize time
    N_bins = int(T / delta_t)
    events_by_bin = np.zeros((n_nodes, N_bins), dtype=int)
    lambdas_over_time = np.zeros((n_nodes, N_bins))

    # Precompute time decay factors for beta
    feedback_window_beta_bins = int(t_feedback_beta / delta_t)
    time_decay_factors_beta = np.exp(-beta * np.arange(feedback_window_beta_bins) * delta_t)

    # Determine feedback window for gamma
    feedback_window_gamma_bins = int(t_feedback_gamma / delta_t)

    # Normal distribution for mu
    mu = np.random.normal(mu, mu_std, n_nodes)

    # Iterate over time bins
    for t in range(N_bins):
        # Compute intensities
        lambdas = mu.copy()

        if t > 0:
            # Exponential kernel for excitation
            recent_events_beta = events_by_bin[:, max(0, t - feedback_window_beta_bins):t]
            decayed_effects = recent_events_beta @ time_decay_factors_beta[:recent_events_beta.shape[1]]
            lambdas += alpha_base * adj_matrix @ decayed_effects

            # Negative feedback
            recent_events_gamma = events_by_bin[:, max(0, t - feedback_window_gamma_bins):t]
            recent_activity = np.sum(recent_events_gamma, axis=1)
            lambdas -= gamma * recent_activity

        # Cap the intensity before applying feedback
        lambdas = np.clip(lambdas, 0, lambda_max)

        # Ensure non-negative intensity after feedback
        lambdas = np.maximum(lambdas, 0)

        # Save lambda values
        lambdas_over_time[:, t] = lambdas

        # Generate events using Poisson distribution
        events_by_bin[:, t] = np.random.poisson(lambdas * delta_t)

    return events_by_bin, lambdas_over_time, network_graph