#!/usr/bin/env python3
"""Utility functions for the CqaModel based on vectorized computations.

Designed for speed; compatible with Numba acceleration.
"""
import numpy as np
import numba

from cqasim.cqa_utils import get_loop_indices


# numba.set_num_threads(40)  # Set to desired number of threads
current_threads = numba.get_num_threads()
print(f"Numba is using {current_threads} threads.")


def auto_numba(f):
    """Create numba decorator.

    Note: it is easy to turn off by uncommenting
    the second return statement, commenting the
    former.
    """
    return numba.njit(f)
    # return f


@auto_numba
def one_step_dynamics(general_mean, Vc, J, g, kb):
    """Update the state V of a system based on input currents and a transfer function.

    Called: one_step_dynamics_B in FS's code.

    Vc: activity at time step
    general_mean is data_mean, is mean_v, si desired_mean
    J: Connectivity matrix
    g: fixed gain parameter
    kb: omega / sigma / 300 / strength of inhibitory feedback
    """
    # Compute the input currents h
    Vc = np.ascontiguousarray(Vc)
    h = np.dot(J, Vc)
    # Apply cubic transformation using desired general mean and scale by kb
    Th = b_function(np.mean(Vc), general_mean, kb)
    # Shift the input currents by Th
    h -= Th
    # Apply transfer function
    V = relu(h)
    # Scale output by g
    V = g * V

    return V


@auto_numba
def relu(h):
    """RELU or threshold-linear transfer function."""
    return np.maximum(0, h)


@auto_numba
def b_function(v, desired_mean, omega):
    """Cubic transformation function applied to input v.

    Note:
    - called b_function in FS's code
    - omega == sigma in the paper
    """
    # NOTE: how did factor 4 and exp 3 came about?
    # TODO: understand why
    return 4 * omega * (v - desired_mean) ** 3


@auto_numba
def cosine_similarity_vec(d, V):
    """Calculate the vectorized cosine similarity.

    Between a vector `V` and the columns of matrix `d`.

    Parameters:
    - d (ndarray): A 2D array (matrix) with shape (n, m), where `n` is
        the dimension of each vector, and `m` is the number of vectors
        to compare.
    - V (ndarray): A 1D array (vector) with the same number of elements
        as the rows of `d`.
    Returns:
    - ndarray: A 1D array with the cosine similarity between `V` and each column of `d`.
    """
    # Compute the dot product of `V` with each column of `d`
    dot_product = np.dot(d.T, V)
    # Compute the magnitudes (Euclidean norms) of `V` and the columns of `d`
    V_magnitude = np.sqrt(np.sum(V**2))  # Magnitude of vector V
    d_magnitude = np.sqrt(np.sum(d**2, axis=0))  # Magnitudes of columns in d

    # Return the cosine similarity between `V` and each column in `d`
    return dot_product / (V_magnitude * d_magnitude)


@auto_numba
def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Avoid division by zero

    return np.dot(v1, v2) / (norm_v1 * norm_v2)


@auto_numba
def calc_hessian(v, general_mean, kb, N):
    """Compute scaled squared deviation of `v` from `general_mean`, normalized by `N`.

    Parameters:
    v : float or array-like
        Input values for which the deviation is computed.
    general_mean : float
        Reference mean value to compute deviation from.
    kb : float
        Scaling factor (could be a parameter related to curvature or weighting).
    N : int
        Normalization factor (possibly the number of samples or degrees of freedom).

    Returns:
    float or array-like
        The scaled squared deviation.
    """
    # Compute squared deviation from the mean
    squared_deviation = (v - general_mean) ** 2
    # Scale by kb and normalize by N with a factor of 12.0
    a = (12.0 / N) * kb * squared_deviation
    return a


@auto_numba
def calc_sparsity(V):
    # TODO: Function unused
    """Calculate the sparcity or flatness in neural activity.

    - Is between 0 and 1 if all elements in V are non-negative.
    - Is close to 1 if all elements in V are equal.
    - Drops as V becomes more "spiky" — i.e., a few large values
        and many zeros or small values.
    """
    return np.mean(V) ** 2 / np.mean(V**2)


@auto_numba
def calc_support(V, thresh=0.02):
    """Calculate the support or active fraction of neural activiy."""
    return np.sum(V > thresh) / len(V)


@auto_numba
def std_cdm(o1):
    """Compute circular dispersion measure (CDM) for weighted data.

    On periodic scale.

    Parameters:
    o1  : numpy array of similarity values (should be non-negative)

    Returns:
    delq : Circular dispersion measure (standard deviation in a circular space)
            Low delq: points tightly clustered around some circular mean.
            High delq: points spread out evenly around the circle.
    """
    T = len(o1)
    Len = np.arange(T)  # array of positions (indices) from 0 to T

    # Ensure non-negative weights
    o1 = np.maximum(o1, 0)
    # Convert positions to circular coordinates
    theta = 2 * np.pi * Len / (T - 1)
    # Compute weighted mean cosine and sine components
    total_weight = np.sum(o1)
    co = np.sum(o1 * np.cos(theta)) / total_weight
    si = np.sum(o1 * np.sin(theta)) / total_weight
    # Compute the circular mean angle in radians
    qui = np.arctan2(-si, -co) + np.pi  # Ensures output in [0, 2π]
    # Convert circular mean back to original scale
    cm = (T - 1) * qui / (2 * np.pi)
    # Compute absolute circular differences
    dif = np.abs(Len - cm)
    # Adjust distances for circularity in a vectorized way
    dif = np.where(dif >= (T - 1) / 2.0, dif - (T - 1), dif)
    # Compute weighted circular dispersion (similar to standard deviation)
    delq = np.sqrt(np.sum(o1 * dif**2) / (total_weight * (1 / 12.0) * (T - 1) ** 2))

    return delq  # normalized circular standard deviation.


@auto_numba
def gaussian_1d(dist, radius):
    """Compute 1D Gaussian for given distance and radius (stand. deviation).

    Returns:
        float or np.ndarray: Gaussian value(s) at the given distance(s).
    """
    return np.exp(-dist**2 / (2 * radius**2)) / np.sqrt(2 * np.pi)


@auto_numba
def broken_gaussian_1d(x, x_center, C, radius, L):
    """Compute clipped Gaussian with periodic boundary conditions in 1D.

    Parameters:
    - x: np.array, positions where the function is evaluated.
    - x_center: float, center of the Gaussian.
    - C: float, scaling factor.
    - radius: float, standard deviation-like parameter.
    - L: float, domain length for periodic boundary conditions.

    Returns:
    - V: np.array, values of the "broken" Gaussian function.
    """
    # Compute absolute distance
    dist = np.abs(x - x_center)
    # Apply periodic boundary correction for distances > L/2
    dist = np.minimum(dist, L - dist)  # dist[dist >= L / 2] -= L
    # Precompute constants
    exp_neg_half = np.exp(-0.5)
    norm_factor = 1 / (1 - exp_neg_half)
    # Compute Gaussian function with scaling
    V = C * (np.exp(-dist**2 / (2 * radius**2)) - exp_neg_half) * norm_factor

    # Ensure non-negative values (clipping)
    return np.maximum(V, 0)


### TODO: UNFINISHED ANALYSES!

def count_decreasing_steps(values: np.ndarray) -> int:  # TODO: necessary?
    """Count decreasing transitions in an array.

    From left to right.
    Ex: values = [5,4,3,4,1], Returns: 3.
    """
    return np.sum(values[1:] < values[:-1])


# def track_tangent_overlap(self, thresh=1e-3):  # TODO: finish writing
#     """Analyze most unstable direction after removing near-zero activations.

#     Note: This analysis is done on the last activation, taking in account
#     the activation of the 2nd-last and of the current activation.

#     Parameters:
#     - V: Activation vector
#     - W: Weight matrix
#     - dx: Direction vector to compare against
#     - g: Gain term (used to shift eigenvalues)
#     - general_mean: Background mean input
#     - kb: Some parameter for curvature (passed to B_hessian)
#     - N: Dimensionality
#     - threshold: Cutoff to determine weak activations (default=1e-3)

#     Returns:
#     - min_cos_sim: Cosine similarity between most unstable eigenvector and dx
#     """
#     # Indices of near-zero activations
#     inactive_indices = np.where(self.par["V_prec"] < thresh)[0]

#     # Remove corresponding rows/cols from weight matrix
#     Wq = self.par["W"].copy()
#     Wq = np.delete(Wq, inactive_indices, axis=0)
#     Wq = np.delete(Wq, inactive_indices, axis=1)

#     # Add Hessian-like correction to capture local curvature
#     hessian_correction = calc_hessian(
#         np.mean(self.par["V_prec"]), self.ctx["data_mean"],
#         self.par["kb"], self.par["N"],
#     )
#     Wq = -1 * Wq + hessian_correction

#     # Compute eigen decomposition
#     eigenvalues, eigenvectors = np.linalg.eig(Wq)
#     eigenvalues = np.real(eigenvalues) + 1 / self.par["g"]

#     # Find the direction of smallest (most unstable) eigenvalue
#     argmin = np.argmin(eigenvalues)
#     eigen_min = eigenvalues[argmin]
#     unstable_direction = eigenvectors[:, argmin]

#     # TODO
#     # Attempt to compute centered or one-sided derivative
#     forward = self.par["time_step"] # TODO: Check
#     backward = self.par["time_step"] - 2 > 0

#     if forward and backward:
#         # Centered difference
#         dx = forward[0] - backward[0]
#         V = ((V1 + V2) / 2.0).copy()
#         o = ((o1 + o2) / 2.0).copy()
#     elif forward:
#         # One-sided forward difference
#         V = np.copy(V2)
#         o = np.copy(o2)
#         dx = forward[0] - V
#     elif backward:
#         # One-sided backward difference
#         V = np.copy(V1)
#         o = np.copy(o1)
#         dx = backward[0] - V
#     else:
#         raise ValueError("No valid forward or backward passaggi to compute derivative.")

#     # Remove same indices from dx to match reduced Wq shape
#     dx = np.delete(dx, inactive_indices)
#     dx_reduced = np.delete(dx, inactive_indices)

#     # Compute cosine similarity with dx (reshape if needed)
#     cos_sim = cosine_similarity(
#         unstable_direction.reshape(1, -1),
#         dx.reshape(1, -1)
#     )[0, 0]

#     # Compute cosine similarity
#     cos_sim = np.abs(cosine_similarity(unstable_direction, dx_reduced))

#     return eigen_min, cos_sim


def detect_significant_jump(curr_idx: int, max_idx: int,  # TODO: necessary?
                            similarity_values: np.ndarray,
                            threshold=5) -> bool:
    """Detect meaningful jump and return jump info."""
    T = len(similarity_values)
    # Compute distance between current index (prima) and max one (poi)
    dist_idx = np.abs(curr_idx - max_idx)
    if dist_idx <= threshold:
        return False  # No significant movement
    # Count how many decreasing similarity_values values in movement
    # (handling wraparound)
    loop_indices = get_loop_indices(curr_idx, max_idx, T)
    similarity_values_values = np.asarray(similarity_values)[loop_indices]
    nb_decreasing_steps = count_decreasing_steps(similarity_values_values)

    if nb_decreasing_steps > threshold:
        return True
    return False
