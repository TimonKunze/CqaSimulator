#!/usr/bin/env python3
"""Generate data for CQA model."""
import numpy as np
from itertools import combinations
from typing import Optional

from cqasim.cqa_utils import pad_with_nans
from cqasim.cqa_vectutils import broken_gaussian_1d, gaussian_1d
def generate_gauss_tuning(n_cells=100, x_range=(0, 10), num_points=1000, sigma=0.1, a=1):
    """Generate 1D Gaussian tuning curves with centers evenly spaced across the environment.

    Args:
        num_cells (int): Number of tuning curves (cells).
        x_range (tuple): Environment spatial range (start, end).
        num_points (int): Number of spatial sampling points.
        sigma (float): Width (standard deviation) of Gaussian tuning curves.
        a (float): Prefactor

    Returns:
        tuning_curves (np.array): shape (num_cells, num_points), Gaussian tuning curves.
        x (np.array): spatial positions.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Evenly spaced centers across environment
    centers = np.linspace(x_range[0], x_range[1], n_cells)

    tuning_curves = []
    for c in centers:
        tuning = a * np.exp(-0.5 * ((x - c) / sigma) ** 2)
        tuning_curves.append(tuning)

    return np.array(tuning_curves), x


def gen_simplest_complex_tuning(N, L, T, scale=None):
    """Generate the simplest complex tuning curves.

    According to Malerba et al. 2024.

    Args:
        N (int): number of cells.
        L (int): length of environment.
        T (int): points in environment.
    Returns:
        tuning_curves (np.array): shape (num_cells, num_points)
        x (np.array): spatial positions.

    """
    if scale is None:
        scale = 1/L

    curves, x = generate_gauss_tuning(
        n_cells=N, x_range=(0, L), num_points=T
    )
    W = np.random.normal(0, scale=scale, size=(N, N))

    return W @ curves, x


def generate_1d_grid_tuning(
        num_cells=999, x_range=(0, 10),
        num_points=1000, period="rand", n_modules=1,
        noise_std=0.0,
    ):
    """Generate 1D grid tuning curves divided into n_modules.

    Args:
        num_cells (int): total number of cells.
        x_range (tuple): spatial range.
        num_points (int): number of spatial points.
        period (float or "rand"): base period or random period within each module.
        n_modules (int): number of grid modules.
            Stensola et al. (2012) typically find about 4 to 6 modules.
            Between 4 and 10 is a good number.
        noise_std (float): Additive noise for the tuning curves.
            If tuning curves have a max amplitude of 1 (like cosine output),
            a noise_std of 0.1 means ~10% fluctuations.
    Returns:
        tuning_curves (np.array): shape (num_cells, num_points), rectified cosine tuning curves.
        x (np.array): spatial positions.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    tuning_curves = []

    # Divide cells roughly equally into modules
    cells_per_module = num_cells // n_modules
    remainder = num_cells % n_modules

    # Define range for random periods if period=="rand"
    min_period, max_period = 0.5, (x_range[1] - x_range[0]) / 2

    for mod_idx in range(n_modules):
        # Generate a random period per module if period=="rand", else use fixed period scaled by module idx
        if period == "rand":
            module_period = np.random.uniform(min_period, max_period)
        else:
            # e.g. scale period geometrically across modules (like grid cell modules)
            module_period = period * (1.4 ** mod_idx)  # 1.4 is a common scale ratio

        # Determine how many cells in this module
        n_cells_module = cells_per_module + (1 if mod_idx < remainder else 0)

        for _ in range(n_cells_module):
            phase = np.random.uniform(0, 2 * np.pi)
            tuning = np.cos(2 * np.pi * x / module_period + phase)

            if noise_std != 0.0:
                noise = np.random.normal(0, noise_std, size=tuning.shape)
                tuning += noise

            tuning_curves.append(tuning)

    tuning_curves = np.maximum(np.array(tuning_curves), 0)  # rectify

    return tuning_curves, x


# def generate_1d_grid_tuning(
#         num_cells=999, x_range=(0, 10), num_points=1000, period=2, n_modules):
#     """Generate 1D grid tuning ...
#     """
#     x = np.linspace(x_range[0], x_range[1], num_points)
#     tuning_curves = []
#     # maybe 100-300 grid cells per module?

#     for i in range(num_cells):
#         # phase = 2 * np.pi * i / num_cells  # phase offset evenly spaced in [0, 2pi)
#         phase = np.random.uniform(0, 2 * np.pi)  # random phase offsets
#         if period == "rand":
#             period = np.random.uniform(0, num_points)
#         tuning = np.cos(2 * np.pi * x / period + phase)
#         tuning_curves.append(tuning)

#     return np.maximum(np.array(tuning_curves), 0), x


def gen_p_data(P, N, T, L, Zeta,
               diametro_m, diametro_delta,
               height_m, height_delta,
               correlated_peaks,
               gamma,
               correlated_dims,
               M_fixed,
               simplified=True,
               verbose=0, seed=None):
    """Gen. fully random tuning curves."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize data
    dt = np.zeros((P, N, T))
    diams_per_nrn = []
    heights_per_nrn = []
    fields_per_nrn = []

    for p in range(P):
        # Fully random tuning curves
        # dt[p] = np.random.uniform(0.0, 1.0, (N, T))

        # Simple possible model that generates complex tuning curves (Malerba, et al. 2025)
        # TODO: TBD
        dt[p], _ = gen_simplest_complex_tuning(N, L, T)

        # Grid cell tuning curves ...
        # ... with fixed period for the full population
        # dt[p], _ = generate_1d_grid_tuning(
        #     num_cells=999, x_range=(0, 1000), period=100)
        # ... with random period for the full population
        # dt[p], _ = generate_1d_grid_tuning(
        #     num_cells=999, x_range=(0, 1000), period="rand", n_modules=1,
        #     noise_std=0.1)
        # ... with random period for the N modules
        # dt[p], _ = generate_1d_grid_tuning(
        #     num_cells=999, x_range=(0, 1000), period="rand", n_modules=6,
        #     noise_std=0.1)

    return dt, diams_per_nrn, heights_per_nrn, fields_per_nrn



# def gen_p_data(P, N, T, L, Zeta,
#                diametro_m, diametro_delta,
#                height_m, height_delta,
#                correlated_peaks,
#                gamma,
#                correlated_dims,
#                M_fixed,
#                simplified=True,
#                verbose=0, seed=None):
#     """Gen. simplif. multidim. data with only variability in peak widths."""
#     if seed is not None:
#         np.random.seed(seed)

#     # Initialize data
#     dt = np.zeros((P, N, T))
#     diams_per_nrn = []
#     heights_per_nrn = []
#     fields_per_nrn = []

#     for p in range(P):
#         # Generate simplified data (i.e. only variation in peak width)
#         # ================================================================
#         if simplified:
#             # Peaks are non-overlapping in all dimensions, i.e. shifted peak centers
#             if correlated_dims == "min":
#                 shift = p*50
#                 if verbose:
#                     print(f"NOTE: Using shift {shift} for 'min' correlation.")
#                 # TODO: shift should be dependent on dimetro_delta, L, T and N (>9)
#                 # must be manually adapted here
#             # Peaks are maximally overlapping in all dimensions, i.e. same peak center
#             elif correlated_dims == "max":
#                 if verbose:
#                     print("NOTE: Using 'max' correlation with overlapping peaks.")
#                 shift = 0
#             else:
#                 raise ValueError("As of now, correlations can only be 'max' or 'min'.")

#             dt[p], diams, heights = gen_simplified_data(
#                     N, T, L, diametro_m, diametro_delta, shift=p*50,
#                     seed=seed + p)
#             fields = np.ones_like(diams)  # all 1 for simpl. dt

#         # Realistic data (i.e. variation in peak width, height, and number)
#         # ================================================================
#         else:
#             # Generate realistic data
#             if seed is None:
#                 spec_seed = seed
#             else:
#                 spec_seed = seed + p
#             dt[p], diams, heights, fields = gen_realistic_data2(
#                 N, Zeta, T, L,
#                 diametro_m, diametro_delta,
#                 height_m,
#                 height_delta,
#                 correlated_peaks,
#                 gamma,
#                 M_fixed=M_fixed,
#                 verbose=verbose,
#                 seed=spec_seed,
#             )

#         diams_per_nrn.append(diams)
#         heights_per_nrn.append(heights)
#         fields_per_nrn.append(fields)

#     if verbose > 0:
#         print(f"Avg. corr. betw. dimensions = {avg_dim_correlations(dt):.3f}.")

#     return dt, diams_per_nrn, heights_per_nrn, fields_per_nrn


def gen_realistic_data2(  # TODO: write unit tests!
    N: int, Zeta: float, T: int, L: float,
    diametro_m: float, diametro_delta: float,
    altezza_m: float, altezza_delta: float,
    correlazione: bool,
    gamma: float,
    M_fixed: int = None,
    verbose: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic neural activity data based on place cell statistics.

    Parameters:
    - N: Number of neurons
    - Zeta: Exponential distribution parameter for number of fields
    - T: Number of time steps
    - L: Length of the environment
    - diametro_m, diametro_delta: Mean and standard deviation for field diameters
    - altezza_m, altezza_delta: Mean and standard deviation for peak heights
    - correlazione: Whether there’s a correlation in peak heights ('no' or other)
    - gamma: Correlation strength for peak heights
    - M_fixed: Fixed number of fields per neuron (optional)
    - verbose: Verbosity level
    - seed: Random seed for reproducibility

    Returns:
    - dati: Neural activity data
    - diams_per_nrn, heights_per_nrn, fields_per_nrn: Properties of place fields for each neuron
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize general grid and data
    x_general = np.linspace(0, L, T, endpoint=False)
    dati = np.zeros((N, T))

    diams_per_nrn = []  # diameters of place fields for each neuron
    heights_per_nrn = []  # heights of place fields
    fields_per_nrn = []  # position of place fields
    # nb_fields_per_nrn = []

    # Loop over each neuron
    for i in range(N):
        # Determine the number of fields per neuron
        if M_fixed is not None:
            nb_fields = M_fixed
        else:
            nb_fields = int(np.random.exponential(Zeta))

            # Ensure valid number of fields
            while nb_fields == 0 or nb_fields > 21:
                nb_fields = int(np.random.exponential(Zeta))

        # Draw field diameters
        diametri = draw_diameters(diametro_m, diametro_delta, nb_fields, L)

        if verbose > 0:
            print(f"n={i}, Total Diameters: {np.sum(diametri)}, "
                  f" nb_fields={nb_fields}")

        # Draw peak heights based on correlation
        if not correlazione or correlazione == "no":
            picchi = np.random.lognormal(altezza_m, altezza_delta, nb_fields)
        else:
            picchi = []
            for d in diametri:
                # NOTE: FS uses unnecessary while loop here qith q==0
                # NOTE: FS has an if condition here testing whether 0<2*expis which
                # is trivially true, therefore removed
                specific_mean = altezza_m + gamma * \
                    np.log(d / np.exp(diametro_m + diametro_delta**2 / 2.0))
                # Draw peak firing rate from log normal
                q = np.random.lognormal(specific_mean, altezza_delta,)
                # Modify peak firing rate to prevent peak firing rate higher
                # than 40Hz
                p = 30 * np.arctan(q / 30.0) + 6 * np.arctan((q / 120.0) ** 2)
                picchi.append(p)

        # NOTE: here FS sampled dovesono_interno from a uniform distribution
        # but this is unnecessary, given that she defines it as an empty list
        # one step later.
        # But this influences the random number generator so I leave it in
        # as dummy.
        np.random.uniform(0, L, nb_fields)

        # Sample peak locations ensuring fields don't overlap
        dovesono_interno = [None] * nb_fields
        ordered_diametri_inds = np.argsort(diametri)[::-1]  # Sort diameters in descending order

        # Try to place fields randomly
        max_iterations = 10**5
        for j in ordered_diametri_inds:
            diametro = diametri[j]
            posizione = np.random.uniform(0, L)
            valid_position = False
            iterations = 0
            while not valid_position:
                valid_position = True
                for ij in range(nb_fields):
                    if ij != j and dovesono_interno[ij] is not None:
                        d = np.abs(dovesono_interno[ij] - posizione)
                        d = min(d, L - d)  # Adjust for circular boundary
                        if d < diametro / 2.0 + diametri[ij] / 2.0:
                            valid_position = False
                            posizione = np.random.uniform(0, L)
                iterations += 1
                if iterations > max_iterations:
                    raise RuntimeError("Max iterations to place non-overlapping fields reached.")
            dovesono_interno[j] = posizione

        # Store results for this neuron
        diams_per_nrn.append(diametri)
        heights_per_nrn.append(picchi)
        fields_per_nrn.append(dovesono_interno)
        # nb_fields_per_nrn.append(nb_fields)

        # Generate neural activity for this neuron
        for j in range(nb_fields):
            centro = dovesono_interno[j]
            altezza = picchi[j]
            radius = diametri[j] / 2.0
            dati[i, :] += broken_gaussian_1d(x_general, centro, altezza, radius, L)

    # print(np.shape(diams_per_nrn))
    # print(np.shape(heights_per_nrn))
    # print(np.shape(fields_per_nrn))
    # print(len(diams_per_nrn), "items")
    # print("Example entry shapes:", [len(d) if hasattr(d, '__len__') else type(d) for d in diams_per_nrn[:5]])
    # print(len(fields_per_nrn), "items")
    # print("Example entry shapes:", [len(d) if hasattr(d, '__len__') else type(d) for d in fields_per_nrn[:5]])

    diams_per_nrn = pad_with_nans(diams_per_nrn)
    heights_per_nrn = pad_with_nans(heights_per_nrn)
    fields_per_nrn = pad_with_nans(fields_per_nrn)

    return dati, diams_per_nrn, heights_per_nrn, fields_per_nrn


# Tested
def gen_simplified_data(
        N, T, L, diametro_m, diametro_delta, shift=0, seed=None):
    """Create a dataset with Gaussian-like profiles.

    Old name: crea_dati_regolare(N, T, L, diametri)

    Gaussians are centered at evenly spaced locations.
    "Simplified dataset", i.e. with lognormal variations only in
    peak widths. Heights vary because of normalization.

    Parameters:
    - N (int): Number of center profiles.
    - T (int): Number of sampled places (points).
    - L (float): Total length of the profile.
    - diameters (array-like): Diameters of the Gaussians.

    Returns:
    - data (ndarray): A (N, T) array containing the generated values.
    """
    if seed is not None:
        np.random.seed(seed)

    # Draw diameters from log normal
    diameter_per_nrn = []
    for _ in range(N):
        diameter = draw_diameters(diametro_m, diametro_delta, 1, L)
        diameter_per_nrn.append(diameter)
    diameter_per_nrn = np.array(diameter_per_nrn).flatten()
    # (flatten because of broadcasting issue)

    # Create evenly spaced center points
    x_centri = np.linspace(0, L, N, endpoint=False)  # N centers, shape=(N,)
    x_centri += shift
    # Create general x values
    x_general = np.linspace(0, L, T, endpoint=False)  # T points, shape=(T,)

    # Compute all distances efficiently using broadcasting
    dist = np.abs(x_general[:, None] - x_centri)  # Shape: (T, N)
    dist = np.minimum(dist, L - dist)  # Apply periodic boundary conditions

    # Compute (non-normalized) Gaussian values
    radius = diameter_per_nrn / 2.0  # Convert diameters to radii
    # data = np.exp(-dist**2 / (2 * radius**2)) / np.sqrt(2 * np.pi)
    data = gaussian_1d(dist, radius)
    heights_per_nrn = 1 / (radius * np.sqrt(2 * np.pi))

    data = data.T  # Transpose to match original (N, T) shape

    return data, diameter_per_nrn, heights_per_nrn


# Tested
def draw_diameters(mean_diameter, var_diameter, num_fields, L, seed=None):
    """Draw diameters from a log-normal distribution.

    Parameters:
        mean_diameter (float): The mean diameter of the objects.
        var_diameter (float): The standard deviation (sigma) for the
            log-normal distribution.
        num_fields (int): The number of diameters to generate.
        L (float): The total available length within which diameters must fit.
        max_attempts (int): The max nb of attempts to generate valid diameters.

    Returns:
        np.ndarray or None: Array of diameters if valid, otherwise None after max attempts.
    """
    max_attempts = 100

    if seed is not None:
        np.random.seed(seed)

    # Try up to `max_attempts` using a for loop
    for _ in range(max_attempts):
        # Generate diameters from the log-normal distribution
        diameters = np.random.lognormal(mean_diameter, var_diameter, num_fields)
        # Check if the total sum of diameters fits within the available space,
        # considering some gap between them
        if np.sum(diameters) <= (L - 2 * num_fields):  # Allow small gaps per field
            return diameters.tolist()

    print(f"Warning: Unable to generate valid diameters after {max_attempts} attempts.")
    return None


def avg_dim_correlations(tensor):
    """Compute mean Pearson corr. betw. all pairs of dimension slices in a tensor of shape (D, N, T).

    Returns:
        float: average correlation coefficient across all dimension pairs.
    """
    D = tensor.shape[0]
    flattened = tensor.reshape(D, -1)  # Shape: (D, N*T)

    corrs = []
    for i, j in combinations(range(D), 2):
        a = flattened[i]
        b = flattened[j]
        if np.std(a) > 0 and np.std(b) > 0:
            corr = np.corrcoef(a, b)[0, 1]
            corrs.append(corr)

    return np.mean(corrs) if corrs else 0.0
