#!/usr/bin/env python3
import numpy as np
from itertools import combinations

from cqasim.cqa_utils import pad_with_nans
from cqasim.cqa_vectutils import broken_gaussian_1d, gaussian_1d


def gen_p_data(P, N, T, L, M,
               diametro_m, diametro_delta,
               height_m, height_delta,
               correlated_peaks,
               gamma,
               correlated_dims,
               M_fixed,
               simplified=True,
               verbose=0, seed=None):
    """Gen. simplif. multidim. data with only variability in peak widths."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize data
    dt = np.zeros((P, N, T))
    diams_per_nrn = [] #np.zeros((P, N))
    heights_per_nrn = [] # np.zeros((P, N))
    fields_per_nrn = [] # np.zeros((P, N))
    # print("fielskdf", fields_per_nrn.shape)

    for p in range(P):
        # Generate simplified data (i.e. only variation in peak width)
        # ================================================================
        if simplified:
            # Peaks are non-overlapping in all dimensions, i.e. shifted peak centers
            if correlated_dims == "min":
                shift = p*50
                if verbose:
                    print(f"NOTE: Using shift {shift} for 'min' correlation.")
                # TODO: shift should be dependent on dimetro_delta, L, T and N (>9)
                # must be manually adapted here
            # Peaks are maximally overlapping in all dimensions, i.e. same peak center
            elif correlated_dims == "max":
                if verbose:
                    print("NOTE: Using 'max' correlation with overlapping peaks.")
                shift = 0
            else:
                raise ValueError("As of now, correlations can only be 'max' or 'min'.")

            dt[p], diams, heights = gen_simplified_data(
                    N, T, L, diametro_m, diametro_delta, shift=p*50,
                    seed=seed)
            fields = np.ones_like(diams)  # all 1 for simpl. dt

        # Realistic data (i.e. variation in peak width, height, and number)
        # ================================================================
        else:
            if correlated_dims == "max":
                if seed is None:
                    seed = 1
                print("NOTE: Data is necessarily seeded for correlated_dims=='max'.")
            elif correlated_dims == "random":
                print("NOTE: Data is not seeded because correlated_dims=='random'.")
                seed = None
            else:
                # Raise error for unsupported correlation mode
                raise ValueError("As of now, correlations can only be 'max' or 'random' for realistic data.")

            # Generate realistic data
            dt[p], diams, heights, fields = gen_realistic_data2(
                N, M, T, L,
                diametro_m, diametro_delta,
                height_m, height_delta,
                correlated_peaks, gamma,
                M_fissato=M_fixed,
                verbose=verbose,
                seed=seed
            )

        diams_per_nrn.append(diams)
        heights_per_nrn.append(heights)
        fields_per_nrn.append(fields)

    if verbose > 0:
        print(f"Avg. corr. betw. dimensions = {avg_dim_correlations(dt):.3f}.")

    print("fields", np.shape(fields_per_nrn))

    return dt, diams_per_nrn, heights_per_nrn, fields_per_nrn


def gen_realistic_data2(  # TODO: write unit tests!
    N, M, T, L,
    diametro_m, diametro_delta,
    altezza_m, altezza_delta,
    correlazione,
    gamma,
    M_fissato=True,  # Default value: True (fixed number of fields)
    verbose=1,
    seed=4,
):
    """Generate synthetic neural activity data based on place cell statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize arrays and lists
    x_general = np.linspace(0, L, T, endpoint=False)
    dati = np.zeros((N, T))

    diams_per_nrn = []  # diameters of place fields for each neuron
    heights_per_nrn = []  # heights of place fields
    fields_per_nrn = []  # position of place fields
    # nb_fields_per_nrn = []

    if M_fissato or M_fissato == "yes":
        M = int(M)

    # Loop over number of neurons N
    for i in range(N):
        # Determine the number of fields per neuron
        if M_fissato or M_fissato == "yes":
            nb_fields = M
        else:
            nb_fields = int(np.random.exponential(M))

        # Ensure valid number of fields
        while nb_fields == 0 or nb_fields > 21:
            nb_fields = int(np.random.exponential(M))

        # Draw field diameters
        diametri = draw_diameters(diametro_m, diametro_delta, nb_fields, L)

        if verbose > 0:
            print(f"n={i}, Total Diameters: {np.sum(diametri)}, nb_fieldss={nb_fields}")

        # Draw peak heights based on correlation
        if not correlazione or correlazione == "no":
            picchi = np.random.lognormal(altezza_m, altezza_delta, nb_fields)
        else:
            picchi = []
            for d in diametri:
                # NOTE: FS uses unnecessary while loop here qith q==0
                # NOTE: FS has an if condition here testing whether 0<2*expis which
                # is trivially true, therefore removed
                specific_mean = altezza_m + gamma * np.log(d / np.exp(diametro_m + diametro_delta**2 / 2.0))
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
    print("height", heights_per_nrn)
    print("data shape", data.shape)

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
