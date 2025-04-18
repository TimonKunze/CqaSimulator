#!/usr/bin/env python3
import numpy as np


def create_hebb_p_weights(tensor_dt):
    """Create hebbian weights for different charts (P).

    See Battaglia & Treves, 1998, Physical Review E, Equ. 24.
    """
    if len(tensor_dt.shape) == 2:
        tensor_dt = tensor_dt[np.newaxis, :, :]

    assert len(tensor_dt.shape) == 3, "Data is not tensor."

    P, N, _ = tensor_dt.shape
    summed_W = np.zeros((N, N))
    for p in range(P):
        summed_W += create_hebb_weights(tensor_dt[p])

    # Maybe normalize the sum by P?

    return summed_W


# Tested
# @sut.auto_numba
def create_hebb_weights(dati, exp=0):
    """Create a Hebbian weight matrix based on input data.

    Parameters:
    - dati (ndarray): A (N, T) array where N is the number of neurons and T
      is the number of samples.

    Returns:
    - W (ndarray): A (N, N) symmetric Hebbian weight matrix.
    """
    N, T = dati.shape  # Extract dimensions
    mean = np.mean(dati)  # Global mean
    #
    # Mean of each row (column-wise)
    mean_rows = np.mean(dati, axis=1, keepdims=True)
    # Get zero mean da  # NOTE: problem in FS's code?  # TODO:
    dat = dati / ((mean_rows ** exp) * (mean ** (1 - exp)))  # NOTE: What written in the paper MISTAKE
    # dat = dati / ((mean_rows ** (1 - exp)) * (mean ** exp)) # NOTE: What FS has written in the code
    # but she didn't make any mistake for exp=0 because it is caught with an if conditions.
    # Simplifies to for exp=0:
    # dat = dati / mean  # Get zero mean data
    dat -= 1

    # Compute Hebbian weights using matrix operations
    W = dat @ dat.T  # Compute full matrix
    np.fill_diagonal(W, 0)  # Set diagonal to zero

    # Normalize
    W /= N * T

    return W
