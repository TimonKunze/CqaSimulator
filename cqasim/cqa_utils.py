#!/usr/bin/env python3
"""Utility functions for the CQA Simulator."""
import numpy as np


def get_loop_indices(curr_idx, max_idx, T):
    """Return sequence of indices on a circular domain of length T.

    Supporting both forward and backward trajectories with wraparound.
    Includes both curr_idx and max_idx as the start and end of the sequence.
    """
    if curr_idx == max_idx:
        return np.array([])

    forward_dist = (max_idx - curr_idx) % T
    backward_dist = (curr_idx - max_idx) % T

    if forward_dist <= backward_dist:
        return (np.arange(curr_idx, curr_idx + forward_dist + 1) % T)
    else:
        return (np.arange(curr_idx, curr_idx - backward_dist - 1, -1) % T)


def pad_with_nans(nested_list):
    """Pad nested list with nans for equal length and convert to array.

    Pad that all sublists are of same length, then
    convert to a numpy array.
    """
    # Determine maximum length of inner lists
    max_length = max(len(sub) for sub in nested_list)
    padded_list = [list(sub) + [np.nan]
                   * (max_length - len(sub)) for sub in nested_list]

    return np.array(padded_list)


def pad_with_nans_3d(nested_list):
    """
    Pads a 3D list of shape [P][N][T] so that all [N][T] entries have the same shape.
    """
    if not nested_list:
        return np.array([])

    max_n = max(len(outer) for outer in nested_list)
    max_t = max(
        len(inner) for outer in nested_list for inner in outer if inner
    )

    padded = []
    for outer in nested_list:
        padded_outer = []
        for inner in outer:
            # Pad each inner list to max_t
            padded_inner = list(inner) + [np.nan] * (max_t - len(inner))
            padded_outer.append(padded_inner)
        # Pad outer list to max_n
        while len(padded_outer) < max_n:
            padded_outer.append([np.nan] * max_t)
        padded.append(padded_outer)

    return np.array(padded)

def pad_with_nans_3d(nested_list):
    """Pad a twice-nested list with NaNs to create a rectangular NumPy array.

    The input should be a list of lists of lists. The function:
    - Pads inner lists (e.g., per object) to equal length (e.g., same vector size)
    - Pads middle lists (e.g., per frame) to equal length
    - Returns a 3D NumPy array: (outer, middle, inner)
    """
    # Step 1: Pad innermost lists (e.g., vectors) with NaNs
    max_inner_len = max(
        len(inner) for outer in nested_list for inner in outer
    )
    padded_inner = [
        [list(inner) + [np.nan] * (max_inner_len - len(inner)) for inner in middle]
        for middle in nested_list
    ]

    # Step 2: Pad middle lists to equal length (e.g., same number of objects per frame)
    max_middle_len = max(len(middle) for middle in padded_inner)
    padded_middle = [
        middle + [[np.nan] * max_inner_len] * (max_middle_len - len(middle))
        for middle in padded_inner
    ]

    return np.array(padded_middle)
