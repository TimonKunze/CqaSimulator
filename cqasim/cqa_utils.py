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


