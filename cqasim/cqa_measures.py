#!/usr/bin/env python3
"""Measures and metrics for CQA dynamics."""
import numpy as np


def detect_nonmonotonic_jumps(values, step_threshold=5):
    """Detect non-monotonic 3-point windows with large changes.

    Detect positions in a sequence where a non-monotonic change occurs
    over 3-point windows, and the magnitude of that change exceeds the
    threshold.

    Returns:
    - List of indices `i` where non-monotonic jump occurs.
    """
    jump_indices = []
    n = len(values)

    for i in range(2, n):
        a = values[i - 2]
        b = values[i - 1]
        c = values[i]

        # Monotonic check (early continue)
        if (a <= b <= c) or (a >= b >= c):
            continue

        # Check if the change is significant
        vmin = min(a, b, c)
        vmax = max(a, b, c)
        if vmax - vmin > step_threshold:
            jump_indices.append(i)

    return jump_indices


def calc_perc_vanished_manifold(overlap_max_pos, step_threshold=5):
    """Calculate the percentage of vanished manifolds per dimension.

    Parameters:
    - overlap_max_pos: List of lists of arrays. Each outer list represents a dimension.
    - step_threshold: Threshold to detect non-monotonic jumps.

    Returns:
    - List of percentages (floats) of vanished manifolds per dimension.
    """
    perc_vanished_manifold_per_dim = []

    for dim in overlap_max_pos:
        # Filter out NaN-starting sequences and compute jumps
        non_monotonic_jumps = [
            detect_nonmonotonic_jumps(o_init, step_threshold)
            for o_init in dim
            if not np.isnan(o_init[0])
        ]

        if non_monotonic_jumps:
            non_empty_count = sum(bool(jump) for jump in non_monotonic_jumps)
            percentage = non_empty_count / len(non_monotonic_jumps)
        else:
            percentage = 0.0

        perc_vanished_manifold_per_dim.append(percentage)

    return perc_vanished_manifold_per_dim


# non-mono
# for init_pos in overlap_max_pos[0]:
#     x = detect_nonmonotonic_jumps(init_pos)
