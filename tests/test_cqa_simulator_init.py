#!/usr/bin/env python3
import pytest
import numpy as np
import tempfile
# from unittest.mock import patch
from pathlib import Path

from cqasim.cqa_simulator import CqaSimulator


@pytest.fixture
def simulator():
    return CqaSimulator()


def test_with_init_full_set(simulator):
    """Test Initialization with __init__ and non-default parameters."""
    dpar = {
            # Model parameters
            "P": 11,
            "N": 5,  # Neurons
            "T": 999,  # nb places
            "L": 200,  # sampling of places
            # Data parameters
            "mean_diameter": 1.57,  # field mean
            "var_diameter": 2.0,  # field width diameter
            "correlated_peaks": False,  # Corrs. between field height and diameter
            "simplified_data": True,  # TODO: mistake
            "correlated_dimensions": "max",  # only generating data for p > 1
            "mean_height": 1.549,
            "var_height": 0.0,
            "exponent": 0,
            "gamma": 0.5,
            "Zeta": 4.3,  # average nb of fields (4.4 in real dt)
            "M_fixed": 4,
            # Simulation parameters
            "seed": 1,
            "track_dynamics_flag": True,
            "record_final_flag": True,  # TODO: test
            "g": 18.0,
            "kb": 300,  # omega / sigma / strenght of inh. feedback
            "initial_step_size": 0.08,
            "converge_eps": 1e-5,
            "max_iterations": 6e4,
        }
    simulator.par = dpar.copy()
    simulator.__init__(simulator.par)

    assert (simulator.ctx["data"].shape == (dpar["P"], dpar["N"], dpar["T"])), \
        "Incorrect data shape."


def test_with_init_subset_modification(simulator):
    """Test Initialization with __init__ and non-default parameters.

    With a modification of only a subset of paramters.
    """
    P = 11
    N = 3
    T = 890
    simulator.par["P"] = P
    simulator.par["N"] = N
    simulator.par["T"] = T
    simulator.__init__(simulator.par)

    assert (simulator.ctx["data"].shape == (P, N, T)), \
        "Incorrect data shape."

def test_without_init_full_set(simulator):
    """Test Initialization with __init__ and non-default parameters.

    With full set of citionary paramters.
    """
    dpar = {
            # Model parameters
            "P": 11,
            "N": 5,  # Neurons
            "T": 999,  # nb places
            "L": 200,  # sampling of places
            # Data parameters
            "mean_diameter": 1.57,  # field mean
            "var_diameter": 2.0,  # field width diameter
            "correlated_peaks": False,  # Corrs. between field height and diameter
            "simplified_data": True,
            "correlated_dimensions": "max",  # only generating data for p > 1
            "mean_height": 1.549,
            "var_height": 0.0,
            "exponent": 0,
            "gamma": 0.5,
            "Zeta": 4.3,  # average nb of fields (4.4 in real dt)
            "M_fixed": 4,
            # Simulation parameters
            "seed": 1,
            "track_dynamics_flag": True,
            "record_final_flag": True,
            "g": 18.0,
            "kb": 300,  # omega / sigma / strenght of inh. feedback
            "initial_step_size": 0.08,
            "converge_eps": 1e-5,
            "max_iterations": 6e4,
            "verbose": False,
            "track_quantified_dynamics": False,
            "track_full_dynamics": False,
        }
    simulator.par = dpar.copy()
    simulator.update_ctx_state()

    assert (simulator.ctx["data"].shape == (dpar["P"], dpar["N"], dpar["T"])), \
        "Incorrect data shape."


def test_without_init_subset_modification(simulator):
    """Test Initialization with __init__ and non-default parameters.

    With a modification of only a subset of paramters.
    """
    P = 11
    N = 3
    T = 890
    simulator.par["P"] = P
    simulator.par["N"] = N
    simulator.par["T"] = T
    simulator.update_ctx_state()

    assert (simulator.ctx["data"].shape == (P, N, T)), \
        "Incorrect data shape."
