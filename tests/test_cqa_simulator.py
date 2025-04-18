#!/usr/bin/env python3
import pytest
import numpy as np
import tempfile
# from unittest.mock import patch
from pathlib import Path

from src.cqa_simulator import CqaSimulator


@pytest.fixture
def simulator():
    return CqaSimulator()

def test_initialization(simulator):
    """Test if simulator initializes with default parameters correctly."""
    assert isinstance(simulator.par, dict)
    assert simulator.par["P"] == 1
    assert simulator.par["seed"] == 1
    assert simulator.ctx["V"] is not None  # State variables
    assert simulator.ctx["sim_data"].shape == (
        simulator.par["P"], simulator.par["N"], simulator.par["T"]
    )

def test_data_generation_with_simplified_data(simulator):
    """Test data generation when using simplified data flag."""
    simulator.par["simplified_data"] = True
    simulator.par["P"] = 2  # Multiple positions for testing
    simulator.par["N"] = 500
    simulator.par["T"] = 100
    simulator.par["L"] = 50

    simulator.__init__(simulator.par)  # Reinitialize with new parameters

    assert "data" in simulator.ctx
    assert simulator.ctx["data"].shape == (
        simulator.par["P"], simulator.par["N"], simulator.par["T"]
    )

def test_execute_step_simplified_data(simulator):
    """Test if execute_step method updates the state correctly."""
    simulator.par["simplified_data"] = True
    simulator.__init__(simulator.par)  # Reinitialize with default parameters
    initial_V = simulator.ctx["V"].copy()
    simulator.execute_step(verbose=0)
    assert not np.allclose(simulator.ctx["V"], initial_V), \
        "State V should change after step execution."

def test_execute_step_realistic_data(simulator):
    """Test if execute_step method updates the state correctly."""
    simulator.par["simplified_data"] = False
    simulator.__init__(simulator.par)  # Reinitialize with default parameters
    initial_V = simulator.ctx["V"].copy()
    simulator.execute_step(verbose=0)
    assert not np.allclose(simulator.ctx["V"], initial_V), \
        "State V should change after step execution."

# def test_convergence_criterion(simulator):  # This test does not work because one cannot change the paramters without reinitializing!
#     """Test if the convergence criterion works correctly."""
#     simulator.ctx["V"] = np.ones_like(simulator.ctx["V"])  # Set initial state
#     simulator.ctx["V_prec"] = np.zeros_like(simulator.ctx["V"])  # Set previous state to zero

#     # Execute one step, should not converge initially
#     simulator.execute_step(verbose=0)
#     assert not simulator.ctx["converged"], "Model should not converge after one step."

#     # Force convergence by setting a small difference
#     simulator.ctx["V"] = simulator.ctx["V_prec"]
#     simulator.__init__(simulator.par)  # Reinitialize with new parameters
#     simulator.execute_step(verbose=0)
#     assert simulator.ctx["converged"], "Model should converge when state difference is below criterion."

def test_run_until_max_iter_reached(simulator):
    """Test if the model runs until convergence."""
    simulator.par["max_iterations"] = 100  # Set small number of iterations for testing
    simulator.run_until_convergence(init_pos=0, p=0, verbose=1)
    assert simulator.ctx["max_iter_reached"], \
        "The model should reach max_iterations within the given iterations."

def test_run_n_steps(simulator):
    """Test running the model for a fixed number of steps."""
    history = simulator.run_N_steps(n_steps=50, track_dynamics_flag=True)

    assert history.shape == (50, simulator.par["N"], simulator.par["T"]), \
        "History should contain the correct number of steps."

def test_save_and_load_run_data(simulator):
    """Test saving and loading the model's data."""
    simulator.par["simplified_data"] = False
    simulator.__init__(simulator.par)  # Reinitialize with new parameters
    path = Path("/tmp/simulator_test")
    simulator.save_run_data(path)
    path = path / "reali_data/g17.0_kb300_step0.08" / "P1_N999" / "T1000_L200"
    assert (path / "data_fieldM1_heightM1.549_diaM1.57_diaVar2.0_corrP0_gamma0.5_seed1.npy").exists(), "Data file was not saved correctly."
    assert (path / "sim_data_fieldM1_heightM1.549_diaM1.57_diaVar2.0_corrP0_gamma0.5_seed1.npy").exists(), "Sim data file was not saved correctly."

# def test_save_full_model(simulator):
#     """Test saving and loading the full model."""
#     path = Path("/tmp/simulator_test_model")
#     simulator.save_full_model(path)

#     assert (path / "full_model.pkl").exists(), \
#         "Full model file was not saved correctly."

def test_save_full_model_and_load(simulator):
    """Test saving and loading the full model using a temporary directory."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)
        print(path)
        # Save the full model
        simulator.save_full_model(path)
        # Assert that the full model file exists
        model_file = path / "cqa_model.pkl"
        assert model_file.exists(), "Full model file was not saved correctly."
        # Load the model
        loaded_simulator = CqaSimulator.load_full_model(model_file)
        # Assert that the loaded model is of the correct type
        assert isinstance(loaded_simulator, CqaSimulator), \
            "Loaded model should be of type CqaSimulator."
    # Temporary directory and its contents will be deleted automatically after the with block



def test_string_representation(simulator):
    """Test the string and repr representation."""
    s = str(simulator)
    r = repr(simulator)
    assert "CqaSimulator" in s
    assert "CqaSimulator" in r
    assert "params" in s
    assert "ctx" in r
