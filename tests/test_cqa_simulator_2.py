#!/usr/bin/env python3

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

def test_correct_data_loaded_reinit(simulator):
    """Test whether model updates the context state with new paramters. Reinitialization method."""
    dpar = {
            # Model parameters
            "P": 1,  # NOTE: should be other than default
            "N": 4,  # Neurons
            "T": 100,  # nb places
        }
    simulator.par = dpar.copy()
    simulator.__init__(simulator.par)  # This is necessary

    expected_shape = (simulator.par["P"], simulator.par["N"], simulator.par["T"])
    assert simulator.ctx["data"].shape == expected_shape

def test_correct_data_loaded_asarg(simulator):
    """Test whether model updates the context state with new paramters. Argument method."""
    dpar = {# NOTE: should be other than default
            "P": 1,
            "N": 4,  # Neurons
            "T": 100,  # nb places
        }
    simulator = CqaSimulator(par=dpar)

    expected_shape = (simulator.par["P"], simulator.par["N"], simulator.par["T"])
    assert simulator.ctx["data"].shape == expected_shape


def test_correct_data_loaded_updctx(simulator):
    """Test whether model updates the context state with new paramters. Argument update_ctx method."""
    simulator.par["P"] = 1
    simulator.par["N"] = 4
    simulator.par["T"] = 100

    simulator.update_ctx_state()  # Do this after paramter change!

    expected_shape = (simulator.par["P"], simulator.par["N"], simulator.par["T"])
    assert simulator.ctx["data"].shape == expected_shape


def test_save_and_load_run_data(simulator):
    """Test saving and loading the model's data."""
    dpar = {
            # Model parameters
            "P": 1,
            "N": 4,  # Neurons
            "T": 1000,  # nb places
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
            "M": 1,  # average nb of fields (4.4 in real dt)
            "M_fixed": True,
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
    simulator.__init__(simulator.par)  # Reinitialize with new parameters # TODO: test with and withoout
    base_path = Path("/tmp/simulator_test")
    simulator.save_run_data(base_path)

    base_path = Path(base_path)
    fp = base_path / ("simpl_data" if dpar["simplified_data"] else "reali_data")
    fp = fp / f"g{dpar['g']}_kb{dpar['kb']}_step{dpar['initial_step_size']}"
    fp = fp / f"P{dpar['P']}_N{dpar['N']}"
    fp = fp / f"T{dpar['T']}_L{dpar['L']}"

    fn_params = [
        f"fieldM{dpar['M']}",
        f"heightM{dpar['mean_height']}",
        f"diaM{dpar['mean_diameter']}",
        f"diaVar{dpar['var_diameter']}",
        f"corrP{int(dpar['correlated_peaks'])}",
        f"gamma{dpar['gamma']}",
        # f"corrD{int(self.dpar['correlated_dimensions'])}",
        # f"exp{self.dpar['exponent']}",
        f"seed{dpar['seed']}",
    ]
    fn = "_".join(fn_params)
    assert (fp / f"data_{fn}.npy").exists(), "Data file not saved correctly."
    assert (fp / f"sim_data_{fn}.npy").exists(), "Sim data file not saved correctly."


# def test_record_final_false(simulator):
#     """Test saving and loading the full model using a temporary directory."""
#     # Create a temporary directory for testing
#     dpar = {
#             # Model parameters
#             "P": 1,
#             "N": 4,  # Neurons
#             "T": 1000,  # nb places
#             "L": 200,  # sampling of places
#             # Data parameters
#             "mean_diameter": 1.57,  # field mean
#             "var_diameter": 2.0,  # field width diameter
#             "correlated_peaks": False,  # Corrs. between field height and diameter
#             "simplified_data": False,
#             "correlated_dimensions": "max",  # only generating data for p > 1
#             "mean_height": 1.549,
#             "var_height": 0.0,
#             "exponent": 0,
#             "gamma": 0.5,
#             "M": 1,  # average nb of fields (4.4 in real dt)
#             "M_fixed": True,
#             # Simulation parameters
#             "seed": 1,
#             "track_dynamics_flag": True,
#             "record_final_flag": False,
#             "g": 18.0,
#             "kb": 300,  # omega / sigma / strenght of inh. feedback
#             "initial_step_size": 0.08,
#             "converge_eps": 1e-5,
#             "max_iterations": 6e4,
#         }
#     simulator.par = dpar.copy()
#     simulator.__init__(simulator.par)  # Reinitialize with new parameters
#     base_path = Path("/tmp/simulator_test")
#     simulator.save_run_data(base_path)

#     fn_params = [
#         f"fieldM{dpar['M']}",
#         f"heightM{dpar['mean_height']}",
#         f"diaM{dpar['mean_diameter']}",
#         f"diaVar{dpar['var_diameter']}",
#         f"corrP{int(dpar['correlated_peaks'])}",
#         f"gamma{dpar['gamma']}",
#         # f"corrD{int(self.dpar['correlated_dimensions'])}",
#         # f"exp{self.dpar['exponent']}",
#         f"seed{dpar['seed']}",
#     ]
#     fn = "_".join(fn_params)
#     with tempfile.TemporaryDirectory() as temp_dir:

#         base_path = Path(temp_dir)
#         fp = base_path / ("simpl_data" if dpar["simplified_data"] else "reali_data")
#         fp = fp / f"g{dpar['g']}_kb{dpar['kb']}_step{dpar['initial_step_size']}"
#         fp = fp / f"P{dpar['P']}_N{dpar['N']}"
#         fp = fp / f"T{dpar['T']}_L{dpar['L']}"

#         print(fp)
#         simulator.save_run_data(base_path)
#         x = (fp / f"data_{fn}.npy").exists()
#         assert not x, "Data file not saved correctly."
#         # assert (fp / f"sim_data_{fn}.npy").exists(), "Sim data file not saved correctly."
