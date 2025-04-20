#!/usr/bin/env python3
"""Default parameters for cqa_simulator.py and for cqa_fileloader.py."""
from pathlib import Path


class CqaDefaults:
    """Defines the default parameters for the CQA model.

    This class initializes the model with a set of default parameters
    for simulation and data processing. The parameters can be
    overridden by passing a dictionary when initializing a derived class.

    Used as superclass for the CqaSimulator and the CqaFileLoader.

    Attributes:
        default_par (dict): A dictionary containing default values for model,
                             data, and simulation parameters.
    Methods:
        build_fp(base_path): Constructs a file path based on the default
                              parameters.
    """

    def __init__(self):
        """Initialize with default paramters."""
        self.default_par = {
            # Model parameters
            "P": 1,
            "N": 999,  # Neurons
            "T": 1000,  # nb places
            "L": 200,  # sampling of places
            # Data parameters
            "mean_diameter": 1.57,  # field mean
            "var_diameter": 2.0,  # field width diameter
            "correlated_peaks": False,  # Corrs. between field height and diameter
            "simplified_data": True,
            "correlated_dimensions": "min",  # only generating data for p > 1
            "mean_height": 1.549,
            "var_height": 0.0,
            "exponent": 0,
            "gamma": 0.5,
            "M": 1,  # average nb of fields (4.4 in real dt)
            "M_fixed": True,
            # Simulation parameters
            "seed": 1,
            "track_dynamics_flag": False,
            "record_final_flag": True,
            "g": 17.0,
            "kb": 300,  # omega / sigma / strenght of inh. feedback
            "initial_step_size": 0.08,
            "converge_eps": 1e-5,
            "max_iterations": 6e4,
            "verbose": False,
            "spacing": None,
        }

    def build_fp(self, base_path):
        """Build folder path from default parameters."""
        if not hasattr(self, 'par'):
            self.par = self.default_par

        base_path = Path(base_path)
        fp = base_path / ("simpl_data" if self.par["simplified_data"] else "reali_data")
        fp = fp / f"g{self.par['g']}_kb{self.par['kb']}_step{self.par['initial_step_size']}"
        fp = fp / f"P{self.par['P']}_N{self.par['N']}"
        fp = fp / f"T{self.par['T']}_L{self.par['L']}"
        fp = fp / f"fieldM{self.par['M']}_fixedM{self.par['M_fixed']}"
        fp = fp / f"corrP{int(self.par['correlated_peaks'])}_gamma{self.par['gamma']}"
        fp = fp / f"heightM{self.par['mean_height']}_heightVar{self.par['mean_height']}"
        fp = fp / f"diaM{self.par['mean_diameter']}_diaVar{self.par['var_diameter']}"

        return fp

    def build_fn(self):
        """Build file name from default parameters."""
        # if not self.par:  # TODO: test with simulator, does it really update even if a paramter is changed?
        if not hasattr(self, 'par'):
            self.par = self.default_par
        fn_params = [
            # f"corrD{int(self.default_par['correlated_dimensions'])}",
            # f"exp{self.default_par['exponent']}",
            f"seed{self.par['seed']}",
        ]
        fn = "_".join(fn_params)
        return fn

    def __str__(self):
        """Return a human-readable string representation of the CqaModel."""
        return (f"CqaModel(par={self.default_par})")

    def __repr__(self):
        """Return a detailed string representation of the CqaModel."""
        return (f"CqaModel(par={self.default_par})")
