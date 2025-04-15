#!/usr/bin/env python3
"""TBC.

TBC...
"""
import argparse
import pickle
import copy
from pathlib import Path
from itertools import combinations
import numpy as np

import src.utils as sut
import src.model as smo

from src.config import PATHS


def pad_with_nans(nested_list):
    """Pad nested list with nans for equal length and convert to array.

    Pad that all sublists are of same length, then
    convert to a numpy array.
    """
    # Determine maximum length of inner lists
    max_length = max(len(sublist) for sublist in nested_list)
    padded_list = [sublist + [np.nan]
                   * (max_length - len(sublist)) for sublist in nested_list]

    return np.array(padded_list)


# @sut.auto_numba
def gen_p_data(P, N, T, L,
               diametro_m, diametro_delta,
               correlations="min", simplified=True,
               verbose=0, seed=None):
    """Generate simplified multidimensional data with only variability in peak widths."""
    if seed is not None:
        np.random.seed(seed)

    if not simplified:
        raise NotImplementedError("Only simplified mode currently supported.")

    # Initialize data
    dt = np.zeros((P, N, T))

    # Generate simplified data (i.e. only variation in peak width)
    if simplified:
        # Peaks are non-overlapping in all dimensions, i.e. shifted peak centers
        if correlations == "min":
            for i in range(P):
                dt[i] = smo.gen_simplified_data(
                    N, T, L, diametro_m, diametro_delta, shift=i*50,  # TODO: shift should be dependent on dimetro_delta, L, T and N (>9)
                    seed=seed)
        # Peaks are maximally overlapping in all dimensions, i.e. same peak center
        elif correlations == "max":
            for i in range(P):
                dt[i] = smo.gen_simplified_data(
                    N, T, L, diametro_m, diametro_delta, seed=seed)
        else:
            raise ValueError("As of now correlations must be 0 or 1.")

    if verbose > 0:
        # print(f"Avg. corr. betw. dimensions = {avg_dim_correlations(dt):.3f}.")
        print("Avg. corr. betw. dimensions = {:.3f}.".format(avg_dim_correlations(dt)))

    return dt


# @sut.auto_numba
def create_hebb_p_weights(tensor_dt):
    """Create hebbian weights for different charts (P)."""
    if len(tensor_dt.shape) == 2:
        tensor_dt = tensor_dt[np.newaxis, :, :]

    assert len(tensor_dt.shape) == 3, "Data is not tensor."

    P, N, _ = tensor_dt.shape
    summed_W = np.zeros((N, N))
    for d in range(P):
        summed_W += smo.create_hebb_weights(tensor_dt[d])

    # Maybe normalize the sum by P?

    return summed_W


class CQA_Model():
    """Implementation of the Quasi Continous Attractor Model.

    Based on the original model of Schönsberg, Monaisson &
    Treves, 2024.
    """

    def __init__(self, params):
        """Initialize the model with a parameter dictionary."""
        # Initialize Parameters
        # ---------------------
        self.par = params

        # Add additional (fixed) parameters
        self.par["record_flag"] = False
        self.par["g"] = 17.0  # fixed gain paramter
        self.par["kb"] = 300  # omega / sigma / strenght of inh. feedback
        self.par["initial_step_size"] = 0.08
        self.par["convergence_criterion"] = 1e-5
        self.par["max_iterations"] = 6e4
        # Set seed
        if self.par["seed"] is not None:
            np.random.seed(self.par["seed"])

        # Initialize State variables
        # --------------------------
        self.ctx = {}

        # Generate data
        self.ctx["data"] = gen_p_data(
            self.par["P"], self.par["N"],
            self.par["T"], self.par["L"],
            self.par["mean_diameter"], self.par["var_diameter"],
            correlations=self.par["correlated_dimensions"],
            simplified=self.par["simplified_data"],
            seed=self.par["seed"],
        )
        self.ctx["data_mean"] = np.mean(self.ctx["data"][0])
        # Record simulated data
        self.ctx["sim_data"] = np.full_like(self.ctx["data"], np.nan)
        self.ctx["sim_pos"] = np.full((self.par["P"], self.par["T"]),
                                      np.nan, dtype=np.float64)
        # Record evolving states
        lst_init = [[] for _ in range(self.par["T"])]
        self.ctx["overlap_max"] = copy.deepcopy(lst_init)
        self.ctx["overlap_max_pos"] = copy.deepcopy(lst_init)
        self.ctx["overlap_disp"] = copy.deepcopy(lst_init)
        self.ctx["overlap_disp_clip"] = copy.deepcopy(lst_init)
        self.ctx["v_disp"] = copy.deepcopy(lst_init)

        # Get weights
        self.ctx["W"] = create_hebb_p_weights(self.ctx["data"])
        # Initialize state vector V
        self.ctx["V"] = self.ctx["data"][:, 0]
        self.ctx["V_prec"] = self.ctx["V"].copy()
        # Initialize state variables
        self.ctx["step_count"] = 0
        self.ctx["current_step_size"] = self.par["initial_step_size"]
        self.ctx["time_step"] = 0
        self.ctx["v_diff"] = 1
        self.ctx["converged"] = False

    def execute_step(self, verbose=0):
        """Perform one update step using Euler integration."""
        # Update variables
        self.ctx["time_step"] += 1

        # Update step
        aa = smo.one_step_dynamics_B(
            self.ctx["data_mean"], self.ctx["V"], self.ctx["W"],
            self.par["g"], self.par["kb"],
        )
        self.ctx["V"] = (1 - self.ctx["current_step_size"]) * self.ctx["V"] \
            + self.ctx["current_step_size"] * aa

        # Check convergence
        self.ctx["v_diff"] = np.sum(np.abs(self.ctx["V"] - self.ctx["V_prec"]))
        self.ctx["converged"] = self.ctx["v_diff"] < \
            self.par["convergence_criterion"]
        if verbose > 0 and self.ctx["converged"]:
            print(f"Model converged after {self.ctx["time_step"]} steps.")

        # Potentially shorten step size for next step
        # (if v_diff is still high after modulo 600 steps)
        # NOTE: v_diff is diff_vec in FS's code
        if self.ctx["time_step"] % 600 == 0 and self.ctx["v_diff"] > 0.07:
            self.ctx["current_step_size"] -= self.ctx["current_step_size"] / 3.0
        self.ctx["V_prec"] = self.ctx["V"].copy()

    def run_until_convergence_over_positions(
            self, spacing=20, record=True, verbose=2):
        """Run until model converges, for all intialization in data."""
        # Run over all dimensions
        for p in range(self.par["P"]):
            # Run over evenly sampled positions, each until convergence
            sampled_pos_indices = np.arange(0, self.par["T"], spacing)
            for init_pos in sampled_pos_indices:
                if verbose > 1:
                    print("p =", p, ", init_pos =", init_pos,
                          " (var_dia =", self.par["var_diameter"], ")")

                self.run_until_convergence(init_pos, p=0, record=record,  # TODO: change p
                                           verbose=verbose)

    def run_until_convergence(
            self, init_pos, p=0, record_flag=True, verbose=2):
        """Run until model converges, for specific position."""
        # self.curr_p_and_pos = (p, curr_pos)
        self.par["record_flag"] = record_flag

        # Reinitialize variables
        self.ctx["V"] = self.ctx["data"][p][:, init_pos]
        self.ctx["V_prec"] = self.ctx["V"].copy()
        self.ctx["current_step_size"] = self.par["initial_step_size"]

        self.ctx["time_step"] = 0
        self.ctx["converged"] = False
        max_iter_reached = False
        # Run until convergence or max_iter. per dim and sampled pos
        while not (self.ctx["converged"] or max_iter_reached):
            # Exist
            max_iter_reached = self.ctx["time_step"] > self.par["max_iterations"]
            if max_iter_reached:
                print("Maximum steps reached. Stopping.")

            # Execute step updating V, V_prec, time_step
            self.execute_step(verbose=verbose)

            if self.par["record_flag"]:
                o_curr = smo.cosine_similarity_vec(self.ctx["data"][p],
                                                   self.ctx["V"])
                o_max, o_max_pos = np.max(o_curr), np.argmax(o_curr)
                o_curr_clip = np.maximum(o_curr - 0.1, 0)

                self.ctx["overlap_max"][init_pos].append(o_max)
                self.ctx["overlap_max_pos"][init_pos].append(o_max_pos)
                self.ctx["overlap_disp"][init_pos].append(smo.std_cdm(o_curr))
                self.ctx["overlap_disp_clip"][init_pos].append(smo.std_cdm(o_curr_clip))
                self.ctx["v_disp"][init_pos].append(smo.std_cdm(self.ctx["V"]))

        if self.par["record_flag"]:
            # Record only if converged.
            if self.ctx["converged"]:
                self.ctx["sim_data"][p][:, init_pos] = self.ctx["V"]
                self.ctx["sim_pos"][p][init_pos] = o_max_pos
            # Set inf if timeout
            elif max_iter_reached:
                self.ctx["sim_data"][p][:, init_pos] = np.inf

    def run_N_steps(self, n_steps=100, record_flag=False):
        """Run the dynamics for N steps.

        Args:
            steps (int): Number of steps.
            dt (float): Time step.
            record (bool): Whether to record the states over time.

        Returns:
            np.ndarray: Recorded states if `record=True`, else None.
        """
        self.par["record_flag"] = record_flag
        history = [] if self.par["record_flag"] else None

        self.ctx["time_step"] = 0
        while self.ctx["time_step"] < n_steps:
            self.execute_step()
            self.ctx["time_step"] += 1

            if self.par["record_flag"]:
                history.append(self.ctx["V"].copy())

        return np.array(history) if self.par["record_flag"] else None

    def save_run_data(self, path, create_dir=True):
        """Save data of a run."""
        assert self.par["record_flag"] is True, "Data was not recorded."

        path = Path(path)
        if create_dir:
            path.mkdir(parents=True, exist_ok=True)

        if self.par["simplified_data"]:
            fn_params = [
                f"P{self.par['P']}",
                f"N{self.par['N']}",
                f"T{self.par['T']}",
                f"L{self.par['L']}",
                # f"mean{self.par['mean_diameter']}",
                f"var{self.par['var_diameter']}",
                f"simp{'1' if self.par['simplified_data'] else '0'}",
                f"corrD{self.par['correlated_dimensions']}",
                # f"g{self.par['g']}",
                # f"kb{self.par['kb']}",
                # f"step{self.par['initial_step_size']}",
                # f"eps{self.par['convergence_criterion']:.0e}",
                # f"maxIter{int(self.par['max_iterations']):.0e}",
                f"seed{self.par['seed']}",
            ]
        fn = "_".join(fn_params)

        # Save simulated data (converged)
        np.save(path / f"sim_data{fn}.npy", self.ctx["sim_data"])
        np.save(path / f"sim_pos{fn}.npy", self.ctx["sim_pos"])
        # Save evolving data
        np.save(path / f"overlap_max{fn}.npy",
                pad_with_nans(self.ctx["overlap_max"]))
        np.save(path / f"overlap_max_pos{fn}.npy",
                pad_with_nans(self.ctx["overlap_max_pos"]))
        np.save(path / f"overlap_disp{fn}.npy",
                pad_with_nans(self.ctx["overlap_disp"]))
        np.save(path / f"overlap_disp_clip{fn}.npy",
                pad_with_nans(self.ctx["overlap_disp_clip"]))
        np.save(path / f"v_disp{fn}.npy",
                pad_with_nans(self.ctx["v_disp"]))

    def save_full_model(self, path):
        """Save entire model."""
        path = Path(path)
        with open(path / "full_model.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_full_model(path):
        """Load model."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __str__(self):
        """Report essential model information."""
        return (
            f"{self.__class__.__name__}("
            f"params={self.par})"
        )

    def __repr__(self):
        """Report precise model information."""
        return (
            f"{self.__class__.__name__}("
            f"params={self.par}"
            f"ctx={self.ctx})"
        )


if __name__ == '__main__':

    params = {
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
        "correlated_dimensions": "max",
        # for now unused
        "mean_height": 1.549,
        "exponent": 0,
        "M": 4.4,  # average nb of fields
        # Simulation parameters
        "seed": 1,
    }
    fp = PATHS["simpl_data"]
    var_dias = np.round(np.arange(0.0, 2.0, 0.05), 2).tolist()
    var_dias = [0.65]
    for i, var_d in enumerate(var_dias):
        print(f"\n Run {i}/{len(var_dias)} with diameter variance {var_d}")

        params["var_diameter"] = var_d
        cqa = CQA_Model(params)
        cqa.run_until_convergence_over_positions(spacing=100)
        cqa.save_run_data(fp)
    cqa.save_full_model(fp)
