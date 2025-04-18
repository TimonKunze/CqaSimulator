#!/usr/bin/env python3
"""Reimplementation of the Quasi Continous Attractor Model.

But generalized to multi-dimensional input.
Based on the original model of Schönsberg, Monaisson & Treves, 2024.
"""
import os
import pickle
import copy
import numpy as np

from pathlib import Path
from typing import Optional, Dict, Any

from cqasim.cqa_defaults import CqaDefaults
from cqasim.cqa_gendata import gen_p_data
from cqasim.cqa_weights import create_hebb_p_weights
from cqasim.cqa_utils import get_loop_indices, pad_with_nans
from cqasim.cqa_vizutils import animate_1d_activity
from cqasim.cqa_vectutils import calc_hessian, calc_sparsity, \
    calc_support, cosine_similarity, one_step_dynamics, relu, \
    cosine_similarity_vec, std_cdm

from src.config import PATHS


class CqaSimulator(CqaDefaults):
    """Simulator class for the model."""

    def __init__(self, par: Optional[Dict[str, Any]] = None):
        """Initialize the model with a parameter dictionary."""
        # Initialize Parameters
        # ---------------------

        # Get par dictionary from CqaDefault.py
        super().__init__()
        self.par = self.default_par.copy()
        # Ammend with custom entries potentially
        if par is not None:
            self.par.update(par)

        if self.par["seed"] is not None:
            np.random.seed(self.par["seed"])

        # Initialize State variables
        # --------------------------
        self.ctx = {}
        self.update_ctx_state()

    def update_ctx_state(self):
        """Initialize context states.

        Run manually after each paramter change!
        """
        # Generate data
        if self.par["M_fixed"]:
            self.par["M"] = int(self.par["M"])

        dt, diams_per_nrn, heights_per_nrn, fields_per_nrn = gen_p_data(
            self.par["P"], self.par["N"],
            self.par["T"], self.par["L"],
            self.par["M"],
            self.par["mean_diameter"], self.par["var_diameter"],
            self.par["mean_height"], self.par["var_height"],
            self.par["correlated_peaks"], self.par["gamma"],
            self.par["correlated_dimensions"],
            self.par["M_fixed"],
            simplified=self.par["simplified_data"],
            verbose=self.par["verbose"],
            seed=self.par["seed"],
            )
        self.ctx["data"] = dt
        self.ctx["diams_per_nrn"] = diams_per_nrn
        self.ctx["heights_per_nrn"] = heights_per_nrn
        self.ctx["fields_per_nrn"] = fields_per_nrn

        self.ctx["data_mean"] = np.mean(self.ctx["data"][0])  # TODO: p index for multid data!
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

        # Initialize state vector V and preceding ones
        self.ctx["V"] = self.ctx["data"][0]  # TODO: p index for multid data!
        self.ctx["V_prec"] = self.ctx["V"].copy()
        self.ctx["V_prec_prec"] = self.ctx["V"].copy()

        # Initialize state variables
        self.ctx["step_count"] = 0
        self.ctx["current_step_size"] = self.par["initial_step_size"]
        self.ctx["time_step"] = 0
        self.ctx["v_diff"] = 1
        self.ctx["converged"] = False
        self.ctx["max_iter_reached"] = False

    def execute_step(self, verbose=0):
        """Perform one update step using Euler integration."""
        # Update variables
        self.ctx["time_step"] += 1

        # Update step
        aa = one_step_dynamics(
            self.ctx["data_mean"], self.ctx["V"], self.ctx["W"],
            self.par["g"], self.par["kb"],
        )
        self.ctx["V"] = (1 - self.ctx["current_step_size"]) * self.ctx["V"] \
            + self.ctx["current_step_size"] * aa

        # Check convergence
        self.ctx["v_diff"] = np.sum(np.abs(self.ctx["V"] - self.ctx["V_prec"]))
        self.ctx["converged"] = self.ctx["v_diff"] < self.par["converge_eps"]
        if verbose > 0 and self.ctx["converged"]:
            print(f"Model converged after {self.ctx["time_step"]} steps.")

        # Potentially shorten step size for next step
        # (if v_diff is still high after modulo 600 steps)
        # NOTE: v_diff is diff_vec in FS's code
        if self.ctx["time_step"] % 600 == 0 and self.ctx["v_diff"] > 0.07:
            self.ctx["current_step_size"] -= self.ctx["current_step_size"] / 3.0
        self.ctx["V_prec"] = self.ctx["V"].copy()

    def run_until_convergence(
            self, init_pos, p=0, record_final_flag=None,
            visualize=False, verbose=2):
        """Run until model converges, for specific position."""
        # self.curr_p_and_pos = (p, curr_pos)
        if record_final_flag is not None:
            self.par["record_final_flag"] = record_final_flag

        # Reinitialize variables
        self.ctx["V"] = self.ctx["data"][p][:, init_pos]
        self.ctx["V_prec"] = self.ctx["V"].copy()
        self.ctx["current_step_size"] = self.par["initial_step_size"]

        self.ctx["time_step"] = 0
        self.ctx["converged"] = False
        self.ctx["max_iter_reached"] = False

        visualize_V = []  # for visualization
        visualize_o = []  # for visualization

        # Run until convergence or max_iter. per dim and sampled pos
        # curr_pos = np.copy(init_pos)
        while not (self.ctx["converged"] or self.ctx["max_iter_reached"]):
            # Exit loop
            self.ctx["max_iter_reached"] = self.ctx["time_step"] > \
                                           self.par["max_iterations"]
            if self.ctx["max_iter_reached"]:
                print(self.par["max_iterations"])
                print("Maximum steps reached. Stopping.")

            # Execute step updating V, V_prec, time_step
            self.execute_step(verbose=verbose)

            if visualize:
                o_curr = cosine_similarity_vec(
                    self.ctx["data"][p], self.ctx["V"])
                visualize_o.append(o_curr)
                visualize_V.append(self.ctx["V"])

            if self.par["track_dynamics_flag"]:
                o_curr = cosine_similarity_vec(self.ctx["data"][p],
                                               self.ctx["V"])
                o_max, o_max_pos = np.max(o_curr), np.argmax(o_curr)
                o_curr_clip = np.maximum(o_curr - 0.1, 0)

                self.ctx["overlap_max"][init_pos].append(o_max)
                self.ctx["overlap_max_pos"][init_pos].append(o_max_pos)
                self.ctx["overlap_disp"][init_pos].append(std_cdm(o_curr))
                self.ctx["overlap_disp_clip"][init_pos].append(std_cdm(o_curr_clip))
                self.ctx["v_disp"][init_pos].append(std_cdm(self.ctx["V"]))

        if self.par["record_final_flag"]:
            # Record only if converged.
            if self.ctx["converged"]:
                o_curr = cosine_similarity_vec(self.ctx["data"][p],
                                               self.ctx["V"])
                o_max_pos = np.argmax(o_curr)
                self.ctx["sim_data"][p][:, init_pos] = self.ctx["V"]
                self.ctx["sim_pos"][p][init_pos] = o_max_pos
            # Set inf if timeout reached
            elif self.ctx["max_iter_reached"]:
                self.ctx["sim_data"][p][:, init_pos] = np.inf

        if visualize:
            if self.par["P"] == 1:
                animate_1d_activity(visualize_V, visualize_o,
                                    self.ctx["data"][0].T[init_pos],
                                    ymin=np.min(visualize_V),
                                    ymax=np.max(visualize_V),
                                    pause=0.05)
            elif self.par["P"] == 2:
                pass
            else:
                print(f"No visualization for d={self.par["P"]}")

            # # Detect significant jump
            # jump_detected = detect_significant_jump(
            #     curr_pos, o_max_pos, o, threshold=5)
            # # NOTE: Maybe make 6 or 7 to be on par with FS. because
            # # it now includes start and end index

            # # Append data
            # if jump_detected:
            #     if init_pos not in saltato_da:
            #         saltato_da.append(init_pos)
            #     if o_max_pos not in approdato_a:
            #         approdato_a.append(o_max_pos)

            # # Prepare next iteration
            # curr_pos = np.copy(o_max_pos)

    def run_until_convergence_over_positions(
            self, spacing=20, record_final_flag=None, verbose=2):
        """Run until model converges, for all intialization in data."""
        if record_final_flag is not None:
            self.par["record_final_flag"] = record_final_flag
        # Run over all dimensions
        for p in range(self.par["P"]):
            # Run over evenly sampled positions, each until convergence
            sampled_pos_indices = np.arange(0, self.par["T"], spacing)
            for init_pos in sampled_pos_indices:
                if verbose > 1:
                    print("p =", p, ", init_pos =", init_pos,
                          " (var_dia =", self.par["var_diameter"], ")")

                self.run_until_convergence(
                    init_pos, p=0,  # TODO: change p
                    record_final_flag=self.par["record_final_flag"],
                    verbose=verbose)

    def run_N_steps(self, n_steps=100, track_dynamics_flag=None):
        """Run the dynamics for N steps.

        Args:
            steps (int): Number of steps.
            dt (float): Time step.
            record (bool): Whether to record the states over time.

        Returns:
            np.ndarray: Recorded states if `record=True`, else None.
        """
        if track_dynamics_flag is not None:
            self.par["track_dynamics_flag"] = track_dynamics_flag

        history = []
        self.ctx["time_step"] = 0
        while self.ctx["time_step"] < n_steps:
            self.execute_step()
            if self.par["track_dynamics_flag"]:
                history.append(self.ctx["V"].copy())

        return np.array(history)

    def save_run_data(self, path, parameter_dir=True):
        """Save data of a run."""
        assert self.par["record_final_flag"] is True, "No data recorded."

        if parameter_dir:
            fp = self.build_fp(path)
            print(fp)
            os.makedirs(fp, exist_ok=True)
        else:
            fp = Path(path)

        fn = self.build_fn()
        print(fn)
        # TODO: test with simulator, does it really update even if a paramter is changed?

        # Save initialization data
        np.save(fp / f"data_{fn}.npy", self.ctx["data"])
        np.save(fp / f"diams_per_nrn_{fn}.npy", self.ctx["diams_per_nrn"])
        np.save(fp / f"heights_per_nrn_{fn}.npy", self.ctx["heights_per_nrn"])
        np.save(fp / f"fields_per_nrn_{fn}.npy", self.ctx["fields_per_nrn"])
        # Save simulated data (converged)
        np.save(fp / f"sim_data_{fn}.npy", self.ctx["sim_data"])
        np.save(fp / f"sim_pos_{fn}.npy", self.ctx["sim_pos"])
        # Save evolving data
        if self.par["track_dynamics_flag"]:
            np.save(fp / f"overlap_max_{fn}.npy", pad_with_nans(self.ctx["overlap_max"]))
            np.save(fp / f"overlap_max_pos_{fn}.npy", pad_with_nans(self.ctx["overlap_max_pos"]))
            np.save(fp / f"overlap_disp_{fn}.npy", pad_with_nans(self.ctx["overlap_disp"]))
            np.save(fp / f"overlap_disp_clip_{fn}.npy", pad_with_nans(self.ctx["overlap_disp_clip"]))
            np.save(fp / f"v_disp_{fn}.npy", pad_with_nans(self.ctx["v_disp"]))

    def save_full_model(self, path, parameter_dir=False):
        """Save entire model."""
        if parameter_dir:
            fp = self.build_fp(path)
        else:
            fp = Path(path)

        with open(fp / "cqa_model.pkl", "wb") as f:
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

    # Run and save for multiple positions
    # ===================================
    # fp = PATHS["runs"]
    # # var_dias = np.round(np.arange(0.0, 2.0, 0.05), 2).tolist()
    # var_dias = [0.5]
    # for i, var_d in enumerate(var_dias):
    #     print(f"\n Run {i}/{len(var_dias)} with diameter variance {var_d}")

    #     cqa = CqaSimulator()
    #     cqa.par["var_diameter"] = var_d
    #     cqa.run_until_convergence_over_positions(spacing=1)
    #     cqa.save_run_data(fp)
    # cqa.save_full_model(fp)

    # Run and visualize for specific position
    # =======================================
    cqa = CqaSimulator({
        "var_diameter": 0.1,
        "var_heights": 0.1,
        "N": 3000,
        "M": 4.4,
        "correlated_peaks": True,
        "M_fixed": False,
        "seed": 1,
    })
    # cqa.par["var_diameter"] = 1.3
    # cqa.par["N"] = 6000
    # cqa.par["M"] = 4.4
    # cqa.par["correlated_peaks"] = True
    # cqa.par["M_fixed"] = False
    # cqa.update_ctx_state()
    print(cqa)
    init_pos = 100
    cqa.run_until_convergence(
        init_pos, p=0, record_final_flag=None, verbose=2,
        visualize=True,
    )
