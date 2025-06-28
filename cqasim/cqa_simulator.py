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


class CqaSimulator(CqaDefaults):
    """Simulator class for the model."""

    def __init__(self, par: Optional[Dict[str, Any]] = None):
        """Initialize the model with a parameter dictionary."""
        # Initialize Parameters
        # ---------------------

        # Get par dictionary from CqaDefault.py
        super().__init__()

        self.par = self.default_par.copy()

        # print("track_quantified_dynamics" in self.par.keys())
        # print("track_quantified_dynamics" in self.default_par.keys())

        # Ammend with custom entries potentially
        if par is not None:
            self.par.update(par)

        if self.par["seed"] is not None:
            np.random.seed(self.par["seed"])

        # Initialize State variables
        # --------------------------
        self.ctx = {}
        self.update_ctx_state()
    #     self.update_states()

    # TODO: maybe find better update logic without :
    # cqa.__init__(cqa.par)  # This is important!!
    # cqa.update_ctx_state()  # This is important!!

    # def update_states(self):
    #     """Update model states."""
    #     self.update_par_state()
    #     self.update_ctx_state()

    # def update_par_state(self):
    #     """Update parameter states."""
    #     self.__init__(self.par)  # Reinitialize with new parameters

    def update_ctx_state(self):
        """Initialize context states.

        Run manually after each parameter change!
        """
        # Generate data
        dt, diams_per_nrn, heights_per_nrn, fields_per_nrn = gen_p_data(
            self.par["P"], self.par["N"],
            self.par["T"], self.par["L"],
            self.par["Zeta"],
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

        # Record evolving states
        lst_init = [[] for _ in range(self.par["T"])]
        if self.par["track_quantified_dynamics"]:
            self.ctx["overlap_max"] = copy.deepcopy(lst_init)
            self.ctx["overlap_max_pos"] = copy.deepcopy(lst_init)
            self.ctx["overlap_disp"] = copy.deepcopy(lst_init)
            self.ctx["overlap_disp_clip"] = copy.deepcopy(lst_init)
            self.ctx["v_disp"] = copy.deepcopy(lst_init)
        # if self.par["track_full_dynamics"]:
        self.ctx["V_dynamics"] = copy.deepcopy(lst_init)

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
                visualize_V.append(self.ctx["V"].copy())

            if self.par["track_full_dynamics"]:
                self.ctx["V_dynamics"][init_pos].append(self.ctx["V"].copy())

            if self.par["track_quantified_dynamics"]:
                o_curr = cosine_similarity_vec(self.ctx["data"][p],
                                               self.ctx["V"])
                o_max, o_max_pos = np.max(o_curr), np.argmax(o_curr)
                o_curr_clip = np.maximum(o_curr - 0.2, 0)

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
        self.ctx["spacing"] = spacing

        if record_final_flag is not None:
            self.par["record_final_flag"] = record_final_flag
        # Run over all dimensions
        for p in range(self.par["P"]):
            # Run over evenly sampled positions, each until convergence
            sampled_pos_indices = np.arange(0, self.par["T"], spacing)
            for init_pos in sampled_pos_indices:
                if verbose > 1:
                    print(f"p = {p}, init_pos = {init_pos} "
                          f"(Zeta={self.par['Zeta']}/"
                          f"M_fixed={self.par['M_fixed']},"
                          f" var_height={self.par['var_height']}, "
                          f"var_dia={self.par['var_diameter']})")

                self.run_until_convergence(
                    init_pos, p=0,  # TODO: change p
                    record_final_flag=self.par["record_final_flag"],
                    verbose=verbose)

    def save_run_data(
            self, path, spacing=None, parameter_dir=True, verbose=True):
        """Save data of a run."""
        assert self.par["record_final_flag"] is True, "No data was recorded."

        path = Path(path)
        fp = self.build_fp(path) if parameter_dir else path
        fp.mkdir(parents=True, exist_ok=True)  # pathlib way

        self.data_to_save(spacing)

        for fn, data in self.ctx["to_save"].items():
            if fn.endswith(".npy"):
                np.save(fp / fn, data)
            elif fn.endswith(".npz"):
                np.savez(fp / fn, **{f"{i}": arr for i, arr in enumerate(data)})
            else:
                print("Error: Unsupported file format. "
                      "Supported formats are: .npy, .npz")
            if verbose:
                print(f"[SAVED] {fp / fn}")

    def load_run_data(self, path, parameter_dir=True):
        """Load data of a full run."""
        if self.check_files_exist(path):
            fp = self.build_fp(path) if parameter_dir else path
            fn = self.build_fn(path)

            data = np.load(fp / fn)  # TODO: finish


            # # Load them back
            # loaded = np.load("arrays.npz")
            # arr_list_loaded = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]

    def check_files_exist(self, path, parameter_dir=True,
                          spacing=None, verbose=True):
        """Check whether each file in ctx['to_save'] exists on disk."""
        # Check file path
        path = Path(path)
        fp = self.build_fp(path) if parameter_dir else path
        self.data_to_save(spacing)

        # Create file status dir
        file_statuses = {f: (fp / f).exists() for f in self.ctx["to_save"]}
        if verbose:
            print(f"[INFO] CHECKING FILE EXIST FOR: {fp.resolve()}")
            for f, exists in file_statuses.items():
                status = "FOUND" if exists else "MISSING"
                print(f"[CHECK] {f}: {status}")

        return file_statuses, fp

    def data_to_save(self, spacing=None):
        """Determine which files to save."""
        fn = self.build_fn()
        if spacing is not None:
            fn += f"_spc{spacing}"

        self.ctx["to_save"] = {
            # Save initialization data
            f"data_{fn}.npy": self.ctx["data"],
            f"diams_per_nrn_{fn}.npy": self.ctx["diams_per_nrn"],
            f"heights_per_nrn_{fn}.npy": self.ctx["heights_per_nrn"],
            f"fields_per_nrn_{fn}.npy": self.ctx["fields_per_nrn"],
            # Save simulated data (converged)
            f"sim_data_{fn}.npy": self.ctx["sim_data"],
            f"sim_pos_{fn}.npy": self.ctx["sim_pos"],
        }
        # Add dynamic tracking data if enabled
        if self.par["track_quantified_dynamics"]:
            self.ctx["to_save"].update({
                f"overlap_max_{fn}.npy": pad_with_nans(self.ctx["overlap_max"]),
                f"overlap_max_pos_{fn}.npy": pad_with_nans(self.ctx["overlap_max_pos"]),
                f"overlap_disp_{fn}.npy": pad_with_nans(self.ctx["overlap_disp"]),
                f"overlap_disp_clip_{fn}.npy": pad_with_nans(self.ctx["overlap_disp_clip"]),
                f"v_disp_{fn}.npy": pad_with_nans(self.ctx["v_disp"]),
            })
        if self.par["track_full_dynamics"]:
            v_dyn_arr_lst = [np.array(pos) for pos in self.ctx["V_dynamics"]]
            self.ctx["to_save"].update({f"v_dynamics_{fn}.npz": v_dyn_arr_lst})

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

    # # Run and save for multiple positions
    # # ===================================
    fp = "./temp"
    # var_dias = np.round(np.arange(0.0, 2.0, 0.05), 2).tolist()
    var_dias = [0.5]
    for i, var_d in enumerate(var_dias):
        print(f"\n Run {i}/{len(var_dias)} with diameter variance {var_d}")

        # cqa = CqaSimulator()
        # cqa.par["var_diameter"] = var_d
        # cqa.par["track_full_dynamics"] = True
        # cqa.__init__(cqa.par)  # This is important!!
        cqa = CqaSimulator(par={
            "var_diameter": var_d,
            "track_full_dynamics": True,
            "track_quantified_dynamics": False,
            "record_final_flag": True,
        })

        files_exist, _ = cqa.check_files_exist(fp, spacing=500)

        cqa.run_until_convergence_over_positions(spacing=500)
        cqa.save_run_data(fp)
        cqa.check_files_exist(fp)
    cqa.save_full_model(fp)

    # # Run and visualize for specific position
    # # =======================================
    # cqa = CqaSimulator({
    #     "var_diameter": 1.2,
    #     "var_heights": 0.1,
    #     "N": 1000,
    #     "M": 4.4,
    #     # "M": 1.0,
    #     "correlated_peaks": False,
    #     "M_fixed": False,
    #     "seed": 1,
    # })
    # # cqa.par["var_diameter"] = 1.3
    # # cqa.par["N"] = 6000
    # # cqa.par["M"] = 4.4
    # # cqa.par["correlated_peaks"] = True
    # # cqa.par["M_fixed"] = False
    # # cqa.update_ctx_state()
    # print(cqa)
    # init_pos = 600
    # cqa.run_until_convergence(
    #     init_pos, p=0, record_final_flag=None, verbose=2,
    #     visualize=True,
    # )
