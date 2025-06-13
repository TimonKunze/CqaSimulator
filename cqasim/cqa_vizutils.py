#!/usr/bin/env python3
"""Visualization function for the Cqa Simulator."""
import numpy as np
import matplotlib.pyplot as plt

from cqasim.cqa_vectutils import calc_hessian, calc_sparsity, \
    calc_support, cosine_similarity, one_step_dynamics, relu, \
    cosine_similarity_vec, std_cdm


def animate_1d_activity(V_data, o_data, data, ymin=-1, ymax=1, pause=0.2):
    """Animate 1D data over time steps."""
    assert len(np.shape(V_data)) == 2, \
        f"Data is not 2D but of shape {np.shape(V_data)}."

    o_max_dt = [[np.argmax(o_curr), np.max(o_curr)] for o_curr in o_data]

    o_disp_dt = [std_cdm(o_curr) for o_curr in o_data]
    o_disp_clip_dt = [std_cdm(np.maximum(o_curr - 0.2, 0)) for o_curr in o_data] # Love from Ani

    x_0 = np.arange(len(o_data[1]))
    x_1 = np.arange(len(V_data[1]))

    fig, axs = plt.subplots(2, 1, figsize=(13, 6))

    point_0 = axs[0].scatter(o_max_dt[0][0], o_max_dt[0][1],
                             color='black', marker='.', s=100)
    line_0, = axs[0].plot(x_0, o_data[0], 'r-')
    axs[1].plot(x_1, data*ymax/10, color='gray', alpha=0.5)  # initialization
    line_1, = axs[1].plot(x_1, V_data[0], 'b-')

    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(ymin, ymax)
    axs[0].set_xlabel("Position")
    axs[1].set_xlabel("Neuron")
    axs[0].set_ylabel("Overlap")
    axs[1].set_ylabel("Activation")

    text_disp = axs[0].text(
        0.95, 0.95,                 # X, Y in axes coordinates (0 to 1)
        # np.round(o_disp_dt[0], 3),
        f"Disp={o_disp_dt[0]:.2f}",           # The text
        transform=axs[0].transAxes,     # Use axes coords, not data coords
        ha="right", va="top",       # Align text to the top-right corner
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Optional styling
    )
    text_disp_clip = axs[0].text(
        0.95, 0.85,                 # X, Y in axes coordinates (0 to 1)
        f"DispC={o_disp_clip_dt[0]:.2f}",           # The text
        transform=axs[0].transAxes,     # Use axes coords, not data coords
        ha="right", va="top",       # Align text to the top-right corner
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Optional styling
    )

    fig.tight_layout()

    i = 0
    while True:
        # Update the line
        line_0.set_data(x_0, o_data[i])
        point_0.set_offsets(o_max_dt[i])
        line_1.set_data(x_1, V_data[i])
        text_disp.set_text(f"Disp={o_disp_dt[0]:.2f}")
        text_disp_clip.set_text(f"DispC={o_disp_clip_dt[0]:.2f}")
        fig.suptitle(f"Timestep: {i}/{len(V_data)}")
        plt.pause(pause)  # Pause to visualize

        i = (i + 1) % len(V_data)  # Loop index

    plt.close()


def animate_2d_activity(data, ymin=-1, ymax=1, pause=0.2):
    # TODO: Implement
    pass
