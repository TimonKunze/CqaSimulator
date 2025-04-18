#!/usr/bin/env python3
"""Visualization function for the Cqa Simulator."""
import numpy as np
import matplotlib.pyplot as plt


def animate_1d_activity(V_data, o_data, data, ymin=-1, ymax=1, pause=0.2):
    """Animate 1D data over time steps."""
    assert len(np.shape(V_data)) == 2, \
        f"Data is not 2D but of shape {np.shape(V_data)}."

    o_max_dt = [[np.argmax(o_curr), np.max(o_curr)] for o_curr in o_data]
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

    fig.tight_layout()

    i = 0
    while True:
        # Update the line
        line_0.set_data(x_0, o_data[i])
        point_0.set_offsets(o_max_dt[i])
        line_1.set_data(x_1, V_data[i])
        fig.suptitle(f"Timestep: {i}/{len(V_data)}")
        plt.pause(pause)  # Pause to visualize

        i = (i + 1) % len(V_data)  # Loop index

    plt.close()


def animate_2d_activity(data, ymin=-1, ymax=1, pause=0.2):
    # TODO: Implement
    pass
