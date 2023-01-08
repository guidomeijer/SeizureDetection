# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:09:46 2022 by Guido Meijer
"""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seizure_functions import paths

N_PLOTS = 250

# Get paths
path_dict = paths()

# Load in training labels
train_labels = pd.read_csv(join(path_dict["data_path"], "train", "train_labels.csv"))

# Select first X data snippets to plot
seizure_ind = train_labels.loc[train_labels["label"] == 1, "filepath"].index[:N_PLOTS]

# Positive training data samples
for i, filepath in enumerate(train_labels.loc[seizure_ind, "filepath"]):

    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict["data_path"], "train", filepath))

    # Plot data
    time_ax = np.array(data_snippet["utc_timestamp"] - data_snippet.loc[0, "utc_timestamp"]) / 60
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5))

    ax1.plot(time_ax, data_snippet["acc_mag"])
    ax1.set(ylabel="Accelerometer mag.", xlabel="Time (min)")

    ax2.plot(time_ax, data_snippet["bvp"])
    ax2.set(ylabel="Blood volume pulse", xlabel="Time (min)")

    ax3.plot(time_ax, data_snippet["eda"])
    ax3.set(ylabel="Electrodermal act.", xlabel="Time (min)")

    ax4.plot(time_ax, data_snippet["hr"])
    ax4.set(ylabel="Heart rate", xlabel="Time (min)")

    ax5.plot(time_ax, data_snippet["temp"])
    ax5.set(ylabel="Temperature", xlabel="Time (min)")

    ax6.axis("off")

    f.suptitle(f"{filepath}")
    plt.tight_layout()
    plt.savefig(
        join(
            path_dict["fig_path"],
            "Raw",
            "Seizure",
            filepath[:-8].replace("/", "_") + ".jpg",
        ),
        dpi=300,
    )
    plt.close(f)

# Control training data samples
control_ind = np.random.choice(
    train_labels[train_labels["label"] == 0].index, N_PLOTS, replace=False
)

for i, filepath in enumerate(train_labels.loc[control_ind, "filepath"]):

    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict["data_path"], "train", filepath))

    # Plot data
    time_ax = np.array(data_snippet["utc_timestamp"] - data_snippet.loc[0, "utc_timestamp"]) / 60
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5), dpi=200)

    ax1.plot(time_ax, data_snippet["acc_mag"])
    ax1.set(ylabel="Accelerometer mag.", xlabel="Time (min)")

    ax2.plot(time_ax, data_snippet["bvp"])
    ax2.set(ylabel="Blood volume pulse", xlabel="Time (min)")

    ax3.plot(time_ax, data_snippet["eda"])
    ax3.set(ylabel="Electrodermal act.", xlabel="Time (min)")

    ax4.plot(time_ax, data_snippet["hr"])
    ax4.set(ylabel="Heart rate", xlabel="Time (min)")

    ax5.plot(time_ax, data_snippet["temp"])
    ax5.set(ylabel="Temperature", xlabel="Time (min)")

    f.suptitle(f"{filepath}")
    plt.tight_layout()
    plt.savefig(
        join(
            path_dict["fig_path"],
            "Raw",
            "Control",
            filepath[:-8].replace("/", "_") + ".jpg",
        )
    )
    plt.close(f)
