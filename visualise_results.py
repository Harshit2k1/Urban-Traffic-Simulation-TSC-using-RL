#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
visualise_results.py
Reads comparison_plots/per_step_results.npz produced by
run_all.py and creates two figures:
Average waiting time per step  (mean ±95% CI)
Total waiting time per step    (mean ±95% CI)

PNG files are written to comparison_plots/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------
RESULT_FILE = "comparison_plots/per_step_results.npz"
os.makedirs("comparison_plots", exist_ok=True)

# ---------------------------------------------------------------------
#  LOAD
# ---------------------------------------------------------------------
data = np.load(RESULT_FILE, allow_pickle=True)
algos = ["Baseline", "Webster", "Max-Pressure", "RL"]
colours = ["k", "tab:blue", "tab:orange", "tab:green"]

avg_arrays = [
    data["baseline_avg"],
    data["webster_avg"],
    data["mp_avg"],
    data["rl_avg"],
]
total_arrays = [
    data["baseline_total"],
    data["webster_total"],
    data["mp_total"],
    data["rl_total"],
]
steps_axis = data["steps"]
num_runs = avg_arrays[0].shape[0]

# ---------------------------------------------------------------------
#  STATS HELPERS
# ---------------------------------------------------------------------
def mean_ci(arr, conf=0.95):
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    sem = stats.sem(arr, axis=0)
    t_val = stats.t.ppf((1 + conf) / 2.0, n - 1)
    hw = t_val * sem
    return mean, hw


# ---------------------------------------------------------------------
#  FIGURE 1  – Average wait / step  (running vehicles)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 4))
for a, lab, col in zip(avg_arrays, algos, colours):
    m, hw = mean_ci(a)
    plt.plot(steps_axis, m, label=lab, color=col)
    plt.fill_between(steps_axis, m - hw, m + hw, alpha=0.2, color=col)
plt.xlabel("Simulation step")
plt.ylabel("Average waiting time of running vehicles [s]")
plt.title(
    f"(Mean ±95 % CI, No. of Runs = {num_runs})"
)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("comparison_plots/avg_wait_vs_step.png", dpi=300)

# ---------------------------------------------------------------------
#  FIGURE 2  – Total wait / step  (running vehicles)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 4))
for a, lab, col in zip(total_arrays, algos, colours):
    m, hw = mean_ci(a)
    plt.plot(steps_axis, m, label=lab, color=col)
    plt.fill_between(steps_axis, m - hw, m + hw, alpha=0.2, color=col)
plt.xlabel("Simulation step")
plt.ylabel("Total waiting time of running vehicles [s]")
plt.title(
    f"(Mean ±95 % CI, No. of Runs = {num_runs})"
)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("comparison_plots/total_wait_vs_step.png", dpi=300)

print("Figures written to comparison_plots/ with updated labels.")

