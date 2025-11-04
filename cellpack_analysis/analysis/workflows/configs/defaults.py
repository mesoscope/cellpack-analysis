STRUCTURE_ID = "SLC25A17"
"""Default structure ID."""

PACKING_ID = "peroxisome"
"""Default packing ID."""

STRUCTURE_NAME = "peroxisome"
"""Default structure name."""

PACKING_OUTPUT_FOLDER = "packing_outputs/8d_sphere_data/rules_shape/"
"""Default relative path to packing outputs."""

SAVE_FORMAT = "pdf"
"""Default save format for figures."""

BANDWIDTH = 0.2
"""Default bandwidth for KDE."""

DISTANCE_MEASURES = [
    "nearest",
    "pairwise",
    "nucleus",
    "z",
]
"""Default distance measures for distance analysis."""

"""Parameters for KS test analysis."""
KS_SIGNIFICANCE_LEVEL = 0.05
"""Default significance level for KS test."""
N_BOOTSTRAP = 1000
"""Default number of bootstrap samples for KS test."""

"""Parameters for occupancy analysis."""
OCCUPANCY_DISTANCE_MEASURES = [
    "nucleus",
    "z",
]
"""Default distance measures for occupancy analysis."""
OCCUPANCY_PARAMS = {
    "nucleus": {"xlim": 6, "ylim": 3, "bandwidth": 0.2},
    "z": {"xlim": 8, "ylim": 2, "bandwidth": 0.2},
    "fig_params": {"dpi": 300, "figsize": [3.5, 2.5]},
    "plot_individual": True,
    "show_legend": True,
}
"""Default occupancy analysis KDE parameters."""

RECALCULATE = {
    "load_common_data": False,
    "calculate_distances": False,
    "plot_distance_distributions": False,
    "run_emd_analysis": False,
    "run_ks_analysis": False,
    "run_occupancy_analysis": False,
}
"""Default recalculation flags for analysis workflow."""

NUM_WORKERS = 16
"""Default number of workers for parallel processing."""
