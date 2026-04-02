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

DISTRIBUTION_METHOD = "discrete"
"""Default method for plotting distance distributions and occupancy.
Options: "discrete" (histogram-based) or "kde" (kernel density estimate).
Note: "discrete" is the default; set to "kde" in your config to use the old KDE behaviour."""

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
"""Default occupancy analysis parameters (used for both KDE and discrete methods)."""

BIN_WIDTH_MAP = {
    "nucleus": 0.2,
    "z": 0.2,
    "nearest": 0.2,
    "pairwise": 0.2,
    "membrane": 0.2,
    "scaled_nucleus": 0.05,
    "scaled_z": 0.05,
}
"""Default histogram bin width per distance measure (used in discrete mode)."""

DISCRETE_OCCUPANCY_PARAMS = {
    "pseudocount": 1e-10,
    "min_count": 5,
    "x_min": 0.0,
}
"""Extra parameters specific to discrete (histogram-based) occupancy."""

ENVELOPE_TEST_PARAMS = {
    "alpha": 0.05,
    "r_grid_size": 150,
    "bin_width": 0.2,
    "statistic": "intdev",
}
"""Default parameters for pairwise Monte Carlo envelope tests."""

RECALCULATE = {
    "load_common_data": False,
    "calculate_distances": False,
    "plot_distance_distributions": False,
    "run_emd_analysis": False,
    "run_ks_analysis": False,
    "run_pairwise_envelope_test": False,
    "run_occupancy_analysis": False,
    "run_occupancy_emd_analysis": False,
    "run_occupancy_pairwise_envelope_test": False,
    "run_occupancy_interpolation_analysis": False,
    "run_rule_interpolation_cv": False,
}
"""Default recalculation flags for analysis workflow."""

RULE_INTERPOLATION_CV_PARAMS: dict = {
    "n_folds": 5,
    "random_state": None,
    "generate_packing_configs": False,
    "packing_config_scope": "joint",
    "cv_use_slurm": True,
}
"""Default parameters for rule interpolation cross-validation."""

NUM_WORKERS = 16
"""Default number of workers for parallel processing."""
