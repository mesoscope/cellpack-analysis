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

DISTRIBUTION_METHOD = "kde"
"""Default method for plotting distance distributions and occupancy.
Options: "discrete" (histogram-based) or "kde" (kernel density estimate).
Note: "kde" is the default; set to "discrete" in your config to use the histogram-based approach."""

FILTER_MINIMUM_DISTANCE: float | None = -1
"""Minimum distance passed to ``filter_invalids_from_distance_distribution_dict``.
Set to ``-1`` in configs that use KDE to allow small negative distances."""

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
    "nucleus": {"xlim": 6, "ylim": 3, "bandwidth": 0.2, "num_points": 250, "x_min": 0},
    "z": {"xlim": 8, "ylim": 2, "bandwidth": 0.2, "num_points": 250, "x_min": 0},
    "fig_params": {"dpi": 300, "figsize": [3.5, 2.5]},
    "plot_individual": False,
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
    "run_occupancy_ks_analysis": False,
    "run_rule_interpolation_cv": False,
}
"""Default recalculation flags for analysis workflow."""

RULE_INTERPOLATION_CV_PARAMS: dict = {
    "n_folds": 5,
    "n_repeats": 10,
    "random_state": None,
    "grouping": "combined",
    "generate_packing_configs": False,
    "packing_config_scope": "joint",
    "cv_use_slurm": True,
}
"""Default parameters for rule interpolation cross-validation."""

NUM_WORKERS = 8
"""Default number of workers for parallel processing."""

DISTANCE_PDF_PARAMS: dict = {
    "bandwidth": 0.2,
    "envelope_alpha": 0.05,
    "bin_width": 0.01,
}
"""Extra kwargs forwarded to ``distance.compute_distance_pdfs``."""

DISTANCE_PLOT_PARAMS: dict = {
    "plot_individual_curves": False,
    "figure_size": [2.7, 3.4],
    "envelope_alpha": 0.05,
    "production_mode": False,
    "overlay_mean_and_std": False,
}
"""Extra kwargs forwarded to ``visualization.plot_distance_distributions``."""

EMD_PLOT_PARAMS: dict = {
    "figure_size": [7, 7],
    "minimum_distance": -1,
}
"""Extra kwargs forwarded to ``visualization.plot_pairwise_emd_matrix``."""

ENVELOPE_PLOT_PARAMS: dict = {
    "per_dm_matrix_figsize": [7, 4],
    "per_dm_matrix_font_scale": 1.1,
    "joint_matrix_figsize": [7, 4],
    "joint_matrix_font_scale": 1.1,
    "rejection_bars_figsize": [4, 1.8],
    "rejection_bars_font_scale": 1.1,
    "overlaid_figsize": [3, 1.5],
}
"""Figsize / font-scale kwargs for envelope-test visualizations.
Shared by both ``_run_pairwise_envelope_test`` and
``_run_occupancy_pairwise_envelope_test``."""
