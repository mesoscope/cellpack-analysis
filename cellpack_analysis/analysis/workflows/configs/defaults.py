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

MINIMUM_DISTANCE_UM: float | None = -0.108
"""Floor in **microns** for observed and simulated distances (round 1: -1 px).
``None`` drops only NaN/inf."""

AVAILABLE_MINIMUM_DISTANCE_UM: float | None = -1.0
"""Floor in **microns** for available-space distances (occupancy denominator; round 1)."""

DISTANCE_MEASURES = [
    "nearest",
    "pairwise",
    "nucleus",
    "z",
]
"""Default distance measures for distance analysis."""

EMD_POOL_SEEDS = False
"""Pool each cell's seeds into one curve before the pairwise EMD.

The pairwise EMD is quadratic in the number of curves, so a replicate dataset
passed per-seed is infeasible: ``replicate_99_v2`` gives 7.3e9 pairs for
peroxisome and 1.5e10 for endosome. Pooling drops this to ~1.2e6, matching the
granularity the occupancy EMD already uses. Off by default so single-seed
analyses are unchanged."""

EMD_SAME_CELL = False
"""Pair each cell only with itself in cross-mode EMD comparisons.

Off by default so population-level comparisons (e.g. variation sources) stay pooled."""

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
OCCUPANCY_NUM_POINTS = 250
"""Number of points on the common occupancy grid.

Used as the fallback whenever a config's ``occupancy_params[dm]`` omits
``num_points``, which is the common case: the merge in
``AnalysisConfig._setup_occupancy_params`` is shallow, so a config that overrides
any key of an inner dict replaces the whole dict.
"""

OCCUPANCY_PARAMS = {
    "nucleus": {
        "xlim": 6,
        "ylim": 3,
        "bandwidth": 0.2,
        "num_points": OCCUPANCY_NUM_POINTS,
        "x_min": 0,
    },
    "z": {
        "xlim": 8,
        "ylim": 2,
        "bandwidth": 0.2,
        "num_points": OCCUPANCY_NUM_POINTS,
        "x_min": 0,
    },
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

DISTANCE_BACKEND = "thread"
"""Default executor backend for the per-cell distance calculation.
Options: "thread" (shares meshes in-memory, GIL-bound) or "process" (bypasses the
GIL via separate processes; faster for many cells at the cost of pickling meshes)."""

OCCUPANCY_BACKEND = "thread"
"""Default executor backend for the per-cell KDE occupancy calculation.
Options: "thread" (shares the KDE dict in-memory) or "process" (bypasses the GIL;
only this cell's mode + available KDEs are pickled per task, so the payload stays
small). Defaults to "thread" so existing configs are unaffected."""

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
    "minimum_distance": None,
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
