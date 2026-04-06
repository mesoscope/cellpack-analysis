# %% [markdown]
"""
# Rule interpolation workflow for punctate structures.

Fit the observed (baseline) occupancy ratio curve as a non-negative linear
combination of simulated packing-rule occupancy curves (NNLS), evaluate
fit quality with k-fold cross-validation, and optionally generate mixed-rule
packing configurations for held-out cells.

## Workflow steps:
1. Load positions and mesh information
2. Compute distance dictionaries and normalize → ``all_distance_dict``
3. Compute binned occupancy ratios directly from distance arrays
   → ``combined_binned_occupancy_dict``
4. Fit rule interpolation on all baseline cells → ``fit_result``
5. Plot fit overlay on occupancy ratio curves per distance measure
6. Run k-fold cross-validation → ``cv_result``
7. Plot CV MSE summary and coefficient stability across folds
8. (Optional) Generate mixed-rule packing configs for held-out cells
9. (Optional) Validate mixed-rule packings against observed data
"""

# %% [markdown]
import logging
import time

from IPython.display import display

from cellpack_analysis.analysis import rule_interpolation
from cellpack_analysis.lib import distance, occupancy, visualization
from cellpack_analysis.lib.file_io import get_project_root, make_dir
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

logger = logging.getLogger(__name__)

start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure IDs and packing configuration
# SLC25A17: peroxisomes
# RAB5A: early endosomes
# SEC61B: ER
# ST6GAL1: Golgi
STRUCTURE_ID = "SLC25A17"
"""ID for the packed punctate structure; used as packing mode for observed data."""

CELL_STRUCTURE_ID = "SLC25A17"
"""ID of the cell shapes used for packing (used for mesh lookup for simulated modes)."""

PACKING_ID = "peroxisome"
"""Packing configuration ID — used for naming outputs and folders."""

STRUCTURE_NAME = "peroxisome"
"""Name of the packed punctate structure in cellPACK output files."""

CONDITION = "rules_shape_with_seed"
"""Experimental condition / packing output subfolder."""

RESULT_SUBFOLDER = "interpolation_analysis/rules_shape_with_seed_test"
"""Subfolder within results/ to save outputs for this workflow."""
# %% [markdown]
# ### Set packing modes and channel map
save_format = "pdf"

baseline_mode = STRUCTURE_ID

channel_map = {
    STRUCTURE_ID: STRUCTURE_ID,
    "random": CELL_STRUCTURE_ID,
    "nucleus_gradient": CELL_STRUCTURE_ID,
    "membrane_gradient": CELL_STRUCTURE_ID,
    "apical_gradient": CELL_STRUCTURE_ID,
    # "struct_gradient": CELL_STRUCTURE_ID,
}

# relative path to packing outputs
packing_output_folder = f"packing_outputs/8d_sphere_data/{CONDITION}/"

packing_modes = list(channel_map.keys())
all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = make_dir(base_results_dir / RESULT_SUBFOLDER)

figures_dir = make_dir(results_dir / "figures")
# %% [markdown]
# ### Distance measures to use
# Options: "nucleus", "z", "scaled_nucleus", "membrane"
occupancy_distance_measures = [
    "nucleus",
    "z",
    # "scaled_nucleus",
    # "scaled_z",
]
# %% [markdown]
# ### Set normalization
# options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = None
suffix = ""
if normalization is not None:
    suffix = f"_normalized_{normalization}"
# %% [markdown]
# ### Set binning parameters per distance measure
bin_width_map: dict[str, float] = {
    "nucleus": 0.2,
    "z": 0.2,
}
occupancy_params: dict[str, dict] = {
    "nucleus": {"xlim": 6, "ylim": 3},
    "z": {"xlim": 8, "ylim": 2},
    # "scaled_nucleus": {"xlim": 1.0, "ylim": 3},
    # "scaled_z": {"xlim": 1.0, "ylim": 2},
}
# %% [markdown]
# ### Set rule interpolation parameters
n_folds = 5
"""Number of cross-validation folds."""

random_state = 42
"""Random seed for CV fold splitting (None = non-deterministic)."""

recalculate_cv = True
"""If True, recompute CV even when a cached result exists."""

generate_configs = True
"""Set to True to generate mixed-rule packing configs after CV."""

mode_to_gradient_name: dict[str, str] = {
    "random": "uniform",
    "nucleus_gradient": "nucleus_gradient",
    "membrane_gradient": "membrane_gradient",
    "apical_gradient": "apical_gradient",
}
"""Maps packing mode names to gradient field names in the cellPACK recipe,
e.g. ``{"random": "uniform"}``.
Required when ``generate_configs=True``."""

base_packing_config_path = (
    project_root / "cellpack_analysis/packing/configs/rules_shape_with_seed/peroxisome.json"
)
"""Path to the base cellPACK config used as template for config generation."""
# %% [markdown]
# ## Data loading
# %% [markdown]
# ### Read position data from outputs
all_positions = get_position_data_from_outputs(
    structure_id=STRUCTURE_ID,
    packing_id=PACKING_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
    ingredient_key=f"membrane_interior_{STRUCTURE_NAME}",
    recalculate=False,
)
# %% [markdown]
# ### Get mesh information
combined_mesh_information_dict: dict = {}
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    combined_mesh_information_dict[structure_id] = mesh_information_dict
# %% [markdown]
# ## Distance pipeline
# %% [markdown]
# ### Calculate distance measures
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=occupancy_distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
    num_workers=8,
)
# %% [markdown]
# ### Filter invalid distances
all_distance_dict_filtered = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict,
    minimum_distance=0,
)
# %% [markdown]
# ### Normalize distances
all_distance_dict_normalized = distance.normalize_distance_dictionary(
    all_distance_dict=all_distance_dict_filtered,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)
# %% [markdown]
# ## Build discrete occupancy dictionaries
# %% [markdown]
# ### Compute binned occupancy ratio for each distance measure
combined_binned_occupancy_dict: dict = {}
for dm in occupancy_distance_measures:
    logger.info("Computing binned occupancy for distance measure: %s", dm)
    combined_binned_occupancy_dict[dm] = occupancy.get_binned_occupancy_dict_from_distance_dict(
        all_distance_dict=all_distance_dict_normalized,
        combined_mesh_information_dict=combined_mesh_information_dict,
        channel_map=channel_map,
        distance_measure=dm,
        bin_width=bin_width_map.get(dm, 0.4),
        x_min=0.0,
        results_dir=results_dir,
        pseudocount=1e-10,
        min_count=5,
        recalculate=False,
        suffix=suffix,
    )
# %% [markdown]
# ### Visualize occupancy curves
for dm in occupancy_distance_measures:
    occ_fig = visualization.plot_occupancy_ratio(
        occupancy_dict=combined_binned_occupancy_dict[dm],
        channel_map=channel_map,
        figures_dir=figures_dir,
        normalization=normalization,
        suffix=suffix,
        distance_measure=dm,
        xlim=occupancy_params[dm]["xlim"],
        ylim=occupancy_params[dm]["ylim"],
        save_format=save_format,
        fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
    )
    display(occ_fig)
# %% [markdown]
# ## Full-data rule interpolation fit
# %% [markdown]
# ### Fit NNLS interpolation on all baseline cells
fit_result = rule_interpolation.fit_rule_interpolation(
    occupancy_dict=combined_binned_occupancy_dict,
    channel_map=channel_map,
    baseline_mode=baseline_mode,
    distance_measures=occupancy_distance_measures,
)
# %% [markdown]
# ### Plot fit overlay per distance measure
for dm in occupancy_distance_measures:
    interp_figures_dir = figures_dir / dm
    interp_figures_dir.mkdir(exist_ok=True, parents=True)
    for plot_type in ("individual", "joint"):
        fig_fit, ax_fit = visualization.plot_rule_interpolation_fit(
            fit_result=fit_result,
            occupancy_dict=combined_binned_occupancy_dict,
            channel_map=channel_map,
            baseline_mode=baseline_mode,
            distance_measure=dm,
            plot_type=plot_type,
            figures_dir=interp_figures_dir,
            xlim=occupancy_params[dm]["xlim"],
            ylim=occupancy_params[dm]["ylim"],
            suffix=suffix,
            save_format=save_format,
        )
        # display(fig_fit)
# %% [markdown]
# ### Log full-data fit coefficients
rule_interpolation.log_rule_interpolation_coeffs(
    fit_result=fit_result,
    baseline_mode=baseline_mode,
    file_path=results_dir / f"{STRUCTURE_NAME}_rule_interpolation_coefficients{suffix}.log",
)
# %% [markdown]
# ## Cross-validation
# %% [markdown]
# ### Run k-fold cross-validation on baseline cell IDs
cv_result = rule_interpolation.run_rule_interpolation_cv(
    occupancy_dict=combined_binned_occupancy_dict,
    channel_map=channel_map,
    baseline_mode=baseline_mode,
    n_folds=n_folds,
    random_state=random_state,
    distance_measures=occupancy_distance_measures,
    results_dir=results_dir,
    recalculate=recalculate_cv,
    suffix=suffix,
    grouping="combined",
)
# %% [markdown]
# ### Tabulate CV results
cv_df = rule_interpolation.summarize_cv_results(cv_result)
display(cv_df.groupby(["scope", "distance_measure", "split"])["mse"].describe())
# %% [markdown]
# ### Plot CV MSE summary (train vs. test per distance measure)
cv_figures_dir = figures_dir / "cross_validation"
cv_figures_dir.mkdir(exist_ok=True, parents=True)

fig_mse, axs_mse = visualization.plot_cv_mse_summary(
    cv_df=cv_df,
    figures_dir=cv_figures_dir,
    suffix=suffix,
    save_format=save_format,
)
display(fig_mse)
# %% [markdown]
# ### Plot coefficient stability across CV folds
fig_coef, axs_coef = visualization.plot_cv_coefficient_stability(
    cv_result=cv_result,
    figures_dir=cv_figures_dir,
    suffix=suffix,
    save_format=save_format,
)
display(fig_coef)
# %% [markdown]
# ### Log CV summary
rule_interpolation.log_cv_summary(
    cv_result=cv_result,
    file_path=results_dir / f"{STRUCTURE_NAME}_rule_interpolation_cv_summary{suffix}.log",
)
# %% [markdown]
# ## (Optional) Generate mixed-rule packing configurations
# Set ``generate_configs = True`` and provide ``base_packing_config_path``
# and ``mode_to_gradient_name`` in the parameters section above to activate.

if generate_configs:
    if base_packing_config_path is None:
        raise ValueError("Set base_packing_config_path to use generate_mixed_rule_packing_configs.")
    output_config_dir = results_dir / "mixed_rule_configs"
    config_paths = rule_interpolation.generate_mixed_rule_packing_configs(
        cv_result=cv_result,
        base_config_path=base_packing_config_path,
        output_config_dir=output_config_dir,
        mode_to_gradient_name=mode_to_gradient_name,
        scope="joint",
        dry_run=True,
    )
    logger.info(
        "Generated %d mixed-rule packing configs in %s", len(config_paths), output_config_dir
    )
# %% [markdown]
# ## (Optional) Validate mixed-rule packings
# Runs automatically when a ``mixed_rule_packings/`` subdirectory exists inside
# ``results_dir`` (populated after running the packing workflow on generated configs).
mixed_rule_results_dir = results_dir / "mixed_rule_packings"
if mixed_rule_results_dir.exists():
    validation_result = rule_interpolation.run_mixed_rule_validation(
        combined_occupancy_dict=combined_binned_occupancy_dict,
        channel_map=channel_map,
        baseline_mode=baseline_mode,
        packing_modes=packing_modes,
        distance_measures=occupancy_distance_measures,
        results_dir=results_dir,
        recalculate=False,
    )
    display(validation_result.emd_df)
    display(validation_result.ks_df)
# %%
logger.info("Time taken to complete workflow: %.2f s", time.time() - start_time)
