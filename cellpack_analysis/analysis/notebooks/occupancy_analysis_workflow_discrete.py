# %% [markdown]
"""
# Occupancy workflow for punctate structures (discrete / histogram-based).

Calculate and plot the occupancy ratio for different observed and simulated data
using *discrete histogram* methods

# ## Workflow steps:
1. Load positions and mesh information
2. Compute distance dictionaries and normalize → ``all_distance_dict``
3. Compute binned occupancy ratios directly from distance arrays
   → ``combined_binned_occupancy_dict``
4. Plot occupancy illustration (histogram overlay + ratio curve for one cell)
5. Plot occupancy ratio: mean (thick line) + 95 % pointwise envelope
6. Occupancy EMD: bar/violin comparisons + pairwise EMD matrix
7. Pairwise envelope test on occupancy ratio curves
8. Pairwise KS test on occupancy distributions
"""

# %% [markdown]
import logging
import time

import pandas as pd
from IPython.display import display

from cellpack_analysis.lib import distance, occupancy, visualization
from cellpack_analysis.lib.file_io import get_project_root
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

RESULT_SUBFOLDER = "occupancy_discrete_test_rules_shape_with_seed"
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

results_dir = base_results_dir / RESULT_SUBFOLDER / PACKING_ID
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
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
        # bin_width=0.5,
        x_min=0.0,
        # x_max=occupancy_params[dm]["xlim"],
        results_dir=results_dir,
        pseudocount=1e-10,
        min_count=5,
        recalculate=False,
        suffix=suffix,
    )
# %% [markdown]
# ### Plot discrete occupancy illustration for one example cell
for dm in occupancy_distance_measures:
    occupancy_figures_dir = figures_dir / dm
    occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
    fig_ill, axs_ill = visualization.plot_occupancy_illustration_discrete(
        binned_occupancy_dict=combined_binned_occupancy_dict[dm],
        packing_mode="SLC25A17",
        # cell_id_or_index=5,
        figures_dir=occupancy_figures_dir,
        suffix=suffix,
        distance_measure=dm,
        normalization=normalization,
        # xlim=occupancy_params[dm]["xlim"],
        # ylim_ratio=occupancy_params[dm]["ylim"],
        save_format=save_format,
    )

# %% [markdown]
# ### Plot occupancy ratio (mean + 95 % pointwise envelope)
for dm in occupancy_distance_measures:
    occupancy_figures_dir = figures_dir / dm
    occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = visualization.plot_binned_occupancy_ratio(
        binned_occupancy_dict=combined_binned_occupancy_dict[dm],
        channel_map=channel_map,
        figures_dir=occupancy_figures_dir,
        normalization=normalization,
        suffix=suffix,
        distance_measure=dm,
        xlim=occupancy_params[dm]["xlim"],
        ylim=occupancy_params[dm]["ylim"],
        save_format=save_format,
        envelope_alpha=0.05,
        fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
    )
# %% [markdown]
# ## Occupancy EMD analysis
# %% [markdown]
# ### Create EMD output folder
occ_emd_figures_dir = figures_dir / "occupancy_emd"
occ_emd_figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Calculate pairwise occupancy EMD
occupancy_emd_df = occupancy.get_occupancy_emd_df(
    combined_occupancy_dict=combined_binned_occupancy_dict,
    packing_modes=packing_modes,
    distance_measures=occupancy_distance_measures,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ### Plot occupancy EMD comparisons (baseline)
fig_v_emd, ax_v_emd, fig_b_emd, ax_b_emd = visualization.plot_emd_comparisons(
    df_emd=occupancy_emd_df,
    distance_measures=occupancy_distance_measures,
    comparison_type="baseline",
    baseline_mode=baseline_mode,
    suffix=suffix,
    figures_dir=occ_emd_figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ### Plot pairwise occupancy EMD matrix per distance measure
for dm in occupancy_distance_measures:
    occupancy_figures_dir = figures_dir / dm
    occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
    fig_emd_mat, axs_emd_mat = visualization.plot_pairwise_occupancy_emd_matrix(
        df_emd=occupancy_emd_df,
        binned_occupancy_dict=combined_binned_occupancy_dict[dm],
        packing_modes=packing_modes,
        distance_measure=dm,
        normalization=normalization,
        xlim=occupancy_params[dm]["xlim"],
        ylim=occupancy_params[dm]["ylim"],
        figures_dir=occupancy_figures_dir,
        suffix=suffix,
        save_format=save_format,
    )
    display(fig_emd_mat)
# %% [markdown]
# ## Pairwise envelope test on occupancy ratios
# %% [markdown]
# ### Run pairwise envelope test on occupancy ratio curves
occ_pairwise_results = occupancy.pairwise_envelope_test_occupancy(
    combined_binned_occupancy_dict=combined_binned_occupancy_dict,
    packing_modes=packing_modes,
    alpha=0.05,
    statistic="intdev",
    # comparison_type="ecdf",
)
# %% [markdown]
# ### Plot pairwise occupancy envelope matrix per distance measure
envelope_figures_dir = figures_dir / "pairwise_envelope"
envelope_figures_dir.mkdir(exist_ok=True, parents=True)

for dm in occupancy_distance_measures:
    fig_env, axs_env = visualization.plot_pairwise_envelope_matrix(
        pairwise_results=occ_pairwise_results,
        distance_measure=dm,
        figures_dir=envelope_figures_dir,
        suffix=suffix,
        save_format=save_format,
    )
    display(fig_env)
# %% [markdown]
# ### Plot pairwise occupancy envelope matrix — joint test
fig_env_joint, axs_env_joint = visualization.plot_pairwise_envelope_matrix(
    pairwise_results=occ_pairwise_results,
    distance_measure=None,
    figures_dir=envelope_figures_dir,
    figsize=(5, 3),
    suffix=suffix,
    save_format=save_format,
)
display(fig_env_joint)
# %% [markdown]
# ## Pairwise KS test analysis on occupancy distributions
# Compare each mode against every other mode (pairwise, not just vs. baseline).
# %% [markdown]
# ### Run pairwise KS tests across all mode pairs
ks_significance_level = 0.05
pairwise_ks_figures_dir = figures_dir / "pairwise_ks_test"
pairwise_ks_figures_dir.mkdir(exist_ok=True, parents=True)

pairwise_occ_ks_dfs: list[pd.DataFrame] = []
for ref_mode in packing_modes:
    occ_ks_df = occupancy.get_occupancy_ks_test_df(
        distance_measures=occupancy_distance_measures,
        packing_modes=packing_modes,
        combined_occupancy_dict=combined_binned_occupancy_dict,
        baseline_mode=ref_mode,
        significance_level=ks_significance_level,
        save_dir=None,
        recalculate=False,
    )
    occ_ks_df["baseline_mode"] = ref_mode
    pairwise_occ_ks_dfs.append(occ_ks_df)

pairwise_occ_ks_test_df = pd.concat(pairwise_occ_ks_dfs, ignore_index=True)
# %% [markdown]
# ### Bootstrap pairwise KS tests
pairwise_occ_ks_bootstrap_dfs: list[pd.DataFrame] = []
for ref_mode in packing_modes:
    ref_ks_df = pairwise_occ_ks_test_df.query("baseline_mode == @ref_mode")
    other_modes = [m for m in packing_modes if m != ref_mode]
    df_boot = distance.bootstrap_ks_tests(
        ks_test_df=ref_ks_df,
        distance_measures=occupancy_distance_measures,
        packing_modes=other_modes,
        n_bootstrap=1000,
    )
    df_boot["baseline_mode"] = ref_mode
    pairwise_occ_ks_bootstrap_dfs.append(df_boot)

pairwise_occ_ks_bootstrap_df = pd.concat(pairwise_occ_ks_bootstrap_dfs, ignore_index=True)
# %% [markdown]
# ### Plot pairwise KS results per baseline mode
for ref_mode in packing_modes:
    ref_boot_df = pairwise_occ_ks_bootstrap_df.query("baseline_mode == @ref_mode")
    fig_list, ax_list = visualization.plot_ks_test_results(
        df_ks_bootstrap=ref_boot_df,
        distance_measures=occupancy_distance_measures,
        figures_dir=pairwise_ks_figures_dir,
        suffix=f"{suffix}_vs_{ref_mode}",
        save_format=save_format,
    )
# %% [markdown]
# ### Log pairwise KS central tendencies
pairwise_ks_log_file_path = (
    results_dir / f"{STRUCTURE_NAME}_pairwise_occupancy_ks_central_tendencies{suffix}.log"
)
for _ref_mode in packing_modes:
    ref_boot_df = pairwise_occ_ks_bootstrap_df.query("baseline_mode == @_ref_mode")
    distance.log_central_tendencies_for_ks(
        df_ks_bootstrap=ref_boot_df,
        distance_measures=occupancy_distance_measures,
        file_path=pairwise_ks_log_file_path,
    )
# %%
logger.info("Time taken to complete workflow: %.2f s", time.time() - start_time)
# %%
