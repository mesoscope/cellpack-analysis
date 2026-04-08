# %% [markdown]
"""
# Occupancy workflow for punctate structures (KDE-based).

Calculate and plot the occupancy ratio for different observed and simulated data
using *KDE* methods

## Workflow steps:
1. Load positions and mesh information
2. Compute distance dictionaries and normalize → ``all_distance_dict``
3. Compute occupancy ratio KDE dictionary directly from distance arrays
   → ``combined_kde_occupancy_dict``
4. Plot occupancy illustration (KDE occupied + available + ratio curve for one cell)
5. Plot occupancy ratio: mean (thick line) + 95 % pointwise envelope
6. Occupancy EMD: bar/violin comparisons + pairwise EMD matrix
7. Pairwise envelope test on occupancy ratio curves
8. Pairwise KS test on occupancy distributions
"""

# %% [markdown]
import logging
import time

import pandas as pd

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

RESULT_SUBFOLDER = "occupancy_analysis/rules_shape_with_seed"
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

figures_dir = make_dir(results_dir / "figures/")

log_dir = make_dir(results_dir / "logs/")
# %% [markdown]
# ### Distance measures to use
# Options: "nucleus", "z", "scaled_nucleus", "membrane"
occupancy_distance_measures = [
    "nucleus",
    "z",
]
# %% [markdown]
# ### Set normalization
# options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = None
suffix = ""
if normalization is not None:
    suffix = f"_normalized_{normalization}"

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
    recalculate=True,
)
# %% [markdown]
# ### Get mesh information
combined_mesh_information_dict = {}
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=True,
    )
    combined_mesh_information_dict[structure_id] = mesh_information_dict
# %% [markdown]
# ## Distance analysis
# %% [markdown]
# ### Calculate distance measures and normalize
all_distance_dict_raw = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=occupancy_distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=True,
    num_workers=8,
)
all_distance_dict_filtered = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict_raw,
    minimum_distance=-1,
)
all_distance_dict = distance.normalize_distance_dictionary(
    all_distance_dict=all_distance_dict_filtered,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)
# %% [markdown]
# ## Occupancy Analysis
# %% [markdown]
# ### Set limits and bandwidths for plotting
occupancy_params = {
    "nucleus": {"xlim": 6, "ylim": 3.2, "bandwidth": 0.2},
    "z": {"xlim": 8, "ylim": 2, "bandwidth": 0.2},
}
# %% [markdown]
# ### Create kde dictionary for individual distance distributions
# occupancy_distance_measure = "nucleus"  # for testing
occupancy_axes_dict = {}
minimum_distance = -1  # allowance for small negative distances
occupancy_figures_dir = {}
for dm in occupancy_distance_measures:
    logger.info(f"Starting occupancy analysis for distance measure: {dm}")
    occupancy_figures_dir[dm] = make_dir(figures_dir / dm)
# %% [markdown]
# ### Calculate distance kde dictionary
distance_kde_dict = {}
for dm in occupancy_distance_measures:
    distance_kde_dict[dm] = distance.get_distance_distribution_kde(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=combined_mesh_information_dict,
        channel_map=channel_map,
        save_dir=results_dir,
        recalculate=True,
        suffix=suffix,
        normalization=normalization,
        distance_measure=dm,
        minimum_distance=minimum_distance,
    )

# %% [markdown]

# %% [markdown]
# ### Calculate occupancy ratio
combined_occupancy_dict = {}
for dm in occupancy_distance_measures:
    combined_occupancy_dict[dm] = occupancy.get_kde_occupancy_dict(
        distance_kde_dict=distance_kde_dict[dm],
        channel_map=channel_map,
        results_dir=results_dir,
        recalculate=True,
        suffix=suffix,
        distance_measure=dm,
        bandwidth=occupancy_params[dm]["bandwidth"],
        # bandwidth="scott",
        # num_cells=5,
        num_points=250,
        x_min=0,
        x_max=occupancy_params[dm]["xlim"],
        num_workers=8,
    )

# %% [markdown]
# ### Plot illustration for one example cell
for dm in occupancy_distance_measures:
    fig_ill, axs_ill = visualization.plot_occupancy_illustration(
        distance_kde_dict=distance_kde_dict[dm],
        packing_mode="random",
        figures_dir=occupancy_figures_dir[dm],
        suffix=suffix,
        distance_measure=dm,
        normalization=normalization,
        cell_id_or_index=25,
        num_points=250,
        bandwidth=0.4,
        save_format=save_format,
        xlim=occupancy_params[dm]["xlim"],
    )

# %% [markdown]
# ### Plot occupancy ratio
for dm in occupancy_distance_measures:
    fig, ax = visualization.plot_occupancy_ratio(
        occupancy_dict=combined_occupancy_dict[dm],
        channel_map=channel_map,
        baseline_mode=baseline_mode,
        figures_dir=occupancy_figures_dir[dm],
        suffix=suffix,
        normalization=normalization,
        distance_measure=dm,
        save_format=save_format,
        xlim=occupancy_params[dm]["xlim"],
        ylim=occupancy_params[dm]["ylim"],
        fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
    )

# %% [markdown]
# ## Calculate occupancy EMD
# TODO: use combined grid evaluated occupancy
occupancy_emd_df = occupancy.get_occupancy_emd_df(
    combined_occupancy_dict=combined_occupancy_dict,
    packing_modes=packing_modes,
    distance_measures=occupancy_distance_measures,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% [markdown]
# ## Plot occupancy EMD
emd_figures_dir = make_dir(figures_dir / "emd")
fig_bar, axs_bar, fig_violin, axs_violin = visualization.plot_emd_comparisons(
    df_emd=occupancy_emd_df,
    distance_measures=occupancy_distance_measures,
    comparison_type="baseline",
    baseline_mode=baseline_mode,
    figures_dir=emd_figures_dir,
    suffix=suffix,
    save_format=save_format,
    annotate_significance=False,
)
# %% [markdown]
# ### Plot pairwise occupancy EMD matrix per distance measure
for dm in occupancy_distance_measures:
    occupancy_figures_dir = make_dir(figures_dir / dm)
    fig_emd_mat, axs_emd_mat = visualization.plot_pairwise_emd_matrix(
        df_emd=occupancy_emd_df,
        binned_occupancy_dict=combined_occupancy_dict[dm],
        packing_modes=packing_modes,
        distance_measure=dm,
        normalization=normalization,
        xlim=occupancy_params[dm]["xlim"],
        ylim=occupancy_params[dm]["ylim"],
        figures_dir=occupancy_figures_dir,
        suffix=suffix,
        save_format=save_format,
    )
# %% [markdown]
# ## Pairwise envelope test on occupancy ratios
# %% [markdown]
# ### Run pairwise envelope test on occupancy ratio curves
occ_pairwise_results = occupancy.pairwise_envelope_test_occupancy(
    combined_occupancy_dict=combined_occupancy_dict,
    packing_modes=packing_modes,
    alpha=0.05,
    statistic="intdev",
    comparison_type="ecdf",
)
# %% [markdown]
# ### Plot pairwise occupancy envelope matrix per distance measure
envelope_figures_dir = make_dir(figures_dir / "pairwise_envelope")

for dm in occupancy_distance_measures:
    fig_env, axs_env = visualization.plot_pairwise_envelope_matrix(
        pairwise_results=occ_pairwise_results,
        distance_measure=dm,
        figures_dir=envelope_figures_dir,
        suffix=suffix,
        save_format=save_format,
        figure_size=(3.5, 2.5),
        font_scale=0.8,
    )
# %% [markdown]
# ### Plot pairwise occupancy envelope matrix — joint test
fig_env_joint, axs_env_joint = visualization.plot_pairwise_envelope_matrix(
    pairwise_results=occ_pairwise_results,
    distance_measure=None,
    figures_dir=envelope_figures_dir,
    suffix=suffix,
    save_format=save_format,
    figure_size=(7, 3.5),
    font_scale=1.1,
)
# %% [markdown]
# ### Per distance measure rejection bars (per reference mode)
# %%
for ref_mode in packing_modes:
    for joint_test in [False, True]:
        fig_env, axs_env = visualization.plot_per_dm_rejection_bars(
            pairwise_results=occ_pairwise_results,
            reference_mode=ref_mode,
            joint_test=joint_test,
            figures_dir=envelope_figures_dir,
            figsize=(3.5, 2),
            suffix=suffix,
            save_format=save_format,
        )
# %% [markdown]
# ### Per distance measure envelope overlays
# %%
fig_env, axs_env = visualization.plot_per_dm_envelopes_overlaid(
    pairwise_results=occ_pairwise_results,
    figures_dir=envelope_figures_dir,
    suffix=suffix,
    figsize=(6, 1.5),
    save_format=save_format,
)
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
        combined_occupancy_dict=combined_occupancy_dict,
        baseline_mode=ref_mode,
        significance_level=ks_significance_level,
        results_dir=None,
        recalculate=True,
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
    log_dir / f"{STRUCTURE_NAME}_pairwise_occupancy_ks_central_tendencies{suffix}.log"
)
for _ref_mode in packing_modes:
    ref_boot_df = pairwise_occ_ks_bootstrap_df.query("baseline_mode == @_ref_mode")
    distance.log_central_tendencies_for_ks(
        df_ks_bootstrap=ref_boot_df,
        distance_measures=occupancy_distance_measures,
        file_path=pairwise_ks_log_file_path,
    )
# %%
logger.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
