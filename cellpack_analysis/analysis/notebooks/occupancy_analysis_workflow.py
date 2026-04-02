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

CONDITION = "norm_weights"
"""Experimental condition / packing output subfolder."""

RESULT_SUBFOLDER = "occupancy_test_kde_norm_weights"
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

results_dir = make_dir(base_results_dir / RESULT_SUBFOLDER / PACKING_ID)

figures_dir = make_dir(results_dir / "figures/")
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
if normalization is not None:
    suffix = f"_normalized_{normalization}"
else:
    suffix = ""

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
combined_mesh_information_dict = {}
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    combined_mesh_information_dict[structure_id] = mesh_information_dict
# %% [markdown]
# ## Distance analysis
# %% [markdown]
# ### Calculate distance measures and normalize
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=occupancy_distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
    num_workers=8,
)
all_distance_dict = distance.normalize_distance_dictionary(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)
# %% [markdown]
# ## Occupancy Analysis
# %% [markdown]
# ### Set limits and bandwidths for plotting
occupancy_params = {
    "nucleus": {"xlim": 6, "ylim": 3, "bandwidth": 0.2},
    "z": {"xlim": 8, "ylim": 2, "bandwidth": 0.2},
}
# %% [markdown]
# ### Create kde dictionary for individual distance distributions
# occupancy_distance_measure = "nucleus"  # for testing
occupancy_axes_dict = {}
occupancy_figures_dir = {}
minimum_distance = 0  # allowance for small negative distances
for dm in occupancy_distance_measures:
    logger.info(f"Starting occupancy analysis for distance measure: {dm}")
    occupancy_figures_dir[dm] = figures_dir / dm
    occupancy_figures_dir[dm].mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Calculate distance kde dictionary
distance_kde_dict = {}
for dm in occupancy_distance_measures:
    distance_kde_dict[dm] = distance.get_distance_distribution_kde(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=combined_mesh_information_dict,
        channel_map=channel_map,
        save_dir=results_dir,
        recalculate=False,
        suffix=suffix,
        normalization=normalization,
        distance_measure=dm,
        minimum_distance=minimum_distance,
    )

# %% [markdown]
# ### Plot illustration for occupancy distribution
for dm in occupancy_distance_measures:
    pdf_occupied, pdf_available, xvals, occupancy_vals, fig_ill, axs_ill = (
        visualization.plot_occupancy_illustration(
            kde_dict=distance_kde_dict[dm],
            packing_mode="random",
            suffix=suffix,
            distance_measure=dm,
            normalization=normalization,
            method="pdf",
            cellid_index=743916,
            figures_dir=occupancy_figures_dir[dm],
            save_format=save_format,
            xlim=occupancy_params[dm]["xlim"],
            bandwidth=0.4,
        )
    )

# %% [markdown]
# ### Calculate occupancy ratio
combined_occupancy_dict = {}
for dm in occupancy_distance_measures:
    combined_occupancy_dict[dm] = occupancy.get_kde_occupancy_dict(
        distance_kde_dict=distance_kde_dict[dm],
        channel_map=channel_map,
        results_dir=results_dir,
        recalculate=False,
        suffix=suffix,
        distance_measure=dm,
        bandwidth=occupancy_params[dm]["bandwidth"],
        # bandwidth="scott",
        # num_cells=5,
        num_points=250,
        x_min=0,
        x_max=occupancy_params[dm]["xlim"],
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
        # xlim=12,
        ylim=occupancy_params[dm]["ylim"],
        fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
    )

# %% [markdown]
# ## Calculate occupancy EMD
occupancy_emd_df = occupancy.get_occupancy_emd_df(
    combined_occupancy_dict=combined_occupancy_dict,
    packing_modes=packing_modes,
    distance_measures=occupancy_distance_measures,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ## Plot occupancy EMD
emd_figures_dir = make_dir(figures_dir / "emd")
fig_v_emd, ax_v_emd, fig_b_emd, ax_b_emd = visualization.plot_emd_comparisons(
    df_emd=occupancy_emd_df,
    distance_measures=occupancy_distance_measures,
    comparison_type="baseline",
    baseline_mode=baseline_mode,
    suffix=suffix,
    figures_dir=emd_figures_dir,
    save_format=save_format,
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
# ## Calculate KS observed values
occupancy_ks_observed_df = occupancy.get_occupancy_ks_test_df(
    distance_measures=occupancy_distance_measures,
    packing_modes=packing_modes,
    combined_occupancy_dict=combined_occupancy_dict,
    baseline_mode=baseline_mode,
    save_dir=results_dir,
    recalculate=True,
)
# %% [markdown]
# ## Bootstrap KS test
occupancy_ks_bootstrap_df = distance.bootstrap_ks_tests(
    ks_test_df=occupancy_ks_observed_df,
    distance_measures=occupancy_distance_measures,
    packing_modes=[pm for pm in packing_modes if pm != baseline_mode],
    n_bootstrap=1000,
)
# %% [markdown]
# ## Plot KS test results
ks_figures_dir = make_dir(figures_dir / "ks_tests")
visualization.plot_ks_test_results(
    df_ks_bootstrap=occupancy_ks_bootstrap_df,
    distance_measures=occupancy_distance_measures,
    suffix=suffix,
    figures_dir=ks_figures_dir,
    baseline_mode=baseline_mode,
    save_format=save_format,
)
# %% [markdown]
# ### Interpolate occupancy ratio
interp_occupancy_dict = occupancy.interpolate_occupancy_dict(
    occupancy_dict=combined_occupancy_dict,
    channel_map=channel_map,
    baseline_mode=baseline_mode,
    results_dir=results_dir,
    suffix=suffix,
)
# %% [markdown]
# ### Plot interpolated occupancy ratio
interp_figures_dir = make_dir(figures_dir / "interpolated")
for dm in occupancy_distance_measures:
    for plot_type in ["individual", "joint"]:
        fig, ax = visualization.plot_occupancy_ratio(
            occupancy_dict=combined_occupancy_dict[dm],
            channel_map=channel_map,
            baseline_mode=baseline_mode,
            suffix=suffix,
            normalization=normalization,
            distance_measure=dm,
            xlim=occupancy_params[dm]["xlim"],
            # xlim=12,
            ylim=occupancy_params[dm]["ylim"],
            fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
        )
        fig_interp, ax_interp = visualization.add_baseline_occupancy_interpolation_to_plot(
            ax=ax,
            interpolated_occupancy_dict=interp_occupancy_dict,
            baseline_mode=baseline_mode,
            distance_measure=dm,
            figures_dir=interp_figures_dir,
            suffix=suffix,
            save_format=save_format,
            plot_type=plot_type,
        )
# %%
logger.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
