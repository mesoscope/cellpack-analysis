# %% [markdown]
# # Occupancy workflow for punctate structures
#
# Calculate and plot the occupancy ratio for different observed and simulated data
import logging
import time

from IPython.display import display

from cellpack_analysis.lib import distance, occupancy, visualization
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import normalize_distances

logger = logging.getLogger(__name__)

start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID
# SLC25A17: peroxisomes
# RAB5A: early endosomes
# SEC61B: ER
# ST6GAL1: Golgi
STRUCTURE_ID = "SLC25A17"
PACKING_ID = "peroxisome"
STRUCTURE_NAME = "peroxisome"
# %% [markdown]
# ### Set packing modes and channel map
save_format = "pdf"
packing_modes = [
    STRUCTURE_ID,
    "random",
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    # "apical_gradient",
    "apical_gradient_weak",
    # "struct_gradient_weak",
]

channel_map = {
    "SLC25A17": "SLC25A17",
    "random": "SLC25A17",
    "nucleus_gradient_strong": "SLC25A17",
    "membrane_gradient_strong": "SLC25A17",
    "apical_gradient_weak": "SLC25A17",
    # "struct_gradient_weak": "ST6GAL1",
    # "struct_gradient_weak": "ST6GAL1",
    # "struct_gradient": "ST6GAL1",
}

# relative path to packing outputs
packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
baseline_mode = STRUCTURE_ID

all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = base_results_dir / f"occupancy_analysis/{PACKING_ID}"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Distance measures to use
# Options are "nucleus", "z", "scaled_nucleus", "membrane"
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
# ### Calculate distance measures and normalize
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=occupancy_distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
    num_workers=32,
)
all_distance_dict = normalize_distances(
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
# occupancy_distance_measure = "z"
combined_occupancy_dict = {}
occupancy_axes_dict = {}
minimum_distance = -1  # allowance for small negative distances
for occupancy_distance_measure in occupancy_distance_measures:
    logger.info(f"Starting occupancy analysis for distance measure: {occupancy_distance_measure}")
    occupancy_figures_dir = figures_dir / occupancy_distance_measure
    occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
    distance_kde_dict = distance.get_distance_distribution_kde(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=combined_mesh_information_dict,
        channel_map=channel_map,
        save_dir=results_dir,
        recalculate=False,
        suffix=suffix,
        normalization=normalization,
        distance_measure=occupancy_distance_measure,
        minimum_distance=minimum_distance,
    )

    # ### Plot illustration for occupancy distribution
    pdf_occupied, pdf_available, xvals, occupancy_vals, fig_ill, axs_ill = (
        visualization.plot_occupancy_illustration(
            kde_dict=distance_kde_dict,
            baseline_mode="random",
            suffix=suffix,
            distance_measure=occupancy_distance_measure,
            normalization=normalization,
            method="pdf",
            seed_index=743916,
            figures_dir=occupancy_figures_dir,
            save_format=save_format,
            xlim=occupancy_params[occupancy_distance_measure]["xlim"],
            bandwidth=0.4,
        )
    )

    # ### Calculate occupancy ratio
    occupancy_dict = occupancy.get_kde_occupancy_dict(
        distance_kde_dict=distance_kde_dict,
        channel_map=channel_map,
        results_dir=results_dir,
        recalculate=False,
        suffix=suffix,
        distance_measure=occupancy_distance_measure,
        bandwidth=occupancy_params[occupancy_distance_measure]["bandwidth"],
        # bandwidth="scott",
        # num_cells=5,
        num_points=250,
        x_min=0,
        x_max=occupancy_params[occupancy_distance_measure]["xlim"],
    )

    # ### Plot occupancy ratio
    fig, ax = visualization.plot_occupancy_ratio(
        occupancy_dict=occupancy_dict,
        channel_map=channel_map,
        baseline_mode=baseline_mode,
        figures_dir=occupancy_figures_dir,
        suffix=suffix,
        normalization=normalization,
        distance_measure=occupancy_distance_measure,
        save_format=save_format,
        xlim=occupancy_params[occupancy_distance_measure]["xlim"],
        # xlim=12,
        ylim=occupancy_params[occupancy_distance_measure]["ylim"],
        fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
    )
    combined_occupancy_dict[occupancy_distance_measure] = occupancy_dict


# %% [markdown]
# ### Test interpolation
interp_occupancy_dict = occupancy.interpolate_occupancy_dict(
    occupancy_dict=combined_occupancy_dict,
    channel_map=channel_map,
    baseline_mode=baseline_mode,
    results_dir=results_dir,
    suffix=suffix,
)
# %% [markdown]
# ### Plot interpolated occupancy ratio
interp_figures_dir = figures_dir / "interpolation"
interp_figures_dir.mkdir(exist_ok=True, parents=True)
for occupancy_distance_measure in occupancy_distance_measures:
    for plot_type in ["individual", "joint"]:
        fig, ax = visualization.plot_occupancy_ratio(
            occupancy_dict=combined_occupancy_dict[occupancy_distance_measure],
            channel_map=channel_map,
            baseline_mode=baseline_mode,
            suffix=suffix,
            normalization=normalization,
            distance_measure=occupancy_distance_measure,
            xlim=occupancy_params[occupancy_distance_measure]["xlim"],
            # xlim=12,
            ylim=occupancy_params[occupancy_distance_measure]["ylim"],
            fig_params={"dpi": 300, "figsize": (3.5, 2.5)},
        )
        fig_interp, ax_interp = visualization.add_baseline_occupancy_interpolation_to_plot(
            ax=ax,
            interpolated_occupancy_dict=interp_occupancy_dict,
            baseline_mode=baseline_mode,
            distance_measure=occupancy_distance_measure,
            figures_dir=interp_figures_dir,
            suffix=suffix,
            save_format=save_format,
            plot_type=plot_type,
        )
        display(fig_interp)
# %%
logger.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
