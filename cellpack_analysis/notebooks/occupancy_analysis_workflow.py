# %% [markdown]
# # Occupancy workflow for punctate structures
#
# Calculate and plot the occupancy ratio for different observed and simulated data
import logging
import time

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
    # "struct_gradient_weak",
    # "struct_gradient",
]

channel_map = {
    "SLC25A17": "SLC25A17",
    # "RAB5A": "RAB5A",
    "random": "SLC25A17",
    "nucleus_gradient_strong": "SLC25A17",
    "membrane_gradient_strong": "SLC25A17",
    # "apical_gradient": "RAB5A",
    # "struct_gradient": "SEC61B",
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

results_dir = base_results_dir / f"occupancy_analysis/{PACKING_ID}/test_negative_distances"
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
    structure_name=PACKING_ID,
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
# ### Calculate distance measures and normalize
minimum_distance = -1  # in microns
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=occupancy_distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=True,
    num_workers=16,
    minimum_distance=minimum_distance,
)
all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict, minimum_distance=minimum_distance
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
    "nucleus": {"xlim": 6, "ylim": 4, "bandwidth": 0.4},
    "z": {"xlim": 8, "ylim": 4, "bandwidth": 0.4},
}
# %% [markdown]
# ### Create kde dictionary for individual distance distributions
occupancy_distance_measure = "nucleus"
# for occupancy_distance_measure in occupancy_distance_measures:
occupancy_figures_dir = figures_dir / occupancy_distance_measure
occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
logger.info(f"Starting occupancy analysis for distance measure: {occupancy_distance_measure}")
distance_kde_dict = distance.get_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    save_dir=results_dir,
    recalculate=True,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    minimum_distance=minimum_distance,
)

# %% [markdown]
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
        bandwidth=occupancy_params[occupancy_distance_measure]["bandwidth"],
    )
)
# %% [markdown]
# ### Calculate occupancy ratio
occupancy_dict = occupancy.get_kde_occupancy_dict(
    distance_kde_dict=distance_kde_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    bandwidth=occupancy_params[occupancy_distance_measure]["bandwidth"],
    # bandwidth="scott",
    # num_cells=5,
    num_points=250,
)

# %% [markdown]
# ### Plot occupancy ratio
fig, ax = visualization.plot_occupancy_ratio(
    occupancy_dict=occupancy_dict,
    channel_map=channel_map,
    # figures_dir=occupancy_figures_dir,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    save_format=save_format,
    xlim=occupancy_params[occupancy_distance_measure]["xlim"],
    # xlim=12,
    ylim=occupancy_params[occupancy_distance_measure]["ylim"],
)


# %% [markdown]
# ### Test interpolation
interp_occupancy_dict = occupancy.interpolate_occupancy_dict(
    occupancy_dict=occupancy_dict,
    baseline_mode=baseline_mode,
    results_dir=results_dir,
    suffix=suffix,
)
# %% [markdown]
# ### Plot interpolated occupancy ratio
fig_interp, ax_interp = visualization.plot_occupancy_ratio_interpolation(
    interpolated_occupancy_dict=interp_occupancy_dict,
    baseline_mode=baseline_mode,
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    xlim=occupancy_params[occupancy_distance_measure]["xlim"],
    ylim=occupancy_params[occupancy_distance_measure]["ylim"],
    save_format=save_format,
)
# %% [markdown]
# ### Get binned occupancy ratio
binned_occupancy_dict = occupancy.get_binned_occupancy_dict(
    distance_kde_dict=distance_kde_dict,
    channel_map=channel_map,
    bin_width=0.4,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    x_max=occupancy_params[occupancy_distance_measure]["xlim"],
    # x_max=12,
    # num_cells=5,
)
# %% [markdown]
# ### Pot binned occupancy ratio
fig_binned, ax_binned = visualization.plot_binned_occupancy_ratio(
    binned_occupancy_dict=binned_occupancy_dict,
    channel_map=channel_map,
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    normalization=normalization,
    xlim=occupancy_params[occupancy_distance_measure]["xlim"],
    # xlim=12,
    ylim=occupancy_params[occupancy_distance_measure]["ylim"],
    save_format=save_format,
)
# %% [markdown]
# ### Test interpolation
interp_binned_occupancy_dict = occupancy.interpolate_occupancy_dict(
    occupancy_dict=binned_occupancy_dict,
    baseline_mode=baseline_mode,
    results_dir=results_dir,
    suffix=f"_binned_{occupancy_distance_measure}",
)
# %% [markdown]
# ### Plot interpolated occupancy ratio
fig_interp, ax_interp = visualization.plot_occupancy_ratio_interpolation(
    interpolated_occupancy_dict=interp_binned_occupancy_dict,
    baseline_mode=baseline_mode,
    figures_dir=occupancy_figures_dir,
    suffix="_binned",
    distance_measure=occupancy_distance_measure,
    xlim=occupancy_params[occupancy_distance_measure]["xlim"],
    ylim=occupancy_params[occupancy_distance_measure]["ylim"],
    save_format=save_format,
)
# %%
logger.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
