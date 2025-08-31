# %% [markdown]
# # Occupancy analysis workflow
# Plot occupancy ratio for observed and simulated data
import logging
import time

import matplotlib.pyplot as plt

from cellpack_analysis.analysis.punctate_analysis.lib import (
    distance,
    occupancy,
    visualization,
)
from cellpack_analysis.analysis.punctate_analysis.lib.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 14})
start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID and radius
# SLC25A17: peroxisomes
# RAB5A: early endosomes
# SEC61B: ER
# ST6GAL1: Golgi
STRUCTURE_ID = "SLC25A17"
PACKING_ID = "peroxisome"
STRUCTURE_NAME = "peroxisome"
# %% [markdown]
# ### Set packing modes to analyze
save_format = "svg"
packing_modes = [
    STRUCTURE_ID,
    "random",
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    # "apical_gradient",
    # "struct_gradient",
]

channel_map = {
    "SLC25A17": "SLC25A17",
    "random": "SLC25A17",
    "nucleus_gradient_strong": "SLC25A17",
    "membrane_gradient_strong": "SLC25A17",
    # "struct_gradient": "SEC61B",
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

folder_name = "rules"
results_dir = base_results_dir / f"occupancy_analysis/{PACKING_ID}/{folder_name}"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
save_format = "svg"
# %% [markdown]
# ### Distance measures to use
distance_measures = [
    "nearest",
    "pairwise",
    "nucleus",
    "scaled_nucleus",
    # "z",
    # "membrane",
]
# %% [markdown]
# ### Set normalization
# options: None, "intracellular_radius", "cell_diameter", "max_distance"
# normalization = "cell_diameter"
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
    distance_measures=distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
)

all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict,
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
# ### Choose distance measure to use for occupancy analysis
# Options are "pairwise", "nucleus", "nearest", "z", "scaled_nucleus", "membrane"
occupancy_distance_measure = "nucleus"
# %% [markdown]
# ### Create kde dictionary for individual distance distributions
distance_kde_dict = distance.get_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
)

# %% [markdown]
# ### Plot illustration for occupancy distribution
kde_distance, kde_available_space, xvals, yvals, fig_ill, axs_ill = (
    visualization.plot_occupancy_illustration(
        distance_dict=all_distance_dict[occupancy_distance_measure],
        kde_dict=distance_kde_dict,
        baseline_mode="random",
        suffix=suffix,
        distance_measure=occupancy_distance_measure,
        normalization=normalization,
        method="pdf",
        seed_index=10,
        # xlim=3.01,
        figures_dir=figures_dir,
        save_format=save_format,
    )
)

# %% [markdown]
# ### Plot individual occupancy ratio
figs_ind, axs_ind = visualization.plot_individual_occupancy_ratio(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    suffix=suffix,
    method="pdf",
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    xlim=4,
    ylim=3,
    # sample_size=10,
    figures_dir=figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ### plot mean and std of occupancy ratio
figs_ci, axs_ci = visualization.plot_mean_and_std_occupancy_ratio_kde(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    method="pdf",
    xlim=3,
    ylim=3,
    # sample_size=10,
    figures_dir=figures_dir,
    save_format=save_format,
)
# # %%
# for fig, ax in zip(figs_ci, axs_ci):
#     ax.set_xlim([0, 1])
#     ax.set_xlabel("Scaled Distance from Nucleus")
# fig
# %% [markdown]
# ### get combined space corrected kde
combined_kde_dict = occupancy.get_combined_occupancy_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    # sample_size=10,
)
# %% [markdown]
# ### plot combined space corrected kde
# aspect = 0.02
occupancy_start_time = time.time()
aspect = None
method = "pdf"
fig_combined, ax_combined = visualization.plot_combined_occupancy_ratio(
    combined_kde_dict=combined_kde_dict,
    packing_modes=packing_modes,
    suffix=suffix,
    normalization=normalization,
    aspect=aspect,
    save_intermediates=True,
    distance_measure=occupancy_distance_measure,
    num_points=1000,
    method=method,
    xlim=4,
    ylim=3,
    figures_dir=figures_dir,
    save_format=save_format,
)
log.info(f"Time taken to plot occupancy ratio: {time.time() - occupancy_start_time:.2f} s")
# %% [markdown]
# ### plot binned occupancy ratio
fig_binned, ax_binned = visualization.plot_binned_occupancy_ratio(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    packing_modes=packing_modes,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
    # num_bins=16,
    bin_width=0.2,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    xlim=4,
    ylim=3,
    # sample_size=10,
    figures_dir=figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ### get EMD between occupied and available distances
emd_occupancy_dict = occupancy.get_occupancy_emd(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ### box and whisker plot for occupancy EMD values
visualization.plot_occupancy_emd_boxplot(
    emd_occupancy_dict=emd_occupancy_dict,
    suffix=suffix,
    figures_dir=figures_dir,
)
# %% [markdown]
# ### run ks test for occupancy distributions
ks_occupancy_dict = occupancy.get_occupancy_ks_test_dict(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ### plot ks test results
visualization.plot_occupancy_ks_test(
    ks_occupancy_dict=ks_occupancy_dict,
    distance_measure=occupancy_distance_measure,
    figures_dir=figures_dir,
)

log.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
