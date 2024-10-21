# %% [markdown]
# # Occupancy analysis workflow
# Plot occupancy ratio for observed and simulated data
import logging
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cellpack_analysis.analysis.punctate_analysis import distance
from cellpack_analysis.analysis.punctate_analysis.stats_functions import (
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
STRUCTURE_NAME = "ER_peroxisome"
STRUCT_RADIUS = 2.37  # 2.37 um for peroxisomes, 2.6 um for early endosomes
# %% [markdown]
# ### Set packing modes to analyze
packing_modes = [
    STRUCTURE_ID,
    "random",
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    # "apical_gradient",
    "struct_gradient",
]

channel_map = {
    "SLC25A17": "SLC25A17",
    "random": "SEC61B",
    "nucleus_gradient_strong": "SEC61B",
    "membrane_gradient_strong": "SEC61B",
    "struct_gradient": "SEC61B",
}
all_structures = list(set(channel_map.values()))

# relative path to packing outputs
packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
baseline_mode = STRUCTURE_ID
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

folder_name = "rules"
results_dir = (
    base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_NAME}/{folder_name}"
)
results_dir.mkdir(exist_ok=True, parents=True)

occupancy_figures_dir = results_dir / "figures/occupancy"
occupancy_figures_dir.mkdir(exist_ok=True, parents=True)
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
    structure_name=STRUCTURE_NAME,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
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
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=True,
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
    recalculate=True,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
)

# %% [markdown]
# ### Plot illustration for occupancy distribution
kde_distance, kde_available_space, xvals, yvals, fig_ill, axs_ill = (
    distance.plot_occupancy_illustration(
        distance_dict=all_distance_dict[occupancy_distance_measure],
        kde_dict=distance_kde_dict,
        baseline_mode="random",
        figures_dir=occupancy_figures_dir,
        suffix=suffix,
        distance_measure=occupancy_distance_measure,
        normalization=normalization,
        method="pdf",
        seed_index=10,
        # xlim=3.01,
    )
)

# %% [markdown]
# ### Plot individual occupancy ratio
figs_ind, axs_ind = distance.plot_individual_occupancy_ratio(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=[
        STRUCTURE_ID,
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        "struct_gradient",
        # "apical_gradient",
    ],
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    method="pdf",
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    xlim=4,
    ylim=3,
    # sample_size=10,
)
# %% [markdown]
# ### plot mean and std of occupancy ratio
figs_ci, axs_ci = distance.plot_mean_and_std_occupancy_ratio_kde(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=[
        STRUCTURE_ID,
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        "struct_gradient",
        # "apical_gradient",
    ],
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
    method="pdf",
    xlim=3,
    ylim=3,
    # sample_size=10,
)
# # %%
# for fig, ax in zip(figs_ci, axs_ci):
#     ax.set_xlim([0, 1])
#     ax.set_xlabel("Scaled Distance from Nucleus")
# fig
# %% [markdown]
# ### get combined space corrected kde
combined_kde_dict = distance.get_combined_occupancy_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
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
fig_combined, ax_combined = distance.plot_combined_occupancy_ratio(
    combined_kde_dict=combined_kde_dict,
    # packing_modes=packing_modes,
    packing_modes=[
        STRUCTURE_ID,
        "random",
        "nucleus_gradient_strong",
        "membrane_gradient_strong",
        # "apical_gradient",
        "struct_gradient",
    ],
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    normalization=normalization,
    aspect=aspect,
    save_format=save_format,
    save_intermediates=True,
    distance_measure=occupancy_distance_measure,
    num_points=1000,
    method=method,
    xlim=4,
    ylim=3,
)
log.info(
    f"Time taken to plot occupancy ratio: {time.time() - occupancy_start_time:.2f} s"
)
# %% [markdown]
# #### edit figure as needed
ax = ax_combined
fig = fig_combined

aspect = 0.1
ax.set_aspect(aspect)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(4))
# ax.legend(loc="upper right", prop={"size": 14})
# ax.legend().set_visible(False)
ax.set_xlim((0, 6))
ax.set_ylim((0, 3))
ax.set_ylabel("Occupancy ratio")
ax.set_xlabel("Distance from Nucleus \u03bcm")
plt.tight_layout()
display(fig)
# %% [markdown]
# #### save edited figure
filename = (
    occupancy_figures_dir
    / f"{occupancy_distance_measure}_combined_{method}_occupancy_ratio_edited{suffix}"
)
for format in ["png", "svg"]:
    fig.savefig(
        f"{filename}.{format}",
        dpi=300,
    )

# %% [markdown]
# ### plot binned occupancy ratio
fig_binned, ax_binned = distance.plot_binned_occupancy_ratio(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    packing_modes=packing_modes,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    figures_dir=occupancy_figures_dir,
    normalization=normalization,
    # num_bins=16,
    bin_width=0.2,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    xlim=4,
    ylim=3,
    # sample_size=10,
)
# %% [markdown]
# ### get EMD between occupied and available distances
emd_occupancy_dict = distance.get_occupancy_emd(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% [markdown]
# ### box and whisker plot for occupancy EMD values
distance.plot_occupancy_emd_boxplot(
    emd_occupancy_dict=emd_occupancy_dict,
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
)
# %% [markdown]
# ### run ks test for occupancy distributions
ks_occupancy_dict = distance.get_occupancy_ks_test_dict(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=distance_kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% [markdown]
# ### plot ks test results
distance.plot_occupancy_ks_test(
    ks_occupancy_dict=ks_occupancy_dict,
    figures_dir=occupancy_figures_dir,
    distance_measure=occupancy_distance_measure,
)

log.info(f"Time taken to complete workflow: {time.time() - start_time:.2f} s")

# %%
