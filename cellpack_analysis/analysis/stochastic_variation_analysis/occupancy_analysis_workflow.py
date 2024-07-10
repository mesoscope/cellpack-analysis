# %% [markdown]
# # Occupancy analysis workflow
from pathlib import Path
import time

import matplotlib.pyplot as plt

from cellpack_analysis.analysis.stochastic_variation_analysis import distance
from cellpack_analysis.analysis.stochastic_variation_analysis.load_data import (
    get_position_data_from_outputs,
)
from cellpack_analysis.analysis.stochastic_variation_analysis.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

plt.rcParams.update({"font.size": 14})
start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID and radius
STRUCTURE_ID = "SLC25A17"  # "SLC25A17" for peroxisomes, "RAB5A" for early endosomes
STRUCT_RADIUS = 2.37  # 2.37 um for peroxisomes, 2.6 um for early endosomes
# %% [markdown]
# ### Set packing modes to analyze
baseline_analysis = False
packing_modes_baseline = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "shape",
]
packing_modes_rules = [
    STRUCTURE_ID,
    "random",
    "nucleus_moderate",
    "nucleus_moderate_invert",
    "planar_gradient_Z_moderate",
]

# relative path to packing outputs
if baseline_analysis:
    packing_output_folder = "packing_outputs/stochastic_variation_analysis/"
    packing_modes = packing_modes_baseline
    baseline_mode = "mean_count_and_size"
else:
    # TODO: update path with new packings
    packing_output_folder = "packing_outputs/8d_sphere_data/RS/"
    packing_modes = packing_modes_rules
    baseline_mode = STRUCTURE_ID
# %% [markdown]
# ### Set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

folder_name = "baseline" if baseline_analysis else "rules"
results_dir = (
    base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_ID}/{folder_name}"
)
results_dir.mkdir(exist_ok=True, parents=True)

occupancy_figures_dir = results_dir / "figures/occupancy"
occupancy_figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Distance measures to use
distance_measures = ["pairwise", "nucleus", "nearest", "z"]
# %% [markdown]
# ### Set normalization
# options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = "cell_diameter"
if normalization is not None:
    suffix = f"_normalized_{normalization}"

# %% [markdown]
# ### Read position data from outputs
all_positions = get_position_data_from_outputs(
    structure_id=STRUCTURE_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
    recalculate=True,
    baseline_analysis=baseline_analysis,
)
# %% [markdown]
# ### Check number of packings for each result
for mode, position_dict in all_positions.items():
    print(mode, len(position_dict))
# %% [markdown]
# ### Get mesh information
mesh_information_dict = get_mesh_information_dict(
    structure_id=STRUCTURE_ID,
    base_datadir=base_datadir,
    recalculate=True,
)
# %% [markdown]
# ### Plot distribution of cell diameters
distance.plot_cell_diameter_distribution(mesh_information_dict)

# %% [markdown]
# ### Calculate distance measures and normalize
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=mesh_information_dict,
    results_dir=results_dir,
    recalculate=True,
)

all_distance_dict = normalize_distances(
    all_distance_dict, normalization, mesh_information_dict
)

# %% [markdown]
# ## Occupancy Analysis
# %% [markdown]
# ### Choose distance measure to use for occupancy analysis
# Options are "pairwise", "nucleus", "nearest", "z"
occupancy_distance_measure = "nucleus"
# %% [markdown]
# ### Create kde dictionary for individual distance distributions
kde_dict = distance.get_individual_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
)
# %% [markdown]
# ### Plot illustration for occupancy distribution
kde_distance, kde_available_space, xvals, yvals = distance.plot_occupancy_illustration(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=kde_dict,
    baseline_mode="random",
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    struct_diameter=STRUCT_RADIUS,
    mesh_information_dict=mesh_information_dict,
    ratio_to_plot="occupancy",
)

# %% [markdown]
# ### plot individual space corrected occupancy kde
distance.plot_individual_occupancy_ratio(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
)
# %% [markdown]
# ### get combined space corrected kde
combined_kde_dict = distance.get_combined_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    # sample=0.01,
)
# %% [markdown]
# ### plot combined space corrected kde
# aspect = 0.02
aspect = None
fig, ax = distance.plot_combined_occupancy_ratio(
    combined_kde_dict=combined_kde_dict,
    packing_modes=packing_modes,
    figures_dir=occupancy_figures_dir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
    normalization=normalization,
    aspect=aspect,
    save_format="png",
    distance_measure=occupancy_distance_measure,
    num_points=100,
    ratio_to_plot="occupancy",
)
# %% [markdown]
# #### edit figure as needed
# aspect = 0.01
# ax.set_aspect(aspect)
# from matplotlib.ticker import MaxNLocator

# plt.rcParams.update({"font.size": 16})
# ax.set_ylim([-5, 20])
# ax.set_xlim([0, 0.25])
# # ax.set_xlabel("")
# # ax.set_ylabel("")
# ax.xaxis.set_major_locator(MaxNLocator(5))
# plt.tight_layout()
# fig
# # %% [markdown]
# # #### save edited figure
# fig.savefig(
#     occupancy_figures_dir / f"combined_space_corrected_kde_aspect{suffix}.svg", dpi=300
# )
# %% [markdown]
# ### get EMD between occupied and available distances
emd_occupancy_dict = distance.get_occupancy_emd(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
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
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ### plot ks test results
distance.plot_occupancy_ks_test(
    ks_occupancy_dict=ks_occupancy_dict,
    figures_dir=occupancy_figures_dir,
    distance_measure=occupancy_distance_measure,
)

print(f"Time taken: {time.time() - start_time:.2f} s")

# %%
