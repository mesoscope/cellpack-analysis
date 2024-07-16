# %%
from pathlib import Path

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
# %% set structure ID
STRUCTURE_ID = "SLC25A17"
STRUCT_RADIUS = 2.37  # 2.37 um for peroxisomes, 2.6 um for early endosomes
# %% set packing modes
baseline_analysis = False
packing_modes_baseline = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    # "variable_count_and_size",
    "shape",
]
packing_modes_rules = [
    STRUCTURE_ID,
    "random",
    "nucleus_moderate",
    "nucleus_moderate_invert",
    # TODO: add bias towards membrane
    # TODO: add bias away from membrane
    "planar_gradient_Z_moderate",
]
packing_modes = packing_modes_baseline if baseline_analysis else packing_modes_rules
baseline_mode = "mean_count_and_size" if baseline_analysis else STRUCTURE_ID
# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

folder_name = "baseline" if baseline_analysis else "rules"
results_dir = (
    base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_ID}/{folder_name}"
)
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% distance measures to use
distance_measures = ["pairwise", "nucleus", "nearest", "z"]
# %% set suffix
normalization = "cell_diameter"  # options: None, "intracellular_radius", "cell_diameter", "max_distance"
if normalization is not None:
    suffix = f"_normalized_{normalization}"

# %% read position data from outputs
all_positions = get_position_data_from_outputs(
    structure_id=STRUCTURE_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    recalculate=False,
    baseline_analysis=baseline_analysis,
)
# %% check number of packings for each result
for mode, position_dict in all_positions.items():
    print(mode, len(position_dict))
# %% calculate mesh information
mesh_information_dict = get_mesh_information_dict(
    structure_id=STRUCTURE_ID,
    base_datadir=base_datadir,
    recalculate=False,
)
# %% plot distribution of cell diameters
distance.plot_cell_diameter_distribution(mesh_information_dict)
# %%
avg_struct_diameter, std_struct_diameter = distance.get_average_scaled_diameter(
    struct_diameter=STRUCT_RADIUS, mesh_information_dict=mesh_information_dict
)
# %% Calculate distance measures
all_distance_dict = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=mesh_information_dict,
    results_dir=results_dir,
    recalculate=False,
)
# %% Normalize distances
all_distance_dict = normalize_distances(
    all_distance_dict, normalization, mesh_information_dict
)
# %% plot distance PDFs
distance.plot_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% plot distance PDFs overlaid
distance.plot_distance_distributions_overlay(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% plot distance distribution histograms
distance.plot_distance_distributions_histogram(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% [markdown]
# ## KS Test Analysis
# %% ks test between observed and other modes
ks_observed_dict = distance.get_ks_observed_dict(
    distance_measures,
    packing_modes,
    all_distance_dict,
    baseline_mode=baseline_mode,
)
# %% plot ks test results
distance.plot_ks_observed_barplots(
    ks_observed_dict=ks_observed_dict,
    figures_dir=figures_dir,
    suffix=suffix,
)
# %% plot colorized ks distance distributions
distance.plot_ks_test_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    ks_observed_dict=ks_observed_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
    baseline_mode=baseline_mode,
)
# %% [markdown]
# # Pairwise EMD Analysis
# %% Get pairwise earth movers distances between distance distributions
all_pairwise_emd = distance.get_pairwise_emd_dictionary(
    all_distance_dict=all_distance_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% create pairwise emd folders
pairwise_emd_dir = figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)
# %% calculate pairwise EMD distances across modes
distance.plot_pairwise_emd_heatmaps(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot emd correlation heatmap
corr_df_dict = distance.get_average_emd_correlation(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    pairwise_emd_dir=pairwise_emd_dir,
    baseline_mode=baseline_mode,
    suffix=suffix,
)
# %% plot EMD heatmaps with variation
distance.plot_emd_correlation_circles(
    distance_measures=distance_measures,
    corr_df_dict=corr_df_dict,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD box plots
distance.plot_emd_boxplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD histograms
distance.plot_emd_histograms(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD kdeplots
distance.plot_emd_kdeplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% [markdown]
# # Ripley's K Analysis
# %% calculate ripleyK for all positions
all_ripleyK, mean_ripleyK, ci_ripleyK, r_values = distance.calculate_ripley_k(
    all_positions=all_positions,
    mesh_information_dict=mesh_information_dict,
)
# %% plot ripleyK distributions
distance.plot_ripley_k(
    mean_ripleyK=mean_ripleyK,
    ci_ripleyK=ci_ripleyK,
    r_values=r_values,
    figures_dir=figures_dir,
    suffix=suffix,
)

# %% [markdown]
# # Occupancy Analysis
# %% create figure directory
occupancy_figdir = figures_dir / "occupancy"
occupancy_figdir.mkdir(exist_ok=True, parents=True)
# %% choose distance measure
occupancy_distance_measure = "z"
# %% create kde dictionary
kde_dict = distance.get_individual_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    normalization=normalization,
    distance_measure=occupancy_distance_measure,
)
# %% plot illustration for space corrected kde
kde_distance, kde_available_space, xvals = distance.plot_occupancy_illustration(
    distance_dict=all_distance_dict[occupancy_distance_measure],
    kde_dict=kde_dict,
    baseline_mode="random",
    figures_dir=occupancy_figdir,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
    struct_diameter=avg_struct_diameter,
)

# %% plot individual space corrected individual kde values
distance.plot_individual_occupancy_ratio(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=[STRUCTURE_ID, "random", "nucleus_moderate"],
    figures_dir=occupancy_figdir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
)
# %% get combined space corrected kde
combined_kde_dict = distance.get_combined_distance_distribution_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    distance_measure=occupancy_distance_measure,
)
# %% plot combined space corrected kde
# aspect = 0.02
aspect = None
fig, ax = distance.plot_combined_occupancy_ratio(
    combined_kde_dict=combined_kde_dict,
    packing_modes=[
        "random",
        "nucleus_moderate",
        STRUCTURE_ID,
        # "nucleus_moderate_invert",
        # "planar_gradient_Z_moderate",
    ],
    figures_dir=occupancy_figdir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
    normalization=normalization,
    aspect=aspect,
    save_format="png",
    distance_measure=occupancy_distance_measure,
)
# %% adjust aspect ratio
# aspect = 0.01
# ax.set_aspect(aspect)
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 18})
ax.set_ylim([0, 1.5])
# ax.set_xlim([0, 0.4])
# ax.set_xlabel("")
# ax.set_ylabel("")
ax.xaxis.set_major_locator(MaxNLocator(5))
plt.tight_layout()
fig
# %%
fig.savefig(
    occupancy_figdir / f"combined_space_corrected_kde_aspect{suffix}.svg", dpi=300
)
# %% get EMD between occupied and available distances
emd_occupancy_dict = distance.get_occupancy_emd(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% occupancy emd kdeplot
distance.plot_occupancy_emd_kdeplot(
    emd_occupancy_dict=emd_occupancy_dict,
    packing_modes=packing_modes,
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% create box and whisker plot for occupancy EMD values
distance.plot_occupancy_emd_boxplot(
    emd_occupancy_dict=emd_occupancy_dict,
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% run ks test for occupancy distributions
ks_occupancy_dict = distance.get_occupancy_ks_test_dict(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% plot ks test results
distance.plot_occupancy_ks_test(
    ks_occupancy_dict=ks_occupancy_dict, figures_dir=occupancy_figdir
)
