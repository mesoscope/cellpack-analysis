# %%
import matplotlib.pyplot as plt
from pathlib import Path


from cellpack_analysis.analyses.stochastic_variation_analysis import distance
from cellpack_analysis.analyses.stochastic_variation_analysis.load_data import (
    get_position_data_from_outputs,
)
from cellpack_analysis.analyses.stochastic_variation_analysis.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

plt.rcParams.update({"font.size": 14})
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
    "observed_data",
    "random",
    "nucleus_moderate",
    "nucleus_moderate_invert",
]
packing_modes = packing_modes_baseline if baseline_analysis else packing_modes_rules
baseline_mode = "mean_count_and_size" if baseline_analysis else "observed_data"
# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

folder_name = "baseline" if baseline_analysis else "rules"
results_dir = base_results_dir / f"stochastic_variation_analysis/{folder_name}"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)


# %% set structure ID
STRUCTURE_ID = "SLC25A17"
STRUCT_RADIUS = 2.37
# %% distance measures to use
distance_measures = ["pairwise", "nucleus", "nearest"]
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
import numpy as np
cell_diameters = [cellid_dict["cell_diameter"] for _, cellid_dict in mesh_information_dict.items()]
nuc_diameters = [cellid_dict["nuc_diameter"] for _, cellid_dict in mesh_information_dict.items()]
fig, ax = plt.subplots()
ax.hist(cell_diameters, bins=20, alpha=0.5, label="cell")
ax.hist(nuc_diameters, bins=20, alpha=0.5, label="nucleus")
ax.set_title(
    f"Mean cell diameter: {np.mean(cell_diameters):.2f}\n"
    f"Mean nucleus diameter: {np.mean(nuc_diameters):.2f}"
)
plt.tight_layout()
plt.show()
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
    recalculate=False,
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
corr_df_dict = distance.plot_average_emd_correlation_heatmap(
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
# %% create kde dictionary
kde_dict = distance.get_space_corrected_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    normalization=normalization,
)
# %% plot illustration for space corrected kde
distance.plot_space_corrected_kde_illustration(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    baseline_mode="random",
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% plot individual space corrected individual kde values
distance.plot_individual_space_corrected_kde(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=["observed_data", "random", "nucleus_moderate"],
    figures_dir=occupancy_figdir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
)
# %% get combined space corrected kde
combined_kde_dict = distance.get_combined_space_corrected_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    normalization=normalization,
)
# %% plot combined space corrected kde
fig, ax = distance.plot_combined_space_corrected_kde(
    combined_kde_dict=combined_kde_dict,
    packing_modes=[
        "random",
        "nucleus_moderate",
        "observed_data",
    ],
    figures_dir=occupancy_figdir,
    suffix=suffix,
    mesh_information_dict=mesh_information_dict,
    struct_diameter=STRUCT_RADIUS,
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

# %%
