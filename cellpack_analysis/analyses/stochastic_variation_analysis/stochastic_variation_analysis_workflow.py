# %%
import matplotlib.pyplot as plt
from pathlib import Path

from cellpack_analysis.analyses.stochastic_variation_analysis.distance import (
    get_combined_space_corrected_kde,
    get_distance_dictionary,
    get_ks_observed_dict,
    get_occupancy_emd,
    get_occupancy_ks_test_dict,
    get_pairwise_emd_dictionary,
    get_space_corrected_kde,
    plot_distance_distributions_histogram,
    plot_distance_distributions_kde,
    plot_average_emd_correlation_heatmap,
    plot_distance_distributions_overlay,
    plot_emd_boxplots,
    plot_emd_correlation_circles,
    plot_emd_histograms,
    plot_individual_space_corrected_kde,
    plot_combined_space_corrected_kde,
    plot_ks_observed_barplots,
    plot_occupancy_emd_boxplot,
    plot_occupancy_emd_kdeplot,
    plot_occupancy_ks_test,
    plot_pairwise_emd_heatmaps,
    plot_emd_kdeplots,
    calculate_ripley_k,
    plot_ripley_k,
)
from cellpack_analysis.analyses.stochastic_variation_analysis.load_data import (
    get_position_data_from_outputs,
)
from cellpack_analysis.analyses.stochastic_variation_analysis.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

plt.rcParams.update({"font.size": 14})

# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

results_dir = base_results_dir / "stochastic_variation_analysis/baseline"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% set packing modes
baseline_analysis = True
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
# %% set structure ID
STRUCTURE_ID = "SLC25A17"
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
# %% Calculate distance measures
all_distance_dict = get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=mesh_information_dict,
    results_dir=results_dir,
    recalculate=True,
)
# %% Normalize distances
all_distance_dict = normalize_distances(
    all_distance_dict, normalization, mesh_information_dict
)
# %% plot distance PDFs
plot_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% plot distance PDFs overlaid
plot_distance_distributions_overlay(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% plot distance distribution histograms
plot_distance_distributions_histogram(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% [markdown]
# # KS Test Analysis
# %% ks test between observed and other modes
ks_observed_dict = get_ks_observed_dict(
    distance_measures,
    packing_modes,
    all_distance_dict,
    baseline_mode="observed_data",
)
# %% plot ks test results
plot_ks_observed_barplots(
    ks_observed_dict=ks_observed_dict,
    figures_dir=figures_dir,
    suffix=suffix,
)
# %% [markdown]
# # Pairwise EMD Analysis
# %% Get pairwise earth movers distances between distance distributions
all_pairwise_emd = get_pairwise_emd_dictionary(
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
plot_pairwise_emd_heatmaps(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot emd correlation heatmap
corr_df_dict = plot_average_emd_correlation_heatmap(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD heatmaps with variation
plot_emd_correlation_circles(
    distance_measures=distance_measures,
    corr_df_dict=corr_df_dict,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD box plots
plot_emd_boxplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD histograms
plot_emd_histograms(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% plot EMD kdeplots
plot_emd_kdeplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% [markdown]
# # Ripley's K Analysis
# %% calculate ripleyK for all positions
all_ripleyK, mean_ripleyK, ci_ripleyK, r_values = calculate_ripley_k(
    all_positions=all_positions,
    mesh_information_dict=mesh_information_dict,
)
# %% plot ripleyK distributions
plot_ripley_k(
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
kde_dict = get_space_corrected_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    normalization=normalization,
)
# %% plot individual space corrected individual kde values
plot_individual_space_corrected_kde(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=["observed_data", "random", "nucleus_moderate"],
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% get combined space corrected kde
combined_kde_dict = get_combined_space_corrected_kde(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=mesh_information_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% plot combined space corrected kde
plot_combined_space_corrected_kde(
    combined_kde_dict=combined_kde_dict,
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% get EMD between occupied and available distances
emd_occupancy_dict = get_occupancy_emd(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% occupancy emd kdeplot
plot_occupancy_emd_kdeplot(
    emd_occupancy_dict=emd_occupancy_dict,
    packing_modes=packing_modes,
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% create box and whisker plot for occupancy EMD values
plot_occupancy_emd_boxplot(
    emd_occupancy_dict=emd_occupancy_dict,
    figures_dir=occupancy_figdir,
    suffix=suffix,
)
# %% run ks test for occupancy distributions
ks_occupancy_dict = get_occupancy_ks_test_dict(
    distance_dict=all_distance_dict["nucleus"],
    kde_dict=kde_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% plot ks test results
plot_occupancy_ks_test(
    ks_occupancy_dict=ks_occupancy_dict, figures_dir=occupancy_figdir
)

# %%
