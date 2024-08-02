# %% [markdown]
# # Distance analysis workflow
import time
from pathlib import Path

import matplotlib.pyplot as plt

from cellpack_analysis.analysis.stochastic_variation_analysis import distance
from cellpack_analysis.analysis.stochastic_variation_analysis.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

plt.rcParams.update({"font.size": 14})
start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID and radius
STRUCTURE_ID = "RAB5A"  # "SLC25A17" for peroxisomes, "RAB5A" for early endosomes
STRUCT_RADIUS = 2.6  # 2.37 um for peroxisomes, 2.6 um for early endosomes
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
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    "apical_gradient",
]

# relative path to packing outputs
if baseline_analysis:
    packing_output_folder = "packing_outputs/stochastic_variation_analysis/"
    packing_modes = packing_modes_baseline
    baseline_mode = "mean_count_and_size"
else:
    # TODO: update path with new packings
    packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
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

distance_figures_dir = results_dir / "figures/distance"
distance_figures_dir.mkdir(exist_ok=True, parents=True)

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
# ### Get average structure radius
avg_struct_radius, std_struct_radius = distance.get_average_scaled_value(
    value=STRUCT_RADIUS, mesh_information_dict=mesh_information_dict
)
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
# ### plot distance PDFs as kde plots
distance.plot_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    overlay=True,
)
# %% [markdown]
# ### plot distance PDFs overlaid for comparison
distance.plot_distance_distributions_overlay(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% [markdown]
# ### plot distance distribution histograms
distance.plot_distance_distributions_histogram(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
)
# %% [markdown]
# ## KS Test Analysis for distance distributions
# %% [markdown]
# ### Run ks test between observed and other modes
ks_observed_dict = distance.get_ks_observed_dict(
    distance_measures,
    packing_modes,
    all_distance_dict,
    baseline_mode=baseline_mode,
)
# %% [markdown]
# ### plot ks test results as barplots
distance.plot_ks_observed_barplots(
    ks_observed_dict=ks_observed_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
)
# %% [markdown]
# ### plot colorized ks distance distributions
distance.plot_ks_test_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    ks_observed_dict=ks_observed_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    baseline_mode=baseline_mode,
)
# %% [markdown]
# ## Pairwise EMD Analysis for distance distributions
# %% [markdown]
# ### create pairwise emd folders
pairwise_emd_dir = distance_figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Get pairwise earth movers distances between distance distributions
all_pairwise_emd = distance.get_pairwise_emd_dictionary(
    all_distance_dict=all_distance_dict,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% [markdown]
# ### calculate pairwise EMD distances across modes
distance.plot_pairwise_emd_heatmaps(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% [markdown]
# ### get average emd correlation dataframe
corr_df_dict = distance.get_average_emd_correlation(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
)
# %% [markdown]
# ### plot EMD correlation circles
distance.plot_emd_correlation_circles(
    distance_measures=distance_measures,
    corr_df_dict=corr_df_dict,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% [markdown]
# ### plot EMD box plots
distance.plot_emd_boxplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    pairwise_emd_dir=pairwise_emd_dir,
    suffix=suffix,
)

# %%
