# %% [markdown]
# # Distance analysis workflow
# Compare distributions of various measures of distance using:
# 1. Pairwise EMD
# 2. KS test
# 3. Ripley's K
#
# Can be used to compare distirbutions in the presence or absence
# of other influencing structures
# Current structure pairs include:
# * Peroxisomes (SLC25A17) and Endoplasmic Reticulum (SEC61B)
# * Endosomes (RAB5A) and Golgi (ST6GAL1)
import logging
import time

import matplotlib.pyplot as plt

from cellpack_analysis.analysis.punctate_analysis.lib import distance, visualization
from cellpack_analysis.analysis.punctate_analysis.lib.stats_functions import (
    normalize_distances,
)
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
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
    # "RAB5A": "RAB5A",
    "random": "SLC25A17",
    "nucleus_gradient_strong": "SLC25A17",
    "membrane_gradient_strong": "SLC25A17",
    # "struct_gradient": "SEC61B",
}

packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
baseline_mode = STRUCTURE_ID

all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

folder_name = "rules"
results_dir = base_results_dir / f"punctate_analysis/{STRUCTURE_NAME}/{folder_name}"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Distance measures to use
distance_measures = [
    "nearest",
    "pairwise",
    "nucleus",
    "scaled_nucleus",
    # "z",
]
# %% [markdown]
# ### Set normalization
# options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = None
suffix = ""
if normalization is not None:
    suffix = f"_normalized_{normalization}"

# %% [markdown]
# ### Read position data from outputs
all_positions = get_position_data_from_outputs(
    structure_id=STRUCTURE_ID,
    structure_name=STRUCTURE_NAME,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
    recalculate=False,
)
# %% [markdown]
# ### Get mesh information
combined_mesh_information_dict = {}
for structure in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure,
        base_datadir=base_datadir,
        recalculate=False,
    )
    combined_mesh_information_dict[structure] = mesh_information_dict
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
# ## Distance distributions
distance_figures_dir = figures_dir / "distance_distributions"
distance_figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### plot distance distributions as vertical kde
fig_list, ax_list = visualization.plot_distance_distributions_kde_vertical(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    suffix=suffix,
    normalization=normalization,
    overlay=True,
    distance_limits=DISTANCE_LIMITS,
    figures_dir=distance_figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ## EMD Analysis for distance distributions
# %% [markdown]
# ### create emd analysis folders
emd_figures_dir = figures_dir / "emd"
emd_figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Get earth movers distances between distance distributions
all_pairwise_emd = distance.get_distance_distribution_emd_dictionary(
    all_distance_dict=all_distance_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %%
visualization.plot_emd_kdeplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    suffix=suffix,
    emd_figures_dir=emd_figures_dir,
    save_format=save_format,
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
visualization.plot_emd_correlation_circles(
    distance_measures=distance_measures,
    corr_df_dict=corr_df_dict,
    suffix=suffix,
    figures_dir=emd_figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ### plot EMD barplots
visualization.plot_emd_barplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    suffix=suffix,
    figures_dir=emd_figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ### plot EMD violinplots
visualization.plot_emd_violinplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    suffix=suffix,
    figures_dir=emd_figures_dir,
    save_format=save_format,
)
# %% [markdown]
# ## KS Test Analysis for distance distributions
# %% [markdown]
# ### create KS test  folders
ks_figures_dir = figures_dir / "ks_test"
ks_figures_dir.mkdir(exist_ok=True, parents=True)
ks_significance_level = 0.05
# %% [markdown]
# ### Run KS test between observed and other modes
ks_observed_combined_df = distance.get_ks_observed_combined_df(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    baseline_mode=baseline_mode,
    significance_level=ks_significance_level,
    save_dir=results_dir,
    recalculate=False,
)
# %% [markdown]
# ### Melt ks observed dataframe for plotting
distance_measures_to_plot = ["nucleus", "scaled_nucleus"]
df_plot = ks_observed_combined_df.loc[
    ks_observed_combined_df["distance_measure"].isin(distance_measures_to_plot)
]
df_melt = distance.melt_df_for_plotting(df_plot)

# %% [markdown]
# ### Plot KS observed results
fig, ax = visualization.plot_ks_observed_barplots(
    df_melt=df_melt,
    figures_dir=ks_figures_dir,
    suffix=suffix,
    significance_level=ks_significance_level,
    save_format=save_format,
)

# %% [markdown]
# ## Ripley's K Analysis
# %% [markdown]
# ### create ripley analysis folders
ripley_figures_dir = figures_dir / "ripley"
ripley_figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ## calculate ripleyK for all positions
all_ripleyK, mean_ripleyK, ci_ripleyK, r_values = distance.calculate_ripley_k(
    all_positions=all_positions,
    mesh_information_dict=combined_mesh_information_dict[baseline_mode],
)
# %% [markdown]
# ## plot ripleyK distributions
visualization.plot_ripley_k(
    mean_ripleyK=mean_ripleyK,
    ci_ripleyK=ci_ripleyK,
    r_values=r_values,
    figures_dir=ripley_figures_dir,
    save_format=save_format,
)
