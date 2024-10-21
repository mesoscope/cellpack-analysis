# %% [markdown]
# # Distance analysis workflow
# Used to compare distance distributions for punctate structures packed in the presence
# or absence of other influencing structures
# Current structure pairs include:
# * Peroxisomes (SLC25A17) and Endoplasmic Reticulum (SEC61B)
# * Endosomes (RAB5A) and Golgi (ST6GAL1)
import logging
import time

import matplotlib.pyplot as plt

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
STRUCTURE_NAME = "peroxisome"
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
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    # "apical_gradient",
    "struct_gradient",
]

channel_map = {
    "SLC25A17": "SLC25A17",
    "random": "SLC25A17",
    "nucleus_gradient_strong": "SLC25A17",
    "membrane_gradient_strong": "SLC25A17",
    # "struct_gradient": "SEC61B",
}
all_structures = list(set(channel_map.values()))
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
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

folder_name = "baseline" if baseline_analysis else "rules"
results_dir = base_results_dir / f"punctate_analysis/{STRUCTURE_NAME}/{folder_name}"
results_dir.mkdir(exist_ok=True, parents=True)

distance_figures_dir = results_dir / "figures/distance"
distance_figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Distance measures to use
distance_measures = [
    "pairwise",
    "nearest",
    "nucleus",
    # "scaled_nucleus",
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
    recalculate=True,
    baseline_analysis=baseline_analysis,
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
# ### plot distance distribution histograms
distance.plot_distance_distributions_histogram(
    distance_measures,
    packing_modes,
    all_distance_dict,
    suffix=suffix,
    normalization=normalization,
    figures_dir=distance_figures_dir,
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
    distance_limits={
        "pairwise": (-2, 25),
        "nucleus": (-1, 8),
        "nearest": (-0.2, 5),
        "z": (-2, 12),
        "scaled_nucleus": (-0.2, 1.2),
        "membrane": (-0.4, 3.2),
    },
)

# %% [markdown]
# ### plot distance distributions as vertical kde
plt.rcParams.update({"font.size": 10})
fig_list, ax_list = distance.plot_distance_distributions_kde_vertical(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    suffix=suffix,
    normalization=normalization,
    overlay=True,
    figures_dir=distance_figures_dir,
    distance_limits={
        "pairwise": (-2, 25),
        "nucleus": (-1, 8),
        "nearest": (-0.2, 5),
        "z": (-2, 12),
        "scaled_nucleus": (-0.2, 1.2),
        "membrane": (-0.4, 3.2),
    },
)

# %% [markdown]
# ## Pairwise EMD Analysis for distance distributions
# %% [markdown]
# ### create pairwise emd folders
pairwise_emd_dir = distance_figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Get pairwise earth movers distances between distance distributions
all_pairwise_emd = distance.get_distance_distribution_emd_dictionary(
    all_distance_dict=all_distance_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    results_dir=results_dir,
    recalculate=True,
    suffix=suffix,
)
# %% [markdown]
# ### calculate pairwise EMD distances across modes
distance.plot_emd_heatmaps(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    figures_dir=pairwise_emd_dir,
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
    figures_dir=pairwise_emd_dir,
    suffix=suffix,
)
# %% [markdown]
# ### plot EMD box plots
distance.plot_emd_boxplots(
    distance_measures=distance_measures,
    all_pairwise_emd=all_pairwise_emd,
    baseline_mode=baseline_mode,
    figures_dir=pairwise_emd_dir,
    suffix=suffix,
)

# %%
