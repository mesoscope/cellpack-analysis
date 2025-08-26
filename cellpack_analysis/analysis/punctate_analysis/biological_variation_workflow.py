# %% [markdown]
# # Biological variation workflow
# Compare distributions of various measures of distance using:
# 1. Pairwise EMD
# 2. KS test
# 3. Ripley's K
#
# Used to compare variation due to
# 1. Mean count and size
# 2. Variable size
# 3. Variable count
# 4. Shape

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
STRUCTURE_ID = "SLC25A17"
STRUCTURE_NAME = "peroxisome"
STRUCT_RADIUS = 2.37  # 2.37 um for peroxisomes
# %% [markdown]
# ### Set packing modes to analyze
save_format = "svg"
packing_modes = [
    "mean_count_and_size",
    "variable_count",
    "variable_size",
    "shape",
]

channel_map = {
    "mean_count_and_size": "mean",
    "variable_count": "mean",
    "variable_size": "mean",
    "shape": "SLC25A17",
}

packing_output_folder = "packing_outputs/stochastic_variation_analysis/"
baseline_mode = "mean_count_and_size"

all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

folder_name = "baseline"
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
