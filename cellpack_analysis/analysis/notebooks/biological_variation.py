# %% [markdown]
# # Biological variation workflow
# Compare variation in spatial organization due to biological factors

# Factors affecting spatial organization:
# 1. Size variation
# 2. Count variation
# 3. Variation in cell and nucleus shape

import logging
import time

from cellpack_analysis.lib import distance, visualization
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

logger = logging.getLogger(__name__)

start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID and radius
STRUCTURE_ID = "SLC25A17"
PACKING_ID = "peroxisome"
STRUCTURE_NAME = "peroxisome"
# %% [markdown]
# ### Set packing modes to analyze
save_format = "pdf"

channel_map = {
    "mean_count_and_size": "mean",
    "variable_count": "mean",
    "variable_size": "mean",
    "shape": "SLC25A17",
}

packing_output_folder = "packing_outputs/stochastic_variation_analysis/"
baseline_mode = "mean_count_and_size"

all_structures = list(set(channel_map.values()))
packing_modes = list(channel_map.keys())
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = base_results_dir / f"biological_variation/{STRUCTURE_NAME}/update_emd/"
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
    "z",
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
    packing_id=STRUCTURE_NAME,
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
    recalculate=True,
)

all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict, minimum_distance=None
)

all_distance_dict = distance.normalize_distance_dictionary(
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
# ### plot distance distribution kde
fig, axs = visualization.plot_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    distance_limits=DISTANCE_LIMITS,
    bandwidth=0.2,
    save_format=save_format,
)
# %% [markdown]
# ### log central tendencies for distance distributions
log_file_path = (
    results_dir / f"{STRUCTURE_NAME}_distance_distribution_central_tendencies{suffix}.log"
)
distance.log_central_tendencies_for_distance_distributions(
    all_distance_dict=all_distance_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    file_path=log_file_path,
)
# %% [markdown]
# ## EMD Analysis for distance distributions
# %% [markdown]
# ### create emd analysis folders
emd_figures_dir = figures_dir / "emd"
emd_figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Get earth movers distances between distance distributions
df_emd = distance.get_distance_distribution_emd_df(
    all_distance_dict=all_distance_dict,
    packing_modes=packing_modes,
    distance_measures=distance_measures,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
)
# %% [markdown]
# ### Create plots for within rule EMD
for comparison_type in ["intra_mode", "baseline"]:
    fig_bar, axs_bar, fig_violin, axs_violin = visualization.plot_emd_comparisons(
        df_emd=df_emd,
        distance_measures=distance_measures,
        baseline_mode=baseline_mode,
        comparison_type=comparison_type,  # type: ignore
        figures_dir=emd_figures_dir,
        suffix=suffix,
        save_format=save_format,
        annotate_significance=False,
    )
# %% [markdown]
# ### Log statistics for EMD comparisons
for comparison_type in ["intra_mode", "baseline"]:
    emd_log_file_path = (
        results_dir / f"{STRUCTURE_NAME}_emd_central_tendencies_{comparison_type}{suffix}.log"
    )
    distance.log_central_tendencies_for_emd(
        df_emd=df_emd,
        distance_measures=distance_measures,
        packing_modes=packing_modes,
        baseline_mode=baseline_mode,
        log_file_path=emd_log_file_path,
        comparison_type=comparison_type,
    )

# %%
