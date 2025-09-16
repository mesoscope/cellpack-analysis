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

from cellpack_analysis.lib import distance, visualization
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats_functions import normalize_distances

log = logging.getLogger(__name__)

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
PACKING_ID = "peroxisome"
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
    # "apical_gradient": "SLC25A17",
    # "struct_gradient": "SEC61B",
}

# relative path to packing outputs
packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
baseline_mode = STRUCTURE_ID

all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = base_results_dir / f"punctate_analysis/{PACKING_ID}/data"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir.parent / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Distance measures to use
distance_measures = [
    # "nearest",
    # "pairwise",
    "nucleus",
    # "scaled_nucleus",
    "z",
    # "membrane",
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
    structure_name=PACKING_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
    ingredient_key=f"membrane_interior_{STRUCTURE_NAME}",
    recalculate=False,
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
# ### plot distance distribution kde
fig, axs = visualization.plot_distance_distributions_kde(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    overlay=True,
    distance_limits=DISTANCE_LIMITS,
    bandwidth=0.4,
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
_ = visualization.plot_intra_mode_emd(
    df_emd=df_emd,
    distance_measures=distance_measures,
    figures_dir=emd_figures_dir,
    suffix=suffix,
    save_format=save_format,
    baseline_mode=baseline_mode,
    annotate_significance=False,
)
# %% [markdown]
# ### Create plots for baseline comparison EMD
_ = visualization.plot_baseline_mode_emd(
    df_emd=df_emd,
    distance_measures=distance_measures,
    baseline_mode=baseline_mode,
    figures_dir=emd_figures_dir,
    suffix=suffix,
    save_format=save_format,
    annotate_significance=False,
)
# %% [markdown]
# ### Log statistics for EMD comparisons
emd_log_file_path = results_dir / f"{PACKING_ID}_emd_central_tendencies{suffix}.log"
for comparison_type in ["within_rule", "baseline"]:
    distance.log_central_tendencies_for_emd(
        df_emd=df_emd,
        distance_measures=distance_measures,
        packing_modes=packing_modes,
        baseline_mode=baseline_mode,
        log_file_path=emd_log_file_path,
        comparison_type=comparison_type,
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
ks_test_df = distance.get_ks_test_df(
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    all_distance_dict=all_distance_dict,
    baseline_mode=baseline_mode,
    significance_level=ks_significance_level,
    save_dir=results_dir,
    recalculate=False,
)
# %% [markdown]
# ### Bootstrap KS test
df_ks_bootstrap = distance.bootstrap_ks_tests(
    ks_test_df=ks_test_df,
    distance_measures=distance_measures,
    packing_modes=[pm for pm in packing_modes if pm != baseline_mode],
    n_bootstrap=1000,
)
# %% [markdown]
# ### Plot KS observed results
fig_list, ax_list = visualization.plot_ks_observed_barplots(
    df_ks_bootstrap=df_ks_bootstrap,
    distance_measures=distance_measures,
    figures_dir=ks_figures_dir,
    suffix=suffix,
    save_format=save_format,
)
# %% [markdown]
# ### Log statistics for KS test comparisons
ks_log_file_path = results_dir / f"{STRUCTURE_NAME}_ks_test_central_tendencies{suffix}.log"
distance.log_central_tendencies_for_ks(
    df_ks_bootstrap=df_ks_bootstrap,
    distance_measures=distance_measures,
    file_path=ks_log_file_path,
)
