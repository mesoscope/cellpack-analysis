# %% [markdown]
"""
# Complete Spatial Randomness (CSR) Test Workflow

Compares observed cellular structure positions against simulation-based null models
using ECDF-based envelope tests with Monte Carlo p-values and Benjamini-Hochberg
multi-testing correction across cells.

Key steps:
1. Compute ECDFs for multiple distance metrics per cell
2. Build pointwise MC envelopes from simulation replicates
3. Global test via supremum of standardized deviations
4. Joint test across metrics via concatenated standardized curves
5. Aggregate results across cells with BH correction
"""

import logging
import time

from cellpack_analysis.lib import distance
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import (
    analyze_cell_from_metrics,
    normalize_distances,
    summarize_across_cells,
)
from cellpack_analysis.lib.visualization import plot_envelope_for_cell, plot_rejection_bars

logger = logging.getLogger(__name__)

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
CONDITION = "replicate"
# %% [markdown]
# ### Set packing modes to analyze
save_format = "pdf"
channel_map = {
    "SLC25A17": "SLC25A17",
    "random": "SLC25A17",
    # "nucleus_gradient": "SLC25A17",
    # "membrane_gradient": "SLC25A17",
    # "apical_gradient": "SLC25A17",
}

# relative path to packing outputs
packing_output_folder = f"packing_outputs/8d_sphere_data/{CONDITION}"
baseline_mode = STRUCTURE_ID

packing_modes = list(channel_map.keys())
all_structures = list(set(channel_map.values()))
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = base_results_dir / f"csr_test/{PACKING_ID}/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Distance measures to use
distance_measures = [
    "nearest",
    "pairwise",
    "nucleus",
    # "scaled_nucleus",
    "z",
    "membrane",
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
    packing_id=PACKING_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    packing_output_folder=packing_output_folder,
    ingredient_key=f"membrane_interior_{STRUCTURE_NAME}",
    recalculate=False,
    drop_random_seed=False,
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
all_distance_dict = distance.get_distance_dictionary_serial(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=True,
    # num_workers=8,
    drop_random_seed=False,
)

all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict, minimum_distance=None
)

all_distance_dict = normalize_distances(
    all_distance_dict=all_distance_dict,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)

# %%
# Configuration
C = 305  # number of cells
models = ["unbiased", "towards_nucleus", "towards_membrane", "towards_apical"]
R = 64  # number of simulation replicates per model
metrics_order = ["dist_nucleus", "dist_membrane", "dist_apical", "dist_basal", "nn", "pair"]

# Initialize data containers
# obs_metrics_all[c] -> dict {metric_name: 1D array of distances}
obs_metrics_all = [{} for _ in range(C)]

# sims_metrics_all[c][model] -> list of R dicts, each {metric_name: 1D array}
sims_metrics_all = [{model: [] for model in models} for _ in range(C)]

# TODO: Populate obs_metrics_all and sims_metrics_all from your data pipeline
# Example structure:
# obs_metrics_all[0] = {
#     'dist_nucleus': np.array([...]),
#     'dist_membrane': np.array([...]),
#     'nn': np.array([...]),
#     ...
# }
# sims_metrics_all[0]['unbiased'] = [
#     {'dist_nucleus': np.array([...]), ...},  # replicate 1
#     {'dist_nucleus': np.array([...]), ...},  # replicate 2
#     ...
# ]

# ---- Run per-cell analysis ----
all_results = []
for c in range(C):
    # Skip cells with missing data
    if not obs_metrics_all[c] or not all(sims_metrics_all[c][m] for m in models):
        print(f"Warning: Cell {c} has missing data, skipping")
        continue

    res_c = analyze_cell_from_metrics(
        obs_metrics=obs_metrics_all[c],
        sims_metrics_per_model=sims_metrics_all[c],
        alpha=0.05,
        metrics_order=metrics_order,
        r_grid_size=150,
    )
    all_results.append(res_c)

    if (c + 1) % 50 == 0:
        print(f"Processed {c + 1}/{C} cells")

# ---- Summarize across 305 cells ----
summary = summarize_across_cells(all_results, alpha=0.05)

# Optional: save tables
summary["per_metric_pvals"].to_csv("per_metric_pvals.csv", index_label="cell_id")
summary["per_metric_qvals"].to_csv("per_metric_qvals.csv", index_label="cell_id")
summary["joint_pvals"].to_csv("joint_pvals.csv", index_label="cell_id")
summary["joint_qvals"].to_csv("joint_qvals.csv", index_label="cell_id")

# ---- Quick visualizations ----
# 1) For a specific cell, model, metric
plot_envelope_for_cell(all_results[0], model="unbiased", metric="nn", title="Cell 0: NN vs CSR")

# 2) Rejection bars (joint)
plot_rejection_bars(summary["rejection_joint"], title="Joint rejections (q<0.05) by model")

# 3) Rejection bars per-metric
plot_rejection_bars(summary["rejection_per_metric"], title="Per-metric rejections (q<0.05)")
