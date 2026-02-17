# %% [markdown]
"""
Complete Spatial Randomness (CSR) Test Workflow.

Compares observed cellular structure positions against simulation-based null models
using ECDF-based envelope tests with Monte Carlo p-values and Benjamini-Hochberg
multi-testing correction across cells.

Key steps:
1. Compute ECDFs for multiple distance metrics per cell
2. Build pointwise MC envelopes from simulation replicates
3. Global test via supremum of standardized deviations
4. Joint test across metrics via concatenated standardized curves
5. Aggregate results across cells with BH correction.
"""

import logging
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

from cellpack_analysis.lib import distance
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import DISTANCE_MEASURE_LABELS, MODE_LABELS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import monte_carlo_per_cell, summarize_across_cells
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
STRUCTURE_ID = "RAB5A"
PACKING_ID = "endosome"
STRUCTURE_NAME = "endosome"
CONDITION = "replicate_99"
# %% [markdown]
# ### Set packing modes to analyze
save_format = "pdf"
channel_map = {
    STRUCTURE_ID: STRUCTURE_ID,
    "random": STRUCTURE_ID,
    "nucleus_gradient": STRUCTURE_ID,
    "membrane_gradient": STRUCTURE_ID,
    "apical_gradient": STRUCTURE_ID,
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
    recalculate=True,
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
    num_workers=64,
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

# %%
# Configuration
num_cells = len(next(iter(all_distance_dict.values())).get(baseline_mode, {}))
num_replicates = len(
    next(iter(next(iter(all_distance_dict.values())).get("random", {}).values())).values()
)  # number of simulation replicates per model
alpha = 0.05
simulated_packing_modes = [mode for mode in packing_modes if mode != baseline_mode]
logger.info(
    "Configuration: num_cells=%s, num_replicates=%s, alpha=%s, simulated_packing_modes=%s",
    num_cells,
    num_replicates,
    alpha,
    simulated_packing_modes,
)
# %%
# Initialize data containers
# obs_metrics_all[c] -> dict {metric_name: 1D array of distances}
obs_metrics_all = [{} for _ in range(num_cells)]

# sims_metrics_all[c][model] -> list of R dicts, each {metric_name: 1D array}
sims_metrics_all = [
    {mode: [{} for _ in range(num_replicates)] for mode in simulated_packing_modes}
    for _ in range(num_cells)
]
# %%
for measure, measure_dict in all_distance_dict.items():
    for mode, mode_dict in measure_dict.items():
        for cell_idx, (_cell_id, seed_dict) in enumerate(mode_dict.items()):
            # Observed data
            for seed_idx, (_seed, distances) in enumerate(seed_dict.items()):
                if mode == baseline_mode:
                    obs_metrics_all[cell_idx][measure] = distances
                else:
                    # Simulation data
                    sims_metrics_all[cell_idx][mode][seed_idx][measure] = distances

# %%
# Get monte carlo envelopes per cell
all_results = []
for cell_idx in tqdm(range(num_cells)):
    # Skip cells with missing data
    if not obs_metrics_all[cell_idx] or not all(
        sims_metrics_all[cell_idx][m] for m in simulated_packing_modes
    ):
        logger.warning("Cell index %s has missing data, skipping", cell_idx)
        continue

    cell_result = monte_carlo_per_cell(
        observed_distances=obs_metrics_all[cell_idx],
        simulated_distances_by_mode=sims_metrics_all[cell_idx],
        alpha=alpha,
        distance_measures=distance_measures,
        r_grid_size=150,
        statistic="supremum",
    )
    all_results.append(cell_result)
# %%
# Summarize across all cells
summary = summarize_across_cells(all_results, alpha=alpha)

# Optional: save tables
summary["per_distance_measure_pvals"].to_csv(
    results_dir / "per_distance_measure_pvals.csv", index_label="cell_id"
)
summary["per_distance_measure_qvals"].to_csv(
    results_dir / "per_distance_measure_qvals.csv", index_label="cell_id"
)
summary["joint_pvals"].to_csv(results_dir / "joint_pvals.csv", index_label="cell_id")
summary["joint_qvals"].to_csv(results_dir / "joint_qvals.csv", index_label="cell_id")

# %%
# Plot Monte Carlo envelopes for a specific cell, rule, distance measure
fig, axs = plt.subplots(
    len(distance_measures) // 2, 3, dpi=300, figsize=(7, 7), squeeze=False, sharey=True
)
axs = axs.flatten()
cell_idx = 0
packing_mode = "random"
for ct, distance_measure in enumerate(distance_measures):
    ax = axs[ct]
    fig, ax = plot_envelope_for_cell(
        all_results[cell_idx],
        packing_mode=packing_mode,
        distance_measure=distance_measure,
        title=DISTANCE_MEASURE_LABELS[distance_measure],
        ax=ax,
    )
fig.suptitle(f"Cell {cell_idx} envelopes for {MODE_LABELS[packing_mode]}")
fig.tight_layout()
plt.show()
fig.savefig(
    figures_dir / f"cell_{cell_idx}_envelopes_{packing_mode}{suffix}.{save_format}",
    format=save_format,
)
# %%
# Plot joint test rejections by packing mode (combined over all distance measures)
fig, ax = plot_rejection_bars(
    summary["rejection_joint"], title=f"Joint rejections (q<{alpha}) by packing mode"
)
plt.show()
fig.savefig(
    figures_dir / f"joint_rejections_by_packing_mode{suffix}.{save_format}",
    format=save_format,
)
# %%
# Plot per distance measure rejections by packing mode
fig, ax = plot_rejection_bars(
    summary["rejection_per_distance_measure"], title=f"Per distance measure rejections (q<{alpha})"
)
plt.show()
fig.savefig(
    figures_dir / f"per_distance_measure_rejections_by_packing_mode{suffix}.{save_format}",
    format=save_format,
)

# %%
