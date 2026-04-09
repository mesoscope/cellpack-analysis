# %% [markdown]
"""
# Biological variation workflow
This notebook compares variation in spatial organization due to biological factors.
Uses a KDE based approach for visualization and statistical comparison of distance distributions,
as well as pairwise KS tests and Monte Carlo Envelope tests.

Factors affecting spatial organization:
1. Size variation
2. Count variation
3. Variation in cell and nucleus shape

## Workflow steps:
1. Calculate distance distributions for each mode (size, count, shape) and a baseline mode.
2. Visualize distance distribution histograms for each distance measure and mode.
3. Calculate and visualize Earth Mover's Distance (EMD) between distance distributions of different
modes.
4. Perform pairwise Monte Carlo Envelope Tests to compare distance distributions between modes.
5. Plot pairwise envelope test results in a matrix format to identify significant differences
between modes for each distance measure.
"""

import logging
import time

from cellpack_analysis.lib import distance, visualization
from cellpack_analysis.lib.file_io import get_project_root, make_dir
from cellpack_analysis.lib.label_tables import DISTANCE_LIMITS
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import pairwise_envelope_test

logger = logging.getLogger(__name__)

start_time = time.time()
# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### Set structure ID and radius
STRUCTURE_ID = "SLC25A17"
"""This is the ID for the packed structure, it is used to get the cell containers for the shape
variation mode."""

PACKING_ID = "peroxisome"
"""This is the ID for the overall packing configuration, it is used for naming outputs
and folders."""

STRUCTURE_NAME = "peroxisome"
"""This is the name of the structure being analyzed, it is used in cellPACK output files."""

RESULT_SUBFOLDER = "biological_variation"
"""Subfolder within results/ to save outputs for this workflow."""
# %% [markdown]
# ### Set packing modes to analyze
save_format = "pdf"

channel_map = {
    "baseline": "mean",
    "variable_count": "mean",
    "variable_size": "mean",
    "shape": STRUCTURE_ID,
}

# packing_output_folder = "packing_outputs/stochastic_variation_analysis/"
packing_output_folder = "packing_outputs/biological_variation/"
baseline_mode = "baseline"

all_structures = list(set(channel_map.values()))
packing_modes = list(channel_map.keys())
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = make_dir(base_results_dir / RESULT_SUBFOLDER)
figures_dir = make_dir(results_dir / "figures")
log_dir = make_dir(results_dir / "logs")

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
all_distance_dict_raw = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
)
all_distance_dict_filtered = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict_raw, minimum_distance=None
)

all_distance_dict = distance.normalize_distance_dictionary(
    all_distance_dict=all_distance_dict_filtered,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)

# %% [markdown]
# ## Distance distributions
distance_figures_dir = make_dir(figures_dir / "distance_distributions")
# %% [markdown]
# ### compute distance PDFs
distance_pdf_dict = distance.compute_distance_pdfs(
    all_distance_dict=all_distance_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    method="kde",
    bin_width=0.2,
    bandwidth=0.4,
    distance_limits=DISTANCE_LIMITS,
    minimum_distance=0,
)
# %% [markdown]
# ### plot distance distribution histograms
fig, axs = visualization.plot_distance_distributions(
    distance_pdf_dict=distance_pdf_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    save_format=save_format,
    production_mode=True,
)
# %% [markdown]
# ### log central tendencies for distance distributions
log_file_path = log_dir / f"{STRUCTURE_NAME}_distance_distribution_central_tendencies{suffix}.log"
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
emd_figures_dir = make_dir(figures_dir / "emd")
# %% [markdown]
# ### Get earth movers distances between distance distributions
df_emd = distance.get_distance_distribution_emd_df(
    all_distance_dict=all_distance_dict,
    packing_modes=packing_modes,
    distance_measures=distance_measures,
    results_dir=results_dir,
    recalculate=False,
    suffix=suffix,
    num_workers=8,
)
# %% [markdown]
# ### Create plots for EMD comparisons
for comparison_type in ["intra_mode", "baseline"]:
    fig_bar, axs_bar, fig_violin, axs_violin = visualization.plot_emd_comparisons(
        df_emd=df_emd,
        distance_measures=distance_measures,
        comparison_type=comparison_type,  # type: ignore
        baseline_mode=baseline_mode,
        figures_dir=emd_figures_dir,
        suffix=suffix,
        save_format=save_format,
        annotate_significance=False,
    )
# %% [markdown]
# ### Plot pairwise EMD matrix for each distance measure
for dm in distance_measures:
    fig, axs = visualization.plot_pairwise_emd_matrix(
        df_emd=df_emd,
        distance_pdf_dict=distance_pdf_dict,
        packing_modes=packing_modes,
        distance_measure=dm,
        normalization=normalization,
        figures_dir=emd_figures_dir,
        suffix=suffix,
        figure_size=(4.5, 3),
        font_scale=0.8,
        save_format=save_format,
    )
# %% [markdown]
# ### Log statistics for EMD comparisons
emd_log_file_path = log_dir / f"{STRUCTURE_NAME}_emd_pairwise_central_tendencies{suffix}.log"
distance.log_pairwise_emd_central_tendencies(
    df_emd=df_emd,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    log_file_path=emd_log_file_path,
)

# %% [markdown]
# ## Pairwise Monte Carlo Envelope Test
# Compare each mode against every other mode's envelope.
# Comparisons are asymmetric: row = test mode, column = reference (envelope source).
# %% [markdown]
# ### Run pairwise envelope test
# %%
pairwise_results = pairwise_envelope_test(
    all_distance_dict=all_distance_dict,
    packing_modes=packing_modes,
    distance_measures=distance_measures,
    alpha=0.05,
    bin_width=0.1,
    statistic="intdev",
)
# %% [markdown]
# ### Plot pairwise envelope matrix per distance measure
envelope_figures_dir = make_dir(figures_dir / "envelope_tests/")
# %%
for dm in distance_measures:
    fig, axs = visualization.plot_pairwise_envelope_matrix(
        pairwise_results=pairwise_results,
        distance_measure=dm,
        figures_dir=envelope_figures_dir,
        suffix=suffix,
        save_format=save_format,
        figure_size=(3.5, 2.5),
        font_scale=0.8,
    )
    # break
# %% [markdown]
# ### Plot pairwise envelope matrix - joint test
# %%
fig, axs = visualization.plot_pairwise_envelope_matrix(
    pairwise_results=pairwise_results,
    distance_measure=None,
    figures_dir=envelope_figures_dir,
    suffix=suffix,
    save_format=save_format,
    figure_size=(7, 3.5),
    font_scale=1.1,
)
# %% [markdown]
# ### Per distance measure rejection bars (per test mode)
# %%
rej_dict = {}
for test_mode in packing_modes:
    rej_dict[test_mode] = visualization.plot_per_dm_rejection_bars(
        pairwise_results=pairwise_results,
        test_mode=test_mode,
        joint_test=True,
        figures_dir=envelope_figures_dir,
        figsize=(3.5, 2),
        suffix=suffix,
        save_format=save_format,
    )
    break
# %% [markdown]
# ### Per distance measure envelope overlays
# %%
fig, axs = visualization.plot_per_dm_envelopes_overlaid(
    pairwise_results=pairwise_results,
    figures_dir=envelope_figures_dir,
    suffix=suffix,
    figsize=(6, 1.5),
    save_format=save_format,
)
# %%
logger.info(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")

# %%
