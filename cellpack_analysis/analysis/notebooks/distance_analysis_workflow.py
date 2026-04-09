# %% [markdown]
"""
# Distance analysis workflow.

Compare distributions of various measures of distance:
1. Nearest neighbor distance
2. Pairwise distance
3. Distance to nucleus
4. Distance to membrane
5. Distance from basal surface (z-distance)

Can be used to compare distributions in the presence or absence of other influencing structures.
Uses a KDE based approach for visualization and statistical comparison of distance distributions.

Workflow steps:
1. Calculate distance distributions for each distance measure and packing mode.
2. Visualize distance distribution histograms for each distance measure and mode.
3. Calculate and visualize Earth Mover's Distance (EMD) between distance distributions of different
modes.
4. Perform pairwise Monte Carlo Envelope Tests to compare distance distributions between modes.
5. Plot pairwise envelope test results in a matrix format to identify significant differences
between modes for each distance measure.
6. Perform pairwise KS tests to compare distance distributions between modes, and bootstrap results
to get confidence intervals.
"""

import logging
import time

import pandas as pd

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
# SLC25A17: peroxisomes
# RAB5A: early endosomes
# SEC61B: ER
# ST6GAL1: Golgi
PUNCTATE_STRUCTURE_ID = "SLC25A17"
"""This is the ID for the packed structure, it is used as the packing mode for observed data"""

CELL_STRUCTURE_ID = "SLC25A17"
"""This is the ID for the cell shapes used for packing"""

PACKING_ID = "peroxisome"
"""This is the ID for the overall packing configuration,
it is used for naming outputs and folders"""

STRUCTURE_NAME = "peroxisome"
"""This is the name of the structure being analyzed, it is used in cellPACK output files"""

CONDITION = "rules_shape_with_seed"
"""This is the experimental condition or packing output subfolder to analyze"""

RESULTS_SUBFOLDER = f"{CONDITION}/{PACKING_ID}"
"""Subfolder within results/ to save outputs for this workflow."""

FIGURE_SUBFOLDER = "figures/test"
"""Subfolder within results subfolder to save figures for this workflow."""
# %% [markdown]
# ### Set packing modes to analyze
save_format = "pdf"

channel_map = {
    PUNCTATE_STRUCTURE_ID: PUNCTATE_STRUCTURE_ID,
    "random": CELL_STRUCTURE_ID,
    "nucleus_gradient": CELL_STRUCTURE_ID,
    "membrane_gradient": CELL_STRUCTURE_ID,
    "apical_gradient": CELL_STRUCTURE_ID,
    # "struct_gradient": CELL_STRUCTURE_ID,
}

# relative path to packing outputs
packing_output_folder = f"packing_outputs/8d_sphere_data/{CONDITION}/"
baseline_mode = PUNCTATE_STRUCTURE_ID

all_structures = list(set(channel_map.values()))
packing_modes = list(channel_map.keys())
# %% [markdown]
# ### Set file paths and setup parameters
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = make_dir(base_results_dir / RESULTS_SUBFOLDER)

figures_dir = make_dir(results_dir / FIGURE_SUBFOLDER)

log_dir = make_dir(results_dir / "logs")
# %% [markdown]
# ### Distance measures to use
distance_measures = [
    # "nearest",
    # "pairwise",
    "nucleus",
    # "scaled_nucleus",
    # "scaled_z",
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
    structure_id=PUNCTATE_STRUCTURE_ID,
    packing_id=PACKING_ID,
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
all_distance_dict_raw = distance.get_distance_dictionary(
    all_positions=all_positions,
    distance_measures=distance_measures,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    results_dir=results_dir,
    recalculate=False,
    num_workers=8,
)

all_distance_dict_filtered = distance.filter_invalids_from_distance_distribution_dict(
    distance_distribution_dict=all_distance_dict_raw, minimum_distance=-1
)

all_distance_dict = distance.normalize_distance_dictionary(
    all_distance_dict=all_distance_dict_filtered,
    mesh_information_dict=combined_mesh_information_dict,
    channel_map=channel_map,
    normalization=normalization,
)

# %% [markdown]
# ## Distance distributions
distance_figures_dir = make_dir(figures_dir / "distance_distributions/")
# %% [markdown]
# ### compute distance PDFs
distance_pdf_dict = distance.compute_distance_pdfs(
    all_distance_dict=all_distance_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    method="kde",
    bin_width=0.01,
    bandwidth=0.2,
    distance_limits=DISTANCE_LIMITS,
    minimum_distance=-1,
    # n_grid=1000,
    results_dir=results_dir,
    recalculate=False,
)
# %% [markdown]
# ### plot distance distributions
fig, axs = visualization.plot_distance_distributions(
    distance_pdf_dict=distance_pdf_dict,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    figures_dir=distance_figures_dir,
    suffix=suffix,
    normalization=normalization,
    save_format=save_format,
    figure_size=(2.7, 3.4),
    # production_mode=True,
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
# ### Create plots for pairwise EMD matrix
for dm in distance_measures:
    fig, axs = visualization.plot_pairwise_emd_matrix(
        df_emd=df_emd,
        distance_pdf_dict=distance_pdf_dict,
        packing_modes=packing_modes,
        distance_measure=dm,
        normalization=normalization,
        figure_size=(6, 4.5),
        figures_dir=emd_figures_dir,
        suffix=suffix,
        # font_scale=0.6,
        save_format=save_format,
    )
# %% [markdown]
# ### Log pairwise EMD central tendencies
emd_pairwise_log_file_path = log_dir / f"{PACKING_ID}_emd_pairwise_central_tendencies{suffix}.log"
distance.log_pairwise_emd_central_tendencies(
    df_emd=df_emd,
    distance_measures=distance_measures,
    packing_modes=packing_modes,
    log_file_path=emd_pairwise_log_file_path,
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
    bin_width=0.2,
    statistic="intdev",
)
# %% [markdown]
# ### Plot pairwise envelope matrix
envelope_figures_dir = make_dir(figures_dir / "envelope_tests/")
# %%
# for dm in [*distance_measures, None]:
for dm in [None]:
    fig, axs = visualization.plot_pairwise_envelope_matrix(
        pairwise_results=pairwise_results,
        distance_measure=dm,
        figures_dir=envelope_figures_dir,
        suffix=suffix,
        save_format=save_format,
        figure_size=(7, 3.5),
        font_scale=1.1,
    )
# %% [markdown]
# ### Plot rejection bars
for joint_test in [False, True]:
    fig, axs = visualization.plot_per_dm_rejection_bars(
        pairwise_results=pairwise_results,
        test_mode=baseline_mode,
        joint_test=joint_test,
        figures_dir=envelope_figures_dir,
        figsize=(4, 1.8),
        font_scale=1.1,
        suffix=suffix,
        save_format=save_format,
    )
# %% [markdown]
# ### Per distance measure envelope overlays
fig, axs = visualization.plot_per_dm_envelopes_overlaid(
    pairwise_results=pairwise_results,
    figures_dir=envelope_figures_dir,
    suffix=suffix,
    figsize=(3, 1.5),
    save_format=save_format,
)
# %% [markdown]
# ## Pairwise KS Test Analysis
# Compare each mode against every other mode (not just the baseline).
# %% [markdown]
# ### Run pairwise KS tests across all mode pairs
# %%
ks_significance_level = 0.05
pairwise_ks_figures_dir = figures_dir / "pairwise_ks_test"
pairwise_ks_figures_dir.mkdir(exist_ok=True, parents=True)

pairwise_ks_dfs: list[pd.DataFrame] = []
for ref_mode in packing_modes:
    other_modes = [m for m in packing_modes if m != ref_mode]
    ks_df = distance.get_ks_test_df(
        distance_measures=distance_measures,
        packing_modes=packing_modes,
        all_distance_dict=all_distance_dict,
        baseline_mode=ref_mode,
        significance_level=ks_significance_level,
        save_dir=None,
        recalculate=False,
    )
    ks_df["baseline_mode"] = ref_mode
    pairwise_ks_dfs.append(ks_df)

pairwise_ks_test_df = pd.concat(pairwise_ks_dfs, ignore_index=True)
# %% [markdown]
# ### Bootstrap pairwise KS tests
# %%
pairwise_ks_bootstrap_dfs: list[pd.DataFrame] = []
for ref_mode in packing_modes:
    ref_ks_df = pairwise_ks_test_df.query("baseline_mode == @ref_mode")
    other_modes = [m for m in packing_modes if m != ref_mode]
    df_boot = distance.bootstrap_ks_tests(
        ks_test_df=ref_ks_df,
        distance_measures=distance_measures,
        packing_modes=other_modes,
        n_bootstrap=1000,
    )
    df_boot["baseline_mode"] = ref_mode
    pairwise_ks_bootstrap_dfs.append(df_boot)

pairwise_ks_bootstrap_df = pd.concat(pairwise_ks_bootstrap_dfs, ignore_index=True)
# %% [markdown]
# ### Plot pairwise KS results per baseline mode
# %%
for ref_mode in packing_modes:
    ref_boot_df = pairwise_ks_bootstrap_df.query("baseline_mode == @ref_mode")
    fig_list, ax_list = visualization.plot_ks_test_results(
        df_ks_bootstrap=ref_boot_df,
        distance_measures=distance_measures,
        figures_dir=pairwise_ks_figures_dir,
        suffix=f"{suffix}_vs_{ref_mode}",
        save_format=save_format,
    )
# %% [markdown]
# ### Log pairwise KS central tendencies
pairwise_ks_log_file_path = log_dir / f"{STRUCTURE_NAME}_pairwise_ks_central_tendencies{suffix}.log"
for ref_mode in packing_modes:  # noqa:B007
    ref_boot_df = pairwise_ks_bootstrap_df.query("baseline_mode == @ref_mode")
    distance.log_central_tendencies_for_ks(
        df_ks_bootstrap=ref_boot_df,
        distance_measures=distance_measures,
        file_path=pairwise_ks_log_file_path,
    )
# %% [markdown]
logger.info(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
