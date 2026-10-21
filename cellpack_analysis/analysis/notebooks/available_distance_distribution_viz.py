# %% [markdown]
"""
# Available-space distance distributions (Fig 4b)

Per-cell KDEs (grey) and the pooled KDE (black) of available-space grid distances to the
nucleus and along z, for the peroxisome and endosome cells pooled (n = 305 + 442 = 747).

Inputs: ``data/structure_data/{SLC25A17,RAB5A}/mesh_information.dat`` (read only; per-cell
grid distances).

Output: ``results/illustrations/Fig4/Fig4b_grid_distance_distribution.pdf``
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.distance import filter_invalid_distances
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import (
    DISTANCE_LIMITS,
    DISTANCE_MEASURE_LABELS,
    GRID_DISTANCE_LABELS,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

# %% [markdown]
# ## Frozen parameters (Feb-2026 published settings)
STRUCTURE_IDS = ["SLC25A17", "RAB5A"]  # pooled: peroxisome + endosome cells
DISTANCE_MEASURES = ["nucleus", "z"]  # panel rows
MINIMUM_DISTANCE = -1  # µm; drop grid distances below this
BANDWIDTH = 0.2  # gaussian_kde bw_method factor
FIG_SIZE = (2.5, 2.5)  # in
FONT_SIZE = 6  # published font size
DISTANCE_YLIMS = {"nucleus": (0, 0.35), "z": (0, 0.25)}  # density axis limits

plt.rcParams["font.size"] = FONT_SIZE

project_root = get_project_root()
base_datadir = project_root / "data"
figures_dir = project_root / "results/illustrations/Fig4"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load mesh information
combined_mesh_information_dict: dict = {
    structure_id: get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    for structure_id in STRUCTURE_IDS
}

# %% [markdown]
# ## Plot Fig 4b
fig, axs = plt.subplots(len(DISTANCE_MEASURES), 1, figsize=FIG_SIZE, dpi=300)
for ax, distance_measure in zip(axs, DISTANCE_MEASURES, strict=True):
    combined_distances: list = []
    for structure_id in STRUCTURE_IDS:
        mesh_information_dict = combined_mesh_information_dict[structure_id]
        for cellid_dict in tqdm(
            mesh_information_dict.values(), desc=f"{structure_id} - {distance_measure}"
        ):
            cell_grid_distances = cellid_dict[GRID_DISTANCE_LABELS[distance_measure]]
            cell_distances_um = filter_invalid_distances(
                cell_grid_distances * PIXEL_SIZE_IN_UM, minimum_distance=MINIMUM_DISTANCE
            )
            combined_distances.extend(cell_distances_um)
            sns.kdeplot(
                cell_distances_um,
                ax=ax,
                color="gray",
                alpha=0.1,
                linewidth=0.1,
                cut=0,
                label="_nolegend_",
                bw_method=BANDWIDTH,
            )

    sns.kdeplot(
        combined_distances,
        ax=ax,
        color="black",
        linewidth=1.5,
        cut=0,
        bw_method=BANDWIDTH,
    )
    ax.set_xlabel(f"{DISTANCE_MEASURE_LABELS[distance_measure]}")
    sns.despine(ax=ax)
    ax.set_ylim(DISTANCE_YLIMS[distance_measure])
    ax.set_xlim(DISTANCE_LIMITS[distance_measure])
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))

fig.tight_layout()
fig.savefig(figures_dir / "Fig4b_grid_distance_distribution.pdf", bbox_inches="tight")

# %%
