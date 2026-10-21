# %% [markdown]
"""
# Occupancy illustration for one cell (Fig 4c)

Occupied PDF (distances from the nucleus of one unbiased packing), available PDF (the cell's
cytoplasm grid points) and their ratio, the occupancy ratio, for peroxisome cell 743916.

Inputs:
- ``data/packing_outputs/8d_sphere_data/norm_weights/random/peroxisome/spheresSST/``
  ``positions_peroxisome_rules_shape_random_743916.json`` (byte-identical to the
  ``rules_shape`` copy)
- ``data/structure_data/SLC25A17/meshes/{nuc,mem}_mesh_743916.obj``
- ``data/structure_data/SLC25A17/grid_distances/{nuc,mem,z}_distances_743916.npy``

Output: ``results/illustrations/Fig4/Fig4c_nucleus_pdf_occupancy_ratio_743916.pdf``
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.mesh_tools import (
    get_distances_from_mesh,
    get_mesh_information_for_shape,
)
from cellpack_analysis.lib.stats import normalize_pdf, pdf_ratio

# %% [markdown]
# ## Frozen parameters (published panel, Sep-2025 run)
STRUCTURE_ID = "SLC25A17"  # peroxisome cell shapes
CELL_ID = "743916"  # illustrated cell
POSITIONS_FILE = (
    "packing_outputs/8d_sphere_data/norm_weights/random/peroxisome/spheresSST/"
    "positions_peroxisome_rules_shape_random_743916.json"
)  # one unbiased packing, relative to data/
INGREDIENT_KEY = "membrane_interior_peroxisome"  # punctum ingredient in the packing
SEED = "0"  # packing seed key
BANDWIDTH = 0.4  # gaussian_kde bandwidth factor (both PDFs)
NUM_POINTS = 250  # evaluation points on [0, max occupied distance]
XLIM = 6  # µm, x-axis limit
RATIO_YLIM = 2  # occupancy-ratio axis limit
FIG_SIZE = (2.5, 2.5)  # in
FONT_SIZE = 6  # published font size
LINE_COLOR = "black"  # published line colour

plt.rcParams["font.size"] = FONT_SIZE

project_root = get_project_root()
base_datadir = project_root / "data"
figures_dir = project_root / "results/illustrations/Fig4"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Occupied and available distances from the nucleus (µm)
with open(base_datadir / POSITIONS_FILE) as f:
    positions = np.array(json.load(f)[SEED][INGREDIENT_KEY])

mesh_info = get_mesh_information_for_shape(CELL_ID, base_datadir, STRUCTURE_ID)
nuc_distances = get_distances_from_mesh(positions, mesh_info["nuc_mesh"], invert=True)
mem_distances = get_distances_from_mesh(positions, mesh_info["mem_mesh"])
occupied = nuc_distances[(nuc_distances > 0) & (mem_distances > 0)]  # cytoplasm only

available = mesh_info["nuc_grid_distances"].flatten() * PIXEL_SIZE_IN_UM
available = available[np.isfinite(available) & (available > 0)]  # cytoplasm points

# %% [markdown]
# ## KDEs, normalized PDFs and their ratio
occupied_kde = gaussian_kde(occupied.astype(np.float32))
available_kde = gaussian_kde(available.astype(np.float32))
occupied_kde.set_bandwidth(BANDWIDTH)
available_kde.set_bandwidth(BANDWIDTH)

xvals = np.linspace(0, occupied.max(), NUM_POINTS)
occupancy_ratio, occupied_pdf, available_pdf = pdf_ratio(
    xvals,
    normalize_pdf(xvals, occupied_kde.evaluate(xvals)),
    normalize_pdf(xvals, available_kde.evaluate(xvals)),
)

# %% [markdown]
# ## Plot Fig 4c
fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=FIG_SIZE, sharex=True)
for ax, yvals, ylabel in zip(
    axs,
    [occupied_pdf, available_pdf, occupancy_ratio],
    ["Occupied \nPDF", "Available \nPDF", "Occupancy ratio"],
    strict=True,
):
    ax.plot(xvals, yvals, color=LINE_COLOR, linewidth=1.5)
    ax.set_xlim([0, XLIM])
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
axs[-1].set_xlabel("Distance from nucleus (µm)")

pdf_ylim = [0, max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])]
axs[0].set_ylim(pdf_ylim)
axs[1].set_ylim(pdf_ylim)
axs[2].set_ylim([0, RATIO_YLIM])
axs[2].axhline(1, color="gray", linestyle="--")

sns.despine(fig=fig)
fig.tight_layout()
fig.savefig(
    figures_dir / f"Fig4c_nucleus_pdf_occupancy_ratio_{CELL_ID}.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
