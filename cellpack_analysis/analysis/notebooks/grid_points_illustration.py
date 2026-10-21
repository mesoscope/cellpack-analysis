# %% [markdown]
"""
# Grid-point distance and weight maps (Fig 4a, Fig 5a)

Illustration for one peroxisome cell: a uniform grid inside the cell is sliced through its
middle, and each cytoplasm point is coloured by its distance from the nucleus (Fig 4a) or by its
rule weight (Fig 5a). The rule weights and the mixture are illustration values, not the fitted
model.

Inputs: ``data/structure_data/SLC25A17/meshes/{nuc,mem}_mesh_834350.obj``.

Outputs:
- ``results/illustrations/Fig4/Fig4a_nucleus_distance_xy_834350.pdf``
- ``results/illustrations/Fig5/Fig5a_<rule>_weights_<xy|xz>_834350.pdf`` for rules
  ``rnd`` (unbiased), ``nuc``, ``mem``, ``z`` (apical) and ``mixed``
"""

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LinearSegmentedColormap

from cellpack_analysis.lib.default_values import PIXEL_SIZE_IN_UM
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import PROJECTION_TO_INDEX_MAP
from cellpack_analysis.lib.mesh_tools import (
    get_distances_from_mesh,
    get_grid_points_slice,
    get_inside_outside_check,
    get_list_of_grid_points,
    get_weights_from_distances,
    round_away_from_zero,
)
from cellpack_analysis.lib.visualization import plot_grid_points_slice

# %% [markdown]
# ## Frozen parameters (illustration values, not fitted)
STRUCTURE_ID = "SLC25A17"  # peroxisome
CELL_ID = "834350"  # illustrated cell
SPACING = 2  # grid spacing (px)
DECAY_LENGTH = 0.1  # illustration λ for every rule (not Table 3)
# illustration mixture (not the fitted coefficients)
MIXED_WEIGHT_FRACTION = {"rnd": 0, "nuc": 0.33, "mem": 0.34, "z": 0.33}
DOT_SIZE = 6  # scatter marker size
FONT_SIZE = 10  # font size
# Fig 4a style (Nov-2025, commit 800a9c9): cyan-to-magenta map, grey nucleus
FIG4A_CMAP = LinearSegmentedColormap.from_list("cyan_to_magenta", ["cyan", "magenta"])
FIG4A_NUCLEUS_COLOR = "gray"
# projection axis -> slice plane name
PLANE_NAMES = {"z": "xy", "y": "xz"}

plt.rcParams["font.size"] = FONT_SIZE

project_root = get_project_root()
mesh_folder = project_root / f"data/structure_data/{STRUCTURE_ID}/meshes"
fig4_dir = project_root / "results/illustrations/Fig4"
fig5_dir = project_root / "results/illustrations/Fig5"
fig4_dir.mkdir(exist_ok=True, parents=True)
fig5_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load meshes and build the grid
nuc_mesh = trimesh.load_mesh(mesh_folder / f"nuc_mesh_{CELL_ID}.obj")
mem_mesh = trimesh.load_mesh(mesh_folder / f"mem_mesh_{CELL_ID}.obj")
bounding_box = round_away_from_zero(mem_mesh.bounds)
all_grid_points = get_list_of_grid_points(bounding_box, SPACING)

# %% [markdown]
# ## Distances and weights on each slice
grid_map: dict = {}
for projection_axis in PLANE_NAMES:
    grid_points_slice = get_grid_points_slice(
        all_grid_points=all_grid_points, projection_axis=projection_axis, spacing=SPACING
    )
    inside_nuc, _, inside_mem_outside_nuc = get_inside_outside_check(
        nuc_mesh=nuc_mesh, mem_mesh=mem_mesh, grid_points_slice=grid_points_slice
    )
    cyto_points = grid_points_slice[inside_mem_outside_nuc]
    distances = {
        "rnd": np.ones(len(cyto_points)),
        "nuc": get_distances_from_mesh(points=cyto_points, mesh=nuc_mesh, invert=True),
        "mem": get_distances_from_mesh(points=cyto_points, mesh=mem_mesh),
        "z": mem_mesh.bounds[1][2] - cyto_points[:, 2],  # depth below cell top (px)
    }
    weights = {
        rule: get_weights_from_distances(dist, decay_length=DECAY_LENGTH)
        for rule, dist in distances.items()
    }
    mixed_weights = sum(MIXED_WEIGHT_FRACTION[rule] * weights[rule] for rule in weights)
    weights["mixed"] = mixed_weights / np.max(mixed_weights)
    grid_map[projection_axis] = {
        "grid_points_slice": grid_points_slice,
        "inside_nuc": inside_nuc,
        "inside_mem_outside_nuc": inside_mem_outside_nuc,
        "distances": distances,
        "weights": weights,
    }

# %% [markdown]
# ## Fig 4a: distance from the nucleus, xy slice, grey nucleus
axis_data = grid_map["z"]
fig, ax = plot_grid_points_slice(
    grid_points_slice=axis_data["grid_points_slice"],
    inside_mem_outside_nuc=axis_data["inside_mem_outside_nuc"],
    color_var=axis_data["distances"]["nuc"],
    cbar_label="Distance from nucleus (µm)",
    dot_size=DOT_SIZE,
    projection_axis="z",
    cmap=FIG4A_CMAP,
)
# Nucleus points, centred as in plot_grid_points_slice
grid_points_um = axis_data["grid_points_slice"] * PIXEL_SIZE_IN_UM
centroid = np.mean(grid_points_um, axis=0)
x_index, y_index = PROJECTION_TO_INDEX_MAP["z"]
nuc_points_um = grid_points_um[axis_data["inside_nuc"]]
ax.scatter(
    x=nuc_points_um[:, x_index] - centroid[x_index],
    y=nuc_points_um[:, y_index] - centroid[y_index],
    c=FIG4A_NUCLEUS_COLOR,
    s=DOT_SIZE,
    marker=".",
    edgecolor="none",
)
fig.savefig(fig4_dir / f"Fig4a_nucleus_distance_xy_{CELL_ID}.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Fig 5a: rule weight maps, xy and xz slices
weight_labels = {
    "rnd": "Uniform Weight",
    "nuc": "Nucleus Weight",
    "mem": "Membrane Weight",
    "z": "Z Weight",
    "mixed": "Mixed Weight",
}
for projection_axis, plane_name in PLANE_NAMES.items():
    axis_data = grid_map[projection_axis]
    for rule, cbar_label in weight_labels.items():
        fig, ax = plot_grid_points_slice(
            grid_points_slice=axis_data["grid_points_slice"],
            inside_mem_outside_nuc=axis_data["inside_mem_outside_nuc"],
            color_var=axis_data["weights"][rule],
            cbar_label=cbar_label,
            dot_size=DOT_SIZE,
            projection_axis=projection_axis,
            clim=(0, 1),
        )
        fig.savefig(
            fig5_dir / f"Fig5a_{rule}_weights_{plane_name}_{CELL_ID}.pdf", bbox_inches="tight"
        )
        plt.show()

# %%
