# %% [markdown]
# # Get positions of structure centroids from images

import logging

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from skimage import io, measure

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.label_tables import STRUCTURE_NAME_DICT
from cellpack_analysis.preprocessing.get_structure_coordinates import (
    get_positions_from_single_image,
)

logger = logging.getLogger(__name__)

# %% [markdown]
# ### Set structure ID and name
STRUCTURE_ID = "SLC25A17"  # peroxisome: "SLC25A17", early endosome: "RAB5A"
STRUCTURE_NAME = STRUCTURE_NAME_DICT[STRUCTURE_ID]
subfolder = "sample_8d"

# %% [markdown]
# ### Get path for images
datadir = get_datadir_path() / f"structure_data/{STRUCTURE_ID}/{subfolder}"
img_path = datadir / "segmented"

# %% [markdown]
# ### Get list of files
file_path_list = [f for f in img_path.glob("*.tiff") if not f.name.startswith(".")]
logger.info(f"Number of files: {len(file_path_list)}")

# %% [markdown]
# ### Set image channels
struct_channel_index = 3
nuc_channel_index = 0
mem_channel_index = 1

# %% [markdown]
# ## Test single image
index = 0
file = file_path_list[index]
img = io.imread(file)
img_struct = img[:, struct_channel_index]
img_nuc = img[:, nuc_channel_index]
img_mem = img[:, mem_channel_index]
img_struct_max = np.max(img_struct, axis=0)
img_nuc_max = np.max(img_nuc, axis=0)
img_mem_max = np.max(img_mem, axis=0)
label_img = measure.label(img_struct)
logger.info("Image shape: %s", img.shape)

cell_id, positions, nuc_centroid, mem_centroid, nuc_distances_img, mem_distances_img = (
    get_positions_from_single_image(file)
)
positions = np.array(positions)
nuc_centroid = np.array(nuc_centroid)
mem_centroid = np.array(mem_centroid)
nuc_distances_img = np.array(nuc_distances_img)
mem_distances_img = np.array(mem_distances_img)

# %% [markdown]
# ### Load nucleus and membrane mesh
cell_id = file.stem.split("_")[1]
nuc_mesh_path = file.parents[2] / f"meshes/nuc_mesh_{cell_id}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
# nuc_mesh.apply_translation(-nuc_mesh.centroid)

mem_mesh_path = file.parents[2] / f"meshes/mem_mesh_{cell_id}.obj"
mem_mesh = trimesh.load_mesh(mem_mesh_path)
# mem_mesh.apply_translation(-mem_mesh.centroid)

# %% [markdown]
# ### Get distances from meshes
nuc_distances_mesh = -trimesh.proximity.signed_distance(nuc_mesh, positions)
mem_distances_mesh = trimesh.proximity.signed_distance(mem_mesh, positions)
fraction_inside_nuc = np.sum(nuc_distances_mesh < 0) / len(nuc_distances_mesh)
fraction_outside_mem = np.sum(mem_distances_mesh < 0) / len(mem_distances_mesh)

# %% [markdown]
# ### Compare image and mesh distances
fig, axs = plt.subplots(1, 2)
axs[0].scatter(nuc_distances_img, nuc_distances_mesh, s=1)
axs[0].set_xlabel("Image distances")
axs[0].set_ylabel("Mesh distances")
axs[0].set_title(f"Nucleus distances\nCell {cell_id}")

axs[1].scatter(mem_distances_img, mem_distances_mesh, s=1)
axs[1].set_xlabel("Image distances")
axs[1].set_ylabel("Mesh distances")
axs[1].set_title(f"Membrane distances\nCell {cell_id}")
plt.show()

# %% [markdown]
# ### Plot mesh distances
fig, axs = plt.subplots(1, 2)

axs[0].hist(nuc_distances_mesh)
axs[0].set_title(f"Cell {cell_id}\nFraction inside nuc: {fraction_inside_nuc:.2f}")
axs[0].set_xlabel("Distance from nucleus")

axs[1].hist(mem_distances_mesh)
axs[1].set_title(f"Cell {cell_id}\nFraction outside mem: {fraction_outside_mem:.2f}")
axs[1].set_xlabel("Distance from membrane")
plt.show()
# %% find points outside membrane and inside nucleus
inside_nuc_indices = nuc_distances_mesh < 0
outside_mem_indices = mem_distances_mesh < 0
good_indices = np.logical_and(~inside_nuc_indices, ~outside_mem_indices)
bad_indices = np.logical_or(inside_nuc_indices, outside_mem_indices)
# %% plot composite image with structure positions and negative distances
fig, axs = plt.subplots(2, 2, dpi=300)
for ct, projection_axis in enumerate(["x", "y", "z"]):
    ax = axs[ct // 2, ct % 2]
    projection_axis_index = {"x": 2, "y": 1, "z": 0}[projection_axis]
    plot_index_1 = {"x": 1, "y": 0, "z": 0}[projection_axis]
    plot_index_2 = {"x": 2, "y": 2, "z": 1}[projection_axis]
    axis_indices = {0: "x", 1: "y", 2: "z"}

    # Get max projections for each channel
    img_struct_proj = np.max(img_struct, axis=projection_axis_index)
    img_nuc_proj = np.max(img_nuc, axis=projection_axis_index)
    img_mem_proj = np.max(img_mem, axis=projection_axis_index)

    # Normalize intensity values to 0-1 range
    img_struct_norm = (
        img_struct_proj / np.max(img_struct_proj)
        if np.max(img_struct_proj) > 0
        else img_struct_proj
    )
    img_nuc_norm = img_nuc_proj / np.max(img_nuc_proj) if np.max(img_nuc_proj) > 0 else img_nuc_proj
    img_mem_norm = img_mem_proj / np.max(img_mem_proj) if np.max(img_mem_proj) > 0 else img_mem_proj

    # Create binary masks for each channel
    threshold = 0.1
    struct_mask = img_struct_norm > threshold
    nuc_mask = img_nuc_norm > threshold
    mem_mask = img_mem_norm > threshold

    # Create RGB composite with priority overlay
    height, width = img_struct_proj.shape
    composite = np.zeros((height, width, 3))

    # Layer 1 (bottom): Membrane - Magenta (1, 0, 1)
    composite[mem_mask, 0] = img_mem_norm[mem_mask]  # Red channel
    composite[mem_mask, 2] = img_mem_norm[mem_mask]  # Blue channel

    # Layer 2 (middle): Nucleus - Cyan (0, 1, 1)
    composite[nuc_mask, 0] = 0  # Clear red channel to avoid white with membrane
    composite[nuc_mask, 1] = img_nuc_norm[nuc_mask]  # Green channel
    composite[nuc_mask, 2] = img_nuc_norm[nuc_mask]  # Blue channel

    # Layer 3 (top): Structure - Yellow/Green (1, 1, 0)
    composite[struct_mask, :] = 0  # Clear all channels first
    composite[struct_mask, 1] = img_struct_norm[struct_mask]  # Green channel

    ax.imshow(composite, origin="lower")

    # Plot structure positions with different colors
    ax.scatter(
        positions[good_indices, plot_index_1],
        positions[good_indices, plot_index_2],
        c="white",
        s=5,
        marker="x",
    )  # plot good positions
    ax.scatter(
        positions[inside_nuc_indices, plot_index_1],
        positions[inside_nuc_indices, plot_index_2],
        c="red",
        s=5,
        marker="x",
    )  # inside nucleus
    ax.scatter(
        positions[outside_mem_indices, plot_index_1],
        positions[outside_mem_indices, plot_index_2],
        c="blue",
        s=5,
        marker="x",
    )  # outside membrane

    ax.set_xlabel(axis_indices[plot_index_1])
    ax.set_ylabel(axis_indices[plot_index_2])
    ax.set_title(f"{projection_axis.upper()}-projection")

axs[-1, -1].axis("off")
# Add legend for the composite colors
legend_text = (
    "Membrane: Magenta\nNucleus: Cyan\nStructure: Green\n"
    "Good positions: White\nInside nucleus: Red\nOutside membrane: Blue"
)
axs[-1, -1].text(0.1, 0.5, legend_text, fontsize=8, verticalalignment="center")
plt.tight_layout()
plt.show()
# %%
