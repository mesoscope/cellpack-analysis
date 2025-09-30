# %% [markdown]
# # Get positions of structure centroids from images

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy import ndimage
from skimage import io, measure
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.label_tables import STRUCTURE_NAME_DICT

logger = logging.getLogger(__name__)

# %% [markdown]
# ### Set structure id
STRUCTURE_ID = "SLC25A17"  # peroxisome: "SLC25A17", early endosome: "RAB5A"
STRUCTURE_NAME = STRUCTURE_NAME_DICT[STRUCTURE_ID]
dsphere = True
subfolder = "sample_8d" if dsphere else "full"

# %% [markdown]
# ### Get path for images
datadir = get_datadir_path() / f"structure_data/{STRUCTURE_ID}/{subfolder}"
img_path = datadir / "segmented"

# %% [markdown]
# ### Get list of files
file_path_list = [f for f in img_path.glob("*.tiff") if not f.name.startswith(".")]
logger.info(f"Number of files: {len(file_path_list)}")

# %% [markdown]
# Set image parameters
struct_channel_index = 3
nuc_channel_index = 0
mem_channel_index = 2


# %% [markdown]
# ### Define function to get positions from a single image
def get_positions_from_single_image(
    file: Path,
) -> tuple[str, list[list[float]], list[float], list[float], list[float], list[float]]:
    """
    Get the positions of structures from a single image.

    Parameters
    ----------
    file
        The path to the image file.

    Returns
    -------
    cell_id
        The cell ID extracted from the file name.
    positions
        A list of positions of the structures.
    nuc_centroid
        The centroid of the nucleus.
    mem_centroid
        The centroid of the membrane.
    struct_nuc_distances
        A list of distances from each structure to the nucleus.
    struct_mem_distances
        A list of distances from each structure to the membrane.
    """
    cell_id = file.stem.split("_")[1]
    img = io.imread(file)
    img_pex = img[:, struct_channel_index]
    img_nuc = img[:, nuc_channel_index]
    img_mem = img[:, mem_channel_index]
    label_img_pex, n_pex = measure.label(img_pex, return_num=True)  # type: ignore
    label_img_nuc, n_nuc = measure.label(img_nuc, return_num=True)  # type: ignore
    label_img_mem, n_mem = measure.label(img_mem, return_num=True)  # type: ignore

    nuc_positions = []
    nuc_sizes = []
    if n_nuc > 10:
        print(f"Warning: {n_nuc} nuclei detected in cell {cell_id}")
    for i in range(1, n_nuc + 1):
        zcoords, ycoords, xcoords = np.where(label_img_nuc == i)
        nuc_positions.append(
            [
                np.mean(xcoords),
                np.mean(ycoords),
                np.mean(zcoords),
            ]
        )
        nuc_sizes.append(len(xcoords))
    nuc_lcc = np.argmax(nuc_sizes)
    nuc_distances = ndimage.distance_transform_edt(
        ~(label_img_nuc == (nuc_lcc + 1)), return_indices=False
    )
    nuc_centroid = nuc_positions[nuc_lcc]

    mem_positions = []
    mem_sizes = []
    if n_mem > 10:
        print(f"Warning: {n_mem} membranes detected in cell {cell_id}")
    for i in range(1, n_mem + 1):
        zcoords, ycoords, xcoords = np.where(label_img_mem == i)
        mem_positions.append(
            [
                np.mean(xcoords),
                np.mean(ycoords),
                np.mean(zcoords),
            ]
        )
        mem_sizes.append(len(xcoords))
    mem_lcc = np.argmax(mem_sizes)
    mem_distances = ndimage.distance_transform_edt(
        ~(label_img_mem == (mem_lcc + 1)), return_indices=False
    )
    mem_centroid = mem_positions[mem_lcc]

    positions = []
    struct_nuc_distances = []
    struct_mem_distances = []
    for i in range(1, n_pex + 1):
        zcoords, ycoords, xcoords = np.where(label_img_pex == i)
        centroid = [np.mean(xcoords), np.mean(ycoords), np.mean(zcoords)]
        centroid_inds = np.round(centroid).astype(int)
        positions.append(
            centroid,
        )
        struct_nuc_distances.append(
            nuc_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]  # type: ignore
        )
        struct_mem_distances.append(
            mem_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]  # type: ignore
        )
    return (
        cell_id,
        positions,
        nuc_centroid,
        mem_centroid,
        struct_nuc_distances,
        struct_mem_distances,
    )


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
print(img_struct.shape, img_struct_max.shape)

cell_id, positions, nuc_centroid, mem_centroid, nuc_distances_img, mem_distances_img = (
    get_positions_from_single_image(file)
)
positions = np.array(positions)
nuc_centroid = np.array(nuc_centroid)
mem_centroid = np.array(mem_centroid)

# %% load nucleus and membrane mesh
cell_id = file.stem.split("_")[1]
nuc_mesh_path = file.parents[2] / f"meshes/nuc_mesh_{cell_id}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
# nuc_mesh.apply_translation(-nuc_mesh.centroid)

mem_mesh_path = file.parents[2] / f"meshes/mem_mesh_{cell_id}.obj"
mem_mesh = trimesh.load_mesh(mem_mesh_path)
# mem_mesh.apply_translation(-mem_mesh.centroid)

# %% get distances
nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions)
mem_distances = trimesh.proximity.signed_distance(mem_mesh, positions)
fraction_inside_nuc = np.sum(nuc_distances < 0) / len(nuc_distances)
fraction_outside_mem = np.sum(mem_distances < 0) / len(mem_distances)

# %% plot distances
fig, ax = plt.subplots(1, 2)

ax[0].hist(nuc_distances)
ax[0].set_title(f"Cell {cell_id}\nFraction inside nuc: {fraction_inside_nuc:.2f}")
ax[0].set_xlabel("Distance from nucleus")

ax[1].hist(mem_distances)
ax[1].set_title(f"Cell {cell_id}\nFraction outside mem: {fraction_outside_mem:.2f}")
ax[1].set_xlabel("Distance from membrane")
plt.show()
# %% find points outside membrane and inside nucleus
inside_nuc_indices = nuc_distances < 0
outside_mem_indices = mem_distances < 0
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

# %% process all images
num_processes = 32
structure_name = f"membrane_interior_{STRUCTURE_NAME}"
positions_dict = {}
centroids_dict = {}
parallel = True
if parallel:
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for _, (cell_id, positions, nuc_centroid, mem_centroid, _, _) in tqdm(
            zip(
                file_path_list,
                executor.map(get_positions_from_single_image, file_path_list),
                strict=False,
            ),
            total=len(file_path_list),
        ):
            positions_dict[cell_id] = {}
            positions_dict[cell_id][structure_name] = positions
            centroids_dict[cell_id] = {}
            centroids_dict[cell_id]["nucleus"] = nuc_centroid
            centroids_dict[cell_id]["membrane"] = mem_centroid
else:
    for file in tqdm(file_path_list):
        cell_id, positions, nuc_centroid, mem_centroid, _, _ = get_positions_from_single_image(file)
        positions_dict[cell_id] = {}
        positions_dict[cell_id][structure_name] = positions
        centroids_dict[cell_id] = {}
        centroids_dict[cell_id]["nucleus"] = nuc_centroid
        centroids_dict[cell_id]["membrane"] = mem_centroid

# %% save positions
save_path = datadir / f"positions_{STRUCTURE_ID}.json"
with open(save_path, "w") as f:
    json.dump(positions_dict, f, indent=4, sort_keys=True)
print(f"Saved positions to {save_path}")
# %% save centroids
save_path = datadir / f"centroids/centroids_{STRUCTURE_ID}.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(centroids_dict, f, indent=4, sort_keys=True)
print(f"Saved centroids to {save_path}")

# %%
