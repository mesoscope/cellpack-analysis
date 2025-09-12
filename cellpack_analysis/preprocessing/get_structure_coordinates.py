# %% [markdown]
# # Get positions of structure centroids from images

import json
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

# %% [markdown]
# ### Set structure id
STRUCTURE_ID = "SLC25A17"  # peroxisome: "SLC25A17", early endosome: "RAB5A"
STRUCTURE_NAME = STRUCTURE_NAME_DICT[STRUCTURE_ID]
dsphere = True
subfolder = "sample_8d" if dsphere else "full"

# %% [markdown]
# ### Get path for images
img_path = get_datadir_path() / f"structure_data/{STRUCTURE_ID}/{subfolder}/segmented/"

# %% [markdown]
# ### Get list of files
file_path = img_path.glob("*.tiff")
file_path_list = list(file_path)
print(f"Number of files: {len(file_path_list)}")

# %% [markdown]
# Set image parameters
struct_channel_index = 3
nuc_channel_index = 0
mem_channel_index = 2


# %% [markdown]
# ### Define function to get positions from a single image
def get_positions_from_single_image(file):
    """
    Get the positions of structures from a single image.

    Args:
    ----
        file (str): The file path of the image.

    Returns:
    -------
        tuple: A tuple containing the cell ID and a list of positions of structures.
    """
    cell_id = file.stem.split("_")[1]
    img = io.imread(file)
    img_pex = img[:, struct_channel_index]
    img_nuc = img[:, nuc_channel_index]
    img_mem = img[:, mem_channel_index]
    label_img_pex, n_pex = measure.label(img_pex, return_num=True)
    label_img_nuc, n_nuc = measure.label(img_nuc, return_num=True)
    label_img_mem, n_mem = measure.label(img_mem, return_num=True)

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
            nuc_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]
        )
        struct_mem_distances.append(
            mem_distances[centroid_inds[2], centroid_inds[1], centroid_inds[0]]
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
# %% plot nucleus with structure positions and negative distances
fig, axs = plt.subplots(2, 2, dpi=300)
# projection_axis = "y"
for ct, projection_axis in enumerate(["x", "y", "z"]):
    ax = axs[ct // 2, ct % 2]
    projection_axis_index = {"x": 2, "y": 1, "z": 0}[projection_axis]
    plot_index_1 = {"x": 1, "y": 0, "z": 0}[projection_axis]
    plot_index_2 = {"x": 2, "y": 2, "z": 1}[projection_axis]
    axis_indices = {0: "x", 1: "y", 2: "z"}
    binary_label_img = np.max(label_img > 0, axis=projection_axis_index)
    binary_img_nuc = np.max(img_nuc > 0, axis=projection_axis_index)
    binary_img_mem = np.max(img_mem > 0, axis=projection_axis_index)
    ax.imshow(binary_label_img, cmap="Greens", origin="lower")
    ax.imshow(binary_img_nuc, cmap="Blues", alpha=0.4, origin="lower")
    ax.imshow(binary_img_mem, cmap="Reds", alpha=0.4, origin="lower")
    ax.scatter(
        positions[good_indices, plot_index_1],
        positions[good_indices, plot_index_2],
        c="y",
        s=1,
    )  # plot good positions
    ax.scatter(
        positions[inside_nuc_indices, plot_index_1],
        positions[inside_nuc_indices, plot_index_2],
        c="r",
        s=1,
    )  # inside nucleus
    ax.scatter(
        positions[outside_mem_indices, plot_index_1],
        positions[outside_mem_indices, plot_index_2],
        c="b",
        s=1,
    )  # outside membrane

    ax.set_xlabel(axis_indices[plot_index_1])
    ax.set_ylabel(axis_indices[plot_index_2])
axs[-1, -1].axis("off")
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
save_path = datadir / f"structure_data/{STRUCTURE_ID}/{subfolder}/positions_{STRUCTURE_ID}.json"
with open(save_path, "w") as f:
    json.dump(positions_dict, f, indent=4, sort_keys=True)
print(f"Saved positions to {save_path}")
# %% save centroids
save_path = datadir / f"structure_data/{STRUCTURE_ID}/centroids/centroids_{STRUCTURE_ID}.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(centroids_dict, f, indent=4, sort_keys=True)
print(f"Saved centroids to {save_path}")

# %%
