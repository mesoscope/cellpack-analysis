# %% [markdown]
# # Get positions of structure centroids from images

import json
from concurrent.futures import ProcessPoolExecutor

# %% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from skimage import io, measure
from tqdm import tqdm

# %% [markdown]
# ### Set structure id
structure_name_dict = {
    "SLC25A17": "peroxisome",
    "RAB5A": "endosome",
}
STRUCTURE_ID = "RAB5A"  # peroxisome: "SLC25A17", early endosome: "RAB5A"
STRUCTURE_NAME = structure_name_dict[STRUCTURE_ID]
# %% [markdown]
# ### get data directory
datadir = Path(__file__).parents[2] / "data"
# %% [markdown]
# ### Get path for images
img_path = datadir / f"structure_data/{STRUCTURE_ID}/sample_8d/segmented/"

# %% [markdown]
# ### Get list of files
file_path = img_path.glob("*.tiff")
file_path_list = [x for x in file_path]
print(f"Number of files: {len(file_path_list)}")

# %% [markdown]
# Set image parameters
struct_channel_index = 4
nuc_channel_index = 0


# %% [markdown]
# ### Define function to get positions from a single image
def get_positions_from_single_image(file):
    """
    Get the positions of structures from a single image.

    Args:
        file (str): The file path of the image.

    Returns:
        tuple: A tuple containing the cell ID and a list of positions of structures.
    """
    cellid = file.stem.split("_")[1]
    img = io.imread(file)
    img_pex = img[:, struct_channel_index]
    img_nuc = img[:, nuc_channel_index]
    label_img_pex = measure.label(img_pex)
    label_img_nuc = measure.label(img_nuc)

    nuc_positions = []
    n_nuc = label_img_nuc.max()
    if n_nuc > 10:
        print(f"Warning: {n_nuc} nuclei detected in cell {cellid}")
    for i in range(1, n_nuc + 1):
        zcoords, ycoords, xcoords = np.where(label_img_nuc == i)
        nuc_positions.append(
            [
                np.mean(xcoords),
                np.mean(ycoords),
                np.mean(zcoords),
            ]
        )
    nuc_positions = np.array(nuc_positions)
    nuc_centroid = np.mean(nuc_positions, axis=0)

    positions = []
    for i in range(1, label_img_pex.max() + 1):
        zcoords, ycoords, xcoords = np.where(label_img_pex == i)
        positions.append(
            [
                np.mean(xcoords) - nuc_centroid[0],
                np.mean(ycoords) - nuc_centroid[1],
                np.mean(zcoords) - nuc_centroid[2],
            ]
        )
    return cellid, positions, nuc_centroid


# %% [markdown]
# ## Test single image
index = 50
file = file_path_list[index]
img = io.imread(file)
img_struct = img[:, struct_channel_index]
img_nuc = img[:, nuc_channel_index]
img_struct_max = np.max(img_struct, axis=0)
img_nuc_max = np.max(img_nuc, axis=0)
label_img = measure.label(img_struct)
print(img_struct.shape, img_struct_max.shape)

cellid, positions_corrected, nuc_centroid = get_positions_from_single_image(file)
positions_corrected = np.array(positions_corrected)
# %% load nucleus mesh
cellid = file.stem.split("_")[1]
nuc_mesh_path = file.parents[2] / f"meshes/nuc_mesh_{cellid}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)

# %% get distances
nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions_corrected)
fraction_negative = np.sum(nuc_distances < 0) / len(nuc_distances)
fig, ax = plt.subplots(1, 1)
ax.hist(nuc_distances)
ax.set_title(f"Cell {cellid}\nFraction negative: {fraction_negative:.2f}")
ax.set_xlabel("Distance from nucleus")
ax.set_ylabel("Frequency")
plt.show()
# %% find points with negative distance
neg_distance_indices = np.where(nuc_distances < 0)[0]
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
    ax.imshow(binary_label_img, cmap="Greens", origin="lower")
    ax.imshow(binary_img_nuc, cmap="Greys", alpha=0.5, origin="lower")
    ax.scatter(
        positions_corrected[:, plot_index_1] + nuc_centroid[plot_index_1],
        positions_corrected[:, plot_index_2] + nuc_centroid[plot_index_2],
        c="b",
        s=1,
    )  # plot all positions
    # ax.scatter(
    #     positions_corrected[neg_distance_indices, plot_index_1]
    #     + nuc_centroid[plot_index_1],
    #     positions_corrected[neg_distance_indices, plot_index_2]
    #     + nuc_centroid[plot_index_2],
    #     marker="*",
    #     c="r",
    #     s=10,
    # )  # only plot negative distances
    ax.set_xlabel(axis_indices[plot_index_1])
    ax.set_ylabel(axis_indices[plot_index_2])
axs[-1, -1].axis("off")
plt.tight_layout()
plt.show()

# %% process all images
num_processes = 32
structure_name = f"membrane_interior_{STRUCTURE_NAME}"
positions_dict = {}
parallel = True
if parallel:
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for file, (cellid, positions, nuc_centroid) in tqdm(
            zip(
                file_path_list,
                executor.map(get_positions_from_single_image, file_path_list),
            ),
            total=len(file_path_list),
        ):
            positions_dict[cellid] = {}
            positions_dict[cellid][structure_name] = positions
else:
    for file in tqdm(file_path_list):
        cellid, positions, nuc_centroid = get_positions_from_single_image(file)
        positions_dict[cellid] = {}
        positions_dict[cellid][structure_name] = positions
# %% save positions
save_path = (
    datadir / f"structure_data/{STRUCTURE_ID}/sample_8d/positions_{STRUCTURE_ID}.json"
)
with open(save_path, "w") as f:
    json.dump(positions_dict, f, indent=4, sort_keys=True)
print(f"Saved positions to {save_path}")
# %%
