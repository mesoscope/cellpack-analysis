# %% [markdown]
# # Get positions of structure centroids from images

# %%
from pathlib import Path
import numpy as np
from skimage import io, measure
import io as io_base
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import json
import trimesh

from concurrent.futures import ProcessPoolExecutor

# %%
img_path = Path(
    "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/structure_data/SLC25A17/sample_8d/raw_imgs_for_PILR/"
)

# %%
file_path = img_path.glob("*.tiff")
file_path_list = [x for x in file_path]

# %%
struct_channel_index = 4
num_processes = 32
structure_name = "membrane_interior_peroxisome"


# %%
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
    mid_z, mid_y, mid_x = img_pex.shape[0] // 2, img_pex.shape[1] // 2, img_pex.shape[2] // 2
    label_img = measure.label(img_pex)
    positions = []
    for i in range(1, label_img.max() + 1):
        zcoords, ycoords, xcoords = np.where(label_img == i)
        positions.append(
            [
                np.mean(xcoords) - mid_x,
                np.mean(ycoords) - mid_y,
                np.mean(zcoords) - mid_z,
            ]
        )
    return cellid, positions


# %% process images
positions_dict = {}
parallel = True
if parallel:
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for file, (cellid, positions) in tqdm(
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
        cellid, positions = get_positions_from_single_image(file)
        positions_dict[cellid] = {}
        positions_dict[cellid][structure_name] = positions


# %% Test single image
index = 0
file = file_path_list[index]
img = io.imread(file)
img_pex = img[:, 4]
img_nuc = img[:, 0]
img_pex_max = np.max(img_pex, axis=0)
img_nuc_max = np.max(img_nuc, axis=0)
print(img_pex.shape, img_pex_max.shape)
# %% get peroxisome positions
label_img = measure.label(img_pex)
mid_z, mid_y, mid_x = (
    img_pex.shape[0] // 2,
    img_pex.shape[1] // 2,
    img_pex.shape[2] // 2,
)
positions = []
for i in range(1, label_img.max() + 1):
    zcoords, ycoords, xcoords = np.where(label_img == i)
    positions.append(
        [
            np.mean(xcoords),
            np.mean(ycoords),
            np.mean(zcoords),
        ]
    )
positions = np.array(positions)
positions_corrected = positions - np.array([mid_x, mid_y, mid_z])
# %% load nucleus mesh
cellid = file.stem.split("_")[1]
nuc_mesh_path = file.parents[2] / f"meshes/nuc_mesh_{cellid}.obj"
nuc_mesh = trimesh.load_mesh(nuc_mesh_path)
# %% plot nucleus mesh as image
data = nuc_mesh.save_image()
nuc_img = Image.open(io_base.BytesIO(data))
print(nuc_img.shape)
nuc_img_max = np.max(nuc_img, axis=0)
fig, ax = plt.subplots(1, 1)
ax.imshow(nuc_img_max, origin="lower", cmap="gray")

# %% get distances
nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions_corrected)
fig, ax = plt.subplots(1, 1)
ax.hist(nuc_distances)
ax.set_title(f"Cell {cellid}")
ax.set_xlabel("Distance from nucleus")
ax.set_ylabel("Frequency")
plt.show()
# %% find points with negative distance
neg_distance_indices = np.where(nuc_distances < 0)[0]
# %% plot
fig, ax = plt.subplots(1, 1)
ax.imshow(np.max(label_img, axis=0), cmap="tab20")
ax.imshow(img_nuc_max, cmap="gray", alpha=0.5)
# ax.scatter(positions[:, 0], positions[:, 1], c="r", s=1)
ax.scatter(positions[neg_distance_indices, 0], positions[neg_distance_indices, 1], c="r", s=1)  # only plot negative distances
# ax.text(centroids[1], centroids[0], f"{i}", c="r")
# ax.text(centroids_corrected[1], centroids_corrected[0], f"{i}", c="g")
plt.show()

# %%
positions = np.array(positions)

# %%
fig, ax = plt.subplots(1, 2)
ax[0].hist(positions[:, 0], bins=20)
ax[1].hist(positions[:, 1], bins=20)

# %%
with open(
    "/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/structure_data/SLC25A17/sample_8d/positions_SLC25A17.json",
    "w",
) as f:
    json.dump(positions_dict, f, indent=4, sort_keys=True)

# %%
xcoords, ycoords, zcoords = [], [], []
for cellid, position in positions_dict.items():
    xcoords.extend([x[0] for x in position[structure_name]])
    ycoords.extend([x[1] for x in position[structure_name]])
    zcoords.extend([x[2] for x in position[structure_name]])
xcoords = np.array(xcoords)
ycoords = np.array(ycoords)
zcoords = np.array(zcoords)
print(xcoords.shape)


# %%
fig, ax = plt.subplots(1, 3)
ax[0].hist(xcoords, bins=100)
ax[1].hist(ycoords, bins=100)
ax[2].hist(zcoords, bins=100)
plt.tight_layout()
plt.show()

# %%
