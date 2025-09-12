# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cellpack_analysis.lib.file_io import read_json
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.mesh_tools import get_average_shape_mesh_objects
from cellpack_analysis.lib.PILR_tools import (
    average_over_dimension,
    get_domain,
    get_embeddings,
    get_parametrized_coords_for_avg_shape,
    morph_PILRs_into_average_shape,
)
from cellpack_analysis.lib.plotting_tools import plot_and_save_center_slice, plot_PILR

# %% [markdown]
# ### Read in individual PILRs
pilr_path = (
    "/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/PILR_correlation_analysis/"
    "combined/individual_PILR.json"
)
individual_PILR_dict = read_json(pilr_path)

# %% [markdown]
# ### Process PILRs

average_over_phi = False

for key, value in individual_PILR_dict.items():
    value = np.array(value)

    if average_over_phi:
        avg_value = np.zeros((value.shape[0], value.shape[1], 64))
        for ind in range(value.shape[0]):
            avg_value[ind] = average_over_dimension(value[ind])
        value = avg_value

    # mask out nucleus
    new_values = np.zeros((value.shape[0], value.shape[1] // 2, value.shape[2]))
    for ind in range(value.shape[0]):
        new_values[ind] = value[ind][(len(value[ind]) // 2 + 1) :]
        new_values[ind] = new_values[ind] / np.max(new_values[ind])
    individual_PILR_dict[key] = new_values.reshape(new_values.shape[0], -1)
    print(key, individual_PILR_dict[key].shape)

# %% [markdown]
# ### Create index to cell_id dictionary
struct_list = ["SLC25A17", "RAB5A"]
cell_id_dict = {}
for struct in struct_list:
    cell_id_list = get_cell_id_list_for_structure(struct)
    cell_id_dict[struct] = cell_id_list

# %% [markdown]
# ## Plot pacmap/pca for individual channels

# %% [markdown]
# ### Plot all channels
channels_to_use = list(individual_PILR_dict.keys())

# %% [markdown]
# ### plot selected channels
channels_to_use = [
    "SLC25A17",
    # "RAB5A",
    # "random",
    # "membrane_moderate_gradient",
    # "nucleus_moderate_gradient"
]

# %%
PILR_dict_to_use = {key: individual_PILR_dict[key] for key in channels_to_use}

# %%
metric = "pca"
channels_for_embedding = channels_to_use
# channels_for_embedding = [ch for ch in channels_to_use if ch != "SLC25A17"]
# channels_for_embedding = ["SLC25A17"]

# %%
# n_components_pca = 2
n_components_pca = min([min(pilr.shape) for pilr in PILR_dict_to_use.values()])
print(n_components_pca)

# %%
embedding_dict, embedding = get_embeddings(
    PILR_dict_to_use,
    metric=metric,
    channels_for_embedding=channels_for_embedding,
    n_components_pca=n_components_pca,
    n_components=n_components_pca,
)

# %% [markdown]
# ### Plot explained variance ratio

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax.plot(
    np.cumsum(embedding.explained_variance_ratio_), lw=2, color="k"
)  # , marker="o", markersize=5)
ax.set_xlabel("Principal Component Dimension")
ax.set_ylabel("Cumulative explained Variance Ratio")

# %%
for ch, ch_dict in embedding_dict.items():
    print(ch, ch_dict["embedding"], ch_dict["values"].shape)

# %% [markdown]
# ### Plot colorized pacmap/pca

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
prev_ind = 0
for _col_ind, (ch, ch_dict) in enumerate(embedding_dict.items()):
    if "invert" in ch:
        continue
    embedding_type = ch_dict["embedding"]
    values = ch_dict["values"]
    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=20,
        label=ch,
        alpha=0.6,
        marker="o" if embedding_type == "fit" else "x",
        # color=f"C{col_ind}",
        zorder=1 if embedding == "fit" else 2,
        edgecolors="none" if embedding_type == "fit" else None,
    )
if metric.lower() == "pacmap_pca":
    ax.set_title(f"Init PCA dims: {n_components_pca}")
ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
# ax.set_xlim(-5, 15)
# ax.set_ylim(-5, 10)
ax.set_xlabel(f"{metric} 1")
ax.set_ylabel(f"{metric} 2")
plt.show()

# %%
# zoom in to plot
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
plt.show()

# %% [markdown]
# ## Latent walk along PCA embedding

# %%
ch_name = "SLC25A17"

input_data = embedding_dict[ch_name]["values"]
print(input_data.shape)

# %% [markdown]
# Histogram of pca values

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
for walk_dim in (0, 1):
    ax[walk_dim].hist(input_data[:, walk_dim], bins=20)
    ax[walk_dim].axvline(np.mean(input_data[:, walk_dim]), color="k", lw=2)
    ax[walk_dim].set_title(f"{ch_name} PCA {walk_dim}")
    ax[walk_dim].set_xlabel("PCA value")
    ax[walk_dim].set_ylabel("Frequency")

plt.show()

# %% [markdown]
# ## Get latent walk points

# %%
walk_dim = 1

# %% [markdown]
# get up to 2 stds away from mean
num_std = 2
walk_pts = np.zeros((num_std * 2 + 1, input_data.shape[1]))
avg_dim_val = np.mean(input_data[:, walk_dim])
std_dim_val = np.std(input_data[:, walk_dim])
walk_pts[:, walk_dim] = avg_dim_val + np.arange(-num_std, num_std + 1) * std_dim_val
print(walk_pts.shape)

# %% [markdown]
# sample points from min to max
num_pts = 5
walk_pts = np.zeros((num_pts, input_data.shape[1]))
walk_pts[:, walk_dim] = np.linspace(
    min(input_data[:, walk_dim]), max(input_data[:, walk_dim]), num_pts
)
print(walk_pts.shape)

# %% [markdown]
# get inverse transform
generated_pts = []
for walk_pt in walk_pts:
    generated_pt = embedding.inverse_transform(walk_pt)
    generated_pts.append(generated_pt)
generated_pts = np.array(generated_pts)
print(generated_pts.shape)

# %% [markdown]
# ## Plot PILRs from latent walk
fig, axs = plt.subplots(generated_pts.shape[0], 1, figsize=(generated_pts.shape[0] * 2, 5), dpi=300)
pilr_list = []
for ct, generated_pt in enumerate(generated_pts):
    # plot the pilrs
    ax = axs[ct]
    pilr = generated_pt.reshape(32, -1)
    _, ax = plot_PILR(pilr, ax=ax, save_dir=None, aspect=20, vmax=0.05, vmin=0)
    # ax.set_title(f"{ct - len(generated_pts)//2} SD", c="w")
    ax.set_title(f"{ct / (len(generated_pts) - 1) * 100:.3g} percentile", c="w")

    # re-pad PILR to add nucleus
    padded_pilr = np.zeros((pilr.shape[0] * 2 + 1, pilr.shape[1]))
    padded_pilr[pilr.shape[0] // 2 : pilr.shape[0] // 2 + pilr.shape[0], :] = pilr
    pilr_list.append(padded_pilr)

fig.suptitle(f"{ch_name} PCA Walk, dimension {walk_dim}", c="w")
plt.tight_layout()
plt.show(fig)

# %% [markdown]
# ## Morph inverse transformed PILRs into average shape

# %% [markdown]
# Get domain and coords_param for the average shape meshes
mesh_path = Path(__file__).parents[3] / "data/average_shape_meshes"
mesh_dict = get_average_shape_mesh_objects(mesh_path)
domain = get_domain(mesh_dict)
coords_param = get_parametrized_coords_for_avg_shape(domain)
#  %% [markdown]
# create spherical domain
domain = np.zeros((300, 300, 300))
domain[:250, :250, :250] = 1
domain[:100, :100, :100] = 2
coords_param = get_parametrized_coords_for_avg_shape(domain)

# %% [markdown]
# get reconstructions
morphed = morph_PILRs_into_average_shape(
    pilr_list=pilr_list,
    domain=domain,
    coords_param=coords_param,
    mesh_dict=mesh_dict,
)

# %%
dim_to_axis_map = {
    0: "XZ",
    1: "XY",
    2: "YZ",
}

# %%
fig, axs = plt.subplots(2, len(morphed), figsize=(len(morphed) * 2, 4), dpi=300)
for ct, morph in enumerate(morphed):
    for dim in range(2):
        fig, axs[dim][ct] = plot_and_save_center_slice(
            morph,
            structure=ch_name,
            dim=dim,
            ax=axs[dim][ct],
            # title=f"CellId: {cell_id_dict[ch_name][pilr_inds[ct]]}",
            title=f"Walk point {ct}",
            ylabel=f"{dim_to_axis_map[dim]}" if ct == 0 else None,
            showfig=False,
        )

fig.suptitle(f"{ch_name} PCA walk center slice, dimension {walk_dim} ", c="w")
plt.tight_layout()
plt.show()


# %% [markdown]
# Look at outliers

# %%
ch_name = "RAB5A"

# %%
walk_dim = 1

# %%
cell_id_list = cell_id_dict[ch_name]

# %%
input_data.shape

# %% [markdown]
# get outliers along dimension

# %%
sort_inds = np.argsort(input_data[:, walk_dim])
walk_pts = input_data[sort_inds[-3:]]
outlier_cell_id = [cell_id_list[ind] for ind in sort_inds[-2:]]
print(walk_pts[:, walk_dim], outlier_cell_id)

# %% [markdown]
# get outlier based on distance

# %%
pt = np.array([10, 20])
distances = np.linalg.norm(input_data[:, :2] - pt, axis=1)
min_ind = np.argmin(distances)
outlier_cell_id = cell_id_list[min_ind]
print(outlier_cell_id)


# %%
