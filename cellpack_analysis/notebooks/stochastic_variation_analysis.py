# %%
from unittest import result
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.spatial.distance import cdist, squareform
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import trimesh
import seaborn as sns

from cellpack_analysis.utilities.data_tools import (
    get_positions_dictionary_from_file,
    combine_multiple_seeds_to_dictionary,
    get_pairwise_wasserstein_distance_dict,
)

# %% set file paths
base_datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/")
base_results_dir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/")

results_dir = base_results_dir / "stochastic_variation_analysis/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% load data
packing_modes = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "variable_count_and_size",
    "shape",
]
all_positions = {}
for mode in packing_modes:
    print(mode)
    data_folder = file_path = (
        base_datadir
        / f"packing_outputs/stochastic_variation_analysis/{mode}/peroxisome/spheresSST/"
    )
    if mode == "shape":
        combine_multiple_seeds_to_dictionary(data_folder)

    file_path = data_folder / "positions_peroxisome_analyze_random_mean.json"
    positions = get_positions_dictionary_from_file(file_path)

    all_positions[mode] = positions
# %% save all positions dictionary
file_path = results_dir / "packing_modes_positions.dat"
with open(file_path, "wb") as f:
    pickle.dump(all_positions, f)

# %% load saved dictionary
file_path = results_dir / "packing_modes_positions.dat"
with open(file_path, "rb") as f:
    all_positions = pickle.load(f)    

# %% Calculate distance measures
# %%
nuc_mesh_path = base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))

# %%
all_pairwise_distances = {}  # pairwise distance between particles
all_nuc_distances = {}  # distance to nucleus surface
all_nearest_distances = {}  # distance to nearest neighbor
for mode, position_dict in all_positions.items():
    print(mode)
    all_pairwise_distances[mode] = {}
    all_nuc_distances[mode] = {}
    all_nearest_distances[mode] = {}
    for seed, positions in tqdm(position_dict.items()):
        nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions)
        all_nuc_distances[mode][seed] = nuc_distances / np.max(nuc_distances)
        all_distances = cdist(positions, positions, metric="euclidean")
        nearest_distances = np.min(all_distances + np.eye(len(positions)) * 1e6, axis=1)
        all_nearest_distances[mode][seed] = nearest_distances / np.max(
            nearest_distances
        )
        pairwise_distances = squareform(all_distances)
        all_pairwise_distances[mode][seed] = pairwise_distances / np.max(
            pairwise_distances
        )

# %% save distance dictionaries
for distance_dict, distance_measure in zip(
    [all_pairwise_distances, all_nuc_distances, all_nearest_distances],
    ["pairwise", "nucleus", "nearest"],
):
    file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
    with open(file_path, "wb") as f:
        pickle.dump(distance_dict, f)

# %% load saved dictionary
for distance_dict, distance_measure in zip(
    [all_pairwise_distances, all_nuc_distances, all_nearest_distances],
    ["pairwise", "nucleus", "nearest"],
):
    file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
    with open(file_path, "rb") as f:
        distance_dict = pickle.load(f)

# %%
num_rows = len(all_positions)
num_cols = 3

fig, axs = plt.subplots(
    num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), dpi=300
)

for i, (mode, position_dict) in enumerate(all_positions.items()):
    cmap = plt.get_cmap("jet", len(position_dict))
    for j, (distance_measure, distance_dict) in enumerate(
        [
            ("pairwise", all_pairwise_distances[mode]),
            ("nucleus", all_nuc_distances[mode]),
            ("nearest", all_nearest_distances[mode]),
        ]
    ):
        for k, (seed, distances) in enumerate(distance_dict.items()):
            sns.kdeplot(distances, ax=axs[i, j], color=cmap(k), alpha=0.2)

        axs[i, j].set_title(f"{mode} {distance_measure}")
        axs[i, j].set_xlim([0, 1])
        axs[i, j].set_xlabel("Distance")
        axs[i, j].set_ylabel("PDF")

fig.tight_layout()
fig.savefig(figures_dir / "distance_distributions.png", dpi=300)

plt.show()

# %% Get pairwise earth movers distances
all_pairwise_emd = {}
for distribution_dict, distance_measure in zip(
    [all_pairwise_distances, all_nuc_distances, all_nearest_distances],
    ["pairwise", "nucleus", "nearest"],
):
    print(distance_measure)
    measure_pairwise_emd = {}
    for mode_1 in packing_modes:
        if mode_1 not in measure_pairwise_emd:
            measure_pairwise_emd[mode_1] = {}
        distribution_dict_1 = distribution_dict[mode_1]
        for mode_2 in packing_modes:
            if (
                measure_pairwise_emd.get(mode_2) is not None
                and measure_pairwise_emd[mode_2].get(mode_1) is not None
            ):
                continue
            print(mode_1, mode_2)

            if mode_1 == mode_2:
                distribution_dict_2 = None
            else:
                distribution_dict_2 = distribution_dict[mode_2]
            measure_pairwise_emd[mode_1][
                mode_2
            ] = get_pairwise_wasserstein_distance_dict(
                distribution_dict_1,
                distribution_dict_2,
            )

            if mode_2 not in measure_pairwise_emd:
                measure_pairwise_emd[mode_2] = {}
            measure_pairwise_emd[mode_2][mode_1] = measure_pairwise_emd[mode_1][mode_2]
    all_pairwise_emd[distance_measure] = measure_pairwise_emd

# %% Save EMD dict
file_path = results_dir / "packing_modes_pairwise_emd.dat"
with open(file_path, "wb") as f:
    pickle.dump(all_pairwise_emd, f)

# %% load saved dictionary
file_path = results_dir / "packing_modes_pairwise_emd.dat"
with open(file_path, "rb") as f:
    all_pairwise_emd = pickle.load(f)

# %% Plot EMD
pairwise_emd_dir = figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)
for distance_measure, emd_dict in all_pairwise_emd.items():
    fig, axs = plt.subplots(len(emd_dict), len(emd_dict), figsize=(10, 10), dpi=300)
    for rt, (mode_1, mode_1_dict) in enumerate(emd_dict.items()):
        for ct, (mode_2, emd) in enumerate(mode_1_dict.items()):
            print(distance_measure, mode_1, mode_2)
            if mode_1 == mode_2:
                values = squareform(list(emd.values()))
            else:
                values = list(emd.values())
                values = np.array(values)
                dim_len = int(np.sqrt(len(values)))
                values = values.reshape((dim_len, dim_len))
            ax = axs[rt, ct]
            sns.heatmap(
                values,
                ax=ax,
                cmap="PuOr",
                square=True,
                cbar_kws={"label": "EMD"},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{mode_1} vs {mode_2}")
    fig.tight_layout()
    fig.savefig(
        pairwise_emd_dir / f"{distance_measure}_emd.png", dpi=300
    )
    plt.show()
    fig.suptitle(f"{distance_measure} EMD")

# %%
