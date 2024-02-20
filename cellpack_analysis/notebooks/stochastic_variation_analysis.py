# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.spatial.distance import cdist, squareform

from tqdm.notebook import tqdm
import trimesh
import seaborn as sns
import pandas as pd

from cellpack_analysis.utilities.data_tools import (
    get_positions_dictionary_from_file,
    combine_multiple_seeds_to_dictionary,
    get_pairwise_wasserstein_distance_dict,
)
from matplotlib.collections import PatchCollection
from cellpack_analysis.utilities.analysis_tools import ripleyK
from matplotlib.patches import Circle
# %% set file paths and setup parameters
base_datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/")
base_results_dir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/")

results_dir = base_results_dir / "stochastic_variation_analysis/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

RULE = "random"

packing_modes = [
    "raw_data",
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "variable_count_and_size",
    "shape",
]
# %% read data from cellpack outputs
all_positions = {}
for mode in packing_modes:
    print(mode)
    if mode == "raw_data":
        file_path = base_datadir / "structure_data/SLC25A17/sample_8d/positions_SLC25A17.json"
    else:
        data_folder = (
            base_datadir
            / f"packing_outputs/stochastic_variation_analysis/{mode}/peroxisome/spheresSST/"
        )
        if mode == "shape":
            combine_multiple_seeds_to_dictionary(
                data_folder,
                ingredient_key="membrane_interior_peroxisome",
                search_prefix="positions_",
                save_name="positions_peroxisome_analyze_random_mean",
            )

        file_path = data_folder / "positions_peroxisome_analyze_random_mean.json"

    positions = get_positions_dictionary_from_file(file_path)

    all_positions[mode] = positions
# %% save all positions dictionary
file_path = results_dir / "packing_modes_positions.dat"
with open(file_path, "wb") as f:
    pickle.dump(all_positions, f)

# %% load saved positions dictionary
file_path = results_dir / "packing_modes_positions.dat"
with open(file_path, "rb") as f:
    all_positions = pickle.load(f)

# %% Calculate distance measures
nuc_mesh_path = base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
mean_nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))

all_pairwise_distances = {}  # pairwise distance between particles
all_pairwise_distances_normalized = {}  # pairwise distance between particles
all_nuc_distances = {}  # distance to nucleus surface
all_nuc_distances_normalized = {}  # distance to nucleus surface
all_nearest_distances = {}  # distance to nearest neighbor
all_nearest_distances_normalized = {}  # distance to nearest neighbor
for mode, position_dict in all_positions.items():
    print(mode)
    all_pairwise_distances[mode] = {}
    all_pairwise_distances_normalized[mode] = {}
    all_nuc_distances[mode] = {}
    all_nuc_distances_normalized[mode] = {}
    all_nearest_distances[mode] = {}
    all_nearest_distances_normalized[mode] = {}
    for seed, positions in tqdm(position_dict.items()):
        # Distance from nucleus
        if mode not in ["shape", "raw_data"]:
            nuc_mesh = mean_nuc_mesh
        else:
            cellid = seed.split("_")[0]
            nuc_mesh_path = base_datadir / f"structure_data/SLC25A17/meshes/nuc_mesh_{cellid}.obj"
            nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))
        nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions)
        all_nuc_distances_normalized[mode][seed] = nuc_distances / np.max(nuc_distances)
        all_nuc_distances[mode][seed] = nuc_distances

        all_distances = cdist(positions, positions, metric="euclidean")
        # Nearest neighbor distance
        nearest_distances = np.min(all_distances + np.eye(len(positions)) * 1e6, axis=1)
        all_nearest_distances_normalized[mode][seed] = nearest_distances / np.max(
            nearest_distances
        )
        all_nearest_distances[mode][seed] = nearest_distances

        # Pairwise distance
        pairwise_distances = squareform(all_distances)
        all_pairwise_distances_normalized[mode][seed] = pairwise_distances / np.max(
            pairwise_distances
        )
        all_pairwise_distances[mode][seed] = pairwise_distances

# %% save distance dictionaries
for distance_dict, norm_distance_dict, distance_measure in zip(
    [all_pairwise_distances, all_nuc_distances, all_nearest_distances],
    [all_pairwise_distances_normalized, all_nuc_distances_normalized, all_nearest_distances_normalized],
    ["pairwise", "nucleus", "nearest"],
):
    file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
    with open(file_path, "wb") as f:
        pickle.dump(distance_dict, f)

    file_path = results_dir / f"packing_modes_{distance_measure}_distances_normalized.dat"
    with open(file_path, "wb") as f:
        pickle.dump(norm_distance_dict, f)

# %% load saved distance dictionary
print("Loading saved distance dictionaries")
suffix = "_normalized"
all_distance_dict = {}
for distance_measure in ["pairwise", "nucleus", "nearest"]:
    file_path = results_dir / f"packing_modes_{distance_measure}_distances{suffix}.dat"
    with open(file_path, "rb") as f:
        all_distance_dict[distance_measure] = pickle.load(f)

# %% plot distance distributions
print("Plotting distance distributions")
num_rows = 3
num_cols = len(all_positions)

fig, axs = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 3, num_rows * 3),
    dpi=300,
    sharex=True,
    sharey=True,
)

for i, (distance_measure, distance_dict) in enumerate(all_distance_dict.items()):
    for j, (mode, mode_dict) in enumerate(distance_dict.items()):
        print(mode)
        cmap = plt.get_cmap("jet", len(mode_dict))
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            sns.kdeplot(distances, ax=axs[i, j], color=cmap(k), alpha=0.2)

        if i == 0:
            axs[i, j].set_title(f"{mode}")
        # axs[i, j].set_xlim([0, 1])
        axs[i, j].set_xlabel("Distance")
        if j == 0:
            axs[i, j].set_ylabel(f"{distance_measure} PDF")

fig.tight_layout()
fig.savefig(figures_dir / "distance_distributions_norm.png", dpi=300)

plt.show()

# %% Get pairwise earth movers distances
print("Calculating pairwise EMDs")
all_pairwise_emd = {}
for distance_measure, distribution_dict in all_distance_dict.items():
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
suffix = "_normalized"
file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
with open(file_path, "wb") as f:
    pickle.dump(all_pairwise_emd, f)

# %% load saved EMD dictionary
suffix = "_normalized"
file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
with open(file_path, "rb") as f:
    all_pairwise_emd = pickle.load(f)

# %% create pairwise emd folders
pairwise_emd_dir = figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)
# %% plot individual pairwise EMD distributions
print("Plotting individual pairwise EMD distributions")
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
            if ct == len(emd_dict) - 1:
                cbar = True
                cbar_kws = {"label": "EMD"}
            else:
                cbar = False
                cbar_kws = None
            sns.heatmap(
                values,
                ax=ax,
                cmap="PuOr",
                vmin=0,
                vmax=0.05,
                square=True,
                cbar=cbar,
                cbar_kws=cbar_kws,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if rt == 0:
                ax.set_title(mode_2)
            if ct == 0:
                ax.set_ylabel(mode_1)
    fig.suptitle(f"{distance_measure} EMD")
    fig.tight_layout()
    fig.savefig(pairwise_emd_dir / f"{distance_measure}_emd_raw.png", dpi=300)
    plt.show()
    break

# %% compare average EMDs with baseline
baseline_mode = "mean_count_and_size"
mode_names = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "variable_count_and_size",
    "shape",
    "raw_data",
]
plt.rcdefaults()
corr_df_dict = {}
for distance_measure, emd_dict in all_pairwise_emd.items():
    corr_df_dict[distance_measure] = {}
    df_corr = pd.DataFrame(columns=mode_names, index=mode_names)
    df_std = pd.DataFrame(columns=mode_names, index=mode_names)
    fig, ax = plt.subplots(dpi=300)
    for rt, (mode_1, mode_1_dict) in enumerate(emd_dict.items()):
        for ct, (mode_2, emd) in enumerate(mode_1_dict.items()):
            if not np.isnan(df_corr.loc[mode_1, mode_2]):
                continue
            if mode_1 == mode_2:
                values = squareform(list(emd.values()))
            else:
                values = list(emd.values())
                values = np.array(values)
                dim_len = int(np.sqrt(len(values)))
                values = values.reshape((dim_len, dim_len))
            df_corr.loc[mode_1, mode_2] = np.mean(values)
            df_corr.loc[mode_2, mode_1] = df_corr.loc[mode_1, mode_2]
            df_std.loc[mode_1, mode_2] = np.std(values)
            df_std.loc[mode_2, mode_1] = df_std.loc[mode_1, mode_2]
    df_corr = df_corr.div(df_corr.loc[baseline_mode, baseline_mode])
    df_std = df_std.div(df_corr.loc[baseline_mode, baseline_mode])
    df_corr = df_corr.astype(float)
    df_std = df_std.astype(float)
    corr_df_dict[distance_measure]["mean"] = df_corr
    corr_df_dict[distance_measure]["std"] = df_std
    sns.heatmap(df_corr, cmap="PuOr", annot=True, ax=ax)
    plt.tight_layout()
    ax.set_title(f"{distance_measure} EMD")
    fig.savefig(pairwise_emd_dir / f"{distance_measure}_emd_heatmap_norm.png", dpi=300)
print(df_corr)
# %% plot EMD heatmaps
for distance_measure, corr_dict in corr_df_dict.items():
    df_corr = corr_dict["mean"]
    df_std = corr_dict["std"]

    N = M = len(df_corr)
    xlabels = ylabels = df_corr.columns

    xvals = np.arange(M)
    yvals = np.arange(N)

    x, y = np.meshgrid(xvals, yvals)
    s = df_std.to_numpy()
    c = df_corr.to_numpy()

    fig, ax = plt.subplots(dpi=300)

    R = s / s.max() / 2
    circles = [Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap="Reds", edgecolor="black")
    ax.add_collection(col)

    ax.set(
        xticks=np.arange(M),
        yticks=np.arange(N),
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.invert_yaxis()

    # ax.grid(which="minor")

    for i in range(N):
        for j in range(M):
            ax.text(j, i, f"{c[i, j]:.3g}", ha="center", va="center", color="black")

    cbar = fig.colorbar(col)
    cbar.set_label("EMD")
    # ax.set_frame_on(False)
    ax.set_title(f"{distance_measure} EMD")
    fig.savefig(pairwise_emd_dir / f"{distance_measure}_emd_heatmap_circ_norm.png", dpi=300)
    plt.show()

# %% calculate ripleyK for all positions
print("Calculating Ripley K")
all_ripleyK = {}
mean_ripleyK = {}
ci_ripleyK = {}
r_values = np.linspace(0, 50, 100)
for mode, position_dict in all_positions.items():
    print(mode)
    all_ripleyK[mode] = {}
    for seed, positions in tqdm(position_dict.items()):
        mean_k_values, _ = ripleyK(positions, r_values, bootstrap_count=1)
        all_ripleyK[mode][seed] = mean_k_values
    mean_ripleyK[mode] = np.mean(
        np.array([np.array(v) for v in all_ripleyK[mode].values()]), axis=0
    )
    ci_ripleyK[mode] = np.percentile(
        np.array([np.array(v) for v in all_ripleyK[mode].values()]), [2.5, 97.5], axis=0
    )
# %% plot ripleyK distributions
fig, ax = plt.subplots(dpi=300)
plt.rcParams.update({"font.size": 18})

baseline_mode = "mean_count_and_size"
baseline_ripleyk_mean = mean_ripleyK[baseline_mode]
baseline_ripleyk_ci = ci_ripleyK[baseline_mode]
for mode, mode_values in mean_ripleyK.items():
    if mode == baseline_mode:
        continue
    mode_values = mode_values / baseline_ripleyk_mean
    err_values = ci_ripleyK[mode] / baseline_ripleyk_mean

    ax.plot(r_values, mode_values, label=mode)
    ax.fill_between(r_values, err_values[0], err_values[1], alpha=0.2)
ax.axhline(1, color="k", linestyle="--", label=baseline_mode)
ax.set_xlabel("r")
ax.set_ylabel("$K(r) / K_b(r)$")
ax.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
# fig.tight_layout()
fig.savefig(figures_dir / "ripleyK.png", dpi=300)
plt.show()
# %%
