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
from cellpack_analysis.utilities.analysis_tools import normalize_distances, ripley_k
from matplotlib.patches import Circle

# %% set file paths and setup parameters
base_datadir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/")
base_results_dir = Path("/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/")

results_dir = base_results_dir / "stochastic_variation_analysis/rules/"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

packing_modes = [
    "observed_data",
    # "mean_count_and_size",
    # "variable_size",
    # "variable_count",
    # "variable_count_and_size",
    "random",
    "nucleus_moderate",
    "nucleus_moderate_invert",
]
# %% set recalculate flag
recalculate = True
# %% set suffix
# normalization = "cell_diameter"  # options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = None  # options: None, "intracellular_radius", "cell_diameter", "max_distance"
if normalization is not None:
    suffix = f"_normalized_{normalization}"

# %% read data from cellpack outputs
if recalculate:
    print("Reading data from cellpack outputs")
    all_positions = {}
    for mode in packing_modes:
        print(mode)
        if mode == "observed_data":
            file_path = (
                base_datadir
                / "structure_data/SLC25A17/sample_8d/positions_SLC25A17.json"
            )
        else:
            data_folder = (
                base_datadir
                / f"packing_outputs/8d_sphere_data/RS/peroxisome/spheresSST/"
            )

            file_path = data_folder / f"all_positions_peroxisome_analyze_{mode}.json"

            if file_path.exists() and not recalculate:
                print(f"Loading {file_path}")
                positions = get_positions_dictionary_from_file(file_path)
                all_positions[mode] = positions
                continue

            combine_multiple_seeds_to_dictionary(
                data_folder,
                ingredient_key="membrane_interior_peroxisome",
                search_prefix="positions_",
                rule_name=mode,
                save_name=f"all_positions_peroxisome_analyze_{mode}",
            )

        positions = get_positions_dictionary_from_file(
            file_path,
            drop_random_seed=True,
        )

        all_positions[mode] = positions
    # save all positions dictionary
    file_path = results_dir / "packing_modes_positions.dat"
    with open(file_path, "wb") as f:
        pickle.dump(all_positions, f)

# %% load saved positions dictionary
file_path = results_dir / "packing_modes_positions.dat"
with open(file_path, "rb") as f:
    all_positions = pickle.load(f)

# %% check number of packings for each result
for mode, position_dict in all_positions.items():
    print(mode, len(position_dict))
# %% calculate mesh information
recalculate_mesh = False
if recalculate_mesh:
    print("Calculating mesh information")
    mesh_information_dict = {}
    seed_list = all_positions["observed_data"].keys()

    for seed in tqdm([*seed_list, "mean"], total=len(seed_list) + 1):
        if seed == "mean":
            nuc_mesh_path = base_datadir / "average_shape_meshes/nuc_mesh_mean.obj"
            mem_mesh_path = base_datadir / "average_shape_meshes/mem_mesh_mean.obj"
        else:
            nuc_mesh_path = (
                base_datadir / f"structure_data/SLC25A17/meshes/nuc_mesh_{seed}.obj"
            )
            mem_mesh_path = (
                base_datadir / f"structure_data/SLC25A17/meshes/mem_mesh_{seed}.obj"
            )

        nuc_mesh = trimesh.load_mesh(str(nuc_mesh_path))
        mem_mesh = trimesh.load_mesh(str(mem_mesh_path))

        nuc_bounds = nuc_mesh.bounds
        cell_bounds = mem_mesh.bounds

        nuc_diameter = np.diff(nuc_bounds, axis=0).max()
        cell_diameter = np.diff(cell_bounds, axis=0).max()

        intracellular_radius = (cell_diameter - nuc_diameter) / 2

        mesh_information_dict[seed] = {
            "nuc_mesh": nuc_mesh,
            "mem_mesh": mem_mesh,
            "nuc_diameter": nuc_diameter,
            "cell_diameter": cell_diameter,
            "intracellular_radius": intracellular_radius,
        }

    # save mesh information dictionary
    file_path = results_dir / "mesh_information.dat"
    with open(file_path, "wb") as f:
        pickle.dump(mesh_information_dict, f)
# %% load mesh information
file_path = results_dir / "mesh_information.dat"
with open(file_path, "rb") as f:
    mesh_information_dict = pickle.load(f)

# %% Calculate distance measures
if recalculate:
    print("Calculating distance measures")
    all_pairwise_distances = {}  # pairwise distance between particles
    all_nuc_distances = {}  # distance to nucleus surface
    all_nearest_distances = {}  # distance to nearest neighbor

    for mode, position_dict in all_positions.items():
        print(mode)
        all_pairwise_distances[mode] = {}
        all_nuc_distances[mode] = {}
        all_nearest_distances[mode] = {}
        for seed, positions in tqdm(position_dict.items()):
            if "mean" in mode:
                seed_to_use = "mean"
            else:
                seed_to_use = seed.split("_")[0]

            nuc_mesh = mesh_information_dict[seed_to_use]["nuc_mesh"]
            nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions)

            all_nuc_distances[mode][seed_to_use] = nuc_distances

            all_distances = cdist(positions, positions, metric="euclidean")
            # Nearest neighbor distance
            nearest_distances = np.min(
                all_distances + np.eye(len(positions)) * 1e6, axis=1
            )
            all_nearest_distances[mode][seed_to_use] = nearest_distances

            # Pairwise distance
            pairwise_distances = squareform(all_distances)
            all_pairwise_distances[mode][seed_to_use] = pairwise_distances

    # save distance dictionaries
    for distance_dict, distance_measure in zip(
        [all_pairwise_distances, all_nuc_distances, all_nearest_distances],
        ["pairwise", "nucleus", "nearest"],
    ):
        file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
        with open(file_path, "wb") as f:
            pickle.dump(distance_dict, f)

# %% load saved distance dictionary
print("Loading saved distance dictionaries")
all_distance_dict = {}
for distance_measure in ["pairwise", "nucleus", "nearest"]:
    file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
    with open(file_path, "rb") as f:
        all_distance_dict[distance_measure] = pickle.load(f)
# %% Normalize distances
if normalization is not None:
    all_distance_dict = normalize_distances(
        all_distance_dict, normalization, mesh_information_dict
    )
# %% plot distance distributions
print("Plotting distance distributions")
plt.rcdefaults()
num_rows = 3
num_cols = len(packing_modes)

fig, axs = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 3, num_rows * 3),
    dpi=300,
    sharex="row",
    sharey="row",
)

fig_hist, axs_hist = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 3, num_rows * 3),
    dpi=300,
    sharex="row",
    sharey="row",
)

for i, (distance_measure, distance_dict) in enumerate(all_distance_dict.items()):
    print(distance_measure)
    for j, mode in enumerate(packing_modes):
        mode_dict = distance_dict[mode]
        # for j, (mode, mode_dict) in enumerate(distance_dict.items()):
        #     if mode not in packing_modes:
        #         continue
        print(mode)
        cmap = plt.get_cmap("jet", len(mode_dict))
        combined_mode_distances = np.concatenate(list(mode_dict.values()))
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            sns.kdeplot(distances, ax=axs[i, j], color=cmap(k), alpha=0.2)

        mean_distance = combined_mode_distances.mean()

        if i == 0:
            axs[i, j].set_title(f"{mode}\nMean: {mean_distance:.2f}")
            axs_hist[i, j].set_title(f"{mode}\nMean: {mean_distance:.2f}")
        else:
            axs[i, j].set_title(f"Mean: {mean_distance:.2f}")
            axs_hist[i, j].set_title(f"Mean: {mean_distance:.2f}")

        axs[i, j].set_xlabel("distance")
        axs_hist[i, j].set_xlabel("distance")

        if j == 0:
            axs[i, j].set_ylabel(f"{distance_measure} PDF")
            axs_hist[i, j].set_ylabel(f"{distance_measure} counts")

        axs[i, j].axvline(mean_distance, color="k", linestyle="--")
        axs_hist[i, j].hist(combined_mode_distances, bins=50, alpha=0.5, color=cmap(0))
        axs_hist[i, j].axvline(mean_distance, color="k", linestyle="--")

        # low, high = np.mean(combined_mode_distances) - 3 * np.std(
        #     combined_mode_distances
        # ), np.mean(combined_mode_distances) + 3 * np.std(combined_mode_distances)
        # axs[i, j].set_xlim(low, high)
        # axs_hist[i, j].set_xlim(low, high)

        # if "normalized" in suffix:
        # axs[i, j].set_xlim(-0.15, 1)
        # axs_hist[i, j].set_xlim(-0.15, 1)

fig.tight_layout()
fig.savefig(figures_dir / f"distance_distributions{suffix}.png", dpi=300)

fig_hist.tight_layout()
fig_hist.savefig(figures_dir / f"distance_distributions_hist{suffix}.png", dpi=300)

plt.show()

# %% Get pairwise earth movers distances
if recalculate:
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
                measure_pairwise_emd[mode_1][mode_2] = (
                    get_pairwise_wasserstein_distance_dict(
                        distribution_dict_1,
                        distribution_dict_2,
                    )
                )

                if mode_2 not in measure_pairwise_emd:
                    measure_pairwise_emd[mode_2] = {}
                measure_pairwise_emd[mode_2][mode_1] = measure_pairwise_emd[mode_1][
                    mode_2
                ]
        all_pairwise_emd[distance_measure] = measure_pairwise_emd

    # Save EMD dict
    file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
    with open(file_path, "wb") as f:
        pickle.dump(all_pairwise_emd, f)

# %% load saved EMD dictionary
file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
with open(file_path, "rb") as f:
    all_pairwise_emd = pickle.load(f)

# %% create pairwise emd folders
pairwise_emd_dir = figures_dir / "pairwise_emd"
pairwise_emd_dir.mkdir(exist_ok=True, parents=True)
# %% calculate pairwise EMD distances across modes
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
                # vmin=0,
                # vmax=0.05,
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
    fig.savefig(pairwise_emd_dir / f"{distance_measure}_emd{suffix}.png", dpi=300)
    plt.show()

# %% compare average EMDs with baseline
print("calculating correlations for EMDs")
baseline_mode = "observed_data"
mode_names = list(all_pairwise_emd["pairwise"].keys())
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
    fig.savefig(
        pairwise_emd_dir / f"{distance_measure}_emd_heatmap{suffix}.png", dpi=300
    )
print(df_corr)
# %% plot EMD heatmaps
print("Plotting EMD variation heatmaps")
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
    for item in ax.spines.__dict__["_dict"].values():
        item.set_visible(False)
    ax.tick_params(which="both", length=0)

    for i in range(N):
        for j in range(M):
            ax.text(j, i, f"{c[i, j]:.3g}", ha="center", va="center", color="black")

    cbar = fig.colorbar(col)
    cbar.set_label("EMD")
    # ax.set_frame_on(False)
    ax.set_title(f"{distance_measure} EMD")
    fig.savefig(
        pairwise_emd_dir / f"{distance_measure}_emd_heatmap_circ{suffix}.png", dpi=300
    )
    plt.show()

# %% calculate ripleyK for all positions
print("Calculating Ripley K")
all_ripleyK = {}
mean_ripleyK = {}
ci_ripleyK = {}
r_max = 50
num_bins = 1000
r_values = np.linspace(0, r_max, num_bins)
for mode, position_dict in all_positions.items():
    print(mode)
    all_ripleyK[mode] = {}
    for seed, positions in tqdm(position_dict.items()):
        radius = mesh_information_dict[seed]["cell_diameter"] / 2
        volume = 4 / 3 * np.pi * radius**3
        # mean_k_values, _ = ripley_k(
            # positions, volume, r_max, num_bins=num_bins, norm_factor=(radius * 2)
        # )
        mean_k_values, _ = ripley_k(
            positions, volume, r_max, num_bins=num_bins, norm_factor=1
        )
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

baseline_mode = "observed_data"
baseline_ripleyk_mean = mean_ripleyK[baseline_mode]
baseline_ripleyk_ci = ci_ripleyK[baseline_mode]
for mode, mode_values in mean_ripleyK.items():
    # if mode == baseline_mode:
    # continue
    # mode_values = mode_values / baseline_ripleyk_mean
    # err_values = ci_ripleyK[mode] / baseline_ripleyk_mean
    mode_values = mode_values - np.pi * r_values**2
    err_values = ci_ripleyK[mode] - np.pi * r_values**2

    ax.plot(r_values, mode_values, label=mode)
    ax.fill_between(r_values, err_values[0], err_values[1], alpha=0.2)
# ax.axhline(1, color="k", linestyle="--", label=baseline_mode)
ax.set_xlabel("r")
ax.set_ylabel("$K(r) - \\pi r^2$")
ax.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
# fig.tight_layout()
fig.savefig(figures_dir / f"ripleyK{suffix}.png", dpi=300)
plt.show()

# %%
r_max = 1
positions = np.random.rand(1000, 2) * r_max
volume = r_max ** 2
n_boot = 100
all_ripley_k = []
for bc in tqdm(range(n_boot)):
    boot_indices = np.random.choice(len(positions), len(positions), replace=True)
    ripley_k_values, r_values = ripley_k(positions[boot_indices], volume, r_max / 4)
    all_ripley_k.append(ripley_k_values)
all_ripley_k = np.array(all_ripley_k)
mean_ripley_k = np.mean(all_ripley_k, axis=0)

# %%
adj_factor = np.pi * r_values**2
adj_factor = 0
adjusted_ripley_k = mean_ripley_k - adj_factor
ci_ripley_k = np.percentile(all_ripley_k, [2.5, 97.5], axis=0) - adj_factor
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(positions[:, 0], positions[:, 1])
ax[0].set_aspect("equal")
ax[1].plot(r_values, adjusted_ripley_k)
ax[1].fill_between(r_values, ci_ripley_k[0], ci_ripley_k[1], alpha=0.2)
plt.tight_layout()
plt.show()

# %%
