# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist, squareform

from tqdm import tqdm
import trimesh
import seaborn as sns
import pandas as pd
from scipy.stats import wasserstein_distance, ttest_ind, ks_2samp


from cellpack_analysis.utilities.data_tools import (
    get_positions_dictionary_from_file,
    combine_multiple_seeds_to_dictionary,
    get_pairwise_wasserstein_distance_dict,
)
from matplotlib.collections import PatchCollection
from cellpack_analysis.utilities.analysis_tools import (
    divide_pdfs,
    normalize_distances,
    ripley_k,
)
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 14})

# %% set file paths and setup parameters
base_datadir = Path(__file__).parents[2] / "data"
base_results_dir = Path(__file__).parents[2] / "results"

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
# %% create label dictionary
MODE_LABELS = {
    "observed_data": "Observed Data",
    "mean_count_and_size": "Mean Count and Size",
    "variable_size": "Variable Size",
    "variable_count": "Variable Count",
    "variable_count_and_size": "Variable Count and Size",
    "random": "Random",
    "nucleus_moderate": "Bias towards nucleus",
    "nucleus_moderate_invert": "Bias away from nucleus",
}
STRUCTURE_NAME_DICT = {
    "SLC25A17": "peroxisome",
    "RAB5A": "endosome",
}
# set structure ID
STRUCTURE_ID = "SLC25A17"
STRUCTURE_NAME = STRUCTURE_NAME_DICT[STRUCTURE_ID]
# %% distance measures to use
distance_measures = ["pairwise", "nucleus", "nearest"]
# %% set recalculate flag
recalculate = False
# %% set suffix
# normalization = "cell_diameter"  # options: None, "intracellular_radius", "cell_diameter", "max_distance"
normalization = "cell_diameter"  # options: None, "intracellular_radius", "cell_diameter", "max_distance"
if normalization is not None:
    suffix = f"_normalized_{normalization}"

# %% read position data from outputs
if recalculate:
    print("Reading position data from outputs")
    all_positions = {}
    for mode in packing_modes:
        print(mode)
        if mode == "observed_data":
            file_path = (
                base_datadir
                / f"structure_data/{STRUCTURE_ID}/sample_8d/positions_{STRUCTURE_ID}.json"
            )
        else:
            data_folder = (
                base_datadir
                / f"packing_outputs/8d_sphere_data/RS/{STRUCTURE_NAME}/spheresSST/"
            )

            file_path = (
                data_folder / f"all_positions_{STRUCTURE_NAME}_analyze_{mode}.json"
            )

            if file_path.exists() and not recalculate:
                print(f"Loading {file_path}")
                positions = get_positions_dictionary_from_file(file_path)
                all_positions[mode] = positions
                continue

            combine_multiple_seeds_to_dictionary(
                data_folder,
                ingredient_key=f"membrane_interior_{STRUCTURE_NAME}",
                search_prefix="positions_",
                rule_name=mode,
                save_name=f"all_positions_{STRUCTURE_NAME}_analyze_{mode}",
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
                base_datadir
                / f"structure_data/{STRUCTURE_ID}/meshes/nuc_mesh_{seed}.obj"
            )
            mem_mesh_path = (
                base_datadir
                / f"structure_data/{STRUCTURE_ID}/meshes/mem_mesh_{seed}.obj"
            )
        nuc_grid_distance_path = (
            base_datadir
            / f"structure_data/{STRUCTURE_ID}/grid_distances/nuc_distances_{seed}.npy"
        )
        mem_grid_distance_path = (
            base_datadir
            / f"structure_data/{STRUCTURE_ID}/grid_distances/mem_distances_{seed}.npy"
        )
        nuc_grid_distances = np.load(nuc_grid_distance_path)
        mem_grid_distances = np.load(mem_grid_distance_path)

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
            "nuc_bounds": nuc_bounds,
            "cell_bounds": cell_bounds,
            "intracellular_radius": intracellular_radius,
            "nuc_grid_distances": nuc_grid_distances,
            "mem_grid_distances": mem_grid_distances,
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
    all_z_distances = {}  # distance from z-axis

    for mode, position_dict in all_positions.items():
        print(mode)
        all_pairwise_distances[mode] = {}
        all_nuc_distances[mode] = {}
        all_nearest_distances[mode] = {}
        all_z_distances[mode] = {}
        for seed, positions in tqdm(position_dict.items()):
            if "mean" in mode:
                seed_to_use = "mean"
            else:
                seed_to_use = seed.split("_")[0]

            all_distances = cdist(positions, positions, metric="euclidean")

            # Distance from the nucleus surface
            nuc_mesh = mesh_information_dict[seed_to_use]["nuc_mesh"]
            nuc_distances = -trimesh.proximity.signed_distance(nuc_mesh, positions)
            all_nuc_distances[mode][seed_to_use] = nuc_distances

            # Nearest neighbor distance
            nearest_distances = np.min(
                all_distances + np.eye(len(positions)) * 1e6, axis=1
            )
            all_nearest_distances[mode][seed_to_use] = nearest_distances

            # Pairwise distance
            pairwise_distances = squareform(all_distances)
            all_pairwise_distances[mode][seed_to_use] = pairwise_distances

            # Z distance
            z_min = mesh_information_dict[seed_to_use]["cell_bounds"][:, 2].min()
            z_distances = positions[:, 2] - z_min
            all_z_distances[mode][seed_to_use] = z_distances

    # save distance dictionaries
    for distance_dict, distance_measure in zip(
        [
            all_pairwise_distances,
            all_nuc_distances,
            all_nearest_distances,
            all_z_distances,
        ],
        ["pairwise", "nucleus", "nearest", "z"],
    ):
        file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
        with open(file_path, "wb") as f:
            pickle.dump(distance_dict, f)

# %% load saved distance dictionary
print("Loading saved distance dictionaries")
all_distance_dict = {}
for distance_measure in ["pairwise", "nucleus", "nearest", "z"]:
    file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
    with open(file_path, "rb") as f:
        all_distance_dict[distance_measure] = pickle.load(f)
# %% Normalize distances
if normalization is not None:
    all_distance_dict = normalize_distances(
        all_distance_dict, normalization, mesh_information_dict
    )
# %% plot distance PDFs
print("Plotting distance distributions")
num_rows = len(distance_measures)
num_cols = len(packing_modes)

fig, axs = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 3, num_rows * 3),
    dpi=300,
    sharex="row",
    sharey="row",
)

for i, distance_measure in enumerate(distance_measures):
    distance_dict = all_distance_dict[distance_measure]
    print(distance_measure)
    for j, mode in enumerate(packing_modes):
        mode_dict = distance_dict[mode]
        # for j, (mode, mode_dict) in enumerate(distance_dict.items()):
        #     if mode not in packing_modes:
        #         continue
        print(mode)
        cmap = plt.get_cmap("jet", len(mode_dict))

        # plot individual kde plots of distance distributions
        combined_available_space_kde = []
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            sns.kdeplot(
                distances, ax=axs[i, j], color=cmap(k + 1), linewidth=1, alpha=0.2
            )

        # plot combined kde plot of distance distributions
        combined_mode_distances = np.concatenate(list(mode_dict.values()))
        sns.kdeplot(combined_mode_distances, ax=axs[i, j], color=cmap(0), linewidth=2)

        # plot mean distance and add title
        mean_distance = combined_mode_distances.mean()
        title_str = f"Mean: {mean_distance:.2f}"
        if i == 0:
            axs[i, j].set_title(f"{MODE_LABELS[mode]}\n{title_str}")
        else:
            axs[i, j].set_title(title_str)

        axs[i, j].axvline(mean_distance, color="k", linestyle="--")

        # add y label
        if j == 0:
            axs[i, j].set_ylabel(f"{distance_measure} PDF")
distance_label = "distance"
if normalization is not None:
    distance_label = f"{distance_label} / {normalization}"
fig.supxlabel(distance_label)
fig.tight_layout()
fig.savefig(figures_dir / f"distance_distributions{suffix}.png", dpi=300)

plt.show()
# %% plot distance distribution histograms
print("Plotting distance histograms")
num_rows = len(distance_measures)
num_cols = len(packing_modes)

fig_hist, axs_hist = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 3, num_rows * 3),
    dpi=300,
    sharex="row",
    sharey="row",
)

for i, distance_measure in enumerate(distance_measures):
    distance_dict = all_distance_dict[distance_measure]
    print(distance_measure)
    for j, mode in enumerate(packing_modes):
        mode_dict = distance_dict[mode]

        combined_mode_distances = np.concatenate(list(mode_dict.values()))

        # plot histogram
        axs_hist[i, j].hist(combined_mode_distances, bins=50, alpha=0.5, color=cmap(0))

        # plot mean distance and add title
        mean_distance = combined_mode_distances.mean()
        title_str = f"Mean: {mean_distance:.2f}"
        if i == 0:
            axs_hist[i, j].set_title(f"{MODE_LABELS[mode]}\n{title_str}")
        else:
            axs_hist[i, j].set_title(title_str)

        axs_hist[i, j].axvline(mean_distance, color="k", linestyle="--")

        # add x and y labels
        axs_hist[i, j].set_xlabel("distance")
        axs_hist[i, j].set_ylabel(f"{distance_measure} counts")

fig_hist.tight_layout()
fig_hist.savefig(figures_dir / f"distance_distributions_hist{suffix}.png", dpi=300)

plt.show()

# %% Get pairwise earth movers distances between distance distributions
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
for distance_measure in distance_measures:
    emd_dict = all_pairwise_emd[distance_measure]
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
corr_df_dict = {}
for distance_measure in distance_measures:
    emd_dict = all_pairwise_emd[distance_measure]
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
# %% plot EMD heatmaps
print("Plotting EMD variation heatmaps")
for distance_measure in distance_measures:
    corr_dict = corr_df_dict[distance_measure]
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
r_max = 0.5
num_bins = 100
r_values = np.linspace(0, r_max, num_bins)
for mode, position_dict in all_positions.items():
    print(mode)
    all_ripleyK[mode] = {}
    for seed, positions in tqdm(position_dict.items()):
        radius = mesh_information_dict[seed]["cell_diameter"] / 2
        volume = 4 / 3 * np.pi * radius**3
        mean_k_values, _ = ripley_k(
            positions, volume, r_max, num_bins=num_bins, norm_factor=(radius * 2)
        )  # normalized ripley k values
        # mean_k_values, _ = ripley_k(
        #     positions, volume, r_max, num_bins=num_bins, norm_factor=1
        # )  # non-normalized ripley k values
        all_ripleyK[mode][seed] = mean_k_values
    mean_ripleyK[mode] = np.mean(
        np.array([np.array(v) for v in all_ripleyK[mode].values()]), axis=0
    )
    ci_ripleyK[mode] = np.percentile(
        np.array([np.array(v) for v in all_ripleyK[mode].values()]), [2.5, 97.5], axis=0
    )
# %% plot ripleyK distributions
fig, ax = plt.subplots(dpi=300)
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

    ax.plot(r_values, mode_values, label=MODE_LABELS[mode])
    ax.fill_between(r_values, err_values[0], err_values[1], alpha=0.2)
# ax.axhline(1, color="k", linestyle="--", label=baseline_mode)
ax.set_xlabel("r")
ax.set_ylabel("$K(r) - \\pi r^2$")
ax.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
# fig.tight_layout()
fig.savefig(figures_dir / f"ripleyK{suffix}.png", dpi=300)
plt.show()

# %% create kde dictionary
if recalculate:
    distance_dict = all_distance_dict["nucleus"]
    kde_dict = {}
    for mode in packing_modes:
        print(mode)
        # if mode == "nucleus_moderate_invert":
        #    continue
        mode_dict = distance_dict[mode]
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            if seed not in kde_dict:
                kde_dict[seed] = {}
                available_space_distances = mesh_information_dict[seed][
                    "nuc_grid_distances"
                ].flatten()
                # available_space_distances = available_space_distances[
                #     available_space_distances >= 0
                # ]
                if normalization is not None:
                    available_space_distances /= mesh_information_dict[seed][
                        normalization
                    ]
                kde_available_space = gaussian_kde(available_space_distances)
                kde_dict[seed]["available_distance"] = {
                    "distances": available_space_distances,
                    "kde": kde_available_space,
                }
            kde_distance = gaussian_kde(distances)
            kde_dict[seed][mode] = {
                "distances": distances,
                "kde": kde_distance,
            }

    # save kde dictionary
    file_path = results_dir / f"packing_modes_kde_vals{suffix}.dat"
    with open(file_path, "wb") as f:
        pickle.dump(kde_dict, f)
# %% load kde dictionary
file_path = results_dir / f"packing_modes_kde_vals{suffix}.dat"
with open(file_path, "rb") as f:
    kde_dict = pickle.load(f)
# %% plot individual space corrected individual kde values
print("Plotting individual space corrected kde values")
fig, ax = plt.subplots(dpi=300)
cmap = plt.get_cmap("tab10")
for ct, mode in enumerate(packing_modes):
    if mode == "nucleus_moderate_invert":
        continue
    print(mode)
    mode_dict = distance_dict[mode]
    for k, (seed, distances) in tqdm(
        enumerate(mode_dict.items()), total=len(mode_dict)
    ):
        kde_distance = kde_dict[seed][mode]["kde"]
        kde_available_space = kde_dict[seed]["available_distance"]["kde"]
        xvals = np.linspace(distances.min(), distances.max(), 100)
        kde_distance_values = kde_distance(xvals)
        kde_distance_values_normalized = kde_distance_values / np.trapz(
            kde_distance_values, xvals
        )
        kde_available_space_values = kde_available_space(xvals)
        kde_available_space_values_normalized = kde_available_space_values / np.trapz(
            kde_available_space_values, xvals
        )
        yvals = kde_distance_values_normalized / kde_available_space_values_normalized
        if k == 0:
            ax.plot(xvals, yvals, label=MODE_LABELS[mode], c=cmap(ct), alpha=0.2)
        else:
            ax.plot(xvals, yvals, c=cmap(ct), alpha=0.25, label="_nolegend_")
        # if k > 10:
        #     break
    # break
ax.axhline(1, color="k", linestyle="--")
ax.set_xlim([0, 0.2])
ax.set_ylim([0, 4])
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
ax.legend()
ax.set_xlabel("distance")
ax.set_ylabel("individual space corrected kde")
plt.tight_layout()
plt.show()
fig.savefig(figures_dir / f"space_corrected_kde{suffix}.png", dpi=300)
# %% get combined space corrected kde
if recalculate:
    distance_dict = all_distance_dict["nucleus"]
    combined_kde_dict = {}
    for mode in packing_modes:
        print(mode)
        mode_dict = distance_dict[mode]

        combined_mode_distances = np.concatenate(list(mode_dict.values()))
        kde_distance = gaussian_kde(combined_mode_distances)

        combined_available_distances = np.concatenate(
            [
                mesh_information_dict[seed]["nuc_grid_distances"].flatten()
                / mesh_information_dict[seed]["cell_diameter"]
                for seed in mode_dict.keys()
            ]
        )
        kde_available_space = gaussian_kde(combined_available_distances)

        combined_kde_dict[mode] = {
            "mode_distances": combined_mode_distances,
            "kde_distance": kde_distance,
            "kde_available_space": kde_available_space,
        }
    file_path = results_dir / f"combined_space_corrected_kde{suffix}.dat"
    with open(file_path, "wb") as f:
        pickle.dump(combined_kde_dict, f)
# %% load combined space corrected kde values
file_path = results_dir / f"combined_space_corrected_kde{suffix}.dat"
with open(file_path, "rb") as f:
    combined_kde_dict = pickle.load(f)
# %% plot combined space corrected kde
print("Plotting combined space corrected kde values")
fig, ax = plt.subplots(dpi=300)
for mode, kde_mode_dict in combined_kde_dict.items():
    if mode == "nucleus_moderate_invert":
        continue
    print(mode)
    mode_distances = kde_mode_dict["mode_distances"]
    kde_distance = kde_mode_dict["kde_distance"]
    kde_available_space = kde_mode_dict["kde_available_space"]
    xvals = np.linspace(mode_distances.min(), mode_distances.max(), 100)
    kde_distance_values = kde_distance(xvals)
    kde_distance_values /= np.trapz(kde_distance_values, xvals)
    kde_available_space_values = kde_available_space(xvals)
    kde_available_space_values /= np.trapz(kde_available_space_values, xvals)
    yvals = kde_distance_values / kde_available_space_values
    ax.plot(xvals, yvals, label=MODE_LABELS[mode])
ax.axhline(1, color="k", linestyle="--")
ax.set_ylim([0, 4])
ax.set_xlim([0, 0.2])
ax.set_xlabel("distance")
ax.set_ylabel("PDF(distance) / PDF(available space)")
ax.legend()
plt.show()
fig.savefig(figures_dir / f"combined_space_corrected_kde{suffix}.png", dpi=300)
# %% get EMD between occupied and available distances
if recalculate:
    emd_occupancy_dict = {}
    for mode in packing_modes:
        print(mode)
        mode_dict = distance_dict[mode]
        emd_occupancy_dict[mode] = {}
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            occupied_distance = kde_dict[seed][mode]["distances"]
            available_distance = kde_dict[seed]["available_distance"]["distances"]
            emd_occupancy_dict[mode][seed] = wasserstein_distance(
                occupied_distance, available_distance
            )
    # save occupancy EMD dictionary
    file_path = results_dir / f"packing_modes_occupancy_emd{suffix}.dat"
    with open(file_path, "wb") as f:
        pickle.dump(emd_occupancy_dict, f)
# %% load occupancy EMD dictionary
file_path = results_dir / f"packing_modes_occupancy_emd{suffix}.dat"
with open(file_path, "rb") as f:
    emd_occupancy_dict = pickle.load(f)
# %% plot occupancy EMD values
fig, ax = plt.subplots(dpi=300)
cmap = plt.get_cmap("tab10")
for ct, mode in enumerate(packing_modes):
    emd_values = list(emd_occupancy_dict[mode].values())
    mean_emd = np.mean(emd_values)
    sns.kdeplot(emd_values, ax=ax, label=MODE_LABELS[mode], c=cmap(ct))
    ax.axvline(mean_emd, color=cmap(ct), linestyle="--", label="_nolegend_")
ax.set_xlabel("EMD")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / f"occupancy_emd{suffix}.png", dpi=300)
plt.show()
# %% create box and whisker plot for occupancy EMD values
fig, ax = plt.subplots(dpi=300)
emd_df = pd.DataFrame(emd_occupancy_dict)
sns.boxplot(data=emd_df, ax=ax)
xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
ax.set_xticklabels(xticklabels, rotation=45)
ax.set_ylabel("EMD")
fig.tight_layout()
fig.savefig(figures_dir / f"occupancy_emd_boxplot{suffix}.png", dpi=300)
# %% run t test for occupancy EMD values
for mode_1 in packing_modes:
    for mode_2 in packing_modes:
        if mode_1 == mode_2:
            continue
        t_stat, p_val = ttest_ind(
            list(emd_occupancy_dict[mode_1].values()),
            list(emd_occupancy_dict[mode_2].values()),
        )
        print(f"{mode_1} vs {mode_2}: p_val = {p_val}")
# %% run ks test for occupancy distributions
if recalculate:
    ks_occupancy_dict = {}
    for mode in packing_modes:
        print(mode)
        mode_dict = distance_dict[mode]
        ks_occupancy_dict[mode] = {}
        for k, (seed, distances) in tqdm(
            enumerate(mode_dict.items()), total=len(mode_dict)
        ):
            occupied_distance = kde_dict[seed][mode]["distances"]
            available_distance = kde_dict[seed]["available_distance"]["distances"]
            ks_stat, p_val = ks_2samp(occupied_distance, available_distance)
            ks_occupancy_dict[mode][seed] = p_val
    file_path = results_dir / f"packing_modes_occupancy_ks{suffix}.dat"
    with open(file_path, "wb") as f:
        pickle.dump(ks_occupancy_dict, f)
# %% load ks test results
file_path = results_dir / f"packing_modes_occupancy_ks{suffix}.dat"
with open(file_path, "rb") as f:
    ks_occupancy_dict = pickle.load(f)
# %% plot ks test results
fig, ax = plt.subplots(dpi=300)
ks_df = pd.DataFrame(ks_occupancy_dict)
ks_df_test = ks_df < 0.05
ax = sns.barplot(data=ks_df_test, ax=ax)
xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
ax.set_xticklabels(xticklabels, rotation=45)
ax.set_ylabel("Fraction of observations with KS test p < 0.05")
plt.tight_layout()
plt.show()
fig.savefig(figures_dir / f"occupancy_ks_test{suffix}.png", dpi=300)
# %% ks test between observed and other modes
ks_observed_dict = {}
distance_dict = all_distance_dict["nucleus"]
for mode in packing_modes:
    if mode == "observed_data":
        continue
    print(mode)
    mode_dict = distance_dict[mode]
    ks_observed_dict[mode] = {}
    for k, (seed, distances) in tqdm(
        enumerate(mode_dict.items()), total=len(mode_dict)
    ):
        observed_distances = distance_dict["observed_data"][seed]
        mode_distances = distance_dict[mode][seed]
        ks_stat, p_val = ks_2samp(observed_distances, mode_distances)
        ks_observed_dict[mode][seed] = p_val
# %% plot ks test results
fig, ax = plt.subplots(dpi=300)
ks_df = pd.DataFrame(ks_observed_dict)
ks_df_test = ks_df < 0.05
ax = sns.barplot(data=ks_df_test, ax=ax)
xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
ax.set_xticklabels(xticklabels, rotation=45)
ax.set_ylabel("Fraction of observations with KS test p < 0.05")
plt.tight_layout()
plt.show()
fig.savefig(figures_dir / f"observed_ks_test{suffix}.png", dpi=300)
