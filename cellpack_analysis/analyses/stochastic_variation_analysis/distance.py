import pickle
from turtle import distance
from matplotlib import lines
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from trimesh import proximity
from tqdm import tqdm
from scipy.spatial.distance import cdist, squareform
from scipy.stats import wasserstein_distance, gaussian_kde, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection

from cellpack_analysis.analyses.stochastic_variation_analysis.label_tables import (
    MODE_LABELS,
    VARIABLE_SHAPE_MODES,
)
from cellpack_analysis.analyses.stochastic_variation_analysis.stats_functions import (
    ripley_k,
)


def get_distance_dictionary(
    all_positions,
    distance_measures,
    mesh_information_dict,
    results_dir=None,
    recalculate=False,
):
    """
    Calculate or load distance measures between particles in different modes.

    Parameters:
        all_positions (dict): A dictionary containing positions of particles in different modes.
        mesh_information_dict (dict): A dictionary containing mesh information.
        results_dir (str, optional): The directory to save or load distance dictionaries. Defaults to None.
        recalculate (bool, optional): Whether to recalculate the distance measures. Defaults to False.

    Returns:
        dict: A dictionary containing distance measures between particles in different modes.
    """
    if not recalculate and results_dir is not None:
        # load saved distance dictionary
        print("Loading saved distance dictionaries")
        all_distance_dict = {}
        for distance_measure in distance_measures:
            file_path = results_dir / f"packing_modes_{distance_measure}_distances.dat"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    all_distance_dict[distance_measure] = pickle.load(f)
            else:
                print(f"File not found: {file_path}")
        if len(all_distance_dict) == len(distance_measures):
            return all_distance_dict

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
            if mode in VARIABLE_SHAPE_MODES:
                seed_to_use = seed.split("_")[0]
            else:
                seed_to_use = seed

            all_distances = cdist(positions, positions, metric="euclidean")

            # Distance from the nucleus surface
            nuc_mesh = mesh_information_dict.get(
                seed_to_use, mesh_information_dict["mean"]
            )["nuc_mesh"]
            nuc_distances = -proximity.signed_distance(nuc_mesh, positions)
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
            z_min = mesh_information_dict.get(
                seed_to_use, mesh_information_dict["mean"]
            )["cell_bounds"][:, 2].min()
            z_distances = positions[:, 2] - z_min
            all_z_distances[mode][seed_to_use] = z_distances

    # save distance dictionaries
    if results_dir is not None:
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

    all_distance_dict = {
        "pairwise": all_pairwise_distances,
        "nucleus": all_nuc_distances,
        "nearest": all_nearest_distances,
        "z": all_z_distances,
    }

    return all_distance_dict


def plot_distance_distributions_kde(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=None,
    suffix="",
    normalization=None,
):
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
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            print(f"Distance measure: {distance_measure}, Mode: {mode}")
            cmap = plt.get_cmap("jet", len(mode_dict))

            # plot individual kde plots of distance distributions
            for k, (seed, distances) in tqdm(
                enumerate(mode_dict.items()), total=len(mode_dict)
            ):
                ax = axs[i, j]
                color = cmap(k)

                sns.kdeplot(distances, ax=ax, color=color, linewidth=1, alpha=0.2)

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))

            # plot mean distance and add title
            mean_distance = combined_mode_distances.mean()
            title_str = f"Mean: {mean_distance:.2f}"

            if i == 0:
                ax.set_title(f"{MODE_LABELS[mode]}\n{title_str}")
            else:
                ax.set_title(title_str)

            ax.axvline(mean_distance, color="k", linestyle="--")

            # add y label
            if j == 0:
                ax.set_ylabel(f"{distance_measure} PDF")
    distance_label = "distance"
    if normalization is not None:
        distance_label = f"{distance_label} / {normalization}"

    fig.supxlabel(distance_label)
    fig.tight_layout()

    if figures_dir is not None:
        fig.savefig(figures_dir / f"distance_distributions{suffix}.png", dpi=300)

    plt.show()


def plot_ks_test_distance_distributions_kde(
    distance_measures,
    packing_modes,
    all_distance_dict,
    ks_observed_dict,
    figures_dir=None,
    suffix="",
    normalization=None,
    baseline_mode=None,
    significance_level=0.05,
):
    print("Plotting distance distributions")

    if baseline_mode is not None:
        packing_modes = [mode for mode in packing_modes if mode != baseline_mode]

    num_cols = len(packing_modes)

    for i, distance_measure in enumerate(distance_measures):
        distance_dict = all_distance_dict[distance_measure]
        ks_measure_dict = ks_observed_dict[distance_measure]
        fig, axs = plt.subplots(
            2,
            num_cols,
            figsize=(num_cols * 4, 8),
            dpi=300,
            sharex="row",
            sharey="row",
        )
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            ks_mode_dict = ks_measure_dict[mode]
            print(f"Distance measure: {distance_measure}, Mode: {mode}")

            sig_ax = axs[0, j]
            ns_ax = axs[1, j]

            # get significant and non-significant seeds
            all_seeds = list(mode_dict.keys())
            significant_seeds = [
                seed
                for seed, p_value in ks_mode_dict.items()
                if p_value < significance_level
            ]
            ns_seeds = [seed for seed in all_seeds if seed not in significant_seeds]

            # kde plots for significant seeds
            for seed in tqdm(significant_seeds):
                distances = mode_dict[seed]
                sns.kdeplot(distances, ax=sig_ax, color="r", linewidth=1, alpha=0.2)
            # plot combined kde plot of distance distributions
            if len(significant_seeds) > 0:
                combined_sig_distances = np.concatenate(
                    [
                        distance
                        for seed, distance in mode_dict.items()
                        if seed in significant_seeds
                    ]
                )
                sns.kdeplot(combined_sig_distances, ax=sig_ax, color="k", linewidth=2)
                # plot mean distance and add title
                mean_distance = combined_sig_distances.mean()
                title_str = (
                    f"Mean: {mean_distance:.2f}\nKS p-value < {significance_level}"
                )
                sig_ax.axvline(mean_distance, color="k", linestyle="--")
            else:
                title_str = "No significant obs."
            if i == 0:
                sig_ax.set_title(f"{MODE_LABELS[mode]}\n{title_str}")
            else:
                sig_ax.set_title(title_str)

            # kde plots for non-significant seeds
            for seed in tqdm(ns_seeds):
                distances = mode_dict[seed]
                sns.kdeplot(distances, ax=ns_ax, color="g", linewidth=1, alpha=0.2)
            # plot combined kde plot of distance distributions
            if len(ns_seeds) > 0:
                combined_ns_distances = np.concatenate(
                    [
                        distance
                        for seed, distance in mode_dict.items()
                        if seed in ns_seeds
                    ]
                )
                sns.kdeplot(combined_ns_distances, ax=ns_ax, color="k", linewidth=2)
                # plot mean distance and add title
                mean_distance = combined_ns_distances.mean()
                title_str = (
                    f"Mean: {mean_distance:.2f}\nKS p-value >= {significance_level}"
                )
                ns_ax.axvline(mean_distance, color="k", linestyle="--")
            else:
                title_str = "No non-significant obs."
            ns_ax.set_title(title_str)

        distance_label = "distance"
        if normalization is not None:
            distance_label = f"{distance_label} / {normalization}"
        fig.supylabel(f"{distance_measure} PDF")
        fig.supxlabel(distance_label)
        fig.tight_layout()

        if figures_dir is not None:
            fig.savefig(
                figures_dir
                / f"distance_distributions_ks_test_{distance_measure}{suffix}.png",
                dpi=300,
            )

    plt.show()


def plot_distance_distributions_overlay(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=None,
    suffix="",
    normalization=None,
):
    print("Plotting overlaid distance distributions")

    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
        cmap = plt.get_cmap("tab10")
        for j, mode in enumerate(packing_modes):
            mode_dict = distance_dict[mode]
            print(f"Distance measure: {distance_measure}, Mode: {mode}")

            # plot combined kde plot of distance distributions
            combined_mode_distances = np.concatenate(list(mode_dict.values()))
            sns.kdeplot(
                combined_mode_distances,
                ax=ax,
                color=cmap(j),
                linewidth=2,
                label=MODE_LABELS[mode],
            )

            # plot mean distance
            mean_distance = combined_mode_distances.mean()
            ax.axvline(mean_distance, color=cmap(j), linestyle="--", label="_nolegend_")

        distance_label = "distance"
        if normalization is not None:
            distance_label = f"{distance_label} / {normalization}"
        if distance_measure == "nearest":
            xlim = ax.get_xlim()
            ax.set_xlim([xlim[0], 0.2])
        ax.set_xlabel(distance_label)
        ax.set_title(f"{distance_measure} distance")
        ax.legend()
        fig.tight_layout()

        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"distance_distributions_{distance_measure}_{suffix}.png",
                dpi=300,
            )

        plt.show()


def plot_distance_distributions_histogram(
    distance_measures,
    packing_modes,
    all_distance_dict,
    figures_dir=None,
    suffix="",
    normalization=None,
):
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
            axs_hist[i, j].hist(
                combined_mode_distances,
                bins=50,
            )

            # plot mean distance and add title
            mean_distance = combined_mode_distances.mean()
            title_str = f"Mean: {mean_distance:.2f}\nn={len(combined_mode_distances)}"
            if i == 0:
                axs_hist[i, j].set_title(f"{MODE_LABELS[mode]}\n{title_str}")
            else:
                axs_hist[i, j].set_title(title_str)

            axs_hist[i, j].axvline(mean_distance, color="k", linestyle="--")

            # add x and y labels
            axs_hist[i, j].set_ylabel(f"{distance_measure} counts")

    distance_label = "distance"
    if normalization is not None:
        distance_label = f"{distance_label} / {normalization}"
    fig_hist.supxlabel(distance_label)
    fig_hist.tight_layout()

    if figures_dir is not None:
        fig_hist.savefig(
            figures_dir / f"distance_distributions_hist{suffix}.png", dpi=300
        )

    plt.show()


def get_ks_observed_dict(
    distance_measures,
    packing_modes,
    all_distance_dict,
    baseline_mode="observed_data",
):
    ks_observed_dict = {}
    for distance_measure in distance_measures:
        distance_dict = all_distance_dict[distance_measure]
        ks_observed_dict[distance_measure] = {}
        for mode in packing_modes:
            if mode == baseline_mode:
                continue
            print(
                f"KS test between {baseline_mode} and {mode}, distance: {distance_measure}"
            )
            mode_dict = distance_dict[mode]
            ks_observed_dict[distance_measure][mode] = {}
            for seed, distances in tqdm(mode_dict.items(), total=len(mode_dict)):
                observed_distances = distance_dict[baseline_mode][seed]
                _, p_val = ks_2samp(observed_distances, distances)
                ks_observed_dict[distance_measure][mode][seed] = p_val
    return ks_observed_dict


def plot_ks_observed_barplots(
    ks_observed_dict,
    figures_dir=None,
    suffix="",
    significance_level=0.05,
):
    for distance_measure, ks_observed_mode_dict in ks_observed_dict.items():
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
        ks_df = pd.DataFrame(ks_observed_mode_dict)
        ks_df_test = ks_df >= significance_level
        ax = sns.barplot(data=ks_df_test, ax=ax)
        xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_ylabel(
            f"Fraction of observations with\n p-value >= {significance_level}"
        )
        ax.set_title(f"KS test for {distance_measure} distance")
        ax.set_ylim(0, 0.65)
        plt.tight_layout()
        if figures_dir is not None:
            fig.savefig(
                figures_dir / f"observed_ks_test_{distance_measure}{suffix}.png",
                dpi=300,
            )
        plt.show()


def get_pairwise_wasserstein_distance_dict(
    distribution_dict_1,
    distribution_dict_2=None,
):
    """
    distribution_dict is a dictionary with distances or other values for multiple seeds
    it has the form: {seed1: [value1, value2, ...], seed2: [value1, value2, ...], ...}

    The output has the form: {(seed1, seed2): distance, (seed1, seed3): distance, ...}
    """
    pairwise_wasserstein_distances = {}

    keys_1 = list(distribution_dict_1.keys())
    if distribution_dict_2 is None:
        for i in tqdm(range(len(keys_1))):
            for j in range(i + 1, len(keys_1)):
                seed_1 = keys_1[i]
                seed_2 = keys_1[j]
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_1[seed_2]
                )
    else:
        keys_2 = list(distribution_dict_2.keys())
        for i in tqdm(range(len(keys_1))):
            for j in range(len(keys_2)):
                seed_1 = keys_1[i]
                seed_2 = keys_2[j]
                pairwise_wasserstein_distances[(seed_1, seed_2)] = wasserstein_distance(
                    distribution_dict_1[seed_1], distribution_dict_2[seed_2]
                )

    return pairwise_wasserstein_distances


def get_pairwise_emd_dictionary(
    all_distance_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    baseline_mode=None,
    suffix="",
):
    if not recalculate and results_dir is not None:
        # load saved EMD dictionary
        file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                all_pairwise_emd = pickle.load(f)
            return all_pairwise_emd

    print("Calculating pairwise EMDs")
    all_pairwise_emd = {}
    for distance_measure, distribution_dict in all_distance_dict.items():
        print(distance_measure)
        measure_pairwise_emd = {}
        for mode_1 in packing_modes:
            if baseline_mode is not None and mode_1 != baseline_mode:
                continue
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
    if results_dir is not None:
        file_path = results_dir / f"packing_modes_pairwise_emd{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(all_pairwise_emd, f)

    return all_pairwise_emd


def plot_pairwise_emd_heatmaps(
    distance_measures,
    all_pairwise_emd,
    pairwise_emd_dir=None,
    suffix="",
):
    print("Plotting pairwise EMD heatmaps")
    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure]
        fig, axs = plt.subplots(len(emd_dict), len(emd_dict), figsize=(10, 10), dpi=300)
        for rt, (mode_1, mode_1_dict) in enumerate(emd_dict.items()):
            mode_1_label = MODE_LABELS[mode_1]
            for ct, (mode_2, emd) in enumerate(mode_1_dict.items()):
                print(distance_measure, mode_1, mode_2)
                mode_2_label = MODE_LABELS[mode_2]
                if mode_1 == mode_2:
                    values = squareform(list(emd.values()))
                else:
                    values = list(emd.values())
                    values = np.array(values)
                    dim_len = int(np.sqrt(len(values)))
                    values = values.reshape((dim_len, -1))
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
                    ax.set_title(mode_2_label)
                if ct == 0:
                    ax.set_ylabel(mode_1_label)
        fig.suptitle(f"{distance_measure} EMD")
        fig.tight_layout()

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd{suffix}.png", dpi=300
            )

        plt.show()


def plot_average_emd_correlation_heatmap(
    distance_measures,
    all_pairwise_emd,
    pairwise_emd_dir=None,
    baseline_mode="observed_data",
    suffix="",
):
    print("calculating correlations for EMDs")
    mode_names = list(all_pairwise_emd["pairwise"].keys())
    corr_df_dict = {}
    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure]
        corr_df_dict[distance_measure] = {}
        df_corr = pd.DataFrame(columns=mode_names, index=mode_names)
        df_std = pd.DataFrame(columns=mode_names, index=mode_names)
        fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
        for mode_1, mode_1_dict in emd_dict.items():
            for mode_2, emd in mode_1_dict.items():
                if not np.isnan(df_corr.loc[mode_1, mode_2]):
                    continue
                if mode_1 == mode_2:
                    values = squareform(list(emd.values()))
                else:
                    values = list(emd.values())
                    values = np.array(values)
                    dim_len = int(np.sqrt(len(values)))
                    values = values.reshape((dim_len, -1))
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
        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_heatmap{suffix}.png",
                dpi=300,
            )

    return corr_df_dict


def plot_emd_correlation_circles(
    distance_measures,
    corr_df_dict,
    pairwise_emd_dir=None,
    suffix="",
):
    print("Plotting EMD variation circles")

    for distance_measure in distance_measures:
        corr_dict = corr_df_dict[distance_measure]
        df_corr = corr_dict["mean"]
        df_std = corr_dict["std"]

        N = M = len(df_corr)
        xlabels = ylabels = [MODE_LABELS.get(col, col) for col in df_corr.columns]

        xvals = np.arange(M)
        yvals = np.arange(N)

        x, y = np.meshgrid(xvals, yvals)
        s = df_std.to_numpy()
        c = df_corr.to_numpy()

        fig, ax = plt.subplots(dpi=300)

        R = s / s.max() / 2
        circles = [Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
        col = PatchCollection(
            circles, array=c.flatten(), cmap="Reds", edgecolor="black"
        )
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

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_heatmap_circ{suffix}.png",
                dpi=300,
            )
        plt.show()


def plot_emd_boxplots(
    distance_measures,
    all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=None,
    suffix="",
):
    print("Plotting EMD variation boxplots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
        sns.boxplot(data=emd_df, ax=ax, whis=(0, 100))
        ax.set_ylabel(f"EMD")
        ax.set_title(f"{distance_measure} distance")
        ax.set_xticklabels([MODE_LABELS[m] for m in emd_df.columns], rotation=45)
        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_boxplot{suffix}.png",
                dpi=300,
            )


def plot_emd_histograms(
    distance_measures,
    all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=None,
    suffix="",
):
    print("Plotting EMD variation histograms")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        ncol = len(emd_df.columns)
        fig, axs = plt.subplots(
            1, ncol, dpi=300, sharey=True, sharex=True, figsize=(ncol * 5, 5)
        )
        for ct, col in enumerate(emd_df.columns):
            axs[ct].hist(emd_df[col], bins=50, density=True)
            axs[ct].set_title(
                f"{MODE_LABELS[col]}\nn={np.count_nonzero(~np.isnan(emd_df[col]))}"
            )
        fig.supxlabel(f"{distance_measure} EMD")
        fig.supylabel("Density")
        plt.tight_layout()

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_boxplot{suffix}.png",
                dpi=300,
            )


def plot_emd_kdeplots(
    distance_measures,
    all_pairwise_emd,
    baseline_mode="observed_data",
    pairwise_emd_dir=None,
    suffix="",
):
    print("Plotting EMD variation kde plots")

    for distance_measure in distance_measures:
        emd_dict = all_pairwise_emd[distance_measure][baseline_mode]
        emd_df = pd.DataFrame.from_dict(emd_dict)
        fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
        for col in emd_df.columns:
            sns.kdeplot(emd_df[col], ax=ax, label=MODE_LABELS[col])
        ax.set_xlabel(f"{distance_measure} EMD")
        ax.legend()
        fig.tight_layout()

        if pairwise_emd_dir is not None:
            fig.savefig(
                pairwise_emd_dir / f"{distance_measure}_emd_kdeplot{suffix}.png",
                dpi=300,
            )


def calculate_ripley_k(
    all_positions,
    mesh_information_dict,
):
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
                positions, volume, r_values, norm_factor=(radius * 2)
            )
            all_ripleyK[mode][seed] = mean_k_values
        mean_ripleyK[mode] = np.mean(
            np.array([np.array(v) for v in all_ripleyK[mode].values()]), axis=0
        )
        ci_ripleyK[mode] = np.percentile(
            np.array([np.array(v) for v in all_ripleyK[mode].values()]),
            [2.5, 97.5],
            axis=0,
        )

    return all_ripleyK, mean_ripleyK, ci_ripleyK, r_values


def plot_ripley_k(
    mean_ripleyK,
    ci_ripleyK,
    r_values,
    figures_dir=None,
    suffix="",
):
    print("Plotting Ripley K")
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
    for mode, mode_values in mean_ripleyK.items():
        mode_values = mode_values - np.pi * r_values**2
        err_values = ci_ripleyK[mode] - np.pi * r_values**2

        ax.plot(r_values, mode_values, label=MODE_LABELS[mode])
        ax.fill_between(r_values, err_values[0], err_values[1], alpha=0.2)
    ax.set_xlabel("r")
    ax.set_ylabel("$K(r) - \\pi r^2$")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)

    if figures_dir is not None:
        fig.savefig(figures_dir / f"ripleyK{suffix}.png", dpi=300)

    plt.show()


def get_space_corrected_kde(
    all_distance_dict,
    mesh_information_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
    normalization=None,
):
    if not recalculate and results_dir is not None:
        file_path = results_dir / f"packing_modes_kde_vals{suffix}.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                kde_dict = pickle.load(f)
            return kde_dict

    print("Calculating space corrected kde values")
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
    if results_dir is not None:
        file_path = results_dir / f"packing_modes_kde_vals{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(kde_dict, f)

    return kde_dict


def plot_space_corrected_kde_illustration(
    distance_dict,
    kde_dict,
    baseline_mode="random",
    figures_dir=None,
    suffix="",
):
    print("Plotting space corrected kde illustration")
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300, figsize=(7, 7))
    mode_dict = distance_dict[baseline_mode]
    seed = list(mode_dict.keys())[0]
    distances = mode_dict[seed]

    kde_distance = kde_dict[seed][baseline_mode]["kde"]
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

    # plot occupied distance values
    ax = axs[0]
    ax.plot(xvals, kde_distance_values_normalized, c="r")
    ax.set_xlim([0, 0.2])
    ax.axhline(1, color="k", linestyle="--")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.set_title("occupied space")

    # plot available space values
    ax = axs[1]
    ax.plot(
        xvals, kde_available_space_values_normalized, label="available space", c="b"
    )
    ax.set_xlim([0, 0.2])
    ax.set_ylim(axs[0].get_ylim())
    ax.axhline(1, color="k", linestyle="--")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.set_title("available space")

    # plot ratio
    ax = axs[2]
    ax.plot(xvals, yvals, label="space corrected", c="g")
    ax.set_xlim([0, 0.2])
    ax.set_ylim([0, 2])
    ax.axhline(1, color="k", linestyle="--")
    ax.set_xlabel("distance")
    ax.set_ylabel("conditional density")
    ax.set_title("ratio")
    plt.tight_layout()
    plt.show()

    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"space_corrected_kde_illustration{suffix}.png",
            dpi=300,
        )
    plt.show()


def plot_individual_space_corrected_kde(
    distance_dict,
    kde_dict,
    packing_modes,
    figures_dir=None,
    suffix="",
    mesh_information_dict=None,
    struct_diameter=4.74,
):
    print("Plotting individual space corrected kde values")
    cmap = plt.get_cmap("tab10")
    nrows = len(packing_modes)
    fig, axs = plt.subplots(dpi=300, figsize=(9, 3 * nrows), nrows=nrows)
    for ct, mode in enumerate(packing_modes):
        print(mode)
        mode_dict = distance_dict[mode]
        ax = axs[ct]
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
            kde_available_space_values_normalized = (
                kde_available_space_values / np.trapz(kde_available_space_values, xvals)
            )
            yvals = (
                kde_distance_values_normalized / kde_available_space_values_normalized
            )
            if k == 0:
                ax.plot(xvals, yvals, label=MODE_LABELS[mode], c=cmap(ct), alpha=0.2)
            else:
                ax.plot(xvals, yvals, c=cmap(ct), alpha=0.25, label="_nolegend_")
            # break

        if mesh_information_dict is not None and struct_diameter:
            avg_diameter, std_diameter = get_average_scaled_diameter(
                struct_diameter=struct_diameter,
                mesh_information_dict=mesh_information_dict,
            )
            ax.axvspan(
                avg_diameter - std_diameter,
                avg_diameter + std_diameter,
                color="r",
                alpha=0.2,
            )
            ax.axvline(avg_diameter, color="r", linestyle="--")

        ax.axhline(1, color="k", linestyle="--")
        ax.set_xlim([0, 0.2])
        ax.set_ylim([0, 4])
        ax.set_xlabel("distance")
        ax.set_ylabel("conditional PDF")
        ax.set_title(f"{MODE_LABELS[mode]}")
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(
            figures_dir / f"individual_space_corrected_kde_{suffix}.png",
            dpi=300,
        )
    plt.show()


def get_combined_space_corrected_kde(
    all_distance_dict,
    mesh_information_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
):
    if not recalculate and results_dir is not None:
        # load combined space corrected kde values
        file_path = results_dir / f"combined_space_corrected_kde{suffix}.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                combined_kde_dict = pickle.load(f)
            return combined_kde_dict

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
    if results_dir is not None:
        file_path = results_dir / f"combined_space_corrected_kde{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(combined_kde_dict, f)

    return combined_kde_dict


def plot_combined_space_corrected_kde(
    combined_kde_dict,
    packing_modes,
    figures_dir,
    suffix="",
    mesh_information_dict=None,
    struct_diameter=2.37,
    normalization=None,
):
    print("Plotting combined space corrected kde values")
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
    for ct, mode in enumerate(packing_modes):
        print(mode)
        kde_mode_dict = combined_kde_dict[mode]
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
        if mesh_information_dict is not None and struct_diameter and ct == 0:
            avg_diameter, std_diameter = get_average_scaled_diameter(
                struct_diameter=struct_diameter, mesh_information_dict=mesh_information_dict
            )
            ax.axvspan(
                avg_diameter - std_diameter,
                avg_diameter + std_diameter,
                color="r",
                alpha=0.2,
            )
            ax.axvline(avg_diameter, color="r", linestyle="--")
        if ct == 0:
            ax.axhline(1, color="k", linestyle="--")
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, 0.2])
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        xlabel = "distance"
        if normalization:
            xlabel = f"{xlabel} / {normalization}"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("conditional PDF")
        ax.legend()
        plt.show()
        fig.savefig(figures_dir / f"combined_space_corrected_kde_{ct}{suffix}.png", dpi=300)
    plt.show()
    ax.set_aspect(0.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=len(packing_modes))
    fig.savefig(figures_dir / f"combined_space_corrected_kde_aspect{suffix}.png", dpi=300)
    return fig, ax


def get_occupancy_emd(
    distance_dict,
    kde_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
):
    if not recalculate and results_dir is not None:
        file_path = results_dir / f"packing_modes_occupancy_emd{suffix}.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                emd_occupancy_dict = pickle.load(f)
            return emd_occupancy_dict

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
    if results_dir is not None:
        file_path = results_dir / f"packing_modes_occupancy_emd{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(emd_occupancy_dict, f)

    return emd_occupancy_dict


def plot_occupancy_emd_kdeplot(
    emd_occupancy_dict,
    packing_modes,
    figures_dir=None,
    suffix="",
):
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for ct, mode in enumerate(packing_modes):
        emd_values = list(emd_occupancy_dict[mode].values())
        mean_emd = np.mean(emd_values)
        sns.kdeplot(emd_values, ax=ax, label=MODE_LABELS[mode], c=cmap(ct))
        ax.axvline(mean_emd, color=cmap(ct), linestyle="--", label="_nolegend_")
    ax.set_xlabel("EMD")
    ax.legend()
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / f"occupancy_emd{suffix}.png", dpi=300)
    plt.show()


def plot_occupancy_emd_boxplot(
    emd_occupancy_dict,
    figures_dir=None,
    suffix="",
):
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
    emd_df = pd.DataFrame(emd_occupancy_dict)
    sns.boxplot(data=emd_df, ax=ax)
    xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel("EMD")
    fig.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / f"occupancy_emd_boxplot{suffix}.png", dpi=300)
    plt.show()


def get_occupancy_ks_test_dict(
    distance_dict,
    kde_dict,
    packing_modes,
    results_dir=None,
    recalculate=False,
    suffix="",
):
    if not recalculate and results_dir is not None:
        # load ks test results
        file_path = results_dir / f"packing_modes_occupancy_ks{suffix}.dat"
        if file_path.exists():
            with open(file_path, "rb") as f:
                ks_occupancy_dict = pickle.load(f)
            return ks_occupancy_dict

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
            _, p_val = ks_2samp(occupied_distance, available_distance)
            ks_occupancy_dict[mode][seed] = p_val
    if results_dir is not None:
        file_path = results_dir / f"packing_modes_occupancy_ks{suffix}.dat"
        with open(file_path, "wb") as f:
            pickle.dump(ks_occupancy_dict, f)

    return ks_occupancy_dict


def plot_occupancy_ks_test(
    ks_occupancy_dict,
    figures_dir=None,
    suffix="",
):
    fig, ax = plt.subplots(dpi=300, figsize=(7, 7))
    ks_df = pd.DataFrame(ks_occupancy_dict)
    ks_df_test = ks_df < 0.05
    ax = sns.barplot(data=ks_df_test, ax=ax)
    xticklabels = [MODE_LABELS[mode._text] for mode in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_ylabel(
        "Fraction of observations with\n occupancy different from available space"
    )
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / f"occupancy_ks_test{suffix}.png", dpi=300)
    plt.show()


def get_average_scaled_diameter(struct_diameter, mesh_information_dict):
    print("Calculating average scaled diameter")
    scaled_diameter = []
    for _, mesh_info in mesh_information_dict.items():
        scaled_diameter.append(struct_diameter / mesh_info["cell_diameter"])
    scaled_diameter = np.array(scaled_diameter)
    average_scaled_diameter = np.mean(scaled_diameter)
    std_scaled_diameter = np.std(scaled_diameter)
    return average_scaled_diameter, std_scaled_diameter
