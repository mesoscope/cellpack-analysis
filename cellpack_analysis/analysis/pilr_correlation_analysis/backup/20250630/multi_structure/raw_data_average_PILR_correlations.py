# %% [markdown]
# # Average PILR correlation analysis
# This workflow is used to compare the average PILRs and the radial profiles of the PILRs
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cellpack_analysis.lib.mesh_tools import get_average_shape_mesh_objects
from cellpack_analysis.lib.PILR_tools import (
    add_contour_to_axis,
    get_domain,
    get_parametrized_coords_for_avg_shape,
    get_processed_PILR_from_dict,
    get_projection,
    morph_PILRs_into_average_shape,
)
from cellpack_analysis.lib.plotting_tools import plot_and_save_center_slice

log = logging.getLogger(__name__)

# %% [markdown]
# ## Define channel names
STRUCTURE_IDS = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
    # "random": "Random",
    # "nucleus_gradient_strong": "Nucleus Gradient",
    # "membrane_gradient_strong": "Membrane Gradient",
}
num_struct = len(STRUCTURE_IDS)

# %% [markdown]
# ## Load average PILRs
folder_id = "raw_data"
recalculate = False

base_folder = Path(__file__).parents[4] / "results/PILR_correlation_analysis/"
results_folder = base_folder / "average_PILR_correlations"
results_folder.mkdir(parents=True, exist_ok=True)

avg_pilr = {}

log.info("Calculating average PILRs")
for structure_id in STRUCTURE_IDS:
    structure_folder = base_folder / structure_id / folder_id
    with open(structure_folder / "avg_PILR.json") as f:
        struct_pilr = json.load(f)

    struct_pilr = {k: np.expand_dims(np.array(v), 0) for k, v in struct_pilr.items()}
    avg_pilr[structure_id] = struct_pilr[structure_id]


# %% [markdown]
# ## Get domain and coords_param for the average shape meshes
mesh_path = Path(__file__).parents[4] / "data/average_shape_meshes"
mesh_dict = get_average_shape_mesh_objects(mesh_path)
domain = get_domain(mesh_dict)
domain = np.transpose(domain, (1, 0, 2))
domain_nuc = (255 * (domain > 1)).astype(np.uint8)
domain_mem = (255 * (domain > 0)).astype(np.uint8)
coords_param = get_parametrized_coords_for_avg_shape(domain)

# %% [markdown]
# ## Morph PILRs into the average shape
morphed_pilrs = {}
for structure_id, pilr in avg_pilr.items():
    pilr_vals = morph_PILRs_into_average_shape(
        pilr_list=[pilr],
        domain=domain,
        coords_param=coords_param,
        mesh_dict=mesh_dict,
    )[0]
    pilr_vals = pilr_vals / np.max(pilr_vals)
    morphed_pilrs[structure_id] = pilr_vals

# %% [markdown]
# ## Plot the center slice of the morphed PILRs
dim_to_axis_map = {
    0: "XY",
    1: "XZ",
    2: "YZ",
}

fig, axs = plt.subplots(
    2, len(STRUCTURE_IDS), figsize=(len(STRUCTURE_IDS) * 2, 4), dpi=300
)
for ct, (structure_id, morphed_pilr) in enumerate(morphed_pilrs.items()):
    morphed_pilr = (255 * (morphed_pilr / np.max(morphed_pilr))).astype(np.uint8)
    morphed_pilr[domain > 0] += 3
    morphed_pilr[domain == 0] = 0
    for dim in range(2):
        fig, ax = plot_and_save_center_slice(
            morphed_pilr,
            structure=STRUCTURE_IDS[structure_id],
            dim=dim,
            title=f"{STRUCTURE_IDS[structure_id]} {dim_to_axis_map[dim]}",
            ax=axs[dim][ct] if len(morphed_pilrs) > 1 else axs[dim],
            ylabel=f"{dim_to_axis_map[dim]}" if ct == 0 else None,
            showfig=False,
            add_contour=True,
            domain_nuc=domain_nuc,
            domain_mem=domain_mem,
            vmin=0,
            pct_max=97.5,
            lw=1,
            fontcolor="black",
        )
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot the average overlap PILRs
threshold_pct = 95
projections = {
    "mean": "Mean Projection",
    "center": "Center Slice",
    "max": "Max Projection",
}
projection = "center"
struct_x_list = ["SLC25A17", "RAB5A"]
num_x = len(struct_x_list)
struct_y_list = ["SEC61B", "ST6GAL1"]
num_y = len(struct_y_list)
for dim in range(2):
    figsize = (num_x * 2.5, num_y * 2) if dim == 0 else (num_y * 2.5, num_x * 1.5)
    fig, axs = plt.subplots(
        num_y,
        num_x,
        figsize=figsize,
        dpi=300,
    )
    fig.set_facecolor((0, 0, 0, 0))
    for ct1, struct_x in enumerate(struct_x_list):
        morphed_pilr1 = morphed_pilrs[struct_x]
        morphed_pilr1 = (255 * (morphed_pilr1 / np.max(morphed_pilr1))).astype(np.uint8)
        morphed_pilr1[domain > 0] += 3
        morphed_pilr1[domain == 0] = 0
        for ct2, struct_y in enumerate(struct_y_list):
            morphed_pilr2 = morphed_pilrs[struct_y]
            morphed_pilr2 = (255 * (morphed_pilr2 / np.max(morphed_pilr2))).astype(
                np.uint8
            )
            morphed_pilr2[domain > 0] += 3
            morphed_pilr2[domain == 0] = 0
            ax = axs[ct2, ct1] if len(morphed_pilrs) > 1 else axs
            # if ct2 > ct1:
            #     ax.set_visible(False)
            #     continue
            combined_pilr = (morphed_pilr1 + morphed_pilr2) / 2
            overlap_fraction = 1
            # if ct1 != ct2:
            threshold1 = np.percentile(morphed_pilr1, threshold_pct)
            threshold2 = np.percentile(morphed_pilr2, threshold_pct)
            above_threshold1 = morphed_pilr1 > threshold1
            above_threshold2 = morphed_pilr2 > threshold2
            overlap_fraction = np.sum(
                np.logical_and(above_threshold1, above_threshold2)
            ) / np.sum(np.logical_or(above_threshold1, above_threshold2))
            combined_pilr[
                np.logical_or(morphed_pilr1 < threshold1, morphed_pilr2 < threshold2)
            ] = 0
            maxval = np.nanmax(combined_pilr)
            combined_pilr = (255 * (combined_pilr / np.max(combined_pilr))).astype(
                np.uint8
            )
            combined_pilr[domain > 0] += 3

            # ax.set_title(f"Overlap: {overlap_fraction:.2f}", color="white")

            pilr_proj = get_projection(combined_pilr, dim, projection)

            maxval = np.nanmax(pilr_proj)
            # if maxval != 0:
            #     pilr_proj = pilr_proj / maxval
            vmin = 0
            vmax = np.min([threshold1, threshold2])
            ax.imshow(
                pilr_proj,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            ax = add_contour_to_axis(ax, projection, dim, domain_nuc, domain_mem)
            ax.xaxis.set_visible(False)
            # make spines (the box) invisible
            plt.setp(ax.spines.values(), visible=False)
            # remove ticks and labels for the left axis
            ax.tick_params(left=False, labelleft=False)
            # remove background patch (only needed for non-white background)
            ax.patch.set_visible(False)

            if ct1 == 0:
                ax.set_ylabel(STRUCTURE_IDS[struct_y], color="black")
            if ct2 == num_y - 1:
                ax.set_xlabel(STRUCTURE_IDS[struct_x], color="black")
    # fig.suptitle(
    #     (
    #         f"Threshold: {threshold_pct} percentile, "
    #         f"{projections[projection]}, {dim_to_axis_map[dim]}"
    #     ),
    #     c="w",
    #     fontsize=10,
    # )
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Plot the radial profile
fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=300)

for structure_id, pilr in avg_pilr.items():
    pilr = pilr.squeeze()
    pilrsum = np.sum(pilr, axis=1)
    pilrsum = pilrsum / np.sum(pilrsum)
    ax.plot(
        np.linspace(0, 1, len(pilrsum)),
        pilrsum,
        label=STRUCTURE_IDS[structure_id],
        linewidth=2,
        alpha=1,
        # color=ch_colors_dict[ch],
    )

ax.axvline(0.5, ls="--", color="black", label="Nuclear boundary")
ax.set_xlabel("Normalized distance from the nuclear centroid")
ax.set_ylabel("Normalized radial PILR")
ax.set_xlim((0, 1))
ax.set_ylim((0, 0.07))
# put legend on the right outside the axis
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
fig.savefig(f"{results_folder}/avg_PILR_squeezed.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# # Calculate correlations between average PILRs
structure_id_list = list(STRUCTURE_IDS.values())
df = pd.DataFrame(index=structure_id_list, columns=structure_id_list, dtype=float)
std_df = pd.DataFrame(index=structure_id_list, columns=structure_id_list, dtype=float)

# %% [markdown]
# ## Process PILRs and calculate correlations
mask_nucleus = True
average_over_phi = False

for ch_ind, ch_name in STRUCTURE_IDS.items():
    if ch_ind not in avg_pilr:
        continue
    pilr1 = get_processed_PILR_from_dict(
        avg_pilr, ch_ind, average_over_phi=average_over_phi, mask_nucleus=mask_nucleus
    )
    std_pilr1 = np.std(pilr1)

    for ch_ind2, ch_name2 in STRUCTURE_IDS.items():
        if ch_ind2 not in avg_pilr:
            continue
        pilr2 = get_processed_PILR_from_dict(
            avg_pilr,
            ch_ind2,
            average_over_phi=average_over_phi,
            mask_nucleus=mask_nucleus,
        )

        corrcoef = np.corrcoef(pilr1, pilr2)[0, 1]
        df.loc[ch_name, ch_name2] = corrcoef
        std_pilr2 = np.std(pilr2)
        std_df.loc[ch_name, ch_name2] = (
            np.sqrt((1 - corrcoef**2) / (len(pilr1) - 2)) * std_pilr1 * std_pilr2
        )
print(df)

# %% [markdown]
# ## Create heatmap
mask = np.zeros_like(df, dtype=bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
sns.heatmap(
    df,
    ax=ax,
    annot=True,
    cmap="viridis",
    mask=mask,
)
fig.savefig(f"{results_folder}/avg_PILR_correlation_heatmap.png", dpi=300)

# %%
