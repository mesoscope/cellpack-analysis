# %% [markdown]
# # Compare average PILRs for observed and simulated data
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cellpack_analysis.lib.plotting_tools import plot_PILR

# %% [markdown]
# Read in average PILR

# %%
STRUCTURE_ID = "SLC25A17"
EXPERIMENT = "rules_shape"

# %%
base_folder = (
    Path(__file__).parents[3] / f"results/PILR_correlation_analysis/{STRUCTURE_ID}/{EXPERIMENT}/"
)

# %%
with open(f"{base_folder}/avg_PILR.json") as f:
    avg_pilr = json.load(f)

# %%
for key in avg_pilr.keys():
    avg_pilr[key] = np.array(avg_pilr[key])

# %%
print(avg_pilr.keys())

# %% [markdown]
# Channel name dictionary

# %%
ch_names_dict = {
    "SLC25A17": "Peroxisome",
    # "RAB5A": "Endosome",
    "random": "Random",
    # "membrane_weak_gradient": "Membrane Weak",
    # "membrane_moderate": "Membrane Moderate",
    "membrane_gradient_strong": "Membrane",
    # "nucleus_weak_gradient": "Nucleus Weak",
    # "nucleus_moderate": "Nucleus Moderate",
    "nucleus_gradient_strong": "Nucleus Strong",
    # "membrane_weak_gradient_invert": "Membrane Weak inverted",
    # "membrane_moderate_invert": "Membrane Moderate inverted",
    # "membrane_strong_gradient_invert": "Membrane Strong inverted",
    # "nucleus_weak_gradient_invert": "Nucleus Weak inverted",
    # "nucleus_moderate_invert": "Nucleus Moderate inverted",
    # "nucleus_strong_gradient_invert": "Nucleus Strong inverted",
    "apical_gradient": "Apical gradient",
}

# %%
ch_colors_dict = {key: f"C{i}" for i, key in enumerate(ch_names_dict.keys())}

# %% [markdown]
# Set raw image channel

# %%
raw_image_channel = STRUCTURE_ID
raw_image_label = ch_names_dict[raw_image_channel]

# %%
raw_image_channel = None
raw_image_label = "Peroxisome"

# %% [markdown]
# ### Save average PILR images

# %%
for ch, pilr in avg_pilr.items():

    # if ch == raw_image_channel:
    #     vmin, vmax = np.percentile(pilr, [1, 90])
    # else:
    #     vmin, vmax = None, None

    plot_PILR(
        pilr,
        ch_name=ch,
        # save_dir=base_folder,
        # vmin=vmin,
        # vmax=vmax,
    )

# %% [markdown]
# ### Plot radial profile

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=300)
# for ch in avg_pilr:
# for ch in ["SLC25A17", "membrane", "random", "nucleus_square"]:
# for ch in [raw_image_channel, "nucleus_strong_gradient", "nucleus_strong_gradient_invert", "membrane_strong_gradient", "membrane_strong_gradient_invert"]:
for ch in ch_names_dict:
    if ch not in avg_pilr:
        continue
    pilr = avg_pilr[ch]
    pilrsum = np.sum(pilr, axis=1)
    pilrsum = pilrsum / np.sum(pilrsum)
    linewidth = 4 if ch == raw_image_channel else 2
    alpha = 1 if ch == raw_image_channel else 0.6
    ax.plot(
        np.linspace(0, 1, len(pilrsum)),
        pilrsum,
        label=ch_names_dict[ch],
        linewidth=linewidth,
        alpha=alpha,
        color=ch_colors_dict[ch],
    )

ax.axvline(0.5, ls="--", color="black", label="Nuclear boundary")
ax.set_xlabel("Normalized distance from the nuclear centroid")
ax.set_ylabel("Normalized radial PILR")
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim([0, 1])
ax.set_ylim([0, 0.06])
# put legend on the right outside the axis
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
# ax.legend()
fig.savefig(f"{base_folder}/avg_PILR_squeezed.png", dpi=300, bbox_inches="tight")


# %% [markdown]
# ### Get correlations between average PILRs

# %%
df = pd.DataFrame(index=ch_names_dict.values(), columns=ch_names_dict.values(), dtype=float)
std_df = pd.DataFrame(index=ch_names_dict.values(), columns=ch_names_dict.values(), dtype=float)

# %% [markdown]
# ### Calculate correlations between channels

# %%
from cellpack_analysis.utilities.PILR_tools import get_processed_PILR_from_dict

# %%
mask_nucleus = True
average_over_phi = False

for ch_ind, ch_name in ch_names_dict.items():
    if ch_ind not in avg_pilr:
        continue
    pilr1, std_pilr1 = get_processed_PILR_from_dict(
        avg_pilr, ch_ind, average_over_phi=average_over_phi, mask_nucleus=mask_nucleus
    )

    for ch_ind2, ch_name2 in ch_names_dict.items():
        if ch_ind2 not in avg_pilr:
            continue
        pilr2, std_pilr2 = get_processed_PILR_from_dict(
            avg_pilr,
            ch_ind2,
            average_over_phi=average_over_phi,
            mask_nucleus=mask_nucleus,
        )

        corrcoef = np.corrcoef(pilr1, pilr2)[0, 1]
        df.loc[ch_name, ch_name2] = corrcoef
        std_df.loc[ch_name, ch_name2] = (
            np.sqrt((1 - corrcoef**2) / (len(pilr1) - 2)) * std_pilr1 * std_pilr2
        )


# %%
df = df.sort_values(by=raw_image_label, ascending=False).sort_values(
    by=raw_image_label, axis=1, ascending=False
)
print(df)

# %%
bar_vals = df.loc[raw_image_label, [col for col in df.columns if col != raw_image_label]].values
bar_errs = std_df.loc[raw_image_label, [col for col in df.columns if col != raw_image_label]].values

# %% [markdown]
# Create bar plot with values and errors

# %%
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
y_pos = np.arange(len(bar_vals))

ax.barh(
    y_pos,
    bar_vals,
    xerr=bar_errs,
    color="gray",
)
ax.set_yticks(y_pos, labels=[col for col in df.columns if col != raw_image_label])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel("Correlation")
# ax.set_xlim([-0.3, 0.3])
plt.show()


# %% [markdown]
# ### Create heatmap

# %%
import seaborn as sns

# %%
df_sorted = df.sort_values(by=raw_image_label, ascending=False).sort_values(
    by=raw_image_label, axis=1, ascending=False
)

# %%
df_sorted

# %%
save_dir = (
    "/allen/aics/animated-cell/Saurabh/cellpack-analysis/results/SLC25A17/sample_8d_actual_shape"
)

# %%
mask = np.zeros_like(df, dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.close("all")
pex_vals = df.loc[raw_image_label].values
min_val = np.min(pex_vals[pex_vals > 0])
max_val = np.max(pex_vals[pex_vals < 0.9])
sns.heatmap(
    df,
    annot=True,
    cmap="cool",
    vmin=min_val,
    vmax=max_val,
    mask=mask,
)
# plt.savefig(
#     f"{save_dir}/PILR_correlation_bias_avg_phi.png",
#     dpi=300,
#     bbox_inches="tight",
# )

# %% [markdown]
# ### Plot different gradients

# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
def normalize_values(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))


# %%
xvals = np.linspace(0.001, 1, 1000)
weak_vals = normalize_values(np.exp(-xvals / 0.9))  # (1 - xvals)
mod_vals = normalize_values(np.exp(-xvals / 0.3))  # (1 - xvals) ** 3
str_vals = normalize_values(np.exp(-xvals / 0.1))  # (1 - xvals) ** 8

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
ax.plot(xvals, weak_vals, label="Weak", linewidth=4, c="C3")
ax.plot(xvals, mod_vals, label="Moderate", linewidth=4, c="C4")
ax.plot(xvals, str_vals, label="Strong", linewidth=4, c="C5")
ax.legend(fontsize=18)
ax.set_yticks([])
ax.set_xticks([0, 1])
ax.set_xticklabels(["nucleus", "cell membrane"], fontsize=18)
ax.set_ylabel("Relative probability", fontsize=18)
plt.show()

# %% [markdown]
# Plot average PILR

# %%
pilr_vals = avg_pilr["planar_gradient_Z"]


# %%
fig, ax = plot_PILR(
    pilr_vals,
    ch_name=None,
    save_dir=None,
    suffix="",
    aspect=20,
    vmin=0,
    max_pct=90,
)
fig.show()
