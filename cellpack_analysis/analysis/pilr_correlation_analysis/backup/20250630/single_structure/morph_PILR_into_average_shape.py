# %% [markdown]
# # Morph PILRs into the average cell and nuclear shape
import matplotlib.pyplot as plt
import numpy as np

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.label_tables import MODE_LABELS
from cellpack_analysis.lib.mesh_tools import get_average_shape_mesh_objects
from cellpack_analysis.lib.PILR_tools import (
    get_domain,
    get_parametrized_coords_for_avg_shape,
    morph_PILRs_into_average_shape,
)
from cellpack_analysis.lib.plotting_tools import plot_and_save_center_slice

# %% [markdown]
# ## Set data directory path
STRUCTURE_NAME = "peroxisome"
FOLDER_ID = "rules_shape"
datadir = get_datadir_path()

base_folder = datadir / f"PILR/{STRUCTURE_NAME}/{FOLDER_ID}"
# %% [markdown]
# ## Load average PILR
RULES = {
    "SLC25A17": "Observed",
    "random": "Random",
    "nucleus_gradient_strong": "Nucleus",
    "membrane_gradient_strong": "Membrane",
}

avg_pilr = {}
for rule in RULES.keys():
    avg_pilr[rule] = np.load(
        base_folder / rule / f"{rule}_average_PILR.npy",
    )

# %% [markdown]
# ## Get domain and parameterized coordinates for the average shape meshes
mesh_path = datadir / "average_shape_meshes"
mesh_dict = get_average_shape_mesh_objects(mesh_path)
domain = get_domain(mesh_dict)
domain = np.transpose(domain, (1, 0, 2))
domain_nuc = (255 * (domain > 1)).astype(np.uint8)
domain_mem = (255 * (domain > 0)).astype(np.uint8)
coords_param = get_parametrized_coords_for_avg_shape(domain)

# %% [markdown]
# ## Get reconstructions
morphed = {}
for key in avg_pilr.keys():
    morphed[key] = morph_PILRs_into_average_shape(
        pilr_list=[avg_pilr[key]],
        domain=domain,
        coords_param=coords_param,
        mesh_dict=mesh_dict,
    )
morphed = {k: np.array(v).squeeze() for k, v in morphed.items()}

# %% [markdown]
# ## Create folder to save morphed PILRs
morphed_folder = base_folder / "morphed_PILRs"
morphed_folder.mkdir(exist_ok=True)

# %% [markdown]
# ## Plot morphed PILRs
for mode, morph in morphed.items():
    morph = (255 * (morph / np.max(morph))).astype(np.uint8)
    morph[domain > 0] += 3
    morph[domain == 0] = 0
    for dim in range(2):
        fig, ax = plot_and_save_center_slice(
            morph,
            structure=f"{STRUCTURE_NAME}_{MODE_LABELS[mode]}",
            dim=dim,
            showfig=False,
            output_dir=morphed_folder,
            add_contour=True,
            domain_nuc=domain_nuc,
            domain_mem=domain_mem,
            vmin=0,
            pct_max=97.5,
            lw=2,
        )

        plt.tight_layout()
        plt.show()
# %% [markdown]
# ## Plot radial profile
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
STRUCTURE_ID = "SLC25A17"
for ch, pilr in avg_pilr.items():
    pilr = pilr.squeeze()
    pilrsum = np.sum(pilr, axis=1).astype(np.float64)
    pilrsum = pilrsum / np.sum(pilrsum)
    linewidth = 4 if ch == STRUCTURE_ID else 2
    alpha = 1 if ch == STRUCTURE_ID else 0.6
    ax.plot(
        np.linspace(0, 1, len(pilrsum)),
        pilrsum,
        label=MODE_LABELS[ch],
        linewidth=linewidth,
        alpha=alpha,
    )

ax.axvline(0.5, ls="--", color="black", label="Nuclear boundary")
ax.set_xlabel("Normalized distance from the nuclear centroid")
ax.set_ylabel("Normalized radial PILR")
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim([0, 1])
# ax.set_ylim([0, 0.06])
# put legend on the right outside the axis
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
# ax.legend()
fig.savefig(f"{base_folder}/avg_PILR_squeezed.png", dpi=300, bbox_inches="tight")

# %%
