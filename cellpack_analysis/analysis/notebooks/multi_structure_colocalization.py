# %% [markdown]
# # Workflow to analyze multi-structure colocalization
#
# Compare colocalization of endosomes and peroxisomes with the ER and Golgi
import itertools
import logging
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from statannotations.Annotator import Annotator
from tqdm import tqdm

from cellpack_analysis.lib import distance
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import COLOR_PALETTE
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

logger = logging.getLogger(__name__)
# %% [markdown]
# ## Set up parameters
save_format = "pdf"
distance_measures = [
    "nucleus",
    "z",
]

distance_dict = {
    "peroxisome": {
        "structure_id": "SLC25A17",
        "packing_mode": "SLC25A17",
        "ingredient_key": "membrane_interior_peroxisome",
        "distances": {dm: {} for dm in distance_measures},
    },
    "endosome": {
        "structure_id": "RAB5A",
        "packing_mode": "RAB5A",
        "ingredient_key": "membrane_interior_endosome",
        "distances": {dm: {} for dm in distance_measures},
    },
    "ER_peroxisome": {
        "structure_id": "SEC61B",
        "packing_mode": "struct_gradient",
        "ingredient_key": "membrane_interior_peroxisome",
        "distances": {dm: {} for dm in distance_measures},
    },
    "golgi_peroxisome": {
        "structure_id": "ST6GAL1",
        "packing_mode": "struct_gradient_weak",
        "ingredient_key": "membrane_interior_peroxisome",
        "distances": {dm: {} for dm in distance_measures},
    },
    "ER_endosome": {
        "structure_id": "SEC61B",
        "packing_mode": "struct_gradient",
        "ingredient_key": "membrane_interior_endosome",
        "distances": {dm: {} for dm in distance_measures},
    },
    "golgi_endosome": {
        "structure_id": "ST6GAL1",
        "packing_mode": "struct_gradient_weak",
        "ingredient_key": "membrane_interior_endosome",
        "distances": {dm: {} for dm in distance_measures},
    },
}
all_structures = list({v["structure_id"] for v in distance_dict.values()})
row_modes = ["peroxisome", "endosome"]
col_modes = ["ER", "golgi"]

# %% [markdown]
# ### Set file paths and setup parameters
packing_output_folder = "packing_outputs/8d_sphere_data/rules_shape/"
project_root = get_project_root()
base_datadir = project_root / "data"
base_results_dir = project_root / "results"

results_dir = base_results_dir / "multi_structure_colocalization"
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures/"
figures_dir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Set normalziation
normalization = None
suffix = ""
# %% [markdown]
# ### Get combined mesh dictionary
combined_mesh_dict = {}
for structure_id in all_structures:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=structure_id,
        base_datadir=base_datadir,
        recalculate=False,
    )
    combined_mesh_dict[structure_id] = mesh_information_dict

# %% [markdown]
# ## Read position and distance data
for packing_id, mode_dict in distance_dict.items():

    packing_mode = mode_dict["packing_mode"]
    structure_id = mode_dict["structure_id"]

    logger.info(f"Processing {packing_id} - {packing_mode} - {structure_id}")

    mode_results_dir = results_dir / packing_id
    mode_results_dir.mkdir(exist_ok=True, parents=True)

    mode_positions = get_position_data_from_outputs(
        structure_id=structure_id,
        packing_id=packing_id,
        packing_modes=[packing_mode],
        base_datadir=base_datadir,
        results_dir=mode_results_dir,
        packing_output_folder=packing_output_folder,
        ingredient_key=mode_dict["ingredient_key"],
        recalculate=False,
    )

    all_distances = distance.get_distance_dictionary(
        all_positions=mode_positions,
        distance_measures=distance_measures,
        mesh_information_dict=combined_mesh_dict,
        channel_map={packing_mode: structure_id},
        results_dir=mode_results_dir,
        recalculate=False,
        num_workers=16,
    )

    for distance_measure in distance_measures:
        distance_dict[packing_id]["distances"][distance_measure] = all_distances[distance_measure][
            packing_mode
        ]

with open(results_dir / "distances.dat", "wb") as f:
    pickle.dump(distance_dict, f)
# %% [markdown]
records = []
for row_mode in row_modes:
    for col_mode in col_modes:
        mode1 = distance_dict[row_mode]
        mode2 = distance_dict[f"{col_mode}_{row_mode}"]
        for distance_measure in distance_measures:
            logger.info(f"Calculating {distance_measure} EMD for {row_mode} and {col_mode}")
            distance_dict_1 = mode1["distances"][distance_measure]
            distance_dict_2 = mode2["distances"][distance_measure]

            cell_ids_1 = list(distance_dict_1.keys())
            cell_ids_2 = list(distance_dict_2.keys())
            pairwise_combinations = list(itertools.product(cell_ids_1, cell_ids_2))
            for cell_id_1, cell_id_2 in tqdm(pairwise_combinations):
                distances_1 = distance_dict_1[cell_id_1]
                distances_2 = distance_dict_2[cell_id_2]
                emd_record = {
                    "mode_1": row_mode,
                    "mode_2": col_mode,
                    "cell_id_1": cell_id_1,
                    "cell_id_2": cell_id_2,
                    "distance_measure": distance_measure,
                    "emd": wasserstein_distance(distances_1, distances_2),
                }
                records.append(emd_record)

emd_df = pd.DataFrame(records)
emd_df.to_parquet(results_dir / "emd_dataframe.parquet", index=False)

# %% [markdown]
# ## Plot results
pairs = [
    (("ER", "peroxisome"), ("ER", "endosome")),
    (("golgi", "peroxisome"), ("golgi", "endosome")),
]
for distance_measure in distance_measures:
    emd_df_subset = emd_df[emd_df["distance_measure"] == distance_measure]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    plot_params = {
        "data": emd_df_subset,
        "x": "mode_2",
        "y": "emd",
        "hue": "mode_1",
        "errorbar": "sd",
        "palette": COLOR_PALETTE,
    }
    sns.barplot(
        ax=ax,
        **plot_params,
    )
    ax.legend()
    ax.set_xlabel("Structure")
    ax.set_ylabel(f"EMD for {distance_measure}")
    sns.despine()
    annotator = Annotator(ax=ax, pairs=pairs, plot="barplot", **plot_params)
    annotator.configure(
        test="Mann-Whitney",
        verbose=False,
        comparisons_correction="Bonferroni",
        # loc="outside",
    ).apply_and_annotate()
    fig.savefig(figures_dir / f"emd_barplot_{distance_measure}{suffix}.{save_format}")
# %%
