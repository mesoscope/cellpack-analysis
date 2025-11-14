# %% [markdown]
# # Workflow to analyze multi-structure colocalization
#
# Compare colocalization of endosomes and peroxisomes with the ER and Golgi
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cellpack_analysis.lib import distance
from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.label_tables import COLOR_PALETTE
from cellpack_analysis.lib.load_data import get_position_data_from_outputs
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure
from cellpack_analysis.lib.stats import normalize_distances

logger = logging.getLogger(__name__)
# %% [markdown]
# ## Set up parameters
save_format = "pdf"
distance_measures = [
    "nucleus",
    "z",
]

distance_dict = {
    "ER_peroxisome": {
        "channel_map": {
            "SLC25A17": "SLC25A17",
            "random": "SEC61B",
            "nucleus_gradient_strong": "SEC61B",
            "membrane_gradient_strong": "SEC61B",
            "apical_gradient_weak": "SEC61B",
            "struct_gradient": "SEC61B",
        },
        "structure_id": "SLC25A17",
        "structure_name": "peroxisome",
    },
    "golgi_peroxisome": {
        "channel_map": {
            "SLC25A17": "SLC25A17",
            "random": "ST6GAL1",
            "nucleus_gradient_strong": "ST6GAL1",
            "membrane_gradient_strong": "ST6GAL1",
            "apical_gradient_weak": "ST6GAL1",
            "struct_gradient_weak": "ST6GAL1",
        },
        "structure_id": "SLC25A17",
        "structure_name": "peroxisome",
    },
    "ER_endosome": {
        "channel_map": {
            "RAB5A": "RAB5A",
            "random": "SEC61B",
            "nucleus_gradient_strong": "SEC61B",
            "membrane_gradient_strong": "SEC61B",
            "apical_gradient_weak": "SEC61B",
            "struct_gradient": "SEC61B",
        },
        "structure_id": "RAB5A",
        "structure_name": "endosome",
    },
    "golgi_endosome": {
        "channel_map": {
            "RAB5A": "RAB5A",
            "random": "ST6GAL1",
            "nucleus_gradient_strong": "ST6GAL1",
            "membrane_gradient_strong": "ST6GAL1",
            "apical_gradient_weak": "ST6GAL1",
            "struct_gradient_weak": "ST6GAL1",
        },
        "structure_id": "RAB5A",
        "structure_name": "endosome",
    },
}
all_structures = list(
    set([v["channel_map"][k] for v in distance_dict.values() for k in v["channel_map"]])
)

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
# ## Read emd data
df_list = []
for packing_id, mode_dict in distance_dict.items():

    mode_results_dir = base_results_dir / f"distance_analysis/{packing_id}/"
    mode_results_dir.mkdir(exist_ok=True, parents=True)

    channel_map = mode_dict["channel_map"]
    structure_id = mode_dict["structure_id"]
    structure_name = mode_dict["structure_name"]
    packing_modes = list(channel_map.keys())

    all_positions = get_position_data_from_outputs(
        structure_id=structure_id,
        packing_id=packing_id,
        packing_modes=packing_modes,
        base_datadir=base_datadir,
        results_dir=mode_results_dir,
        packing_output_folder=packing_output_folder,
        ingredient_key=f"membrane_interior_{structure_name}",
        recalculate=False,
    )

    all_distance_dict = distance.get_distance_dictionary(
        all_positions=all_positions,
        distance_measures=distance_measures,
        mesh_information_dict=combined_mesh_dict,
        channel_map=channel_map,
        results_dir=mode_results_dir,
        recalculate=False,
    )

    all_distance_dict = distance.filter_invalids_from_distance_distribution_dict(
        distance_distribution_dict=all_distance_dict, minimum_distance=None
    )

    all_distance_dict = normalize_distances(
        all_distance_dict=all_distance_dict,
        mesh_information_dict=combined_mesh_dict,
        channel_map=channel_map,
        normalization=normalization,
    )

    df_emd = distance.get_distance_distribution_emd_df(
        all_distance_dict=all_distance_dict,
        packing_modes=packing_modes,
        distance_measures=distance_measures,
        results_dir=mode_results_dir,
        recalculate=False,
        suffix=suffix,
    )
    df_emd["packing_id"] = packing_id
    df_plot = (
        df_emd.query("packing_mode_1 == @structure_id" " and packing_mode_2 != @structure_id")
        .copy()
        .reset_index(drop=True)
    )
    df_list.append(df_plot)
df_emd_all = pd.concat(df_list, axis=0).reset_index(drop=True)
# %%
distance_measure_to_plot = "nucleus"
df_emd_plot = df_emd_all.query("distance_measure == @distance_measure_to_plot").copy()
# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(
    data=df_emd_plot,
    x="emd",
    y="packing_mode_2",
    hue="packing_mode_1",
    palette=COLOR_PALETTE,
    ax=ax,
    orient="h",
    cut=0,
    split=True,
)

# %%
