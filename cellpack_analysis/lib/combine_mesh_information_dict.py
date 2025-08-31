# %% [markdown]
# # Create a combined mesh_information_dict for all structures
# %%
import pickle
from pathlib import Path

from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

# %%
base_datadir = Path(__file__).parents[2] / "data"
save_path = base_datadir / "mesh_information_dicts" / "combined_mesh_information_dict.dat"
save_path.parent.mkdir(exist_ok=True, parents=True)
# %%
STRUCTURE_IDS = [
    "SLC25A17",  # Peroxisomes
    "RAB5A",  # Early endosomes
    "SEC61B",  # ER
    "ST6GAL1",  # Golgi
]

# %%
combined_mesh_information_dict = {}
for STRUCTURE_ID in STRUCTURE_IDS:
    mesh_information_dict = get_mesh_information_dict_for_structure(
        structure_id=STRUCTURE_ID,
        base_datadir=base_datadir,
        recalculate=True,
    )
    combined_mesh_information_dict[STRUCTURE_ID] = mesh_information_dict
    with open(save_path, "wb") as f:
        pickle.dump(combined_mesh_information_dict, f)
