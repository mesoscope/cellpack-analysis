# %%
from pathlib import Path

from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict_for_structure

# %%
base_datadir = Path(__file__).parents[2] / "data"
# %%
STRUCTURE_ID = "SEC61B"

# %%
mesh_information_dict = get_mesh_information_dict_for_structure(
    structure_id=STRUCTURE_ID,
    base_datadir=base_datadir,
    recalculate=True,
)

# %%
