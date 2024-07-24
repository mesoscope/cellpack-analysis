# %%
from pathlib import Path

from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

# %%
base_datadir = Path(__file__).parents[3] / "data"
# %%
STRUCTURE_ID = "SLC25A17"

# %%
mesh_information_dict = get_mesh_information_dict(
    structure_id=STRUCTURE_ID,
    base_datadir=base_datadir,
    recalculate=True,
)

# %%
