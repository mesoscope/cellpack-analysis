# %%
import logging
import shutil
from pathlib import Path

from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure

log = logging.getLogger(__name__)

# %%
base_datadir = Path(__file__).parents[2] / "data"

subfolder = "8d_sphere_data"

condition_name = "rules_shape"

rule = "random"

structure_name = "peroxisome"

structure_id = "SLC25A17"

packing_output_folder = (
    base_datadir
    / "packing_outputs"
    / subfolder
    / condition_name
    / rule
    / structure_name
    / "spheresSST"
)

mesh_folder = base_datadir / "structure_data" / structure_id / "meshes"

log.info(f"packing_output_folder: {packing_output_folder}")
log.info(f"mesh_folder: {mesh_folder}")

# %% [markdown]
# ## Copy meshes to packing output folder
cell_id_list = get_cell_id_list_for_structure(structure_id=structure_id, dsphere=True)
num_cells = 3
cell_ids = cell_id_list[:num_cells]
print(cell_ids)

# %%
cell_ids = ["mean"]
mesh_folder = base_datadir / "average_shape_meshes"
packing_output_folder = (
    base_datadir
    / "packing_outputs"
    / "stochastic_variation_analysis"
    / "variable_count_and_size"
    / "peroxisome"
    / "spheresSST"
)

# %%
# prefixes = ["mem", "nuc", "struct"]
prefixes = ["mem", "nuc"]
for cell_id in cell_ids:
    for prefix in prefixes:
        mesh_file = mesh_folder / f"{prefix}_mesh_{cell_id}.obj"
        new_mesh_file = packing_output_folder / f"{prefix}_mesh_{cell_id}.obj"
        shutil.copy(mesh_file, new_mesh_file)
log.info(f"{num_cells} meshes copied to packing output folder")

# %% [markdown]
# ## Invert mesh faces for mem
invert_prefix = "mem"
for cell_id in cell_ids:
    mesh_file_path = packing_output_folder / f"{invert_prefix}_mesh_{cell_id}.obj"
    orig_mesh_file_path = packing_output_folder / f"{invert_prefix}_mesh_{cell_id}_orig.obj"
    # save copy with orig prefix
    shutil.copy(
        mesh_file_path,
        orig_mesh_file_path,
    )

    # read obj file
    with open(orig_mesh_file_path) as f:
        lines = f.readlines()

    with open(mesh_file_path, "w") as f:
        for line in lines:
            if line[0] == "f":
                faces = line.split(" ")
                faces[-1] = faces[-1].split("\n")[0]
                faces = [faces[0], faces[-1], faces[2], f"{faces[1]}\n"]
                line = " ".join(faces).lstrip()
            f.write(line)

# %%
