# %%
import concurrent.futures
import logging

import vtk
from bioio import BioImage
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.mesh_tools import get_mesh_from_image

logger = logging.getLogger(__name__)

# NOTE: Raw images should be downloaded prior to running this script using `get_structure_images.py`
# Run this script using: `python get_meshes_from_images.py`
# %%
STRUCTURE_ID = "SLC25A17"
RECALCULATE = False

datadir = get_project_root() / f"data/structure_data/{STRUCTURE_ID}"
image_path = datadir / "sample_8d/segmented/"

save_folder = datadir / "meshes/test"
save_folder.mkdir(exist_ok=True, parents=True)

nuc_channel = 0
mem_channel = 1
struct_channel = 3


# %%
def decimation_pro(data, ratio):
    sim = vtk.vtkDecimatePro()
    sim.SetTargetReduction(ratio)
    sim.SetInputData(data)
    sim.PreserveTopologyOn()
    sim.SplittingOff()
    sim.BoundaryVertexDeletionOff()
    sim.Update()
    return sim.GetOutput()


def get_meshes_for_file(
    file,
    nuc_channel,
    mem_channel,
    struct_channel,
    save_folder=save_folder,
    subsample=0.95,
    recalculate=RECALCULATE,
):
    cell_id = file.stem.split("_")[1]
    reader = BioImage(file)
    data = reader.get_image_data("CZYX", S=0, T=0)
    writer = vtk.vtkOBJWriter()
    for name, channel in zip(
        ["nuc", "mem", "struct"], [nuc_channel, mem_channel, struct_channel], strict=False
    ):
        save_path = save_folder / f"{name}_mesh_{cell_id}.obj"
        if save_path.exists() and not recalculate:
            logger.info(f"Mesh for {file.stem} already exists. Skipping.")
            return
        mesh = get_mesh_from_image(data[channel], translate_to_origin=False)
        if subsample:
            subsampled_mesh = decimation_pro(mesh[0], 0.95)
        else:
            subsampled_mesh = mesh[0]
        writer.SetFileName(f"{save_path}")
        writer.SetInputData(subsampled_mesh)
        writer.Write()


# %% run function in parallel using concurrent futures
files_to_use = list(image_path.glob("*.tiff"))
subsample = True
input_files = []
for file in files_to_use:
    if (
        (STRUCTURE_ID not in file.stem)
        or (".tiff" not in file.suffix)
        or (file.name.startswith("."))
    ):
        logger.info(f"Skipping {file.stem}")
        continue
    input_files.append(file)
input_files = input_files[:10]  # for testing
logger.info(f"Processing {len(input_files)} files")
for file in input_files:
    logger.info(f"File to process: {file.stem}")

num_cores = 16
if len(input_files):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for file in input_files:
            future = executor.submit(
                get_meshes_for_file,
                file,
                nuc_channel,
                mem_channel,
                struct_channel,
                save_folder,
                subsample,
                RECALCULATE,
            )
            futures.append(future)

        # Wait for all futures to complete
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
            unit="file",
        ):
            pass
else:
    logger.info("No files to process")

# %%
