# %%
import concurrent.futures
from pathlib import Path

import vtk
from aicsimageio.aics_image import AICSImage
from aicsshparam import shtools

# NOTE: Raw images should be downloaded prior to running this script
# Run this script using: `python get_meshes_from_raw_images.py`
# %%
STRUCTURE_NAME = "RAB5A"

datadir = Path(__file__).parents[2] / f"data/structure_data/{STRUCTURE_NAME}"
image_path = datadir / "full/segmented/"

save_folder = datadir / "meshes/"
save_folder.mkdir(exist_ok=True, parents=True)

nuc_channel = 0
mem_channel = 1


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


def get_mesh_for_file(
    file, nuc_channel, mem_channel, save_folder=save_folder, subsample=0.95
):
    cellID = file.stem.split("_")[1]
    reader = AICSImage(file)
    data = reader.get_image_data("CZYX", S=0, T=0)
    writer = vtk.vtkOBJWriter()
    for name, channel in zip(["nuc", "mem"], [nuc_channel, mem_channel]):
        save_path = save_folder / f"{name}_mesh_{cellID}.obj"
        if save_path.exists():
            print(f"Mesh for {file.stem} already exists. Skipping.")
            return
        mesh = shtools.get_mesh_from_image(data[channel])
        if subsample:
            subsampled_mesh = decimation_pro(mesh[0], 0.95)
        else:
            subsampled_mesh = mesh[0]
        writer.SetFileName(save_path)
        writer.SetInputData(subsampled_mesh)
        writer.Write()


# %% run function in parallel using concurrent futures
files_to_use = list(image_path.glob("*.tiff"))
subsample = True
input_files = []
for file in files_to_use:
    if (STRUCTURE_NAME not in file.stem) or (".tiff" not in file.suffix):
        print(f"Skipping {file.stem}")
        continue
    input_files.append(file)

print(f"Processing {len(input_files)} files")

num_cores = 64
if len(input_files):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        futures = executor.map(
            get_mesh_for_file,
            input_files,
            [nuc_channel] * len(input_files),
            [mem_channel] * len(input_files),
            [save_folder] * len(input_files),
            [subsample] * len(input_files),
        )
        # get number of completed futures
        done = 0
        for _ in concurrent.futures.as_completed(futures):  # type: ignore
            done += 1
            print(f"Completed {done} meshes")
else:
    print("No files to process")

# %%
