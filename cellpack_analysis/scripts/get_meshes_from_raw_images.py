from aicsimageio.aics_image import AICSImage
from aicsshparam import shtools

from pathlib import Path
import vtk
import concurrent.futures

# NOTE: Raw images should be downloaded prior to running this script
# Run this script using: `python get_meshes_from_raw_images.py`

structure_name = "RAB5A"
datadir = Path(
    f"/allen/aics/animated-cell/Saurabh/cellpack-analysis/data/structure_data/{structure_name}"
)
image_path = datadir / "sample_8d/raw_imgs_for_PILR/"
file_glob = image_path.glob("*.tiff")

save_folder = datadir / "meshes"
save_folder.mkdir(exist_ok=True, parents=True)

nuc_channel = 0
mem_channel = 1


def decimation_pro(data, ratio):
    sim = vtk.vtkDecimatePro()
    sim.SetTargetReduction(ratio)
    sim.SetInputData(data)
    sim.PreserveTopologyOn()
    sim.SplittingOff()
    sim.BoundaryVertexDeletionOff()
    sim.Update()
    return sim.GetOutput()


def get_mesh_for_file(file, nuc_channel, mem_channel, save_folder=save_folder):
    cellID = file.stem.split("_")[1]
    reader = AICSImage(file)
    data = reader.get_image_data("CZYX", S=0, T=0)
    writer = vtk.vtkOBJWriter()
    for name, channel in zip(["nuc", "mem"], [nuc_channel, mem_channel]):
        mesh = shtools.get_mesh_from_image(data[channel])
        subsampled_mesh = decimation_pro(mesh[0], 0.95)
        writer.SetFileName(save_folder / f"{name}_mesh_{cellID}.obj")
        writer.SetInputData(subsampled_mesh)
        writer.Write()


# run function in parallel using concurrent futures
files_to_use = list(file_glob)
input_files = []
for file in files_to_use:
    if (structure_name not in file.stem) or (".tiff" not in file.suffix):
        continue
    input_files.append(file)

num_cores = 64
with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = []
    for file in input_files:
        futures.append(
            executor.submit(get_mesh_for_file, file, nuc_channel, mem_channel)
        )
    # get number of completed futures
    done = 0
    for fs in concurrent.futures.as_completed(futures):
        done += 1
        print(f"Completed {done} meshes")
