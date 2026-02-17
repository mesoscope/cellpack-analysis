#!/usr/bin/env python3
"""
Script to extract meshes from multichannel image files.

This script processes TIFF images containing multichannel segmentation data
and extracts 3D meshes for nucleus, membrane, and structure channels.

Usage:
    python get_meshes_from_images_script.py --structure-id SLC25A17 --sample-dir sample_8d
    
Example:
    python get_meshes_from_images_script.py --structure-id SLC25A17 \
        --sample-dir sample_8d --num-cores 8 --recalculate
"""

import argparse
import concurrent.futures
import logging
from pathlib import Path

import vtk
from bioio import BioImage
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.mesh_tools import get_mesh_from_image

logger = logging.getLogger(__name__)


def decimation_pro(data: vtk.vtkPolyData, ratio: float) -> vtk.vtkPolyData:
    """
    Reduce polygon count of mesh using decimation algorithm.

    Parameters
    ----------
    data
        Input mesh data to decimate
    ratio
        Target reduction ratio (0.0 to 1.0), where 0.95 reduces by 95%

    Returns
    -------
    :
        Decimated mesh with reduced polygon count
    """
    sim = vtk.vtkDecimatePro()
    sim.SetTargetReduction(ratio)
    sim.SetInputData(data)
    sim.PreserveTopologyOn()
    sim.SplittingOff()
    sim.BoundaryVertexDeletionOff()
    sim.Update()
    return sim.GetOutput()


def get_meshes_for_file(
    file: Path,
    nuc_channel: int,
    mem_channel: int,
    struct_channel: int,
    save_folder: Path,
    subsample: float | bool = 0.95,
    recalculate: bool = False,
) -> None:
    """
    Extract meshes from multichannel image file and save as OBJ files.

    Parameters
    ----------
    file
        Path to input TIFF image file containing multichannel data
    nuc_channel
        Channel index for nucleus segmentation
    mem_channel
        Channel index for membrane segmentation
    struct_channel
        Channel index for structure segmentation
    save_folder
        Directory to save generated mesh files
    subsample
        If float, target reduction ratio for decimation. If False, no subsampling
    recalculate
        If True, regenerate meshes even if they already exist
    """
    cell_id = file.stem.split("_")[1]
    reader = BioImage(file)
    data = reader.get_image_data("CZYX", S=0, T=0)
    writer = vtk.vtkOBJWriter()

    for name, channel in zip(
        ["nuc", "mem", "struct"], [nuc_channel, mem_channel, struct_channel], strict=False
    ):
        save_path = save_folder / f"{name}_mesh_{cell_id}.obj"
        if save_path.exists() and not recalculate:
            logger.info(f"Mesh for {file.stem} {name} already exists. Skipping.")
            continue

        mesh = get_mesh_from_image(data[channel], translate_to_origin=False)
        if subsample:
            ratio = subsample if isinstance(subsample, float) else 0.95
            subsampled_mesh = decimation_pro(mesh[0], ratio)
        else:
            subsampled_mesh = mesh[0]

        writer.SetFileName(str(save_path))
        writer.SetInputData(subsampled_mesh)
        writer.Write()
        logger.debug(f"Saved mesh: {save_path}")


def main():
    """Parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Extract meshes from multichannel image files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--structure-id", required=True, help="Structure ID to process (e.g., SLC25A17)"
    )

    parser.add_argument(
        "--sample-dir", default="sample_8d", help="Sample directory name (e.g., sample_8d)"
    )

    # Optional arguments
    parser.add_argument(
        "--nuc-channel",
        type=int,
        default=0,
        help="Channel index for nucleus segmentation (default: 0)",
    )

    parser.add_argument(
        "--mem-channel",
        type=int,
        default=1,
        help="Channel index for membrane segmentation (default: 1)",
    )

    parser.add_argument(
        "--struct-channel",
        type=int,
        default=3,
        help="Channel index for structure segmentation (default: 3)",
    )

    parser.add_argument(
        "--num-cores",
        type=int,
        default=16,
        help="Number of CPU cores to use for parallel processing (default: 16)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing, default: process all)",
    )

    parser.add_argument(
        "--subsample-ratio",
        type=float,
        default=0.95,
        help="Decimation ratio for mesh subsampling (0.0-1.0, default: 0.95)",
    )

    parser.add_argument(
        "--no-subsample", action="store_true", help="Disable mesh subsampling/decimation"
    )

    parser.add_argument(
        "--recalculate", action="store_true", help="Recalculate meshes even if they already exist"
    )

    parser.add_argument(
        "--output-dir",
        help="Custom output directory name within the structure data folder (default: meshes)",
    )

    args = parser.parse_args()

    # Setup paths
    datadir = get_project_root() / f"data/structure_data/{args.structure_id}"
    image_path = datadir / args.sample_dir / "segmented"

    if not image_path.exists():
        logger.error(f"Image path does not exist: {image_path}")
        return 1

    # Setup output directory
    if args.output_dir:
        save_folder = datadir / args.output_dir
    else:
        save_folder = datadir / "meshes"
    save_folder.mkdir(exist_ok=True, parents=True)

    # Configure subsampling
    if args.no_subsample:
        subsample = False
    else:
        subsample = args.subsample_ratio

    # Find input files
    files_to_use = list(image_path.glob("*.tiff"))
    input_files = []

    for file in files_to_use:
        if (
            (args.structure_id not in file.stem)
            or (".tiff" not in file.suffix)
            or (file.name.startswith("."))
        ):
            logger.debug(f"Skipping {file.stem}")
            continue
        input_files.append(file)
        if args.max_files and len(input_files) >= args.max_files:
            logger.info(f"Limited to {args.max_files} files for testing")
            break

    if not input_files:
        logger.error("No files to process")
        return 1

    logger.info(f"Processing {len(input_files)} files")
    logger.info(f"Using {args.num_cores} CPU cores")
    logger.info(f"Output directory: {save_folder}")
    logger.info(f"Subsample: {subsample}")
    logger.info(f"Recalculate: {args.recalculate}")

    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        futures = []
        for file in input_files:
            future = executor.submit(
                get_meshes_for_file,
                file,
                args.nuc_channel,
                args.mem_channel,
                args.struct_channel,
                save_folder,
                subsample,
                args.recalculate,
            )
            futures.append(future)

        # Wait for all futures to complete
        completed = 0
        failed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
            unit="file",
        ):
            try:
                future.result()  # This will raise an exception if the task failed
                completed += 1
            except Exception as e:
                logger.error(f"Task failed: {e}")
                failed += 1

    logger.info(f"Processing complete. {completed} successful, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
