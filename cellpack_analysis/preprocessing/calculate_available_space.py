#!/usr/bin/env python3
"""
Calculate available space for a structure.

This script calculates the available space for intracellular structures by discretizing
the space into a grid and calculating the distance of each grid point to the
nearest nucleus and membrane.
The distance is calculated using the signed distance function from the trimesh library and the
distances are saved in a grid directory for each cell_id.
Distances are normalized by the cell diameter and saved in the grid directory.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_project_root
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.mesh_tools import calculate_grid_distances

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate available intracellular space.")

    parser.add_argument(
        "--structure-id", type=str, default=None, help="Structure ID to process (default: None)"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=2.0,
        help="Grid spacing for distance calculations (default: 2.0)",
    )
    parser.add_argument(
        "--use-struct-mesh",
        action="store_true",
        default=True,
        help="Use structure mesh in calculations (default: True)",
    )
    parser.add_argument(
        "--no-struct-mesh",
        action="store_false",
        dest="use_struct_mesh",
        help="Don't use structure mesh in calculations",
    )
    parser.add_argument(
        "--use-mean-shape",
        action="store_true",
        help="Use mean shape instead of individual cell shapes",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=8,
        help="Number of cores to use for parallel processing (default: 8)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20000, help="Chunk size for processing (default: 20000)"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        default=True,
        help="Recalculate existing results (default: True)",
    )
    parser.add_argument(
        "--no-recalculate", action="store_false", dest="recalculate", help="Skip existing results"
    )
    return parser.parse_args()


def main():
    """Main function to run the available space calculation."""
    # Parse command line arguments
    args = parse_arguments()

    # Extract parameters from arguments
    structure_id = args.structure_id
    spacing = args.spacing
    use_struct_mesh = args.use_struct_mesh
    use_mean_shape = args.use_mean_shape
    num_cores = args.num_cores
    chunk_size = args.chunk_size
    recalculate = args.recalculate

    # Validate required arguments
    if not structure_id:
        logger.error("Structure ID is required. Use --structure-id to specify.")
        return  # Set file paths
    base_datadir = get_project_root() / "data"
    logger.info(f"Data directory: {base_datadir}")

    # Select cell_ids to use
    if use_mean_shape:
        cell_ids_to_use = ["mean"]
    else:
        cell_ids_to_use = get_cell_id_list_for_structure(structure_id)

    mesh_folder = base_datadir / f"structure_data/{structure_id}/meshes/"
    logger.info(f"Using {len(cell_ids_to_use)} cell_ids")

    # Get meshes for cell_ids used and prepare mesh lists
    mesh_data = []
    for cell_id in cell_ids_to_use:
        nuc_mesh_path = mesh_folder / f"nuc_mesh_{cell_id}.obj"
        mem_mesh_path = mesh_folder / f"mem_mesh_{cell_id}.obj"
        struct_mesh_path = mesh_folder / f"struct_mesh_{cell_id}.obj" if use_struct_mesh else None

        if nuc_mesh_path.exists() and mem_mesh_path.exists():
            struct_mesh = (
                struct_mesh_path if (struct_mesh_path and struct_mesh_path.exists()) else None
            )
            mesh_data.append((cell_id, nuc_mesh_path, mem_mesh_path, struct_mesh))
        else:
            logger.warning(f"Missing mesh for cell_id {cell_id}, skipping")

    logger.info(f"Found {len(mesh_data)} valid meshes")

    # Set up grid results directory
    grid_dir = base_datadir / f"structure_data/{structure_id}/grid_distances/"
    grid_dir.mkdir(exist_ok=True, parents=True)

    # Distance calculation parameters
    calc_nuc_distances = True
    calc_mem_distances = True
    calc_z_distances = True
    calc_scaled_nuc_distances = True

    # Run the workflow with parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for cell_id, nuc_mesh_path, mem_mesh_path, struct_mesh_path in mesh_data:
            results.append(
                executor.submit(
                    calculate_grid_distances,
                    nuc_mesh_path,
                    mem_mesh_path,
                    cell_id,
                    spacing,
                    grid_dir,
                    recalculate,
                    calc_nuc_distances,
                    calc_mem_distances,
                    calc_z_distances,
                    calc_scaled_nuc_distances,
                    chunk_size,
                    struct_mesh_path,
                )
            )

        with tqdm(total=len(results), desc="cells") as pbar:
            for result in as_completed(results):
                if result.result():
                    pbar.update(1)

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
