#!/usr/bin/env python3
"""
Script to download raw or segmented images for a specific cellular structure.

This script downloads images from the hiPSC single cell image dataset:
https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset

It can download either raw (unsegmented) or segmented images for various cellular structures.

Usage:
    python get_structure_images.py --structure-id SLC25A17 --download-raw

Example:
    python get_structure_images.py --structure-id RAB5A --sample-dir sample_8d --max-cells 5 --redownload

Available structures:
    - SLC25A17 (peroxisomes)
    - RAB5A (early endosomes)
    - LAMP1 (lysosomes)
    - SEC61B (ER)
    - ATP2A2 (smooth ER)
    - TOMM20 (mitochondria)
    - ST6GAL1 (Golgi)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import quilt3
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.get_cell_id_list import get_cell_id_list_for_structure
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe

logger = logging.getLogger(__name__)


def download_image(row: pd.Series, col_name: str, save_path: Path, pkg: quilt3.Package) -> Path:
    """
    Download a single image file from the dataset.

    Parameters
    ----------
    row
        DataFrame row containing image metadata
    col_name
        Column name indicating which image type to download ('crop_raw' or 'crop_seg')
    save_path
        Directory path where the image will be saved
    pkg
        Quilt package containing the dataset

    Returns
    -------
    :
        Path to the downloaded image file
    """
    subdir_name = row[col_name].split("/")[0]
    file_name = row[col_name].split("/")[1]
    local_filename = (
        save_path
        / f"{row.structure_name}_{row.CellId}_ch_{row.ChannelNumberStruct}_{col_name}_original.tiff"
    )

    if not local_filename.exists():
        logger.debug(f"Downloading {local_filename.name}")
        pkg[subdir_name][file_name].fetch(local_filename)
    else:
        logger.debug(f"{local_filename.name} already exists. Skipping download.")

    return local_filename


def download_structure_images(
    structure_id: str,
    sample_dir: str = "sample_8d",
    download_raw: bool = True,
    max_cells: int | None = None,
    redownload: bool = False,
    output_dir: str | None = None,
) -> int:
    """
    Download images for a specific cellular structure.

    Parameters
    ----------
    structure_id
        Structure identifier (e.g., 'SLC25A17', 'RAB5A')
    sample_dir
        Sample directory name ('sample_8d' for dsphere samples or 'full' for complete dataset)
    download_raw
        If True, download raw/unsegmented images. If False, download segmented images
    max_cells
        Maximum number of cells to download (None for all available)
    redownload
        If True, re-download variance dataset metadata
    output_dir
        Custom output directory name (default: auto-generated based on image type)

    Returns
    -------
    :
        Exit code (0 for success, 1 for failure)
    """
    # Set up data directory
    datadir = get_datadir_path()

    # Load dataset package from quilt
    logger.info("Loading dataset package from quilt...")
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")

    # Load dataframe
    logger.info("Loading metadata...")
    meta_df = get_variance_dataframe(redownload, pkg)
    meta_df.index = meta_df.index.astype(str)

    # Get cell_id list for structure
    dsphere = sample_dir == "sample_8d"
    cell_id_list = get_cell_id_list_for_structure(structure_id, dsphere=dsphere)
    logger.info(f"Found {len(cell_id_list)} cell IDs for {structure_id}")
    if len(cell_id_list) == 0:
        logger.error(f"No cell IDs found for structure {structure_id}. Exiting.")
        return 1

    # Limit number of cells if specified
    if max_cells is not None:
        cell_id_list = cell_id_list[:max_cells]
        logger.info(f"Limited to {len(cell_id_list)} cells")

    # Create dataframe for structure metadata
    meta_df_struct = meta_df.loc[cell_id_list].reset_index()

    # Prepare file paths to save images
    subfolder_name = "sample_8d" if dsphere else "full"
    if output_dir:
        folder_name = output_dir
    else:
        folder_name = "unsegmented" if download_raw else "segmented"

    save_path = datadir / f"structure_data/{structure_id}/{subfolder_name}/{folder_name}"
    save_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Images will be saved to {save_path}")

    # Start download
    col_name = "crop_raw" if download_raw else "crop_seg"
    logger.info(f"Starting download of {len(meta_df_struct)} images...")

    # Download images with progress bar
    for _, row in tqdm(
        meta_df_struct.iterrows(), total=len(meta_df_struct), desc="Downloading images"
    ):
        download_image(row, col_name, save_path, pkg)

    logger.info("Download complete!")
    return 0


def main():
    """Main function to parse arguments and download images."""
    parser = argparse.ArgumentParser(
        description="Download raw or segmented images for cellular structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--structure-id",
        required=True,
        help="Structure ID to download (e.g., SLC25A17, RAB5A, LAMP1, SEC61B, TOMM20, ST6GAL1)",
    )

    # Optional arguments
    parser.add_argument(
        "--sample-dir",
        default="sample_8d",
        choices=["sample_8d", "full"],
        help="Sample directory: 'sample_8d' for dsphere samples or 'full' for complete dataset (default: sample_8d)",
    )

    parser.add_argument(
        "--download-raw",
        action="store_true",
        help="Download raw/unsegmented images (default: download segmented images)",
    )

    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Maximum number of cells to download (default: download all available)",
    )

    parser.add_argument(
        "--redownload", action="store_true", help="Re-download variance dataset metadata"
    )

    parser.add_argument(
        "--output-dir",
        help="Custom output directory name within structure data folder (default: auto-generated)",
    )

    args = parser.parse_args()

    # Validate structure ID
    valid_structures = ["SLC25A17", "RAB5A", "SEC61B", "ST6GAL1"]
    if args.structure_id not in valid_structures:
        logger.warning(f"Structure ID '{args.structure_id}' not in known list: {valid_structures}")
        logger.warning("Proceeding anyway - make sure this is a valid structure ID")

    logger.info(f"Structure ID: {args.structure_id}")
    logger.info(f"Sample directory: {args.sample_dir}")
    logger.info(f"Download raw images: {args.download_raw}")
    logger.info(f"Max cells: {args.max_cells}")
    logger.info(f"Redownload metadata: {args.redownload}")
    logger.info(f"Output directory: {args.output_dir}")

    # Download images
    return download_structure_images(
        structure_id=args.structure_id,
        sample_dir=args.sample_dir,
        download_raw=args.download_raw,
        max_cells=args.max_cells,
        redownload=args.redownload,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    exit(main())
