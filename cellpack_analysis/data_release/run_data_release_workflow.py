"""
Workflow to export simularium paths as CSV for BFF data release.

This workflow supports modular execution of the following steps:
1. Upload meshes to S3
2. Update simularium files locally (update mesh URLs and colors)
3. Upload updated simularium files to S3
4. Generate and upload thumbnails
5. Create CSV files with S3 paths and metadata
6. Upload CSV files to S3 (optional, controlled by config)

Usage:
    python run_data_release_workflow.py --config_file path/to/config.json

    # Or run specific steps:
    python run_data_release_workflow.py --config_file path/to/config.json --upload_meshes
    python run_data_release_workflow.py --config_file path/to/config.json --update_simularium
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cellpack_analysis.data_release.csv_metadata_config import get_metadata_dict
from cellpack_analysis.data_release.data_release_config import DataReleaseConfig
from cellpack_analysis.lib.get_structure_stats_dataframe import get_structure_stats_dataframe
from cellpack_analysis.lib.io import format_time
from cellpack_analysis.lib.s3_utils import get_s3_client, upload_meshes_for_structure, upload_to_s3
from cellpack_analysis.lib.simularium_utils import (
    generate_and_upload_thumbnail,
    update_simularium_colors,
    update_simularium_mesh_urls,
)

logger = logging.getLogger(__name__)


def upload_meshes(config: DataReleaseConfig) -> None:
    """
    Upload meshes to S3 for all structures in the configuration.

    Parameters
    ----------
    config
        Data release configuration
    """
    logger.info("Step 1: Uploading meshes to S3")

    for structure_config in config.structures:
        structure_id = structure_config["structure_id"]
        logger.info(f"Processing meshes for structure: {structure_id}")

        count = upload_meshes_for_structure(
            structure_id=structure_id,
            base_datadir=config.base_datadir,
            bucket=config.s3_bucket,
            use_inverted_meshes=config.use_inverted_meshes,
            reinvert=config.reinvert_meshes,
            reupload=config.upload_meshes_to_s3,
            max_workers=config.max_workers,
        )
        logger.info(f"Processed {count} meshes for {structure_id}")


def collect_simularium_files(config: DataReleaseConfig) -> list[dict]:
    """
    Collect all simularium files to process.

    Parameters
    ----------
    config
        Data release configuration

    Returns
    -------
    list[dict]
        List of file information dictionaries
    """
    files_to_process = []

    for structure_config in config.structures:
        structure_id = structure_config["structure_id"]
        structure_name = structure_config["structure_name"]
        packing_id = structure_config["packing_id"]
        structure_color = structure_config["color"]

        # Create channel colors with the correct structure color
        channel_colors = config.get_channel_colors(structure_color)

        # Create structure-specific mesh URL
        structure_mesh_url = config.get_structure_mesh_url(structure_id)

        for rule in config.rules:
            simularium_path = (
                config.base_datadir
                / "packing_outputs"
                / config.dataset
                / config.experiment
                / rule
                / packing_id
                / "spheresSST"
            )
            figure_path = simularium_path / "figures"

            for file_path in simularium_path.rglob("*.simularium"):
                files_to_process.append(
                    {
                        "file_path": file_path,
                        "structure_id": structure_id,
                        "structure_name": structure_name,
                        "packing_id": packing_id,
                        "rule": rule,
                        "figure_path": figure_path,
                        "channel_colors": channel_colors.copy(),
                        "mesh_url": structure_mesh_url,
                    }
                )

    logger.info(f"Found {len(files_to_process)} simularium files to process")
    return files_to_process


def update_simularium_files_locally(
    files_to_process: list[dict],
) -> None:
    """
    Update simularium files locally (mesh URLs and colors).

    Parameters
    ----------
    files_to_process
        List of file information dictionaries
    """
    logger.info("Step 2: Updating simularium files locally")

    for file_info in tqdm(files_to_process, desc="Updating simularium files"):
        file_path = file_info["file_path"]
        cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

        # Update mesh URLs
        update_simularium_mesh_urls(file_path, file_info["mesh_url"], cell_id)

        # Update colors
        update_simularium_colors(
            file_path, file_info["channel_colors"], file_info["structure_name"]
        )

    logger.info(f"Updated {len(files_to_process)} simularium files")


def upload_simularium_files_to_s3(config: DataReleaseConfig, files_to_process: list[dict]) -> None:
    """
    Upload simularium files to S3.

    Parameters
    ----------
    config
        Data release configuration
    files_to_process
        List of file information dictionaries
    """
    logger.info("Step 3: Uploading simularium files to S3")

    s3_client = get_s3_client()
    uploaded_count = 0

    for file_info in tqdm(files_to_process, desc="Uploading simularium files"):
        file_path = file_info["file_path"]
        s3_key = file_path.relative_to(config.base_datadir).as_posix()

        if config.reupload_simularium_files or not _file_exists_on_s3(
            s3_client, config.s3_bucket, s3_key
        ):
            upload_to_s3(s3_client, file_path, config.s3_bucket, s3_key)
            uploaded_count += 1

    logger.info(f"Uploaded {uploaded_count} simularium files to S3")


def generate_and_upload_all_thumbnails(
    config: DataReleaseConfig, files_to_process: list[dict]
) -> None:
    """
    Generate and upload thumbnails for all simularium files.

    Parameters
    ----------
    config
        Data release configuration
    files_to_process
        List of file information dictionaries
    """
    logger.info("Step 4: Generating and uploading thumbnails")

    s3_client = get_s3_client()
    generated_count = 0

    for file_info in tqdm(files_to_process, desc="Generating thumbnails"):
        file_path = file_info["file_path"]
        cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

        thumbnail_url = generate_and_upload_thumbnail(
            s3_client=s3_client,
            figure_path=file_info["figure_path"],
            file_stem=file_path.stem,
            packing_id=file_info["packing_id"],
            condition=config.condition,
            rule=file_info["rule"],
            cell_id=cell_id,
            thumbnail_dir=config.thumbnail_dir,
            base_datadir=config.base_datadir,
            file_path=file_path,
            channel_colors=file_info["channel_colors"],
            bucket=config.s3_bucket,
            base_s3_url=config.base_s3_url,
            reupload=config.reupload_thumbnails,
        )

        if thumbnail_url:
            generated_count += 1

    logger.info(f"Generated {generated_count} thumbnails")


def create_csv_files(
    config: DataReleaseConfig,
    files_to_process: list[dict],
    structure_stats_df: pd.DataFrame,
) -> None:
    """
    Create CSV files with S3 paths and metadata.

    Parameters
    ----------
    config
        Data release configuration
    files_to_process
        List of file information dictionaries
    structure_stats_df
        DataFrame containing structure statistics
    """
    logger.info("Step 5: Creating CSV files")

    records = []

    for file_info in tqdm(files_to_process, desc="Building CSV records"):
        file_path = file_info["file_path"]
        cell_id = file_path.stem.split("_seed")[0].split("_")[-1]

        # Build S3 paths
        simularium_s3_key = file_path.relative_to(config.base_datadir).as_posix()
        simularium_s3_path = f"{config.base_s3_url}{simularium_s3_key}"

        # Get thumbnail path
        thumbnail_filename = f"{file_path.stem}_thumbnail.png"
        thumbnail_s3_key = (
            file_path.relative_to(config.base_datadir).parent.as_posix()
            + f"/thumbnails/{thumbnail_filename}"
        )
        thumbnail_s3_path = f"{config.base_s3_url}{thumbnail_s3_key}"

        # Get structure stats
        stats = _get_cell_stats(cell_id, file_info["structure_id"], structure_stats_df)

        records.append(
            {
                "File Name": f"{file_info['packing_id']}_{file_info['rule']}_{cell_id}",
                "Cell ID": cell_id,
                "Rule": file_info["rule"],
                "Packing ID": file_info["packing_id"],
                "Structure ID": file_info["structure_id"],
                "Structure Name": file_info["structure_name"],
                "Count": stats["count"],
                "Dataset": config.dataset,
                "Condition": config.condition,
                "Experiment": config.experiment,
                "Cell Volume": stats["cell_volume"],
                "Nucleus Volume": stats["nuc_volume"],
                "Cell Height": stats["cell_height"],
                "Nucleus Height": stats["nuc_height"],
                "Cell Sphericity": stats["sphericity"],
                "File Type": "simularium",
                "File Path": simularium_s3_path,
                "Thumbnail": thumbnail_s3_path,
            }
        )

    # Save main CSV
    df = pd.DataFrame.from_records(records)
    csv_path = config.csv_output_dir / f"{config.output_name}_paths.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved paths CSV: {csv_path}")

    # Create metadata CSV if enabled
    metadata_csv_path = None
    if config.create_metadata_csv:
        metadata_csv_path = _create_metadata_csv(config)

    # Upload CSVs to S3 if enabled
    if config.upload_csv_to_s3:
        _upload_csv_files_to_s3(config, csv_path, metadata_csv_path)


def _get_cell_stats(
    cell_id: str,
    structure_id: str,
    structure_stats_df: pd.DataFrame,
) -> dict:
    """
    Extract cell statistics for a given cell and structure from the stats dataframe.

    Parameters
    ----------
    cell_id
        Cell identifier
    structure_id
        Structure identifier (e.g., "SLC25A17")
    structure_stats_df
        DataFrame containing structure statistics

    Returns
    -------
    dict
        Dictionary with keys: count, cell_volume, nuc_volume, cell_height, nuc_height, sphericity
    """
    cellid_structure_stats = structure_stats_df[
        (structure_stats_df["CellId"] == cell_id)
        & (structure_stats_df["structure_name"] == structure_id)
    ]

    def _get(col):
        return cellid_structure_stats[col].values[0] if not cellid_structure_stats.empty else None

    return {
        "count": _get("count"),
        "cell_volume": _get("cell_volume"),
        "nuc_volume": _get("nuc_volume"),
        "cell_height": _get("cell_height"),
        "nuc_height": _get("nuc_height"),
        "sphericity": _get("sphericity"),
    }


def update_csv_cell_stats(config: DataReleaseConfig, structure_stats_df: pd.DataFrame) -> None:
    """
    Update cell statistics columns in an existing CSV without modifying simularium files.

    This allows recalculating cell metrics (count, volumes, heights, sphericity) independently
    of the simularium file update workflow.

    Parameters
    ----------
    config
        Data release configuration
    structure_stats_df
        DataFrame containing updated structure statistics
    """
    csv_path = config.csv_output_dir / f"{config.output_name}_paths.csv"
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}. Run 'create_csv' first.")
        return

    logger.info(f"Updating cell stats in: {csv_path}")
    df = pd.read_csv(csv_path)

    col_map = {
        "Count": "count",
        "Cell Volume": "cell_volume",
        "Nucleus Volume": "nuc_volume",
        "Cell Height": "cell_height",
        "Nucleus Height": "nuc_height",
        "Cell Sphericity": "sphericity",
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating cell stats"):
        stats = _get_cell_stats(str(row["Cell ID"]), str(row["Structure ID"]), structure_stats_df)
        for col, key in col_map.items():
            df.at[idx, col] = stats[key]

    df.to_csv(csv_path, index=False)
    logger.info(f"Updated cell stats for {len(df)} rows in {csv_path}")

    if config.upload_csv_to_s3:
        metadata_csv_path = None
        if config.create_metadata_csv:
            metadata_csv_path = _create_metadata_csv(config)
        _upload_csv_files_to_s3(config, csv_path, metadata_csv_path)


def _file_exists_on_s3(s3_client, bucket: str, s3_key: str) -> bool:
    """Check if a file exists on S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except s3_client.exceptions.ClientError:
        return False


def _create_metadata_csv(config: DataReleaseConfig) -> Path:
    """Create metadata CSV describing the output columns.

    Returns
    -------
    Path
        Path to the created metadata CSV file
    """
    metadata_dict = get_metadata_dict()
    metadata_df = pd.DataFrame(list(metadata_dict.items()), columns=["Column Name", "Description"])
    metadata_csv_path = config.csv_output_dir / f"{config.output_name}_metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)
    logger.info(f"Created metadata CSV: {metadata_csv_path}")
    return metadata_csv_path


def _upload_csv_files_to_s3(
    config: DataReleaseConfig, csv_path: Path, metadata_csv_path: Path | None
) -> None:
    """Upload CSV files to S3 bucket root with public ACL.

    Parameters
    ----------
    config
        Data release configuration
    csv_path
        Path to the main CSV file
    metadata_csv_path
        Path to the metadata CSV file (optional, can be None)
    """
    logger.info("Uploading CSV files to S3")

    s3_client = get_s3_client()

    # Upload main CSV to bucket root
    csv_s3_key = csv_path.name
    if config.reupload_csv_files or not _file_exists_on_s3(s3_client, config.s3_bucket, csv_s3_key):
        upload_to_s3(s3_client, csv_path, config.s3_bucket, csv_s3_key)
        csv_s3_url = f"{config.base_s3_url}{csv_s3_key}"
        logger.info(f"Uploaded main CSV to: {csv_s3_url}")
    else:
        logger.info(f"Main CSV already exists on S3, skipping upload: {csv_s3_key}")

    # Upload metadata CSV to bucket root if enabled and exists
    if config.upload_metadata_csv and metadata_csv_path is not None:
        metadata_s3_key = metadata_csv_path.name
        if config.reupload_csv_files or not _file_exists_on_s3(
            s3_client, config.s3_bucket, metadata_s3_key
        ):
            upload_to_s3(s3_client, metadata_csv_path, config.s3_bucket, metadata_s3_key)
            metadata_s3_url = f"{config.base_s3_url}{metadata_s3_key}"
            logger.info(f"Uploaded metadata CSV to: {metadata_s3_url}")
        else:
            logger.info(f"Metadata CSV already exists on S3, skipping upload: {metadata_s3_key}")
    elif not config.upload_metadata_csv:
        logger.info("Metadata CSV upload disabled in config")
    else:
        logger.info("Metadata CSV not created, skipping upload")


def run_data_release_workflow(config_file: Path, steps_to_run: list[str] | None = None) -> None:
    """
    Run the data release workflow.

    Parameters
    ----------
    config_file
        Path to the configuration file
    steps_to_run
        List of specific steps to run. If None, runs all enabled steps from config.
        Valid steps: 'upload_meshes', 'update_simularium', 'upload_simularium',
                     'upload_thumbnails', 'create_csv'
    """
    # Load configuration
    config = DataReleaseConfig(config_file)
    logger.info(f"Loaded configuration: {config}")

    # Determine which steps to run
    if steps_to_run is None:
        # Run based on config settings
        run_upload_meshes = config.upload_meshes_to_s3
        run_update_simularium = config.update_simularium_files
        run_upload_simularium = config.upload_simularium_to_s3
        run_upload_thumbnails = config.generate_thumbnails
        run_create_csv = config.create_csv
        run_update_csv_stats = config.update_csv_stats
    else:
        # Run specific steps requested
        run_upload_meshes = "upload_meshes" in steps_to_run
        run_update_simularium = "update_simularium" in steps_to_run
        run_upload_simularium = "upload_simularium" in steps_to_run
        run_upload_thumbnails = "upload_thumbnails" in steps_to_run
        run_create_csv = "create_csv" in steps_to_run
        run_update_csv_stats = "update_csv_stats" in steps_to_run

    # Step 1: Upload meshes
    if run_upload_meshes:
        upload_meshes(config)

    # Collect simularium files for subsequent steps
    files_to_process = []
    if any([run_update_simularium, run_upload_simularium, run_upload_thumbnails, run_create_csv]):
        files_to_process = collect_simularium_files(config)

    # Step 2: Update simularium files locally
    if run_update_simularium and files_to_process:
        update_simularium_files_locally(files_to_process)

    # Step 3: Upload simularium files to S3
    if run_upload_simularium and files_to_process:
        upload_simularium_files_to_s3(config, files_to_process)

    # Step 4: Generate and upload thumbnails
    if run_upload_thumbnails and files_to_process:
        generate_and_upload_all_thumbnails(config, files_to_process)

    # Step 5: Create CSV files
    if run_create_csv and files_to_process:
        structure_stats_df = get_structure_stats_dataframe()
        logger.info(f"Loaded structure statistics: {len(structure_stats_df)} rows")
        create_csv_files(config, files_to_process, structure_stats_df)

    # Step 6: Update cell stats in existing CSV (independent of simularium files)
    if run_update_csv_stats:
        structure_stats_df = get_structure_stats_dataframe()
        logger.info(f"Loaded structure statistics: {len(structure_stats_df)} rows")
        update_csv_cell_stats(config, structure_stats_df)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(description="Export simularium paths as CSV for BFF")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        required=True,
        help="Path to the data release configuration file",
    )
    parser.add_argument(
        "--upload_meshes",
        action="store_true",
        help="Run only the upload meshes step",
    )
    parser.add_argument(
        "--update_simularium",
        action="store_true",
        help="Run only the update simularium files step",
    )
    parser.add_argument(
        "--upload_simularium",
        action="store_true",
        help="Run only the upload simularium files step",
    )
    parser.add_argument(
        "--upload_thumbnails",
        action="store_true",
        help="Run only the upload thumbnails step",
    )
    parser.add_argument(
        "--create_csv",
        action="store_true",
        help="Run only the create CSV step",
    )
    parser.add_argument(
        "--update_csv_stats",
        action="store_true",
        help="Update cell statistics columns in an existing CSV without modifying simularium files",
    )

    args = parser.parse_args()

    # Build list of steps to run from command-line args
    steps_to_run = None
    if any(
        [
            args.upload_meshes,
            args.update_simularium,
            args.upload_simularium,
            args.upload_thumbnails,
            args.create_csv,
            args.update_csv_stats,
        ]
    ):
        steps_to_run = []
        if args.upload_meshes:
            steps_to_run.append("upload_meshes")
        if args.update_simularium:
            steps_to_run.append("update_simularium")
        if args.upload_simularium:
            steps_to_run.append("upload_simularium")
        if args.upload_thumbnails:
            steps_to_run.append("upload_thumbnails")
        if args.create_csv:
            steps_to_run.append("create_csv")
        if args.update_csv_stats:
            steps_to_run.append("update_csv_stats")

    run_data_release_workflow(config_file=Path(args.config_file), steps_to_run=steps_to_run)

    logger.info(f"Total time: {format_time(time.time() - start)}")
