"""S3 utilities for uploading and managing files."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from tqdm import tqdm

from cellpack_analysis.lib.mesh_tools import invert_mesh_faces

logger = logging.getLogger(__name__)


def should_upload_to_s3(s3_client, bucket: str, s3_key: str, force_upload: bool = False) -> bool:
    """
    Check if a file should be uploaded to S3.

    Parameters
    ----------
    s3_client
        Boto3 S3 client
    bucket
        S3 bucket name
    s3_key
        S3 object key
    force_upload
        If True, always return True to force upload

    Returns
    -------
    bool
        True if file should be uploaded, False otherwise
    """
    if force_upload:
        return True
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return False
    except s3_client.exceptions.ClientError:
        return True


def upload_to_s3(s3_client, local_path: Path, bucket: str, s3_key: str) -> None:
    """
    Upload a file to S3 and set it as public-read.

    Parameters
    ----------
    s3_client
        Boto3 S3 client
    local_path
        Local file path to upload
    bucket
        S3 bucket name
    s3_key
        S3 object key
    """
    s3_client.upload_file(str(local_path), bucket, s3_key)
    s3_client.put_object_acl(ACL="public-read", Bucket=bucket, Key=s3_key)
    logger.debug(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")


def get_s3_client():
    """
    Get a boto3 S3 client.

    Returns
    -------
    boto3.client
        Boto3 S3 client
    """
    return boto3.client("s3")


def process_and_upload_mesh(
    file_path: Path,
    base_datadir: Path,
    inverted_dir: Path,
    s3_client,
    bucket: str,
    use_inverted_meshes: bool = False,
    reinvert: bool = False,
    reupload: bool = True,
    invert_prefix: str = "mem",
) -> str:
    """
    Process a single mesh file: invert if needed and upload to S3.

    Parameters
    ----------
    file_path
        Path to the mesh file to process
    base_datadir
        Base data directory path
    inverted_dir
        Directory to store inverted meshes
    s3_client
        Boto3 S3 client
    bucket
        S3 bucket name
    use_inverted_meshes
        Whether to invert meshes before uploading
    reinvert
        Whether to re-invert meshes if they already exist
    reupload
        Whether to actually upload to S3
    invert_prefix
        Prefix in filename to identify meshes that need to be inverted
        (e.g. "mem" for membrane meshes)

    Returns
    -------
    str
        Status message for this file
    """
    s3_key = file_path.relative_to(base_datadir).as_posix()
    path_to_upload = file_path

    if use_inverted_meshes and file_path.name.startswith(invert_prefix):
        # Create inverted mesh in the inverted subfolder
        inverted_mesh_path = inverted_dir / file_path.name
        if reinvert or not inverted_mesh_path.exists():
            invert_mesh_faces(input_mesh_path=file_path, output_mesh_path=inverted_mesh_path)
            logger.debug(f"Inverted mesh: {file_path.name}")
        path_to_upload = inverted_mesh_path

    if reupload:
        upload_to_s3(s3_client, path_to_upload, bucket, s3_key)
        return f"Uploaded {file_path.name}"
    else:
        return f"Processed {file_path.name}"


def upload_meshes_for_structure(
    structure_id: str,
    base_datadir: Path,
    bucket: str,
    use_inverted_meshes: bool = False,
    reinvert: bool = False,
    reupload: bool = True,
    max_workers: int = 8,
) -> int:
    """
    Upload all meshes for a given structure to S3.

    Parameters
    ----------
    structure_id
        Structure identifier (e.g., "SLC25A17")
    base_datadir
        Base data directory path
    bucket
        S3 bucket name
    use_inverted_meshes
        Whether to invert meshes before uploading
    reinvert
        Whether to re-invert meshes if they already exist
    reupload
        Whether to actually upload to S3
    max_workers
        Number of parallel workers

    Returns
    -------
    int
        Number of meshes processed
    """
    s3_client = get_s3_client()
    mesh_dir = base_datadir / "structure_data" / structure_id / "meshes"
    inverted_dir = mesh_dir / "inverted"

    # Create inverted subfolder if inverting meshes
    if use_inverted_meshes:
        inverted_dir.mkdir(exist_ok=True)

    # Collect all mesh files to process
    mesh_files = [
        file_path
        for file_path in mesh_dir.glob("*.obj")
        if "inverted" not in file_path.parts and file_path.is_file()
    ]

    logger.info(f"Found {len(mesh_files)} mesh files to process for {structure_id}")

    # Process and upload files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_and_upload_mesh,
                file_path,
                base_datadir,
                inverted_dir,
                s3_client,
                bucket,
                use_inverted_meshes,
                reinvert,
                reupload,
            )
            for file_path in mesh_files
        ]

        # Display progress with tqdm
        desc = f"Processing {structure_id} meshes"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                logger.debug(result)
            except Exception as e:
                logger.error(f"Error processing file: {e}")

    return len(mesh_files)
