# %%
# Upload meshes to s3 bucket
# %%
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.mesh_tools import invert_mesh_faces

# %%
base_datadir = get_datadir_path()
structure_id = "RAB5A"
S3_CLIENT = boto3.client("s3")
# %%


def process_and_upload_mesh(
    file_path: Path,
    base_datadir: Path,
    inverted_dir: Path,
    use_inverted_meshes: bool,
    reinvert: bool,
    reupload: bool,
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
        path_to_upload = inverted_mesh_path

    if reupload:
        S3_CLIENT.upload_file(str(path_to_upload), "cellpack-analysis-data", s3_key)
        S3_CLIENT.put_object_acl(
            ACL="public-read",
            Bucket="cellpack-analysis-data",
            Key=s3_key,
        )
        return f"Uploaded {file_path.name}"
    else:
        return f"Processed {file_path.name}"


# %%
# Invert and upload meshes to s3 bucket
use_inverted_meshes = True
reupload = True  # Whether to actually upload to S3 or just process files without uploading
reinvert = False  # Whether to re-invert meshes if they already exist in the inverted folder on disk
max_workers = 8  # Number of parallel threads

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

# Process and upload files in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(
            process_and_upload_mesh,
            file_path,
            base_datadir,
            inverted_dir,
            use_inverted_meshes,
            reinvert,
            reupload,
        )
        for file_path in mesh_files
    ]

    # Display progress with tqdm
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing meshes"):
        try:
            result = future.result()
        except Exception as e:
            print(f"Error processing file: {e}")

# %%
