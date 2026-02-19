# %%
# Upload meshes to s3 bucket
# %%
import shutil

import boto3
from tqdm import tqdm

from cellpack_analysis.lib.file_io import get_datadir_path
from cellpack_analysis.lib.mesh_tools import invert_mesh_faces

# %%
base_datadir = get_datadir_path()
structure_id = "SLC25A17"

base_s3_mesh_url = (
    "https://cellpack-analysis-data.s3.us-west-2.amazonaws.com/structure_data/SLC25A17/meshes/"
)
s3_client = boto3.client("s3")
# %%
# Invert and upload meshes to s3 bucket
reupload = False
invert_meshes = False

if reupload:
    mesh_dir = base_datadir / "structure_data" / structure_id / "meshes"
    for file_path in tqdm(mesh_dir.rglob("*")):
        if file_path.is_file():
            # invert mem meshes
            if invert_meshes and "mem" in file_path.name and "original" not in file_path.name:
                orig_mesh_path = file_path.parent / f"{file_path.stem}_original.obj"
                # copy original file with suffix
                shutil.copy(
                    file_path,
                    orig_mesh_path,
                )
                invert_mesh_faces(input_mesh_path=orig_mesh_path, output_mesh_path=file_path)

            # upload to s3
            s3_key = file_path.relative_to(base_datadir).as_posix()
            s3_client.upload_file(str(file_path), "cellpack-analysis-data", s3_key)
            s3_client.put_object_acl(
                ACL="public-read",
                Bucket="cellpack-analysis-data",
                Key=s3_key,
            )
