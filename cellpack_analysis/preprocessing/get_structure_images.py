# %% [markdown]
# # Get raw images for a structure
from pathlib import Path

import pandas as pd
import quilt3
from tqdm import tqdm

from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe

tqdm.pandas()
# %% [markdown]
# ### Set data directory
datadir = Path(__file__).parents[2] / "data"
datadir.mkdir(exist_ok=True, parents=True)
# %% [markdown]
# ### Load variance dataset package from quilt
pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)

# %% [markdown]
# ### Load variance dataset
redownload = False
meta_df = get_variance_dataframe(datadir, redownload, pkg)
print(meta_df.structure_name.unique())

# %% [markdown]
# ### Set structure of interest
# - `SLC25A17` is peroxisomes
# - `RAB5A` is early endosomes
# - `LAMP1` is lysosomes
STRUCTURE_ID = "LAMP1"

# %% [markdown]
# ### Get cellID list for structure
dsphere = True
cellid_list = get_cellid_list_for_structure(STRUCTURE_ID, dsphere=dsphere)
# %% [markdown]
# ### Create dataframe for structure metadata
meta_df_struct = meta_df.loc[cellid_list].reset_index()
# %% [markdown]
# ### Prepare file paths to save images
download_raw = False
subfolder_name = "sample_8d" if dsphere else "full"
folder_name = "unsegmented" if download_raw else "segmented"

save_path = Path(
    datadir / f"structure_data/{STRUCTURE_ID}/{subfolder_name}/{folder_name}"
)
save_path.mkdir(exist_ok=True, parents=True)


# %% [markdown]
# ### Define function to download images
def download_image(row: pd.DataFrame, col_name: str, save_path: Path, pkg) -> Path:
    subdir_name = row[col_name].split("/")[0]
    file_name = row[col_name].split("/")[1]
    local_filename = (
        save_path
        / f"{row.structure_name}_{row.CellId}_ch_{row.ChannelNumberStruct}_{col_name}_original.tiff"
    )
    if not local_filename.exists():
        # print(f"Downloading {local_filename.name}")
        pkg[subdir_name][file_name].fetch(local_filename)
    else:
        print(f"{local_filename.name} already exists. Skipping download.")
    return local_filename


# %% [markdown]
# ### Start download
col_name = "crop_raw" if download_raw else "crop_seg"
meta_df_struct.progress_apply(
    lambda row: download_image(row, col_name, save_path, pkg), axis=1
)
print("Done")
# %%
