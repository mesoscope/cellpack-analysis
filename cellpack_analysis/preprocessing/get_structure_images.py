# %% [markdown]
# # Get raw images for a structure
import logging
from pathlib import Path

import pandas as pd
import quilt3
from tqdm import tqdm

from cellpack_analysis.lib.get_cellid_list import get_cellid_list_for_structure
from cellpack_analysis.lib.get_variance_dataset import get_variance_dataframe

log = logging.getLogger(__name__)

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
meta_df.index = meta_df.index.astype(str)
log.info(meta_df.structure_name.unique())

# %% [markdown]
# ### Set structure of interest
# - `SLC25A17` is peroxisomes
# - `RAB5A` is early endosomes
# - `LAMP1` is lysosomes
# - `SEC61B` is ER
# - `ATP2A2` is smooth ER
# - `TOMM20` is mitochondria
# - `ST6GAL1` is Golgi
STRUCTURE_ID = "SLC25A17"

# %% [markdown]
# ### Get cellID list for structure
dsphere = True
cellid_list = get_cellid_list_for_structure(STRUCTURE_ID, dsphere=dsphere)
log.info(f"Found {len(cellid_list)} cell IDs for {STRUCTURE_ID}")
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
log.info(f"Images will be saved to {save_path}")


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
        # log.info(f"Downloading {local_filename.name}")
        pkg[subdir_name][file_name].fetch(local_filename)
    else:
        log.info(f"{local_filename.name} already exists. Skipping download.")
    return local_filename


# %% [markdown]
# ### Start download
col_name = "crop_raw" if download_raw else "crop_seg"
meta_df_struct.apply(lambda row: download_image(row, col_name, save_path, pkg), axis=1)
log.info("Done")

# %%
