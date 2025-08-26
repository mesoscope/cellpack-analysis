from pathlib import Path

import pandas as pd
import quilt3


def get_local_variance_dataframe_path(datadir: Path | None = None) -> Path:
    if datadir is None:
        datadir = get_datadir_path()
    return datadir / "variance_dataset.parquet"


def get_datadir_path() -> Path:
    datadir = Path(__file__).parents[2] / "data"
    datadir.mkdir(exist_ok=True, parents=True)
    return datadir


def get_variance_dataframe(datadir: Path | None = None, redownload=False, pkg=None):
    df_path = get_local_variance_dataframe_path(datadir)
    if not df_path.exists() or redownload:
        if pkg is None:
            pkg = quilt3.Package.browse(
                "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
            )
        meta_df = pkg["metadata.csv"]()
        meta_df.set_index("CellId", inplace=True)
        meta_df.to_parquet(datadir / "variance_dataset.parquet")
    else:
        meta_df = pd.read_parquet(df_path)

    return meta_df
