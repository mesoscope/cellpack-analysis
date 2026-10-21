from pathlib import Path

import pandas as pd
import quilt3

from cellpack_analysis.lib.file_io import get_datadir_path


def get_local_variance_dataframe_path() -> Path:
    return get_datadir_path() / "variance_dataset.parquet"


def get_variance_dataframe(
    redownload: bool = False, pkg: quilt3.Package | None = None
) -> pd.DataFrame:
    df_path = get_local_variance_dataframe_path()
    if not df_path.exists() or redownload:
        if pkg is None:
            pkg = quilt3.Package.browse(
                "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
            )
        meta_df = pkg["metadata.csv"]()
        meta_df.set_index("CellId", inplace=True)
        meta_df.to_parquet(df_path)
    else:
        meta_df = pd.read_parquet(df_path)

    return meta_df
