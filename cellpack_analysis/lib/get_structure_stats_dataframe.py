from pathlib import Path
from typing import Optional

import pandas as pd


def get_local_stats_dataframe_path(datadir: Optional[Path] = None) -> Path:
    if datadir is None:
        datadir = get_datadir_path()
    return datadir / "structure_stats.parquet"


def get_datadir_path() -> Path:
    datadir = Path(__file__).parents[2] / "data"
    datadir.mkdir(exist_ok=True, parents=True)
    return datadir


def get_structure_stats_dataframe(datadir: Optional[Path] = None):
    df_path = get_local_stats_dataframe_path(datadir)
    if df_path.exists():
        return pd.read_parquet(df_path)

    raise FileNotFoundError(f"Local file {df_path} not found.")
