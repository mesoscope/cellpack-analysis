from pathlib import Path

import pandas as pd

from cellpack_analysis.lib.file_io import get_datadir_path


def get_local_stats_dataframe_path(datadir: Path | None = None) -> Path:
    if datadir is None:
        datadir = get_datadir_path()
    return datadir / "structure_stats.parquet"


def get_structure_stats_dataframe(
    datadir: Path | None = None, structure_id: str | None = None
) -> pd.DataFrame:
    if datadir is None:
        datadir = get_datadir_path()
    df_path = get_local_stats_dataframe_path(datadir)
    if df_path.exists():
        df = pd.read_parquet(df_path)
        if structure_id:
            df = df.loc[df["structure_name"] == structure_id]
        df.index = df.index.astype(str)
        return df

    raise FileNotFoundError(f"Local file {df_path} not found.")


def get_structure_radius(structure_id: str, datadir: Path | None = None) -> float:
    df_struct_stats = get_structure_stats_dataframe(
        structure_id=structure_id, datadir=datadir
    )
    mean_radius, std_radius = df_struct_stats["radius"].agg(["mean", "std"])

    return mean_radius, std_radius  # type: ignore
