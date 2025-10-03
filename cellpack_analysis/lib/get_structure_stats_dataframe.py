from pathlib import Path

import pandas as pd

from cellpack_analysis.lib.io import load_dataframe


def get_structure_stats_dataframe(
    structure_id: str | None = None
) -> pd.DataFrame:
    """
    Get the structure stats DataFrame from the specified parquet file.
    
    Parameters
    ----------
    structure_id : str | None, optional
        Gene ID of the structure to filter by
        
    Returns
    -------
    pd.DataFrame
        The structure stats DataFrame, optionally filtered by structure_id
    """
    df = load_dataframe(load_local=True, prefix="structure_stats")
    
    if structure_id:
        df = df.loc[df["structure_name"] == structure_id]
    return df


def get_structure_radius(structure_id: str) -> float:
    df_struct_stats = get_structure_stats_dataframe(structure_id=structure_id)
    mean_radius, std_radius = df_struct_stats["radius"].agg(["mean", "std"])

    return mean_radius, std_radius  # type: ignore
