from typing import Optional

import pandas as pd

from cellpack_analysis.lib import default_values


def get_cellid_df(
    load_local: Optional[bool] = False, save_local: Optional[bool] = False
) -> pd.DataFrame:
    """
    Retrieves the cell ID DataFrame from the specified parquet file.

    Returns
    -------
        pd.DataFrame: The cell ID DataFrame.
    """
    s3_path = "s3://cellpack-analysis-data/all_cellids.parquet"
    local_path = default_values.DATADIR / "all_cellids.parquet"

    if load_local:
        if not local_path.exists():
            raise FileNotFoundError(f"Local file {local_path} not found.")

    df_path = s3_path if not load_local else local_path

    df_cellID = pd.read_parquet(df_path)

    if save_local:
        df_cellID.to_parquet(local_path, index=False)

    return df_cellID


def get_cellid_list_for_structure(
    structure_id: str,
    df_cellID: Optional[pd.DataFrame] = None,
    dsphere: Optional[bool] = False,
    load_local: Optional[bool] = False,
    save_local: Optional[bool] = False,
) -> list:
    """
    Get a list of cell IDs for a given structure ID.

    Parameters
    ----------
        structure_id (str): The id of the structure.
        df_cellID (Optional[pd.DataFrame]): Optional DataFrame containing cell IDs.
            If not provided, it will be fetched using get_cellid_df().
        dsphere (Optional[bool]): Flag indicating whether to filter by 8dsphere.

    Returns
    -------
        np.ndarray: An array of cell IDs.
    """
    if df_cellID is None:
        df_cellID = get_cellid_df(load_local=load_local, save_local=save_local)

    condition = df_cellID.structure_name == structure_id
    if dsphere:
        condition = condition & df_cellID["8dsphere"]

    cellid_list = df_cellID.loc[condition, "CellId"].astype(str).tolist()
    return cellid_list
