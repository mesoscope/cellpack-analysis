import pandas as pd

from cellpack_analysis.lib import default_values


def get_cell_id_df(
    load_local: bool | None = False, save_local: bool | None = False
) -> pd.DataFrame:
    """
    Retrieves the cell ID DataFrame from the specified parquet file.

    Returns
    -------
        pd.DataFrame: The cell ID DataFrame.
    """
    s3_path = "s3://cellpack-analysis-data/all_cell_ids.parquet"
    local_path = default_values.DATADIR / "all_cell_ids.parquet"

    if load_local:
        if not local_path.exists():
            raise FileNotFoundError(f"Local file {local_path} not found.")

    df_path = s3_path if not load_local else local_path

    df_cell_id = pd.read_parquet(df_path)

    if save_local:
        df_cell_id.to_parquet(local_path, index=False)

    return df_cell_id


def get_cell_id_list_for_structure(
    structure_id: str,
    df_cell_id: pd.DataFrame | None = None,
    dsphere: bool | None = False,
    load_local: bool | None = False,
    save_local: bool | None = False,
) -> list:
    """
    Get a list of cell IDs for a given structure ID.

    Parameters
    ----------
        structure_id (str): The id of the structure.
        df_cell_id (Optional[pd.DataFrame]): Optional DataFrame containing cell IDs.
            If not provided, it will be fetched using get_cell_id_df().
        dsphere (Optional[bool]): Flag indicating whether to filter by 8dsphere.

    Returns
    -------
        np.ndarray: An array of cell IDs.
    """
    if df_cell_id is None:
        df_cell_id = get_cell_id_df(load_local=load_local, save_local=save_local)

    condition = df_cell_id.structure_name == structure_id
    if dsphere:
        condition = condition & df_cell_id["8dsphere"]

    cell_id_list = df_cell_id.loc[condition, "CellId"].astype(str).tolist()
    return cell_id_list
