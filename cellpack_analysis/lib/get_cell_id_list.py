import logging

import pandas as pd

from cellpack_analysis.lib.file_io import get_datadir_path

log = logging.getLogger(__name__)


def get_cell_id_df(load_local: bool = False, save_local: bool = False) -> pd.DataFrame:
    """
    Retrieves the cell ID DataFrame from the specified parquet file.

    Parameters
    ----------
    load_local : bool, default False
        If True, load from local file. If False, load from S3.
    save_local : bool, default False
        If True, save the DataFrame to local storage after loading.

    Returns
    -------
    :
        The cell ID DataFrame.

    Raises
    ------
    FileNotFoundError
        If load_local is True but the local file doesn't exist.
    ValueError
        If the loaded DataFrame is empty or malformed.
    Exception
        For other file read errors.
    """
    s3_path = "s3://cellpack-analysis-data/all_cell_ids.parquet"
    local_path = get_datadir_path() / "all_cell_ids.parquet"

    if load_local and not local_path.exists():
        raise FileNotFoundError(f"Local file {local_path} not found.")

    df_path = local_path if load_local else s3_path
    log.info(f"Loading cell ID data from: {df_path}")

    try:
        df_cell_id = pd.read_parquet(df_path)

        if df_cell_id.empty:
            raise ValueError(f"Loaded DataFrame from {df_path} is empty")

        required_columns = ["CellId", "structure_name"]
        missing_columns = [col for col in required_columns if col not in df_cell_id.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        log.info(f"Successfully loaded {len(df_cell_id)} cell records")

        if save_local and not load_local:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df_cell_id.to_parquet(local_path, index=False)
                log.info(f"Saved cell ID data to local file: {local_path}")
            except Exception as e:
                log.warning(f"Failed to save to local file {local_path}: {e}")

        return df_cell_id

    except Exception as e:
        log.error(f"Failed to load cell ID data from {df_path}: {e}")
        raise


def get_cell_id_list_for_structure(
    structure_id: str,
    df_cell_id: pd.DataFrame | None = None,
    dsphere: bool = False,
    load_local: bool = False,
    save_local: bool = False,
) -> list:
    """
    Get a list of cell IDs for a given structure ID.

    Parameters
    ----------
    structure_id:
        The gene id of the structure.
    df_cell_id:
        Optional DataFrame containing cell IDs.
        If not provided, it will be fetched using get_cell_id_df().
    dsphere:
        If True, filter for 8D sphere data only.
    load_local:
        If True, load the cell ID DataFrame from a local file.
    save_local:
        If True, save the cell ID DataFrame to a local file after loading.

    Returns
    -------
    :
        An array of cell IDs.
    """
    if df_cell_id is None:
        df_cell_id = get_cell_id_df(load_local=load_local, save_local=save_local)

    condition = df_cell_id.structure_name == structure_id
    if dsphere:
        condition = condition & df_cell_id["8dsphere"]

    cell_id_list = df_cell_id.loc[condition, "CellId"].astype(str).tolist()
    return cell_id_list
