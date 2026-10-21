import logging

import numpy as np
import pandas as pd

from cellpack_analysis.lib.io import load_dataframe

logger = logging.getLogger(__name__)


def get_cell_id_df() -> pd.DataFrame:
    """
    Retrieve the cell ID DataFrame from the local release table ``all_cell_ids.parquet``.

    Returns
    -------
    :
        The cell ID DataFrame.

    Raises
    ------
    FileNotFoundError
        If the local file is missing (fetch it with the paper_inputs ``tables`` group).
    ValueError
        If the loaded DataFrame is empty.
    """
    return load_dataframe(prefix="all_cell_ids")


def get_cell_id_list_for_structure(
    structure_id: str,
    df_cell_id: pd.DataFrame | None = None,
    dsphere: bool = False,
) -> list[str]:
    """
    Get list of cell IDs for a given structure ID.

    Parameters
    ----------
    structure_id
        Gene ID of the structure to filter by
    df_cell_id
        DataFrame containing cell IDs. If not provided, fetches using get_cell_id_df()
    dsphere
        If True, filter for 8D sphere data only. Default is False

    Returns
    -------
    :
        List of cell IDs matching the specified structure
    """
    if df_cell_id is None:
        df_cell_id = get_cell_id_df()

    condition = df_cell_id.structure_name == structure_id
    if dsphere:
        condition = condition & df_cell_id["8dsphere"]

    cell_id_list = df_cell_id.loc[condition, "CellId"].astype(str).tolist()
    return cell_id_list


def sample_cell_ids_for_structure(
    structure_id: str,
    num_cells: int | None = None,
    dsphere: bool = True,
) -> list[str]:
    """
    Sample a specified number of cell IDs for a given structure ID.

    Parameters
    ----------
    structure_id
        The gene id of the structure.
    num_cells
        The number of cell IDs to sample.
    dsphere
        If True, filter for 8D sphere data only.

    Returns
    -------
    :
        A list of sampled cell IDs.
    """
    all_cell_ids = get_cell_id_list_for_structure(
        structure_id=structure_id,
        dsphere=dsphere,
    )

    if len(all_cell_ids) == 0:
        logger.warning(f"No cell IDs found for structure_id={structure_id} with dsphere={dsphere}")
        return []

    if num_cells is None or num_cells >= len(all_cell_ids):
        return all_cell_ids

    return np.random.choice(all_cell_ids, size=num_cells, replace=False).tolist()
