import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cellpack_analysis.lib.file_io import get_datadir_path

logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format time in seconds to a human readable format."""
    if seconds == np.inf:
        return "∞"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def is_url(path: Path | str) -> bool:
    """Check if a path is a URL."""
    str_path = str(path)
    return (
        str_path.startswith("http://")
        or str_path.startswith("https://")
        or str_path.startswith("s3://")
    )


FETCH_TABLES_COMMAND = "python -m cellpack_analysis.data_release.paper_inputs fetch --groups tables"


def load_dataframe(prefix: str = "all_cell_ids") -> pd.DataFrame:
    """
    Load a release table (parquet) from the local data directory.

    There is no remote fallback: the tables are versioned release inputs and must be fetched with
    ``python -m cellpack_analysis.data_release.paper_inputs fetch --groups tables``.

    Parameters
    ----------
    prefix
        Prefix for the parquet file name.

    Returns
    -------
    :
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the local parquet file does not exist.
    ValueError
        If the loaded DataFrame is empty.
    """
    local_path = get_datadir_path() / f"{prefix}.parquet"
    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found. Fetch the release tables with: {FETCH_TABLES_COMMAND}"
        )

    logger.debug(f"Loading from local file: {local_path}")
    try:
        df = pd.read_parquet(local_path)
        if df.empty:
            raise ValueError(f"Loaded DataFrame from {local_path} is empty")
    except Exception as e:
        logger.error(f"Failed to load from {local_path}: {e}")
        raise

    logger.debug(f"Successfully loaded {len(df)} records")
    return df
