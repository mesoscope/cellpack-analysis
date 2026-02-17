import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cellpack_analysis.lib.file_io import get_datadir_path

logger = logging.getLogger(__name__)


def count_all_keys(d: dict) -> int:
    """
    Count all keys in a dictionary, including nested dictionaries.

    Parameters
    ----------
    d
        Dictionary to count keys from

    Returns
    -------
    :
        Total number of keys at all nesting levels
    """
    count = len(d)  # Count keys at current level
    for value in d.values():
        if isinstance(value, dict):
            count += count_all_keys(value)  # Recursively count nested keys
    return count


def format_time(seconds):
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


def load_dataframe(load_local: bool = True, prefix: str = "all_cell_ids") -> pd.DataFrame:
    """
    Retrieve the cell ID DataFrame from the specified parquet file.

    Parameters
    ----------
    load_local
        If True, attempt to load from local file first. If False, load from S3.

    Returns
    -------
    :
        The cell ID DataFrame.

    Raises
    ------
    FileNotFoundError
        If load_local is True but the local file doesn't exist and S3 loading fails.
    ValueError
        If the loaded DataFrame is empty or malformed.
    Exception
        For other file read errors.
    """
    s3_path = f"s3://cellpack-analysis-data/{prefix}.parquet"
    local_path = get_datadir_path() / f"{prefix}.parquet"

    # Determine data source
    loaded_from_s3 = False

    if load_local and local_path.exists():
        df_path = local_path
        logger.debug(f"Loading from local file: {df_path}")
    else:
        if load_local and not local_path.exists():
            logger.warning(f"Local file {local_path} not found, loading from S3.")
        df_path = s3_path
        loaded_from_s3 = True
        logger.debug(f"Loading from S3: {df_path}")

    try:
        df = pd.read_parquet(df_path)

        if df.empty:
            raise ValueError(f"Loaded DataFrame from {df_path} is empty")

        logger.debug(f"Successfully loaded {len(df)} cell records")

        # Save to local if we loaded from S3
        if loaded_from_s3:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(local_path, index=False)
                logger.info(f"Saved to local file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to save to local file {local_path}: {e}")

        return df

    except Exception as e:
        logger.error(f"Failed to load from {df_path}: {e}")
        raise
