import pandas as pd

from cellpack_analysis.lib.io import FETCH_TABLES_COMMAND, load_dataframe

# Median puncta radius (um) of the release table is ~0.26-0.28; the legacy pixel file gives ~2.4-2.6
MAX_MEDIAN_RADIUS_UM = 1.0
CHECKED_STRUCTURES = ("SLC25A17", "RAB5A")


def validate_structure_stats_units(df: pd.DataFrame) -> None:
    """
    Check that structure_stats is the micrometer-unit release table.

    Parameters
    ----------
    df
        The full structure stats DataFrame.

    Raises
    ------
    ValueError
        If the table has the legacy layout (``sphericity``, no ``mem_sphericity``) or the median
        radius of SLC25A17 or RAB5A is at least 1.0, i.e. it is in pixels.
    """
    reasons = []
    if "mem_sphericity" not in df.columns:
        reasons.append("missing column 'mem_sphericity'")
    if "sphericity" in df.columns:
        reasons.append("has the legacy column 'sphericity'")
    for structure_id in CHECKED_STRUCTURES:
        radius = df.loc[df["structure_name"] == structure_id, "radius"]
        if len(radius) and radius.median() >= MAX_MEDIAN_RADIUS_UM:
            reasons.append(f"median radius of {structure_id} is {radius.median():.3f} (>= 1.0)")
    if reasons:
        raise ValueError(
            f"pixel-unit structure_stats; fetch the release tables ({FETCH_TABLES_COMMAND}). "
            f"Problems: {'; '.join(reasons)}"
        )


def get_structure_stats_dataframe(structure_id: str | None = None) -> pd.DataFrame:
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
    df = load_dataframe(prefix="structure_stats")
    validate_structure_stats_units(df)

    if structure_id:
        df = df.loc[df["structure_name"] == structure_id]
    return df
