"""Central configuration for CSV column metadata and descriptions."""

from collections import OrderedDict

# Column metadata: maps column names to their descriptions
# This is the single source of truth for column names and descriptions
# used across the data release workflow
CSV_COLUMN_METADATA: dict[str, str] = OrderedDict(
    [
        ("File Name", "Name of the simularium file"),
        ("Cell ID", "Unique identifier for each cell/packing output"),
        ("Rule", "Packing rule used (e.g. random, nucleus_gradient, etc.)"),
        ("Packing ID", "Identifier for the packing directory (e.g. peroxisome)"),
        ("Structure ID", "Identifier for the structure (e.g. SLC25A17)"),
        ("Structure Name", "Name of the structure (e.g. peroxisome)"),
        ("Count", "Count of structures packed in the cell (if available)"),
        ("Dataset", "Dataset name (e.g. 8d_sphere_data)"),
        ("Condition", "Experimental condition (e.g. rules_shape)"),
        ("Experiment", "Experiment name (e.g. norm_weights)"),
        ("Cell Volume", "Volume of the cell"),
        ("Nucleus Volume", "Volume of the nucleus"),
        ("Cell Height", "Height of the cell"),
        ("Nucleus Height", "Height of the nucleus"),
        ("Cell Sphericity", "Sphericity of the cell"),
        ("File Type", "Type of file (e.g. simularium)"),
        ("File Path", "Path to simularium file on s3"),
        ("Thumbnail", "Path to thumbnail file on s3"),
    ]
)


def get_column_names() -> list[str]:
    """
    Get list of column names in the correct order.

    Returns
    -------
    list[str]
        Ordered list of column names
    """
    return list(CSV_COLUMN_METADATA.keys())


def get_column_description(column_name: str) -> str:
    """
    Get description for a specific column.

    Parameters
    ----------
    column_name
        Name of the column

    Returns
    -------
    str
        Description of the column

    Raises
    ------
    KeyError
        If column name is not found in metadata
    """
    return CSV_COLUMN_METADATA[column_name]


def get_metadata_dict() -> dict[str, str]:
    """
    Get the full metadata dictionary.

    Returns
    -------
    dict[str, str]
        Dictionary mapping column names to descriptions
    """
    return dict(CSV_COLUMN_METADATA)
