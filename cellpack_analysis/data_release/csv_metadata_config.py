"""Central configuration for CSV column metadata and descriptions."""

from collections import OrderedDict

# Column metadata: maps column names to their descriptions
# This is the single source of truth for column names and descriptions
# used across the data release workflow
CSV_COLUMN_METADATA: dict[str, str] = OrderedDict(
    [
        ("File Name", "Name of the simularium file"),
        ("Cell ID", "Unique identifier for each cell/packing output"),
        (
            "Rule",
            "Modeling rule: random (unbiased), nucleus_gradient (nuclear bias), membrane_gradient "
            "(membrane bias), apical_gradient (apical bias), or interpolated (mixed rule)",
        ),
        ("Packing ID", "Identifier for the packing directory (e.g. peroxisome)"),
        ("Structure ID", "Identifier for the structure (e.g. SLC25A17)"),
        ("Structure Name", "Name of the structure (e.g. peroxisome)"),
        ("Count", "Number of puncta packed in each replicate (the observed count)"),
        ("Dataset", "Dataset name (e.g. 8d_sphere_data)"),
        ("Condition", "Experimental condition (e.g. rules_shape)"),
        ("Experiment", "Experiment name (e.g. norm_weights)"),
        ("Cell Volume (µm³)", "Volume of the cell"),
        ("Nucleus Volume (µm³)", "Volume of the nucleus"),
        ("Cell Height (µm)", "Height of the cell"),
        ("Nucleus Height (µm)", "Height of the nucleus"),
        ("Cell Sphericity", "Sphericity of the cell"),
        ("File Type", "Type of file (e.g. simularium)"),
        ("File Path", "Path to simularium file on s3 (one replicate per frame)"),
        ("Thumbnail", "Path to thumbnail file on s3 (first replicate)"),
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
