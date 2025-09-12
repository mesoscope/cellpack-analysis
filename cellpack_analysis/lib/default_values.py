from pathlib import Path

# Top level parameters
PIXEL_SIZE_IN_UM = 0.108
"""Pixel size in microns."""
STRUCTURE_NAME = "peroxisome"
"""Name of the structure to pack."""
STRUCTURE_ID = "SLC25A17"
"""ID of the structure to pack."""
CONDITION = "rules_shape"
"""Simulation condition."""
WORKFLOW_CONFIG_PATH = Path(__file__).parents[1] / "packing/configs/peroxisome.json"
"""Path to the workflow configuration file."""

# Data directory
DATADIR = Path(__file__).parents[2] / "data"
"""Path to the data directory."""

# Default simulation options
DRY_RUN = False
"""Flag indicating whether to run a dry run."""
GENERATE_RECIPES = True
"""Flag indicating whether to generate recipes."""
GENERATE_CONFIGS = True
"""Flag indicating whether to generate configs."""
GET_COUNTS_FROM_DATA = False
"""Flag indicating whether to get counts from data."""
GET_SIZE_FROM_DATA = False
"""Flag indicating whether to get size from data."""
GET_BOUNDING_BOX_FROM_MESH = False
"""Flag indicating whether to get bounding box from mesh."""
RESULT_TYPE = "simularium"
"""While skipping the packing, check for simularium file."""
SKIP_COMPLETED = False
"""Flag indicating whether to skip completed packings."""
USE_CELLS_IN_8D_SPHERE = False
"""Flag indicating whether to use cells in 8D sphere."""
USE_MEAN_CELL = False
"""Flag indicating whether additional structure is packed"""
USE_ADDITIONAL_STRUCT = False
"""Structure to apply gradient to."""
GRADIENT_STRUCTURE_NAME = None
"""Flag indicating whether to use mean cell."""
MULTIPLE_REPLICATES = False
"""
Flag indicating whether a single recipe packs multiple replicates.
If this flag is true, the random seed is not assigned based on cell_id and instead
calculated from the cellPACK algorithm.
"""
NUM_PROCESSES = 1
"""Number of processes to use for parallel processing. 1 indicates serial processing."""

# Default gradient settings
SURFACE_GRADIENT = {
    "description": "gradient based on distance from a surface",
    "pick_mode": "rnd",
    "mode": "surface",
    "mode_settings": {
        "object": "nucleus",
        "scale_to_next_surface": False,
    },
    "weight_mode": "exponential",
    "weight_mode_settings": {"decay_length": 0.1},
}
"""Surface gradient settings."""
