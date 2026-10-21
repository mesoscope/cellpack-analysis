from pathlib import Path

MPLSTYLE_PATH = Path(__file__).parent / "cellpack_analysis.mplstyle"
"""Path to the shared matplotlib style sheet."""

# Top level parameters
PIXEL_SIZE_IN_UM = 0.108
"""Pixel size in microns."""

PACKING_ID = "peroxisome"
"""ID of the packing."""

STRUCTURE_ID = "SLC25A17"
"""ID of the structure to pack."""

CONDITION = "replicate_99_v4"
"""Simulation condition."""

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

NUM_CELLS = None
"""Number of cells to use. If None, use all available cells."""

USE_MEAN_CELL = False
"""Flag indicating whether to use mean cell."""

USE_ADDITIONAL_STRUCT = False
"""Flag indicating whether to use additional structure (ER, golgi, etc.)."""

GRADIENT_STRUCTURE_NAME = None
"""Name of the structure to apply gradient to."""

NUM_PROCESSES = 1
"""Number of processes to use for parallel processing. 1 indicates serial processing."""

NUM_REPLICATES = 1
"""Number of replicates to pack."""
