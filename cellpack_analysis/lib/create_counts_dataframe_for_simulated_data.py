# %% [markdown]
# # Creates a dataframe of counts and sizes from simulated data
# %% [markdown]
# ## Imports
from cellpack_analysis.lib.file_io import get_datadir_path

# %% [markdown]
# ## Set structure information
# this is the name of the structure for which the nucleus and membrane shapes are obtained
STRUCTURE_ID = "SEC61B"

CONDITION_NAME = "rules_shape"

SUBFOLDER = "8d_sphere"

RULES = []
# %% [markdown]
# ## Get simulated data
datadir = get_datadir_path()
