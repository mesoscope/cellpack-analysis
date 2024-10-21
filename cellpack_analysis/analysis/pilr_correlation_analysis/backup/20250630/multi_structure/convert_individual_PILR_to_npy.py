# %% [markdown]
# # Split individual PILRs across folders and save in a common file
# %%
import json
import logging
import logging.config
from pathlib import Path

import numpy as np

log_file_path = Path(__file__).parents[4] / "logging.conf"
logging.config.fileConfig(log_file_path, disable_existing_loggers=True)
log = logging.getLogger(__name__)
# %% [markdown]
# ## Define channel names
STRUCTURE_ID = "ER_peroxisome"
EXPERIMENT = "rules_shape"

# %% [markdown]
# ## Set up folders
base_datadir = Path(__file__).parents[4] / "data"
base_result_dir = Path(__file__).parents[4] / "results/PILR_correlation_analysis"

result_dir = base_result_dir / f"{STRUCTURE_ID}/{EXPERIMENT}/"
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load individual PILR dictionary
individual_pilr_path = result_dir / "individual_PILR.json"
log.info(f"Loading individual PILRs from {individual_pilr_path}")
with open(individual_pilr_path) as f:
    individual_pilr_dict = json.load(f)

# %% [markdown]
# ## Split individual PILRs across folders
for channel, pilrs in individual_pilr_dict.items():
    log.info(f"Converting PILRs for {channel} to numpy arrays")
    channel_dir = result_dir / channel
    channel_dir.mkdir(parents=True, exist_ok=True)

    individual_pilrs = np.array(pilrs, dtype=np.float16)
    log.info(f"{channel} PILR shape {individual_pilrs.shape}")

    # Save the PILR as json
    log.info(f"Saving average PILR for {channel}")
    avg_pilr = np.mean(individual_pilrs, axis=0)
    np.save(channel_dir / f"{channel}_average_PILR.npy", avg_pilr)

    # Save individual PILRs
    log.info(f"Saving individual PILR for {channel}")
    np.save(
        channel_dir / f"{channel}_individual_PILR.npy",
        individual_pilrs,
    )

# %% [markdown]
# ## Test whether the individual PILRs have been saved correctly
for ch, pilrs in individual_pilr_dict.items():
    log.info(f"Testing PILRs for {ch}")

    pilrs = np.array(pilrs, dtype=np.float16)
    loaded_pilrs = np.load(result_dir / ch / f"{ch}_individual_PILR.npy")
    assert np.allclose(pilrs, loaded_pilrs)
    log.info(f"PILRs for {ch} are correct")

    avg_pilr = np.mean(pilrs, axis=0)
    loaded_avg_pilr = np.load(result_dir / ch / f"{ch}_average_PILR.npy")
    assert np.allclose(avg_pilr, loaded_avg_pilr)
    log.info(f"Average PILR for {ch} is correct")
# %% [markdown]
# ## Delete individual PILR json file
individual_pilr_path.unlink()
log.info(f"Deleted {individual_pilr_path}")
# %%
