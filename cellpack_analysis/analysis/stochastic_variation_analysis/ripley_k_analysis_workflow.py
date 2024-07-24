# %% [markdown]
# # Ripley's K Analysis Workflow
from pathlib import Path

from cellpack_analysis.analysis.stochastic_variation_analysis import distance
from cellpack_analysis.lib.load_data import (
    get_position_data_from_outputs,
)
from cellpack_analysis.lib.mesh_tools import get_mesh_information_dict

# %% [markdown]
# ## Set up parameters
# %% [markdown]
# ### set structure ID
STRUCTURE_ID = "SLC25A17"
STRUCT_RADIUS = 2.37  # 2.37 um for peroxisomes, 2.6 um for early endosomes
# %% [markdown]
# ### set packing modes
baseline_analysis = False
packing_modes_baseline = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    # "variable_count_and_size",
    "shape",
]
packing_modes_rules = [
    STRUCTURE_ID,
    "random",
    "nucleus_moderate",
    "nucleus_moderate_invert",
    # TODO: add bias towards membrane
    # TODO: add bias away from membrane
    "planar_gradient_Z_moderate",
]
packing_modes = packing_modes_baseline if baseline_analysis else packing_modes_rules
baseline_mode = "mean_count_and_size" if baseline_analysis else STRUCTURE_ID
# %% [markdown]
# ### set file paths and setup parameters
base_datadir = Path(__file__).parents[3] / "data"
base_results_dir = Path(__file__).parents[3] / "results"

folder_name = "baseline" if baseline_analysis else "rules"
results_dir = (
    base_results_dir / f"stochastic_variation_analysis/{STRUCTURE_ID}/{folder_name}"
)
results_dir.mkdir(exist_ok=True, parents=True)

figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### read position data from outputs
all_positions = get_position_data_from_outputs(
    structure_id=STRUCTURE_ID,
    packing_modes=packing_modes,
    base_datadir=base_datadir,
    results_dir=results_dir,
    recalculate=False,
    baseline_analysis=baseline_analysis,
)
# %% [markdown]
# ### check number of packings for each result
for mode, position_dict in all_positions.items():
    print(mode, len(position_dict))
# %% [markdown]
# ### calculate mesh information
mesh_information_dict = get_mesh_information_dict(
    structure_id=STRUCTURE_ID,
    base_datadir=base_datadir,
    recalculate=False,
)
# %% [markdown]
# ## Ripley's K Analysis
# %% [markdown]
# ## calculate ripleyK for all positions
all_ripleyK, mean_ripleyK, ci_ripleyK, r_values = distance.calculate_ripley_k(
    all_positions=all_positions,
    mesh_information_dict=mesh_information_dict,
)
# %% [markdown]
# ## plot ripleyK distributions
distance.plot_ripley_k(
    mean_ripleyK=mean_ripleyK,
    ci_ripleyK=ci_ripleyK,
    r_values=r_values,
    figures_dir=figures_dir,
)
