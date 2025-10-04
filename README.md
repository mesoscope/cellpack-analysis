# cellPACK analysis pipeline

> [!WARNING]
> This repository is a work in progress and is not yet ready for public use.

This repository contains tools that analyze cellPACK generated synthetic data and compare it to experimental data. 

## Installation
This project requires Python 3.11. The installation is managed using [uv](https://docs.astral.sh/uv/).

Dependencies are listed in the `pyproject.toml` file and locked in the `uv.lock` file.

**1. Navigate to where you want to clone this repository**

```bash
cd /path/to/directory/
```

**2. Clone the repo from GitHub**

```bash
git clone git@github.com:mesoscope/cellpack-analysis.git
cd cellpack-analysis
```

**3. Install the dependencies using uv**

```bash
uv sync
```

**4. Activate the virtual environment**

Activate the virtual environment in the terminal:

For Windows:

```powershell
\path\to\venv\Scripts\activate
```

For Linux/Mac:

```bash
source /path/to/venv/bin/activate
```

You can deactivate the virtual environment using:

```
deactivate
```

### Alternative installation using `pip`

This project also includes a `requirements.txt` generated from the `uv.lock` file, which can be used to install requirements using `pip`.

> [!NOTE]
> This installation method will only install core dependencies.
> We recommend using uv to handle more complex installations of development and optional dependencies.

After cloning the repository, follow these steps:

**1. Create and activate a new virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install the dependencies using pip**

```bash
pip install -r requirements.txt
```

**3. Install the package**

```bash
pip install -e .
```

## Example usage

### Analyze the radial bias for peroxisomes in cells from the [hiPSC single cell image dataset](https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset).

**1. Download images**
```bash
python cellpack_analysis/preprocessing/get_structure_images.py --structure-id SLC25A17
```

**2. Get meshes from images**
```bash
python cellpack_analysis/preprocessing/get_meshes_from_images.py --structure-id SLC25A17
```

**3. Run simulations with radial bias rules**
```bash
python cellpack_analysis/packing/run_packing_workflow.py -c cellpack_analysis/packing/configs/examples/peroxisome_example.json
```

**4. Extract structure coordinates from segmented images**
```bash
python cellpack_analysis/preprocessing/get_structure_coordinates.py --structure-id SLC25A17
```

**5. (WIP) Calculate grid distances from membrane and nucleus**
```bash
python cellpack_analysis/preprocessing/calculate_available_space.py --structure-id SLC25A17
```

**6. (WIP) Run distance analysis workflow**
```bash
python cellpack_analysis/workflows/run_analysis_workflow.py -c cellpack_analysis/workflows/configs/distance_analysis_config_peroxisome.json
```