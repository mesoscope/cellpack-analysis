# cellpack-analysis
Analysis pipeline for cellPACK

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

For basic installation with just the core dependencies:

```bash
uv sync --no-dev
```

If you plan to develop code, you should also install the development dependencies:

```bash
uv sync
```

To install extra dependencies:

```bash
uv sync --all-extras
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

## Usage

Coming soon!