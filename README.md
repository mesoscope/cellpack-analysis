# cellpack-analysis
Analysis pipeline for cellPACK generated synthetic data.

**This repository is a work in progress and is not yet ready for public use.**

This repository contains tools that analyze cellPACK generated synthetic data and compare it to experimental data. 

## Pre-requisites and installation
This code is intended to run using a python installation with some other libraries. Using a conda environment is recommended. The following instructions assume that you have conda installed. If you do not, you can install it by following the instructions at https://docs.conda.io/projects/conda/en/latest/user-guide/install/

1. Install cellPACK by following instructions at https://github.com/mesoscope/cellpack
2. Create a conda environment: `conda create -n cellpack-analysis python=3.9`
3. Activate the conda environment: `conda activate cellpack-analysis`
4. Install packages: `pip install -e .`

## Usage

The analysis pipeline contains multiple steps that can be run independently. 

### 1. Download experimental data
The experimental data also contains multi-channel segmented images with the channels corresponding to the cell membrane, nucleus, and the structure being analyzed. The experimental data can be downloaded using the notebook `cellpack_analysis/notebooks/get_raw_images.ipynb`

### 2. Generate synthetic data
To generate synthetic data, it is recommended to use a local copy of the development branch of cellPACK. 
Creating synthetic data will require the conversion of the raw images into 3D meshes which can be done using the script `cellpack_analysis/scripts/get_meshes_from_raw_images.py`. 
The meshes can then be used to generate synthetic data using cellPACK.
The intended output of this step is a set of multi-channel segmented images with the channels corresponding to the cell membrane, nucleus, and the structure being analyzed.

### 3. Calculate parameterized representations for the synthetic and experimental data
The parameterized representations are calculated using the script `cellpack_analysis/scripts/calculate_PILR_from_images.sh`. This script takes as input the structure information, the path to the experimental data, and the path to the synthetic data. The output is a set of parameterized representations for the synthetic and experimental data.

### 4. Calculate the similarity between the synthetic and experimental data
To calculate correlations between individual parameterized representations, use the script `cellpack_analysis/scripts/calculate_individual_PILR_correlations.sh`. This script takes as input the path to the parameterized representations for the synthetic and experimental data. The output is a set of correlation matrices for the synthetic and experimental data saved as dataframes.

### 5. Visualize and create plots
To visualize the correlations between individual parameterized representations, use the notebook `cellpack_analysis/notebooks/individual_PILR_heatmap.ipynb`