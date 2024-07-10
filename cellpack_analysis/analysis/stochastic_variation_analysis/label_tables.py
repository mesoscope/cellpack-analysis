STRUCTURE_NAME_DICT = {
    "SLC25A17": "peroxisome",
    "RAB5A": "endosome",
}

MODE_LABELS = {
    "SLC25A17": "Observed Peroxisomes",
    "RAB5A": "Observed Endosomes",
    "mean_count_and_size": "Mean Count and Size",
    "variable_size": "Variable Size",
    "variable_count": "Variable Count",
    "variable_count_and_size": "Variable Count and Size",
    "random": "Random",
    "shape": "Shape",
    "nucleus_moderate": "Nucleus bias",
    "nucleus_moderate_invert": "Membrane bias",
    "planar_gradient_Z_moderate": "Apical bias",
    "planar_gradient_Z_moderate_invert": "Basal bias",
}

VARIABLE_SHAPE_MODES = [
    "SLC25A17",
    "RAB5A",
    "random",
    "shape",
    "nucleus_moderate",
    "nucleus_moderate_invert",
    "planar_gradient_Z_moderate",
    "planar_gradient_Z_moderate_invert",
]

DISTANCE_MEASURE_LABELS = {
    "nucleus": "Distance from Nucleus",
    "membrane": "Distance from Membrane",
    "z": "Z distance",
}

GRID_DISTANCE_LABELS = {
    "nucleus": "nuc_grid_distances",
    "membrane": "mem_grid_distances",
    "z": "z_grid_distances",
}

NORMALIZATION_LABELS = {
    "cell_diameter": "Cell Diameter",
    "intracellular_radius": "Intracellular Radius",
}
