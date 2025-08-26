STRUCTURE_NAME_DICT = {
    "SLC25A17": "peroxisome",
    "RAB5A": "endosome",
    "SEC61B": "ER",
    "ST6GAL1": "golgi",
    "TOMM20": "mitochondria",
}

STRUCTURE_RADIUS_DICT = {
    "SLC25A17": 2.37,
    "RAB5A": 2.6,
}

MODE_LABELS = {
    "SLC25A17": "Peroxisomes",
    "RAB5A": "Endosomes",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
    "mean_count_and_size": "Baseline",
    "variable_size": "Variable Size",
    "variable_count": "Variable Count",
    "variable_count_and_size": "Variable Count and Size",
    "random": "Random",
    "shape": "Shape",
    "nucleus_moderate": "Nucleus bias",
    "nucleus_gradient": "Nucleus bias 0.3",
    "nucleus_gradient_0pt4": "Nucleus bias 0.4",
    "nucleus_gradient_0pt6": "Nucleus bias 0.6",
    "nucleus_gradient_0pt8": "Nucleus bias 0.8",
    "nucleus_gradient_1": "Nucleus bias",
    "nucleus_gradient_1pt2": "Nucleus bias 1.2",
    "nucleus_gradient_1pt4": "Nucleus bias 1.4",
    "nucleus_gradient_1pt6": "Nucleus bias 1.6",
    "nucleus_gradient_strong": "Nucleus bias",
    "nucleus_moderate_invert": "Bias away from nucleus",
    "membrane_gradient_strong": "Membrane bias",
    "planar_gradient_Z_moderate": "Apical bias",
    "apical_gradient": "Apical bias",
    "planar_gradient_Z_moderate_invert": "Basal bias",
    "struct_gradient": "Structure bias",
}

STATIC_SHAPE_MODES = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "variable_count_and_size",
]

VARIABLE_SHAPE_MODES = [
    "SLC25A17",
    "RAB5A",
    "random",
    "shape",
    "nucleus_moderate",
    "nucleus_gradient",
    "nucleus_moderate_invert",
    "planar_gradient_Z_moderate",
    "planar_gradient_Z_moderate_invert",
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    "apical_gradient",
]

DISTANCE_MEASURE_LABELS = {
    "nucleus": "Distance from Nucleus",
    "membrane": "Distance from Membrane",
    "z": "Z distance",
    "nearest": "Nearest Distance",
    "pairwise": "Pairwise Distance",
    "scaled_nucleus": "Scaled Distance from Nucleus",
}

GRID_DISTANCE_LABELS = {
    "nucleus": "nuc_grid_distances",
    "membrane": "mem_grid_distances",
    "z": "z_grid_distances",
    "scaled_nucleus": "scaled_nuc_grid_distances",
}

NORMALIZATION_LABELS = {
    "cell_diameter": "Cell Diameter",
    "intracellular_radius": "Intracellular Radius",
    "max_distance": "Max Distance",
}

# DISTANCE_LIMITS = {
#     "pairwise": (-0.2, 1.2),
#     "nucleus": (-0.1, 0.5),
#     "nearest": (-0.05, 0.25),
#     "z": (-0.1, 0.6),
#     "scaled_nucleus": (-0.2, 1.2),
# }
DISTANCE_LIMITS = {
    "pairwise": (0, 25),
    "nucleus": (0, 6),
    "nearest": (0, 5),
    "z": (0, 12),
    "scaled_nucleus": (0.0, 1.0),
    "membrane": (0.0, 3.2),
}

COLOR_PALETTE = {
    "SLC25A17": "green",
    "peroxisome": "green",
    "RAB5A": "gold",
    "endosome": "gold",
    "SEC61B": "blue",
    "ER": "blue",
    "ST6GAL1": "orange",
    "golgi": "orange",
    "random": "gray",
    "nucleus_gradient_strong": "cyan",
    "membrane_gradient_strong": "magenta",
    "apical_gradient": "brown",
    "struct_gradient": "pink",
}

DATA_CONFIG = {
    "peroxisome": {
        "rules": [
            "observed",
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
        ],
        "radius": STRUCTURE_RADIUS_DICT["SLC25A17"],
        "label": "Peroxisome",
        "structure_id": "SLC25A17",
        "color": "green",
    },
    "ER_peroxisome": {
        "rules": [
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
            "struct_gradient",
        ],
        "radius": STRUCTURE_RADIUS_DICT["SLC25A17"],
        "label": "ER Peroxisome",
        "structure_id": "SEC61B",
        "color": "C0",
    },
    "ER_peroxisome_no_struct": {
        "rules": [
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
        ],
        "radius": STRUCTURE_RADIUS_DICT["SLC25A17"],
        "label": "ER Peroxisome no struct",
        "structure_id": "SEC61B",
        "color": "C1",
    },
    "endosome": {
        "rules": [
            "observed",
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
            "apical_gradient",
        ],
        "radius": STRUCTURE_RADIUS_DICT["RAB5A"],
        "label": "Endosome",
        "structure_id": "RAB5A",
        "color": "gold",
    },
    "golgi_endosome": {
        "rules": [
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
            "struct_gradient",
        ],
        "radius": STRUCTURE_RADIUS_DICT["RAB5A"],
        "label": "Golgi Endosome",
        "structure_id": "ST6GAL1",
        "color": "C2",
    },
    "golgi_endosome_no_struct": {
        "rules": [
            "random",
            "nucleus_gradient_strong",
            "membrane_gradient_strong",
        ],
        "radius": STRUCTURE_RADIUS_DICT["RAB5A"],
        "label": "Golgi Endosome no struct",
        "structure_id": "ST6GAL1",
        "color": "C3",
    },
}


RAW_CHANNEL_MAP = {
    "gfp": 3,
    "mem": 2,
    "nuc": 0,
}

SIM_CHANNEL_MAP = {
    "gfp": 2,
    "mem": 0,
    "nuc": 1,
}

DUAL_STRUCTURE_SIM_CHANNEL_MAP = {
    "gfp": 3,
    "struct": 2,
    "mem": 0,
    "nuc": 1,
}
