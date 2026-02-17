import colorsys

import seaborn as sns


def adjust_color_saturation(hex_color, saturation):
    """
    Adjust the saturation of a hex color.

    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., '#ff0000' or 'ff0000')
    saturation : float
        Saturation value between 0 and 1

    Returns
    -------
    str
        Hex color code with adjusted saturation
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Update saturation
    s = max(0, min(1, saturation))

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert to hex
    r_hex = format(int(r * 255), "02x")
    g_hex = format(int(g * 255), "02x")
    b_hex = format(int(b * 255), "02x")

    return f"#{r_hex}{g_hex}{b_hex}"


colormap = sns.color_palette("tab10", 10).as_hex()
colormap = [*colormap, "#000aff", "#00ff8a"]

# Names of tagged genes
STRUCTURE_NAME_DICT = {
    "SLC25A17": "peroxisome",
    "RAB5A": "endosome",
    "SEC61B": "ER",
    "ST6GAL1": "golgi",
    "TOMM20": "mitochondria",
}

# Radii of punctate structures in voxels
STRUCTURE_RADIUS = {
    "SLC25A17": 2.37,
    "RAB5A": 2.6,
}

# Labels for cellPACK simulated packing modes
MODE_LABELS = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
    "mean_count_and_size": "Baseline",
    "variable_size": "Size",
    "variable_count": "Count",
    "variable_count_and_size": "Count and size",
    "random": "Random",
    "shape": "Shape",
    "interpolated": "Interpolated",
    "nucleus_moderate": "Nucleus",
    "nucleus_gradient": "Nucleus",
    "nucleus_gradient_strong": "Nucleus",
    "nucleus_moderate_invert": "Bias away from nucleus",
    "membrane_gradient": "Membrane",
    "membrane_gradient_strong": "Membrane",
    "planar_gradient_Z_moderate": "Apical",
    "apical_gradient": "Apical",
    "apical_gradient_weak": "Apical",
    "planar_gradient_Z_moderate_invert": "Basal",
    "struct_gradient": "Structure",
    "struct_gradient_weak": "Structure",
    "1_random": "Spacing 100nm",
    "2_random": "Spacing 200nm",
    "4_random": "Spacing 400nm",
}

# Packing modes in the mean shape - do not have a cell ID associated
STATIC_SHAPE_MODES = [
    "mean_count_and_size",
    "variable_size",
    "variable_count",
    "variable_count_and_size",
]

# Packing modes in observed cell shapes - have a cell ID associated
VARIABLE_SHAPE_MODES = [
    "SLC25A17",
    "RAB5A",
    "random",
    "shape",
    "interpolated",
    "nucleus_moderate",
    "nucleus_gradient",
    "nucleus_moderate_invert",
    "planar_gradient_Z_moderate",
    "planar_gradient_Z_moderate_invert",
    "nucleus_gradient_strong",
    "membrane_gradient_strong",
    "apical_gradient",
    "apical_gradient_weak",
    "struct_gradient",
    "struct_gradient_weak",
]

# Labels for distance measures
DISTANCE_MEASURE_LABELS = {
    "nucleus": "Distance from nucleus",
    "membrane": "Distance from membrane",
    "z": "Z distance",
    "nearest": "Nearest neighbor distance",
    "pairwise": "Pairwise distance",
    "scaled_nucleus": "Scaled distance from nucleus",
}

# Distance measure title strings
DISTANCE_MEASURE_TITLES = {
    "nucleus": "Nucleus",
    "membrane": "Membrane",
    "z": "Z",
    "nearest": "Nearest",
    "pairwise": "Pairwise",
    "scaled_nucleus": "Scaled Nucleus",
}

# Labels for grid distance files
GRID_DISTANCE_LABELS = {
    "nucleus": "nuc_grid_distances",
    "membrane": "mem_grid_distances",
    "z": "z_grid_distances",
    "scaled_nucleus": "scaled_nuc_grid_distances",
}

# Labels for normalization methods
NORMALIZATION_LABELS = {
    "cell_diameter": "Cell Diameter",
    "intracellular_radius": "Intracellular Radius",
    "max_distance": "Max Distance",
}

# Limits for distance plots
DISTANCE_LIMITS = {
    "pairwise": (0, 25),
    "nucleus": (0, 6),
    "nearest": (0, 5),
    "z": (0, 10),
    "scaled_nucleus": (0.0, 1.0),
    "membrane": (0.0, 3.2),
}

STATS_LABELS = {
    "cell_volumes": "Cell Volume (µm\u00b3)",
    "nuc_volumes": "Nuclear Volume (µm\u00b3)",
    "cell_heights": "Cell Height (µm)",
    "nuc_heights": "Nuclear Height (µm)",
    "cell_diameters": "Cell Diameter (µm)",
    "nuc_diameters": "Nuclear Diameter (µm)",
    "intracellular_radii": "Intracellular Radius (µm)",
}

# Color palette for plotting
COLOR_PALETTE = {
    # used in main plotting functions
    "mean_count_and_size": colormap[0],
    "variable_count": colormap[8],
    "variable_size": colormap[4],
    "shape": colormap[5],
    "SLC25A17": colormap[2],
    "peroxisome": colormap[2],
    "RAB5A": colormap[1],
    "endosome": colormap[1],
    "random": colormap[3],
    "nucleus_gradient_strong": colormap[9],
    "nucleus_gradient": colormap[9],
    "membrane_gradient_strong": colormap[6],
    "membrane_gradient": colormap[6],
    "apical_gradient": colormap[10],
    "apical_gradient_weak": colormap[10],
    "struct_gradient": colormap[7],
    "struct_gradient_weak": colormap[7],
    "SEC61B": colormap[7],
    "ER": colormap[7],
    "ST6GAL1": colormap[7],
    "golgi": colormap[7],
    "membrane": colormap[6],
    "nucleus": colormap[9],
    "interpolated": "black",
    "pairwise": "gray",
    "nearest": "darkgray",
    "z": colormap[10],
    "scaled_nucleus": adjust_color_saturation(colormap[10], 0.5),
    # parameter sweep plotting
    "nucleus_gradient_0pt03": adjust_color_saturation(colormap[9], 0.1),
    "nucleus_gradient_0pt05": adjust_color_saturation(colormap[9], 0.4),
    "nucleus_gradient_0pt07": adjust_color_saturation(colormap[9], 0.7),
    "nucleus_gradient_0pt1": adjust_color_saturation(colormap[9], 0.9),
    "apical_gradient_0pt3": adjust_color_saturation(colormap[10], 0.1),
    "apical_gradient_0pt5": adjust_color_saturation(colormap[10], 0.4),
    "apical_gradient_0pt7": adjust_color_saturation(colormap[10], 0.7),
    "apical_gradient_1": adjust_color_saturation(colormap[10], 0.9),
    "1_random": adjust_color_saturation(colormap[3], 0.4),
    "2_random": adjust_color_saturation(colormap[3], 0.6),
    "4_random": adjust_color_saturation(colormap[3], 0.8),
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


AXIS_TO_INDEX_MAP = {"x": 0, "y": 1, "z": 2}
"""Mapping from axis labels to their corresponding indices."""

PROJECTION_TO_LABEL_MAP = {"x": "YZ", "y": "XZ", "z": "XY"}
"""Mapping from projection axis to label strings."""

PROJECTION_TO_INDEX_MAP = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
"""Mapping from projection axis to index tuples."""
