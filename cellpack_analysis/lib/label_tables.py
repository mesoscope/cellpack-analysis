import colorsys

import seaborn as sns


def adjust_color_saturation(hex_color: str, saturation: float) -> str:
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

# Labels for cellPACK simulated packing modes
MODE_LABELS = {
    "SLC25A17": "Peroxisome",
    "RAB5A": "Endosome",
    "SEC61B": "ER",
    "ST6GAL1": "Golgi",
    "TOMM20": "Mitochondria",
    "baseline": "Baseline",
    "variable_size": "Size",
    "variable_count": "Count",
    "variable_count_and_size": "Count and size",
    "random": "Unbiased",
    "shape": "Shape",
    "interpolated": "Mixed",
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
    # lambda ladder rungs
    "nucleus_gradient_0pt01": "Nucleus λ=0.01",
    "nucleus_gradient_0pt03": "Nucleus λ=0.03",
    "nucleus_gradient_0pt1": "Nucleus λ=0.1",
    "nucleus_gradient_0pt3": "Nucleus λ=0.3",
    "nucleus_gradient_1": "Nucleus λ=1",
    "nucleus_gradient_10": "Nucleus λ=10",
    "apical_gradient_0pt01": "Apical λ=0.01",
    "apical_gradient_0pt03": "Apical λ=0.03",
    "apical_gradient_0pt1": "Apical λ=0.1",
    "apical_gradient_0pt3": "Apical λ=0.3",
    "apical_gradient_0pt7": "Apical λ=0.7",
    "apical_gradient_1": "Apical λ=1",
    "apical_gradient_10": "Apical λ=10",
    "membrane_gradient_0pt01": "Membrane λ=0.01",
    "membrane_gradient_0pt03": "Membrane λ=0.03",
    "membrane_gradient_0pt1": "Membrane λ=0.1",
    "membrane_gradient_0pt3": "Membrane λ=0.3",
    "membrane_gradient_1": "Membrane λ=1",
    "membrane_gradient_10": "Membrane λ=10",
    "1_random": "Spacing 100nm",
    "2_random": "Spacing 200nm",
    "4_random": "Spacing 400nm",
}

# Canonical decay length per gradient family, for marking the reported rung on
# lambda-ladder figures.
CANONICAL_DECAY_LENGTHS = {
    "nucleus_gradient": 0.1,
    "membrane_gradient": 0.01,
    "apical_gradient": 0.7,
}

# Distance measures that depend on cell shape
SHAPE_DEPENDENT_MEASURES = frozenset({"nucleus", "scaled_nucleus", "membrane", "z", "scaled_z"})

# Labels for distance measures
DISTANCE_MEASURE_LABELS = {
    "nucleus": "Distance from nucleus (µm)",
    "membrane": "Distance from membrane (µm)",
    "z": "Z distance (µm)",
    "nearest": "Nearest neighbor distance (µm)",
    "pairwise": "Pairwise distance (µm)",
    "scaled_nucleus": "Scaled distance from nucleus",
    "scaled_z": "Scaled Z distance",
}

# Distance measure title strings
DISTANCE_MEASURE_TITLES = {
    "nucleus": "Nucleus",
    "membrane": "Membrane",
    "z": "Z",
    "nearest": "Nearest",
    "pairwise": "Pairwise",
    "scaled_nucleus": "Scaled Nucleus",
    "scaled_z": "Scaled Z",
}

# Labels for grid distance files
GRID_DISTANCE_LABELS = {
    "nucleus": "nuc_grid_distances",
    "membrane": "mem_grid_distances",
    "z": "z_grid_distances",
    "scaled_nucleus": "scaled_nuc_grid_distances",
    "scaled_z": "scaled_z_grid_distances",
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
    "mem_volumes": "Cell Volume (µm\u00b3)",
    "nuc_volumes": "Nuclear Volume (µm\u00b3)",
    "mem_heights": "Cell Height (µm)",
    "nuc_heights": "Nuclear Height (µm)",
    "mem_diameters": "Cell Diameter (µm)",
    "nuc_diameters": "Nuclear Diameter (µm)",
    "intracellular_radii": "Intracellular Radius (µm)",
    "mem_sphericities": "Cell Sphericity",
    "nuc_sphericities": "Nuclear Sphericity",
}

# Color palette for plotting
COLOR_PALETTE = {
    # used in main plotting functions
    "baseline": colormap[0],
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
    "interpolated": "#000000",
    "pairwise": "#808080",
    "nearest": "#A9A9A9",
    "z": colormap[10],
    "scaled_nucleus": adjust_color_saturation(colormap[10], 0.5),
    # parameter sweep plotting: round-1 rungs kept for reproducibility
    "nucleus_gradient_0pt05": adjust_color_saturation(colormap[9], 0.4),
    "nucleus_gradient_0pt07": adjust_color_saturation(colormap[9], 0.7),
    "apical_gradient_0pt5": adjust_color_saturation(colormap[10], 0.55),
    # lambda ladder: saturation increases with lambda within each family
    "nucleus_gradient_0pt01": adjust_color_saturation(colormap[9], 0.15),
    "nucleus_gradient_0pt03": adjust_color_saturation(colormap[9], 0.3),
    "nucleus_gradient_0pt1": adjust_color_saturation(colormap[9], 0.45),
    "nucleus_gradient_0pt3": adjust_color_saturation(colormap[9], 0.6),
    "nucleus_gradient_1": adjust_color_saturation(colormap[9], 0.8),
    "nucleus_gradient_10": adjust_color_saturation(colormap[9], 1.0),
    "apical_gradient_0pt01": adjust_color_saturation(colormap[10], 0.15),
    "apical_gradient_0pt03": adjust_color_saturation(colormap[10], 0.28),
    "apical_gradient_0pt1": adjust_color_saturation(colormap[10], 0.4),
    "apical_gradient_0pt3": adjust_color_saturation(colormap[10], 0.55),
    "apical_gradient_0pt7": adjust_color_saturation(colormap[10], 0.7),
    "apical_gradient_1": adjust_color_saturation(colormap[10], 0.85),
    "apical_gradient_10": adjust_color_saturation(colormap[10], 1.0),
    "membrane_gradient_0pt01": adjust_color_saturation(colormap[6], 0.15),
    "membrane_gradient_0pt03": adjust_color_saturation(colormap[6], 0.3),
    "membrane_gradient_0pt1": adjust_color_saturation(colormap[6], 0.45),
    "membrane_gradient_0pt3": adjust_color_saturation(colormap[6], 0.6),
    "membrane_gradient_1": adjust_color_saturation(colormap[6], 0.8),
    "membrane_gradient_10": adjust_color_saturation(colormap[6], 1.0),
    "1_random": adjust_color_saturation(colormap[3], 0.4),
    "2_random": adjust_color_saturation(colormap[3], 0.6),
    "4_random": adjust_color_saturation(colormap[3], 0.8),
}


AXIS_TO_INDEX_MAP = {"x": 0, "y": 1, "z": 2}
"""Mapping from axis labels to their corresponding indices."""

PROJECTION_TO_LABEL_MAP = {"x": ("Y", "Z"), "y": ("X", "Z"), "z": ("X", "Y")}
"""Mapping from projection axis to label strings."""

PROJECTION_TO_INDEX_MAP = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
"""Mapping from projection axis to index tuples."""
