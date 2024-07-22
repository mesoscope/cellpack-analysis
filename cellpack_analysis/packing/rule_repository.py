GRADIENTS = {
    "nucleus_gradient": {
        "description": "gradient based on distance from the surface of the nucleus mesh",
        "mode": "surface",
        "mode_settings": {"object": "nucleus", "scale_to_next_surface": False},
        "weight_mode": "exponential",
        "weight_mode_settings": {"decay_length": 0.1},
    },
    "membrane_gradient": {
        "description": "gradient based on distance from the surface of the membrane mesh",
        "mode": "surface",
        "mode_settings": {"object": "membrane", "scale_to_next_surface": False},
        "weight_mode": "exponential",
        "weight_mode_settings": {"decay_length": 0.1},
    },
    "apical_gradient": {
        "description": "gradient based on distance from a plane",
        "mode": "vector",
        "mode_settings": {"direction": [0, 0, 1], "center": "max"},
        "weight_mode": "exponential",
        "weight_mode_settings": {"decay_length": 0.1},
    },
}
