from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def generate_composite_thumbnail(
    tiff_path: Path | str,
    output_png_path: Path | str,
    image_type: str = "raw",
    channel_colors: dict | None = None,
    scale_bar_microns: float = 5,
    pixel_size: float = 0.108,
    scale_bar_thickness: int = 3,
    margin: int = 10,
    threshold: float = 0.7,
    dpi: int = 300,
    figsize: tuple = (4, 4),
):
    """
    Generate a composite PNG thumbnail from a multi-channel TIFF image.

    The image is flipped vertically to match microscopy orientation.
    The scale bar is drawn after flipping so it remains on the bottom right.

    Parameters
    ----------
    tiff_path
        Path to the input multi-channel TIFF image.
    output_png_path
        Path to save the output composite PNG thumbnail.
    image_type
        Type of the image, either "raw" or "seg". Determines channel order.
    channel_colors
        Dictionary mapping channel names to RGB color tuples (values between 0 and 1).
    scale_bar_microns
        Length of the scale bar in microns.
    pixel_size
        Size of a pixel in microns.
    scale_bar_thickness
        Thickness of the scale bar in pixels.
    margin
        Margin around the image in pixels to accommodate the scale bar.
    threshold
        Intensity threshold (0-1) to determine which pixels belong to each channel.
    dpi
        Dots per inch for the output PNG image.
    figsize
        Size of the output figure in inches (width, height).
    """
    if channel_colors is None:
        channel_colors = {
            "nucleus": (47, 82, 82),  # Dark Cyan
            "membrane": (111, 67, 108),  # Dark Magenta
            "structure": (44, 160, 44),  # Dark Green
        }
    for channel, channel_color in channel_colors.items():
        if np.any(np.array(channel_color) > 1):
            channel_colors[channel] = tuple(c / 255 for c in channel_color)

    img = io.imread(str(tiff_path))
    if image_type == "raw":
        nuc_channel_index = 0
        mem_channel_index = 1
        struct_channel_index = 2
        img = np.transpose(img, (0, 3, 1, 2))  # ZCYX
    elif image_type == "seg":
        struct_channel_index = 3
        nuc_channel_index = 0
        mem_channel_index = 2
    elif image_type == "cellpack":
        struct_channel_index = 2
        nuc_channel_index = 1
        mem_channel_index = 0
        img = np.transpose(img, (0, 3, 1, 2))  # ZCYX
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

    img_struct = img[:, struct_channel_index]
    img_nuc = img[:, nuc_channel_index]
    img_mem = img[:, mem_channel_index]
    img_struct_max = np.max(img_struct, axis=0)
    img_nuc_max = np.max(img_nuc, axis=0)
    img_mem_max = np.max(img_mem, axis=0)

    # Normalize intensity values to 0-1 range
    img_struct_norm = (
        img_struct_max / np.max(img_struct_max) if np.max(img_struct_max) > 0 else img_struct_max
    )
    img_nuc_norm = img_nuc_max / np.max(img_nuc_max) if np.max(img_nuc_max) > 0 else img_nuc_max
    img_mem_norm = img_mem_max / np.max(img_mem_max) if np.max(img_mem_max) > 0 else img_mem_max

    # Create binary masks for each channel
    struct_mask = img_struct_norm > threshold
    nuc_mask = img_nuc_norm > threshold
    mem_mask = img_mem_norm > threshold

    height, width = img_struct_max.shape
    composite = np.zeros((height, width, 3))

    # Layer 1 (bottom): Membrane
    composite[mem_mask, 0] = img_mem_norm[mem_mask] * channel_colors["membrane"][0]
    composite[mem_mask, 1] = img_mem_norm[mem_mask] * channel_colors["membrane"][1]
    composite[mem_mask, 2] = img_mem_norm[mem_mask] * channel_colors["membrane"][2]

    # Layer 2 (middle): Nucleus
    composite[nuc_mask, 0] = img_nuc_norm[nuc_mask] * channel_colors["nucleus"][0]
    composite[nuc_mask, 1] = img_nuc_norm[nuc_mask] * channel_colors["nucleus"][1]
    composite[nuc_mask, 2] = img_nuc_norm[nuc_mask] * channel_colors["nucleus"][2]

    # Layer 3 (top): Structure
    composite[struct_mask, 0] = img_struct_norm[struct_mask] * channel_colors["structure"][0]
    composite[struct_mask, 1] = img_struct_norm[struct_mask] * channel_colors["structure"][1]
    composite[struct_mask, 2] = img_struct_norm[struct_mask] * channel_colors["structure"][2]

    # Flip vertically to match microscopy orientation
    composite = composite[::-1, :, :]

    # Add scale bar (bottom-right, drawn after flipping)
    scale_bar_length_pixels = int(scale_bar_microns / pixel_size)
    x_start = width - margin - scale_bar_length_pixels
    y_start = height - margin - scale_bar_thickness
    x_end = width - margin
    y_end = height - margin
    composite[y_start:y_end, x_start:x_end, :] = 1.0  # White scale bar

    # Save composite as PNG
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(composite)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
