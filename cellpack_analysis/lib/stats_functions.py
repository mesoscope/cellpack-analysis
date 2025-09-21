from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import integrate


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return np.abs(np.mean(x) - np.mean(y)) / pooled_std


def normalize_distances(
    occupied_distance_dict: dict,
    mesh_information_dict: dict,
    normalization: str | None = None,
    channel_map: dict | None = None,
    pix_size: float = 0.108,
):
    """
    Normalize the distances in a dictionary of distances.

    Args:
    ----
        all_distance_dict (dict): A dictionary containing distances.
        normalization (str, optional): The normalization to use. Defaults to None.
        mesh_information_dict (dict, optional): A dictionary containing mesh information.
                                                Defaults to None.

    Returns:
    -------
        dict: The normalized distances.
    """
    if channel_map is None:
        channel_map = {}
    for measure, mode_distance_dict in occupied_distance_dict.items():
        if "scaled" in measure:
            continue
        for mode, distance_dict in mode_distance_dict.items():
            mode_mesh_dict = mesh_information_dict.get(channel_map.get(mode, ""), {})
            for cell_id, distance in distance_dict.items():
                mesh_info = mode_mesh_dict.get(
                    cell_id,
                    mode_mesh_dict.get("mean", {"intracellular_radius": 1}),
                )

                if normalization == "intracellular_radius":
                    normalization_factor = mesh_info["intracellular_radius"]
                elif normalization == "cell_diameter":
                    normalization_factor = mesh_info["cell_diameter"]
                elif normalization == "max_distance":
                    normalization_factor = distance.max()
                else:
                    normalization_factor = 1 / pix_size

                distance_dict[cell_id] = distance / normalization_factor

    return occupied_distance_dict


def ripley_k(positions, volume, r_values, norm_factor=1, edge_correction=True):
    """
    Calculate Ripley's K metric for a given set of positions in a volume V.

    Parameters
    ----------
        positions (numpy.ndarray): Array of shape (n, 3) representing n points in 3D space.
        volume (float): Volume of the space.
        r_values (numpy.ndarray): Array of distances at which to calculate K(r).
        norm_factor (float, optional): Normalization factor for distance calculation. Default is 1.
        edge_correction (bool, optional): Whether to apply border correction. Default is True.

    Returns
    -------
        tuple: A tuple containing:
            - numpy.ndarray: An array of K(r) values for each distance r.
            - numpy.ndarray: The input r_values array.
    """
    num_positions = positions.shape[0]

    if num_positions < 2:
        return np.zeros_like(r_values), r_values

    # Calculate all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2) / norm_factor

    # Get upper triangular part to avoid double counting and self-distances
    triu_indices = np.triu_indices(num_positions, k=1)
    distances_upper = distances[triu_indices]

    ripley_k_values = np.zeros_like(r_values, dtype=float)

    # For each radius r, count pairs within distance r
    for i, r in enumerate(r_values):
        pairs_within_r = np.sum(distances_upper <= r)

        # Basic Ripley's K formula: K(r) = V * (number of pairs within distance r) / (n * (n-1) / 2)
        ripley_k_values[i] = volume * pairs_within_r / (num_positions * (num_positions - 1) / 2)

    return ripley_k_values, r_values


def normalize_density(xvals: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Normalize density to integrate to 1.

    Parameters
    ----------
    xvals
        The x-values of the density
    density
        The density values

    Returns
    -------
    :
        Normalized density
    """
    return density / integrate.trapezoid(density, xvals)


def density_ratio(
    xvals: np.ndarray, density1: np.ndarray, density2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the density ratio between two densities.

    Parameters
    ----------
    xvals
        The x-values of the densities
    density1
        The first density
    density2
        The second density

    Returns
    -------
    :
        Tuple containing the density ratio, normalized density1, and normalized density2
    """
    # regularize
    min_value = np.minimum(np.min(density1[density1 > 0]), np.min(density2[density2 > 0]))
    density1 = np.where(density1 <= min_value, min_value, density1)
    density2 = np.where(density2 <= min_value, min_value, density2)

    # normalize densities
    density1 = normalize_density(xvals, density1)
    density2 = normalize_density(xvals, density2)

    # calculate ratio and normalize
    ratio = density1 / density2
    ratio = normalize_density(xvals, ratio)

    return ratio, density1, density2


def cumulative_ratio(
    xvals: np.ndarray, density1: np.ndarray, density2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the cumulative ratio between two density distributions.

    Parameters
    ----------
    xvals
        The x-values of the density distributions
    density1
        The first density distribution
    density2
        The second density distribution

    Returns
    -------
    :
        Tuple containing the cumulative ratio, normalized density1, and normalized density2
    """
    cumulative_ratio = np.zeros(len(xvals))
    # density1 = normalize_density(xvals, density1)
    # density2 = normalize_density(xvals, density2)
    for ct in range(len(xvals)):
        cumulative_ratio[ct] = integrate.trapezoid(
            density1[: ct + 1], xvals[: ct + 1]
        ) / integrate.trapezoid(density2[: ct + 1], xvals[: ct + 1])
    return cumulative_ratio, density1, density2


def get_pdf_ratio(
    xvals: np.ndarray,
    density_numerator: np.ndarray,
    density_denominator: np.ndarray,
    method: Literal["pdf", "cumulative"] = "pdf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ratio of two probability density functions (PDFs) based on the given method.

    Parameters
    ----------
    xvals
        The x-values of the PDFs
    density_numerator
        Density values of the numerator PDF
    density_denominator
        Density values of the denominator PDF
    method
        The method to calculate the ratio

    Returns
    -------
    :
        Tuple containing the ratio and normalized densities based on the specified method
    """
    if method == "pdf":
        return density_ratio(xvals, density_numerator, density_denominator)
    elif method == "cumulative":
        return cumulative_ratio(xvals, density_numerator, density_denominator)
    else:
        raise ValueError(f"Invalid ratio method: {method}")


def create_padded_numpy_array(
    lists: Sequence[list[float] | np.ndarray], padding: float = np.nan
) -> np.ndarray:
    """
    Create a padded array with the specified padding value.

    Parameters
    ----------
    lists
        List of arrays or lists to pad
    padding
        Value to use for padding

    Returns
    -------
    :
        Padded numpy array with all sublists having the same length
    """
    max_length = max([len(sublist) for sublist in lists])
    padded_array = np.zeros((len(lists), max_length))
    for ct, sublist in enumerate(lists):
        if len(sublist) < max_length:
            if isinstance(sublist, list):
                sublist += [padding] * (max_length - len(sublist))
            elif isinstance(sublist, np.ndarray):
                sublist = np.append(sublist, [padding] * (max_length - len(sublist)))
        padded_array[ct] = sublist[:]
    return padded_array
