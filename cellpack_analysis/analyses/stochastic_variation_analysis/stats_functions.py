import numpy as np


def normalize_distances(
    all_distance_dict, normalization=None, mesh_information_dict=None
):
    """
    Normalize the distances in a dictionary of distances.

    Args:
        all_distance_dict (dict): A dictionary containing distances.
        normalization (str, optional): The normalization to use. Defaults to None.
        mesh_information_dict (dict, optional): A dictionary containing mesh information. Defaults to None.

    Returns:
        dict: The normalized distances.
    """
    # get mesh information
    if normalization is not None:

        for mode_distance_dict in all_distance_dict.values():
            for distance_dict in mode_distance_dict.values():
                for cellid, distance in distance_dict.items():
                    mesh_info = mesh_information_dict.get(cellid, "mean")

                    if normalization == "intracellular_radius":
                        normalization_factor = mesh_info["intracellular_radius"]
                    elif normalization == "cell_diameter":
                        normalization_factor = mesh_info["cell_diameter"]
                    elif normalization == "max_distance":
                        normalization_factor = distance.max()

                    distance_dict[cellid] = distance / normalization_factor

    return all_distance_dict


def divide_pdfs(pdf_num, pdf_denom, xvals, epsilon=1e-4):
    # Take logarithms
    valid_num_indices = pdf_num > epsilon
    valid_denom_indices = pdf_denom > epsilon

    valid_inds = valid_num_indices & valid_denom_indices

    pdf_num = pdf_num[valid_inds]
    pdf_denom = pdf_denom[valid_inds]

    pdf_result = np.zeros_like(pdf_num)
    pdf_result = pdf_num / pdf_denom

    xvals = xvals[valid_inds]

    return xvals, pdf_result


def ripley_k(positions, volume, r_max, num_bins=100, norm_factor=1):
    """
    Calculate Ripley's K metric for a given set of positions in a volume V.

    Parameters:
        positions (numpy.ndarray): Array of shape (n, 3) representing n points in 3D space.
        volume (float): Volume of the space.
        r_max (float): Maximum distance to consider.
        num_bins (int): Number of bins for histogram. Default is 100.

    Returns:
        numpy.ndarray: An array of K(r) values for each distance r.
    """
    n = positions.shape[0]
    r_values = np.linspace(0, r_max, num_bins)
    num_points = np.zeros(num_bins)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j]) / norm_factor
            if d < r_max:
                bin_index = int((num_bins - 1) * d / r_max)
                num_points[bin_index] += 1 / n

    ripley_k_values = np.cumsum(num_points) / (n / volume)

    return ripley_k_values, r_values
