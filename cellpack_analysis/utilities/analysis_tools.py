import numpy as np


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


def ripleyK(positions, r_values, volume=None, bootstrap_count=100):
    """
    Calculate the Ripley's K function for a set of positions.

    Args:
        positions (np.ndarray): The positions of the points.
        r_values (np.ndarray): The r values at which to calculate the K function.
        volume (float, optional): The volume of the space. If None, the volume is calculated as the volume of the smallest axis-aligned bounding box containing all the points. Defaults to None.
        bootstrap_count (int, optional): The number of bootstrap samples to use for the confidence interval. Defaults to 1000.

    Returns:
        np.ndarray: The values of the K function at the specified r values.
    """
    n_points = positions.shape[0]
    n_r_values = r_values.shape[0]
    all_k_values = np.zeros((bootstrap_count, n_r_values))

    if volume is None:
        volume = np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
    average_point_density = n_points / volume

    for bc in range(bootstrap_count):
        if bootstrap_count > 1:
            bootstrap_indices = np.random.choice(n_points, n_points, replace=True)
        else:
            bootstrap_indices = np.arange(n_points)
        for rc in range(n_r_values):
            r = r_values[rc]
            k = 0
            for j in range(n_points):
                use_indices = bootstrap_indices[j:]
                distances = np.linalg.norm(
                    positions[use_indices] - positions[bootstrap_indices[j]],
                    axis=1,
                )
                k += (distances <= r).sum()             
            all_k_values[bc, rc] += k / n_points / average_point_density

    mean_k_values = np.mean(all_k_values, axis=0)
    ci_k_values = np.percentile(all_k_values, [2.5, 97.5], axis=0)

    return mean_k_values, ci_k_values


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

    for _, mode_distance_dict in all_distance_dict.items():
        for _, distance_dict in mode_distance_dict.items():
            for seed, distance in distance_dict.items():
                mesh_info = mesh_information_dict[seed]

                if normalization is None:
                    normalization_factor = 1
                elif normalization == "intracellular_radius":
                    normalization_factor = mesh_info["intracellular_radius"]
                elif normalization == "cell_diameter":
                    normalization_factor = mesh_info["cell_diameter"]
                elif normalization == "max_distance":
                    normalization_factor = distance.max()

                distance_dict[seed] = distance / normalization_factor

    return all_distance_dict
