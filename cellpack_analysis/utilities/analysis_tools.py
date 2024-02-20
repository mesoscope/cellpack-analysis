import numpy as np


def ripleyK(positions, r_values, volume=None, bootstrap_count=1000):
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
        bootstrap_indices = np.random.choice(n_points, n_points, replace=True)
        for rc in range(n_r_values):
            r = r_values[rc]
            k = 0
            for j in range(n_points):
                distances = np.linalg.norm(positions[bootstrap_indices] - positions[bootstrap_indices[j]], axis=1)
                k += (distances <= r).sum()
            all_k_values[bc, rc] += k / n_points / average_point_density

    mean_k_values = np.mean(all_k_values, axis=0)
    ci_k_values = np.percentile(all_k_values, [2.5, 97.5], axis=0)

    return mean_k_values, ci_k_values
