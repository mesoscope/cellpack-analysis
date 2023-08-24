from aicscytoparam import cytoparam
from aicsimageio.aics_image import AICSImage
import numpy as np

import pacmap
from sklearn.decomposition import PCA


def get_pilr_for_single_image(file, ch_name):
    img = AICSImage(file)

    gfp_channel = 0
    mem_channel = 1
    nuc_channel = 2

    if ch_name == "SLC25A17":
        gfp_channel = 3
        mem_channel = 1
        nuc_channel = 0

    gfp = img.data[0, gfp_channel, :, :, :].squeeze()
    mem = img.data[0, mem_channel, :, :, :].squeeze()
    nuc = img.data[0, nuc_channel, :, :, :].squeeze()

    gfp[gfp > 0] = 1
    mem[mem > 0] = 1
    nuc[nuc > 0] = 1

    # Use aicsshparam to expand both cell and nuclear shapes in terms of spherical
    # harmonics:
    _, coeffs_centroid = cytoparam.parameterize_image_coordinates(
        seg_mem=mem,
        seg_nuc=nuc,
        lmax=16,  # Degree of the spherical harmonics expansion
        nisos=[32, 32],  # Number of interpolation layers
    )
    coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc = coeffs_centroid

    # Run the cellular mapping to create a parameterized intensity representation
    # for the FP image:
    return cytoparam.cellular_mapping(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=[32, 32],
        images_to_probe=[("gfp", gfp)],
    ).data.squeeze()


def average_over_dimension(value, dim=1):
    # averages the PILR over the dimension dim
    # default is phi
    return value[:, 2:].reshape(65, 128, 64).mean(axis=dim).squeeze()  # R, phi, theta


def get_processed_PILR_from_dict(pilr_dict, ch_ind, average_over_phi=True, mask_nucleus=True):
    # Get the flattened and processed PILR from the dictionary

    if average_over_phi:
        pilr = average_over_dimension(pilr_dict[ch_ind])
    else:
        pilr = pilr_dict[ch_ind]

    pilr = pilr.ravel()

    if mask_nucleus:
        # use second half of the PILR
        pilr = pilr[(len(pilr) // 2):]

    pilr = pilr / np.max(pilr)
    std_pilr = np.std(pilr)

    return pilr, std_pilr


def get_embeddings(combined_PILR, metric, num_samples=305, project_channel=True):
    # calculates PCA or pacmap embeddings from input dict
    # returns a list of embeddings for each channel

    if project_channel:
        X = combined_PILR[num_samples:]
        X_exp = combined_PILR[:num_samples]
    else:
        X = combined_PILR
        X_exp = []

    if metric.lower() == "pacmap":
        embedding = pacmap.PaCMAP(n_components=2, save_tree=True)
    elif metric.lower() == "pca":
        embedding = PCA(n_components=2)
    else:
        raise ValueError("Invalid metric")

    X_transformed = embedding.fit_transform(X)

    if project_channel:
        X_exp_transformed = embedding.transform(X_exp)
        return (X_transformed, X_exp_transformed)

    return X_transformed, None
