from pathlib import Path
from aicscytoparam import cytoparam
from aicsimageio.aics_image import AICSImage
import numpy as np
import pandas as pd
from aicsshparam import shtools

import vtk
import pacmap
import umap
from sklearn.decomposition import PCA


def get_pilr_for_single_image(file, ch_name, raw_image_channel="SLC25A17"):
    img = AICSImage(file)

    gfp_channel = 2
    mem_channel = 0
    nuc_channel = 1

    if ch_name == raw_image_channel:
        gfp_channel = 4
        mem_channel = 2
        nuc_channel = 0

    img_data, _ = shtools.align_image_2d(img.data[0], alignment_channel=mem_channel)

    gfp = img_data[gfp_channel, :, :, :].squeeze()
    mem = img_data[mem_channel, :, :, :].squeeze()
    nuc = img_data[nuc_channel, :, :, :].squeeze()

    # gfp[gfp > 0] = 1
    # mem[mem > 0] = 1
    # nuc[nuc > 0] = 1

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


def get_processed_PILR_from_dict(
    pilr_dict, ch_ind, average_over_phi=True, mask_nucleus=True
):
    # Get the flattened and processed PILR from the dictionary

    if average_over_phi:
        pilr = average_over_dimension(pilr_dict[ch_ind])
    else:
        pilr = pilr_dict[ch_ind]

    pilr = pilr.ravel()

    if mask_nucleus:
        # use second half of the PILR
        pilr = pilr[(len(pilr) // 2) :]

    pilr = pilr / np.max(pilr)
    std_pilr = np.std(pilr)

    return pilr, std_pilr


def get_embeddings(individual_PILR_dict, metric, channels_for_embedding=None, **kwargs):
    # calculates PCA or pacmap embeddings from input dict
    # returns a list of embeddings for each channel
    # input_PILR_dict: dictionary of PILR values for each channel

    if channels_for_embedding is None:
        channels_for_embedding = list(individual_PILR_dict.keys())

    X_fit = np.array([])
    X_proj = np.array([])
    index_dict = {}
    for ch, values in individual_PILR_dict.items():
        index_dict[ch] = {}
        if ch in channels_for_embedding:
            index_dict[ch]["embedding"] = "fit"
            index_dict[ch]["start_ind"] = len(X_fit)
            if len(X_fit) == 0:
                X_fit = values
            else:
                X_fit = np.vstack((X_fit, values))
            index_dict[ch]["end_ind"] = len(X_fit)
        else:
            index_dict[ch]["embedding"] = "proj"
            index_dict[ch]["start_ind"] = len(X_proj)
            if len(X_proj) == 0:
                X_proj = values
            else:
                X_proj = np.vstack((X_proj, values))
            index_dict[ch]["end_ind"] = len(X_proj)

    if metric.lower() == "pacmap":
        embedding = pacmap.PaCMAP(
            n_components=kwargs.get("n_components", 2), apply_pca=False, save_tree=True
        )
    elif metric.lower() == "pca":
        embedding = PCA(n_components=kwargs.get("n_components", 2))
    elif metric.lower() == "pacmap_pca":
        embedding = pacmap.PaCMAP(
            n_components=kwargs.get("n_components", 2), apply_pca=False, save_tree=True
        )
        embedding_pca = PCA(n_components=kwargs.get("n_components_pca", 2))
        if len(X_fit):
            X_fit = embedding_pca.fit_transform(X_fit)
        if len(X_proj):
            X_proj = embedding_pca.transform(X_proj)
    elif metric.lower() == "umap":
        embedding = umap.UMAP(n_components=kwargs.get("n_components", 2))
    else:
        raise ValueError("Invalid metric")

    X_fit_transformed = []
    if len(X_fit):
        X_fit_transformed = embedding.fit_transform(X_fit)

    X_proj_transformed = []
    if len(X_proj):
        X_proj_transformed = embedding.transform(X_proj)

    embedding_dict = {}
    for ch in individual_PILR_dict.keys():
        embedding_dict[ch] = {}
        if ch in channels_for_embedding and len(X_fit):
            embedding_dict[ch]["embedding"] = index_dict[ch]["embedding"]
            embedding_dict[ch]["values"] = X_fit_transformed[
                index_dict[ch]["start_ind"] : index_dict[ch]["end_ind"]
            ]
        elif len(X_proj):
            embedding_dict[ch]["embedding"] = index_dict[ch]["embedding"]
            embedding_dict[ch]["values"] = X_proj_transformed[
                index_dict[ch]["start_ind"] : index_dict[ch]["end_ind"]
            ]

    return embedding_dict, embedding


def get_average_shape_mesh_objects(mesh_folder=Path("../../data/average_shape_meshes")):
    # Get the mesh objects for the average shape
    mesh_dict = {}
    for shape in ["NUC", "MEM"]:
        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(mesh_folder / f"mean_{shape}.obj"))
        reader.Update()
        mesh_dict[shape] = reader.GetOutput()
    return mesh_dict


def get_domain(mesh_dict):
    # Get the domain for the average shape
    domain, _ = cytoparam.voxelize_meshes([mesh_dict["MEM"], mesh_dict["NUC"]])
    return domain


def get_parametrized_coords_for_avg_shape(
    domain,
):
    # Get the parametrized coordinates for the average shape
    coords_param, _ = cytoparam.parameterize_image_coordinates(
        seg_mem=(domain > 0).astype(np.uint8),
        seg_nuc=(domain > 1).astype(np.uint8),
        lmax=16,
        nisos=[32, 32],  # nisos = 32
    )
    return coords_param


def morph_PILRs_into_average_shape(
    pilr_list,
    domain=None,
    coords_param=None,
    mesh_dict=None,
):
    # Morph the PILRs into the average shape
    if mesh_dict is None:
        mesh_dict = get_average_shape_mesh_objects()
    if domain is None:
        domain = get_domain(mesh_dict)
    if coords_param is None:
        coords_param = get_parametrized_coords_for_avg_shape(domain)

    morphed = []
    for pilr in pilr_list:
        morphed.append(
            cytoparam.morph_representation_on_shape(
                img=domain,
                param_img_coords=coords_param,
                representation=pilr,
            )
        )

    return morphed


def get_cell_id_list(df_cellID, structure_id):
    # get cell id list for given structure
    all_cellid_as_strings = df_cellID.loc[structure_id, "CellIds"].split(",")

    cellid_list = []
    for cellid in all_cellid_as_strings:
        cellid_list.append(int(cellid.replace("[", "").replace("]", "")))

    return cellid_list
