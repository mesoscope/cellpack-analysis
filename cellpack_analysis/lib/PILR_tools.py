from aicscytoparam import cytoparam
from aicsimageio.aics_image import AICSImage
import numpy as np
import pandas as pd
from aicsshparam import shtools
import matplotlib.pyplot as plt
import seaborn as sns

import pacmap
import umap
from sklearn.decomposition import PCA

from cellpack_analysis.lib.mesh_tools import (
    calculate_scaled_distances_from_mesh,
    get_average_shape_mesh_objects,
)


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


def get_domain(mesh_dict):
    # Get the domain for the average shape
    domain, _ = cytoparam.voxelize_meshes([mesh_dict["mem"], mesh_dict["nuc"]])
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
    mesh_dict,
    domain=None,
    coords_param=None,
):
    # Morph the PILRs into the average shape
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


def get_cellid_ch_seed_from_fname(fname, config_name="analyze", suffix="_pilr"):
    fname = fname.split(suffix)[0]
    seed_name = fname.split("_")[-1].split(".")[0]
    cellid = fname.split("_")[-3]
    ch = fname.split(f"{config_name}_")[-1].split(f"_{cellid}")[0]
    return cellid, ch, seed_name


def get_correlations_between_average_PILRs(
    average_pilr_dict, channel_name_dict, save_dir
):
    # Get correlations between average PILRs
    df = pd.DataFrame(
        index=channel_name_dict.values(),
        columns=channel_name_dict.values(),
        dtype=float,
    )

    for ch_ind, ch_name in channel_name_dict.items():
        if ch_ind not in average_pilr_dict:
            continue
        pilr1 = average_pilr_dict[ch_ind].ravel()
        pilr1 = pilr1 / np.max(pilr1)
        for ch_ind2, ch_name2 in channel_name_dict.items():
            if ch_ind2 not in average_pilr_dict:
                continue
            pilr2 = average_pilr_dict[ch_ind2].ravel()
            pilr2 = pilr2 / np.max(pilr2)
            df.loc[ch_name, ch_name2] = np.corrcoef(pilr1, pilr2)[0, 1]

    df.to_csv(save_dir / "average_PILR_correlations.csv")

    mask = np.zeros_like(df, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    if save_dir is not None:

        plt.close("all")

        sns.heatmap(
            df,
            annot=True,
            cmap="cool",
            mask=mask,
        )
        plt.savefig(
            save_dir / "PILR_correlation_bias.png",
            dpi=300,
            bbox_inches="tight",
        )


def get_individual_PILRs_from_images(pilr_img_folder, channel_name_dict):
    # Get the PILRs for each channel from the images
    pass


def cartesian_to_sph(xyz, center=None):
    """
    Converts cartesian to spherical coordinates
    """
    if center is None:
        center = np.zeros(3)
    xyz = xyz - center
    sph_pts = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    sph_pts[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    sph_pts[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    sph_pts[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])

    return sph_pts


def calculate_simplified_PILR(positions, nuc_mesh, mem_mesh, scale=True):
    """
    Calculate the simplified PILR (Parameterized Intracellular Location Representation)
    value based on the given positions, inner mesh, and outer mesh.

    Parameters
    ----------
    positions : list
        List of positions.
    nuc_mesh : Mesh
        Inner mesh object.
    mem_mesh : Mesh
        Outer mesh object.
    scale : bool, optional
        Flag indicating whether to scale the PILR value, by default True.

    Returns
    -------
    spilr : float
        Simplified PILR value.
    """
    # Convert positions to spherical coordinates
    sph_positions = cartesian_to_sph(positions)

    (
        scaled_radial_position,
        distance_between_surfaces,
        inner_surface_distance,
    ) = calculate_scaled_distances_from_mesh(sph_positions, nuc_mesh, mem_mesh)

    return spilr


import numpy as np
import matplotlib.pyplot as plt
from aicscytoparam import cytoparam
from skimage import measure as skmeasure


def get_mean_shape_as_image(
    outer_mesh, inner_mesh, lmax=16, nisos_outer=32, nisos_inner=32
):

    domain, origin = cytoparam.voxelize_meshes([outer_mesh, inner_mesh])
    coords_param, _ = cytoparam.parameterize_image_coordinates(
        seg_mem=(domain > 0).astype(np.uint8),
        seg_nuc=(domain > 1).astype(np.uint8),
        lmax=lmax,
        nisos=[nisos_inner, nisos_outer],
    )

    domain_nuc = (255 * (domain > 1)).astype(np.uint8)
    domain_mem = (255 * (domain > 0)).astype(np.uint8)

    return domain, domain_nuc, domain_mem, coords_param


class Projector:
    # Expects a 3 channels 3D image with following channels:
    # 0 - nucleus (binary mask)
    # 1 - membrane (binary mask)
    # 2 - structure (np.float32)
    verbose = True
    gfp_pcts = [10, 90]
    CMAPS = {"nuc": "gray", "mem": "gray", "gfp": "inferno"}

    def __init__(self, data, force_fit=False, box_size=300, mask_on=False):
        self.data = {
            "nuc": data[0].copy().astype(bool),
            "mem": data[1].copy().astype(bool),
            "gfp": data[2].copy().astype(np.float32),
        }
        self.local_pct = False
        self.box_size = box_size
        self.force_fit = force_fit
        self.bbox = self.get_bbox_of_chanel("mem")
        self.tight_crop()
        self.pad_data()
        self.gfp_vmin = None
        self.gfp_vmax = None
        self.panel_size = 2
        if mask_on:
            self.mask_gfp_channel()

    def set_panel_size(self, size):
        self.panel_size = size

    def mask_gfp_channel(self):
        mem = self.data["mem"]
        self.data["gfp"][mem == 0] = 0

    def set_verbose_on(self):
        self.verbose = True

    def set_projection_mode(self, ax, mode):
        # mode is a dict with keys: nuc, mem and gfp
        self.proj_ax = ax
        self.proj_mode = mode

    def view(self, alias, ax, chopy=None):
        proj = self.projs[alias]
        args = {}
        if chopy is not None:
            proj = proj[chopy:-chopy]
        if alias == "gfp":
            args = {"vmin": self.gfp_vmin, "vmax": self.gfp_vmax}
        return ax.imshow(proj, cmap=self.CMAPS[alias], origin="lower", **args)

    def display(self, save):
        for alias, proj in self.projs.items():
            fig, ax = plt.subplots(1, 1, figsize=(self.panel_size, self.panel_size))
            _ = self.view(alias=alias, ax=ax)
            ax.axis("off")
            if save is not None:
                fig.savefig(f"{save}_{alias}.png", dpi=150)
            plt.show()

    def get_projections(self):
        self.projs = {}
        ax = ["z", "y", "x"].index(self.proj_ax)
        for alias, img in self.data.items():
            if self.proj_mode[alias] == "max":
                p = img.max(axis=ax)
            if self.proj_mode[alias] == "mean":
                p = img.mean(axis=ax)
            if self.proj_mode[alias] == "top_nuc":
                zc, yc, xc = [int(np.max(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "center_nuc":
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "center_mem":
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["mem"])]
                if self.proj_ax == "z":
                    p = img[zc]
                if self.proj_ax == "y":
                    p = img[:, yc]
                if self.proj_ax == "x":
                    p = img[:, :, xc]
            if self.proj_mode[alias] == "max_buffer_center_nuc":
                buf = 3
                zc, yc, xc = [int(np.mean(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc - buf : zc + buf].max(axis=ax)
                if self.proj_ax == "y":
                    p = img[:, yc - buf : yc + buf].max(axis=ax)
                if self.proj_ax == "x":
                    p = img[:, :, xc - buf : xc + buf].max(axis=ax)
            if self.proj_mode[alias] == "max_buffer_top_nuc":
                buf = 3
                zc, yc, xc = [int(np.max(u)) for u in np.where(self.data["nuc"])]
                if self.proj_ax == "z":
                    p = img[zc - buf : zc + buf].max(axis=ax)
                if self.proj_ax == "y":
                    p = img[:, yc - buf : yc + buf].max(axis=ax)
                if self.proj_ax == "x":
                    p = img[:, :, xc - buf : xc + buf].max(axis=ax)
            if self.verbose:
                print(f"Image shape: {img.shape}, slices used: ({zc},{yc},{xc})")
            self.projs[alias] = p

    def set_gfp_percentiles(self, pcts, local=False):
        self.gfp_pcts = pcts
        self.local_pct = local

    def set_vmin_vmax_gfp_values(self, vmin, vmax):
        self.gfp_vmin = vmin
        self.gfp_vmax = vmax

    def set_gfp_colormap(self, cmap):
        self.CMAPS["gfp"] = cmap

    def calculate_gfp_percentiles(self):
        if self.gfp_vmin is not None:
            if self.verbose:
                print("vmin/vmax values already set...")
            return
        data = self.data
        if self.local_pct:
            data = self.projs
        data = data["gfp"][data["mem"] > 0]
        self.gfp_vmin = np.percentile(data, self.gfp_pcts[0])
        self.gfp_vmax = np.percentile(data, self.gfp_pcts[1])
        if self.verbose:
            print(f"GFP min/max: {self.gfp_vmin:.3f} / {self.gfp_vmax:.3f}")

    def compute(self, scale_bar=None):
        self.get_projections()
        self.calculate_gfp_percentiles()
        if scale_bar is not None:
            self.stamp_scale_bar(**scale_bar)

    def project(self, save=None, scale_bar=None):
        self.compute(scale_bar=scale_bar)
        self.display(save)

    def project_on(self, alias, ax, chopy=None, scale_bar=None):
        self.compute(scale_bar=scale_bar)
        return self.view(alias=alias, ax=ax, chopy=chopy)

    def get_bbox_of_chanel(self, channel):
        img = self.data[channel]
        z, y, x = np.where(img)
        sz, sy, sx = img.shape
        zmin, zmax = np.max([0, np.min(z)]), np.min([sz - 1, np.max(z)])
        ymin, ymax = np.max([0, np.min(y)]), np.min([sy - 1, np.max(y)])
        xmin, xmax = np.max([0, np.min(x)]), np.min([sx - 1, np.max(x)])
        return xmin, xmax, ymin, ymax, zmin, zmax

    def tight_crop(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bbox
        for alias, img in self.data.items():
            img = img[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]
            if self.force_fit:
                zf = np.min([zmax - zmin, self.box_size])
                yf = np.min([ymax - ymin, self.box_size])
                xf = np.min([xmax - xmin, self.box_size])
                img = img[:zf, :yf, :xf]
            self.data[alias] = img

    def pad_data(self):
        mingfp = self.data["gfp"][self.data["mem"] > 0].min()
        for alias, img in self.data.items():
            shape = img.shape
            pad = [int(0.5 * (self.box_size - s)) for s in shape]
            pad = [(p, int(self.box_size - s - p)) for (s, p) in zip(shape, pad)]
            if np.min([np.min([i, j]) for i, j in pad]) < 0:
                raise ValueError(
                    f"Box of size {self.box_size} invalid for image of size: {shape}."
                )
            self.data[alias] = np.pad(
                img,
                pad,
                mode="constant",
                constant_values=0 if alias != "gfp" else mingfp,
            )

    def stamp_scale_bar(self, pixel_size=0.108, length=5):
        xc = int(0.5 * self.box_size)
        n = int(length / pixel_size)
        self.projs["nuc"][20:30, xc : xc + n] = True

    def get_proj_contours(self):
        cts = {}
        for alias in ["nuc", "mem"]:
            im = self.projs[alias]
            cts[alias] = skmeasure.find_contours(im, 0.5)
        return cts

    def show(self):
        proj = self.projs["gfp"].copy()
        contour = self.get_proj_contours()
        fig, ax = plt.subplots(1, 1, figsize=(self.panel_size, self.panel_size))
        ax.imshow(
            proj, cmap="inferno", origin="lower", vmin=self.gfp_vmin, vmax=self.gfp_vmax
        )
        for alias_cont, alias_color in zip(["nuc", "mem"], ["cyan", "magenta"]):
            [
                ax.plot(c[:, 1], c[:, 0], lw=0.5, color=alias_color)
                for c in contour[alias_cont]
            ]
            [
                ax.plot(c[:, 1], c[:, 0], lw=0.5, color=alias_color)
                for c in contour[alias_cont]
            ]
        plt.show()

    @staticmethod
    def get_shared_morphed_max_based_on_pct_for_zy_views(
        instances, pct, mode, func=np.max, include_vmin_as_zero=True, nonzeros_only=True
    ):
        vmax = {"z": [], "y": []}
        for img in instances:
            for ax in ["z", "y"]:
                proj = Projector(img, force_fit=True)
                proj.set_projection_mode(ax=ax, mode=mode)
                proj.compute()
                values = proj.projs["gfp"].flatten()
                if nonzeros_only:
                    values = values[values > 0.0]
                v = 0
                if len(values) > 0:
                    v = np.percentile(values, pct)
                vmax[ax].append(v)
        if include_vmin_as_zero:
            return dict([(ax, (0, func(vals))) for ax, vals in vmax.items()])
        return dict([(ax, func(vals)) for ax, vals in vmax.items()])

    @staticmethod
    def get_shared_gfp_range_for_zy_views_old(instances, pcts, mode):
        minmax = {"z": [], "y": []}
        for k, cellinfos in instances.items():
            for cellinfo in cellinfos:
                img = cellinfo["img"]
                for ax in ["z", "y"]:
                    proj = Projector(img, force_fit=True)
                    proj.set_projection_mode(ax=ax, mode=mode)
                    proj.compute()
                    values = proj.projs["gfp"].flatten()  # [proj.projs["gfp"]>0]
                    if len(values):
                        minmax[ax].append(np.percentile(values, pcts))
        print(minmax)
        return dict([(ax, (np.min(vals), np.max(vals))) for ax, vals in minmax.items()])
