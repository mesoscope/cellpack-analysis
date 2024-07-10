import matplotlib.pyplot as plt
import numpy as np
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
