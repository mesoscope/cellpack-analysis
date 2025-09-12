import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from aicsimageio.aics_image import AICSImage

from cellpack_analysis.lib.archive.PILR_tools import add_contour_to_axis


def plot_PILR(
    avg_gfp,
    ch_name=None,
    save_dir=None,
    label=None,
    suffix="",
    aspect=20,
    ax=None,
    **kwargs,
):
    # Save the PILR as an image
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()

    if ch_name is None:
        ch_name = ""

    if label is None:
        label = f"avg_PILR_{ch_name}"

    plot_values = avg_gfp / np.max(avg_gfp)
    # plot_values = avg_gfp

    min_pct = kwargs.get("min_pct", 0)
    max_pct = kwargs.get("max_pct", 90)

    vmin = kwargs.get("vmin", np.percentile(plot_values, min_pct))
    vmax = kwargs.get("vmax", np.percentile(plot_values, max_pct))

    ax.imshow(
        plot_values,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    ax.axis("off")
    fig.set_facecolor("black")
    ax.set_aspect(aspect)
    if save_dir is not None:
        fig.savefig(
            f"{save_dir}/{label}{suffix}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
    return fig, ax


def get_center_slice(morph, dim=1):
    # Get the center slice of a morphological stack
    center_slice = np.take(morph, morph.shape[dim] // 2, axis=dim)
    return center_slice


def plot_and_save_center_slice(
    morph,
    structure,
    output_dir=None,
    dim=1,
    ax=None,
    showfig=True,
    title=None,
    xlabel=None,
    ylabel=None,
    add_contour=False,
    **kwargs,
):
    # Plot the center slice of a morphological stack
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig = ax.get_figure()

    center_slice = get_center_slice(morph, dim=dim)

    pct_min = kwargs.get("pct_min", 0)
    pct_max = kwargs.get("pct_max", 99)

    vmin = kwargs.get("vmin", np.percentile(center_slice, pct_min))
    vmax = kwargs.get("vmax", np.percentile(center_slice, pct_max))

    cmap = cm.get_cmap(kwargs.get("cmap", "inferno"))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(center_slice))
    rgba[center_slice == 0] = (0, 0, 0, 0)
    ax.imshow(
        rgba,
        # cmap=cmap,
        # vmin=vmin,
        # vmax=vmax,
        origin="lower",
    )
    if add_contour:
        ax = add_contour_to_axis(
            ax,
            "center",
            dim,
            kwargs["domain_nuc"],
            kwargs["domain_mem"],
            lw=kwargs.get("lw"),
        )
    if title is not None:
        ax.set_title(title, color=kwargs.get("fontcolor", "white"))
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=kwargs.get("fontcolor", "white"))
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=kwargs.get("fontcolor", "white"))
    ax.axis("off")
    fig.set_facecolor((0, 0, 0, 0))
    ax.set_facecolor((0, 0, 0, 0))
    fig.set_facecolor("white")
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.tight_layout()
    if showfig:
        plt.show()
    if output_dir is not None:
        fig.savefig(str(output_dir / f"{structure}_center_slice_{dim}.png"))

    return fig, ax


def plot_and_save_max_proj(morph, structure, output_dir=None, dim=1, ax=None):
    # Plot the max intensity projection of a morphological stack
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig = ax.get_figure()

    max_proj = np.max(morph, axis=dim)

    ax.imshow(
        max_proj,
        cmap="inferno",
        # vmin=vmin,
        # vmax=vmax,
        origin="lower",
    )
    ax.set_title(structure, color="white")
    ax.axis("off")
    fig.set_facecolor("black")
    plt.tight_layout()
    plt.show()
    if output_dir is not None:
        fig.savefig(str(output_dir / f"{structure}_max_proj_{dim}.png"))


def make_individual_PILR_heatmap(
    df_corr,
    ch_list,
    save_dir=None,
    suffix="",
    vmin=-0.01,
    vmax=0.01,
    drawlines=True,
    cmap="PuOr",
    ch_labels=None,
    **kwargs,
):
    s = 0
    mx = -df_corr.values
    ax = sns.heatmap(mx, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    fig = ax.get_figure()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # get equally spaced x and y ticks
    xticks = np.linspace(xmin, xmax, len(ch_list) + 1)
    xticks = (xticks[1:] + xticks[:-1]) / 2

    yticks = np.linspace(ymin, ymax, len(ch_list) + 1)
    yticks = (yticks[1:] + yticks[:-1]) / 2
    yticks = yticks[::-1]

    # set the ticks
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # set the tick labels
    ch_labels = ch_labels if ch_labels is not None else ch_list
    ax.set_xticklabels(ch_list, rotation=45)
    ax.set_yticklabels(ch_list, rotation=0)

    for ch in ch_list:
        try:
            df_ch = df_corr.loc[ch]
            if df_ch.ndim == 1:
                s += 1
            else:
                s += df_ch.shape[0]
            if drawlines:
                ax.axvline(x=s, color="black", lw=1)
                ax.axhline(y=s, color="black", lw=1)
        except:
            pass
    # ax.axis("off")
    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / f"individual_correlations{suffix}.png", dpi=300)
    plt.show()
    return fig, ax


def save_PILR_as_tiff(pilr, save_dir, fname, writer):
    writer.save(
        pilr.astype(np.float32),
        str(save_dir / f"{fname}.ome.tiff"),
    )


def read_PILR_from_tiff(fname):

    pilr = AICSImage(fname)

    return pilr.data.squeeze()
