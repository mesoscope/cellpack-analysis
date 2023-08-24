import matplotlib.pyplot as plt
import numpy as np


def save_PILR_image(avg_gfp, ch_name, save_dir="./results", suffix="", aspect=20):
    # Save the PILR as an image
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        avg_gfp,
        cmap="inferno",
        vmin=np.min(avg_gfp),
        vmax=np.percentile(avg_gfp, 90),
        origin="lower",
    )
    ax.axis("off")
    fig.set_facecolor("black")
    ax.set_aspect(aspect)
    fig.savefig(
        f"{save_dir}/avg_PILR_{ch_name}{suffix}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
