import json
from pathlib import Path

import numpy as np
import trimesh
from skimage import io

from cellpack_analysis.lib.img_io import generate_composite_thumbnail, voxelize_packing


def _write_recipe(tmp_path: Path) -> Path:
    objects = {}
    for name, extents in [("membrane_mesh", [16, 16, 8]), ("nucleus_mesh", [6, 6, 4])]:
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation([10, 10, 5])
        mesh.export(tmp_path / f"{name}.obj")
        objects[name] = {
            "type": "mesh",
            "representations": {
                "mesh": {"path": str(tmp_path), "name": f"{name}.obj", "format": "obj"}
            },
        }
    objects["peroxisome"] = {"type": "single_sphere", "radius": 1.5}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps({"bounding_box": [[0, 0, 0], [20, 20, 10]], "objects": objects})
    )
    return recipe_path


def test_voxelize_packing_fills_meshes_and_spheres(tmp_path: Path) -> None:
    output_path = tmp_path / "voxelized_image.ome.tiff"
    voxelize_packing(_write_recipe(tmp_path), [[4.2, 5.7, 3.1]], output_path)

    img = io.imread(output_path)  # Z, Y, X, C
    assert img.shape == (10, 20, 20, 3)
    membrane, nucleus, puncta = (img[..., c] > 0 for c in range(3))
    assert membrane[5, 10, 10] and membrane[5, 4, 4] and not membrane[5, 0, 0]
    assert nucleus[5, 10, 10] and not nucleus[5, 4, 4]
    assert puncta[3, 5, 4] and puncta.sum() == 19  # lattice points within 1.5 of (4, 5, 3)


def test_voxelized_image_renders_as_cellpack_thumbnail(tmp_path: Path) -> None:
    tiff_path = tmp_path / "voxelized_image.ome.tiff"
    voxelize_packing(_write_recipe(tmp_path), [[4.2, 5.7, 3.1]], tiff_path)

    png_path = tmp_path / "thumbnail.png"
    generate_composite_thumbnail(tiff_path, png_path, image_type="cellpack")

    png = io.imread(png_path)
    assert png.ndim == 3 and np.any(png[..., :3] > 0)
