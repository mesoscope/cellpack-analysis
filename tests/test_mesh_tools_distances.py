"""
Tests for scaled-nucleus-distance calculations in mesh_tools.py.

Uses concentric icosphere meshes as ground-truth geometry.  For a point at
(r, 0, 0) inside the cytoplasm (R_nuc < r < R_mem):

    nuc_surface_distance     = r - R_nuc
    distance_between_surfaces = R_mem - R_nuc
    scaled_nuc_distance       = (r - R_nuc) / (R_mem - R_nuc)

These analytical values hold for ANY direction by spherical symmetry, making
icospheres ideal fixtures.  The cross-comparison between the serial and
vectorized implementations is the key diagnostic for the trimesh index_ray
reordering bug.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh

# ---------------------------------------------------------------------------
# VTK stub – mesh_tools imports several vtkmodules sub-packages at module
# level even though the functions under test never call any VTK code.  On
# some cluster nodes these imports hang or are unavailable, so we inject
# MagicMock stubs for every sub-module before the first import of mesh_tools.
#
# All sub-module stubs must be in sys.modules BEFORE the import runs so that
# Python's import machinery resolves them from the cache without attempting
# to load the real packages.
# ---------------------------------------------------------------------------
_VTK_SUB_MODULES = [
    "vtkmodules",
    "vtkmodules.util",
    "vtkmodules.util.numpy_support",
    "vtkmodules.vtkCommonCore",
    "vtkmodules.vtkCommonDataModel",
    "vtkmodules.vtkFiltersCore",
    "vtkmodules.vtkIOGeometry",
    "vtkmodules.vtkIOParallel",
    "vtkmodules.vtkRenderingCore",
]
for _vtk_name in _VTK_SUB_MODULES:
    if _vtk_name not in sys.modules:
        sys.modules[_vtk_name] = MagicMock()

# Wire sub-module attributes so `from vtkmodules.util import numpy_support`
# and similar dotted imports resolve without triggering real package loading.
sys.modules["vtkmodules"].util = sys.modules["vtkmodules.util"]
sys.modules["vtkmodules.util"].numpy_support = sys.modules["vtkmodules.util.numpy_support"]

from cellpack_analysis.lib.mesh_tools import (  # noqa: E402
    _compute_distances_for_points,
    calc_scaled_distance_to_nucleus_surface,
    calc_scaled_distance_to_nucleus_surface_serial,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_NUC = 5.0  # inner (nucleus) sphere radius
R_MEM = 10.0  # outer (membrane) sphere radius
# For icosphere with subdivisions=4, surface deviates < 0.2 % from a perfect
# sphere so small absolute tolerances on distances are sufficient.
ATOL_DIST = 0.1  # tolerance for absolute distance values (μm)
ATOL_SCALED = 0.05  # tolerance for dimensionless scaled distances


# ---------------------------------------------------------------------------
# Module-scoped fixture: two concentric icosphere meshes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sphere_meshes():
    """Return concentric icosphere trimesh objects and their radii."""
    nuc_mesh = trimesh.creation.icosphere(subdivisions=4, radius=R_NUC)
    mem_mesh = trimesh.creation.icosphere(subdivisions=4, radius=R_MEM)
    # Force BVH build so workers don't rebuild per-ray
    _ = nuc_mesh.nearest
    _ = mem_mesh.ray
    return {"nuc_mesh": nuc_mesh, "mem_mesh": mem_mesh}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analytical_scaled(r: float) -> float:
    """Expected scaled nucleus distance for a point at radius r."""
    return (r - R_NUC) / (R_MEM - R_NUC)


def _points_at_radii(radii, axis=0):
    """Return (N, 3) array of points along the given axis at the supplied radii."""
    pts = np.zeros((len(radii), 3))
    pts[:, axis] = radii
    return pts


# ---------------------------------------------------------------------------
# 1. Serial – single axis-aligned point
# ---------------------------------------------------------------------------


class TestSerialSinglePoint:
    def test_midpoint_x_axis(self, sphere_meshes):
        r = 7.5  # halfway through the cytoplasm
        pts = np.array([[r, 0.0, 0.0]])

        nuc_d, scaled_d, surf_d = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        assert nuc_d[0] == pytest.approx(r - R_NUC, abs=ATOL_DIST)
        assert scaled_d[0] == pytest.approx(_analytical_scaled(r), abs=ATOL_SCALED)
        assert surf_d[0] == pytest.approx(R_MEM - R_NUC, abs=ATOL_DIST)

    def test_not_nan(self, sphere_meshes):
        pts = np.array([[7.0, 0.0, 0.0]])
        nuc_d, scaled_d, surf_d = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        assert not np.isnan(scaled_d[0]), "scaled distance should not be nan for a valid point"


# ---------------------------------------------------------------------------
# 2. Vectorized – single axis-aligned point
# ---------------------------------------------------------------------------


class TestVectorizedSinglePoint:
    def test_midpoint_x_axis(self, sphere_meshes):
        r = 7.5
        pts = np.array([[r, 0.0, 0.0]])

        nuc_d, scaled_d, surf_d = calc_scaled_distance_to_nucleus_surface(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        assert nuc_d[0] == pytest.approx(r - R_NUC, abs=ATOL_DIST)
        assert scaled_d[0] == pytest.approx(_analytical_scaled(r), abs=ATOL_SCALED)
        assert surf_d[0] == pytest.approx(R_MEM - R_NUC, abs=ATOL_DIST)

    def test_not_nan(self, sphere_meshes):
        pts = np.array([[7.0, 0.0, 0.0]])
        nuc_d, scaled_d, surf_d = calc_scaled_distance_to_nucleus_surface(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        assert not np.isnan(
            scaled_d[0]
        ), "vectorized scaled distance should not be nan for valid point"


# ---------------------------------------------------------------------------
# 3. Serial – batch of points at known radii along all three axes
# ---------------------------------------------------------------------------

# fractional position through the cytoplasm, mapped to r: r = R_NUC + frac * (R_MEM - R_NUC)
_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
_RADII = [R_NUC + f * (R_MEM - R_NUC) for f in _FRACS]


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
class TestSerialBatchPoints:
    def test_scaled_distances(self, sphere_meshes, axis):
        pts = _points_at_radii(_RADII, axis=axis)
        expected = np.array(_FRACS)

        _, scaled_d, _ = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        assert not np.any(np.isnan(scaled_d)), f"Unexpected NaN in serial batch along axis {axis}"
        np.testing.assert_allclose(scaled_d, expected, atol=ATOL_SCALED)

    def test_nuc_distances(self, sphere_meshes, axis):
        pts = _points_at_radii(_RADII, axis=axis)
        expected = np.array(_RADII) - R_NUC

        nuc_d, _, _ = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        np.testing.assert_allclose(nuc_d, expected, atol=ATOL_DIST)


# ---------------------------------------------------------------------------
# 4. Vectorized – batch of points at known radii along all three axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
class TestVectorizedBatchPoints:
    def test_scaled_distances(self, sphere_meshes, axis):
        """Primary regression test for the index_ray reordering bug."""
        pts = _points_at_radii(_RADII, axis=axis)
        expected = np.array(_FRACS)

        _, scaled_d, _ = calc_scaled_distance_to_nucleus_surface(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        assert not np.any(np.isnan(scaled_d)), (
            f"Unexpected NaN in vectorized batch along axis {axis} – "
            "possible trimesh index_ray reordering bug"
        )
        np.testing.assert_allclose(scaled_d, expected, atol=ATOL_SCALED)

    def test_nuc_distances(self, sphere_meshes, axis):
        pts = _points_at_radii(_RADII, axis=axis)
        expected = np.array(_RADII) - R_NUC

        nuc_d, _, _ = calc_scaled_distance_to_nucleus_surface(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        np.testing.assert_allclose(nuc_d, expected, atol=ATOL_DIST)


# ---------------------------------------------------------------------------
# 5. Cross-comparison: vectorized must match serial on a mixed batch
# ---------------------------------------------------------------------------


class TestVectorizedMatchesSerial:
    @pytest.fixture(scope="class")
    def mixed_points(self):
        """Points along all three axes at multiple radii – ~15 points total."""
        pts = []
        for axis in range(3):
            pts.append(_points_at_radii(_RADII, axis=axis))
        return np.vstack(pts)

    def test_scaled_distances_match(self, sphere_meshes, mixed_points):
        nuc_s, sc_s, surf_s = calc_scaled_distance_to_nucleus_surface_serial(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        nuc_v, sc_v, surf_v = calc_scaled_distance_to_nucleus_surface(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        # Same NaN positions
        assert np.array_equal(np.isnan(sc_s), np.isnan(sc_v)), (
            "Serial and vectorized implementations disagree on which points are NaN – "
            "possible index_ray reordering or missed-ray bug in vectorized path"
        )
        # Same values where not NaN
        valid = ~np.isnan(sc_s)
        np.testing.assert_allclose(
            sc_v[valid],
            sc_s[valid],
            atol=ATOL_SCALED,
            err_msg="Vectorized scaled distances differ from serial – index_ray bug suspected",
        )

    def test_surface_distances_match(self, sphere_meshes, mixed_points):
        _, _, surf_s = calc_scaled_distance_to_nucleus_surface_serial(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        _, _, surf_v = calc_scaled_distance_to_nucleus_surface(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        valid = ~np.isnan(surf_s) & ~np.isnan(surf_v)
        np.testing.assert_allclose(surf_v[valid], surf_s[valid], atol=ATOL_DIST)

    def test_nuc_distances_match(self, sphere_meshes, mixed_points):
        nuc_s, _, _ = calc_scaled_distance_to_nucleus_surface_serial(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        nuc_v, _, _ = calc_scaled_distance_to_nucleus_surface(
            mixed_points, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        # nuc_distances are computed from the same proximity query in both paths
        np.testing.assert_allclose(nuc_v, nuc_s, atol=ATOL_DIST)


# ---------------------------------------------------------------------------
# 6. Invalid points are masked as NaN in both implementations
# ---------------------------------------------------------------------------


class TestInvalidPointsReturnNan:
    @pytest.mark.parametrize(
        "point, description",
        [
            (np.array([[2.0, 0.0, 0.0]]), "inside nucleus"),
            (np.array([[12.0, 0.0, 0.0]]), "outside membrane"),
        ],
    )
    def test_serial_returns_nan(self, sphere_meshes, point, description):
        _, scaled_d, _ = calc_scaled_distance_to_nucleus_surface_serial(
            point, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        assert np.isnan(scaled_d[0]), f"Expected NaN for {description} (serial)"

    @pytest.mark.parametrize(
        "point, description",
        [
            (np.array([[2.0, 0.0, 0.0]]), "inside nucleus"),
            (np.array([[12.0, 0.0, 0.0]]), "outside membrane"),
        ],
    )
    def test_vectorized_returns_nan(self, sphere_meshes, point, description):
        _, scaled_d, _ = calc_scaled_distance_to_nucleus_surface(
            point, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        assert np.isnan(scaled_d[0]), f"Expected NaN for {description} (vectorized)"

    def test_valid_interior_not_nan(self, sphere_meshes):
        """Sanity check: a clearly interior cytoplasmic point should not be NaN."""
        pts = np.array([[7.5, 0.0, 0.0]])
        _, scaled_s, _ = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        _, scaled_v, _ = calc_scaled_distance_to_nucleus_surface(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )
        assert not np.isnan(scaled_s[0])
        assert not np.isnan(scaled_v[0])


# ---------------------------------------------------------------------------
# 7. Pre-computed mem_distances gives the same result as on-the-fly computation
# ---------------------------------------------------------------------------


class TestPrecomputedMemDistances:
    @pytest.fixture(scope="class")
    def batch_pts(self):
        return _points_at_radii(_RADII, axis=0)

    def test_serial_precomputed(self, sphere_meshes, batch_pts):
        from trimesh import proximity

        pre_mem = np.array(proximity.signed_distance(sphere_meshes["mem_mesh"], batch_pts))
        nuc_a, sc_a, surf_a = calc_scaled_distance_to_nucleus_surface_serial(
            batch_pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"], mem_distances=None
        )
        nuc_b, sc_b, surf_b = calc_scaled_distance_to_nucleus_surface_serial(
            batch_pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"], mem_distances=pre_mem
        )
        np.testing.assert_allclose(sc_a, sc_b, equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(surf_a, surf_b, equal_nan=True, atol=1e-10)

    def test_vectorized_precomputed(self, sphere_meshes, batch_pts):
        from trimesh import proximity

        pre_mem = np.array(proximity.signed_distance(sphere_meshes["mem_mesh"], batch_pts))
        nuc_a, sc_a, surf_a = calc_scaled_distance_to_nucleus_surface(
            batch_pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"], mem_distances=None
        )
        nuc_b, sc_b, surf_b = calc_scaled_distance_to_nucleus_surface(
            batch_pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"], mem_distances=pre_mem
        )
        np.testing.assert_allclose(sc_a, sc_b, equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(surf_a, surf_b, equal_nan=True, atol=1e-10)


# ---------------------------------------------------------------------------
# 8. Vectorized fallback: exception in batch ray cast triggers serial path
# ---------------------------------------------------------------------------


class TestVectorizedFallback:
    def test_fallback_on_ray_exception(self, sphere_meshes):
        """
        When mem_mesh.ray.intersects_location raises on the *batch* call the
        vectorized function must fall back to serial and return correct results.

        The serial fallback also calls intersects_location, one ray at a time.
        We therefore only raise on batch calls (len(ray_origins) > 1) and
        delegate single-ray calls back to the real implementation.
        """
        pts = _points_at_radii(_RADII, axis=0)

        # Compute expected answer via serial reference (uses the real ray caster)
        nuc_exp, sc_exp, surf_exp = calc_scaled_distance_to_nucleus_surface_serial(
            pts, sphere_meshes["nuc_mesh"], sphere_meshes["mem_mesh"]
        )

        mem_mesh = sphere_meshes["mem_mesh"]
        real_intersects = mem_mesh.ray.intersects_location

        def _batch_only_raiser(ray_origins, ray_directions, **kwargs):
            if len(ray_origins) > 1:
                raise RuntimeError("simulated trimesh failure")
            return real_intersects(ray_origins, ray_directions, **kwargs)

        with patch.object(mem_mesh.ray, "intersects_location", side_effect=_batch_only_raiser):
            nuc_fb, sc_fb, surf_fb = calc_scaled_distance_to_nucleus_surface(
                pts, sphere_meshes["nuc_mesh"], mem_mesh
            )

        np.testing.assert_allclose(
            sc_fb,
            sc_exp,
            equal_nan=True,
            atol=ATOL_SCALED,
            err_msg="Fallback to serial produced wrong scaled distances",
        )
        np.testing.assert_allclose(
            nuc_fb,
            nuc_exp,
            atol=ATOL_DIST,
            err_msg="Fallback to serial produced wrong nuc distances",
        )


# ---------------------------------------------------------------------------
# 9. _compute_distances_for_points — analytical correctness
# ---------------------------------------------------------------------------

_ALL_INTERIOR_MEASURES = {"membrane", "nucleus", "scaled_nucleus", "z", "scaled_z"}


class TestComputeDistancesForPoints:
    """Verify the shared primitive against analytical values for concentric spheres."""

    @pytest.fixture(scope="class")
    def batch_pts(self):
        """Cytoplasmic points along the x-axis at known radii."""
        return _points_at_radii(_RADII, axis=0)

    def test_returns_requested_keys_only(self, sphere_meshes, batch_pts):
        for measure in _ALL_INTERIOR_MEASURES:
            result = _compute_distances_for_points(
                batch_pts,
                sphere_meshes["nuc_mesh"],
                sphere_meshes["mem_mesh"],
                {measure},
            )
            assert set(result.keys()) == {measure}, (
                f"Expected only key '{measure}', got {set(result.keys())}"
            )

    def test_nucleus_distances_analytical(self, sphere_meshes, batch_pts):
        result = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"nucleus"},
        )
        expected = np.array(_RADII) - R_NUC
        np.testing.assert_allclose(result["nucleus"], expected, atol=ATOL_DIST)

    def test_scaled_nucleus_distances_analytical(self, sphere_meshes, batch_pts):
        result = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"scaled_nucleus"},
        )
        expected = np.array(_FRACS)
        valid = ~np.isnan(result["scaled_nucleus"])
        np.testing.assert_allclose(result["scaled_nucleus"][valid], expected[valid], atol=ATOL_SCALED)

    def test_membrane_distances_positive_interior(self, sphere_meshes, batch_pts):
        result = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"membrane"},
        )
        # All points are inside the membrane → signed distance > 0
        assert np.all(result["membrane"] > 0), "Interior points must have positive mem distance"

    def test_z_distances_from_mesh_bounds(self, sphere_meshes, batch_pts):
        """Verify z_min is derived from mem_mesh.bounds (single source of truth)."""
        mem_mesh = sphere_meshes["mem_mesh"]
        z_min = float(mem_mesh.bounds[0, 2])
        expected = np.abs(batch_pts[:, 2] - z_min)
        result = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            mem_mesh,
            {"z"},
        )
        np.testing.assert_allclose(result["z"], expected, atol=1e-10)

    def test_scaled_z_in_unit_interval(self, sphere_meshes):
        """Cytoplasmic points along all axes should have scaled_z in [0, 1]."""
        pts = np.vstack([_points_at_radii(_RADII, axis=ax) for ax in range(3)])
        result = _compute_distances_for_points(
            pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"scaled_z"},
        )
        valid = ~np.isnan(result["scaled_z"])
        assert np.all((result["scaled_z"][valid] >= 0) & (result["scaled_z"][valid] <= 1))

    def test_precomputed_mem_distances_reused(self, sphere_meshes, batch_pts):
        """Supplying pre-computed mem_distances must give the same result."""
        from trimesh import proximity as _prox

        pre_mem = np.array(_prox.signed_distance(sphere_meshes["mem_mesh"], batch_pts))
        result_auto = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"nucleus", "scaled_nucleus", "membrane"},
        )
        result_pre = _compute_distances_for_points(
            batch_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {"nucleus", "scaled_nucleus", "membrane"},
            mem_distances=pre_mem,
        )
        for dm in ("nucleus", "scaled_nucleus", "membrane"):
            np.testing.assert_allclose(
                result_pre[dm], result_auto[dm], equal_nan=True, atol=1e-10,
                err_msg=f"Pre-computed mem_distances changed result for '{dm}'",
            )


# ---------------------------------------------------------------------------
# 10. Single-source-of-truth regression gate
#     _calculate_distances_for_cell_id must agree with _compute_distances_for_points
#     on all shared measures (nucleus, scaled_nucleus, membrane, z, scaled_z).
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """Regression gate: particle pipeline and grid primitive must agree."""

    @pytest.fixture(scope="class")
    def cytoplasmic_pts(self):
        """Mix of cytoplasmic points along all three axes."""
        return np.vstack([_points_at_radii(_RADII, axis=ax) for ax in range(3)])

    @pytest.fixture(scope="class")
    def mesh_dict(self, sphere_meshes):
        mem_mesh = sphere_meshes["mem_mesh"]
        return {
            "test_cell": {
                "nuc_mesh": sphere_meshes["nuc_mesh"],
                "mem_mesh": mem_mesh,
                "mem_bounds": mem_mesh.bounds,
            }
        }

    @pytest.mark.parametrize(
        "measure",
        ["nucleus", "scaled_nucleus", "membrane", "z", "scaled_z"],
    )
    def test_particle_pipeline_matches_primitive(self, sphere_meshes, mesh_dict, cytoplasmic_pts, measure):
        """The particle distance pipeline must produce values identical to the primitive."""
        # Defer this import so the VTK stub is already installed.
        from cellpack_analysis.lib.distance import _calculate_distances_for_cell_id

        cell_id = "test_cell"
        _, particle_result = _calculate_distances_for_cell_id(
            cell_id,
            cytoplasmic_pts,
            mesh_dict,
            distance_measures=[measure],
        )
        prim_result = _compute_distances_for_points(
            cytoplasmic_pts,
            sphere_meshes["nuc_mesh"],
            sphere_meshes["mem_mesh"],
            {measure},
        )

        # The particle pipeline applies filter_invalid_distances; the primitive does not.
        # Compare only valid (non-NaN, non-inf) entries from the primitive.
        prim_arr = prim_result[measure]
        valid = ~np.isnan(prim_arr) & ~np.isinf(prim_arr)
        prim_valid = prim_arr[valid]

        particle_arr = particle_result.get(measure, np.array([]))

        assert len(particle_arr) == len(prim_valid), (
            f"Length mismatch for '{measure}': "
            f"particle pipeline={len(particle_arr)}, primitive={len(prim_valid)}"
        )
        np.testing.assert_allclose(
            particle_arr,
            prim_valid,
            atol=ATOL_DIST if measure in ("nucleus", "membrane", "z") else ATOL_SCALED,
            equal_nan=True,
            err_msg=(
                f"Particle pipeline disagrees with primitive for '{measure}' – "
                "single source of truth invariant violated"
            ),
        )
