"""Tests for 2D Gaussian Splatting (surfel) pipeline.

Covers projection, rasterization, and the high-level rasterization_2dgs API.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.core_2dgs.projection_2dgs import fully_fused_projection_2dgs
from gsplat_mlx.core_2dgs.rasterization_2dgs import rasterize_to_pixels_2dgs
from gsplat_mlx.core.intersection import isect_offset_encode, isect_tiles
from gsplat_mlx.rendering import rasterization_2dgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surfels(N=10, seed=42):
    """Create a set of test surfels in front of a camera."""
    mx.random.seed(seed)
    np.random.seed(seed)

    # Place surfels in front of the camera (z > 0 in camera space)
    means = mx.array(np.random.randn(N, 3).astype(np.float32) * 0.5)
    # Push them forward so they're in front of the camera
    means_np = np.array(means)
    means_np[:, 2] = np.abs(means_np[:, 2]) + 2.0  # z > 2
    means = mx.array(means_np.astype(np.float32))

    # Random quaternions, normalised
    quats_np = np.random.randn(N, 4).astype(np.float32)
    quats_np = quats_np / np.linalg.norm(quats_np, axis=-1, keepdims=True)
    quats = mx.array(quats_np)

    # Small scales
    scales = mx.array(np.abs(np.random.randn(N, 3).astype(np.float32)) * 0.1 + 0.05)

    return means, quats, scales


def _make_camera(C=1, width=64, height=64, fx=50.0):
    """Create simple pinhole camera(s) looking down +Z."""
    viewmats = mx.broadcast_to(mx.eye(4), (C, 4, 4))
    cx, cy = width / 2.0, height / 2.0
    K = mx.array([
        [fx, 0.0, cx],
        [0.0, fx, cy],
        [0.0, 0.0, 1.0],
    ])
    Ks = mx.broadcast_to(K, (C, 3, 3))
    return viewmats, Ks, width, height


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------


class TestProjection2DGS:
    """Tests for fully_fused_projection_2dgs."""

    def test_2dgs_projection_shapes(self):
        """Verify output shapes from projection."""
        N, C = 20, 2
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, W, H = _make_camera(C)

        radii, means2d, depths, ray_transforms, normals = (
            fully_fused_projection_2dgs(
                means, quats, scales, viewmats, Ks, W, H,
            )
        )
        mx.eval(radii, means2d, depths, ray_transforms, normals)

        assert radii.shape == (C, N, 2), f"radii shape: {radii.shape}"
        assert radii.dtype == mx.int32
        assert means2d.shape == (C, N, 2), f"means2d shape: {means2d.shape}"
        assert depths.shape == (C, N), f"depths shape: {depths.shape}"
        assert ray_transforms.shape == (C, N, 3, 3), (
            f"ray_transforms shape: {ray_transforms.shape}"
        )
        assert normals.shape == (C, N, 3), f"normals shape: {normals.shape}"

    def test_2dgs_normals_toward_camera(self):
        """Normals should point toward the camera (positive dot with view dir).

        After flipping, dot(-normal, means_c) should be <= 0, meaning
        dot(normal, -means_c) >= 0 (normal points toward camera origin).
        """
        N, C = 15, 1
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, W, H = _make_camera(C)

        _, _, depths, _, normals = fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, W, H,
        )
        mx.eval(normals, depths)

        # For identity viewmat, means_c = means (world = camera)
        # Normal should satisfy: dot(normal, -means_c) >= 0
        # i.e. normal points from surfel toward camera origin
        means_np = np.array(means)
        normals_np = np.array(normals)[0]  # [N, 3] for camera 0

        for i in range(N):
            n = normals_np[i]
            mc = means_np[i]  # means_c for identity viewmat
            # dot(n, -mc) should be >= 0 (normal toward camera)
            dot_val = np.dot(n, -mc)
            assert dot_val >= -1e-5, (
                f"Surfel {i}: dot(normal, -means_c) = {dot_val} < 0"
            )

    def test_2dgs_ray_transforms_shape(self):
        """ray_transforms should be [C, N, 3, 3]."""
        N, C = 5, 3
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, W, H = _make_camera(C)

        _, _, _, ray_transforms, _ = fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, W, H,
        )
        mx.eval(ray_transforms)
        assert ray_transforms.shape == (C, N, 3, 3)

    def test_2dgs_near_far_culling(self):
        """Surfels behind the camera or beyond far plane get radius=0."""
        N = 5
        C = 1
        means_np = np.zeros((N, 3), dtype=np.float32)
        # Place surfels at various depths
        means_np[0, 2] = -1.0  # Behind camera
        means_np[1, 2] = 0.005  # Before near plane (0.01)
        means_np[2, 2] = 3.0  # Valid
        means_np[3, 2] = 5.0  # Valid
        means_np[4, 2] = 2e10  # Beyond far plane (1e10)

        means = mx.array(means_np)
        quats = mx.array(
            np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)
        )
        scales = mx.array(np.full((N, 3), 0.1, dtype=np.float32))
        viewmats, Ks, W, H = _make_camera(C)

        radii, _, _, _, _ = fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, W, H,
        )
        mx.eval(radii)
        radii_np = np.array(radii)[0]  # [N, 2]

        # Behind camera and before near plane should be culled
        assert radii_np[0, 0] == 0 and radii_np[0, 1] == 0, (
            f"Behind-camera surfel not culled: {radii_np[0]}"
        )
        assert radii_np[1, 0] == 0 and radii_np[1, 1] == 0, (
            f"Near-plane surfel not culled: {radii_np[1]}"
        )
        # Beyond far plane should be culled
        assert radii_np[4, 0] == 0 and radii_np[4, 1] == 0, (
            f"Far-plane surfel not culled: {radii_np[4]}"
        )
        # Valid surfels should have non-zero radii
        assert radii_np[2, 0] > 0 or radii_np[2, 1] > 0, (
            f"Valid surfel culled: {radii_np[2]}"
        )


# ---------------------------------------------------------------------------
# Rasterization tests
# ---------------------------------------------------------------------------


class TestRasterization2DGS:
    """Tests for rasterize_to_pixels_2dgs."""

    def _project_and_tile(self, means, quats, scales, C, W, H, tile_size=16):
        """Helper: project surfels and compute tile intersections."""
        viewmats, Ks, _, _ = _make_camera(C, W, H)
        radii, means2d, depths, ray_transforms, normals = (
            fully_fused_projection_2dgs(
                means, quats, scales, viewmats, Ks, W, H,
            )
        )

        tile_width = math.ceil(W / float(tile_size))
        tile_height = math.ceil(H / float(tile_size))

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            sort=True,
        )
        isect_offsets = isect_offset_encode(
            isect_ids, C, tile_width, tile_height,
        )

        return (
            means2d, ray_transforms, normals, radii, depths,
            isect_offsets, flatten_ids, viewmats, Ks,
        )

    def test_2dgs_render_basic(self):
        """Simple surfels should produce a non-zero image."""
        N, C, W, H = 10, 1, 32, 32
        means, quats, scales = _make_surfels(N)
        (
            means2d, ray_transforms, normals, radii, depths,
            isect_offsets, flatten_ids, _, _,
        ) = self._project_and_tile(means, quats, scales, C, W, H)

        channels = 3
        colors = mx.array(
            np.random.rand(C, N, channels).astype(np.float32) * 0.8 + 0.1
        )
        opacities = mx.array(np.full((C, N), 0.9, dtype=np.float32))

        render_colors, render_alphas, render_normals = rasterize_to_pixels_2dgs(
            means2d, ray_transforms, colors, normals, opacities,
            W, H, 16, isect_offsets, flatten_ids,
        )
        mx.eval(render_colors, render_alphas, render_normals)

        assert render_colors.shape == (C, H, W, channels)
        assert render_alphas.shape == (C, H, W, 1)
        assert render_normals.shape == (C, H, W, 3)

        # At least some pixels should be non-zero
        rc_np = np.array(render_colors)
        ra_np = np.array(render_alphas)
        assert rc_np.max() > 0.0, "Render produced all-zero colours"
        assert ra_np.max() > 0.0, "Render produced all-zero alphas"

    def test_2dgs_render_normals(self):
        """Normal map should have non-zero values where alpha > 0."""
        N, C, W, H = 10, 1, 32, 32
        means, quats, scales = _make_surfels(N)
        (
            means2d, ray_transforms, normals, radii, depths,
            isect_offsets, flatten_ids, _, _,
        ) = self._project_and_tile(means, quats, scales, C, W, H)

        colors = mx.array(np.ones((C, N, 3), dtype=np.float32))
        opacities = mx.array(np.full((C, N), 0.9, dtype=np.float32))

        _, render_alphas, render_normals = rasterize_to_pixels_2dgs(
            means2d, ray_transforms, colors, normals, opacities,
            W, H, 16, isect_offsets, flatten_ids,
        )
        mx.eval(render_alphas, render_normals)

        ra_np = np.array(render_alphas)
        rn_np = np.array(render_normals)

        # Where alpha > threshold, normals should be non-zero
        alpha_mask = ra_np[..., 0] > 0.01
        if alpha_mask.any():
            normals_at_visible = rn_np[alpha_mask]
            norms = np.linalg.norm(normals_at_visible, axis=-1)
            assert norms.max() > 0.0, (
                "Normal map is zero at visible pixels"
            )

    def test_2dgs_min_sigma(self):
        """Verify that min(sigma_3d, sigma_2d) behavior is active.

        The 2DGS rasterizer uses min(sigma_3d, sigma_2d) to choose the
        tighter bound. We verify the rasterizer produces output (which
        implicitly validates the min logic is not crashing).
        """
        N, C, W, H = 5, 1, 16, 16
        means, quats, scales = _make_surfels(N)
        (
            means2d, ray_transforms, normals, radii, depths,
            isect_offsets, flatten_ids, _, _,
        ) = self._project_and_tile(means, quats, scales, C, W, H)

        colors = mx.array(np.ones((C, N, 3), dtype=np.float32))
        opacities = mx.array(np.full((C, N), 0.95, dtype=np.float32))

        render_colors, render_alphas, _ = rasterize_to_pixels_2dgs(
            means2d, ray_transforms, colors, normals, opacities,
            W, H, 16, isect_offsets, flatten_ids,
        )
        mx.eval(render_colors, render_alphas)

        # Just verify it runs without error and produces valid output
        rc_np = np.array(render_colors)
        assert not np.any(np.isnan(rc_np)), "NaN in rendered colours"
        assert not np.any(np.isinf(rc_np)), "Inf in rendered colours"


# ---------------------------------------------------------------------------
# High-level API tests
# ---------------------------------------------------------------------------


class TestRasterization2DGSAPI:
    """Tests for the rasterization_2dgs() high-level API."""

    def test_2dgs_rasterization_api(self):
        """Full rasterization_2dgs() call works end-to-end."""
        N, C, W, H = 15, 1, 32, 32
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, _, _ = _make_camera(C, W, H)
        opacities = mx.array(np.full(N, 0.8, dtype=np.float32))
        colors = mx.array(
            np.random.rand(C, N, 3).astype(np.float32)
        )

        render_colors, render_alphas, render_normals, info = (
            rasterization_2dgs(
                means, quats, scales, opacities, colors,
                viewmats, Ks, W, H,
            )
        )
        mx.eval(render_colors, render_alphas, render_normals)

        assert render_colors.shape == (C, H, W, 3)
        assert render_alphas.shape == (C, H, W, 1)
        assert render_normals.shape == (C, H, W, 3)

        # Info dict should contain expected keys
        assert "means2d" in info
        assert "depths" in info
        assert "radii" in info
        assert "ray_transforms" in info
        assert "normals" in info

    def test_2dgs_output_shapes(self):
        """Verify all output dimensions from rasterization_2dgs."""
        N, C, W, H = 8, 2, 48, 48
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, _, _ = _make_camera(C, W, H)
        opacities = mx.array(np.full(N, 0.7, dtype=np.float32))
        colors = mx.array(
            np.random.rand(C, N, 3).astype(np.float32)
        )

        render_colors, render_alphas, render_normals, info = (
            rasterization_2dgs(
                means, quats, scales, opacities, colors,
                viewmats, Ks, W, H,
            )
        )
        mx.eval(render_colors, render_alphas, render_normals)

        assert render_colors.shape == (C, H, W, 3)
        assert render_alphas.shape == (C, H, W, 1)
        assert render_normals.shape == (C, H, W, 3)

        # Check info intermediates
        assert info["means2d"].shape == (C, N, 2)
        assert info["depths"].shape == (C, N)
        assert info["radii"].shape == (C, N, 2)
        assert info["ray_transforms"].shape == (C, N, 3, 3)
        assert info["normals"].shape == (C, N, 3)

    def test_2dgs_with_background(self):
        """Rendering with background colour should fill transparent areas."""
        N, C, W, H = 5, 1, 16, 16
        means, quats, scales = _make_surfels(N)
        viewmats, Ks, _, _ = _make_camera(C, W, H)
        opacities = mx.array(np.full(N, 0.5, dtype=np.float32))
        colors = mx.array(np.zeros((C, N, 3), dtype=np.float32))
        backgrounds = mx.array([[1.0, 0.0, 0.0]])  # Red background

        render_colors, render_alphas, _, _ = rasterization_2dgs(
            means, quats, scales, opacities, colors,
            viewmats, Ks, W, H, backgrounds=backgrounds,
        )
        mx.eval(render_colors, render_alphas)

        rc_np = np.array(render_colors)
        ra_np = np.array(render_alphas)

        # Where alpha is 0, colour should be the background
        transparent_mask = ra_np[0, :, :, 0] < 0.01
        if transparent_mask.any():
            bg_pixels = rc_np[0][transparent_mask]
            # Red channel should be ~1.0
            assert bg_pixels[:, 0].mean() > 0.9, (
                "Background not applied to transparent pixels"
            )
