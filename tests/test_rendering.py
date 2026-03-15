"""Tests for the high-level rasterization() rendering API (PRD-09).

Covers basic RGB rendering, SH evaluation, depth modes, antialiased
mode, multi-camera, background blending, camera models, edge cases,
and output shape / info-dict validation.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from conftest import make_camera_intrinsics, make_gaussians, make_view_matrix
from gsplat_mlx.rendering import (
    RasterizeMode,
    RenderMode,
    rasterization,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_expected_depth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_scene(
    N: int = 100,
    C: int = 1,
    width: int = 64,
    height: int = 64,
    sh_degree: int = 0,
    seed: int = 42,
):
    """Build a minimal test scene with Gaussians in front of the camera."""
    g = make_gaussians(N=N, sh_degree=sh_degree, seed=seed)

    # Place Gaussians in a small volume in front of the camera
    np.random.seed(seed)
    means = mx.array(
        np.random.uniform(-2.0, 2.0, (N, 3)).astype(np.float32)
    )
    # Make sure z > 0 (in front of camera at origin looking down -Z)
    means_np = np.array(means)
    means_np[:, 2] = np.abs(means_np[:, 2]) + 1.0
    means = mx.array(means_np)

    # Reasonable scales (exponentiated)
    scales = mx.exp(
        mx.array(np.random.uniform(-3.0, -1.0, (N, 3)).astype(np.float32))
    )

    # Opacities in [0, 1]
    opacities = mx.sigmoid(g["opacities"])

    # Camera(s)
    viewmats = []
    for i in range(C):
        angle = 2.0 * math.pi * i / max(C, 1)
        eye = (5.0 * math.sin(angle), 0.0, 5.0 * math.cos(angle))
        viewmats.append(make_view_matrix(eye=eye, target=(0, 0, 0)))
    viewmats = mx.stack(viewmats, axis=0)  # [C, 4, 4]

    K = make_camera_intrinsics(width=width, height=height, fx=50.0, fy=50.0)
    Ks = mx.stack([K] * C, axis=0)  # [C, 3, 3]

    return {
        "means": means,
        "quats": g["quats"],
        "scales": scales,
        "opacities": opacities,
        "sh_coeffs": g["sh_coeffs"],
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRasterizationRGBBasic:
    """Basic RGB rendering with direct colors."""

    def test_rasterization_rgb_basic(self):
        """N=100 Gaussians, 1 camera, 64x64 -> valid output shape."""
        scene = _make_test_scene(N=100, C=1)
        N = scene["means"].shape[0]
        C = 1

        # Direct RGB colors [C, N, 3]
        colors = mx.array(
            np.random.rand(C, N, 3).astype(np.float32)
        )

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 3)
        assert render_alphas.shape == (1, 64, 64, 1)
        # Values should be finite
        assert not np.any(np.isnan(np.array(render_colors)))
        assert not np.any(np.isnan(np.array(render_alphas)))


class TestRasterizationSH:
    """SH coefficient rendering modes."""

    def test_rasterization_sh_degree0(self):
        """SH degree 0 color."""
        scene = _make_test_scene(N=50, C=1, sh_degree=0)
        # degree 0: K = 1
        sh_coeffs = mx.array(
            np.random.randn(50, 1, 3).astype(np.float32) * 0.1
        )

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=sh_coeffs,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=0,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 3)
        assert not np.any(np.isnan(np.array(render_colors)))

    def test_rasterization_sh_degree3(self):
        """SH degree 3 color."""
        scene = _make_test_scene(N=50, C=1, sh_degree=3)

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=scene["sh_coeffs"],
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=3,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 3)
        assert not np.any(np.isnan(np.array(render_colors)))


class TestRasterizationDirectColor:
    """Direct color mode (no SH)."""

    def test_rasterization_direct_color(self):
        """colors=[C, N, 3] without SH."""
        scene = _make_test_scene(N=80, C=1)
        N = 80
        C = 1
        colors = mx.array(np.random.rand(C, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
        )

        mx.eval(render_colors)
        assert render_colors.shape == (1, 64, 64, 3)


class TestRasterizationDepthModes:
    """Depth rendering modes."""

    def test_rasterization_depth_mode(self):
        """render_mode='D' produces 1-channel output."""
        scene = _make_test_scene(N=50, C=1)
        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            render_mode="D",
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 1)

    def test_rasterization_rgb_plus_depth(self):
        """render_mode='RGB+D' produces 4-channel output."""
        scene = _make_test_scene(N=50, C=1)
        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            render_mode="RGB+D",
        )

        mx.eval(render_colors)
        assert render_colors.shape == (1, 64, 64, 4)

    def test_rasterization_expected_depth(self):
        """render_mode='ED' normalises depth by alpha."""
        scene = _make_test_scene(N=50, C=1)
        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            render_mode="ED",
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 1)
        # Expected depth should be finite
        assert not np.any(np.isnan(np.array(render_colors)))


class TestRasterizationAntialiased:
    """Antialiased rasterization mode."""

    def test_rasterization_antialiased(self):
        """rasterize_mode='antialiased' runs without error."""
        scene = _make_test_scene(N=50, C=1)
        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            rasterize_mode="antialiased",
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 3)


class TestRasterizationBackground:
    """Background color handling."""

    def test_rasterization_background(self):
        """Background color fills transparent areas."""
        scene = _make_test_scene(N=10, C=1)
        N = 10
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))
        bg = mx.array([[1.0, 0.0, 0.0]])  # red background

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            backgrounds=bg,
        )

        mx.eval(render_colors, render_alphas)
        rc = np.array(render_colors)
        ra = np.array(render_alphas)

        # Where alpha is 0, the color should be the background (red)
        transparent_mask = ra[0, :, :, 0] < 0.01
        if np.any(transparent_mask):
            # Red channel should be close to 1.0 for transparent pixels
            red_values = rc[0, transparent_mask, 0]
            assert np.all(red_values > 0.9), (
                f"Background not applied correctly. "
                f"Min red in transparent area: {red_values.min()}"
            )


class TestRasterizationMultiCamera:
    """Multi-camera rendering."""

    def test_rasterization_multi_camera(self):
        """C=2 cameras produce correct shape."""
        scene = _make_test_scene(N=50, C=2)
        N = 50
        C = 2
        colors = mx.array(np.random.rand(C, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (2, 64, 64, 3)
        assert render_alphas.shape == (2, 64, 64, 1)
        assert info["n_cameras"] == 2


class TestRasterizationCameraModels:
    """All three camera models."""

    @pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
    def test_rasterization_camera_models(self, camera_model):
        """Each camera model runs without error."""
        scene = _make_test_scene(N=30, C=1)
        N = 30
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            camera_model=camera_model,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, 3)


class TestRasterizationOutputShapes:
    """Comprehensive output shape validation."""

    @pytest.mark.parametrize(
        "render_mode, expected_channels",
        [
            ("RGB", 3),
            ("D", 1),
            ("ED", 1),
            ("RGB+D", 4),
            ("RGB+ED", 4),
        ],
    )
    def test_rasterization_output_shapes(self, render_mode, expected_channels):
        """Verify all output shapes for each render mode."""
        scene = _make_test_scene(N=30, C=1)
        N = 30
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
            render_mode=render_mode,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, 64, 64, expected_channels)
        assert render_alphas.shape == (1, 64, 64, 1)


class TestRasterizationInfoDict:
    """Info dict validation."""

    def test_rasterization_info_dict(self):
        """Verify info dict has all required keys."""
        scene = _make_test_scene(N=50, C=1)
        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        _, _, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
        )

        required_keys = {
            "means2d",
            "depths",
            "conics",
            "radii",
            "width",
            "height",
            "tile_size",
            "n_cameras",
            "tiles_per_gauss",
            "isect_offsets",
            "flatten_ids",
        }
        assert required_keys.issubset(
            set(info.keys())
        ), f"Missing keys: {required_keys - set(info.keys())}"

        # Check shapes of array values
        assert info["means2d"].shape == (1, N, 2)
        assert info["depths"].shape == (1, N)
        assert info["conics"].shape == (1, N, 3)
        assert info["radii"].shape == (1, N, 2)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["tile_size"] == 16
        assert info["n_cameras"] == 1


class TestRasterizationNearFar:
    """Near/far plane culling."""

    def test_rasterization_near_far(self):
        """Gaussians behind camera are culled."""
        scene = _make_test_scene(N=50, C=1)

        # Place all Gaussians behind the camera (negative z in camera space)
        # Camera is at (0, 0, 5) looking at (0, 0, 0), so objects at z > 5
        # in world space would be behind the camera
        means_behind = mx.array(
            np.random.uniform(-1, 1, (50, 3)).astype(np.float32)
        )
        # Set z to be far behind the camera
        means_np = np.array(means_behind)
        means_np[:, 2] = -100.0  # well behind camera
        means_behind = mx.array(means_np)

        N = 50
        colors = mx.array(np.random.rand(1, N, 3).astype(np.float32))

        render_colors, render_alphas, info = rasterization(
            means=means_behind,
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=colors,
            viewmats=scene["viewmats"],
            Ks=scene["Ks"],
            width=scene["width"],
            height=scene["height"],
            sh_degree=None,
        )

        mx.eval(render_colors, render_alphas)
        ra = np.array(render_alphas)
        # All pixels should have very low alpha since Gaussians are behind camera
        assert np.all(ra < 0.1), (
            f"Expected near-zero alpha for behind-camera Gaussians, "
            f"got max alpha={ra.max()}"
        )


class TestRasterizationEmpty:
    """Edge case: no Gaussians."""

    def test_rasterization_empty(self):
        """N=0 produces black image with zero alpha."""
        C = 1
        W, H = 32, 32
        viewmat = make_view_matrix()
        K = make_camera_intrinsics(width=W, height=H, fx=50.0, fy=50.0)

        means = mx.zeros((0, 3))
        quats = mx.zeros((0, 4))
        scales = mx.zeros((0, 3))
        opacities = mx.zeros((0,))
        colors = mx.zeros((C, 0, 3))

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,
        )

        mx.eval(render_colors, render_alphas)
        assert render_colors.shape == (1, H, W, 3)
        assert render_alphas.shape == (1, H, W, 1)
        # Should be all zeros (black with no alpha)
        assert np.allclose(np.array(render_colors), 0.0)
        assert np.allclose(np.array(render_alphas), 0.0)


class TestRenderModeHelpers:
    """Render mode helper functions."""

    def test_render_mode_has_color(self):
        assert render_mode_has_color("RGB") is True
        assert render_mode_has_color("RGB+D") is True
        assert render_mode_has_color("RGB+ED") is True
        assert render_mode_has_color("D") is False
        assert render_mode_has_color("ED") is False

    def test_render_mode_has_depth(self):
        assert render_mode_has_depth("D") is True
        assert render_mode_has_depth("ED") is True
        assert render_mode_has_depth("RGB+D") is True
        assert render_mode_has_depth("RGB+ED") is True
        assert render_mode_has_depth("RGB") is False

    def test_render_mode_has_expected_depth(self):
        assert render_mode_has_expected_depth("ED") is True
        assert render_mode_has_expected_depth("RGB+ED") is True
        assert render_mode_has_expected_depth("D") is False
        assert render_mode_has_expected_depth("RGB+D") is False


class TestEndToEndGradient:
    """End-to-end gradient flow through the full rasterization pipeline."""

    def test_end_to_end_gradient_flow(self):
        """Verify mx.grad flows through the ENTIRE rasterization pipeline."""
        N = 50
        # Place Gaussians in a cluster in front of the camera
        mx.random.seed(123)
        means = mx.random.uniform(-0.5, 0.5, (N, 3))
        # Ensure positive z (in front of camera looking down -z at origin)
        means_np = np.array(means)
        means_np[:, 2] = np.abs(means_np[:, 2]) + 2.0
        means = mx.array(means_np)

        quats = mx.concatenate([mx.ones((N, 1)), mx.zeros((N, 3))], axis=1)
        scales = mx.full((N, 3), 0.3)  # Larger Gaussians for visibility
        opacities = mx.ones(N) * 0.9
        colors = mx.random.uniform(0, 1, (N, 1, 3))

        K = mx.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=mx.float32)
        viewmat = mx.eye(4)

        def loss_fn(means_):
            imgs, alphas, info = rasterization(
                means_, quats, scales, opacities, colors,
                viewmat[None], K[None], width=64, height=64,
                sh_degree=0, differentiable=True,
            )
            return mx.sum(imgs)

        grad = mx.grad(loss_fn)(means)
        mx.eval(grad)
        grad_np = np.array(grad)
        assert not np.any(np.isnan(grad_np)), "NaN in gradient"
        assert np.any(grad_np != 0), "All-zero gradient"
