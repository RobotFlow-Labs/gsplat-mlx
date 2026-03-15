"""Tests for the Tier-2 (differentiable MLX) pixel rasterizer.

Validates that rasterize_to_pixels_mlx:
1. Produces the same output as the Tier-1 NumPy reference
2. Preserves the MLX computation graph so mx.grad() works
3. Handles edge cases (empty images, backgrounds, shapes)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    TRANSMITTANCE_THRESHOLD,
)
from gsplat_mlx.core.rasterization import rasterize_to_pixels
from gsplat_mlx.core.rasterization_mlx import rasterize_to_pixels_mlx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_isect(
    C: int,
    tile_height: int,
    tile_width: int,
    tile_gaussians: dict,
):
    """Build isect_offsets and flatten_ids for a simple layout.

    Args:
        C: number of cameras.
        tile_height, tile_width: tile grid dimensions.
        tile_gaussians: dict mapping ``(ty, tx)`` -> list of Gaussian ids.

    Returns:
        isect_offsets: mx.array [C, tile_H, tile_W]
        flatten_ids:   mx.array [n_isects]
    """
    flat_ids = []
    offsets = np.zeros((C, tile_height, tile_width), dtype=np.int32)
    running = 0
    for ty in range(tile_height):
        for tx in range(tile_width):
            offsets[0, ty, tx] = running
            gids = tile_gaussians.get((ty, tx), [])
            flat_ids.extend(gids)
            running += len(gids)
    return mx.array(offsets), mx.array(np.array(flat_ids, dtype=np.int32))


def _make_identity_conic():
    """Return conic (a, b, c) = (1, 0, 1) -- isotropic unit Gaussian."""
    return np.array([1.0, 0.0, 1.0], dtype=np.float32)


def _make_tight_conic(sigma: float = 0.5):
    """Return a tight conic so the Gaussian is concentrated."""
    inv_var = 1.0 / (sigma * sigma)
    return np.array([inv_var, 0.0, inv_var], dtype=np.float32)


def _build_simple_scene(N=1, ch=3, H=8, W=8, ts=8, opacity=0.8):
    """Build a simple scene with N Gaussians near center for testing."""
    C = 1
    tH = (H + ts - 1) // ts
    tW = (W + ts - 1) // ts

    cx, cy = W / 2.0, H / 2.0
    means_np = np.zeros((C, N, 2), dtype=np.float32)
    conics_np = np.zeros((C, N, 3), dtype=np.float32)
    colors_np = np.zeros((C, N, ch), dtype=np.float32)
    opacities_np = np.full((C, N), opacity, dtype=np.float32)

    for i in range(N):
        means_np[0, i] = [cx + i * 0.1, cy + i * 0.1]
        conics_np[0, i] = _make_identity_conic()
        colors_np[0, i] = np.random.RandomState(42 + i).rand(ch).astype(np.float32)

    tile_gaussians = {(0, 0): list(range(N))}
    isect_offsets, flatten_ids = _simple_isect(C, tH, tW, tile_gaussians)

    return (
        mx.array(means_np),
        mx.array(conics_np),
        mx.array(colors_np),
        mx.array(opacities_np),
        W, H, ts,
        isect_offsets,
        flatten_ids,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiffRasterOutputShapes:
    """Correct output dimensions."""

    def test_diff_raster_output_shapes(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 8, 8, 8
        tH, tW = 1, 1

        means2d = mx.array(np.array([[[4.0, 4.0]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.ones((C, N, ch), dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        assert rc.shape == (C, H, W, ch), f"render_colors shape {rc.shape}"
        assert ra.shape == (C, H, W, 1), f"render_alphas shape {ra.shape}"

    def test_multi_channel_shapes(self):
        C, N = 1, 1
        ch = 7
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors_np = np.arange(ch, dtype=np.float32).reshape(1, 1, ch)
        colors = mx.array(colors_np)
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        assert rc.shape == (C, H, W, ch)
        assert ra.shape == (C, H, W, 1)


class TestDiffRasterMatchesReference:
    """Output matches NumPy rasterizer within tolerance."""

    def test_single_gaussian(self):
        """Single Gaussian: MLX matches NumPy reference."""
        C, N, ch = 1, 1, 3
        H, W, ts = 8, 8, 8

        means2d = mx.array(np.array([[[4.5, 4.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.5, 0.2]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        rc_ref, ra_ref = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        rc_mlx, ra_mlx = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc_ref, ra_ref, rc_mlx, ra_mlx)

        np.testing.assert_allclose(
            np.array(rc_mlx), np.array(rc_ref), atol=1e-5,
            err_msg="MLX rasterizer colors don't match NumPy reference",
        )
        np.testing.assert_allclose(
            np.array(ra_mlx), np.array(ra_ref), atol=1e-5,
            err_msg="MLX rasterizer alphas don't match NumPy reference",
        )

    def test_two_gaussians(self):
        """Two overlapping Gaussians: MLX matches NumPy reference."""
        C, N, ch = 1, 2, 3
        H, W, ts = 8, 8, 8

        means2d = mx.array(np.array([[[3.5, 3.5], [5.5, 5.5]]], dtype=np.float32))
        conics = mx.array(np.array([
            [_make_identity_conic(), _make_identity_conic()]
        ], dtype=np.float32))
        colors = mx.array(np.array([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ], dtype=np.float32))
        opacities = mx.array(np.array([[0.8, 0.7]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0, 1]})

        rc_ref, ra_ref = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        rc_mlx, ra_mlx = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc_ref, ra_ref, rc_mlx, ra_mlx)

        np.testing.assert_allclose(
            np.array(rc_mlx), np.array(rc_ref), atol=1e-4,
            err_msg="MLX rasterizer colors don't match for 2 Gaussians",
        )
        np.testing.assert_allclose(
            np.array(ra_mlx), np.array(ra_ref), atol=1e-4,
            err_msg="MLX rasterizer alphas don't match for 2 Gaussians",
        )

    def test_tight_conic(self):
        """Tight conic: MLX matches NumPy reference."""
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_tight_conic(0.5)]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        rc_ref, ra_ref = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        rc_mlx, ra_mlx = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc_ref, ra_ref, rc_mlx, ra_mlx)

        np.testing.assert_allclose(
            np.array(rc_mlx), np.array(rc_ref), atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(ra_mlx), np.array(ra_ref), atol=1e-5,
        )


class TestDiffRasterSingleGaussian:
    """Exact values match manual computation."""

    def test_diff_raster_single_gaussian(self):
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[1.5, 1.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacity_val = 0.8
        opacities = mx.array(np.array([[opacity_val]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        rc_np = np.array(rc)
        ra_np = np.array(ra)

        # At pixel (1, 1) center (1.5, 1.5): dx=0, dy=0 -> sigma=0
        # alpha = min(0.8, MAX_ALPHA) = 0.8, weight = 1.0 * 0.8 = 0.8
        expected_color = opacity_val
        expected_alpha = opacity_val
        np.testing.assert_allclose(rc_np[0, 1, 1, 0], expected_color, atol=1e-5)
        np.testing.assert_allclose(ra_np[0, 1, 1, 0], expected_alpha, atol=1e-5)

        # At pixel (0, 0) center (0.5, 0.5): dx=-1, dy=-1
        # sigma = 0.5*(1+1) = 1.0
        # alpha = 0.8 * exp(-1) = 0.29430
        expected_alpha_corner = opacity_val * math.exp(-1.0)
        expected_color_corner = expected_alpha_corner
        np.testing.assert_allclose(rc_np[0, 0, 0, 0], expected_color_corner, atol=1e-5)


class TestDiffRasterBackground:
    """Background blending works and is differentiable."""

    def test_diff_raster_background(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[0.5, 0.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_tight_conic(0.5)]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        bg = mx.array(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
            backgrounds=bg,
        )
        mx.eval(rc, ra)
        rc_np = np.array(rc)

        # Far pixel should show green background
        green_val = rc_np[0, 3, 3, 1]
        red_val = rc_np[0, 3, 3, 0]
        assert green_val > 0.5, f"Far pixel should show green bg: {green_val:.4f}"
        assert green_val > red_val, "Background should dominate far pixels"

    def test_background_matches_reference(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.7]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})
        bg = mx.array(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))

        rc_ref, ra_ref = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids, backgrounds=bg,
        )
        rc_mlx, ra_mlx = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids, backgrounds=bg,
        )
        mx.eval(rc_ref, ra_ref, rc_mlx, ra_mlx)

        np.testing.assert_allclose(
            np.array(rc_mlx), np.array(rc_ref), atol=1e-5,
        )

    def test_background_gradient_flows(self):
        """Gradient flows through background blending."""
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.5]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})
        bg = mx.array(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

        def loss_fn(bg_input):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, conics, colors, opacities, W, H, ts,
                isect_offsets, flatten_ids, backgrounds=bg_input,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_bg = grad_fn(bg)
        mx.eval(grad_bg)

        # Background contributes to all pixels, so gradient should be non-zero
        grad_np = np.array(grad_bg)
        assert np.any(np.abs(grad_np) > 1e-6), (
            f"Background gradient should be non-zero, got {grad_np}"
        )


class TestDiffRasterEmpty:
    """Empty scene handling."""

    def test_empty_no_background(self):
        C, N, ch = 1, 0, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.zeros((C, 0, 2), dtype=np.float32))
        conics = mx.array(np.zeros((C, 0, 3), dtype=np.float32))
        colors = mx.array(np.zeros((C, 0, ch), dtype=np.float32))
        opacities = mx.array(np.zeros((C, 0), dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {})

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        np.testing.assert_array_equal(np.array(rc), 0.0)
        np.testing.assert_array_equal(np.array(ra), 0.0)

    def test_empty_with_background(self):
        C, N, ch = 1, 0, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.zeros((C, 0, 2), dtype=np.float32))
        conics = mx.array(np.zeros((C, 0, 3), dtype=np.float32))
        colors = mx.array(np.zeros((C, 0, ch), dtype=np.float32))
        opacities = mx.array(np.zeros((C, 0), dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {})
        bg = mx.array(np.array([[0.2, 0.4, 0.6]], dtype=np.float32))

        rc, ra = rasterize_to_pixels_mlx(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids, backgrounds=bg,
        )
        mx.eval(rc, ra)

        rc_np = np.array(rc)
        for py in range(H):
            for px in range(W):
                np.testing.assert_allclose(
                    rc_np[0, py, px], [0.2, 0.4, 0.6], atol=1e-6,
                )


class TestDiffRasterGradientColors:
    """mx.grad flows to colors."""

    def test_diff_raster_gradient_colors(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[0.5, 0.3, 0.1]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(c):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, conics, c, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_colors = grad_fn(colors)
        mx.eval(grad_colors)

        grad_np = np.array(grad_colors)
        # Color gradient should be non-zero: each color channel contributes
        # to the output via weight * color[channel]
        assert grad_np.shape == colors.shape
        assert np.all(np.abs(grad_np) > 1e-6), (
            f"Color gradient should be non-zero everywhere, got {grad_np}"
        )

    def test_gradient_colors_finite_diff(self):
        """Gradient matches finite differences for colors."""
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[0.5]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(c):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, conics, c, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        # Analytical gradient
        grad_fn = mx.grad(loss_fn)
        grad_ana = grad_fn(colors)
        mx.eval(grad_ana)

        # Finite difference gradient
        eps = 1e-4
        colors_plus = colors + eps
        colors_minus = colors - eps
        loss_plus = loss_fn(colors_plus)
        loss_minus = loss_fn(colors_minus)
        mx.eval(loss_plus, loss_minus)
        grad_fd = (np.array(loss_plus) - np.array(loss_minus)) / (2 * eps)

        np.testing.assert_allclose(
            np.array(grad_ana).sum(), grad_fd, rtol=1e-3,
            err_msg="Color gradient doesn't match finite differences",
        )


class TestDiffRasterGradientMeans2d:
    """mx.grad flows to means2d."""

    def test_diff_raster_gradient_means2d(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.5, 0.2]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(m):
            rc, _ = rasterize_to_pixels_mlx(
                m, conics, colors, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_means = grad_fn(means2d)
        mx.eval(grad_means)

        grad_np = np.array(grad_means)
        assert grad_np.shape == means2d.shape
        # When Gaussian is centered, the sum of all pixel contributions is
        # symmetric, but the gradient might be small. Just check it's finite.
        assert np.all(np.isfinite(grad_np)), f"means2d gradient not finite: {grad_np}"

    def test_gradient_means2d_off_center(self):
        """Off-center Gaussian should have non-zero spatial gradient."""
        C, N, ch = 1, 1, 1
        H, W, ts = 8, 8, 8

        # Place Gaussian off-center (not on pixel boundary) so spatial
        # asymmetry produces a non-zero gradient for sum(rc).
        means2d = mx.array(np.array([[[2.0, 2.0]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): list(range(N))})

        def loss_fn(m):
            rc, _ = rasterize_to_pixels_mlx(
                m, conics, colors, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_means = grad_fn(means2d)
        mx.eval(grad_means)

        grad_np = np.array(grad_means)
        # Verify gradient flows and is finite and non-trivial
        assert np.all(np.isfinite(grad_np)), f"means2d gradient not finite: {grad_np}"
        # For a Gaussian at (2.0, 2.0) in an 8x8 image, the total
        # contribution increases when moving toward the image center,
        # so the gradient should point toward higher pixel counts
        # (i.e., positive, pushing toward center at 4.0).
        assert grad_np[0, 0, 0] > 0.1, (
            f"x-gradient should be positive (toward center): {grad_np[0, 0, 0]}"
        )
        assert grad_np[0, 0, 1] > 0.1, (
            f"y-gradient should be positive (toward center): {grad_np[0, 0, 1]}"
        )


class TestDiffRasterGradientOpacities:
    """mx.grad flows to opacities."""

    def test_diff_raster_gradient_opacities(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.5, 0.2]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(o):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, conics, colors, o, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_opa = grad_fn(opacities)
        mx.eval(grad_opa)

        grad_np = np.array(grad_opa)
        assert grad_np.shape == opacities.shape
        # Increasing opacity increases output color => positive gradient
        assert grad_np[0, 0] > 0, (
            f"Opacity gradient should be positive, got {grad_np[0, 0]}"
        )

    def test_gradient_opacities_finite_diff(self):
        """Opacity gradient matches finite differences."""
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.5]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(o):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, conics, colors, o, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_ana = grad_fn(opacities)
        mx.eval(grad_ana)

        eps = 1e-4
        loss_plus = loss_fn(opacities + eps)
        loss_minus = loss_fn(opacities - eps)
        mx.eval(loss_plus, loss_minus)
        grad_fd = (np.array(loss_plus) - np.array(loss_minus)) / (2 * eps)

        np.testing.assert_allclose(
            np.array(grad_ana).sum(), grad_fd, rtol=1e-2,
            err_msg="Opacity gradient doesn't match finite differences",
        )


class TestDiffRasterGradientConics:
    """mx.grad flows to conics."""

    def test_diff_raster_gradient_conics(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.5, 0.2]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(co):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, co, colors, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_conics = grad_fn(conics)
        mx.eval(grad_conics)

        grad_np = np.array(grad_conics)
        assert grad_np.shape == conics.shape
        assert np.all(np.isfinite(grad_np)), f"Conic gradient not finite: {grad_np}"
        # For an isotropic Gaussian at center, increasing a or c (conic diagonal)
        # tightens the Gaussian, reducing total contribution => negative gradient
        assert grad_np[0, 0, 0] < 0, (
            f"Conic 'a' gradient should be negative (tighter = less total), "
            f"got {grad_np[0, 0, 0]}"
        )

    def test_gradient_conics_finite_diff(self):
        """Conic gradient matches finite differences."""
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[[1.0, 0.0, 1.0]]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.5]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, 1, 1, {(0, 0): [0]})

        def loss_fn(co):
            rc, _ = rasterize_to_pixels_mlx(
                means2d, co, colors, opacities, W, H, ts,
                isect_offsets, flatten_ids,
            )
            return mx.sum(rc)

        grad_fn = mx.grad(loss_fn)
        grad_ana = grad_fn(conics)
        mx.eval(grad_ana)

        # Check each conic component
        grad_ana_np = np.array(grad_ana)
        eps = 1e-4
        for i in range(3):
            delta = np.zeros((1, 1, 3), dtype=np.float32)
            delta[0, 0, i] = eps
            conics_plus = conics + mx.array(delta)
            conics_minus = conics - mx.array(delta)
            loss_p = loss_fn(conics_plus)
            loss_m = loss_fn(conics_minus)
            mx.eval(loss_p, loss_m)
            grad_fd = (np.array(loss_p) - np.array(loss_m)) / (2 * eps)
            np.testing.assert_allclose(
                grad_ana_np[0, 0, i], grad_fd, rtol=5e-2, atol=1e-5,
                err_msg=f"Conic gradient[{i}] doesn't match finite diff",
            )
