"""Tests for the Tier-1 (NumPy reference) pixel rasterizer.

Each test builds minimal synthetic inputs (means2d, conics, colors, opacities,
isect_offsets, flatten_ids) by hand so we can reason about exact expected
pixel values without needing the upstream intersection / projection pipeline.
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
        C: number of cameras (only C=1 supported here for simplicity).
        tile_height, tile_width: tile grid dimensions.
        tile_gaussians: dict mapping ``(ty, tx)`` -> list of Gaussian ids.

    Returns:
        isect_offsets: mx.array [C, tile_H, tile_W]
        flatten_ids:   mx.array [n_isects]
    """
    # Build flat list in tile-raster order and cumulative offsets.
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
    """Return a tight conic so the Gaussian is concentrated.

    ``sigma`` is the standard deviation in pixels; the conic entries are the
    inverse covariance diagonal: ``1/sigma^2``.
    """
    inv_var = 1.0 / (sigma * sigma)
    return np.array([inv_var, 0.0, inv_var], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputShapes:
    """test_output_shapes: verify [C, H, W, channels] and [C, H, W, 1]."""

    def test_output_shapes(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 8, 8, 8
        tH, tW = 1, 1

        means2d = mx.array(np.array([[[4.0, 4.0]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.ones((C, N, ch), dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        assert rc.shape == (C, H, W, ch), f"render_colors shape {rc.shape}"
        assert ra.shape == (C, H, W, 1), f"render_alphas shape {ra.shape}"


class TestSingleGaussianCenter:
    """test_single_gaussian_center: one Gaussian at image center -> bright spot."""

    def test_single_gaussian_center(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 16, 16, 16  # single tile
        tH, tW = 1, 1

        cx, cy = W / 2.0, H / 2.0
        means2d = mx.array(np.array([[[cx, cy]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.95]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        # The pixel closest to center should have the highest red value.
        rc_np = np.array(rc)
        center_px = int(cy)  # row
        center_py = int(cx)  # col
        center_color = rc_np[0, center_px, center_py]
        assert center_color[0] > 0.3, "Center pixel should be bright red"

        # Corners should be dimmer.
        corner = rc_np[0, 0, 0]
        assert center_color[0] > corner[0], "Center should be brighter than corner"


class TestSingleGaussianValues:
    """test_single_gaussian_values: verify exact pixel values against manual computation."""

    def test_single_gaussian_values(self):
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        # Place Gaussian at pixel (1.5, 1.5) -- center of pixel (1, 1).
        means2d = mx.array(np.array([[[1.5, 1.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacity_val = 0.8
        opacities = mx.array(np.array([[opacity_val]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        rc_np = np.array(rc)
        ra_np = np.array(ra)

        # At pixel (1, 1) with center (1.5, 1.5): dx=0, dy=0 -> sigma=0
        # alpha = min(0.8 * exp(0), MAX_ALPHA) = 0.8
        # weight = 1.0 * 0.8 = 0.8
        # color = 0.8 * 1.0 = 0.8
        expected_color = opacity_val * 1.0  # T=1, alpha=0.8
        expected_alpha = opacity_val
        np.testing.assert_allclose(
            rc_np[0, 1, 1, 0], expected_color, atol=1e-5,
            err_msg="Center pixel colour mismatch",
        )
        np.testing.assert_allclose(
            ra_np[0, 1, 1, 0], expected_alpha, atol=1e-5,
            err_msg="Center pixel alpha mismatch",
        )

        # At pixel (0, 0) with center (0.5, 0.5): dx=-1, dy=-1
        # sigma = 0.5*(1*1 + 1*1) + 0 = 1.0
        # alpha = min(0.8 * exp(-1), MAX_ALPHA) = 0.8 * 0.36788 = 0.29430
        expected_alpha_corner = opacity_val * math.exp(-1.0)
        expected_color_corner = expected_alpha_corner * 1.0  # T=1
        np.testing.assert_allclose(
            rc_np[0, 0, 0, 0], expected_color_corner, atol=1e-5,
            err_msg="Corner pixel colour mismatch",
        )


class TestTwoGaussiansDepthOrder:
    """test_two_gaussians_depth_order: closer Gaussian occludes farther."""

    def test_two_gaussians_depth_order(self):
        C, N, ch = 1, 2, 3
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        # Both Gaussians at the same pixel center.
        means2d = mx.array(np.array([[[2.5, 2.5], [2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([
            [_make_identity_conic(), _make_identity_conic()]
        ], dtype=np.float32))
        # First: red, second: blue.
        colors = mx.array(np.array([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ], dtype=np.float32))
        opacities = mx.array(np.array([[0.9, 0.9]], dtype=np.float32))
        # Depth order: Gaussian 0 first (closer), Gaussian 1 second.
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0, 1]})

        rc, _ = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc)
        rc_np = np.array(rc)

        # At center pixel (2, 2): first Gaussian contributes mostly red,
        # second adds some blue with reduced transmittance.
        red = rc_np[0, 2, 2, 0]
        blue = rc_np[0, 2, 2, 2]
        assert red > blue, (
            f"Closer red Gaussian should dominate: red={red:.4f}, blue={blue:.4f}"
        )


class TestBackgroundBlending:
    """test_background_blending: transparent areas show background."""

    def test_background_blending(self):
        C, N, ch = 1, 1, 3
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        # Place Gaussian at top-left corner with tight conic.
        means2d = mx.array(np.array([[[0.5, 0.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_tight_conic(0.5)]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[0.9]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        bg = mx.array(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))  # green

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
            backgrounds=bg,
        )
        mx.eval(rc, ra)
        rc_np = np.array(rc)

        # Far-away pixel (3, 3) should be mostly green background.
        green_val = rc_np[0, 3, 3, 1]
        red_val = rc_np[0, 3, 3, 0]
        assert green_val > 0.5, f"Far pixel should show green bg: {green_val:.4f}"
        assert green_val > red_val, "Background should dominate far pixels"


class TestEmptyImage:
    """test_empty_image: no Gaussians -> all zeros (or background)."""

    def test_empty_no_background(self):
        C, N, ch = 1, 0, 3
        H, W, ts = 8, 8, 8
        tH, tW = 1, 1

        means2d = mx.array(np.zeros((C, 0, 2), dtype=np.float32))
        conics = mx.array(np.zeros((C, 0, 3), dtype=np.float32))
        colors = mx.array(np.zeros((C, 0, ch), dtype=np.float32))
        opacities = mx.array(np.zeros((C, 0), dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        np.testing.assert_array_equal(np.array(rc), 0.0)
        np.testing.assert_array_equal(np.array(ra), 0.0)

    def test_empty_with_background(self):
        C, N, ch = 1, 0, 3
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        means2d = mx.array(np.zeros((C, 0, 2), dtype=np.float32))
        conics = mx.array(np.zeros((C, 0, 3), dtype=np.float32))
        colors = mx.array(np.zeros((C, 0, ch), dtype=np.float32))
        opacities = mx.array(np.zeros((C, 0), dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {})
        bg = mx.array(np.array([[0.2, 0.4, 0.6]], dtype=np.float32))

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
            backgrounds=bg,
        )
        mx.eval(rc, ra)

        # Every pixel should equal the background.
        rc_np = np.array(rc)
        for py in range(H):
            for px in range(W):
                np.testing.assert_allclose(
                    rc_np[0, py, px], [0.2, 0.4, 0.6], atol=1e-6,
                )


class TestFullOpacity:
    """test_full_opacity: opacity=1 Gaussian -> fully opaque (clamped to MAX_ALPHA)."""

    def test_full_opacity(self):
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[1.0]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        # At center: sigma=0, alpha = min(1.0, MAX_ALPHA) = MAX_ALPHA = 0.99
        ra_np = np.array(ra)
        np.testing.assert_allclose(
            ra_np[0, 2, 2, 0], MAX_ALPHA, atol=1e-5,
            err_msg="Full-opacity Gaussian should hit MAX_ALPHA at center",
        )


class TestAlphaThreshold:
    """test_alpha_threshold: very low alpha Gaussians are skipped."""

    def test_alpha_threshold(self):
        C, N, ch = 1, 1, 1
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        # Use a very low opacity so that alpha < ALPHA_THRESHOLD everywhere.
        very_low_opacity = ALPHA_THRESHOLD * 0.5
        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors = mx.array(np.array([[[1.0]]], dtype=np.float32))
        opacities = mx.array(np.array([[very_low_opacity]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        # All pixels should be zero since the Gaussian is below threshold.
        np.testing.assert_array_equal(np.array(rc), 0.0)
        np.testing.assert_array_equal(np.array(ra), 0.0)


class TestTransmittanceCutoff:
    """test_transmittance_cutoff: many opaque Gaussians -> early termination."""

    def test_transmittance_cutoff(self):
        C, ch = 1, 1
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        # Stack 50 high-opacity Gaussians at the same location.
        N = 50
        center = np.array([[2.5, 2.5]], dtype=np.float32)
        means2d_np = np.tile(center, (N, 1))[np.newaxis]  # [1, N, 2]
        conics_np = np.tile(_make_identity_conic(), (1, N, 1))
        colors_np = np.ones((1, N, ch), dtype=np.float32)
        opacities_np = np.full((1, N), 0.9, dtype=np.float32)

        means2d = mx.array(means2d_np)
        conics = mx.array(conics_np)
        colors = mx.array(colors_np)
        opacities = mx.array(opacities_np)

        # All 50 Gaussians in the single tile.
        isect_offsets, flatten_ids = _simple_isect(
            C, tH, tW, {(0, 0): list(range(N))}
        )

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        # At center pixel: should be nearly fully opaque.
        ra_np = np.array(ra)
        center_alpha = ra_np[0, 2, 2, 0]
        assert center_alpha > 1.0 - 1e-3, (
            f"Many opaque Gaussians should saturate alpha: {center_alpha:.6f}"
        )


class TestMultiChannel:
    """test_multi_channel: channels > 3 works."""

    def test_multi_channel(self):
        C, N = 1, 1
        ch = 7  # arbitrary non-standard channel count
        H, W, ts = 4, 4, 4
        tH, tW = 1, 1

        means2d = mx.array(np.array([[[2.5, 2.5]]], dtype=np.float32))
        conics = mx.array(np.array([[_make_identity_conic()]], dtype=np.float32))
        colors_np = np.arange(ch, dtype=np.float32).reshape(1, 1, ch)
        colors = mx.array(colors_np)
        opacities = mx.array(np.array([[0.8]], dtype=np.float32))
        isect_offsets, flatten_ids = _simple_isect(C, tH, tW, {(0, 0): [0]})

        rc, ra = rasterize_to_pixels(
            means2d, conics, colors, opacities, W, H, ts,
            isect_offsets, flatten_ids,
        )
        mx.eval(rc, ra)

        assert rc.shape == (C, H, W, ch)
        assert ra.shape == (C, H, W, 1)

        # Center pixel (sigma=0): alpha=0.8, weight=0.8
        rc_np = np.array(rc)
        expected = 0.8 * colors_np[0, 0]
        np.testing.assert_allclose(
            rc_np[0, 2, 2], expected, atol=1e-5,
            err_msg="Multi-channel compositing mismatch at center",
        )
