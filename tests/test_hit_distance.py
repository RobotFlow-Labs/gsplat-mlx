"""Tests for hit-distance render modes and render-mode helper functions."""

import math

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.rendering import (
    rasterization,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_expected_depth,
    render_mode_has_hit_distance,
    render_mode_has_only_depth_channel,
)
from conftest import make_gaussians, make_camera_intrinsics, make_view_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_scene(N=50, width=32, height=32, seed=42):
    """Create a small scene for rendering tests."""
    g = make_gaussians(N=N, sh_degree=0, seed=seed)
    K = make_camera_intrinsics(width=width, height=height, fx=30.0, fy=30.0)
    viewmat = make_view_matrix(eye=(0.0, 0.0, 8.0), target=(0.0, 0.0, 0.0))
    return g, K[None, ...], viewmat[None, ...], width, height


# ---------------------------------------------------------------------------
# Test: render_mode helpers cover all 9 modes
# ---------------------------------------------------------------------------

class TestRenderModeHelpers:
    """Verify all 9 render modes return correct booleans from helpers."""

    ALL_MODES = ["RGB", "D", "ED", "RGB+D", "RGB+ED", "d", "Ed", "RGB+d", "RGB+Ed"]

    def test_has_color(self):
        expected = {"RGB", "RGB+D", "RGB+ED", "RGB+d", "RGB+Ed"}
        for m in self.ALL_MODES:
            assert render_mode_has_color(m) == (m in expected), f"has_color({m})"

    def test_has_depth(self):
        expected = {"D", "ED", "RGB+D", "RGB+ED"}
        for m in self.ALL_MODES:
            assert render_mode_has_depth(m) == (m in expected), f"has_depth({m})"

    def test_has_expected_depth(self):
        expected = {"ED", "RGB+ED", "Ed", "RGB+Ed"}
        for m in self.ALL_MODES:
            assert render_mode_has_expected_depth(m) == (m in expected), f"has_expected_depth({m})"

    def test_has_hit_distance(self):
        expected = {"d", "Ed", "RGB+d", "RGB+Ed"}
        for m in self.ALL_MODES:
            assert render_mode_has_hit_distance(m) == (m in expected), f"has_hit_distance({m})"

    def test_has_only_depth_channel(self):
        expected = {"D", "ED", "d", "Ed"}
        for m in self.ALL_MODES:
            assert render_mode_has_only_depth_channel(m) == (m in expected), f"has_only_depth({m})"


# ---------------------------------------------------------------------------
# Test: render_mode="d" produces a single depth channel
# ---------------------------------------------------------------------------

def test_render_mode_d():
    g, Ks, viewmats, W, H = _setup_scene()
    imgs, alphas, info = rasterization(
        g["means"], g["quats"], g["scales"],
        mx.sigmoid(g["opacities"]), g["sh_coeffs"],
        viewmats, Ks, W, H,
        sh_degree=0, render_mode="d",
    )
    mx.eval(imgs, alphas)
    assert imgs.shape == (1, H, W, 1), f"Expected (1,{H},{W},1), got {imgs.shape}"
    # Hit distances should be non-negative
    assert float(mx.min(imgs)) >= 0.0, "Hit distances must be >= 0"


# ---------------------------------------------------------------------------
# Test: render_mode="Ed" produces expected hit distance
# ---------------------------------------------------------------------------

def test_render_mode_Ed():
    g, Ks, viewmats, W, H = _setup_scene()
    imgs, alphas, info = rasterization(
        g["means"], g["quats"], g["scales"],
        mx.sigmoid(g["opacities"]), g["sh_coeffs"],
        viewmats, Ks, W, H,
        sh_degree=0, render_mode="Ed",
    )
    mx.eval(imgs, alphas)
    assert imgs.shape == (1, H, W, 1)
    # Expected hit distance should be non-negative
    assert float(mx.min(imgs)) >= 0.0


# ---------------------------------------------------------------------------
# Test: render_mode="RGB+d" produces 4 channels (RGB + hit distance)
# ---------------------------------------------------------------------------

def test_render_mode_RGB_d():
    g, Ks, viewmats, W, H = _setup_scene()
    imgs, alphas, info = rasterization(
        g["means"], g["quats"], g["scales"],
        mx.sigmoid(g["opacities"]), g["sh_coeffs"],
        viewmats, Ks, W, H,
        sh_degree=0, render_mode="RGB+d",
    )
    mx.eval(imgs, alphas)
    assert imgs.shape == (1, H, W, 4), f"Expected 4 channels, got {imgs.shape[-1]}"


# ---------------------------------------------------------------------------
# Test: hit distance >= z-depth for off-axis Gaussians
# ---------------------------------------------------------------------------

def test_hit_distance_vs_depth():
    """Hit distance (ray distance) should be >= z-depth for all Gaussians.

    For a Gaussian at camera-space position (x, y, z), the z-depth is z
    while the hit distance is sqrt(x^2 + y^2 + z^2) >= z (assuming z > 0).
    """
    g, Ks, viewmats, W, H = _setup_scene(N=100, seed=123)
    opacities = mx.sigmoid(g["opacities"])

    # Render z-depth
    imgs_D, alphas_D, _ = rasterization(
        g["means"], g["quats"], g["scales"], opacities, g["sh_coeffs"],
        viewmats, Ks, W, H, sh_degree=0, render_mode="D",
    )
    # Render hit distance
    imgs_d, alphas_d, _ = rasterization(
        g["means"], g["quats"], g["scales"], opacities, g["sh_coeffs"],
        viewmats, Ks, W, H, sh_degree=0, render_mode="d",
    )
    mx.eval(imgs_D, imgs_d, alphas_D, alphas_d)

    D_np = np.array(imgs_D).ravel()
    d_np = np.array(imgs_d).ravel()

    # Only compare pixels that have meaningful alpha (were actually rendered)
    alpha_np = np.array(alphas_D).ravel()
    mask = alpha_np > 0.01
    if mask.sum() > 0:
        # hit distance >= z-depth (within numerical tolerance)
        assert np.all(d_np[mask] >= D_np[mask] - 1e-3), (
            "Hit distance should be >= z-depth for visible pixels"
        )


# ---------------------------------------------------------------------------
# Test: RGB+Ed produces 4 channels
# ---------------------------------------------------------------------------

def test_render_mode_RGB_Ed():
    g, Ks, viewmats, W, H = _setup_scene()
    imgs, alphas, info = rasterization(
        g["means"], g["quats"], g["scales"],
        mx.sigmoid(g["opacities"]), g["sh_coeffs"],
        viewmats, Ks, W, H,
        sh_degree=0, render_mode="RGB+Ed",
    )
    mx.eval(imgs, alphas)
    assert imgs.shape == (1, H, W, 4)
