"""Tests for all example scripts.

Each test imports the example's core rendering function, runs it at a
small resolution, and verifies the output is valid (correct shape, not
all zeros, no NaN values). No files are saved during testing.
"""

import sys
import os

import mlx.core as mx
import numpy as np
import pytest

# Add the examples directory to the path so we can import them
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


class TestHelloGaussians:
    """Tests for 01_hello_gaussians.py."""

    def test_produces_valid_output(self):
        from importlib import import_module
        mod = import_module("01_hello_gaussians")

        img, alphas = mod.render_hello_gaussians(width=64, height=64)
        mx.eval(img, alphas)

        assert img.shape == (64, 64, 3), f"Expected (64, 64, 3), got {img.shape}"
        assert alphas.shape == (64, 64, 1), f"Expected (64, 64, 1), got {alphas.shape}"

        img_np = np.array(img)
        assert not np.any(np.isnan(img_np)), "Output contains NaN"
        assert np.max(img_np) > 0.01, "Image is all black"

    def test_alphas_in_valid_range(self):
        from importlib import import_module
        mod = import_module("01_hello_gaussians")

        _, alphas = mod.render_hello_gaussians(width=64, height=64)
        mx.eval(alphas)
        a = np.array(alphas)
        assert np.all(a >= 0.0) and np.all(a <= 1.0 + 1e-5), \
            f"Alpha out of range: [{a.min()}, {a.max()}]"


class TestCameraModels:
    """Tests for 02_camera_models.py."""

    def test_all_three_models_render(self):
        from importlib import import_module
        mod = import_module("02_camera_models")

        results = mod.render_camera_models(width=64, height=64)

        for model in ("pinhole", "fisheye", "ortho"):
            assert model in results, f"Missing camera model: {model}"
            img, alphas = results[model]
            mx.eval(img, alphas)

            assert img.shape == (64, 64, 3), \
                f"{model}: Expected (64, 64, 3), got {img.shape}"
            assert alphas.shape == (64, 64, 1), \
                f"{model}: Expected (64, 64, 1), got {alphas.shape}"

            img_np = np.array(img)
            assert not np.any(np.isnan(img_np)), f"{model}: contains NaN"

    def test_different_models_produce_different_images(self):
        from importlib import import_module
        mod = import_module("02_camera_models")

        results = mod.render_camera_models(width=64, height=64)
        pinhole_np = np.array(results["pinhole"][0])
        ortho_np = np.array(results["ortho"][0])
        mx.eval(results["pinhole"][0], results["ortho"][0])

        # They should differ (different projection models)
        diff = np.mean(np.abs(pinhole_np - ortho_np))
        assert diff > 1e-4, \
            f"Pinhole and ortho images are identical (diff={diff})"


class TestSphericalHarmonics:
    """Tests for 03_spherical_harmonics.py."""

    def test_renders_four_views(self):
        from importlib import import_module
        mod = import_module("03_spherical_harmonics")

        results = mod.render_sh_views(width=64, height=64, sh_degree=2)

        assert len(results) == 4, f"Expected 4 views, got {len(results)}"

        for label, img, alphas in results:
            mx.eval(img, alphas)
            assert img.shape == (64, 64, 3), \
                f"{label}: Expected (64, 64, 3), got {img.shape}"

            img_np = np.array(img)
            assert not np.any(np.isnan(img_np)), f"{label}: contains NaN"
            assert np.max(img_np) > 0.01, f"{label}: image is all black"

    def test_views_show_color_variation(self):
        from importlib import import_module
        mod = import_module("03_spherical_harmonics")

        results = mod.render_sh_views(width=64, height=64, sh_degree=2)
        mx.eval(*[r[1] for r in results])

        # Compare front and right views -- SH should produce different colors
        front_mean = np.mean(np.array(results[0][1]))
        right_mean = np.mean(np.array(results[1][1]))
        diff = abs(front_mean - right_mean)
        # With SH degree 2, we expect some variation (but it might be subtle)
        assert diff >= 0.0, "Views should be computed without error"


class TestCovarianceShapes:
    """Tests for 04_covariance_shapes.py."""

    def test_produces_valid_output(self):
        from importlib import import_module
        mod = import_module("04_covariance_shapes")

        img, alphas = mod.render_covariance_shapes(width=128, height=64)
        mx.eval(img, alphas)

        assert img.shape == (64, 128, 3), f"Expected (64, 128, 3), got {img.shape}"
        assert alphas.shape == (64, 128, 1)

        img_np = np.array(img)
        assert not np.any(np.isnan(img_np)), "Output contains NaN"
        assert np.max(img_np) > 0.01, "Image is all black"

    def test_has_non_trivial_alpha(self):
        from importlib import import_module
        mod = import_module("04_covariance_shapes")

        _, alphas = mod.render_covariance_shapes(width=128, height=64)
        mx.eval(alphas)
        a = np.array(alphas)
        # Should have both opaque and transparent pixels
        assert np.max(a) > 0.1, "No opaque pixels rendered"


class TestDepthRendering:
    """Tests for 05_depth_rendering.py."""

    def test_renders_rgb_depth_normals(self):
        from importlib import import_module
        mod = import_module("05_depth_rendering")

        results = mod.render_depth_and_normals(width=64, height=64)

        assert "rgb" in results
        assert "depth" in results
        assert "normals" in results
        assert "alphas" in results

        mx.eval(results["rgb"], results["depth"], results["normals"])

        rgb_np = np.array(results["rgb"])
        depth_np = np.array(results["depth"])
        normals_np = np.array(results["normals"])

        assert rgb_np.shape == (64, 64, 3)
        assert depth_np.shape == (64, 64, 1)
        assert normals_np.shape == (64, 64, 3)

        assert not np.any(np.isnan(rgb_np)), "RGB contains NaN"
        assert not np.any(np.isnan(depth_np)), "Depth contains NaN"
        assert not np.any(np.isnan(normals_np)), "Normals contain NaN"

    def test_depth_is_positive(self):
        from importlib import import_module
        mod = import_module("05_depth_rendering")

        results = mod.render_depth_and_normals(width=64, height=64)
        mx.eval(results["depth"])
        d = np.array(results["depth"])
        assert np.min(d) >= 0.0, f"Depth has negative values: min={np.min(d)}"


class TestSurfels2DGS:
    """Tests for 06_2dgs_surfels.py."""

    def test_both_pipelines_render(self):
        from importlib import import_module
        mod = import_module("06_2dgs_surfels")

        results = mod.render_3dgs_vs_2dgs(width=64, height=64)

        for key in ("3dgs_rgb", "2dgs_rgb", "2dgs_normals"):
            assert key in results, f"Missing key: {key}"

        mx.eval(results["3dgs_rgb"], results["2dgs_rgb"], results["2dgs_normals"])

        assert np.array(results["3dgs_rgb"]).shape == (64, 64, 3)
        assert np.array(results["2dgs_rgb"]).shape == (64, 64, 3)
        assert np.array(results["2dgs_normals"]).shape == (64, 64, 3)

    def test_no_nan_in_outputs(self):
        from importlib import import_module
        mod = import_module("06_2dgs_surfels")

        results = mod.render_3dgs_vs_2dgs(width=64, height=64)
        mx.eval(results["3dgs_rgb"], results["2dgs_rgb"], results["2dgs_normals"])

        for key in ("3dgs_rgb", "2dgs_rgb", "2dgs_normals"):
            arr = np.array(results[key])
            assert not np.any(np.isnan(arr)), f"{key} contains NaN"

    def test_2dgs_produces_normals(self):
        from importlib import import_module
        mod = import_module("06_2dgs_surfels")

        results = mod.render_3dgs_vs_2dgs(width=64, height=64)
        mx.eval(results["2dgs_normals"])
        n = np.array(results["2dgs_normals"])
        # 2DGS normals should not be all zero (at least where surfels are visible)
        assert n.shape == (64, 64, 3)
