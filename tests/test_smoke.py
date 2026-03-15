"""Smoke tests for gsplat-mlx development environment.

These tests validate that the environment is correctly configured:
- MLX is importable and functional
- Metal GPU backend is available
- Package structure is correct
- Test infrastructure works
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from conftest import check_all_close, make_camera_intrinsics, make_gaussians


# ---------------------------------------------------------------------------
# MLX environment tests
# ---------------------------------------------------------------------------

class TestMLXEnvironment:
    """Verify MLX is installed and functional."""

    def test_mlx_available(self):
        """MLX can be imported."""
        import mlx.core as mx_
        assert mx_ is not None

    def test_metal_backend(self):
        """MLX default device is Metal GPU."""
        device = mx.default_device()
        assert device == mx.gpu, (
            f"Expected Metal GPU backend, got {device}. "
            "Ensure you are running on Apple Silicon."
        )

    def test_mlx_basic_ops(self):
        """Basic MLX operations: array creation, matmul, eval."""
        a = mx.ones((3, 4))
        b = mx.ones((4, 5))
        c = a @ b
        mx.eval(c)
        assert c.shape == (3, 5)
        # Each element should be 4.0 (dot product of ones)
        expected = np.full((3, 5), 4.0, dtype=np.float32)
        np.testing.assert_allclose(np.array(c), expected)

    def test_mlx_custom_function(self):
        """@mx.custom_function decorator works."""

        @mx.custom_function
        def my_relu(x):
            return mx.maximum(x, 0.0)

        @my_relu.vjp
        def my_relu_vjp(primals, cotangent, output):
            (x,) = primals
            return (cotangent * mx.array(x > 0, dtype=cotangent.dtype),)

        x = mx.array([-1.0, 0.0, 1.0, 2.0])
        y = my_relu(x)
        mx.eval(y)
        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(np.array(y), expected)

    def test_mlx_grad(self):
        """mx.grad computes gradients correctly."""

        def f(x):
            return mx.sum(x ** 2)

        x = mx.array([1.0, 2.0, 3.0])
        grad_fn = mx.grad(f)
        g = grad_fn(x)
        mx.eval(g)
        # d/dx sum(x^2) = 2x
        expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(np.array(g), expected)


# ---------------------------------------------------------------------------
# Package structure tests
# ---------------------------------------------------------------------------

class TestPackageStructure:
    """Verify package imports and version."""

    def test_package_import(self):
        """gsplat_mlx package can be imported."""
        import gsplat_mlx
        assert gsplat_mlx is not None

    def test_version(self):
        """Package version is 0.1.0."""
        import gsplat_mlx
        assert gsplat_mlx.__version__ == "0.1.0"

    def test_core_import(self):
        """Core constants module can be imported."""
        from gsplat_mlx.core import constants
        assert constants is not None

    def test_constants_values(self):
        """All four core constants have correct values."""
        from gsplat_mlx.core.constants import (
            ALPHA_THRESHOLD,
            MAX_ALPHA,
            MAX_KERNEL_DENSITY_CUTOFF,
            TRANSMITTANCE_THRESHOLD,
        )
        assert ALPHA_THRESHOLD == pytest.approx(1.0 / 255.0)
        assert MAX_ALPHA == pytest.approx(0.99)
        assert TRANSMITTANCE_THRESHOLD == pytest.approx(1e-4)
        assert MAX_KERNEL_DENSITY_CUTOFF == pytest.approx(0.0113, abs=1e-4)

    def test_constants_invariant(self):
        """TRANSMITTANCE_THRESHOLD is approximately (1 - MAX_ALPHA)^2."""
        from gsplat_mlx.core.constants import MAX_ALPHA, TRANSMITTANCE_THRESHOLD
        expected = (1.0 - MAX_ALPHA) ** 2
        assert TRANSMITTANCE_THRESHOLD == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Test infrastructure tests
# ---------------------------------------------------------------------------

class TestInfrastructure:
    """Verify test helpers and fixtures work correctly."""

    def test_fixture_gaussians(self, gaussians_small):
        """make_gaussians produces valid shapes and dtypes."""
        g = gaussians_small
        N = 10
        K = 16  # (3+1)^2

        assert g["means"].shape == (N, 3)
        assert g["quats"].shape == (N, 4)
        assert g["scales"].shape == (N, 3)
        assert g["opacities"].shape == (N,)
        assert g["sh_coeffs"].shape == (N, K, 3)

        # All should be float32
        for key in g:
            assert g[key].dtype == mx.float32, f"{key} has dtype {g[key].dtype}"

        # Quaternions should be unit length
        mx.eval(g["quats"])
        quat_norms = np.linalg.norm(np.array(g["quats"]), axis=-1)
        np.testing.assert_allclose(quat_norms, 1.0, atol=1e-5)

    def test_fixture_camera(self, camera_640x480):
        """make_camera_intrinsics produces valid K matrix."""
        K = camera_640x480
        assert K.shape == (3, 3)
        assert K.dtype == mx.float32

        mx.eval(K)
        K_np = np.array(K)

        # Check structure: fx at [0,0], fy at [1,1], 1 at [2,2]
        assert K_np[0, 0] == pytest.approx(500.0)
        assert K_np[1, 1] == pytest.approx(500.0)
        assert K_np[2, 2] == pytest.approx(1.0)
        # Principal point at center
        assert K_np[0, 2] == pytest.approx(320.0)
        assert K_np[1, 2] == pytest.approx(240.0)
        # Off-diagonals should be zero
        assert K_np[0, 1] == pytest.approx(0.0)
        assert K_np[1, 0] == pytest.approx(0.0)

    def test_check_all_close_pass(self):
        """check_all_close passes for identical arrays."""
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0, 2.0, 3.0])
        # Should not raise
        check_all_close(a, b)

    def test_check_all_close_fail(self):
        """check_all_close raises AssertionError for different arrays."""
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([1.0, 2.0, 999.0])
        with pytest.raises(AssertionError, match="Arrays not close"):
            check_all_close(a, b)
