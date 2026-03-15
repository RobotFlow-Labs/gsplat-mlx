"""Tests for Gaussian relocation (MCMC paper).

Covers:
- Identity relocation (ratio=1 -> no change)
- Split relocation (ratio=2 -> smaller opacity, adjusted scales)
- Output shape preservation
- Binomial coefficient correctness (Pascal's triangle)
"""

import numpy as np
import pytest

import mlx.core as mx

from gsplat_mlx.relocation import compute_binomial_coefficients, compute_relocation


class TestBinomialCoefficients:
    """Tests for compute_binomial_coefficients."""

    def test_binomial_coefficients_small(self):
        """Verify Pascal's triangle values for n_max=5."""
        binoms = compute_binomial_coefficients(5)
        mx.eval(binoms)
        b = np.array(binoms)

        # Row 0: C(0,0) = 1
        assert b[0, 0] == 1.0
        # Row 1: C(1,0)=1, C(1,1)=1
        assert b[1, 0] == 1.0
        assert b[1, 1] == 1.0
        # Row 2: C(2,0)=1, C(2,1)=2, C(2,2)=1
        assert b[2, 0] == 1.0
        assert b[2, 1] == 2.0
        assert b[2, 2] == 1.0
        # Row 3: 1, 3, 3, 1
        assert b[3, 0] == 1.0
        assert b[3, 1] == 3.0
        assert b[3, 2] == 3.0
        assert b[3, 3] == 1.0
        # Row 4: 1, 4, 6, 4, 1
        assert b[4, 0] == 1.0
        assert b[4, 1] == 4.0
        assert b[4, 2] == 6.0
        assert b[4, 3] == 4.0
        assert b[4, 4] == 1.0

    def test_binomial_shape(self):
        """Output shape is [n_max, n_max]."""
        binoms = compute_binomial_coefficients(10)
        assert binoms.shape == (10, 10)

    def test_binomial_zeros_upper_triangle(self):
        """C(n, k) = 0 for k > n."""
        binoms = compute_binomial_coefficients(6)
        mx.eval(binoms)
        b = np.array(binoms)
        for n in range(6):
            for k in range(n + 1, 6):
                assert b[n, k] == 0.0, f"C({n},{k}) should be 0, got {b[n, k]}"


class TestComputeRelocation:
    """Tests for compute_relocation."""

    def _make_inputs(self, N=10, n_max=10):
        """Create test inputs."""
        np.random.seed(42)
        opacities = mx.array(np.random.uniform(0.1, 0.9, (N,)).astype(np.float32))
        scales = mx.array(np.random.uniform(0.01, 1.0, (N, 3)).astype(np.float32))
        binoms = compute_binomial_coefficients(n_max)
        return opacities, scales, binoms

    def test_relocation_identity(self):
        """ratio=1 should yield (approximately) the same opacity and scale."""
        opacities, scales, binoms = self._make_inputs(N=5, n_max=10)
        ratios = mx.ones((5,), dtype=mx.int32)

        new_opa, new_scales = compute_relocation(opacities, scales, ratios, binoms)
        mx.eval(new_opa, new_scales)

        np.testing.assert_allclose(
            np.array(new_opa), np.array(opacities), atol=1e-5,
            err_msg="ratio=1 should preserve opacity"
        )
        np.testing.assert_allclose(
            np.array(new_scales), np.array(scales), atol=1e-5,
            err_msg="ratio=1 should preserve scales"
        )

    def test_relocation_split(self):
        """ratio=2 should yield smaller opacity."""
        opacities, scales, binoms = self._make_inputs(N=5, n_max=10)
        ratios = mx.full((5,), 2, dtype=mx.int32)

        new_opa, new_scales = compute_relocation(opacities, scales, ratios, binoms)
        mx.eval(new_opa, new_scales)

        # New opacity should be less than original for all Gaussians
        orig = np.array(opacities)
        new = np.array(new_opa)
        assert np.all(new < orig), (
            f"ratio=2 should yield smaller opacity: orig={orig}, new={new}"
        )
        # New opacity should still be positive
        assert np.all(new > 0), f"New opacity should be positive: {new}"

    def test_relocation_shape(self):
        """Output shapes must match input shapes."""
        N = 8
        opacities, scales, binoms = self._make_inputs(N=N, n_max=10)
        ratios = mx.full((N,), 3, dtype=mx.int32)

        new_opa, new_scales = compute_relocation(opacities, scales, ratios, binoms)
        mx.eval(new_opa, new_scales)

        assert new_opa.shape == (N,), f"Expected ({N},), got {new_opa.shape}"
        assert new_scales.shape == (N, 3), f"Expected ({N}, 3), got {new_scales.shape}"

    def test_relocation_high_ratio(self):
        """High ratio should yield very small opacity."""
        N = 3
        opacities = mx.array([0.8, 0.5, 0.3], dtype=mx.float32)
        scales = mx.ones((N, 3), dtype=mx.float32)
        binoms = compute_binomial_coefficients(20)
        ratios = mx.full((N,), 10, dtype=mx.int32)

        new_opa, new_scales = compute_relocation(opacities, scales, ratios, binoms)
        mx.eval(new_opa, new_scales)

        # With ratio=10, opacity should be much smaller
        assert np.all(np.array(new_opa) < np.array(opacities))
