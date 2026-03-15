"""Tests for spherical harmonics evaluation (PRD-04).

Covers:
- SH basis evaluation at known directions (exact test vectors)
- Forward pass for degrees 0-4
- Auto-normalization of non-unit directions
- Batch dimensions
- VJP / gradient flow for coeffs and dirs
- Finite-difference gradient checks
- Cross-framework comparison with PyTorch (requires torch)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.core.spherical_harmonics import (
    _eval_sh_bases_fast,
    spherical_harmonics,
)
from conftest import check_all_close

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===================================================================
# SH Basis Evaluation Tests
# ===================================================================


class TestEvalSHBasesFast:
    """Unit tests for _eval_sh_bases_fast."""

    def test_sh_degree0(self):
        """Degree 0: single constant basis value."""
        dirs = mx.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bases = _eval_sh_bases_fast(1, dirs)
        mx.eval(bases)

        assert bases.shape == (3, 1)
        expected = 0.2820947917738781
        np.testing.assert_allclose(
            np.array(bases), expected, atol=1e-7,
            err_msg="Degree 0 should be constant"
        )

    def test_sh_degree1(self):
        """Degree 1: 4 basis functions, linear in x, y, z."""
        dirs = mx.array([[1.0, 0.0, 0.0]])
        bases = _eval_sh_bases_fast(4, dirs)
        mx.eval(bases)
        assert bases.shape == (1, 4)

        b = np.array(bases)[0]
        fTmpA = -0.48860251190292
        np.testing.assert_allclose(b[0], 0.2820947917738781, atol=1e-7)
        np.testing.assert_allclose(b[1], 0.0, atol=1e-7)  # fTmpA * y
        np.testing.assert_allclose(b[2], 0.0, atol=1e-7)  # -fTmpA * z
        np.testing.assert_allclose(b[3], fTmpA, atol=1e-7)  # fTmpA * x

    def test_sh_degree2(self):
        """Degree 2: 9 basis functions."""
        dirs = mx.array([[0.0, 0.0, 1.0]])
        bases = _eval_sh_bases_fast(9, dirs)
        mx.eval(bases)
        assert bases.shape == (1, 9)

    def test_sh_degree3(self):
        """Degree 3: 16 basis functions."""
        dirs = mx.array([[0.0, 0.0, 1.0]])
        bases = _eval_sh_bases_fast(16, dirs)
        mx.eval(bases)
        assert bases.shape == (1, 16)

    def test_sh_bases_z_axis(self):
        """Verify exact basis values for z-axis direction (0, 0, 1)."""
        dirs = mx.array([[0.0, 0.0, 1.0]])
        bases = _eval_sh_bases_fast(9, dirs)
        mx.eval(bases)

        b = np.array(bases)[0]
        expected = np.array([
            0.2820947917738781,   # 0: constant
            0.0,                  # 1: fTmpA * y = 0
            0.48860251190292,     # 2: -fTmpA * z = 0.48860...
            0.0,                  # 3: fTmpA * x = 0
            0.0,                  # 4: fTmpA * 2xy = 0
            0.0,                  # 5: fTmpB * y = 0
            0.63078313050504,     # 6: 0.9461... * 1 - 0.3153...
            0.0,                  # 7: fTmpB * x = 0
            0.0,                  # 8: fTmpA * (x^2 - y^2) = 0
        ])
        np.testing.assert_allclose(b, expected, atol=1e-7)

    def test_sh_bases_x_axis(self):
        """Verify exact basis values for x-axis direction (1, 0, 0)."""
        dirs = mx.array([[1.0, 0.0, 0.0]])
        bases = _eval_sh_bases_fast(9, dirs)
        mx.eval(bases)

        b = np.array(bases)[0]
        expected = np.array([
            0.2820947917738781,   # 0: constant
            0.0,                  # 1: fTmpA * 0
            0.0,                  # 2: -fTmpA * 0
            -0.48860251190292,    # 3: fTmpA * 1
            0.0,                  # 4: fTmpA * 0
            0.0,                  # 5: 0
            -0.3153915652525201,  # 6: 0.9461... * 0 - 0.3153...
            0.0,                  # 7: 0
            0.5462742152960395,   # 8: fTmpA * (1 - 0)
        ])
        np.testing.assert_allclose(b, expected, atol=1e-7)

    def test_sh_bases_diagonal(self):
        """Verify basis values for diagonal direction (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))."""
        s = 1.0 / math.sqrt(3.0)
        dirs = mx.array([[s, s, s]])
        bases = _eval_sh_bases_fast(9, dirs)
        mx.eval(bases)

        b = np.array(bases)[0]
        fTmpA_d1 = -0.48860251190292
        # Check key values
        np.testing.assert_allclose(b[0], 0.2820947917738781, atol=1e-7)
        np.testing.assert_allclose(b[1], fTmpA_d1 * s, atol=1e-7)
        np.testing.assert_allclose(b[2], -fTmpA_d1 * s, atol=1e-7)
        np.testing.assert_allclose(b[3], fTmpA_d1 * s, atol=1e-7)
        # Index 4: 0.5462742152960395 * 2 * s * s = 0.5462742152960395 * 2/3
        np.testing.assert_allclose(b[4], 0.5462742152960395 * 2.0 / 3.0, atol=1e-7)
        # Index 6: 0.9461746957575601 * 1/3 - 0.3153915652525201 = 0
        np.testing.assert_allclose(b[6], 0.0, atol=1e-7)

    def test_sh_degree4_shape(self):
        """Degree 4: 25 basis functions, verify shape."""
        dirs = mx.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        bases = _eval_sh_bases_fast(25, dirs)
        mx.eval(bases)
        assert bases.shape == (2, 25)


# ===================================================================
# Forward Tests
# ===================================================================


class TestSphericalHarmonics:
    """Forward pass tests for spherical_harmonics."""

    def test_sh_degree0_value(self):
        """Degree 0: output = C0 * coeffs."""
        dirs = mx.array([[0.0, 0.0, 1.0]])
        coeffs = mx.array([[[1.0, 0.5, 0.25]]])  # [1, 1, 3]
        result = spherical_harmonics(0, dirs, coeffs)
        mx.eval(result)

        C0 = 0.2820947917738781
        expected = np.array([[C0 * 1.0, C0 * 0.5, C0 * 0.25]])
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_sh_degree0_direction_invariant(self):
        """Degree 0 gives same output regardless of direction."""
        np.random.seed(123)
        dirs_np = np.random.randn(100, 3).astype(np.float32)
        dirs = mx.array(dirs_np)
        # Same coefficients for all directions
        coeffs = mx.broadcast_to(
            mx.array([[[2.0, 1.0, 0.5]]]),  # [1, 1, 3]
            (100, 1, 3),
        )
        result = spherical_harmonics(0, dirs, coeffs)
        mx.eval(result)

        result_np = np.array(result)
        # All rows should be the same
        for i in range(1, 100):
            np.testing.assert_allclose(result_np[i], result_np[0], atol=1e-7)

    def test_sh_degree1(self):
        """Degree 1: verify direction dependence."""
        # Opposite x-directions should give different colors with x-dependent coeff
        coeffs_np = np.zeros((1, 4, 3), dtype=np.float32)
        coeffs_np[0, 0, :] = [1.0, 1.0, 1.0]  # DC
        coeffs_np[0, 3, :] = [1.0, 0.0, 0.0]  # Y_1^1 (x-direction)

        coeffs = mx.array(coeffs_np)

        d_pos = mx.array([[1.0, 0.0, 0.0]])
        d_neg = mx.array([[-1.0, 0.0, 0.0]])

        r_pos = spherical_harmonics(1, d_pos, coeffs)
        r_neg = spherical_harmonics(1, d_neg, coeffs)
        mx.eval(r_pos, r_neg)

        r_pos_np = np.array(r_pos)
        r_neg_np = np.array(r_neg)

        # R channel should differ (x contributes), G and B should be same
        assert not np.allclose(r_pos_np[0, 0], r_neg_np[0, 0], atol=1e-5)
        np.testing.assert_allclose(r_pos_np[0, 1], r_neg_np[0, 1], atol=1e-6)

    def test_sh_degree1_exact(self):
        """Degree 1: verify exact values from PRD test vector 5."""
        coeffs_np = np.zeros((1, 4, 3), dtype=np.float32)
        coeffs_np[0, 0, :] = [1.0, 1.0, 1.0]
        coeffs_np[0, 3, :] = [1.0, 0.0, 0.0]
        coeffs = mx.array(coeffs_np)

        dirs = mx.array([[1.0, 0.0, 0.0]])
        result = spherical_harmonics(1, dirs, coeffs)
        mx.eval(result)

        C0 = 0.2820947917738781
        fTmpA = -0.48860251190292
        expected_r = C0 * 1.0 + fTmpA * 1.0  # DC + Y_1^1 * coeff
        expected_g = C0 * 1.0
        expected_b = C0 * 1.0

        result_np = np.array(result)[0]
        np.testing.assert_allclose(result_np[0], expected_r, atol=1e-6)
        np.testing.assert_allclose(result_np[1], expected_g, atol=1e-6)
        np.testing.assert_allclose(result_np[2], expected_b, atol=1e-6)

    def test_sh_degree2(self):
        """Degree 2: 9 basis functions produce correct shape and no NaN."""
        np.random.seed(42)
        N = 50
        dirs = mx.array(np.random.randn(N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(N, 9, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(2, dirs, coeffs)
        mx.eval(result)
        assert result.shape == (N, 3)
        assert not np.any(np.isnan(np.array(result)))

    def test_sh_degree3(self):
        """Degree 3: 16 basis functions (most common 3DGS setting)."""
        np.random.seed(42)
        N = 50
        dirs = mx.array(np.random.randn(N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(N, 16, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(3, dirs, coeffs)
        mx.eval(result)
        assert result.shape == (N, 3)
        assert not np.any(np.isnan(np.array(result)))

    def test_sh_degree4(self):
        """Degree 4: 25 basis functions."""
        np.random.seed(42)
        N = 50
        dirs = mx.array(np.random.randn(N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(N, 25, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(4, dirs, coeffs)
        mx.eval(result)
        assert result.shape == (N, 3)
        assert not np.any(np.isnan(np.array(result)))

    def test_sh_auto_normalize(self):
        """Non-unit directions get normalized internally, same result as pre-normalized."""
        dirs_unit = mx.array([[0.0, 0.0, 1.0]])
        dirs_scaled = mx.array([[0.0, 0.0, 5.0]])

        coeffs = mx.array(np.random.randn(1, 9, 3).astype(np.float32) * 0.1)

        r_unit = spherical_harmonics(2, dirs_unit, coeffs)
        r_scaled = spherical_harmonics(2, dirs_scaled, coeffs)
        mx.eval(r_unit, r_scaled)

        np.testing.assert_allclose(np.array(r_unit), np.array(r_scaled), atol=1e-5)

    def test_sh_batch(self):
        """Batch of N=1000 Gaussians."""
        np.random.seed(42)
        N = 1000
        dirs = mx.array(np.random.randn(N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(N, 16, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(3, dirs, coeffs)
        mx.eval(result)
        assert result.shape == (N, 3)
        assert not np.any(np.isnan(np.array(result)))

    def test_sh_multi_camera(self):
        """Multi-camera batch: C=4 cameras x N=500 Gaussians."""
        np.random.seed(42)
        C, N = 4, 500
        dirs = mx.array(np.random.randn(C, N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(C, N, 16, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(3, dirs, coeffs)
        mx.eval(result)
        assert result.shape == (C, N, 3)
        assert not np.any(np.isnan(np.array(result)))

    def test_sh_zero_direction(self):
        """Near-zero direction does not produce NaN or Inf."""
        dirs = mx.array([[1e-10, 1e-10, 1e-10]])
        coeffs = mx.array(np.random.randn(1, 9, 3).astype(np.float32) * 0.1)
        result = spherical_harmonics(2, dirs, coeffs)
        mx.eval(result)
        result_np = np.array(result)
        assert not np.any(np.isnan(result_np))
        assert not np.any(np.isinf(result_np))

    def test_sh_extra_coeffs_ignored(self):
        """Extra coefficients beyond (degree+1)^2 are ignored when using lower degree."""
        np.random.seed(42)
        dirs = mx.array([[0.0, 0.0, 1.0]])
        coeffs_full = mx.array(np.random.randn(1, 25, 3).astype(np.float32))

        # Using degree 1 should only use first 4 bases
        r_deg1 = spherical_harmonics(1, dirs, coeffs_full)
        mx.eval(r_deg1)

        # Compare with manually using only first 4 coeffs
        coeffs_4 = coeffs_full[:, :4, :]
        r_4 = spherical_harmonics(1, dirs, coeffs_4)
        mx.eval(r_4)

        np.testing.assert_allclose(np.array(r_deg1), np.array(r_4), atol=1e-7)


# ===================================================================
# VJP / Gradient Tests
# ===================================================================


class TestSphericalHarmonicsVJP:
    """Backward pass / gradient tests."""

    def test_vjp_coeffs_degree0(self):
        """Gradient w.r.t. coeffs at degree 0: should be C0 * v_colors."""
        dirs = mx.array([[0.0, 0.0, 1.0]])
        coeffs = mx.array([[[1.0, 2.0, 3.0]]])

        def loss_fn(c):
            colors = spherical_harmonics(0, dirs, c)
            return mx.sum(colors)

        grad_fn = mx.grad(loss_fn)
        g = grad_fn(coeffs)
        mx.eval(g)

        C0 = 0.2820947917738781
        # v_colors = [1, 1, 1] (from sum), so grad = C0 * [1, 1, 1]
        expected = np.full((1, 1, 3), C0)
        np.testing.assert_allclose(np.array(g), expected, atol=1e-5)

    def test_vjp_coeffs_degree1(self):
        """Gradient w.r.t. coeffs at degree 1."""
        dirs = mx.array([[1.0, 0.0, 0.0]])
        coeffs = mx.array(np.random.randn(1, 4, 3).astype(np.float32) * 0.1)

        def loss_fn(c):
            return mx.sum(spherical_harmonics(1, dirs, c))

        grad_fn = mx.grad(loss_fn)
        g = grad_fn(coeffs)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (1, 4, 3)
        assert not np.all(g_np == 0), "Gradients should be non-zero"

    def test_vjp_coeffs_degree2(self):
        """Gradient w.r.t. coeffs at degree 2."""
        np.random.seed(42)
        dirs = mx.array(np.random.randn(5, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(5, 9, 3).astype(np.float32) * 0.1)

        def loss_fn(c):
            return mx.sum(spherical_harmonics(2, dirs, c))

        g = mx.grad(loss_fn)(coeffs)
        mx.eval(g)
        assert g.shape == coeffs.shape
        assert not np.all(np.array(g) == 0)

    def test_vjp_coeffs_degree3(self):
        """Gradient w.r.t. coeffs at degree 3."""
        np.random.seed(42)
        dirs = mx.array(np.random.randn(5, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(5, 16, 3).astype(np.float32) * 0.1)

        def loss_fn(c):
            return mx.sum(spherical_harmonics(3, dirs, c))

        g = mx.grad(loss_fn)(coeffs)
        mx.eval(g)
        assert g.shape == coeffs.shape
        assert not np.all(np.array(g) == 0)

    def test_vjp_dirs_degree1(self):
        """Gradient w.r.t. dirs at degree 1 (non-zero for linear SH)."""
        # Use non-axis-aligned direction so normalization Jacobian is non-degenerate
        dirs = mx.array([[1.0, 0.5, 0.3]])
        coeffs_np = np.zeros((1, 4, 3), dtype=np.float32)
        coeffs_np[0, 1, :] = [1.0, 0.0, 0.0]  # Y_1^{-1} depends on y
        coeffs_np[0, 2, :] = [0.0, 1.0, 0.0]  # Y_1^0 depends on z
        coeffs_np[0, 3, :] = [0.0, 0.0, 1.0]  # Y_1^1 depends on x
        coeffs = mx.array(coeffs_np)

        def loss_fn(d):
            return mx.sum(spherical_harmonics(1, d, coeffs))

        g = mx.grad(loss_fn)(dirs)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (1, 3)
        # There should be non-zero gradient in some direction
        assert np.any(np.abs(g_np) > 1e-6), "Dirs gradient should be non-zero"

    def test_vjp_dirs_degree2(self):
        """Gradient w.r.t. dirs at degree 2."""
        np.random.seed(42)
        dirs = mx.array(np.random.randn(5, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(5, 9, 3).astype(np.float32) * 0.1)

        def loss_fn(d):
            return mx.sum(spherical_harmonics(2, d, coeffs))

        g = mx.grad(loss_fn)(dirs)
        mx.eval(g)
        assert g.shape == dirs.shape
        assert np.any(np.abs(np.array(g)) > 1e-6)

    def test_vjp_dirs_degree3(self):
        """Gradient w.r.t. dirs at degree 3."""
        np.random.seed(42)
        dirs = mx.array(np.random.randn(5, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(5, 16, 3).astype(np.float32) * 0.1)

        def loss_fn(d):
            return mx.sum(spherical_harmonics(3, d, coeffs))

        g = mx.grad(loss_fn)(dirs)
        mx.eval(g)
        assert g.shape == dirs.shape
        assert np.any(np.abs(np.array(g)) > 1e-6)

    def test_vjp_batch_shapes(self):
        """Gradient shapes correct for batch input [B, N, 3]."""
        np.random.seed(42)
        B, N = 2, 10
        dirs = mx.array(np.random.randn(B, N, 3).astype(np.float32))
        coeffs = mx.array(np.random.randn(B, N, 9, 3).astype(np.float32) * 0.1)

        def loss_fn_c(c):
            return mx.sum(spherical_harmonics(2, dirs, c))

        def loss_fn_d(d):
            return mx.sum(spherical_harmonics(2, d, coeffs))

        gc = mx.grad(loss_fn_c)(coeffs)
        gd = mx.grad(loss_fn_d)(dirs)
        mx.eval(gc, gd)

        assert gc.shape == coeffs.shape
        assert gd.shape == dirs.shape

    def test_vjp_numerical_coeffs(self):
        """Finite-difference gradient check for coeffs."""
        np.random.seed(42)
        dirs_np = np.random.randn(3, 3).astype(np.float32)
        coeffs_np = np.random.randn(3, 4, 3).astype(np.float32) * 0.1

        dirs = mx.array(dirs_np)

        def loss_fn(c):
            return mx.sum(spherical_harmonics(1, dirs, c))

        # Analytical gradient
        coeffs_mx = mx.array(coeffs_np)
        g_analytical = mx.grad(loss_fn)(coeffs_mx)
        mx.eval(g_analytical)
        g_analytical_np = np.array(g_analytical)

        # Numerical gradient via finite differences
        eps = 1e-4
        g_numerical = np.zeros_like(coeffs_np)
        for idx in np.ndindex(coeffs_np.shape):
            c_plus = coeffs_np.copy()
            c_minus = coeffs_np.copy()
            c_plus[idx] += eps
            c_minus[idx] -= eps
            f_plus = float(np.array(loss_fn(mx.array(c_plus))))
            f_minus = float(np.array(loss_fn(mx.array(c_minus))))
            g_numerical[idx] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            g_analytical_np, g_numerical, rtol=1e-3, atol=1e-5,
            err_msg="Coeffs gradient doesn't match finite differences"
        )

    def test_vjp_numerical_dirs(self):
        """Finite-difference gradient check for dirs."""
        np.random.seed(42)
        dirs_np = np.random.randn(3, 3).astype(np.float32)
        # Make dirs not too small for stable normalization
        dirs_np = dirs_np * 2.0
        coeffs_np = np.random.randn(3, 4, 3).astype(np.float32) * 0.1

        coeffs = mx.array(coeffs_np)

        def loss_fn(d):
            return mx.sum(spherical_harmonics(1, d, coeffs))

        # Analytical gradient
        dirs_mx = mx.array(dirs_np)
        g_analytical = mx.grad(loss_fn)(dirs_mx)
        mx.eval(g_analytical)
        g_analytical_np = np.array(g_analytical)

        # Numerical gradient
        eps = 1e-4
        g_numerical = np.zeros_like(dirs_np)
        for idx in np.ndindex(dirs_np.shape):
            d_plus = dirs_np.copy()
            d_minus = dirs_np.copy()
            d_plus[idx] += eps
            d_minus[idx] -= eps
            f_plus = float(np.array(loss_fn(mx.array(d_plus))))
            f_minus = float(np.array(loss_fn(mx.array(d_minus))))
            g_numerical[idx] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            g_analytical_np, g_numerical, rtol=1e-2, atol=1e-4,
            err_msg="Dirs gradient doesn't match finite differences"
        )


# ===================================================================
# Cross-Framework Tests (require torch)
# ===================================================================


def _torch_eval_sh_bases_fast(basis_dim, dirs):
    """Reference PyTorch SH basis evaluation (inlined from upstream to avoid import issues)."""
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = 0.2820947917738781
    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)
    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y
    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1
    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2
    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _torch_spherical_harmonics(degrees_to_use, dirs, coeffs):
    """Reference PyTorch SH evaluation (inlined from upstream)."""
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degrees_to_use + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _torch_eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestSphericalHarmonicsCrossFramework:
    """Compare MLX implementation against PyTorch reference."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
    def test_cross_framework_sh(self, degree):
        """Compare MLX vs torch _spherical_harmonics for random inputs."""
        np.random.seed(42 + degree)
        N = 1000
        num_bases = (degree + 1) ** 2

        dirs_np = np.random.randn(N, 3).astype(np.float32)
        coeffs_np = np.random.randn(N, num_bases, 3).astype(np.float32) * 0.1

        # MLX
        dirs_mx = mx.array(dirs_np)
        coeffs_mx = mx.array(coeffs_np)
        result_mlx = spherical_harmonics(degree, dirs_mx, coeffs_mx)
        mx.eval(result_mlx)

        # PyTorch
        dirs_t = torch.from_numpy(dirs_np)
        coeffs_t = torch.from_numpy(coeffs_np)
        result_torch = _torch_spherical_harmonics(degree, dirs_t, coeffs_t)

        check_all_close(
            result_mlx, result_torch, atol=1e-5,
            msg=f"Cross-framework SH degree {degree}",
        )

    def test_cross_framework_backward_coeffs(self):
        """Compare VJP w.r.t. coeffs against torch autograd."""
        np.random.seed(42)
        N = 100
        degree = 2
        num_bases = (degree + 1) ** 2

        dirs_np = np.random.randn(N, 3).astype(np.float32)
        coeffs_np = np.random.randn(N, num_bases, 3).astype(np.float32) * 0.1

        # MLX gradient
        dirs_mx = mx.array(dirs_np)
        coeffs_mx = mx.array(coeffs_np)

        def mlx_loss(c):
            return mx.sum(spherical_harmonics(degree, dirs_mx, c))

        g_mlx = mx.grad(mlx_loss)(coeffs_mx)
        mx.eval(g_mlx)

        # PyTorch gradient
        dirs_t = torch.from_numpy(dirs_np)
        coeffs_t = torch.from_numpy(coeffs_np).requires_grad_(True)
        result_t = _torch_spherical_harmonics(degree, dirs_t, coeffs_t)
        loss_t = result_t.sum()
        loss_t.backward()

        check_all_close(
            g_mlx, coeffs_t.grad, atol=1e-4,
            msg="Cross-framework backward coeffs",
        )

    def test_cross_framework_backward_dirs(self):
        """Compare VJP w.r.t. dirs against torch autograd."""
        np.random.seed(42)
        N = 100
        degree = 2
        num_bases = (degree + 1) ** 2

        dirs_np = np.random.randn(N, 3).astype(np.float32) * 2.0
        coeffs_np = np.random.randn(N, num_bases, 3).astype(np.float32) * 0.1

        # MLX gradient
        coeffs_mx = mx.array(coeffs_np)

        def mlx_loss(d):
            return mx.sum(spherical_harmonics(degree, d, coeffs_mx))

        dirs_mx = mx.array(dirs_np)
        g_mlx = mx.grad(mlx_loss)(dirs_mx)
        mx.eval(g_mlx)

        # PyTorch gradient
        dirs_t = torch.from_numpy(dirs_np).requires_grad_(True)
        coeffs_t = torch.from_numpy(coeffs_np)
        result_t = _torch_spherical_harmonics(degree, dirs_t, coeffs_t)
        loss_t = result_t.sum()
        loss_t.backward()

        check_all_close(
            g_mlx, dirs_t.grad, atol=1e-3,
            msg="Cross-framework backward dirs",
        )
