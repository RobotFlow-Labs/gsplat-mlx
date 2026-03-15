"""Tests for gsplat_mlx.core.covariance — PRD-03.

Covers forward correctness, output shapes, symmetry, positive-definiteness,
triu packing, selective computation, batching, and gradient flow.
"""

import math
import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.core.covariance import quat_scale_to_covar_preci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_quat() -> mx.array:
    """Return a single identity quaternion (w,x,y,z) = (1,0,0,0)."""
    return mx.array([[1.0, 0.0, 0.0, 0.0]])


def _z90_quat() -> mx.array:
    """Return quaternion for 90-degree rotation about the z-axis."""
    c = math.cos(math.pi / 4)  # cos(45deg)
    s = math.sin(math.pi / 4)  # sin(45deg)
    return mx.array([[c, 0.0, 0.0, s]])


def _to_np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForwardCorrectness:
    """Verify known analytic results."""

    def test_covar_identity(self):
        """q=identity, s=(1,1,1) -> covariance = I_3."""
        quats = _identity_quat()
        scales = mx.array([[1.0, 1.0, 1.0]])
        covars, precis = quat_scale_to_covar_preci(quats, scales)
        mx.eval(covars, precis)

        np.testing.assert_allclose(_to_np(covars)[0], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(_to_np(precis)[0], np.eye(3), atol=1e-5)

    def test_covar_scaled(self):
        """q=identity, s=(2,3,4) -> covariance = diag(4,9,16)."""
        quats = _identity_quat()
        scales = mx.array([[2.0, 3.0, 4.0]])
        covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False)
        mx.eval(covars)

        expected = np.diag([4.0, 9.0, 16.0])
        np.testing.assert_allclose(_to_np(covars)[0], expected, atol=1e-5)

    def test_covar_rotated(self):
        """90-deg rotation about z with scale (2,1,1).

        R_z(90) maps x->y, y->-x, z->z.
        M = R diag(2,1,1), so columns of M are (0,2,0), (-1,0,0), (0,0,1).
        Sigma = M M^T should give diag entries [1, 4, 1] and off-diags ~ 0.
        """
        quats = _z90_quat()
        scales = mx.array([[2.0, 1.0, 1.0]])
        covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False)
        mx.eval(covars)

        c = _to_np(covars)[0]
        # Diagonal entries
        np.testing.assert_allclose(c[0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(c[1, 1], 4.0, atol=1e-5)
        np.testing.assert_allclose(c[2, 2], 1.0, atol=1e-5)
        # Off-diagonals should be ~0
        np.testing.assert_allclose(c[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(c[0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(c[1, 2], 0.0, atol=1e-5)

    def test_preci_is_inverse(self):
        """For random inputs, precision ~= inv(covariance)."""
        np.random.seed(42)
        quats = mx.array(np.random.randn(8, 4).astype(np.float32))
        scales = mx.array(np.abs(np.random.randn(8, 4)[:, :3]).astype(np.float32) + 0.1)

        covars, precis = quat_scale_to_covar_preci(quats, scales)
        mx.eval(covars, precis)

        c_np = _to_np(covars)
        p_np = _to_np(precis)

        for i in range(8):
            product = c_np[i] @ p_np[i]
            np.testing.assert_allclose(product, np.eye(3), atol=1e-2)


# ---------------------------------------------------------------------------
# triu mode
# ---------------------------------------------------------------------------

class TestTriuMode:
    """Verify the upper-triangle packing."""

    def test_triu_mode(self):
        """triu output has 6 elements matching upper triangle of 3x3."""
        quats = _identity_quat()
        scales = mx.array([[2.0, 3.0, 4.0]])

        covars_full, _ = quat_scale_to_covar_preci(
            quats, scales, compute_preci=False, triu=False
        )
        covars_tri, _ = quat_scale_to_covar_preci(
            quats, scales, compute_preci=False, triu=True
        )
        mx.eval(covars_full, covars_tri)

        full_np = _to_np(covars_full)[0]
        tri_np = _to_np(covars_tri)[0]

        assert tri_np.shape == (6,)

        # Upper-triangle order: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
        expected = [full_np[0, 0], full_np[0, 1], full_np[0, 2],
                    full_np[1, 1], full_np[1, 2], full_np[2, 2]]
        np.testing.assert_allclose(tri_np, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Selective computation
# ---------------------------------------------------------------------------

class TestSelectiveCompute:
    """Test compute_covar / compute_preci flags."""

    def test_covar_only(self):
        """compute_preci=False returns None for precis."""
        covars, precis = quat_scale_to_covar_preci(
            _identity_quat(), mx.array([[1.0, 1.0, 1.0]]), compute_preci=False
        )
        mx.eval(covars)
        assert covars is not None
        assert precis is None

    def test_preci_only(self):
        """compute_covar=False returns None for covars."""
        covars, precis = quat_scale_to_covar_preci(
            _identity_quat(), mx.array([[1.0, 1.0, 1.0]]), compute_covar=False
        )
        mx.eval(precis)
        assert covars is None
        assert precis is not None


# ---------------------------------------------------------------------------
# Batch dimensions
# ---------------------------------------------------------------------------

class TestBatchDims:
    """Verify various batch shapes work correctly."""

    @pytest.mark.parametrize("batch_shape", [
        (1,),
        (5,),
        (2, 3),
        (4, 2, 3),
    ])
    def test_batch_dims(self, batch_shape):
        np.random.seed(0)
        quats = mx.array(np.random.randn(*batch_shape, 4).astype(np.float32))
        scales = mx.array(np.abs(np.random.randn(*batch_shape, 3).astype(np.float32)) + 0.1)

        covars, precis = quat_scale_to_covar_preci(quats, scales)
        mx.eval(covars, precis)

        assert covars.shape == batch_shape + (3, 3)
        assert precis.shape == batch_shape + (3, 3)

    @pytest.mark.parametrize("batch_shape", [
        (5,),
        (2, 3),
    ])
    def test_batch_dims_triu(self, batch_shape):
        np.random.seed(0)
        quats = mx.array(np.random.randn(*batch_shape, 4).astype(np.float32))
        scales = mx.array(np.abs(np.random.randn(*batch_shape, 3).astype(np.float32)) + 0.1)

        covars, precis = quat_scale_to_covar_preci(quats, scales, triu=True)
        mx.eval(covars, precis)

        assert covars.shape == batch_shape + (6,)
        assert precis.shape == batch_shape + (6,)


# ---------------------------------------------------------------------------
# Matrix properties
# ---------------------------------------------------------------------------

class TestMatrixProperties:
    """Covariance must be symmetric and positive-definite."""

    def test_covar_symmetric(self):
        """covars == covars^T."""
        np.random.seed(7)
        quats = mx.array(np.random.randn(10, 4).astype(np.float32))
        scales = mx.array(np.abs(np.random.randn(10, 3).astype(np.float32)) + 0.1)

        covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False)
        mx.eval(covars)
        c_np = _to_np(covars)

        for i in range(10):
            np.testing.assert_allclose(c_np[i], c_np[i].T, atol=1e-6)

    def test_covar_positive_definite(self):
        """All eigenvalues of covariance are > 0 for positive scales."""
        np.random.seed(11)
        quats = mx.array(np.random.randn(10, 4).astype(np.float32))
        scales = mx.array(np.abs(np.random.randn(10, 3).astype(np.float32)) + 0.5)

        covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False)
        mx.eval(covars)
        c_np = _to_np(covars)

        for i in range(10):
            eigvals = np.linalg.eigvalsh(c_np[i])
            assert np.all(eigvals > 0), f"Non-positive eigenvalue at index {i}: {eigvals}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Ensure mx.grad flows through covariance w.r.t. quats and scales."""

    def test_gradient_flows_quats(self):
        """Gradient of scalar loss w.r.t. quats is non-zero."""
        quats = mx.array([[1.0, 0.1, 0.2, 0.3]])
        scales = mx.array([[1.0, 2.0, 3.0]])

        def loss_fn(q):
            covars, _ = quat_scale_to_covar_preci(q, scales, compute_preci=False)
            return mx.sum(covars)

        grad_fn = mx.grad(loss_fn)
        g = grad_fn(quats)
        mx.eval(g)
        g_np = _to_np(g)

        # At least some gradient components should be non-zero
        assert np.any(np.abs(g_np) > 1e-6), f"All-zero gradients w.r.t. quats: {g_np}"

    def test_gradient_flows_scales(self):
        """Gradient of scalar loss w.r.t. scales is non-zero."""
        quats = mx.array([[1.0, 0.1, 0.2, 0.3]])
        scales = mx.array([[1.0, 2.0, 3.0]])

        def loss_fn(s):
            covars, _ = quat_scale_to_covar_preci(quats, s, compute_preci=False)
            return mx.sum(covars)

        grad_fn = mx.grad(loss_fn)
        g = grad_fn(scales)
        mx.eval(g)
        g_np = _to_np(g)

        assert np.any(np.abs(g_np) > 1e-6), f"All-zero gradients w.r.t. scales: {g_np}"
