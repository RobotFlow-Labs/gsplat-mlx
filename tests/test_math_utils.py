"""Comprehensive tests for math_utils.py — MLX port of gsplat _math.py.

Tests compare MLX functions against numpy/known values for numerical correctness.
"""

import numpy as np
import pytest
import mlx.core as mx

from gsplat_mlx.core.math_utils import (
    _assert_shape,
    _cross,
    _numerically_stable_norm2,
    FullPolynomialProxy,
    OddPolynomialProxy,
    EvenPolynomialProxy,
    _eval_poly_inverse_horner_newton,
    _safe_normalize,
    _rotmat_to_quat,
    _quat_normalize_rotation,
    _quat_inverse,
    _quat_rotate,
    _quat_multiply,
    _quat_slerp,
    _quat_to_rotmat,
    _quat_scale_to_matrix,
    _quat_scale_to_covar_preci,
    _quat_scale_to_preci_half,
    compute_inverse_polynomial,
)

ATOL = 1e-5
ATOL_LOOSE = 1e-4


def to_np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.array(x)


# ============================================================================
# assert_shape
# ============================================================================

class TestAssertShape:
    def test_correct_shape(self):
        arr = mx.zeros((3, 4))
        _assert_shape("arr", arr, (3, 4))

    def test_wrong_shape_raises(self):
        arr = mx.zeros((3, 4))
        with pytest.raises(ValueError, match="Expected"):
            _assert_shape("arr", arr, (3, 5))


# ============================================================================
# cross product
# ============================================================================

class TestCross:
    def test_basic(self):
        a = mx.array([[1.0, 0.0, 0.0]])
        b = mx.array([[0.0, 1.0, 0.0]])
        result = to_np(_cross(a, b))
        np.testing.assert_allclose(result, [[0.0, 0.0, 1.0]], atol=ATOL)

    def test_anticommutative(self):
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        ab = to_np(_cross(a, b))
        ba = to_np(_cross(b, a))
        np.testing.assert_allclose(ab, -ba, atol=ATOL)

    def test_batch(self):
        a = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = mx.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = to_np(_cross(a, b))
        expected = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        np.testing.assert_allclose(result, expected, atol=ATOL)


# ============================================================================
# _numerically_stable_norm2
# ============================================================================

class TestNumericallyStableNorm2:
    def test_normal_values(self):
        x = mx.array([3.0, 1.0, 0.0])
        y = mx.array([4.0, 1.0, 1.0])
        result = to_np(_numerically_stable_norm2(x, y))
        expected = np.sqrt(np.array([3.0, 1.0, 0.0]) ** 2 + np.array([4.0, 1.0, 1.0]) ** 2)
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_large_values(self):
        x = mx.array([1e30])
        y = mx.array([1e30])
        result = to_np(_numerically_stable_norm2(x, y))
        expected = np.array([1e30 * np.sqrt(2.0)])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_small_values(self):
        x = mx.array([1e-30])
        y = mx.array([1e-30])
        result = to_np(_numerically_stable_norm2(x, y))
        expected = np.array([1e-30 * np.sqrt(2.0)])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zero(self):
        x = mx.array([0.0])
        y = mx.array([0.0])
        result = to_np(_numerically_stable_norm2(x, y))
        np.testing.assert_allclose(result, [0.0], atol=ATOL)


# ============================================================================
# Polynomial proxies
# ============================================================================

class TestPolynomialFull:
    def test_constant(self):
        # p(x) = 5
        coeffs = mx.array([5.0])
        poly = FullPolynomialProxy(coeffs)
        x = mx.array([2.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [5.0], atol=ATOL)

    def test_linear(self):
        # p(x) = 1 + 2x
        coeffs = mx.array([1.0, 2.0])
        poly = FullPolynomialProxy(coeffs)
        x = mx.array([3.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [7.0], atol=ATOL)

    def test_quadratic(self):
        # p(x) = 1 + 0*x + 1*x^2 = 1 + x^2
        coeffs = mx.array([1.0, 0.0, 1.0])
        poly = FullPolynomialProxy(coeffs)
        x = mx.array([3.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [10.0], atol=ATOL)


class TestPolynomialOdd:
    def test_linear(self):
        # p(x) = 2x
        coeffs = mx.array([2.0])
        poly = OddPolynomialProxy(coeffs)
        x = mx.array([3.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [6.0], atol=ATOL)

    def test_cubic(self):
        # p(x) = 1*x + 1*x^3 for x=2: 2 + 8 = 10
        coeffs = mx.array([1.0, 1.0])
        poly = OddPolynomialProxy(coeffs)
        x = mx.array([2.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [10.0], atol=ATOL)


class TestPolynomialEven:
    def test_constant(self):
        # p(x) = 3
        coeffs = mx.array([3.0])
        poly = EvenPolynomialProxy(coeffs)
        x = mx.array([5.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [3.0], atol=ATOL)

    def test_quadratic(self):
        # p(x) = 1 + 2*x^2 for x=3: 1 + 18 = 19
        coeffs = mx.array([1.0, 2.0])
        poly = EvenPolynomialProxy(coeffs)
        x = mx.array([3.0])
        result = to_np(poly.eval_horner(x))
        np.testing.assert_allclose(result, [19.0], atol=ATOL)


# ============================================================================
# _safe_normalize
# ============================================================================

class TestSafeNormalize:
    def test_unit_vector(self):
        v = mx.array([1.0, 0.0, 0.0])
        result = to_np(_safe_normalize(v))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=ATOL)

    def test_non_unit(self):
        v = mx.array([3.0, 4.0, 0.0])
        result = to_np(_safe_normalize(v))
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=ATOL)

    def test_zero_vector(self):
        v = mx.array([0.0, 0.0, 0.0])
        result = to_np(_safe_normalize(v))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=ATOL)

    def test_batch(self):
        v = mx.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = to_np(_safe_normalize(v))
        expected = [[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_vjp_nonzero(self):
        """Gradient check for safe_normalize with non-zero input."""
        def fn(v):
            return mx.sum(_safe_normalize(v))

        v = mx.array([3.0, 4.0, 0.0])
        grad_fn = mx.grad(fn)
        grad = to_np(grad_fn(v))

        # Numerical gradient check
        eps = 1e-4
        numerical_grad = np.zeros(3)
        for i in range(3):
            v_plus = np.array([3.0, 4.0, 0.0])
            v_minus = np.array([3.0, 4.0, 0.0])
            v_plus[i] += eps
            v_minus[i] -= eps
            f_plus = float(to_np(mx.sum(_safe_normalize(mx.array(v_plus)))))
            f_minus = float(to_np(mx.sum(_safe_normalize(mx.array(v_minus)))))
            numerical_grad[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grad, numerical_grad, atol=1e-3)


# ============================================================================
# _quat_to_rotmat
# ============================================================================

class TestQuatToRotmat:
    def test_identity(self):
        """Identity quaternion (1,0,0,0) should give identity matrix."""
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        R = to_np(_quat_to_rotmat(q))
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_90deg_z(self):
        """90 degree rotation around z-axis."""
        angle = np.pi / 2
        q = mx.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        R = to_np(_quat_to_rotmat(q))
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_90deg_x(self):
        """90 degree rotation around x-axis."""
        angle = np.pi / 2
        q = mx.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        R = to_np(_quat_to_rotmat(q))
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_batch(self):
        """Batch of quaternions."""
        q = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        R = to_np(_quat_to_rotmat(q))
        assert R.shape == (2, 3, 3)
        np.testing.assert_allclose(R[0], np.eye(3), atol=ATOL)
        np.testing.assert_allclose(R[1], np.eye(3), atol=ATOL)


# ============================================================================
# _rotmat_to_quat (roundtrip)
# ============================================================================

class TestRotmatToQuat:
    def test_identity(self):
        """Identity matrix -> (1, 0, 0, 0)."""
        R = mx.array(np.eye(3, dtype=np.float32))
        q = to_np(_rotmat_to_quat(R))
        # Should be (1,0,0,0) or (-1,0,0,0)
        assert abs(abs(q[0]) - 1.0) < ATOL
        np.testing.assert_allclose(q[1:], [0.0, 0.0, 0.0], atol=ATOL)

    def test_roundtrip(self):
        """quat -> rotmat -> quat should give same quaternion (up to sign)."""
        q_orig = mx.array([0.5, 0.5, 0.5, 0.5])  # normalized
        R = _quat_to_rotmat(q_orig)
        q_recovered = to_np(_rotmat_to_quat(R))
        q_orig_np = to_np(q_orig)

        # Quaternions are equivalent up to sign
        if np.dot(q_orig_np, q_recovered) < 0:
            q_recovered = -q_recovered
        np.testing.assert_allclose(q_recovered, q_orig_np, atol=ATOL_LOOSE)

    def test_roundtrip_90deg(self):
        """90-degree rotation roundtrip."""
        angle = np.pi / 2
        q_orig = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], dtype=np.float32)
        R = _quat_to_rotmat(mx.array(q_orig))
        q_recovered = to_np(_rotmat_to_quat(R))

        if np.dot(q_orig, q_recovered) < 0:
            q_recovered = -q_recovered
        np.testing.assert_allclose(q_recovered, q_orig, atol=ATOL_LOOSE)

    def test_batch(self):
        """Batch roundtrip."""
        quats = mx.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ])
        R = _quat_to_rotmat(quats)
        q_back = to_np(_rotmat_to_quat(R))
        q_orig = to_np(quats)

        for i in range(2):
            q_r = q_back[i]
            q_o = q_orig[i]
            if np.dot(q_o, q_r) < 0:
                q_r = -q_r
            np.testing.assert_allclose(q_r, q_o, atol=ATOL_LOOSE)


# ============================================================================
# _quat_normalize_rotation
# ============================================================================

class TestQuatNormalize:
    def test_already_normalized(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        result = to_np(_quat_normalize_rotation(q))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_unnormalized(self):
        q = mx.array([2.0, 0.0, 0.0, 0.0])
        result = to_np(_quat_normalize_rotation(q))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_zero_quat(self):
        q = mx.array([0.0, 0.0, 0.0, 0.0])
        result = to_np(_quat_normalize_rotation(q))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_negative_w(self):
        """If w < 0, negate to get single cover."""
        q = mx.array([-1.0, 0.0, 0.0, 0.0])
        result = to_np(_quat_normalize_rotation(q))
        assert result[0] > 0, f"w should be positive, got {result[0]}"


# ============================================================================
# _quat_inverse
# ============================================================================

class TestQuatInverse:
    def test_identity(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        q_inv = to_np(_quat_inverse(q))
        np.testing.assert_allclose(q_inv, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_q_times_q_inv_is_identity(self):
        """q * q^-1 should be identity (1, 0, 0, 0)."""
        q = mx.array([0.5, 0.5, 0.5, 0.5])  # unit quaternion
        q_inv = _quat_inverse(q)
        product = to_np(_quat_multiply(q, q_inv))
        np.testing.assert_allclose(product, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_conjugate(self):
        q = mx.array([0.5, 0.5, 0.5, 0.5])
        q_inv = to_np(_quat_inverse(q))
        np.testing.assert_allclose(q_inv, [0.5, -0.5, -0.5, -0.5], atol=ATOL)


# ============================================================================
# _quat_rotate
# ============================================================================

class TestQuatRotate:
    def test_identity_rotation(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        v = mx.array([1.0, 2.0, 3.0])
        result = to_np(_quat_rotate(q, v))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=ATOL)

    def test_90deg_z_rotation(self):
        """90 degrees around z: (1,0,0) -> (0,1,0)."""
        angle = np.pi / 2
        q = mx.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        v = mx.array([1.0, 0.0, 0.0])
        result = to_np(_quat_rotate(q, v))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=ATOL)

    def test_90deg_x_rotation(self):
        """90 degrees around x: (0,1,0) -> (0,0,1)."""
        angle = np.pi / 2
        q = mx.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        v = mx.array([0.0, 1.0, 0.0])
        result = to_np(_quat_rotate(q, v))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=ATOL)

    def test_180deg_rotation(self):
        """180 degrees around z: (1,0,0) -> (-1,0,0)."""
        q = mx.array([0.0, 0.0, 0.0, 1.0])  # 180 deg around z
        v = mx.array([1.0, 0.0, 0.0])
        result = to_np(_quat_rotate(q, v))
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=ATOL)


# ============================================================================
# _quat_multiply
# ============================================================================

class TestQuatMultiply:
    def test_identity_left(self):
        q_id = mx.array([1.0, 0.0, 0.0, 0.0])
        q = mx.array([0.5, 0.5, 0.5, 0.5])
        result = to_np(_quat_multiply(q_id, q))
        np.testing.assert_allclose(result, to_np(q), atol=ATOL)

    def test_identity_right(self):
        q_id = mx.array([1.0, 0.0, 0.0, 0.0])
        q = mx.array([0.5, 0.5, 0.5, 0.5])
        result = to_np(_quat_multiply(q, q_id))
        np.testing.assert_allclose(result, to_np(q), atol=ATOL)

    def test_known_product(self):
        """i * j = k in quaternion multiplication."""
        # i = (0, 1, 0, 0), j = (0, 0, 1, 0)
        qi = mx.array([0.0, 1.0, 0.0, 0.0])
        qj = mx.array([0.0, 0.0, 1.0, 0.0])
        result = to_np(_quat_multiply(qi, qj))
        # Expected: k = (0, 0, 0, 1)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0], atol=ATOL)

    def test_batch(self):
        q1 = mx.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        q2 = mx.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 0.0]])
        result = to_np(_quat_multiply(q1, q2))
        assert result.shape == (2, 4)
        # First: identity * q = q
        np.testing.assert_allclose(result[0], [0.5, 0.5, 0.5, 0.5], atol=ATOL)
        # Second: i * j = k
        np.testing.assert_allclose(result[1], [0.0, 0.0, 0.0, 1.0], atol=ATOL)


# ============================================================================
# _quat_slerp
# ============================================================================

class TestQuatSlerp:
    def test_t0_gives_x(self):
        x = mx.array([1.0, 0.0, 0.0, 0.0])
        angle = np.pi / 2
        y = mx.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        t = mx.array(0.0)
        result = to_np(_quat_slerp(x, y, t))
        np.testing.assert_allclose(result, to_np(x), atol=ATOL)

    def test_t1_gives_y(self):
        x = mx.array([1.0, 0.0, 0.0, 0.0])
        angle = np.pi / 2
        y = mx.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        t = mx.array(1.0)
        result = to_np(_quat_slerp(x, y, t))
        np.testing.assert_allclose(result, to_np(y), atol=ATOL)

    def test_midpoint(self):
        """t=0.5 should give midpoint quaternion."""
        x = mx.array([1.0, 0.0, 0.0, 0.0])
        angle = np.pi / 2
        y = mx.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        t = mx.array(0.5)
        result = to_np(_quat_slerp(x, y, t))

        # The midpoint should be at half the angle
        half_angle = np.pi / 4
        expected = np.array([np.cos(half_angle / 2), np.sin(half_angle / 2), 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=ATOL_LOOSE)

    def test_same_quaternion(self):
        """Slerp between identical quaternions gives the same quaternion."""
        x = mx.array([1.0, 0.0, 0.0, 0.0])
        t = mx.array(0.5)
        result = to_np(_quat_slerp(x, x, t))
        np.testing.assert_allclose(result, to_np(x), atol=ATOL)


# ============================================================================
# _quat_scale_to_matrix
# ============================================================================

class TestQuatScaleToMatrix:
    def test_identity_rotation_unit_scale(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([1.0, 1.0, 1.0])
        M = to_np(_quat_scale_to_matrix(q, s))
        np.testing.assert_allclose(M, np.eye(3), atol=ATOL)

    def test_identity_rotation_with_scale(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        M = to_np(_quat_scale_to_matrix(q, s))
        expected = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_allclose(M, expected, atol=ATOL)

    def test_batch(self):
        q = mx.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        s = mx.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        M = to_np(_quat_scale_to_matrix(q, s))
        assert M.shape == (2, 3, 3)
        np.testing.assert_allclose(M[0], np.eye(3), atol=ATOL)
        np.testing.assert_allclose(M[1], 2.0 * np.eye(3), atol=ATOL)


# ============================================================================
# _quat_scale_to_covar_preci
# ============================================================================

class TestQuatScaleToCovarPreci:
    def test_identity_covar(self):
        """Identity rotation + scale s -> covar = diag(s^2)."""
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        covars, precis = _quat_scale_to_covar_preci(q, s, compute_covar=True, compute_preci=False)
        covars_np = to_np(covars)
        expected = np.diag([4.0, 9.0, 16.0])
        np.testing.assert_allclose(covars_np, expected, atol=ATOL)

    def test_identity_preci(self):
        """Identity rotation + scale s -> preci = diag(1/s^2)."""
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        covars, precis = _quat_scale_to_covar_preci(q, s, compute_covar=False, compute_preci=True)
        precis_np = to_np(precis)
        expected = np.diag([0.25, 1.0 / 9.0, 1.0 / 16.0])
        # float32 precision: normalization introduces small error
        np.testing.assert_allclose(precis_np, expected, atol=1e-3)

    def test_both_modes(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        covars, precis = _quat_scale_to_covar_preci(q, s, compute_covar=True, compute_preci=True)
        assert covars is not None
        assert precis is not None

        # covar * preci should be close to identity
        # float32 normalization introduces small errors
        product = to_np(covars) @ to_np(precis)
        np.testing.assert_allclose(product, np.eye(3), atol=1e-3)

    def test_triu(self):
        """Triu mode returns 6 elements."""
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        covars, precis = _quat_scale_to_covar_preci(
            q, s, compute_covar=True, compute_preci=True, triu=True
        )
        covars_np = to_np(covars)
        precis_np = to_np(precis)
        assert covars_np.shape == (6,)
        assert precis_np.shape == (6,)
        # For diagonal covariance, triu should be [4, 0, 0, 9, 0, 16]
        np.testing.assert_allclose(covars_np, [4.0, 0.0, 0.0, 9.0, 0.0, 16.0], atol=ATOL)

    def test_batch(self):
        q = mx.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        s = mx.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        covars, _ = _quat_scale_to_covar_preci(q, s, compute_covar=True, compute_preci=False)
        covars_np = to_np(covars)
        assert covars_np.shape == (2, 3, 3)
        np.testing.assert_allclose(covars_np[0], np.eye(3), atol=ATOL)
        np.testing.assert_allclose(covars_np[1], 4.0 * np.eye(3), atol=ATOL)


# ============================================================================
# _quat_scale_to_preci_half
# ============================================================================

class TestQuatScaleToPreciHalf:
    def test_identity(self):
        q = mx.array([1.0, 0.0, 0.0, 0.0])
        s = mx.array([2.0, 3.0, 4.0])
        M = to_np(_quat_scale_to_preci_half(q, s))
        expected = np.diag([0.5, 1.0 / 3.0, 0.25])
        np.testing.assert_allclose(M, expected, atol=ATOL)


# ============================================================================
# compute_inverse_polynomial
# ============================================================================

class TestComputeInversePolynomial:
    def test_linear(self):
        """Inverse of f(x) = 2x should be g(y) = y/2."""
        coeffs = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        inv_coeffs = compute_inverse_polynomial(coeffs, (0.1, 1.0))
        # inv_coeffs[1] should be ~0.5
        assert abs(inv_coeffs[1] - 0.5) < 0.01

    def test_identity(self):
        """Inverse of f(x) = x should be g(y) = y."""
        coeffs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        inv_coeffs = compute_inverse_polynomial(coeffs, (0.1, 1.0))
        assert abs(inv_coeffs[1] - 1.0) < 0.01

    def test_raises_on_nan(self):
        """Should raise ValueError for pathological polynomials."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            compute_inverse_polynomial(coeffs, (0.1, 1.0))
