"""Mathematical utility functions for gsplat-mlx.

Port of gsplat/cuda/_math.py from PyTorch to Apple MLX.
Contains numerically stable norms, polynomial evaluation/inversion,
and complete quaternion algebra (normalize, rotate, multiply, slerp, convert).

Upstream reference: repositories/gsplat-upstream/gsplat/cuda/_math.py
"""

import mlx.core as mx
from abc import ABC, abstractmethod
from typing import Optional, Tuple


# ============================================================================
# Shape assertion helper
# ============================================================================


def _assert_shape(name: str, arr: mx.array, shape: tuple):
    """Assert that an MLX array has the expected shape.

    Args:
        name: Name of the array (for error messages).
        arr: The MLX array to check.
        shape: Expected shape tuple. Must match exactly.
    """
    if arr.shape != shape:
        raise ValueError(
            f"Expected {name} to have shape {shape}, but got {arr.shape}"
        )


# ============================================================================
# Cross product helper
# ============================================================================


def _cross(a: mx.array, b: mx.array) -> mx.array:
    """Manual cross product since mx.cross may not be available.

    Args:
        a: [..., 3] array
        b: [..., 3] array

    Returns:
        [..., 3] cross product a x b
    """
    a0 = a[..., 0]
    a1 = a[..., 1]
    a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    b2 = b[..., 2]
    return mx.stack(
        [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1
    )


# ============================================================================
# Numerically Stable Norm
# ============================================================================


def _numerically_stable_norm2(x: mx.array, y: mx.array) -> mx.array:
    """Compute 2-norm of [x, y] vectors in a numerically stable way.

    Avoids overflow/underflow for very large or very small values by
    normalizing by the maximum absolute value before computing the norm.

    Args:
        x: [B] x-components
        y: [B] y-components

    Returns:
        norm: [B] ||(x, y)|| = sqrt(x^2 + y^2)
    """
    B = x.shape
    _assert_shape("x", x, B)
    _assert_shape("y", y, B)

    abs_x = mx.abs(x)
    abs_y = mx.abs(y)
    min_val = mx.minimum(abs_x, abs_y)
    max_val = mx.maximum(abs_x, abs_y)

    nonzero_mask = max_val > 0.0

    min_max_ratio = mx.where(nonzero_mask, min_val / mx.maximum(max_val, mx.array(1e-38)), mx.zeros_like(min_val))
    result = mx.where(
        nonzero_mask,
        max_val * mx.sqrt(1.0 + min_max_ratio * min_max_ratio),
        mx.zeros_like(max_val),
    )

    _assert_shape("result", result, B)
    return result


# ============================================================================
# Polynomial Helper Classes and Functions
# ============================================================================


class PolynomialProxy(ABC):
    """Base class for polynomial evaluation with type dispatch.

    Matches the CUDA PolynomialProxy template struct pattern.
    Subclasses implement specific polynomial types (full, even, odd).
    """

    def __init__(self, coeffs: mx.array):
        """Initialize polynomial proxy.

        Args:
            coeffs: [..., B, N] Array of polynomial coefficients.
        """
        self.coeffs = coeffs

    @abstractmethod
    def eval_horner(self, x: mx.array) -> mx.array:
        """Evaluate polynomial at x using Horner's method."""
        ...


class FullPolynomialProxy(PolynomialProxy):
    """Full polynomial: y = c0 + c1*x + c2*x^2 + c3*x^3 + ..."""

    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        _assert_shape("x", x, B + (1,))

        # Start with highest order coefficient
        result = self.coeffs[..., N - 1 : N]

        # Horner's method: iterate backwards through remaining coefficients
        for i in range(N - 2, -1, -1):
            result = result * x + self.coeffs[..., i : i + 1]

        _assert_shape("result", result, B + (1,))
        return result


class OddPolynomialProxy(PolynomialProxy):
    """Odd-only polynomial: y = c0*x + c1*x^3 + c2*x^5 + ..."""

    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        _assert_shape("x", x, B + (1,))

        # Factor out x: y = x * (c0 + c1*x^2 + c2*x^4 + ...)
        result = x * FullPolynomialProxy(self.coeffs).eval_horner(x * x)

        _assert_shape("result", result, B + (1,))
        return result


class EvenPolynomialProxy(PolynomialProxy):
    """Even-only polynomial: y = c0 + c1*x^2 + c2*x^4 + ..."""

    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        _assert_shape("x", x, B + (1,))

        # Substitute x^2 for x
        result = FullPolynomialProxy(self.coeffs).eval_horner(x * x)

        _assert_shape("result", result, B + (1,))
        return result


def _eval_poly_inverse_horner_newton(
    poly: PolynomialProxy,
    dpoly: PolynomialProxy,
    inv_poly_approx: PolynomialProxy,
    y: mx.array,
    n_iterations: int,
) -> Tuple[mx.array, mx.array]:
    """Evaluate inverse polynomial x = f^-1(y) using Newton's method.

    Given a polynomial y = f(x), finds x such that f(x) = y using:
    1. Initial approximation from inv_poly_approx
    2. Newton iterations: x_{n+1} = x_n - (f(x_n) - y) / f'(x_n)

    Args:
        poly: PolynomialProxy for forward polynomial f(x)
        dpoly: PolynomialProxy for derivative f'(x)
        inv_poly_approx: PolynomialProxy for initial approximation
        y: Target values to invert
        n_iterations: Number of Newton iterations

    Returns:
        x: inverted values
        converged: convergence mask (boolean array)
    """
    B = poly.coeffs.shape[:-1]
    M = y.shape[-1]

    # Get initial approximation x0 = approx_f^-1(y)
    x = inv_poly_approx.eval_horner(y)

    converged = mx.zeros_like(x, dtype=mx.bool_)

    # Newton iterations
    for _ in range(n_iterations):
        fx = poly.eval_horner(x)
        dfdx = dpoly.eval_horner(x)

        residual = fx - y
        dx = residual / dfdx

        newly_converged = mx.abs(dx) < 1e-6

        # Only update elements that haven't converged yet
        x = mx.where(~converged, x - dx, x)

        converged = converged | newly_converged

        # MLX is lazy, so we can't do early exit like torch.all(converged)
        # Just run all iterations

    return x, converged


# ============================================================================
# Safe Normalize with Custom VJP
# ============================================================================


@mx.custom_function
def _safe_normalize_last_dim(v: mx.array) -> mx.array:
    """Safely normalize a vector along the last dimension.

    Forward: normalized = v / ||v|| if ||v|| > 0 else v

    Args:
        v: Input tensor to normalize (normalization along axis=-1)

    Returns:
        Normalized tensor with same shape as input
    """
    norm_sq = mx.sum(v * v, axis=-1, keepdims=True)
    inv_norm = mx.where(norm_sq > 0.0, 1.0 / mx.sqrt(norm_sq), mx.zeros_like(norm_sq))
    normalized = v * inv_norm
    return normalized


@_safe_normalize_last_dim.vjp
def _safe_normalize_vjp(primals, cotangent, output):
    """VJP for safe normalize (last dim).

    grad_v = il * grad_out - il^3 * dot(grad_out, v) * v  (for non-zero vectors)
    grad_v = grad_out  (for zero vectors)

    where il = 1/||v||
    """
    v = primals
    grad_output = cotangent

    norm_sq = mx.sum(v * v, axis=-1, keepdims=True)
    inv_norm = mx.where(norm_sq > 0.0, 1.0 / mx.sqrt(norm_sq), mx.zeros_like(norm_sq))

    il = inv_norm
    il3 = mx.where(norm_sq > 0.0, inv_norm / norm_sq, mx.zeros_like(inv_norm))

    dot_product = mx.sum(grad_output * v, axis=-1, keepdims=True)

    grad_v_nonzero = il * grad_output - il3 * dot_product * v
    grad_v = mx.where(norm_sq > 0.0, grad_v_nonzero, grad_output)

    return grad_v


def _safe_normalize(
    v: mx.array,
    dim: int = -1,
    keepdim: bool = False,
) -> mx.array:
    """Safely normalize a vector, returning zero vector if input norm is zero.

    Uses custom VJP that matches CUDA implementation.

    Note: Currently only supports dim=-1. Other dims would require
    transposing before/after normalization.

    Args:
        v: Input tensor to normalize
        dim: Dimension along which to compute the norm (must be -1)
        keepdim: Whether to keep the normalized dimension

    Returns:
        Normalized tensor
    """
    if dim == -1 or dim == v.ndim - 1:
        return _safe_normalize_last_dim(v)
    else:
        # For other dims, move the target dim to last, normalize, move back
        axes = list(range(v.ndim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        v_t = mx.transpose(v, axes)
        result_t = _safe_normalize_last_dim(v_t)
        return mx.transpose(result_t, axes)


# ============================================================================
# Quaternion Operations
# ============================================================================


def _rotmat_to_quat(R: mx.array) -> mx.array:
    """Convert rotation matrix to quaternion (w, x, y, z).

    Direct port of GLM's quat_cast with corrected indexing for row-major matrices.
    Uses nested mx.where instead of boolean indexing (not available in MLX).

    Args:
        R: [..., 3, 3] rotation matrix

    Returns:
        [..., 4] quaternion in (w, x, y, z) format
    """
    B = R.shape[:-2]
    _assert_shape("R", R, B + (3, 3))

    R_flat = mx.reshape(R, (-1, 3, 3))
    n = R_flat.shape[0]

    # Compute decision values
    fourXSquaredMinus1 = R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2]
    fourYSquaredMinus1 = R_flat[:, 1, 1] - R_flat[:, 0, 0] - R_flat[:, 2, 2]
    fourZSquaredMinus1 = R_flat[:, 2, 2] - R_flat[:, 0, 0] - R_flat[:, 1, 1]
    fourWSquaredMinus1 = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    # Find largest component via stacking and argmax
    fourBiggest = mx.stack(
        [fourWSquaredMinus1, fourXSquaredMinus1, fourYSquaredMinus1, fourZSquaredMinus1],
        axis=1,
    )
    biggestIndex = mx.argmax(fourBiggest, axis=1)  # [n]

    # Gather the biggest value
    # take_along_axis equivalent
    biggestSquaredMinus1 = mx.take_along_axis(
        fourBiggest, mx.expand_dims(biggestIndex, axis=1), axis=1
    )
    biggestSquaredMinus1 = mx.squeeze(biggestSquaredMinus1, axis=1)

    biggestVal = mx.sqrt(biggestSquaredMinus1 + 1.0) * 0.5
    mult = 0.25 / biggestVal

    # Case 0: w is largest
    q0_w = biggestVal
    q0_x = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) * mult
    q0_y = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) * mult
    q0_z = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) * mult

    # Case 1: x is largest
    q1_w = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) * mult
    q1_x = biggestVal
    q1_y = (R_flat[:, 1, 0] + R_flat[:, 0, 1]) * mult
    q1_z = (R_flat[:, 0, 2] + R_flat[:, 2, 0]) * mult

    # Case 2: y is largest
    q2_w = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) * mult
    q2_x = (R_flat[:, 1, 0] + R_flat[:, 0, 1]) * mult
    q2_y = biggestVal
    q2_z = (R_flat[:, 2, 1] + R_flat[:, 1, 2]) * mult

    # Case 3: z is largest
    q3_w = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) * mult
    q3_x = (R_flat[:, 0, 2] + R_flat[:, 2, 0]) * mult
    q3_y = (R_flat[:, 2, 1] + R_flat[:, 1, 2]) * mult
    q3_z = biggestVal

    # Use nested mx.where to select the correct case (no boolean indexing)
    is0 = mx.equal(biggestIndex, 0)
    is1 = mx.equal(biggestIndex, 1)
    is2 = mx.equal(biggestIndex, 2)

    qw = mx.where(is0, q0_w, mx.where(is1, q1_w, mx.where(is2, q2_w, q3_w)))
    qx = mx.where(is0, q0_x, mx.where(is1, q1_x, mx.where(is2, q2_x, q3_x)))
    qy = mx.where(is0, q0_y, mx.where(is1, q1_y, mx.where(is2, q2_y, q3_y)))
    qz = mx.where(is0, q0_z, mx.where(is1, q1_z, mx.where(is2, q2_z, q3_z)))

    quat = mx.stack([qw, qx, qy, qz], axis=-1)
    quat = mx.reshape(quat, B + (4,))

    _assert_shape("quat", quat, B + (4,))
    return quat


def _quat_normalize_rotation(q: mx.array, dim: int = -1) -> mx.array:
    """Normalize quaternion, handle zero & double-cover.

    Args:
        q: [..., 4] quaternion

    Returns:
        Normalized quaternion with w >= 0
    """
    assert q.shape[dim] == 4, q.shape

    dim = dim if dim >= 0 else q.ndim + dim

    result = _safe_normalize(q, dim=dim)

    # If quaternion is all zeros, return identity (1, 0, 0, 0)
    all_zero = mx.all(mx.equal(q, 0.0), axis=dim, keepdims=True)

    # Build identity quaternion
    slices = []
    for i in range(4):
        if i == 0:
            slices.append(mx.ones_like(result[..., 0:1]))
        else:
            slices.append(mx.zeros_like(result[..., 0:1]))
    identity = mx.concatenate(slices, axis=-1)

    result = mx.where(all_zero, identity, result)

    # Make "double-cover" into "single-cover": if w < 0, negate
    # Select w component
    w_negative = mx.expand_dims(result[..., 0] < 0, axis=dim)
    result = mx.where(w_negative, -result, result)

    return result


def _quat_inverse(q: mx.array) -> mx.array:
    """Invert unit quaternion via conjugate.

    For unit quaternions: inverse = conjugate = (w, -x, -y, -z)

    Args:
        q: Unit quaternion [..., 4] in (w, x, y, z) format

    Returns:
        Inverted quaternion [..., 4]
    """
    B = q.shape[:-1]
    _assert_shape("q", q, B + (4,))

    result = mx.stack(
        [
            q[..., 0],   # w stays same
            -q[..., 1],  # -x
            -q[..., 2],  # -y
            -q[..., 3],  # -z
        ],
        axis=-1,
    )

    _assert_shape("result", result, B + (4,))
    return result


def _quat_rotate(q: mx.array, v: mx.array) -> mx.array:
    """Rotate vector v by unit quaternion q.

    Computes: q * (0, v) * q_conj
    Using: result = v + 2*cross(q.xyz, cross(q.xyz, v) + w*v)

    Args:
        q: Unit quaternion [..., 4] in (w, x, y, z) format
        v: Vector [..., 3]

    Returns:
        Rotated vector [..., 3]
    """
    B = q.shape[:-1]
    _assert_shape("q", q, B + (4,))
    _assert_shape("v", v, B + (3,))

    q = _safe_normalize(q)

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    qvec = mx.stack([x, y, z], axis=-1)
    uv = _cross(qvec, v)
    uuv = _cross(qvec, uv)
    uv = uv * (2.0 * mx.expand_dims(w, axis=-1))
    uuv = uuv * 2.0
    result = v + uv + uuv

    _assert_shape("result", result, B + (3,))
    return result


def _quat_multiply(q1: mx.array, q2: mx.array, dim: int = -1) -> mx.array:
    """Multiply two quaternions (w, x, y, z format).

    Args:
        q1: First quaternion with 4 components along 'dim'
        q2: Second quaternion with 4 components along 'dim'
        dim: Dimension along which the quaternions are stored

    Returns:
        Product q1 * q2 as quaternion with 4 components along 'dim'
    """
    dim = dim + q1.ndim if dim < 0 else dim
    A = q1.shape[:dim]
    B = q1.shape[dim + 1:]
    _assert_shape("q1", q1, A + (4,) + B)
    _assert_shape("q2", q2, A + (4,) + B)

    # Use slicing with narrow equivalent
    # narrow(tensor, dim, start, length) -> slicing
    def _narrow(arr, d, start, length):
        slices = [slice(None)] * arr.ndim
        slices[d] = slice(start, start + length)
        return arr[tuple(slices)]

    w1 = _narrow(q1, dim, 0, 1)
    v1 = _narrow(q1, dim, 1, 3)
    w2 = _narrow(q2, dim, 0, 1)
    v2 = _narrow(q2, dim, 1, 3)

    w = w1 * w2 - mx.sum(v1 * v2, axis=dim, keepdims=True)

    # Cross product along the quaternion dim
    # For dim=-1, this is straightforward
    v_cross = _cross_along_dim(v1, v2, dim)
    v = w1 * v2 + w2 * v1 + v_cross

    result = mx.concatenate([w, v], axis=dim)

    _assert_shape("result", result, A + (4,) + B)
    return result


def _cross_along_dim(a: mx.array, b: mx.array, dim: int) -> mx.array:
    """Cross product along a specific dimension.

    Args:
        a: [..., 3, ...] array with 3 elements along dim
        b: [..., 3, ...] array with 3 elements along dim
        dim: Dimension along which to compute cross product

    Returns:
        [..., 3, ...] cross product
    """
    def _sel(arr, d, idx):
        slices = [slice(None)] * arr.ndim
        slices[d] = slice(idx, idx + 1)
        return arr[tuple(slices)]

    a0, a1, a2 = _sel(a, dim, 0), _sel(a, dim, 1), _sel(a, dim, 2)
    b0, b1, b2 = _sel(b, dim, 0), _sel(b, dim, 1), _sel(b, dim, 2)

    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    return mx.concatenate([c0, c1, c2], axis=dim)


def _quat_slerp(x: mx.array, y: mx.array, t: mx.array) -> mx.array:
    """Spherical linear interpolation between two quaternions.

    Args:
        x: Start quaternion [..., 4] (normalized)
        y: End quaternion [..., 4] (normalized)
        t: Interpolation parameter [...] in [0, 1]

    Returns:
        Interpolated quaternion [..., 4]
    """
    B = x.shape[:-1]
    _assert_shape("x", x, B + (4,))
    _assert_shape("y", y, B + (4,))
    _assert_shape("t", t, B)

    a = mx.expand_dims(t, axis=-1)

    # Compute dot product (cosTheta)
    cosTheta = mx.sum(x * y, axis=-1, keepdims=True)

    # If cosTheta < 0, negate y to take short path
    z = mx.where(cosTheta < 0, -y, y)
    cosTheta = mx.abs(cosTheta)

    threshold = 1.0 - 1e-6

    # Linear interpolation result
    resultLerp = (1.0 - a) * x + a * z

    # Slerp result
    # Clamp cosTheta for acos stability
    cosTheta_clamped = mx.clip(cosTheta, -1.0, 1.0)
    theta = mx.arccos(cosTheta_clamped)
    sinTheta = mx.sin(theta)
    # Avoid division by zero — where sinTheta is 0, use lerp
    safe_sinTheta = mx.where(sinTheta > 1e-10, sinTheta, mx.ones_like(sinTheta))
    resultSlerp = (
        mx.sin((1.0 - a) * theta) * x + mx.sin(a * theta) * z
    ) / safe_sinTheta

    # Use lerp where close, slerp where distant
    result = mx.where(cosTheta > threshold, resultLerp, resultSlerp)

    _assert_shape("result", result, B + (4,))
    return result


def _quat_to_rotmat(quats: mx.array) -> mx.array:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        quats: [..., 4] quaternion in (w, x, y, z) format

    Returns:
        [..., 3, 3] rotation matrix
    """
    # Normalize: F.normalize(quats, p=2, dim=-1)
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True) + 1e-12)
    quats = quats / norm

    w = quats[..., 0]
    x = quats[..., 1]
    y = quats[..., 2]
    z = quats[..., 3]

    R = mx.stack(
        [
            1 - 2 * (y ** 2 + z ** 2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x ** 2 + z ** 2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x ** 2 + y ** 2),
        ],
        axis=-1,
    )
    return mx.reshape(R, quats.shape[:-1] + (3, 3))


def _quat_scale_to_matrix(
    quats: mx.array,  # [..., 4]
    scales: mx.array,  # [..., 3]
) -> mx.array:
    """Convert quaternion and scale to a 3x3 matrix (R * S).

    Args:
        quats: [..., 4] quaternion
        scales: [..., 3] scale factors

    Returns:
        [..., 3, 3] matrix R * diag(s)
    """
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    R = _quat_to_rotmat(quats)  # [..., 3, 3]
    M = R * mx.expand_dims(scales, axis=-2)  # [..., 3, 3]
    return M


def _quat_scale_to_covar_preci(
    quats: mx.array,  # [..., 4]
    scales: mx.array,  # [..., 3]
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    """Convenience wrapper. Canonical implementation in covariance.py."""
    from gsplat_mlx.core.covariance import quat_scale_to_covar_preci
    return quat_scale_to_covar_preci(quats, scales, compute_covar, compute_preci, triu)


def _quat_scale_to_preci_half(quats: mx.array, scales: mx.array) -> mx.array:
    """Compute M = R * diag(1/s).

    Args:
        quats: [..., 4] quaternion
        scales: [..., 3] scale factors

    Returns:
        [..., 3, 3] precision half matrix
    """
    R = _quat_to_rotmat(quats)
    M = R * (1.0 / mx.expand_dims(scales, axis=-2))
    return M


# ============================================================================
# Polynomial Utilities
# ============================================================================


def compute_inverse_polynomial(forward_poly_coeffs, input_range, num_samples=1000):
    """Compute the inverse polynomial coefficients using least squares fitting.

    Given a polynomial f(x) = c0 + c1*x + c2*x^2 + ... + c5*x^5,
    compute g(y) such that g(f(x)) ~ x using least squares fitting.

    Args:
        forward_poly_coeffs: List or array of 6 coefficients [c0, c1, c2, c3, c4, c5]
        input_range: Tuple (min_val, max_val) for sampling input values
        num_samples: Number of sample points for fitting

    Returns:
        List of 6 inverse polynomial coefficients [k0, k1, k2, k3, k4, k5]

    Raises:
        ValueError: If forward polynomial produces invalid values or if inverse
                   accuracy is insufficient (max error > 0.1% of range)
    """
    import numpy as np

    # Sample uniformly across input range
    x_samples = np.linspace(
        input_range[0], input_range[1], num_samples, dtype=np.float64
    )

    # Evaluate forward polynomial: y = f(x)
    forward_coeffs_desc = np.array(forward_poly_coeffs[::-1], dtype=np.float64)
    y_samples = np.polyval(forward_coeffs_desc, x_samples)

    # Check for numerical issues
    if np.any(np.isnan(y_samples)) or np.any(np.isinf(y_samples)):
        raise ValueError("Forward polynomial evaluation produced NaN or Inf values")

    # Fit inverse: x = g(y) with constraint k0=0
    A = np.column_stack([y_samples ** i for i in range(1, 6)])

    # Solve least squares with Tikhonov regularization
    lambda_reg = 1e-10
    AtA = A.T @ A + lambda_reg * np.eye(5)
    Atb = A.T @ x_samples
    inverse_coeffs_no_k0 = np.linalg.solve(AtA, Atb)

    # Prepend k0=0
    inverse_coeffs = np.concatenate([[0.0], inverse_coeffs_no_k0])

    if np.any(np.isnan(inverse_coeffs)) or np.any(np.isinf(inverse_coeffs)):
        raise ValueError("Least squares solution produced NaN or Inf values")

    # Verify inverse accuracy
    test_indices = np.linspace(0, len(x_samples) - 1, 100, dtype=int)
    x_reconstructed = np.polyval(inverse_coeffs[::-1], y_samples[test_indices])
    errors = np.abs(x_reconstructed - x_samples[test_indices])
    max_error = float(np.max(errors))

    tolerance = (input_range[1] - input_range[0]) * 1e-3
    if max_error > tolerance:
        raise ValueError(
            f"Inverse polynomial accuracy insufficient: max_error={max_error:.6e} "
            f"exceeds tolerance={tolerance:.6e}. Try increasing num_samples."
        )

    return inverse_coeffs.astype(np.float32).tolist()
