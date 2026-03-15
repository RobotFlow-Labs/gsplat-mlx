# PRD-03: Quaternion-Scale to Covariance and Precision Matrices

## Overview

Port the `_QuatScaleToCovarPreci` autograd.Function from gsplat's CUDA backend to Apple MLX using `@mx.custom_function` with an explicit `.vjp` backward pass. This module converts each 3D Gaussian's parameterization -- a unit quaternion encoding rotation and a 3-vector encoding anisotropic scale -- into a full 3x3 covariance matrix (and optionally its inverse, the precision matrix). It is invoked for every Gaussian on every forward pass, making it one of the most performance-critical primitives in the entire pipeline.

### Mathematical Background

In 3D Gaussian Splatting, each Gaussian is parameterized by:
- **Mean position** (3D): `mu` -- the center of the Gaussian
- **Quaternion rotation** (4D): `q = (w, x, y, z)` -- encodes the orientation
- **Scale** (3D): `s = (sx, sy, sz)` -- encodes the anisotropic extent along each principal axis

The **covariance matrix** is constructed as:

```
Sigma = R * S * S^T * R^T = M * M^T    where M = R * diag(s)
```

The **precision matrix** (inverse covariance) is:

```
Sigma^{-1} = R * S^{-1} * (S^{-1})^T * R^T = P * P^T    where P = R * diag(1/s)
```

Where `R = quat_to_rotmat(normalize(q))` is the 3x3 rotation matrix derived from the quaternion.

This factored form (`M * M^T`) guarantees symmetric positive semi-definiteness by construction, which is a hard requirement for valid Gaussian distributions.

## Source Reference

- **Forward (Python)**: `repositories/gsplat-upstream/gsplat/cuda/_math.py:680-710` (`_quat_scale_to_covar_preci`)
- **autograd.Function wrapper**: `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:1323-1367` (`_QuatScaleToCovarPreci`)
- **CUDA forward kernel**: `repositories/gsplat-upstream/gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu:37-105` (`quat_scale_to_covar_preci_fwd_kernel`)
- **CUDA backward kernel**: `repositories/gsplat-upstream/gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu:151-237` (`quat_scale_to_covar_preci_bwd_kernel`)
- **CUDA VJP helpers**: `repositories/gsplat-upstream/gsplat/cuda/include/Utils.cuh:241-320` (`quat_scale_to_covar_vjp`, `quat_scale_to_preci_vjp`)
- **CUDA quat_to_rotmat_vjp**: `repositories/gsplat-upstream/gsplat/cuda/include/Utils.cuh:184-206`
- **Supporting (from PRD-02)**: `_quat_to_rotmat`, `_safe_normalize`, `_quat_scale_to_matrix`
- **Lines**: ~30 lines forward Python, ~80 lines CUDA backward, ~45 lines autograd wrapper

## Scope

### In Scope
- `quat_scale_to_covar_preci(quats, scales, compute_covar, compute_preci, triu)` -- public API function
- Forward pass: quaternion normalization, rotation matrix construction, covariance via `M @ M^T`, precision via `P @ P^T`
- Backward pass (VJP): explicit custom gradients for quaternion and scale parameters, matching the CUDA backward kernel logic
- Upper-triangle (`triu`) output format option: `[..., 6]` instead of `[..., 3, 3]`
- Batch support with arbitrary leading dimensions `[..., N, 4]` / `[..., N, 3]`
- `quat_to_rotmat_vjp` helper function for reuse in other modules (PRD-05 projection)

### Out of Scope
- CUDA kernel implementation (we port the math to pure MLX)
- Sparse gradient support (deferred to packed variants)
- The `_quat_to_rotmat` function itself (implemented in PRD-02's `math_utils.py`)
- The `_quat_scale_to_covar_preci_half` variant (used in 2DGS, deferred to PRD-12)
- Metal shader kernels (future optimization)

## Technical Design

### File: `src/gsplat_mlx/core/covariance.py`

### Key Functions to Port

| Function | Upstream Location | MLX Approach |
|----------|-------------------|-------------|
| `quat_scale_to_covar_preci` | `_wrapper.py:363-395` | Public API, delegates to `@mx.custom_function` |
| `_quat_scale_to_covar_preci` (forward) | `_math.py:680-710` | Pure MLX: normalize, rotmat, einsum |
| `_QuatScaleToCovarPreci.backward` | `_wrapper.py:1345-1367` | `.vjp` method with explicit derivative formulas |
| `quat_scale_to_covar_vjp` | `Utils.cuh:241-278` | Ported to MLX as vectorized helper |
| `quat_scale_to_preci_vjp` | `Utils.cuh:280-320` | Ported to MLX as vectorized helper |
| `quat_to_rotmat_vjp` | `Utils.cuh:184-206` | Ported to MLX as vectorized helper |

### Imports and Dependencies

```python
from typing import Optional, Tuple
import mlx.core as mx

# From PRD-02
from gsplat_mlx.core.math_utils import _quat_to_rotmat
```

---

### Forward Pass: Complete Algorithm

The forward pass converts quaternion + scale parameters into covariance and/or precision matrices.

#### Step 1: Quaternion Normalization

```
q_norm = q / ||q||_2
```

where `||q||_2 = sqrt(w^2 + x^2 + y^2 + z^2)`. This ensures the rotation matrix is orthogonal regardless of input quaternion magnitude.

**MLX implementation:**
```python
norm_sq = mx.sum(quats * quats, axis=-1, keepdims=True)
inv_norm = mx.rsqrt(norm_sq + 1e-12)  # epsilon for numerical safety
quats_n = quats * inv_norm
```

Note: We add a small epsilon to avoid division by zero for degenerate zero quaternions. The upstream CUDA uses `rsqrt(x^2 + y^2 + z^2 + w^2)` without epsilon, but MLX's lazy evaluation means we should be defensive.

#### Step 2: Quaternion to Rotation Matrix

Using `_quat_to_rotmat` from PRD-02 (which internally normalizes, but we pre-normalize for the VJP):

```
w, x, y, z = q_norm

R = | 1 - 2(y^2 + z^2)    2(xy - wz)         2(xz + wy)      |
    | 2(xy + wz)           1 - 2(x^2 + z^2)   2(yz - wx)      |
    | 2(xz - wy)           2(yz + wx)          1 - 2(x^2 + y^2)|
```

Output shape: `[..., 3, 3]`

**Important**: The upstream CUDA stores matrices in column-major (GLM convention), while PyTorch and MLX use row-major. The rotation matrix elements above are in row-major order, matching the PyTorch `_quat_to_rotmat` reference.

#### Step 3: Covariance Matrix (if `compute_covar=True`)

```python
M = R * scales[..., None, :]   # [..., 3, 3] -- broadcasts scale along rows
# This is equivalent to R @ diag(s), but more efficient as element-wise multiply
# M[:, j] = R[:, j] * s[j]

covars = einsum("...ij,...kj->...ik", M, M)   # M @ M^T
# covars[i, k] = sum_j M[i, j] * M[k, j]
```

The result is guaranteed to be symmetric positive semi-definite by construction.

#### Step 4: Precision Matrix (if `compute_preci=True`)

```python
P = R * (1.0 / scales)[..., None, :]   # [..., 3, 3] -- R @ diag(1/s)

precis = einsum("...ij,...kj->...ik", P, P)   # P @ P^T
```

**Mathematical identity**: `covars @ precis = (R S S^T R^T)(R S^{-1} S^{-T} R^T) = R S S^T S^{-1} S^{-T} R^T = R I R^T = I`

#### Step 5: Upper Triangle Extraction (if `triu=True`)

For a symmetric 3x3 matrix, only 6 unique values are needed:

```
Full matrix indices:      Upper triangle (triu) indices:
| [0,0]  [0,1]  [0,2] |   triu[0] = [0,0]  (a)
| [1,0]  [1,1]  [1,2] |   triu[1] = [0,1]  (b)
| [2,0]  [2,1]  [2,2] |   triu[2] = [0,2]  (c)
                            triu[3] = [1,1]  (d)
                            triu[4] = [1,2]  (e)
                            triu[5] = [2,2]  (f)
```

The upstream averages symmetric pairs to ensure exact symmetry:

```python
flat = reshape(matrix, (..., 9))
# Indices: [0,1,2,4,5,8] = positions [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]  (row-major upper)
# Indices: [0,3,6,4,7,8] = positions [0,0], [1,0], [2,0], [1,1], [2,1], [2,2]  (row-major lower)
triu = (flat[..., [0,1,2,4,5,8]] + flat[..., [0,3,6,4,7,8]]) / 2.0
```

This averaging handles any floating-point asymmetry from the `M @ M^T` computation.

#### Complete Forward Implementation

```python
@mx.custom_function
def _quat_scale_to_covar_preci_impl(
    quats: mx.array,      # [..., N, 4]
    scales: mx.array,     # [..., N, 3]
    compute_covar_flag: mx.array,  # scalar bool-as-int (MLX custom_function requires array args)
    compute_preci_flag: mx.array,  # scalar bool-as-int
    triu_flag: mx.array,           # scalar bool-as-int
) -> Tuple[mx.array, mx.array]:
    """Core implementation with @mx.custom_function for VJP support."""
    compute_covar = bool(compute_covar_flag.item())
    compute_preci = bool(compute_preci_flag.item())
    triu = bool(triu_flag.item())

    # Step 1: Build rotation matrix (quat_to_rotmat normalizes internally)
    R = _quat_to_rotmat(quats)  # [..., 3, 3]

    batch_shape = quats.shape[:-1]

    # Step 2: Covariance
    if compute_covar:
        M = R * mx.expand_dims(scales, axis=-2)  # [..., 3, 3]
        covars = mx.einsum("...ij,...kj->...ik", M, M)  # [..., 3, 3]
        if triu:
            covars_flat = mx.reshape(covars, batch_shape + (9,))
            covars = (
                covars_flat[..., mx.array([0, 1, 2, 4, 5, 8])]
                + covars_flat[..., mx.array([0, 3, 6, 4, 7, 8])]
            ) / 2.0  # [..., 6]
    else:
        if triu:
            covars = mx.zeros(batch_shape + (6,), dtype=quats.dtype)
        else:
            covars = mx.zeros(batch_shape + (3, 3), dtype=quats.dtype)

    # Step 3: Precision
    if compute_preci:
        P = R * mx.expand_dims(1.0 / scales, axis=-2)  # [..., 3, 3]
        precis = mx.einsum("...ij,...kj->...ik", P, P)  # [..., 3, 3]
        if triu:
            precis_flat = mx.reshape(precis, batch_shape + (9,))
            precis = (
                precis_flat[..., mx.array([0, 1, 2, 4, 5, 8])]
                + precis_flat[..., mx.array([0, 3, 6, 4, 7, 8])]
            ) / 2.0  # [..., 6]
    else:
        if triu:
            precis = mx.zeros(batch_shape + (6,), dtype=quats.dtype)
        else:
            precis = mx.zeros(batch_shape + (3, 3), dtype=quats.dtype)

    return covars, precis


def quat_scale_to_covar_preci(
    quats: mx.array,      # [..., N, 4]
    scales: mx.array,     # [..., N, 3]
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    """Convert quaternion + scale to covariance and/or precision matrices.

    Args:
        quats: Quaternion rotations (w, x, y, z), not necessarily normalized. [..., N, 4]
        scales: Scale factors (positive). [..., N, 3]
        compute_covar: Whether to compute covariance matrix.
        compute_preci: Whether to compute precision matrix.
        triu: If True, return upper triangle (6 values) instead of full 3x3.

    Returns:
        covars: Covariance matrices [..., N, 3, 3] or [..., N, 6] if triu.
                None if compute_covar=False.
        precis: Precision matrices [..., N, 3, 3] or [..., N, 6] if triu.
                None if compute_preci=False.
    """
    # Encode booleans as scalar int arrays for @mx.custom_function compatibility
    covar_flag = mx.array(int(compute_covar), dtype=mx.int32)
    preci_flag = mx.array(int(compute_preci), dtype=mx.int32)
    triu_f = mx.array(int(triu), dtype=mx.int32)

    covars, precis = _quat_scale_to_covar_preci_impl(
        quats, scales, covar_flag, preci_flag, triu_f
    )

    return (covars if compute_covar else None,
            precis if compute_preci else None)
```

**Design Note on `@mx.custom_function`**: MLX's `@mx.custom_function` requires all arguments to be `mx.array`. Boolean flags must be encoded as scalar integer arrays and decoded inside the function. The public API `quat_scale_to_covar_preci` wraps this to provide a clean interface with Python booleans and `Optional` returns.

**Alternative approach**: If `@mx.custom_function` proves too restrictive with the flag arguments, wrap only the differentiable computation (the part that depends on `quats` and `scales`) and handle flags in the outer function. The VJP closure can capture the flags from its enclosing scope.

---

### Backward Pass (VJP): Complete Derivation

The VJP computes gradients of the loss `L` with respect to `quats` and `scales`, given the cotangents (upstream gradients) `v_covars = dL/d(covars)` and `v_precis = dL/d(precis)`.

#### VJP for Covariance: `Sigma = M * M^T`

**Step 1: Gradient through `M * M^T`**

Given `D = M * M^T` and the matrix calculus identity for `df/dM` when `f` depends on `M * M^T`:

```
Reference: https://math.stackexchange.com/a/3850121

For D = M * M^T and G = dL/dD:
  dL/dM = (G + G^T) * M
```

This is because `d(M * M^T) = dM * M^T + M * dM^T`, so:

```
dL = tr(G^T * dD) = tr(G^T * (dM * M^T + M * dM^T))
   = tr(G^T * dM * M^T) + tr(G^T * M * dM^T)
   = tr(M^T * G^T * dM) + tr(dM^T * G^T * M)
   = tr(M^T * G^T * dM) + tr(M^T * G * dM)
   = tr(M^T * (G + G^T) * dM)
```

Therefore:
```python
v_M = (v_covar + v_covar.T) @ M  # or equivalently: (v_covar + transpose(v_covar)) @ M
```

**Step 2: Gradient through `M = R * diag(s)` w.r.t. scales**

Since `M[:, j] = R[:, j] * s[j]`, we have `dM[:, j]/ds[j] = R[:, j]`. Thus:

```
v_scale[j] = sum_i(R[i, j] * v_M[i, j]) = dot(R[:, j], v_M[:, j])
```

In vectorized form:
```python
v_scale = mx.sum(R * v_M, axis=-2)  # [..., 3]
```

**Note**: The upstream CUDA code computes this with GLM column-major indexing:
```c
v_scale[0] += R[0][0]*v_M[0][0] + R[0][1]*v_M[0][1] + R[0][2]*v_M[0][2];
```
In GLM, `R[col][row]`, so `R[0][0], R[0][1], R[0][2]` is the first column. In row-major (MLX), this corresponds to `R[:, 0]` dotted with `v_M[:, 0]`, which is exactly `mx.sum(R * v_M, axis=-2)[..., 0]`.

**Step 3: Gradient through `M = R * diag(s)` w.r.t. R**

```
v_R = v_M * diag(s) = v_M * s[..., None, :]
```

In vectorized form:
```python
v_R = v_M * mx.expand_dims(scales, axis=-2)  # [..., 3, 3]
```

#### VJP for Precision: `Sigma^{-1} = P * P^T`

The precision VJP follows the same pattern as covariance, but with `P = R * diag(1/s)`.

**Step 1: Same M*M^T gradient rule:**
```python
v_P = (v_preci + v_preci.T) @ P
```

**Step 2: Gradient w.r.t. scales (through `1/s`):**

Since `P[:, j] = R[:, j] / s[j]`, using the chain rule through `1/s`:
```
d(1/s[j])/ds[j] = -1/s[j]^2
```

Therefore:
```python
v_scale_from_preci[j] = -1/s[j]^2 * dot(R[:, j], v_P[:, j])
```

In vectorized form:
```python
v_scale += -(1.0 / scales)**2 * mx.sum(R * v_P, axis=-2)
```

**Step 3: Gradient w.r.t. R (same structure):**
```python
v_R += v_P * mx.expand_dims(1.0 / scales, axis=-2)
```

#### VJP for Quaternion: `R = quat_to_rotmat(q_normalized)`

This is the most mathematically involved part. The gradient flows through:
1. `R` to the normalized quaternion `q_n = (w, x, y, z)`
2. `q_n` to the original quaternion `q` through normalization

**Step 1: dL/dq_n from dL/dR**

Given the rotation matrix formula and `v_R = dL/dR`, the upstream CUDA computes (from `Utils.cuh:184-206`):

```
v_quat_n[0] (v_w) = 2 * (x*(v_R[1][2] - v_R[2][1]) + y*(v_R[2][0] - v_R[0][2]) + z*(v_R[0][1] - v_R[1][0]))
v_quat_n[1] (v_x) = 2 * (-2*x*(v_R[1][1] + v_R[2][2]) + y*(v_R[0][1] + v_R[1][0]) + z*(v_R[0][2] + v_R[2][0]) + w*(v_R[1][2] - v_R[2][1]))
v_quat_n[2] (v_y) = 2 * (x*(v_R[0][1] + v_R[1][0]) - 2*y*(v_R[0][0] + v_R[2][2]) + z*(v_R[1][2] + v_R[2][1]) + w*(v_R[2][0] - v_R[0][2]))
v_quat_n[3] (v_z) = 2 * (x*(v_R[0][2] + v_R[2][0]) + y*(v_R[1][2] + v_R[2][1]) - 2*z*(v_R[0][0] + v_R[1][1]) + w*(v_R[0][1] - v_R[1][0]))
```

**Important GLM-to-row-major translation**: The CUDA code uses GLM column-major convention where `v_R[col][row]`. In row-major (MLX), `v_R[row, col]`. The mapping is:
```
GLM v_R[0][0] = row-major v_R[0, 0]
GLM v_R[0][1] = row-major v_R[1, 0]
GLM v_R[0][2] = row-major v_R[2, 0]
GLM v_R[1][0] = row-major v_R[0, 1]
GLM v_R[1][1] = row-major v_R[1, 1]
GLM v_R[1][2] = row-major v_R[2, 1]
GLM v_R[2][0] = row-major v_R[0, 2]
GLM v_R[2][1] = row-major v_R[1, 2]
GLM v_R[2][2] = row-major v_R[2, 2]
```

Translating to row-major indexing (MLX convention), and using `vR` for `v_R`:
```
v_w = 2 * (x*(vR[2,1] - vR[1,2]) + y*(vR[0,2] - vR[2,0]) + z*(vR[1,0] - vR[0,1]))
v_x = 2 * (-2*x*(vR[1,1] + vR[2,2]) + y*(vR[1,0] + vR[0,1]) + z*(vR[2,0] + vR[0,2]) + w*(vR[2,1] - vR[1,2]))
v_y = 2 * (x*(vR[1,0] + vR[0,1]) - 2*y*(vR[0,0] + vR[2,2]) + z*(vR[2,1] + vR[1,2]) + w*(vR[0,2] - vR[2,0]))
v_z = 2 * (x*(vR[2,0] + vR[0,2]) + y*(vR[2,1] + vR[1,2]) - 2*z*(vR[0,0] + vR[1,1]) + w*(vR[1,0] - vR[0,1]))
```

**Derivation**: Each element of R is a polynomial in `(w, x, y, z)`. For example:
- `R[0,0] = 1 - 2(y^2 + z^2)` => `dR[0,0]/dw = 0`, `dR[0,0]/dx = 0`, `dR[0,0]/dy = -4y`, `dR[0,0]/dz = -4z`
- `R[0,1] = 2(xy - wz)` => `dR[0,1]/dw = -2z`, `dR[0,1]/dx = 2y`, `dR[0,1]/dy = 2x`, `dR[0,1]/dz = -2w`

Summing `v_R[i,j] * dR[i,j]/dq_k` over all 9 elements yields the formulas above.

**Step 2: Chain through quaternion normalization**

The normalized quaternion is `q_n = q / ||q||`. The VJP of normalization (projection onto tangent plane of the unit sphere) is:

```
v_quat = (v_quat_n - dot(v_quat_n, q_n) * q_n) * inv_norm
```

where `inv_norm = 1 / ||q||`. This projects out the component of the gradient along the quaternion direction (since changes along `q` don't change `q_n`).

**MLX implementation**:
```python
dot_product = mx.sum(v_quat_n * quat_n, axis=-1, keepdims=True)
v_quats = (v_quat_n - dot_product * quat_n) * inv_norm
```

#### VJP for triu Mode

When `triu=True`, the upstream gradient `v_covars` has shape `[..., 6]`. Before applying the VJP formulas above, we must expand it back to a full symmetric 3x3 matrix. The upstream CUDA does this with halved off-diagonal entries:

```
v_covar_full = | v[0]      v[1]*0.5  v[2]*0.5 |
               | v[1]*0.5  v[3]      v[4]*0.5 |
               | v[2]*0.5  v[4]*0.5  v[5]     |
```

The factor of 0.5 on off-diagonals compensates for the fact that the forward pass averaged `(M[i,j] + M[j,i]) / 2` for the triu representation, and each off-diagonal gradient contributes to two matrix entries.

**MLX implementation**:
```python
def _triu6_to_symmetric_3x3(triu6: mx.array) -> mx.array:
    """Expand [..., 6] upper triangle to [..., 3, 3] symmetric matrix.

    For gradient flow: off-diagonals are halved since they contribute
    to two entries in the symmetric matrix.
    """
    a, b, c, d, e, f = (triu6[..., i:i+1] for i in range(6))
    row0 = mx.concatenate([a, b * 0.5, c * 0.5], axis=-1)
    row1 = mx.concatenate([b * 0.5, d, e * 0.5], axis=-1)
    row2 = mx.concatenate([c * 0.5, e * 0.5, f], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)  # [..., 3, 3]
```

#### Complete VJP Implementation

```python
@_quat_scale_to_covar_preci_impl.vjp
def _covar_preci_vjp(primals, cotangents, outputs):
    quats, scales, covar_flag, preci_flag, triu_flag = primals
    v_covars, v_precis = cotangents

    compute_covar = bool(covar_flag.item())
    compute_preci = bool(preci_flag.item())
    triu = bool(triu_flag.item())

    # Recompute intermediates
    norm_sq = mx.sum(quats * quats, axis=-1, keepdims=True)
    inv_norm = mx.rsqrt(norm_sq + 1e-12)
    quat_n = quats * inv_norm
    w = quat_n[..., 0]
    x = quat_n[..., 1]
    y = quat_n[..., 2]
    z = quat_n[..., 3]

    R = _quat_to_rotmat(quats)  # [..., 3, 3], normalizes internally

    v_quat_n = mx.zeros_like(quat_n)  # [..., 4]
    v_scales = mx.zeros_like(scales)   # [..., 3]

    if compute_covar and v_covars is not None:
        # Expand triu gradient to full 3x3 if needed
        if triu:
            v_covar_full = _triu6_to_symmetric_3x3(v_covars)
        else:
            v_covar_full = v_covars

        # M = R * diag(s)
        M = R * mx.expand_dims(scales, axis=-2)

        # dL/dM = (G + G^T) @ M
        v_M = (v_covar_full + mx.swapaxes(v_covar_full, -1, -2)) @ M

        # dL/dR from covariance path
        v_R = v_M * mx.expand_dims(scales, axis=-2)

        # dL/d(scales) from covariance path
        v_scales = v_scales + mx.sum(R * v_M, axis=-2)

        # dL/d(quat_n) from dL/dR
        v_quat_n = v_quat_n + _quat_to_rotmat_vjp_qn(w, x, y, z, v_R)

    if compute_preci and v_precis is not None:
        # Expand triu gradient to full 3x3 if needed
        if triu:
            v_preci_full = _triu6_to_symmetric_3x3(v_precis)
        else:
            v_preci_full = v_precis

        inv_scales = 1.0 / scales

        # P = R * diag(1/s)
        P = R * mx.expand_dims(inv_scales, axis=-2)

        # dL/dP = (G + G^T) @ P
        v_P = (v_preci_full + mx.swapaxes(v_preci_full, -1, -2)) @ P

        # dL/dR from precision path
        v_R_preci = v_P * mx.expand_dims(inv_scales, axis=-2)

        # dL/d(scales) from precision path: chain through 1/s
        # d(1/s)/ds = -1/s^2
        v_scales = v_scales + (-(inv_scales ** 2) * mx.sum(R * v_P, axis=-2))

        # dL/d(quat_n) from precision path
        v_quat_n = v_quat_n + _quat_to_rotmat_vjp_qn(w, x, y, z, v_R_preci)

    # Chain through quaternion normalization
    # v_quats = (v_quat_n - dot(v_quat_n, quat_n) * quat_n) * inv_norm
    dot_product = mx.sum(v_quat_n * quat_n, axis=-1, keepdims=True)
    v_quats = (v_quat_n - dot_product * quat_n) * inv_norm

    # No gradients for the boolean flag arrays
    return (v_quats, v_scales, mx.zeros_like(covar_flag),
            mx.zeros_like(preci_flag), mx.zeros_like(triu_flag))
```

#### Helper: `_quat_to_rotmat_vjp_qn`

This computes `dL/dq_n` given `dL/dR`, where `q_n` is the already-normalized quaternion.

```python
def _quat_to_rotmat_vjp_qn(
    w: mx.array,    # [...], already normalized
    x: mx.array,    # [...]
    y: mx.array,    # [...]
    z: mx.array,    # [...]
    v_R: mx.array,  # [..., 3, 3], row-major
) -> mx.array:     # [..., 4]
    """Compute dL/d(normalized_quat) given dL/dR.

    v_R is in row-major order: v_R[..., i, j] = dL/dR_{ij}.
    """
    # Extract v_R components (row-major)
    vR00 = v_R[..., 0, 0]; vR01 = v_R[..., 0, 1]; vR02 = v_R[..., 0, 2]
    vR10 = v_R[..., 1, 0]; vR11 = v_R[..., 1, 1]; vR12 = v_R[..., 1, 2]
    vR20 = v_R[..., 2, 0]; vR21 = v_R[..., 2, 1]; vR22 = v_R[..., 2, 2]

    v_w = 2.0 * (
        x * (vR21 - vR12)
        + y * (vR02 - vR20)
        + z * (vR10 - vR01)
    )
    v_x = 2.0 * (
        -2.0 * x * (vR11 + vR22)
        + y * (vR10 + vR01)
        + z * (vR20 + vR02)
        + w * (vR21 - vR12)
    )
    v_y = 2.0 * (
        x * (vR10 + vR01)
        - 2.0 * y * (vR00 + vR22)
        + z * (vR21 + vR12)
        + w * (vR02 - vR20)
    )
    v_z = 2.0 * (
        x * (vR20 + vR02)
        + y * (vR21 + vR12)
        - 2.0 * z * (vR00 + vR11)
        + w * (vR10 - vR01)
    )

    return mx.stack([v_w, v_x, v_y, v_z], axis=-1)  # [..., 4]
```

**Verification**: This function is a direct translation of `quat_to_rotmat_vjp` from `Utils.cuh:184-206`, with the GLM column-major indexing translated to row-major. The normalization projection step (`(v_qn - dot(v_qn, qn) * qn) * inv_norm`) is handled in the caller, not in this helper, matching the CUDA decomposition.

---

### Data Flow

```
Input:                          Output:
quats [..., N, 4] ──┐
                     ├──> quat_scale_to_covar_preci ──> covars [..., N, 3, 3] or [..., N, 6]
scales [..., N, 3] ──┘                               └──> precis [..., N, 3, 3] or [..., N, 6]

Internal flow:
quats -> normalize -> quat_to_rotmat -> R [..., N, 3, 3]
                                         │
                                         ├── * diag(s) -> M -> M @ M^T -> covars
                                         │
                                         └── * diag(1/s) -> P -> P @ P^T -> precis
```

### Tensor Shapes

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `quats` (input) | `[..., N, 4]` | float32 | (w, x, y, z) order, unnormalized OK |
| `scales` (input) | `[..., N, 3]` | float32 | Must be positive (no validation) |
| `R` (internal) | `[..., N, 3, 3]` | float32 | Orthogonal rotation matrix |
| `M` (internal) | `[..., N, 3, 3]` | float32 | R * diag(s), covariance half |
| `P` (internal) | `[..., N, 3, 3]` | float32 | R * diag(1/s), precision half |
| `covars` (output) | `[..., N, 3, 3]` | float32 | Symmetric PSD, full matrix |
| `covars` (output, triu) | `[..., N, 6]` | float32 | Upper triangle of covariance |
| `precis` (output) | `[..., N, 3, 3]` | float32 | Symmetric PSD, full matrix |
| `precis` (output, triu) | `[..., N, 6]` | float32 | Upper triangle of precision |
| `v_quats` (gradient) | `[..., N, 4]` | float32 | Gradient w.r.t. quaternions |
| `v_scales` (gradient) | `[..., N, 3]` | float32 | Gradient w.r.t. scales |

### Edge Cases and Numerical Considerations

1. **Zero quaternion**: If `||q|| = 0`, the normalization produces `NaN`. The epsilon in `rsqrt(norm_sq + 1e-12)` prevents this. In practice, optimizers should never produce exactly-zero quaternions, but initialization might.

2. **Very small scales**: When `s[j] -> 0`, the precision matrix `P = R * diag(1/s)` produces very large values. No clamping is applied to match upstream behavior. Users should ensure scales are bounded away from zero (typically `s >= 1e-7` in practice).

3. **Very large scales**: The covariance matrix entries grow as `s^2`. For `s = 1e4`, entries reach `1e8`, which is fine for float32 (max ~3.4e38).

4. **Unnormalized quaternions**: The forward pass normalizes quaternions internally, so unnormalized inputs are valid. The VJP correctly handles the normalization gradient.

5. **Negative quaternion components**: `q` and `-q` represent the same rotation. The normalization handles this correctly; both produce the same rotation matrix.

6. **triu averaging**: The averaging `(upper + lower) / 2` in triu mode ensures exact symmetry even with floating-point errors from `einsum`. This matches upstream behavior.

7. **Gradient accumulation**: When both `compute_covar` and `compute_preci` are True, gradients from both paths are accumulated (summed) into `v_quats` and `v_scales`, matching the CUDA kernel which uses `+=` operators.

---

## Test Plan

### File: `tests/test_covariance.py`

### Forward Tests

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_covar_identity_rotation` | `q=(1,0,0,0)`, `s=(1,1,1)` | `covars = I_3` (3x3 identity) |
| `test_covar_scaled_identity_rotation` | `q=(1,0,0,0)`, `s=(2,3,4)` | `covars = diag(4, 9, 16)` |
| `test_covar_90deg_x_rotation` | `q=(cos(pi/4), sin(pi/4), 0, 0)`, `s=(1,2,3)` | Known rotated diagonal |
| `test_covar_90deg_y_rotation` | 90-degree rotation around Y axis | Verify permuted eigenvalues |
| `test_covar_90deg_z_rotation` | 90-degree rotation around Z axis | Verify permuted eigenvalues |
| `test_covar_symmetric` | Random quats/scales, N=1000 | `covars == covars^T` within `atol=1e-6` |
| `test_covar_positive_semidefinite` | Random quats/scales, N=1000 | All eigenvalues >= -1e-6 |
| `test_covar_eigenvalues_match_scales` | Random quats, known scales | Eigenvalues = s^2 (sorted) |
| `test_preci_identity_rotation` | `q=(1,0,0,0)`, `s=(1,1,1)` | `precis = I_3` |
| `test_preci_scaled_identity_rotation` | `q=(1,0,0,0)`, `s=(2,3,4)` | `precis = diag(1/4, 1/9, 1/16)` |
| `test_preci_inverse_of_covar` | Random quats/scales, N=100 | `covars @ precis = I` within `atol=1e-4` |
| `test_triu_mode_covar` | Compare triu output against manually extracted upper triangle | Element-wise match |
| `test_triu_mode_preci` | Same for precision | Element-wise match |
| `test_triu_roundtrip` | Extract triu, reconstruct full matrix, compare | Match within `atol=1e-6` |
| `test_batch_dims_1d` | `quats=[100, 4]`, `scales=[100, 3]` | Correct output shapes |
| `test_batch_dims_2d` | `quats=[2, 50, 4]`, `scales=[2, 50, 3]` | Correct output shapes |
| `test_batch_dims_3d` | `quats=[2, 3, 10, 4]`, `scales=[2, 3, 10, 3]` | Correct output shapes |
| `test_batch_consistency` | Batch result matches element-wise loop | Element-wise match |
| `test_compute_covar_only` | `compute_preci=False` | Returns `(covars, None)` |
| `test_compute_preci_only` | `compute_covar=False` | Returns `(None, precis)` |
| `test_compute_neither` | Both False | Returns `(None, None)` |
| `test_unnormalized_quats` | `q = (2, 0, 0, 0)` (magnitude 2) | Same result as `q = (1, 0, 0, 0)` |
| `test_negative_quat` | `q` and `-q` produce same covariance | Identical output |
| `test_determinant_equals_scale_product` | Random inputs | `det(covar) = (s1*s2*s3)^2` |

#### Detailed Test: `test_covar_90deg_x_rotation`

```python
def test_covar_90deg_x_rotation():
    """90-degree rotation around X axis with anisotropic scale.

    R_x(90) = [[1,  0, 0],
               [0,  0, -1],
               [0,  1,  0]]

    M = R * diag(s) = [[s0, 0,   0  ],
                       [0,  0,   -s2],
                       [0,  s1,  0  ]]

    covar = M @ M^T = [[s0^2,  0,       0     ],
                       [0,     s2^2,    0     ],
                       [0,     0,       s1^2  ]]
    """
    # q for 90-deg around X: (cos(45), sin(45), 0, 0)
    angle = mx.array([math.pi / 2])
    q = mx.array([[math.cos(math.pi/4), math.sin(math.pi/4), 0.0, 0.0]])
    s = mx.array([[2.0, 3.0, 5.0]])

    covars, _ = quat_scale_to_covar_preci(q, s, compute_preci=False)

    expected = mx.array([[[4.0, 0.0, 0.0],
                          [0.0, 25.0, 0.0],
                          [0.0, 0.0, 9.0]]])
    check_all_close(covars, expected, atol=1e-5)
```

#### Detailed Test: `test_preci_inverse_of_covar`

```python
def test_preci_inverse_of_covar(rng):
    """Verify covar @ preci = I for random inputs."""
    N = 100
    quats = mx.array(rng.standard_normal((N, 4)).astype(np.float32))
    scales = mx.array(np.abs(rng.standard_normal((N, 3)).astype(np.float32)) + 0.1)

    covars, precis = quat_scale_to_covar_preci(quats, scales)

    identity = mx.eye(3, dtype=mx.float32)
    product = mx.einsum("...ij,...jk->...ik", covars, precis)  # [N, 3, 3]

    for i in range(N):
        check_all_close(product[i], identity, atol=1e-4,
                        msg=f"covar @ preci != I for Gaussian {i}")
```

#### Detailed Test: `test_triu_roundtrip`

```python
def test_triu_roundtrip(rng):
    """Verify triu format can reconstruct the full matrix."""
    N = 50
    quats = mx.array(rng.standard_normal((N, 4)).astype(np.float32))
    scales = mx.array(np.abs(rng.standard_normal((N, 3)).astype(np.float32)) + 0.1)

    covars_full, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=False)
    covars_triu, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Reconstruct from triu
    # triu indices: [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
    a = covars_triu[..., 0]  # [0,0]
    b = covars_triu[..., 1]  # [0,1]
    c = covars_triu[..., 2]  # [0,2]
    d = covars_triu[..., 3]  # [1,1]
    e = covars_triu[..., 4]  # [1,2]
    f = covars_triu[..., 5]  # [2,2]

    # Compare against full matrix
    check_all_close(a, covars_full[..., 0, 0], atol=1e-6)
    check_all_close(b, covars_full[..., 0, 1], atol=1e-6)
    check_all_close(c, covars_full[..., 0, 2], atol=1e-6)
    check_all_close(d, covars_full[..., 1, 1], atol=1e-6)
    check_all_close(e, covars_full[..., 1, 2], atol=1e-6)
    check_all_close(f, covars_full[..., 2, 2], atol=1e-6)
```

### Backward/VJP Tests

| Test Case | Description | Tolerance |
|-----------|-------------|-----------|
| `test_vjp_quats_covar` | Finite-difference gradient check for quats (covar only) | `atol=1e-4` |
| `test_vjp_scales_covar` | Finite-difference gradient check for scales (covar only) | `atol=1e-4` |
| `test_vjp_quats_preci` | Finite-difference gradient check for quats (preci only) | `atol=1e-4` |
| `test_vjp_scales_preci` | Finite-difference gradient check for scales (preci only) | `atol=1e-4` |
| `test_vjp_both` | Gradient check with both covar and preci | `atol=1e-4` |
| `test_vjp_triu_covar` | Gradient check in triu mode (covar) | `atol=1e-4` |
| `test_vjp_triu_preci` | Gradient check in triu mode (preci) | `atol=1e-4` |
| `test_vjp_batch_2d` | Gradient check with batch dims `[B, N, ...]` | `atol=1e-4` |
| `test_vjp_accumulation` | Verify covar + preci gradients accumulate correctly | `atol=1e-4` |
| `test_vjp_zero_cotangent` | Zero upstream gradient produces zero downstream gradient | exact |

#### Detailed Test: Finite-Difference Gradient Check Pattern

```python
def _finite_diff_gradient(fn, x, idx, eps=1e-4):
    """Compute numerical gradient of scalar fn w.r.t. x at position idx."""
    x_plus = mx.array(np.array(x))
    x_minus = mx.array(np.array(x))
    flat_plus = np.array(x_plus).flatten()
    flat_minus = np.array(x_minus).flatten()
    flat_plus[idx] += eps
    flat_minus[idx] -= eps
    x_plus = mx.array(flat_plus.reshape(x.shape))
    x_minus = mx.array(flat_minus.reshape(x.shape))
    return (fn(x_plus).item() - fn(x_minus).item()) / (2 * eps)


def test_vjp_quats_covar(rng):
    """Verify VJP for quaternion inputs matches finite differences."""
    quats = mx.array(rng.standard_normal((4,)).astype(np.float32))
    scales = mx.array(np.abs(rng.standard_normal((3,)).astype(np.float32)) + 0.5)

    # Scalar loss: sum of all covariance elements
    def loss_fn(q):
        covars, _ = quat_scale_to_covar_preci(q, scales, compute_preci=False)
        return mx.sum(covars)

    # Analytic gradient
    grad_fn = mx.grad(loss_fn)
    analytic_grad = grad_fn(quats)
    mx.eval(analytic_grad)

    # Numerical gradient
    for i in range(4):
        numerical = _finite_diff_gradient(loss_fn, quats, i, eps=1e-4)
        assert abs(np.array(analytic_grad)[i] - numerical) < 1e-3, \
            f"Quaternion gradient mismatch at index {i}: " \
            f"analytic={np.array(analytic_grad)[i]:.6f}, numerical={numerical:.6f}"


def test_vjp_scales_covar(rng):
    """Verify VJP for scale inputs matches finite differences."""
    quats = mx.array(rng.standard_normal((4,)).astype(np.float32))
    scales = mx.array(np.abs(rng.standard_normal((3,)).astype(np.float32)) + 0.5)

    def loss_fn(s):
        covars, _ = quat_scale_to_covar_preci(quats, s, compute_preci=False)
        return mx.sum(covars)

    grad_fn = mx.grad(loss_fn)
    analytic_grad = grad_fn(scales)
    mx.eval(analytic_grad)

    for i in range(3):
        numerical = _finite_diff_gradient(loss_fn, scales, i, eps=1e-4)
        assert abs(np.array(analytic_grad)[i] - numerical) < 1e-3, \
            f"Scale gradient mismatch at index {i}: " \
            f"analytic={np.array(analytic_grad)[i]:.6f}, numerical={numerical:.6f}"
```

### Cross-Framework Tests (requires_torch)

| Test Case | Description | Tolerance |
|-----------|-------------|-----------|
| `test_cross_framework_forward_covar` | Compare MLX covariance output against torch `_quat_scale_to_covar_preci` for 1000 random inputs | `atol=1e-5` |
| `test_cross_framework_forward_preci` | Same for precision matrices | `atol=1e-5` |
| `test_cross_framework_forward_triu` | Compare triu output format | `atol=1e-5` |
| `test_cross_framework_backward_quats` | Compare MLX VJP vs torch autograd for quaternion gradients | `atol=1e-4` |
| `test_cross_framework_backward_scales` | Compare MLX VJP vs torch autograd for scale gradients | `atol=1e-4` |
| `test_cross_framework_backward_both` | Combined covar+preci backward | `atol=1e-4` |
| `test_cross_framework_backward_triu` | Backward in triu mode | `atol=1e-4` |

#### Detailed Test: Cross-Framework Forward

```python
@pytest.mark.requires_torch
def test_cross_framework_forward_covar(rng):
    """Compare MLX and torch forward pass for 1000 random Gaussians."""
    import torch
    import sys
    sys.path.insert(0, "repositories/gsplat-upstream")
    from gsplat.cuda._math import _quat_scale_to_covar_preci as torch_covar_preci

    N = 1000
    quats_np = rng.standard_normal((N, 4)).astype(np.float32)
    scales_np = (np.abs(rng.standard_normal((N, 3))) + 0.1).astype(np.float32)

    # MLX
    quats_mlx = mx.array(quats_np)
    scales_mlx = mx.array(scales_np)
    covars_mlx, precis_mlx = quat_scale_to_covar_preci(quats_mlx, scales_mlx)
    mx.eval(covars_mlx, precis_mlx)

    # Torch
    quats_torch = torch.tensor(quats_np)
    scales_torch = torch.tensor(scales_np)
    covars_torch, precis_torch = torch_covar_preci(quats_torch, scales_torch)

    check_all_close(covars_mlx, covars_torch.numpy(), atol=1e-5,
                    msg="Covariance mismatch between MLX and torch")
    check_all_close(precis_mlx, precis_torch.numpy(), atol=1e-5,
                    msg="Precision mismatch between MLX and torch")
```

#### Detailed Test: Cross-Framework Backward

```python
@pytest.mark.requires_torch
def test_cross_framework_backward_quats(rng):
    """Compare MLX VJP vs torch autograd for quaternion gradients."""
    import torch
    import sys
    sys.path.insert(0, "repositories/gsplat-upstream")
    from gsplat.cuda._math import _quat_scale_to_covar_preci as torch_covar_preci

    N = 50
    quats_np = rng.standard_normal((N, 4)).astype(np.float32)
    scales_np = (np.abs(rng.standard_normal((N, 3))) + 0.1).astype(np.float32)

    # Torch backward
    quats_torch = torch.tensor(quats_np, requires_grad=True)
    scales_torch = torch.tensor(scales_np, requires_grad=True)
    covars_torch, _ = torch_covar_preci(quats_torch, scales_torch, compute_preci=False)
    loss_torch = covars_torch.sum()
    loss_torch.backward()
    grad_quats_torch = quats_torch.grad.numpy()
    grad_scales_torch = scales_torch.grad.numpy()

    # MLX backward
    quats_mlx = mx.array(quats_np)
    scales_mlx = mx.array(scales_np)

    def mlx_loss(q, s):
        c, _ = quat_scale_to_covar_preci(q, s, compute_preci=False)
        return mx.sum(c)

    grad_fn = mx.grad(mlx_loss, argnums=(0, 1))
    grad_quats_mlx, grad_scales_mlx = grad_fn(quats_mlx, scales_mlx)
    mx.eval(grad_quats_mlx, grad_scales_mlx)

    check_all_close(grad_quats_mlx, grad_quats_torch, atol=1e-4,
                    msg="Quaternion gradient mismatch between MLX and torch")
    check_all_close(grad_scales_mlx, grad_scales_torch, atol=1e-4,
                    msg="Scale gradient mismatch between MLX and torch")
```

### Performance Tests

| Test Case | Description | Threshold |
|-----------|-------------|-----------|
| `test_perf_forward_10k` | Forward pass with 10,000 Gaussians | < 10ms on M1/M2 |
| `test_perf_forward_100k` | Forward pass with 100,000 Gaussians | < 100ms on M1/M2 |
| `test_perf_backward_10k` | Backward pass with 10,000 Gaussians | < 20ms on M1/M2 |

```python
@pytest.mark.benchmark
def test_perf_forward_10k(rng, benchmark):
    N = 10_000
    quats = mx.array(rng.standard_normal((N, 4)).astype(np.float32))
    scales = mx.array(np.abs(rng.standard_normal((N, 3)).astype(np.float32)) + 0.1)

    def run():
        covars, precis = quat_scale_to_covar_preci(quats, scales)
        mx.eval(covars, precis)

    benchmark(run)
```

### Tolerances Summary

| Test Category | atol | rtol | Rationale |
|---------------|------|------|-----------|
| Forward (identity/axis-aligned) | 1e-6 | 1e-5 | No chain rule error |
| Forward (random, symmetry/PSD) | 1e-5 | 1e-5 | Standard float32 |
| Forward (cross-framework) | 1e-5 | 1e-5 | Same algorithm, different backends |
| VJP (finite differences) | 1e-3 | 1e-3 | Finite diff has O(eps^2) error |
| VJP (cross-framework) | 1e-4 | 1e-4 | Quaternion chain rule accumulates error |
| triu (roundtrip) | 1e-6 | 1e-6 | Only averaging, minimal error |

---

## Implementation Strategy

### Phase 1: Forward Pass
1. Implement `quat_scale_to_covar_preci` using standard MLX ops (no `@mx.custom_function` yet)
2. Validate all forward tests pass
3. Validate cross-framework forward tests pass

### Phase 2: VJP via Auto-diff
1. Wrap the forward with `@mx.custom_function` but let MLX auto-differentiate through the ops initially
2. Validate VJP tests pass with auto-diff
3. If auto-diff works and is performant, this may be sufficient

### Phase 3: Explicit VJP (if needed)
1. Implement the explicit VJP matching CUDA backward logic
2. Validate VJP tests pass with the explicit implementation
3. Compare performance of explicit VJP vs auto-diff
4. Keep whichever is more performant

**Rationale**: The forward pass is composed entirely of differentiable MLX operations (`normalize`, `einsum`, element-wise multiply, `reshape`). MLX's auto-diff should handle this correctly. The explicit VJP is needed only if:
- Auto-diff is significantly slower due to storing large intermediate computation graphs
- Auto-diff produces incorrect gradients (unlikely given the ops involved)
- We need to save intermediates for efficiency in repeated evaluations

### Alternative: Pure Auto-diff Approach

If `@mx.custom_function` proves problematic with boolean flag arguments, the simplest correct approach is:

```python
def quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=True, triu=False):
    """Pure forward -- MLX auto-diff handles the backward automatically."""
    R = _quat_to_rotmat(quats)
    covars = precis = None

    if compute_covar:
        M = R * mx.expand_dims(scales, axis=-2)
        covars = mx.einsum("...ij,...kj->...ik", M, M)
        if triu:
            covars = _extract_triu(covars)

    if compute_preci:
        P = R * mx.expand_dims(1.0 / scales, axis=-2)
        precis = mx.einsum("...ij,...kj->...ik", P, P)
        if triu:
            precis = _extract_triu(precis)

    return covars, precis
```

This works because `mx.grad` can differentiate through all these ops natively. The `@mx.custom_function` wrapper is only needed if we want to customize the backward for performance.

---

## Dependencies

- **PRD-01**: Dev environment (package structure, test infrastructure, `conftest.py`)
- **PRD-02**: `_quat_to_rotmat` and `_safe_normalize` from `core/math_utils.py`

## Blocks

- **PRD-05** (Projection): Uses covariance matrices to project 3D Gaussians to 2D
- **PRD-09** (Rendering API): Top-level API that calls `quat_scale_to_covar_preci`
- **PRD-12** (2DGS): Uses the `_quat_scale_to_covar_preci_half` variant (deferred)

## Acceptance Criteria

- [ ] `quat_scale_to_covar_preci()` public API exists in `src/gsplat_mlx/core/covariance.py`
- [ ] Forward pass matches torch reference within `atol=1e-5` for 1000 random inputs
- [ ] Covariance matrices are symmetric (`covars == covars^T` within `atol=1e-6`)
- [ ] Covariance matrices are positive semi-definite (all eigenvalues >= -1e-6)
- [ ] `covars @ precis = I` when both computed (within `atol=1e-4`)
- [ ] `det(covars) = (s1*s2*s3)^2` (within `atol=1e-3`)
- [ ] VJP gradients match finite differences within `atol=1e-3`
- [ ] VJP gradients match torch autograd within `atol=1e-4`
- [ ] `triu=True` produces correct 6-element upper triangle format
- [ ] triu gradient correctly expands to 3x3 symmetric gradient
- [ ] Supports arbitrary batch dimensions `[..., N, 4]` / `[..., N, 3]`
- [ ] `compute_covar=False` returns `None` for covars
- [ ] `compute_preci=False` returns `None` for precis
- [ ] Unnormalized quaternions produce valid results (normalization is internal)
- [ ] `q` and `-q` produce identical covariance/precision matrices
- [ ] All tests pass with `pytest tests/test_covariance.py -v`
- [ ] Cross-framework tests pass with `pytest tests/test_covariance.py -v -m requires_torch`
- [ ] No Python loops over individual Gaussians (all operations vectorized)
- [ ] `_quat_to_rotmat_vjp_qn` helper is exposed for reuse by PRD-05
