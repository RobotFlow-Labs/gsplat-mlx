# PRD-02: Math Utilities — Port of `_math.py` to MLX

## 1. Overview

Port **all 782 lines** of `gsplat/cuda/_math.py` from PyTorch to Apple MLX. This module contains the foundational math primitives used by every downstream component: numerically stable norms, polynomial evaluation/inversion, and the complete quaternion algebra (normalize, rotate, multiply, slerp, convert). Getting these right is non-negotiable — every subsequent PRD depends on the numerical correctness established here.

**Source file**: `repositories/gsplat-upstream/gsplat/cuda/_math.py` (782 lines, 22 public symbols)
**Target file**: `src/gsplat_mlx/core/math_utils.py`
**Test file**: `tests/test_math_utils.py`

---

## 2. Source Reference & Line Map

| Symbol | Type | Lines | Category |
|--------|------|-------|----------|
| `_numerically_stable_norm2` | function | 32-73 | Stable numerics |
| `PolynomialProxy` | ABC | 81-118 | Polynomial |
| `FullPolynomialProxy` | class | 121-143 | Polynomial |
| `OddPolynomialProxy` | class | 146-173 | Polynomial |
| `EvenPolynomialProxy` | class | 176-200 | Polynomial |
| `_eval_poly_inverse_horner_newton` | function | 203-272 | Polynomial |
| `SafeNormalize` | autograd.Function | 280-352 | Quaternion/Normalize |
| `_safe_normalize` | function | 355-373 | Quaternion/Normalize |
| `_rotmat_to_quat` | function | 376-454 | Quaternion |
| `_quat_normalize_rotation` | function | 457-474 | Quaternion |
| `_quat_inverse` | function | 477-505 | Quaternion |
| `_quat_rotate` | function | 508-544 | Quaternion |
| `_quat_multiply` | function | 547-579 | Quaternion |
| `_quat_slerp` | function | 582-636 | Quaternion |
| `_quat_scale_to_preci_half` | function | 639-643 | Quaternion/Matrix |
| `_quat_to_rotmat` | function | 646-664 | Quaternion/Matrix |
| `_quat_scale_to_matrix` | function | 667-677 | Quaternion/Matrix |
| `_quat_scale_to_covar_preci` | function | 680-710 | Quaternion/Matrix |
| `compute_inverse_polynomial` | function | 718-782 | Polynomial (numpy) |

---

## 3. Scope

### 3.1 In Scope

Every symbol listed in section 2, plus:
- A local `_assert_shape(name, arr, shape)` helper for MLX arrays
- A manual `_cross(a, b)` helper if `mx.cross` is unavailable
- `@mx.custom_function` with `.vjp` for `SafeNormalize`

### 3.2 Out of Scope

- Camera model classes (deferred to camera PRD)
- CUDA kernel dispatch
- `gsplat._helper.assert_shape` (we write our own for `mx.array`)

---

## 4. Detailed Function Specifications

### 4.1 `_assert_shape` (new helper)

Replaces upstream `gsplat._helper.assert_shape` for `mx.array`.

```python
def _assert_shape(name: str, arr: mx.array, shape: tuple) -> None:
    """Validate that arr.shape is broadcast-compatible with shape and has same rank."""
    if arr.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)} like {shape}, got {arr.shape}")
    # Check broadcast compatibility
    for i, (a, s) in enumerate(zip(arr.shape, shape)):
        if s != 1 and a != 1 and a != s:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
```

### 4.2 `_numerically_stable_norm2(x, y)`

**Signature**:
```python
def _numerically_stable_norm2(x: mx.array, y: mx.array) -> mx.array:
```

**Shapes**:
- Input: `x: [*B]`, `y: [*B]` — arbitrary batch dims, must match
- Output: `norm: [*B]` — `sqrt(x^2 + y^2)` computed stably

**Algorithm**: Avoids overflow/underflow by dividing by max(|x|, |y|).
```
abs_x, abs_y = |x|, |y|
min_val = min(abs_x, abs_y)
max_val = max(abs_x, abs_y)
ratio = min_val / max_val  (where max_val > 0, else 0)
result = max_val * sqrt(1 + ratio^2)  (where max_val > 0, else 0)
```

**Translation notes**:
- `torch.abs` -> `mx.abs` (direct)
- `torch.maximum` -> `mx.maximum` (direct)
- `torch.minimum` -> `mx.minimum` (direct)
- `torch.where` -> `mx.where` (direct)
- `torch.zeros_like` -> `mx.zeros_like` (direct)
- `torch.sqrt` -> `mx.sqrt` (direct)

**MLX implementation**:
```python
def _numerically_stable_norm2(x: mx.array, y: mx.array) -> mx.array:
    _assert_shape("x", x, x.shape)
    _assert_shape("y", y, x.shape)

    abs_x = mx.abs(x)
    abs_y = mx.abs(y)
    min_val = mx.minimum(abs_x, abs_y)
    max_val = mx.maximum(abs_x, abs_y)

    result = mx.zeros_like(max_val)
    nonzero_mask = max_val > 0.0

    min_max_ratio = mx.where(nonzero_mask, min_val / max_val, mx.zeros_like(min_val))
    result = mx.where(
        nonzero_mask,
        max_val * mx.sqrt(1.0 + min_max_ratio * min_max_ratio),
        result,
    )
    return result
```

---

### 4.3 Polynomial Classes

#### 4.3.1 `PolynomialProxy` (ABC)

**Signature**:
```python
class PolynomialProxy(ABC):
    def __init__(self, coeffs: mx.array):
        """coeffs: [..., N] polynomial coefficients."""
        self.coeffs = coeffs

    @abstractmethod
    def eval_horner(self, x: mx.array) -> mx.array:
        """Evaluate polynomial at x using Horner's method.
        coeffs: [..., B, N], x: [..., B, 1] -> result: [..., B, 1]
        """
        ...
```

#### 4.3.2 `FullPolynomialProxy`

**Formula**: `y = c_0 + c_1*x + c_2*x^2 + ... + c_{N-1}*x^{N-1}`

**Horner's method** (reverse iteration):
```
result = c_{N-1}
for i in (N-2, ..., 0):
    result = result * x + c_i
```

**Shapes**:
- `self.coeffs`: `[..., B, N]` where N = number of coefficients
- `x`: `[..., B, 1]` — evaluation point (trailing dim=1 for broadcasting with coeffs slices)
- `result`: `[..., B, 1]`

**MLX implementation**:
```python
class FullPolynomialProxy(PolynomialProxy):
    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        _assert_shape("x", x, B + (1,))

        result = self.coeffs[..., N - 1 : N]
        for i in range(N - 2, -1, -1):
            result = result * x + self.coeffs[..., i : i + 1]

        _assert_shape("result", result, B + (1,))
        return result
```

**Translation notes**: Identical to PyTorch — only array type changes. Slicing `[..., i:i+1]` works the same in MLX.

#### 4.3.3 `OddPolynomialProxy`

**Formula**: `y = c_0*x + c_1*x^3 + c_2*x^5 + ...`
**Factored**: `y = x * FullPoly(coeffs)(x^2)`

```python
class OddPolynomialProxy(PolynomialProxy):
    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        _assert_shape("x", x, B + (1,))
        result = x * FullPolynomialProxy(self.coeffs).eval_horner(x * x)
        _assert_shape("result", result, B + (1,))
        return result
```

#### 4.3.4 `EvenPolynomialProxy`

**Formula**: `y = c_0 + c_1*x^2 + c_2*x^4 + ...`
**Factored**: `y = FullPoly(coeffs)(x^2)`

```python
class EvenPolynomialProxy(PolynomialProxy):
    def eval_horner(self, x: mx.array) -> mx.array:
        B = self.coeffs.shape[:-1]
        _assert_shape("x", x, B + (1,))
        result = FullPolynomialProxy(self.coeffs).eval_horner(x * x)
        _assert_shape("result", result, B + (1,))
        return result
```

---

### 4.4 `_eval_poly_inverse_horner_newton`

**Signature**:
```python
def _eval_poly_inverse_horner_newton(
    poly: PolynomialProxy,
    dpoly: PolynomialProxy,
    inv_poly_approx: PolynomialProxy,
    y: mx.array,       # [..., M]
    n_iterations: int,
) -> Tuple[mx.array, mx.array]:
    """
    Returns:
        x: [..., M] inverted values
        converged: [..., M] boolean convergence mask
    """
```

**Algorithm**:
1. Initial guess: `x_0 = inv_poly_approx.eval_horner(y)`
2. Newton loop (up to `n_iterations`):
   - `fx = poly.eval_horner(x)`
   - `dfdx = dpoly.eval_horner(x)`
   - `dx = (fx - y) / dfdx`
   - `x = x - dx` (only where not yet converged)
   - Mark converged where `|dx| < 1e-6`
3. Early exit when all converged

**Translation notes**:
- `torch.zeros_like(x, dtype=torch.bool)` -> `mx.zeros_like(x).astype(mx.bool_)` or `mx.full(x.shape, False)`
- `torch.all(converged)` -> `mx.all(converged)` — but note MLX lazy eval: call `mx.eval(converged)` before Python-level truth check
- `torch.abs` -> `mx.abs`
- Boolean operations `|` -> use `mx.logical_or` or `|` operator
- `~converged` -> `mx.logical_not(converged)`

**Critical MLX gotcha**: `if mx.all(converged)` requires `mx.eval()` first to materialize the lazy value:
```python
all_converged = mx.all(converged)
mx.eval(all_converged)
if all_converged.item():
    break
```

---

### 4.5 `SafeNormalize` / `_safe_normalize`

This is the **most critical translation** in the module. PyTorch uses `torch.autograd.Function` with explicit `forward`/`backward`. MLX uses `@mx.custom_function` with a `.vjp` decorator.

#### 4.5.1 Forward Pass

```python
def _safe_normalize_impl(v: mx.array, dim: int = -1) -> mx.array:
    """v / ||v|| if ||v|| > 0, else v (unchanged)."""
    norm_sq = mx.sum(v * v, axis=dim, keepdims=True)
    inv_norm = mx.where(norm_sq > 0.0, 1.0 / mx.sqrt(norm_sq), mx.zeros_like(norm_sq))
    return v * inv_norm
```

**Shapes**:
- Input: `v: [..., D]` where D is the dimension being normalized
- `norm_sq: [..., 1]` (keepdims=True)
- `inv_norm: [..., 1]`
- Output: `[..., D]`

#### 4.5.2 VJP (Custom Backward)

The VJP must match the CUDA backward exactly:
```
For non-zero vectors:
  il  = 1/||v||
  il3 = 1/||v||^3 = il / ||v||^2
  dot = sum(grad_out * v, axis=dim, keepdims=True)
  grad_v = il * grad_out - il3 * dot * v

For zero vectors:
  grad_v = grad_out  (pass through)
```

#### 4.5.3 MLX `@mx.custom_function` Pattern

**Important**: `mx.custom_function` signature differs from `torch.autograd.Function`. The decorated function takes only the array arguments. The VJP function receives `(primals, cotangent, output)`.

```python
@mx.custom_function
def _safe_normalize_custom(v: mx.array) -> mx.array:
    """Safe normalize along last dimension."""
    norm_sq = mx.sum(v * v, axis=-1, keepdims=True)
    inv_norm = mx.where(norm_sq > 0.0, 1.0 / mx.sqrt(norm_sq), mx.zeros_like(norm_sq))
    return v * inv_norm

@_safe_normalize_custom.vjp
def _safe_normalize_vjp(primals, cotangent, output):
    (v,) = primals
    grad_out = cotangent

    norm_sq = mx.sum(v * v, axis=-1, keepdims=True)
    inv_norm = mx.where(norm_sq > 0.0, 1.0 / mx.sqrt(norm_sq), mx.zeros_like(norm_sq))
    inv_norm3 = mx.where(norm_sq > 0.0, inv_norm / norm_sq, mx.zeros_like(inv_norm))

    dot_product = mx.sum(grad_out * v, axis=-1, keepdims=True)
    grad_nonzero = inv_norm * grad_out - inv_norm3 * dot_product * v
    grad_v = mx.where(norm_sq > 0.0, grad_nonzero, grad_out)

    return (grad_v,)
```

**Wrapper with `dim` and `keepdim` support**:
```python
def _safe_normalize(v: mx.array, dim: int = -1, keepdim: bool = False) -> mx.array:
    """Safe normalize with arbitrary dim support.

    For dim != -1, transpose so target dim is last, normalize, transpose back.
    """
    if dim == -1 or dim == v.ndim - 1:
        result = _safe_normalize_custom(v)
    else:
        # Move target dim to last position
        perm = list(range(v.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        v_t = mx.transpose(v, perm)
        result_t = _safe_normalize_custom(v_t)
        result = mx.transpose(result_t, perm)

    if not keepdim:
        pass  # normalize doesn't reduce dims, keepdim is about the norm itself
    return result
```

**Note on `keepdim`**: In the upstream, `keepdim` controls whether the output has the normalized dimension squeezed. Since normalization doesn't actually reduce dimensions (unlike a sum), this parameter is effectively a no-op for the output shape. The upstream only squeezes `inv_norm` when `keepdim=False` and `inv_norm.shape != v.shape`, which happens rarely. We handle this edge case explicitly.

---

### 4.6 `_rotmat_to_quat(R)`

**Signature**:
```python
def _rotmat_to_quat(R: mx.array) -> mx.array:
    """
    Args:
        R: [..., 3, 3] rotation matrix
    Returns:
        quat: [..., 4] quaternion (w, x, y, z)
    """
```

**Algorithm** (GLM's `quat_cast`):
1. Compute four candidate squared magnitudes from the trace and diagonal:
   - `fourXSquaredMinus1 = R[0,0] - R[1,1] - R[2,2]`
   - `fourYSquaredMinus1 = R[1,1] - R[0,0] - R[2,2]`
   - `fourZSquaredMinus1 = R[2,2] - R[0,0] - R[1,1]`
   - `fourWSquaredMinus1 = R[0,0] + R[1,1] + R[2,2]`  (trace)
2. Find which of (w, x, y, z) has the largest magnitude: `biggestIndex = argmax([fourW, fourX, fourY, fourZ])`
3. Compute the largest component: `biggestVal = sqrt(fourBiggest + 1) * 0.5`
4. Compute multiplier: `mult = 0.25 / biggestVal`
5. Based on `biggestIndex`, compute remaining 3 components from off-diagonal elements

**Critical MLX translation challenge**: The upstream uses **boolean mask indexing** (`quat[mask, 0] = biggestVal[mask]`), which MLX does NOT support. We must compute ALL four possible quaternion outcomes and select with nested `mx.where`:

```python
# Compute all 4 candidate quaternions for every element
# Case 0: w is largest
q0 = mx.stack([
    biggestVal,
    (R_flat[:, 2, 1] - R_flat[:, 1, 2]) * mult,
    (R_flat[:, 0, 2] - R_flat[:, 2, 0]) * mult,
    (R_flat[:, 1, 0] - R_flat[:, 0, 1]) * mult,
], axis=-1)

# Case 1: x is largest
q1 = mx.stack([
    (R_flat[:, 2, 1] - R_flat[:, 1, 2]) * mult,
    biggestVal,
    (R_flat[:, 1, 0] + R_flat[:, 0, 1]) * mult,
    (R_flat[:, 0, 2] + R_flat[:, 2, 0]) * mult,
], axis=-1)

# Case 2: y is largest
q2 = mx.stack([
    (R_flat[:, 0, 2] - R_flat[:, 2, 0]) * mult,
    (R_flat[:, 1, 0] + R_flat[:, 0, 1]) * mult,
    biggestVal,
    (R_flat[:, 2, 1] + R_flat[:, 1, 2]) * mult,
], axis=-1)

# Case 3: z is largest
q3 = mx.stack([
    (R_flat[:, 1, 0] - R_flat[:, 0, 1]) * mult,
    (R_flat[:, 0, 2] + R_flat[:, 2, 0]) * mult,
    (R_flat[:, 2, 1] + R_flat[:, 1, 2]) * mult,
    biggestVal,
], axis=-1)

# Select based on biggestIndex using nested mx.where
# biggestIndex: [N] — values 0, 1, 2, or 3
idx = biggestIndex[:, None]  # [N, 1] for broadcasting with [N, 4]
quat = mx.where(idx == 0, q0,
        mx.where(idx == 1, q1,
         mx.where(idx == 2, q2, q3)))
```

**Additional translations**:
- `R.reshape(-1, 3, 3)` -> `mx.reshape(R, (-1, 3, 3))` or `R.reshape(-1, 3, 3)`
- `torch.stack([...], dim=1)` -> `mx.stack([...], axis=1)`
- `torch.argmax(x, dim=1)` -> `mx.argmax(x, axis=1)`
- `x.gather(1, idx.unsqueeze(1)).squeeze(1)` -> `mx.take_along_axis(x, idx[:, None], axis=1).squeeze(1)` or fancy indexing `x[mx.arange(x.shape[0]), idx]`
- `torch.zeros((N, 4), dtype=..., device=...)` -> not needed (we compute all cases and select)

---

### 4.7 `_quat_normalize_rotation(q, dim=-1)`

**Signature**:
```python
def _quat_normalize_rotation(q: mx.array, dim: int = -1) -> mx.array:
    """
    Normalize quaternion; zero quaternions become identity (1,0,0,0);
    negative-w quaternions are negated (single-cover).

    Args:
        q: [..., 4] quaternion (w, x, y, z)
        dim: dimension along which quaternion components live (default -1)
    Returns:
        result: [..., 4] normalized quaternion
    """
```

**Algorithm**:
1. `result = _safe_normalize(q, dim=dim)`
2. If all components along `dim` are zero, replace with identity `(1, 0, 0, 0)`
3. If `w < 0`, negate the entire quaternion (double-cover -> single-cover)

**Translation notes**:
- `result.select(dim, 0)` -> index along `dim`, e.g., `result[..., 0]` when `dim=-1`
- `torch.stack([ones, zeros, zeros, zeros], dim=dim)` -> `mx.stack([ones, zeros, zeros, zeros], axis=dim)`
- `torch.all(q == 0, dim=dim, keepdim=True)` -> `mx.all(q == 0, axis=dim, keepdims=True)`
- `(result.select(dim, 0) < 0).unsqueeze(dim)` -> `mx.expand_dims(result[..., 0] < 0, axis=dim)`

```python
def _quat_normalize_rotation(q: mx.array, dim: int = -1) -> mx.array:
    assert q.shape[dim] == 4, q.shape
    dim = dim if dim >= 0 else q.ndim + dim

    result = _safe_normalize(q, dim=dim)

    # Zero quaternions -> identity (1, 0, 0, 0)
    ones = mx.ones_like(result[..., 0] if dim == q.ndim - 1 else ...)  # adapt for dim
    zeros = mx.zeros_like(ones)
    identity = mx.stack([ones, zeros, zeros, zeros], axis=dim)
    is_zero = mx.all(q == 0, axis=dim, keepdims=True)
    result = mx.where(is_zero, identity, result)

    # Single-cover: if w < 0, negate
    # Select w component along dim
    w = ...  # index result along dim at position 0
    w_negative = mx.expand_dims(w < 0, axis=dim)
    result = mx.where(w_negative, -result, result)

    return result
```

**Implementation detail for arbitrary `dim`**: When `dim != -1`, selecting index 0 along an arbitrary dimension requires `mx.take(result, mx.array([0]), axis=dim).squeeze(dim)` or building a slice tuple dynamically. For simplicity, normalize dim to positive and use:
```python
slices = [slice(None)] * result.ndim
slices[dim] = 0
w = result[tuple(slices)]
```

---

### 4.8 `_quat_inverse(q)`

**Signature**:
```python
def _quat_inverse(q: mx.array) -> mx.array:
    """Conjugate of unit quaternion: (w, -x, -y, -z).
    Args:  q: [..., 4]
    Returns: [..., 4]
    """
```

**MLX implementation**:
```python
def _quat_inverse(q: mx.array) -> mx.array:
    return mx.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)
```

Translation is direct: `torch.stack` -> `mx.stack`, `dim` -> `axis`.

---

### 4.9 `_quat_rotate(q, v)`

**Signature**:
```python
def _quat_rotate(q: mx.array, v: mx.array) -> mx.array:
    """Rotate vector v by unit quaternion q.
    Args:
        q: [..., 4] unit quaternion (w, x, y, z)
        v: [..., 3] vector
    Returns:
        rotated: [..., 3]
    """
```

**Algorithm** (equivalent to `q * (0,v) * q_conj`):
```
qvec = (x, y, z)         # imaginary part of q
uv = cross(qvec, v)
uuv = cross(qvec, uv)
result = v + 2*w*uv + 2*uuv
```

**Cross product translation**: `torch.cross(a, b, dim=-1)` -> manual implementation or `mx.cross` if available.

**Manual cross product** (must be included as a helper):
```python
def _cross(a: mx.array, b: mx.array) -> mx.array:
    """Cross product of 3D vectors along last dimension.
    Args: a: [..., 3], b: [..., 3]
    Returns: [..., 3]
    """
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack([
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ], axis=-1)
```

**Translation notes**:
- `torch.unbind(q, dim=-1)` -> `w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]`
- `w[..., None]` -> `mx.expand_dims(w, axis=-1)` or `w[..., None]` (MLX supports this)

---

### 4.10 `_quat_multiply(q1, q2, dim=-1)`

**Signature**:
```python
def _quat_multiply(q1: mx.array, q2: mx.array, dim: int = -1) -> mx.array:
    """Hamilton product of two quaternions.
    Args:
        q1: [..., 4, ...] first quaternion
        q2: [..., 4, ...] second quaternion
        dim: dimension along which quaternion components are stored
    Returns:
        product: [..., 4, ...] Hamilton product q1 * q2
    """
```

**Algorithm**:
```
w1, v1 = q1[..., 0:1], q1[..., 1:4]   (scalar, vector parts)
w2, v2 = q2[..., 0:1], q2[..., 1:4]
w = w1*w2 - sum(v1*v2)
v = w1*v2 + w2*v1 + cross(v1, v2)
result = concat([w, v])
```

**Translation notes**:
- `torch.narrow(q1, dim, 0, 1)` -> `q1[..., 0:1]` (when dim=-1) or dynamic slicing for arbitrary dim
- `torch.sum(v1 * v2, dim=dim, keepdim=True)` -> `mx.sum(v1 * v2, axis=dim, keepdims=True)`
- `torch.cross(v1, v2, dim=dim)` -> `_cross(v1, v2)` (our helper, works only for dim=-1 with [..., 3]). For arbitrary dim, need to move dim to last, cross, move back.
- `torch.cat([w, v], dim=dim)` -> `mx.concatenate([w, v], axis=dim)`

**Important**: When `dim != -1`, the cross product helper needs adaptation. Best approach: transpose target dim to last, compute, transpose back.

---

### 4.11 `_quat_slerp(x, y, t)`

**Signature**:
```python
def _quat_slerp(x: mx.array, y: mx.array, t: mx.array) -> mx.array:
    """Spherical linear interpolation between quaternions.
    Args:
        x: [..., 4] start quaternion (normalized)
        y: [..., 4] end quaternion (normalized)
        t: [...] interpolation parameter in [0, 1]
    Returns:
        result: [..., 4] interpolated quaternion
    """
```

**Algorithm** (GLM slerp):
1. `cosTheta = dot(x, y)` — along last dim
2. If `cosTheta < 0`: negate `y`, take `|cosTheta|` (short path)
3. If `cosTheta > 1 - 1e-6`: use linear interpolation `(1-t)*x + t*y`
4. Else: `theta = acos(cosTheta)`, `result = (sin((1-t)*theta)*x + sin(t*theta)*z) / sin(theta)`

**Translation notes**:
- `(cosTheta <= threshold).any()` -> need `mx.eval()` before Python truth test:
  ```python
  needs_slerp = mx.any(cosTheta <= threshold)
  mx.eval(needs_slerp)
  if needs_slerp.item():
      ...
  ```
- `torch.acos` -> `mx.arccos` (MLX uses `arccos` not `acos`)
- `torch.sin` -> `mx.sin`
- `t[..., None]` -> `t[..., None]` or `mx.expand_dims(t, axis=-1)`

---

### 4.12 `_quat_to_rotmat(quats)`

**Signature**:
```python
def _quat_to_rotmat(quats: mx.array) -> mx.array:
    """Convert quaternion to 3x3 rotation matrix.
    Args:  quats: [..., 4] (w, x, y, z), will be normalized internally
    Returns: R: [..., 3, 3]
    """
```

**Algorithm**: Standard quaternion-to-rotation-matrix formula.
```python
# Normalize
quats = quats / mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

R = mx.stack([
    1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y),
    2*(x*y + w*z),       1 - 2*(x**2 + z**2),  2*(y*z - w*x),
    2*(x*z - w*y),       2*(y*z + w*x),         1 - 2*(x**2 + y**2),
], axis=-1)

return R.reshape(quats.shape[:-1] + (3, 3))
```

**Translation notes**:
- `F.normalize(quats, p=2, dim=-1)` -> manual: `quats / mx.sqrt(mx.sum(quats*quats, axis=-1, keepdims=True))`
- `torch.unbind(quats, dim=-1)` -> `quats[..., 0], quats[..., 1], ...`
- `torch.stack([9 elements], dim=-1)` -> `mx.stack([9 elements], axis=-1)`
- `R.reshape(...)` -> `R.reshape(...)`

---

### 4.13 `_quat_scale_to_matrix(quats, scales)`

**Signature**:
```python
def _quat_scale_to_matrix(quats: mx.array, scales: mx.array) -> mx.array:
    """R * S matrix from quaternion and scales.
    Args:
        quats: [..., 4]
        scales: [..., 3]
    Returns:
        M: [..., 3, 3]  where M = R * diag(scales)
    """
```

**Implementation**: `R = _quat_to_rotmat(quats)`, then `M = R * scales[..., None, :]` (broadcast multiply each column of R by corresponding scale).

---

### 4.14 `_quat_scale_to_covar_preci(quats, scales, compute_covar, compute_preci, triu)`

**Signature**:
```python
def _quat_scale_to_covar_preci(
    quats: mx.array,    # [..., 4]
    scales: mx.array,   # [..., 3]
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    """
    Compute 3D Gaussian covariance and/or precision matrices.

    Covariance: Sigma = R * S * S^T * R^T = M * M^T  where M = R * diag(scales)
    Precision:  P = R * (1/S) * (1/S)^T * R^T

    Args:
        quats: [..., 4] quaternions
        scales: [..., 3] scale factors
        compute_covar: whether to compute covariance
        compute_preci: whether to compute precision
        triu: if True, return upper-triangular (6 elements) instead of full 3x3

    Returns:
        covars: [..., 3, 3] or [..., 6] (triu) or None
        precis: [..., 3, 3] or [..., 6] (triu) or None
    """
```

**Algorithm**:
```python
R = _quat_to_rotmat(quats)  # [..., 3, 3]

if compute_covar:
    M = R * scales[..., None, :]           # [..., 3, 3]
    covars = mx.einsum("...ij,...kj->...ik", M, M)  # M @ M^T
    if triu:
        covars = covars.reshape(batch_dims + (9,))
        # Extract and average: indices [0,1,2,4,5,8] and [0,3,6,4,7,8]
        covars = (covars[..., [0,1,2,4,5,8]] + covars[..., [0,3,6,4,7,8]]) / 2.0

if compute_preci:
    P = R * (1.0 / scales[..., None, :])   # [..., 3, 3]
    precis = mx.einsum("...ij,...kj->...ik", P, P)  # P @ P^T
    if triu:
        # same indexing as above
```

**Translation notes**:
- `torch.einsum` -> `mx.einsum` (MLX supports einsum)
- Fancy indexing `x[..., [0,1,2,4,5,8]]` -> this works in MLX with `mx.take(x, mx.array([0,1,2,4,5,8]), axis=-1)` or direct indexing. **Verify MLX supports integer list indexing on last axis.**

**Fallback for fancy indexing**:
```python
# If MLX doesn't support list indexing:
idx_a = mx.array([0, 1, 2, 4, 5, 8])
idx_b = mx.array([0, 3, 6, 4, 7, 8])
flat = covars.reshape(batch_dims + (9,))
part_a = mx.take(flat, idx_a, axis=-1)
part_b = mx.take(flat, idx_b, axis=-1)
covars = (part_a + part_b) / 2.0
```

---

### 4.15 `_quat_scale_to_preci_half(quats, scales)`

**Signature**:
```python
def _quat_scale_to_preci_half(quats: mx.array, scales: mx.array) -> mx.array:
    """Compute M = R * diag(1/scales).
    Args:  quats: [..., 4], scales: [..., 3]
    Returns: M: [..., 3, 3]
    """
```

**Implementation**:
```python
R = _quat_to_rotmat(quats)
return R * (1.0 / scales[..., None, :])
```

---

### 4.16 `compute_inverse_polynomial` (numpy-based)

**Signature**: Unchanged from upstream. This function uses numpy only (not torch or MLX) and runs offline. **Port as-is**, just remove the torch import dependency.

```python
def compute_inverse_polynomial(
    forward_poly_coeffs: list,
    input_range: Tuple[float, float],
    num_samples: int = 1000,
) -> list:
    """Fit inverse polynomial via least squares.
    Returns list of 6 float32 coefficients [k0, k1, k2, k3, k4, k5].
    """
```

No translation needed — this function only uses `numpy` and `np.linalg.solve`.

---

## 5. torch-to-mlx Translation Reference (Complete)

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.Tensor` | `mx.array` | |
| `torch.abs(x)` | `mx.abs(x)` | Direct |
| `torch.sqrt(x)` | `mx.sqrt(x)` | Direct |
| `torch.rsqrt(x)` | `1.0 / mx.sqrt(x)` | MLX has `mx.rsqrt` in recent versions; verify availability |
| `torch.where(c, a, b)` | `mx.where(c, a, b)` | Direct |
| `torch.zeros_like(x)` | `mx.zeros_like(x)` | Direct |
| `torch.ones_like(x)` | `mx.ones_like(x)` | Direct |
| `torch.stack([...], dim=d)` | `mx.stack([...], axis=d)` | `dim` -> `axis` |
| `torch.cat([...], dim=d)` | `mx.concatenate([...], axis=d)` | Name change |
| `torch.unbind(x, dim=-1)` | `x[..., 0], x[..., 1], ...` | Manual indexing |
| `torch.cross(a, b, dim=-1)` | `_cross(a, b)` (manual helper) | See section 4.9 |
| `torch.narrow(t, dim, s, l)` | `t[..., s:s+l]` | Slice syntax |
| `F.normalize(x, p=2, dim=-1)` | `x / mx.sqrt(mx.sum(x*x, axis=-1, keepdims=True))` | Manual |
| `torch.einsum(eq, a, b)` | `mx.einsum(eq, a, b)` | Direct (verify `...` support) |
| `torch.argmax(x, dim=d)` | `mx.argmax(x, axis=d)` | `dim` -> `axis` |
| `x.gather(1, idx)` | `mx.take_along_axis(x, idx, axis=1)` | Different API |
| `torch.maximum(a, b)` | `mx.maximum(a, b)` | Direct |
| `torch.minimum(a, b)` | `mx.minimum(a, b)` | Direct |
| `torch.acos(x)` | `mx.arccos(x)` | Name differs |
| `torch.sin(x)` | `mx.sin(x)` | Direct |
| `torch.all(x)` | `mx.all(x)` | Must `mx.eval()` before Python bool check |
| `torch.any(x)` | `mx.any(x)` | Must `mx.eval()` before Python bool check |
| `x.unsqueeze(d)` | `mx.expand_dims(x, axis=d)` | or `x[..., None]` for last dim |
| `x.squeeze(d)` | `mx.squeeze(x, axis=d)` | Direct |
| `x.select(dim, idx)` | Dynamic slicing (see 4.7) | No direct equivalent |
| `x.reshape(...)` | `x.reshape(...)` or `mx.reshape(x, ...)` | Direct |
| `x[mask] = val` (bool indexing) | `mx.where(mask, val, x)` | **Critical difference** |
| `torch.autograd.Function` | `@mx.custom_function` + `.vjp` | See section 4.5 |
| `ctx.save_for_backward(...)` | Not needed (recompute in VJP) | MLX recomputes |
| `torch.zeros((N,4), dtype=d)` | `mx.zeros((N,4))` | dtype via `mx.float32` etc. |

---

## 6. MLX-Specific Gotchas & Mitigations

### 6.1 No Boolean Indexing

**Problem**: `quat[mask, 0] = biggestVal[mask]` is used extensively in `_rotmat_to_quat`.
**Solution**: Compute all branches, select with `mx.where`. See section 4.6 for full pattern.

### 6.2 Lazy Evaluation

**Problem**: `if mx.all(converged)` will fail because the value hasn't been computed yet.
**Solution**: Call `mx.eval()` before any Python-level truth check:
```python
flag = mx.all(converged)
mx.eval(flag)
if flag.item():
    break
```

### 6.3 No `torch.unbind`

**Problem**: `w, x, y, z = torch.unbind(quats, dim=-1)` is used in many functions.
**Solution**: Manual indexing: `w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]`

### 6.4 No `torch.cross` (or uncertain availability)

**Problem**: `torch.cross(a, b, dim=-1)` used in `_quat_rotate` and `_quat_multiply`.
**Solution**: Implement `_cross(a, b)` helper (section 4.9). Check `hasattr(mx, 'cross')` at import time and use it if available, falling back to manual.

### 6.5 `@mx.custom_function` Signature

**Problem**: The decorator and VJP have different calling conventions than `torch.autograd.Function`.
**Key differences**:
- Decorated function takes only array args (no `ctx`)
- VJP receives `(primals_tuple, cotangent, output)` — primals is a tuple of inputs
- VJP returns a tuple of gradients, one per primal
- Non-array arguments (like `dim: int`) cannot be passed through; must be captured via closure or hardcoded

### 6.6 Einsum `...` Support

**Verify**: `mx.einsum("...ij,...kj->...ik", M, M)` — confirm MLX supports `...` (ellipsis) in einsum. If not, use explicit dimension letters or fall back to `M @ M.swapaxes(-1, -2)`:
```python
# Fallback:
covars = M @ mx.transpose(M, list(range(M.ndim - 2)) + [M.ndim - 1, M.ndim - 2])
# Or simpler:
covars = M @ mx.swapaxes(M, -1, -2)
```

### 6.7 `mx.take_along_axis` for Gather

**Verify**: `mx.take_along_axis(x, idx[:, None], axis=1)` as replacement for `x.gather(1, idx.unsqueeze(1))`. If unavailable, use:
```python
# Manual gather for 2D case:
result = x[mx.arange(x.shape[0]), idx]
```

---

## 7. File Structure

### 7.1 Target File: `src/gsplat_mlx/core/math_utils.py`

```python
"""Mathematical utility functions for gsplat-mlx.

Port of gsplat/cuda/_math.py from PyTorch to Apple MLX.
"""

import mlx.core as mx
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# =============================================================================
# Shape Assertion Helper
# =============================================================================

def _assert_shape(name, arr, shape): ...

# =============================================================================
# Cross Product Helper
# =============================================================================

def _cross(a, b): ...

# =============================================================================
# Numerically Stable Operations
# =============================================================================

def _numerically_stable_norm2(x, y): ...

# =============================================================================
# Polynomial Evaluation
# =============================================================================

class PolynomialProxy(ABC): ...
class FullPolynomialProxy(PolynomialProxy): ...
class OddPolynomialProxy(PolynomialProxy): ...
class EvenPolynomialProxy(PolynomialProxy): ...

def _eval_poly_inverse_horner_newton(poly, dpoly, inv_poly_approx, y, n_iterations): ...

# =============================================================================
# Safe Normalize with Custom VJP
# =============================================================================

@mx.custom_function
def _safe_normalize_custom(v): ...

@_safe_normalize_custom.vjp
def _safe_normalize_vjp(primals, cotangent, output): ...

def _safe_normalize(v, dim=-1, keepdim=False): ...

# =============================================================================
# Quaternion Operations
# =============================================================================

def _rotmat_to_quat(R): ...
def _quat_normalize_rotation(q, dim=-1): ...
def _quat_inverse(q): ...
def _quat_rotate(q, v): ...
def _quat_multiply(q1, q2, dim=-1): ...
def _quat_slerp(x, y, t): ...

# =============================================================================
# Quaternion-Scale-to-Matrix Operations
# =============================================================================

def _quat_to_rotmat(quats): ...
def _quat_scale_to_matrix(quats, scales): ...
def _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=True, triu=False): ...
def _quat_scale_to_preci_half(quats, scales): ...

# =============================================================================
# Polynomial Utilities (numpy-based, offline)
# =============================================================================

def compute_inverse_polynomial(forward_poly_coeffs, input_range, num_samples=1000): ...
```

### 7.2 Public Exports

All functions should be importable from `gsplat_mlx.core.math_utils`. Additionally, add to `src/gsplat_mlx/core/__init__.py`:

```python
from .math_utils import (
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
```

---

## 8. Test Plan

### 8.1 Test File: `tests/test_math_utils.py`

All tests use `pytest`. Cross-framework tests that compare MLX vs PyTorch are marked `@pytest.mark.requires_torch` and skip gracefully if torch is not installed.

### 8.2 Test Cases (Exhaustive)

#### 8.2.1 `_numerically_stable_norm2`

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_stable_norm2_basic` | `x=[3.0], y=[4.0]` | `[5.0]` | `atol=1e-6` |
| `test_stable_norm2_batch` | `x=[3,0,1], y=[4,0,0]` | `[5,0,1]` | `atol=1e-6` |
| `test_stable_norm2_zeros` | `x=[0.0], y=[0.0]` | `[0.0]` | exact |
| `test_stable_norm2_large` | `x=[1e30], y=[1e30]` | `[sqrt(2)*1e30]` | `rtol=1e-5` |
| `test_stable_norm2_small` | `x=[1e-30], y=[1e-30]` | `[sqrt(2)*1e-30]` | `rtol=1e-5` |
| `test_stable_norm2_asymmetric` | `x=[1e30], y=[1.0]` | `[1e30]` | `rtol=1e-5` |
| `test_stable_norm2_negative` | `x=[-3.0], y=[-4.0]` | `[5.0]` | `atol=1e-6` |
| `test_stable_norm2_vs_torch` | 1000 random | match torch | `atol=1e-5` |

#### 8.2.2 Polynomial Evaluation

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_full_poly_constant` | coeffs=[5.0], x=anything -> 5.0 | `atol=1e-6` |
| `test_full_poly_linear` | coeffs=[2.0, 3.0], x=4.0 -> 14.0 | `atol=1e-6` |
| `test_full_poly_quadratic` | coeffs=[1.0, 0.0, 1.0], x=3.0 -> 10.0 | `atol=1e-6` |
| `test_full_poly_degree5` | coeffs=[1,2,3,4,5,6], x=2.0 | `atol=1e-5` |
| `test_odd_poly` | coeffs=[1.0, 1.0], x=2.0 -> 2+8=10.0 (x + x^3) | `atol=1e-6` |
| `test_even_poly` | coeffs=[1.0, 1.0], x=3.0 -> 1+9=10.0 (1 + x^2) | `atol=1e-6` |
| `test_poly_batch` | coeffs=[B, N], x=[B, 1], B=10 | `atol=1e-5` |
| `test_poly_vs_torch` | Random coefficients, compare with torch | `atol=1e-5` |

#### 8.2.3 Polynomial Inverse (Newton)

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_poly_inverse_linear` | f(x) = 2x, inv should give x/2 | `atol=1e-5` |
| `test_poly_inverse_quadratic` | f(x) = x^2 on [0.1, 2.0], verify f(f_inv(y)) ~ y | `atol=1e-3` |
| `test_poly_inverse_convergence` | Check convergence mask is all True | exact |
| `test_poly_inverse_vs_torch` | Same inputs, compare outputs | `atol=1e-3` |

#### 8.2.4 `compute_inverse_polynomial` (numpy)

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_compute_inverse_poly_linear` | f(x) = x, inverse should be ~ identity | `atol=1e-3` |
| `test_compute_inverse_poly_cubic` | Known cubic, roundtrip g(f(x)) ~ x | `atol=1e-3` |
| `test_compute_inverse_poly_invalid` | NaN coefficients -> ValueError | N/A |

#### 8.2.5 `_safe_normalize` (Forward)

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_safe_normalize_unit` | Already unit vector -> unchanged | `atol=1e-6` |
| `test_safe_normalize_scaled` | [3,4,0] -> [0.6, 0.8, 0] | `atol=1e-5` |
| `test_safe_normalize_zero` | [0,0,0] -> [0,0,0] | exact |
| `test_safe_normalize_batch` | [B, 3] batch of vectors | `atol=1e-5` |
| `test_safe_normalize_large` | [1e20, 0, 0] -> [1, 0, 0] | `atol=1e-5` |
| `test_safe_normalize_small` | [1e-20, 0, 0] -> [1, 0, 0] | `atol=1e-5` |
| `test_safe_normalize_2d` | normalize along dim=0 and dim=1 | `atol=1e-5` |

#### 8.2.6 `_safe_normalize` (VJP / Backward)

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_safe_normalize_vjp_nonzero` | VJP for [3,4,0], compare with finite differences | `atol=1e-4` |
| `test_safe_normalize_vjp_zero` | VJP for [0,0,0] -> grad passes through | `atol=1e-6` |
| `test_safe_normalize_vjp_unit` | VJP for already-unit vector | `atol=1e-4` |
| `test_safe_normalize_vjp_batch` | Batch VJP [B, 3] | `atol=1e-4` |
| `test_safe_normalize_vjp_vs_torch` | Compare with torch SafeNormalize.backward | `atol=1e-4` |

**VJP test pattern**:
```python
def test_safe_normalize_vjp_nonzero():
    v = mx.array([3.0, 4.0, 0.0])

    def f(v):
        return mx.sum(_safe_normalize(v))

    # MLX gradient
    grad_fn = mx.grad(f)
    grad = grad_fn(v)
    mx.eval(grad)

    # Finite difference reference
    eps = 1e-4
    grad_fd = mx.zeros_like(v)
    for i in range(v.shape[0]):
        v_plus = v.at[i].add(eps)  # or manual
        v_minus = v.at[i].add(-eps)
        grad_fd_i = (f(v_plus) - f(v_minus)) / (2 * eps)
        # ... build finite diff grad

    np.testing.assert_allclose(np.array(grad), np.array(grad_fd), atol=1e-4)
```

#### 8.2.7 `_quat_to_rotmat`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_to_rotmat_identity` | [1,0,0,0] -> I_3 | `atol=1e-6` |
| `test_quat_to_rotmat_90x` | 90deg about X: [cos45, sin45, 0, 0] | `atol=1e-5` |
| `test_quat_to_rotmat_90y` | 90deg about Y: [cos45, 0, sin45, 0] | `atol=1e-5` |
| `test_quat_to_rotmat_90z` | 90deg about Z: [cos45, 0, 0, sin45] | `atol=1e-5` |
| `test_quat_to_rotmat_orthogonal` | R @ R^T == I for 100 random quats | `atol=1e-5` |
| `test_quat_to_rotmat_det` | det(R) == 1 for 100 random quats | `atol=1e-5` |
| `test_quat_to_rotmat_batch` | [B, 4] -> [B, 3, 3], B=100 | `atol=1e-5` |
| `test_quat_to_rotmat_vs_torch` | Compare with torch _quat_to_rotmat | `atol=1e-5` |

#### 8.2.8 `_rotmat_to_quat`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_rotmat_to_quat_identity` | I_3 -> [1,0,0,0] (or [-1,0,0,0]) | `atol=1e-5` |
| `test_rotmat_to_quat_roundtrip` | q -> R -> q', verify q' == q (up to sign) | `atol=1e-5` |
| `test_rotmat_to_quat_batch` | [B, 3, 3] -> [B, 4], B=100 | `atol=1e-5` |
| `test_rotmat_to_quat_all_branches` | Craft rotations that trigger each of the 4 cases (w, x, y, z largest) | `atol=1e-5` |
| `test_rotmat_to_quat_vs_torch` | Compare with torch | `atol=1e-5` |

**Roundtrip test**: Due to quaternion double-cover, compare `|dot(q, q')|` ~ 1.0 rather than element-wise equality.

#### 8.2.9 `_quat_normalize_rotation`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_norm_already_unit` | [1,0,0,0] -> [1,0,0,0] | `atol=1e-6` |
| `test_quat_norm_unnormalized` | [2,0,0,0] -> [1,0,0,0] | `atol=1e-5` |
| `test_quat_norm_zero` | [0,0,0,0] -> [1,0,0,0] (identity) | exact |
| `test_quat_norm_negative_w` | [-1,0,0,0] -> [1,0,0,0] (negated) | `atol=1e-6` |
| `test_quat_norm_negative_w_general` | [-0.5, 0.5, 0.5, 0.5] -> [0.5, -0.5, -0.5, -0.5] | `atol=1e-5` |
| `test_quat_norm_batch` | Mixed zero and non-zero in batch | `atol=1e-5` |

#### 8.2.10 `_quat_inverse`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_inverse_identity` | inv([1,0,0,0]) = [1,0,0,0] | exact |
| `test_quat_inverse_general` | inv([w,x,y,z]) = [w,-x,-y,-z] | exact |
| `test_quat_inverse_product` | q * inv(q) == [1,0,0,0] | `atol=1e-5` |
| `test_quat_inverse_batch` | [B, 4] | `atol=1e-6` |

#### 8.2.11 `_quat_rotate`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_rotate_identity` | rotate by [1,0,0,0] -> no change | `atol=1e-6` |
| `test_quat_rotate_90x` | Rotate [0,1,0] by 90deg about X -> [0,0,1] | `atol=1e-5` |
| `test_quat_rotate_90y` | Rotate [1,0,0] by 90deg about Y -> [0,0,-1] | `atol=1e-5` |
| `test_quat_rotate_90z` | Rotate [1,0,0] by 90deg about Z -> [0,1,0] | `atol=1e-5` |
| `test_quat_rotate_vs_rotmat` | `_quat_rotate(q, v)` == `_quat_to_rotmat(q) @ v` | `atol=1e-5` |
| `test_quat_rotate_batch` | [B, 4] quats, [B, 3] vecs | `atol=1e-5` |
| `test_quat_rotate_norm_preserved` | ||rotate(q,v)|| == ||v|| | `atol=1e-5` |

#### 8.2.12 `_quat_multiply`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_multiply_identity_left` | [1,0,0,0] * q == q | `atol=1e-6` |
| `test_quat_multiply_identity_right` | q * [1,0,0,0] == q | `atol=1e-6` |
| `test_quat_multiply_inverse` | q * inv(q) == [1,0,0,0] | `atol=1e-5` |
| `test_quat_multiply_associative` | (a*b)*c == a*(b*c) for random a,b,c | `atol=1e-4` |
| `test_quat_multiply_known` | i*j=k, j*k=i, k*i=j | `atol=1e-6` |
| `test_quat_multiply_batch` | [B, 4] * [B, 4] | `atol=1e-5` |
| `test_quat_multiply_non_commutative` | a*b != b*a in general | N/A |
| `test_quat_multiply_vs_torch` | Compare with torch | `atol=1e-5` |

**Known products (Hamilton)**:
- `i = [0,1,0,0]`, `j = [0,0,1,0]`, `k = [0,0,0,1]`
- `i*j = k` -> `[0,0,0,1]`
- `j*k = i` -> `[0,1,0,0]`
- `k*i = j` -> `[0,0,1,0]`

#### 8.2.13 `_quat_slerp`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_slerp_t0` | slerp(a, b, 0) == a | `atol=1e-6` |
| `test_quat_slerp_t1` | slerp(a, b, 1) == b (or -b if short path) | `atol=1e-6` |
| `test_quat_slerp_t05` | Midpoint is unit quaternion | `atol=1e-5` |
| `test_quat_slerp_close` | Nearly identical quaternions (linear interp path) | `atol=1e-5` |
| `test_quat_slerp_opposite` | Antipodal quaternions (cosTheta < 0) | `atol=1e-5` |
| `test_quat_slerp_same` | slerp(q, q, t) == q for all t | `atol=1e-6` |
| `test_quat_slerp_batch` | [B, 4] slerp with [B] t values | `atol=1e-5` |
| `test_quat_slerp_vs_torch` | Compare with torch | `atol=1e-5` |

#### 8.2.14 `_quat_scale_to_matrix`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_quat_scale_to_matrix_identity` | Identity quat, [1,1,1] scales -> I_3 | `atol=1e-5` |
| `test_quat_scale_to_matrix_scale_only` | Identity quat, [2,3,4] -> diag(2,3,4) | `atol=1e-5` |
| `test_quat_scale_to_matrix_rotation_only` | Random quat, [1,1,1] -> rotation matrix | `atol=1e-5` |
| `test_quat_scale_to_matrix_batch` | [B, 4], [B, 3] | `atol=1e-5` |
| `test_quat_scale_to_matrix_vs_torch` | Compare with torch | `atol=1e-5` |

#### 8.2.15 `_quat_scale_to_covar_preci`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_covar_preci_identity` | Identity quat, [1,1,1] -> covar=I, preci=I | `atol=1e-5` |
| `test_covar_preci_scaled` | Identity quat, [2,3,4] -> covar=diag(4,9,16), preci=diag(1/4,1/9,1/16) | `atol=1e-5` |
| `test_covar_only` | compute_covar=True, compute_preci=False -> (covar, None) | `atol=1e-5` |
| `test_preci_only` | compute_covar=False, compute_preci=True -> (None, preci) | `atol=1e-5` |
| `test_covar_psd` | Covariance is positive semi-definite (eigenvalues >= 0) | `atol=1e-5` |
| `test_covar_preci_inverse` | covar @ preci ~ I for non-degenerate scales | `atol=1e-4` |
| `test_covar_triu` | triu=True gives 6 elements matching upper triangle of full | `atol=1e-5` |
| `test_covar_preci_batch` | [B, 4], [B, 3] | `atol=1e-5` |
| `test_covar_preci_vs_torch` | Compare with torch | `atol=1e-5` |

**Triu format verification**:
```python
# Full matrix:
#   [[a, b, c],
#    [d, e, f],
#    [g, h, i]]
# Flat: [a, b, c, d, e, f, g, h, i] (indices 0-8)
# triu indices: [0,1,2,4,5,8] = [a, b, c, e, f, i]
# Symmetrized: average of [0,1,2,4,5,8] and [0,3,6,4,7,8]
#   = average of [a,b,c,e,f,i] and [a,d,g,e,h,i]
#   = [a, (b+d)/2, (c+g)/2, e, (f+h)/2, i]
# This is the upper triangle of the symmetrized matrix.
```

#### 8.2.16 `_quat_scale_to_preci_half`

| Test | Details | Tolerance |
|------|---------|-----------|
| `test_preci_half_identity` | Identity quat, [1,1,1] -> I_3 | `atol=1e-5` |
| `test_preci_half_scaled` | Identity quat, [2,3,4] -> diag(0.5, 1/3, 0.25) | `atol=1e-5` |
| `test_preci_half_vs_torch` | Compare with torch | `atol=1e-5` |

### 8.3 Test Fixtures (in `conftest.py` or test file)

```python
@pytest.fixture
def random_quaternions():
    """Generate B=100 random normalized quaternions."""
    key = mx.random.key(42)
    q = mx.random.normal(key, (100, 4))
    q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True))
    mx.eval(q)
    return q

@pytest.fixture
def random_scales():
    """Generate B=100 positive random scales."""
    key = mx.random.key(43)
    s = mx.abs(mx.random.normal(key, (100, 3))) + 0.1  # ensure positive
    mx.eval(s)
    return s

@pytest.fixture
def identity_quat():
    return mx.array([1.0, 0.0, 0.0, 0.0])
```

### 8.4 Cross-Framework Comparison Helper

```python
requires_torch = pytest.mark.skipif(
    not _has_torch, reason="PyTorch not installed"
)

def compare_mlx_torch(mlx_fn, torch_fn, *mlx_inputs, atol=1e-5):
    """Run both implementations on same data, assert close."""
    mlx_result = mlx_fn(*mlx_inputs)
    mx.eval(mlx_result)

    torch_inputs = [torch.tensor(np.array(x)) for x in mlx_inputs]
    torch_result = torch_fn(*torch_inputs)

    np.testing.assert_allclose(
        np.array(mlx_result),
        torch_result.numpy(),
        atol=atol,
    )
```

---

## 9. Tolerances Summary

| Category | `atol` | `rtol` | Rationale |
|----------|--------|--------|-----------|
| Pure math (norm, poly) | `1e-5` | — | float32 precision |
| Quaternion ops | `1e-5` | — | float32 precision |
| SafeNormalize VJP | `1e-4` | — | Finite diff introduces error |
| Polynomial inverse (Newton) | `1e-3` | — | Iterative convergence |
| `compute_inverse_polynomial` | `1e-3` | — | Least-squares fitting |
| Roundtrip tests (q->R->q) | `1e-5` | — | Composed operations |
| Associativity tests | `1e-4` | — | Accumulated floating point |

---

## 10. Dependencies

| Dependency | PRD | Status |
|------------|-----|--------|
| Dev environment, package structure | PRD-01 | Must be complete |
| `mlx` >= 0.5.0 | PRD-01 | Installed |
| `numpy` | PRD-01 | Installed |
| `pytest` | PRD-01 | Installed |
| `torch` (optional, for cross-framework tests) | PRD-01 | Optional |

---

## 11. Blocks (Downstream Dependents)

| Downstream PRD | Functions Used |
|----------------|---------------|
| PRD-03 (Covariance) | `_quat_scale_to_covar_preci`, `_quat_to_rotmat`, `_quat_scale_to_matrix` |
| PRD-04 (Spherical Harmonics) | `_safe_normalize` |
| PRD-05 (Projection) | `_quat_to_rotmat`, `_quat_rotate`, `_safe_normalize`, `_numerically_stable_norm2`, polynomial classes |
| PRD-12 (2DGS) | Quaternion ops |

---

## 12. Acceptance Criteria

- [ ] `src/gsplat_mlx/core/math_utils.py` contains all 19 symbols from section 2
- [ ] `_assert_shape` validates rank and broadcast compatibility
- [ ] `_cross` helper correctly computes 3D cross products
- [ ] `_numerically_stable_norm2` handles zero, large (1e30), and small (1e-30) inputs
- [ ] `FullPolynomialProxy`, `OddPolynomialProxy`, `EvenPolynomialProxy` evaluate correctly via Horner's method
- [ ] `_eval_poly_inverse_horner_newton` converges within specified iterations
- [ ] `_safe_normalize` uses `@mx.custom_function` with correct VJP matching CUDA backward
- [ ] `_safe_normalize` VJP passes gradient through for zero vectors
- [ ] `_rotmat_to_quat` uses vectorized `mx.where` (no boolean mask indexing)
- [ ] `_rotmat_to_quat` correctly handles all 4 branches (w, x, y, z largest)
- [ ] `_quat_to_rotmat` produces orthogonal matrices with det=1
- [ ] `_quat_normalize_rotation` returns identity for zero quaternions and negates for negative w
- [ ] `_quat_rotate(q, v)` matches `_quat_to_rotmat(q) @ v`
- [ ] `_quat_multiply` satisfies Hamilton product rules (i*j=k, etc.)
- [ ] `_quat_slerp` returns exact endpoints at t=0 and t=1
- [ ] `_quat_scale_to_covar_preci` produces PSD covariance matrices
- [ ] `_quat_scale_to_covar_preci` triu mode matches upper triangle of full matrix
- [ ] `compute_inverse_polynomial` roundtrips within tolerance
- [ ] All functions handle batch dimensions correctly (scalar, 1D, 2D, 3D inputs where applicable)
- [ ] No Python loops over individual elements (all operations vectorized)
- [ ] `mx.eval()` called before any Python-level truth checks on lazy values
- [ ] All tests pass: `pytest tests/test_math_utils.py -v`
- [ ] Cross-framework tests pass: `pytest tests/test_math_utils.py -v -m requires_torch`

---

## 13. Implementation Order

Implement in this order to enable incremental testing:

1. **`_assert_shape`** and **`_cross`** helpers
2. **`_numerically_stable_norm2`** — standalone, no deps
3. **Polynomial classes** (`FullPolynomialProxy`, `OddPolynomialProxy`, `EvenPolynomialProxy`)
4. **`_eval_poly_inverse_horner_newton`** — depends on polynomial classes
5. **`_safe_normalize`** with `@mx.custom_function` + VJP — critical, test thoroughly
6. **`_quat_to_rotmat`** — foundational for all quat-to-matrix ops
7. **`_quat_scale_to_matrix`** — simple wrapper around `_quat_to_rotmat`
8. **`_quat_scale_to_covar_preci`** — depends on `_quat_to_rotmat`
9. **`_quat_scale_to_preci_half`** — depends on `_quat_to_rotmat`
10. **`_rotmat_to_quat`** — complex, needs vectorized `mx.where`
11. **`_quat_normalize_rotation`** — depends on `_safe_normalize`
12. **`_quat_inverse`** — simple
13. **`_quat_rotate`** — depends on `_safe_normalize`, `_cross`
14. **`_quat_multiply`** — depends on `_cross`
15. **`_quat_slerp`** — depends on basic quaternion ops
16. **`compute_inverse_polynomial`** — numpy only, copy with minimal changes

---

## 14. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| `@mx.custom_function` API differs from docs | High | Test VJP early; read MLX source for exact signature |
| `mx.einsum` doesn't support `...` | Medium | Fallback to `M @ mx.swapaxes(M, -1, -2)` |
| `mx.take_along_axis` unavailable | Medium | Use manual gather with `arange` indexing |
| `mx.rsqrt` unavailable | Low | Use `1.0 / mx.sqrt(x)` |
| float32 precision differences between MLX/PyTorch | Low | Already accounted for in tolerances |
| `_rotmat_to_quat` performance with 4x branching | Low | Vectorized `mx.where` is efficient on GPU |
