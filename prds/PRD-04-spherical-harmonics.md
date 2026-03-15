# PRD-04: Spherical Harmonics

## Overview

Port the Spherical Harmonics (SH) evaluation from PyTorch to MLX. In 3D Gaussian Splatting, each Gaussian stores color as SH coefficients rather than raw RGB values. Given a view direction from the camera to the Gaussian center, the SH coefficients are evaluated to produce a view-dependent RGB color. This enables realistic view-dependent appearance effects such as specular highlights and reflections.

SH evaluation is performed for every visible Gaussian in every rendered frame, making it one of the most frequently called functions in the pipeline. The implementation supports SH degrees 0 through 4, corresponding to 1, 4, 9, 16, and 25 basis functions respectively. Degree 3 (16 coefficients) is the most common setting in practical 3DGS training.

## Source Reference

- **Forward (basis evaluation)**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:673-754` (`_eval_sh_bases_fast`)
- **Forward (SH dot product)**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:757-775` (`_spherical_harmonics`)
- **Public API**: `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:330-360` (`spherical_harmonics`)
- **autograd.Function**: `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:2296-2333` (`_SphericalHarmonics`)
- **CUDA kernels**: `csrc/SphericalHarmonicsCUDA.cu` (reference only, we port the Python impl)
- **Lines**: ~82 lines for `_eval_sh_bases_fast`, ~19 lines for `_spherical_harmonics`
- **Paper**: "Efficient Spherical Harmonic Evaluation", Peter-Pike Sloan, JCGT 2013 (https://jcgt.org/published/0002/02/06/)

## Scope

### In Scope
- `_eval_sh_bases_fast(basis_dim, dirs)` -- evaluate SH basis functions at given unit directions
- `spherical_harmonics(degrees_to_use, dirs, coeffs)` -- evaluate SH to get RGB colors, with `@mx.custom_function` and `.vjp`
- Custom VJP for backward pass (gradients w.r.t. `dirs` and `coeffs`)
- SH degrees 0-4 (basis_dim 1, 4, 9, 16, 25)
- Batch support with arbitrary leading dimensions
- Direction normalization (input dirs need not be unit vectors)

### Out of Scope
- SH degree > 4 (not used in standard 3DGS)
- Per-Gaussian degree masking (`masks` parameter in upstream -- deferred to optimization phase)
- Precomputed SH transfer matrices
- CUDA/Metal kernel implementation (pure MLX array ops for MVP)
- SH rotation (rotating coefficients under scene rotation)

## Background: Spherical Harmonics in 3DGS

### What Are Spherical Harmonics?

Spherical harmonics are an orthonormal basis for functions on the unit sphere S^2, analogous to Fourier series for periodic functions. Any function f(theta, phi) on the sphere can be decomposed:

```
f(theta, phi) = sum_{l=0}^{inf} sum_{m=-l}^{l} c_l^m * Y_l^m(theta, phi)
```

where Y_l^m are the SH basis functions and c_l^m are the coefficients.

### Why SH for Color?

In 3DGS, each Gaussian's color varies with viewing direction. Instead of storing a lookup table or neural network, we store SH coefficients per color channel (R, G, B). Given a view direction d, the color is:

```
color_c = sum_{k=0}^{K-1} coeffs[k, c] * Y_k(d)     for c in {R, G, B}
```

Higher SH degrees capture more angular variation:
- **Degree 0** (1 coeff): constant color (diffuse only)
- **Degree 1** (4 coeffs): linear directional variation
- **Degree 2** (9 coeffs): quadratic variation (soft highlights)
- **Degree 3** (16 coeffs): cubic variation (sharp highlights) -- most common
- **Degree 4** (25 coeffs): quartic variation (very sharp effects)

### Basis Function Ordering

The implementation uses the real SH basis in the order: for each degree l, the basis functions are ordered as Y_l^{-l}, Y_l^{-l+1}, ..., Y_l^{l-1}, Y_l^{l}. The indexing maps to flat array indices as follows:

| Index | Degree l | Order m | Symbol |
|-------|----------|---------|--------|
| 0     | 0        | 0       | Y_0^0  |
| 1     | 1        | -1      | Y_1^{-1} |
| 2     | 1        | 0       | Y_1^0  |
| 3     | 1        | 1       | Y_1^1  |
| 4     | 2        | -2      | Y_2^{-2} |
| 5     | 2        | -1      | Y_2^{-1} |
| 6     | 2        | 0       | Y_2^0  |
| 7     | 2        | 1       | Y_2^1  |
| 8     | 2        | 2       | Y_2^2  |
| 9     | 3        | -3      | Y_3^{-3} |
| 10    | 3        | -2      | Y_3^{-2} |
| 11    | 3        | -1      | Y_3^{-1} |
| 12    | 3        | 0       | Y_3^0  |
| 13    | 3        | 1       | Y_3^1  |
| 14    | 3        | 2       | Y_3^2  |
| 15    | 3        | 3       | Y_3^3  |
| 16    | 4        | -4      | Y_4^{-4} |
| 17    | 4        | -3      | Y_4^{-3} |
| 18    | 4        | -2      | Y_4^{-2} |
| 19    | 4        | -1      | Y_4^{-1} |
| 20    | 4        | 0       | Y_4^0  |
| 21    | 4        | 1       | Y_4^1  |
| 22    | 4        | 2       | Y_4^2  |
| 23    | 4        | 3       | Y_4^3  |
| 24    | 4        | 4       | Y_4^4  |

## Technical Design

### Key Functions to Port

| Function | Upstream Location | MLX Approach |
|----------|-------------------|-------------|
| `_eval_sh_bases_fast(basis_dim, dirs)` | `_torch_impl.py:673-754` | Build list of basis values, `mx.stack` at end (no in-place assignment) |
| `_spherical_harmonics(degrees_to_use, dirs, coeffs)` | `_torch_impl.py:757-775` | Normalize dirs, compute bases, dot product via `mx.sum(expand_dims * coeffs)` |
| `spherical_harmonics` (public) | `_wrapper.py:330-360` | `@mx.custom_function` with `.vjp` for backward |

### Complete SH Coefficient Table

All 25 SH basis functions evaluated at unit direction (x, y, z). These are the real spherical harmonics using the Sloan 2013 fast evaluation scheme. Every constant below is reproduced exactly from the upstream source.

#### Degree 0 (1 basis function)

| Index | Formula | Constant | Derivation |
|-------|---------|----------|------------|
| 0 | `C0` | `0.2820947917738781` | `0.5 * sqrt(1/pi)` |

#### Degree 1 (3 basis functions, indices 1-3)

Computed using `fTmpA = -0.48860251190292` (which is `-0.5 * sqrt(3/pi)`):

| Index | Formula | Expression |
|-------|---------|------------|
| 1 | `fTmpA * y` | `-0.48860251190292 * y` |
| 2 | `-fTmpA * z` | `0.48860251190292 * z` |
| 3 | `fTmpA * x` | `-0.48860251190292 * x` |

#### Degree 2 (5 basis functions, indices 4-8)

Intermediate values:
```
z2 = z * z
fTmpB = -1.092548430592079 * z      # -sqrt(15/pi) * z / 2
fTmpA = 0.5462742152960395           # sqrt(15/pi) / 4
fC1 = x*x - y*y
fS1 = 2*x*y
```

| Index | Formula | Expression |
|-------|---------|------------|
| 4 | `fTmpA * fS1` | `0.5462742152960395 * 2*x*y` |
| 5 | `fTmpB * y` | `-1.092548430592079 * z * y` |
| 6 | `0.9461746957575601 * z2 - 0.3153915652525201` | `0.25*sqrt(5/pi) * (3*z^2 - 1)` |
| 7 | `fTmpB * x` | `-1.092548430592079 * z * x` |
| 8 | `fTmpA * fC1` | `0.5462742152960395 * (x^2 - y^2)` |

#### Degree 3 (7 basis functions, indices 9-15)

Intermediate values:
```
fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
fTmpB = 1.445305721320277 * z
fTmpA = -0.5900435899266435
fC2 = x * fC1 - y * fS1    # cos(2*phi) recurrence
fS2 = x * fS1 + y * fC1    # sin(2*phi) recurrence
```

| Index | Formula | Expression |
|-------|---------|------------|
| 9  | `fTmpA * fS2` | `-0.5900435899266435 * (x*fS1 + y*fC1)` |
| 10 | `fTmpB * fS1` | `1.445305721320277 * z * 2*x*y` |
| 11 | `fTmpC * y` | `(-2.285228997322329*z^2 + 0.4570457994644658) * y` |
| 12 | `z * (1.865881662950577*z2 - 1.119528997770346)` | `z * (1.865881662950577*z^2 - 1.119528997770346)` |
| 13 | `fTmpC * x` | `(-2.285228997322329*z^2 + 0.4570457994644658) * x` |
| 14 | `fTmpB * fC1` | `1.445305721320277 * z * (x^2 - y^2)` |
| 15 | `fTmpA * fC2` | `-0.5900435899266435 * (x*fC1 - y*fS1)` |

#### Degree 4 (9 basis functions, indices 16-24)

Intermediate values:
```
fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
fTmpC = 3.31161143515146 * z2 - 0.47308734787878
fTmpB = -1.770130769779931 * z
fTmpA = 0.6258357354491763
fC3 = x * fC2 - y * fS2    # cos(3*phi) recurrence
fS3 = x * fS2 + y * fC2    # sin(3*phi) recurrence
```

| Index | Formula | Expression |
|-------|---------|------------|
| 16 | `fTmpA * fS3` | `0.6258357354491763 * (x*fS2 + y*fC2)` |
| 17 | `fTmpB * fS2` | `-1.770130769779931 * z * fS2` |
| 18 | `fTmpC * fS1` | `(3.31161143515146*z^2 - 0.47308734787878) * 2*x*y` |
| 19 | `fTmpD * y` | `z*(-4.683325804901025*z^2 + 2.007139630671868) * y` |
| 20 | compound | `1.984313483298443*z^2*(1.865881662950577*z^2 - 1.119528997770346) + -1.006230589874905*(0.9461746957575601*z^2 - 0.3153915652525201)` |
| 21 | `fTmpD * x` | `z*(-4.683325804901025*z^2 + 2.007139630671868) * x` |
| 22 | `fTmpC * fC1` | `(3.31161143515146*z^2 - 0.47308734787878) * (x^2 - y^2)` |
| 23 | `fTmpB * fC2` | `-1.770130769779931 * z * fC2` |
| 24 | `fTmpA * fC3` | `0.6258357354491763 * (x*fC2 - y*fS2)` |

#### Summary of All Named Constants

| Constant | Value | Appears in |
|----------|-------|------------|
| `C0` (degree 0 basis) | `0.2820947917738781` | Basis 0 |
| `fTmpA` (degree 1) | `-0.48860251190292` | Bases 1-3 |
| `fTmpB` (degree 2, z-scaled) | `-1.092548430592079 * z` | Bases 5, 7 |
| `fTmpA` (degree 2) | `0.5462742152960395` | Bases 4, 8 |
| Degree 2 zonal | `0.9461746957575601`, `-0.3153915652525201` | Basis 6 |
| `fTmpC` (degree 3) | `-2.285228997322329 * z^2 + 0.4570457994644658` | Bases 11, 13 |
| `fTmpB` (degree 3) | `1.445305721320277 * z` | Bases 10, 14 |
| `fTmpA` (degree 3) | `-0.5900435899266435` | Bases 9, 15 |
| Degree 3 zonal | `1.865881662950577`, `-1.119528997770346` | Basis 12 |
| `fTmpD` (degree 4) | `z * (-4.683325804901025 * z^2 + 2.007139630671868)` | Bases 19, 21 |
| `fTmpC` (degree 4) | `3.31161143515146 * z^2 - 0.47308734787878` | Bases 18, 22 |
| `fTmpB` (degree 4) | `-1.770130769779931 * z` | Bases 17, 23 |
| `fTmpA` (degree 4) | `0.6258357354491763` | Bases 16, 24 |
| Degree 4 zonal inner | `1.984313483298443` | Basis 20 |
| Degree 4 zonal outer | `-1.006230589874905` | Basis 20 |

### MLX Implementation Details

#### torch-to-mlx Mapping for This Module

| torch | mlx |
|-------|-----|
| `torch.empty(shape, dtype, device)` | `mx.zeros(shape, dtype=dtype)` (MLX has no uninitialized) |
| `result[..., i] = val` | Not supported; build list and `mx.stack` |
| `x.unbind(-1)` | `x[..., 0], x[..., 1], x[..., 2]` |
| `F.normalize(x, p=2, dim=-1)` | `x / mx.maximum(mx.sqrt(mx.sum(x*x, axis=-1, keepdims=True)), 1e-8)` |
| `torch.zeros_like(coeffs[..., 0])` | `mx.zeros(coeffs.shape[:-1], dtype=coeffs.dtype)` |
| `(a * b).sum(dim=-2)` | `mx.sum(a * b, axis=-2)` |
| `mx.full(shape, val)` | `mx.full(shape, val, dtype=dtype)` |

#### _eval_sh_bases_fast (MLX)

The upstream writes to pre-allocated tensor slots `result[..., i] = value`. MLX arrays are immutable, so we build the result by computing all basis values as separate arrays and stacking at the end:

```python
def _eval_sh_bases_fast(basis_dim: int, dirs: mx.array) -> mx.array:
    """Evaluate SH basis functions at unit directions.

    Uses the fast method from "Efficient Spherical Harmonic Evaluation"
    (Peter-Pike Sloan, JCGT 2013).

    Args:
        basis_dim: Number of basis functions (1, 4, 9, 16, or 25).
        dirs: Unit directions [..., 3].

    Returns:
        SH basis values [..., basis_dim].
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    # Degree 0: 1 basis function
    b0 = mx.full(x.shape, 0.2820947917738781, dtype=dirs.dtype)
    bases = [b0]

    if basis_dim <= 1:
        return mx.stack(bases, axis=-1)

    # Degree 1: 3 basis functions (indices 1, 2, 3)
    fTmpA = -0.48860251190292
    b1 = fTmpA * y        # Y_1^{-1}
    b2 = -fTmpA * z       # Y_1^0
    b3 = fTmpA * x        # Y_1^1
    bases.extend([b1, b2, b3])

    if basis_dim <= 4:
        return mx.stack(bases, axis=-1)

    # Degree 2: 5 basis functions (indices 4-8)
    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    b4 = fTmpA * fS1
    b5 = fTmpB * y
    b6 = 0.9461746957575601 * z2 - 0.3153915652525201
    b7 = fTmpB * x
    b8 = fTmpA * fC1
    bases.extend([b4, b5, b6, b7, b8])

    if basis_dim <= 9:
        return mx.stack(bases, axis=-1)

    # Degree 3: 7 basis functions (indices 9-15)
    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    b9  = fTmpA * fS2
    b10 = fTmpB * fS1
    b11 = fTmpC * y
    b12 = z * (1.865881662950577 * z2 - 1.119528997770346)
    b13 = fTmpC * x
    b14 = fTmpB * fC1
    b15 = fTmpA * fC2
    bases.extend([b9, b10, b11, b12, b13, b14, b15])

    if basis_dim <= 16:
        return mx.stack(bases, axis=-1)

    # Degree 4: 9 basis functions (indices 16-24)
    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    b16 = fTmpA * fS3
    b17 = fTmpB * fS2
    b18 = fTmpC * fS1
    b19 = fTmpD * y
    b20 = (1.984313483298443 * z2 *
           (1.865881662950577 * z2 - 1.119528997770346) +
           -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201))
    b21 = fTmpD * x
    b22 = fTmpC * fC1
    b23 = fTmpB * fC2
    b24 = fTmpA * fC3
    bases.extend([b16, b17, b18, b19, b20, b21, b22, b23, b24])

    return mx.stack(bases, axis=-1)
```

#### spherical_harmonics (with custom VJP)

```python
@mx.custom_function
def spherical_harmonics(degrees_to_use: int, dirs: mx.array, coeffs: mx.array) -> mx.array:
    """Evaluate spherical harmonics to produce RGB colors.

    Args:
        degrees_to_use: SH degree (0-4).
        dirs: View directions [..., 3] (not necessarily normalized).
        coeffs: SH coefficients [..., K, 3] where K >= (degrees_to_use+1)^2.

    Returns:
        RGB colors [..., 3].
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], (
        f"coeffs has {coeffs.shape[-2]} bases but degree {degrees_to_use} "
        f"requires {(degrees_to_use + 1) ** 2}"
    )

    # Normalize directions
    dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
    dirs_normalized = dirs / mx.maximum(dirs_norm, 1e-8)

    num_bases = (degrees_to_use + 1) ** 2
    bases = _eval_sh_bases_fast(num_bases, dirs_normalized)  # [..., num_bases]

    # Pad bases to match coeffs dimension if K > num_bases
    K = coeffs.shape[-2]
    if num_bases < K:
        padding = mx.zeros(bases.shape[:-1] + (K - num_bases,), dtype=bases.dtype)
        bases = mx.concatenate([bases, padding], axis=-1)

    # Dot product: sum over basis dimension
    # bases: [..., K], coeffs: [..., K, 3]
    # result: [..., 3]
    return mx.sum(mx.expand_dims(bases, axis=-1) * coeffs, axis=-2)


@spherical_harmonics.vjp
def sh_vjp(primals, cotangent, output):
    degrees_to_use, dirs, coeffs = primals
    v_colors = cotangent  # [..., 3]

    # Recompute forward intermediates
    dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
    dirs_normalized = dirs / mx.maximum(dirs_norm, 1e-8)
    num_bases = (degrees_to_use + 1) ** 2
    bases = _eval_sh_bases_fast(num_bases, dirs_normalized)

    K = coeffs.shape[-2]
    if num_bases < K:
        padding = mx.zeros(bases.shape[:-1] + (K - num_bases,), dtype=bases.dtype)
        bases = mx.concatenate([bases, padding], axis=-1)

    # ----- Gradient w.r.t. coeffs -----
    # colors = sum_k bases[k] * coeffs[k, :]
    # d(loss)/d(coeffs[k, c]) = d(loss)/d(colors[c]) * bases[k]
    # v_coeffs[..., k, c] = v_colors[..., c] * bases[..., k]
    v_coeffs = mx.expand_dims(bases, axis=-1) * mx.expand_dims(v_colors, axis=-2)

    # ----- Gradient w.r.t. dirs -----
    # Use mx.vjp on a helper that computes colors from dirs only (coeffs fixed)
    # This is the practical approach: let MLX auto-diff through the polynomial
    # SH basis computation and the normalization chain rule.
    def fwd_for_dirs_grad(d):
        d_norm = mx.sqrt(mx.sum(d * d, axis=-1, keepdims=True))
        d_normalized = d / mx.maximum(d_norm, 1e-8)
        b = _eval_sh_bases_fast(num_bases, d_normalized)
        if num_bases < K:
            b = mx.concatenate(
                [b, mx.zeros(b.shape[:-1] + (K - num_bases,), dtype=b.dtype)],
                axis=-1,
            )
        return mx.sum(mx.expand_dims(b, axis=-1) * coeffs, axis=-2)

    _, v_dirs_fn = mx.vjp(fwd_for_dirs_grad, (dirs,), (v_colors,))
    v_dirs = v_dirs_fn[0]

    return (None, v_dirs, v_coeffs)  # None for degrees_to_use (int, not differentiable)
```

### Full VJP Derivation

The forward computation is:

```
colors[..., c] = sum_{k=0}^{K-1} bases[..., k] * coeffs[..., k, c]
```

where `bases[..., k] = Y_k(normalize(dirs))`.

#### Gradient w.r.t. coeffs (linear, closed-form)

Since `colors` is linear in `coeffs`:

```
d(loss)/d(coeffs[..., k, c]) = d(loss)/d(colors[..., c]) * d(colors[..., c])/d(coeffs[..., k, c])
                               = v_colors[..., c] * bases[..., k]
```

In array form:
```
v_coeffs = bases[..., :, None] * v_colors[..., None, :]   # [..., K, 3]
```

This is exact and requires no chain rule -- it is a simple outer product.

#### Gradient w.r.t. dirs (chain rule through normalization + polynomial)

The chain has three stages:

**Stage 1: colors -> bases**
```
d(loss)/d(bases[..., k]) = sum_c v_colors[..., c] * coeffs[..., k, c]
```
i.e., `v_bases = mx.sum(v_colors[..., None, :] * coeffs, axis=-1)  # [..., K]`

**Stage 2: bases -> normalized direction (x, y, z)**

Each basis Y_k is a polynomial in (x, y, z). The Jacobian dY_k/d(x,y,z) for each basis is:

Degree 0:
```
dY_0/dx = dY_0/dy = dY_0/dz = 0   (constant)
```

Degree 1 (using A = -0.48860251190292):
```
dY_1/dy = A,          dY_1/dx = dY_1/dz = 0
dY_2/dz = -A,         dY_2/dx = dY_2/dy = 0
dY_3/dx = A,          dY_3/dy = dY_3/dz = 0
```

Degree 2 (using B_coeff = -1.092548430592079, A_coeff = 0.5462742152960395):
```
dY_4/dx = A_coeff * 2y = 2*A_coeff*y,     dY_4/dy = A_coeff * 2x = 2*A_coeff*x,     dY_4/dz = 0
dY_5/dy = B_coeff * z ... (partial derivatives of each quadratic term)
```

The full Jacobian for all 25 bases is a 25x3 matrix of polynomial expressions in (x, y, z). Computing this analytically for all degrees is tedious but straightforward since every basis is a polynomial.

**Stage 3: normalized direction -> raw direction (normalization chain rule)**

Given `d_hat = d / ||d||`, the Jacobian is:
```
d(d_hat_i)/d(d_j) = (delta_ij - d_hat_i * d_hat_j) / ||d||
```

This is the standard normalization Jacobian (projection onto the tangent plane of the sphere).

**Practical approach**: Rather than manually implementing the 25x3 Jacobian for all SH bases plus the normalization Jacobian, we use `mx.vjp` on a helper function that computes `colors` from `dirs` with `coeffs` held fixed. Since `_eval_sh_bases_fast` is composed entirely of standard MLX arithmetic operations (multiply, add, subtract), MLX's automatic differentiation handles the chain rule correctly. This is the approach used in the implementation above.

**Performance note**: The `mx.vjp` approach recomputes the forward through the SH bases. For the MVP this is acceptable. A future optimization could cache the Jacobian or implement closed-form derivatives for the hot path.

### Data Flow

```
dirs [..., 3]            coeffs [..., K, 3]
     |                        |
     v                        |
normalize (/ ||d||)            |
     |                        |
     v                        |
_eval_sh_bases_fast            |
     |                        |
     v                        v
bases [..., K]    *    coeffs [..., K, 3]
            \         /
             \       /
         sum over K axis
                |
                v
         colors [..., 3]
```

**Typical shapes**:
- N=10000 Gaussians, degree 3: `dirs=[10000, 3]`, `coeffs=[10000, 16, 3]` -> `colors=[10000, 3]`
- C cameras, N Gaussians: `dirs=[C, N, 3]`, `coeffs=[C, N, 25, 3]` -> `colors=[C, N, 3]`
- Memory: for N=100k, degree 3: bases is 100k * 16 * 4 bytes = 6.4 MB (negligible)

### File Layout

```
src/gsplat_mlx/core/spherical_harmonics.py
    _eval_sh_bases_fast(basis_dim, dirs)     # internal
    spherical_harmonics(degrees_to_use, dirs, coeffs)  # public, @mx.custom_function
    sh_vjp(primals, cotangent, output)       # VJP registered on spherical_harmonics

tests/test_spherical_harmonics.py
    TestEvalSHBasesFast                      # unit tests for basis evaluation
    TestSphericalHarmonics                   # forward tests
    TestSphericalHarmonicsVJP               # backward/gradient tests
    TestSphericalHarmonicsCrossFramework    # torch comparison tests
```

## Test Plan

### Exact Test Vectors

These test vectors can be used to verify correctness without depending on the torch reference. All values are computed from the exact coefficient formulas.

#### Test Vector 1: Positive z-axis direction

Direction: `d = [0, 0, 1]` (unit vector, so x=0, y=0, z=1)

Expected basis values:
| Index | Value | Computation |
|-------|-------|-------------|
| 0 | `0.2820947917738781` | constant |
| 1 | `0.0` | `fTmpA * y = fTmpA * 0` |
| 2 | `0.48860251190292` | `-fTmpA * z = -(-0.48860251190292) * 1` |
| 3 | `0.0` | `fTmpA * x = fTmpA * 0` |
| 4 | `0.0` | `fTmpA * 2xy = 0` |
| 5 | `0.0` | `fTmpB * y = 0` |
| 6 | `0.63078313050504` | `0.9461746957575601 * 1 - 0.3153915652525201` |
| 7 | `0.0` | `fTmpB * x = 0` |
| 8 | `0.0` | `fTmpA * (x^2 - y^2) = 0` |

#### Test Vector 2: Positive x-axis direction

Direction: `d = [1, 0, 0]` (x=1, y=0, z=0)

Expected basis values (degrees 0-2):
| Index | Value | Computation |
|-------|-------|-------------|
| 0 | `0.2820947917738781` | constant |
| 1 | `0.0` | `fTmpA * 0` |
| 2 | `0.0` | `-fTmpA * 0` |
| 3 | `-0.48860251190292` | `fTmpA * 1` |
| 4 | `0.0` | `fTmpA * 0` |
| 5 | `0.0` | `0` |
| 6 | `-0.3153915652525201` | `0.9461746957575601 * 0 - 0.3153915652525201` |
| 7 | `0.0` | `0` |
| 8 | `0.5462742152960395` | `fTmpA * (1 - 0)` |

#### Test Vector 3: Diagonal direction

Direction: `d = [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]` (x=y=z=0.57735026918962576)

This exercises all cross-terms. Key basis values:
| Index | Value |
|-------|-------|
| 0 | `0.2820947917738781` |
| 1 | `-0.28209479177387814` (= `fTmpA * 1/sqrt(3)`) |
| 2 | `0.28209479177387814` (= `-fTmpA * 1/sqrt(3)`) |
| 3 | `-0.28209479177387814` (= `fTmpA * 1/sqrt(3)`) |
| 4 | `0.36418281193387295` (= `fTmpA * 2/3`) |
| 6 | `0.0` (= `0.9461746957575601/3 - 0.3153915652525201`) |

#### Test Vector 4: Degree 0 constant color

```python
dirs = mx.array([[0.0, 0.0, 1.0]])           # [1, 3]
coeffs = mx.array([[[1.0, 0.5, 0.25]]])       # [1, 1, 3]
degrees_to_use = 0
# Expected output: 0.2820947917738781 * [1.0, 0.5, 0.25] = [0.28209, 0.14105, 0.07052]
```

#### Test Vector 5: Degree 1 directional color

```python
dirs = mx.array([[1.0, 0.0, 0.0]])            # [1, 3]
coeffs = mx.zeros((1, 4, 3))
coeffs[0, 0, :] = [1.0, 1.0, 1.0]            # DC term
coeffs[0, 3, :] = [1.0, 0.0, 0.0]            # Y_1^1 term (x-direction)
degrees_to_use = 1
# Expected: DC_contrib + x_contrib
# = 0.28209 * [1,1,1] + (-0.48860) * [1,0,0]
# = [0.28209 - 0.48860, 0.28209, 0.28209]
# = [-0.20651, 0.28209, 0.28209]
```

### Test Cases

| Test Case | Description | Tolerance |
|-----------|-------------|-----------|
| **Forward Tests** | | |
| `test_sh_degree0` | Single DC coefficient, constant color regardless of direction. Verify `output = C0 * coeffs[0]` | `atol=1e-6` |
| `test_sh_degree1` | 4 coefficients, verify view-direction dependence. Opposite directions should give different colors when non-DC coefficients are nonzero | `atol=1e-6` |
| `test_sh_degree2` | 9 coefficients, verify quadratic terms activate | `atol=1e-5` |
| `test_sh_degree3` | 16 coefficients, most common 3DGS setting | `atol=1e-5` |
| `test_sh_degree4` | 25 coefficients, verify all quartic terms | `atol=1e-5` |
| `test_sh_exact_vectors` | Verify against hand-computed test vectors above (z-axis, x-axis, diagonal) | `atol=1e-7` |
| `test_sh_normalized_dirs` | Pre-normalized and unnormalized (scaled by 5.0) directions produce identical output | `atol=1e-5` |
| `test_sh_batch` | Batch of N=1000 Gaussians with random directions and degree-3 coefficients | `atol=1e-5` |
| `test_sh_multi_camera` | C=4 cameras x N=500 Gaussians, shape `[C, N, 3]` -> `[C, N, 3]` | `atol=1e-5` |
| `test_sh_zero_direction` | Near-zero direction vector `[1e-10, 1e-10, 1e-10]` does not produce NaN or Inf | N/A (no NaN) |
| `test_sh_degree0_direction_invariant` | Degree 0 output is the same for 100 random directions (purely isotropic) | `atol=1e-7` |
| `test_sh_extra_coeffs_ignored` | If coeffs has K=25 but degrees_to_use=1, only first 4 bases used, rest zeroed | `atol=1e-7` |
| **Backward/VJP Tests** | | |
| `test_vjp_coeffs_degree0` | Gradient w.r.t. coeffs at degree 0: `v_coeffs[0, c] = C0 * v_colors[c]` | `atol=1e-5` |
| `test_vjp_coeffs_degree1` | Gradient w.r.t. coeffs at degree 1 | `atol=1e-5` |
| `test_vjp_coeffs_degree2` | Gradient w.r.t. coeffs at degree 2 | `atol=1e-5` |
| `test_vjp_coeffs_degree3` | Gradient w.r.t. coeffs at degree 3 | `atol=1e-5` |
| `test_vjp_dirs_degree1` | Gradient w.r.t. dirs at degree 1 (linear, should be exact) | `atol=1e-4` |
| `test_vjp_dirs_degree2` | Gradient w.r.t. dirs at degree 2 | `atol=1e-4` |
| `test_vjp_dirs_degree3` | Gradient w.r.t. dirs at degree 3 | `atol=1e-4` |
| `test_vjp_numerical_coeffs` | Finite-difference gradient check for coeffs (eps=1e-4, compare to VJP) at each degree | `rtol=1e-3` |
| `test_vjp_numerical_dirs` | Finite-difference gradient check for dirs (eps=1e-4, compare to VJP) at each degree | `rtol=1e-2` |
| `test_vjp_batch` | Gradient shapes correct for batch input [B, N, 3] | N/A (shape check) |
| **Cross-Framework Tests** | | |
| `test_cross_framework_sh_degree0` | Compare MLX vs torch `_spherical_harmonics` at degree 0 for 1000 random inputs | `atol=1e-5` |
| `test_cross_framework_sh_degree1` | Compare at degree 1 | `atol=1e-5` |
| `test_cross_framework_sh_degree2` | Compare at degree 2 | `atol=1e-5` |
| `test_cross_framework_sh_degree3` | Compare at degree 3 | `atol=1e-5` |
| `test_cross_framework_sh_degree4` | Compare at degree 4 | `atol=1e-5` |
| `test_cross_framework_backward_coeffs` | Compare VJP w.r.t. coeffs against torch autograd | `atol=1e-4` |
| `test_cross_framework_backward_dirs` | Compare VJP w.r.t. dirs against torch autograd | `atol=1e-3` |

### Finite Difference Gradient Check Pattern

```python
def _finite_diff_grad(fn, x, idx, eps=1e-4):
    """Compute numerical gradient of scalar fn w.r.t. x at flat index idx."""
    x_plus = x.copy()
    x_minus = x.copy()
    flat = x_plus.reshape(-1)
    flat[idx] = flat[idx] + eps
    x_plus = flat.reshape(x.shape)
    flat = x_minus.reshape(-1)
    flat[idx] = flat[idx] - eps
    x_minus = flat.reshape(x.shape)
    return (fn(x_plus) - fn(x_minus)) / (2 * eps)
```

### Tolerance Rationale

- **Forward (atol=1e-5)**: All SH computations are exact polynomial arithmetic. Float32 has ~7 digits of precision. The compound expressions in degree 4 (basis 20) involve nested multiplications that may lose 1-2 digits, but 1e-5 tolerance is conservative.
- **VJP for coeffs (atol=1e-5)**: The gradient w.r.t. coeffs is a simple multiplication by the basis value -- same precision as forward.
- **VJP for dirs (atol=1e-4)**: The chain rule through normalization involves division by `||d||` and subtraction of `d_hat * d_hat^T`, which can amplify errors. The polynomial derivatives compound through multiple products. 1e-4 is appropriate.
- **Finite difference (rtol=1e-2 to 1e-3)**: Finite differences themselves have O(eps^2) truncation error and O(machine_eps/eps) rounding error. With eps=1e-4 and float32, the best achievable accuracy is ~1e-3.

## Dependencies

- **PRD-01**: Dev environment (MLX installation, test infrastructure)
- **PRD-02**: `_safe_normalize` or inline normalization pattern (shared utility). No hard dependency -- normalization is inlined in this module for simplicity.

## Blocks

- **PRD-05** (Projection): Optionally evaluates SH to compute view-dependent colors during projection
- **PRD-07** (Rasterization): Needs per-pixel colors which come from SH evaluation
- **PRD-09** (Rendering API): Orchestrates SH evaluation as part of the full rendering pipeline

## Acceptance Criteria

- [ ] `_eval_sh_bases_fast` produces correct basis values for degrees 0-4, verified against exact test vectors
- [ ] `spherical_harmonics` matches torch `_spherical_harmonics` reference within `atol=1e-5` for all degrees (0-4) on 1000 random inputs
- [ ] Backward pass w.r.t. `coeffs` is correct: analytical VJP matches finite differences within `rtol=1e-3`
- [ ] Backward pass w.r.t. `dirs` is correct: VJP matches finite differences within `rtol=1e-2`
- [ ] No NaN or Inf for edge cases (zero directions, large coefficients, single-element batches)
- [ ] Supports arbitrary batch dimensions: scalar `[3]`, vector `[N, 3]`, matrix `[C, N, 3]`
- [ ] Extra coefficients beyond `(degrees_to_use+1)^2` are ignored (zeroed bases)
- [ ] Uses `mx.stack` pattern (not in-place assignment) for immutable MLX arrays
- [ ] File: `src/gsplat_mlx/core/spherical_harmonics.py`
- [ ] Tests: `tests/test_spherical_harmonics.py` -- all pass with `pytest tests/test_spherical_harmonics.py -v`

## Implementation Notes

### Why `mx.stack` instead of `mx.concatenate` for building bases

Both work, but `mx.stack` on a list of scalar arrays (each of shape `[...]`) is cleaner:
- `mx.stack([b0, b1, b2, b3], axis=-1)` produces `[..., 4]` directly
- `mx.concatenate` would require each basis to be `[..., 1]` first (via `mx.expand_dims`)
- `mx.stack` is the natural choice when combining same-shaped arrays along a new axis

### Why auto-diff for dirs gradient instead of closed-form

The Jacobian of all 25 SH bases w.r.t. (x, y, z) has 75 entries, each a polynomial expression. Writing and maintaining this manually is error-prone and provides minimal performance benefit for the MVP. The `mx.vjp` approach:
1. Is correct by construction (same code path as forward)
2. Requires no maintenance when the basis computation changes
3. Has acceptable performance (recomputes forward, ~2x cost)

A future optimization can replace this with closed-form derivatives if profiling shows the VJP is a bottleneck.

### MLX `@mx.custom_function` signature note

The `@mx.custom_function` decorator in MLX requires that the decorated function takes array arguments. The `degrees_to_use` parameter is an integer (not differentiable). In the VJP, it is received as part of `primals` but its gradient is `None`. If MLX's `@mx.custom_function` does not support non-array arguments directly, wrap the integer as a closure:

```python
def make_spherical_harmonics(degrees_to_use: int):
    @mx.custom_function
    def _sh(dirs: mx.array, coeffs: mx.array) -> mx.array:
        ...  # use degrees_to_use from closure

    @_sh.vjp
    def _sh_vjp(primals, cotangent, output):
        dirs, coeffs = primals  # only array args
        ...

    return _sh

# Usage:
sh_fn = make_spherical_harmonics(3)
colors = sh_fn(dirs, coeffs)
```

This closure pattern avoids issues with non-differentiable integer parameters. The implementation should test both patterns and use whichever MLX supports.
