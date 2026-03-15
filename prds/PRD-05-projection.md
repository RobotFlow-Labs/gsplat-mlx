# PRD-05: 3D Gaussian Projection Pipeline

## Status

| Field | Value |
|-------|-------|
| **PRD** | PRD-05 |
| **Title** | Fully-Fused 3D-to-2D Gaussian Projection |
| **Status** | Draft |
| **Dependencies** | PRD-01 (dev environment), PRD-02 (math_utils), PRD-03 (covariance) |
| **Blocks** | PRD-06 (tile intersection), PRD-07 (rasterization), PRD-09 (rendering API) |
| **Estimated Lines** | ~450 (projection.py) + ~20 (cameras.py) + ~600 (test_projection.py) |

---

## 1. Overview

This PRD ports the fully-fused 3D-to-2D Gaussian projection pipeline from gsplat's `_torch_impl.py` to Apple MLX. This is the **first stage of the rendering pipeline**: given 3D Gaussian parameters (means, covariances) and camera parameters (view matrices, intrinsics), compute 2D screen-space positions, inverse-covariance conics, z-depths, and per-axis radii.

The projection pipeline is the most mathematically dense component in the splatting renderer. It chains together:

1. **World-to-camera transform** -- rotate and translate 3D Gaussians into camera coordinates
2. **Camera-to-screen projection** -- project 3D Gaussians into 2D ellipses using the EWA (Elliptical Weighted Average) splatting Jacobian approximation
3. **Covariance regularization** -- add eps2d to prevent degenerate 2D covariances
4. **Conic computation** -- invert the 2D covariance to get the conic representation used by the rasterizer
5. **Radius computation** -- compute per-axis screen-space radii for tile intersection
6. **Frustum and screen-bounds culling** -- zero out radii for invisible Gaussians

Three camera models are supported: **pinhole** (perspective), **fisheye** (equidistant), and **ortho** (orthographic).

---

## 2. Source Reference

| Item | Location | Lines |
|------|----------|-------|
| `_persp_proj` | `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:31-86` | 56 |
| `_fisheye_proj` | `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:89-155` | 67 |
| `_ortho_proj` | `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:158-200` | 43 |
| `_world_to_cam` | `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:203-236` | 34 |
| `_fully_fused_projection` | `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:239-327` | 89 |
| `fully_fused_projection` (public API) | `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:467-619` | 153 |
| `_FullyFusedProjection` (autograd) | `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:1431-1561` | 131 |
| `CameraModel` type | `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:36` | 1 |

**Total upstream lines**: ~574

---

## 3. Scope

### 3.1 In Scope

- `world_to_cam(means, covars, viewmats)` -- world-to-camera coordinate transform
- `persp_proj(means, covars, Ks, width, height)` -- perspective projection with EWA Jacobian
- `fisheye_proj(means, covars, Ks, width, height)` -- fisheye (equidistant) projection
- `ortho_proj(means, covars, Ks, width, height)` -- orthographic projection
- `fully_fused_projection(means, covars, viewmats, Ks, ...)` -- fused pipeline entry point
- Public API wrapper supporting both covariance and quaternion+scale input modes
- Custom VJP via `@mx.custom_function` for backward pass through the fused pipeline
- Frustum culling (near/far plane filtering)
- Screen-bounds culling (out-of-image filtering)
- Radius computation from 2D covariance diagonal
- Compensation factor computation for anti-aliased rendering
- `CameraModel` type alias in `cameras.py`
- Arbitrary batch dimensions `[..., N, 3]` / `[..., C, 4, 4]`

### 3.2 Out of Scope

- Packed output format (`_FullyFusedProjectionPacked`) -- deferred to future PRD
- Unscented transform projection (`fully_fused_projection_with_ut`)
- `ftheta` camera model (requires UT)
- `lidar` camera model (upstream-only)
- Rolling shutter, radial distortion coefficients
- 2DGS projection (`_FullyFusedProjection2DGS`)
- Sparse gradient support
- `radius_clip` parameter (set to 0.0 always in MVP)
- Opacity-based radius adjustment

---

## 4. Technical Design

### 4.1 File Layout

```
src/gsplat_mlx/
  core/
    cameras.py          # CameraModel type alias (NEW)
    projection.py       # All projection functions (NEW)
tests/
  test_projection.py    # Comprehensive tests (NEW)
```

### 4.2 CameraModel Type (cameras.py)

```python
"""Camera model definitions for gsplat-mlx."""

from typing import Literal

# Camera models supported in gsplat-mlx MVP.
# "ftheta" and "lidar" are upstream-only and NOT ported.
CameraModel = Literal["pinhole", "ortho", "fisheye"]
```

This is a simple type alias. Downstream code uses it for type annotations and dispatch.

### 4.3 Tensor Shape Conventions

All functions use the following shape conventions, matching upstream exactly:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `means` | `[..., N, 3]` | 3D Gaussian positions in world space |
| `covars` | `[..., N, 3, 3]` | 3D covariance matrices (symmetric PSD) |
| `viewmats` | `[..., C, 4, 4]` | World-to-camera transformation matrices |
| `Ks` | `[..., C, 3, 3]` | Camera intrinsic matrices |
| `width` | `int` | Image width in pixels |
| `height` | `int` | Image height in pixels |

Where `...` denotes optional batch dimensions, `C` is the number of cameras, and `N` is the number of Gaussians.

**After `world_to_cam`**, the camera-space tensors have shape `[..., C, N, 3]` and `[..., C, N, 3, 3]` -- the camera dimension is outer because each camera sees all N Gaussians.

**Outputs:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `radii` | `[..., C, N, 2]` | Per-axis screen-space radii (int32). 0 = culled. |
| `means2d` | `[..., C, N, 2]` | 2D pixel positions |
| `depths` | `[..., C, N]` | Z-depth in camera frame |
| `conics` | `[..., C, N, 3]` | Inverse 2D covariance as `[a, b, c]` where `cov2d_inv = [[a,b],[b,c]]` |
| `compensations` | `[..., C, N]` or `None` | Anti-aliasing compensation factor |

### 4.4 Function: `world_to_cam`

**Purpose**: Transform Gaussian means and covariances from world coordinates to camera coordinates.

**Mathematical formulation**:

Given rotation `R = viewmat[:3, :3]` and translation `t = viewmat[:3, 3]`:

```
mean_c = R @ mean_w + t
covar_c = R @ covar_w @ R^T
```

For multiple cameras `C` and multiple Gaussians `N`, this is broadcast using einsum.

**MLX implementation**:

```python
def world_to_cam(
    means: mx.array,    # [..., N, 3]
    covars: mx.array,   # [..., N, 3, 3]
    viewmats: mx.array, # [..., C, 4, 4]
) -> Tuple[mx.array, mx.array]:
    """Transform Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian centers in world space. [..., N, 3].
        covars: Gaussian covariances in world space. [..., N, 3, 3].
        viewmats: World-to-camera 4x4 matrices. [..., C, 4, 4].

    Returns:
        means_c: Gaussian centers in camera space. [..., C, N, 3].
        covars_c: Gaussian covariances in camera space. [..., C, N, 3, 3].
    """
    R = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t = viewmats[..., :3, 3]   # [..., C, 3]

    # R @ means^T + t, broadcast over C cameras and N Gaussians
    # "...cij,...nj->...cni": for each camera c, apply R_c to each Gaussian n
    means_c = mx.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]

    # R @ covars @ R^T, broadcast similarly
    # "...cij,...njk,...clk->...cnil": R_c @ covar_n @ R_c^T
    covars_c = mx.einsum("...cij,...njk,...clk->...cnil", R, covars, R)

    return means_c, covars_c
```

**Key translation notes**:
- `torch.einsum` -> `mx.einsum` (MLX supports einsum natively)
- The ellipsis notation handles arbitrary batch dimensions
- Broadcasting `t[..., None, :]` adds the Gaussian dimension for addition

**Edge cases**:
- Identity viewmat: `means_c == means`, `covars_c == covars`
- Zero-covariance Gaussians: pass through correctly (zero stays zero under rotation)

### 4.5 Function: `persp_proj` (Perspective Projection)

**Purpose**: Project 3D Gaussians in camera space to 2D screen space using the pinhole camera model with the EWA (Elliptical Weighted Average) splatting approximation.

#### 4.5.1 Perspective Projection Jacobian Derivation

The pinhole projection maps a 3D point `(X, Y, Z)` in camera space to pixel coordinates `(u, v)`:

```
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

The Jacobian of this mapping with respect to `(X, Y, Z)` is:

```
J = d(u,v) / d(X,Y,Z) = [ du/dX  du/dY  du/dZ ]
                          [ dv/dX  dv/dY  dv/dZ ]

J = [ fx/Z    0     -fx*X/Z^2 ]
    [  0     fy/Z   -fy*Y/Z^2 ]
```

This is a `2x3` matrix. The EWA splatting approximation projects the 3D covariance to 2D via:

```
Sigma_2D = J @ Sigma_3D @ J^T
```

This is a first-order (linear) approximation valid when the Gaussian is small relative to the distance from the camera.

#### 4.5.2 FOV Clamping

To handle Gaussians at extreme angles (near the edge of the field of view), the upstream code clamps the normalized camera-space coordinates before computing the Jacobian. This prevents the Jacobian from having extremely large values that cause numerical instability.

The clamping limits are computed as:

```
tan_fovx = 0.5 * width / fx
tan_fovy = 0.5 * height / fy

lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
lim_x_neg = cx / fx + 0.3 * tan_fovx
lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
lim_y_neg = cy / fy + 0.3 * tan_fovy
```

The 0.3 factor provides a 30% margin beyond the image boundary, allowing Gaussians to extend slightly beyond the visible area while still being properly projected.

After clamping:

```
tx_clamped = tz * clip(tx / tz, -lim_x_neg, lim_x_pos)
ty_clamped = tz * clip(ty / tz, -lim_y_neg, lim_y_pos)
```

The clamped values are used in the Jacobian (for the `-fx*X/Z^2` and `-fy*Y/Z^2` terms), but the **original** values are used for computing `means2d`. This is critical: we clamp the Jacobian to stabilize the covariance projection, but we do NOT clamp the mean projection.

#### 4.5.3 MLX Implementation

```python
def persp_proj(
    means: mx.array,   # [..., C, N, 3]  (camera space)
    covars: mx.array,  # [..., C, N, 3, 3]
    Ks: mx.array,      # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Perspective projection of 3D Gaussians to 2D.

    Uses EWA splatting: projects the 3D covariance through the perspective
    Jacobian to obtain a 2D covariance ellipse.

    Args:
        means: Gaussian means in camera space. [..., C, N, 3].
        covars: Gaussian covariances in camera space. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        means2d: Projected 2D means. [..., C, N, 2].
        cov2d: Projected 2D covariances. [..., C, N, 2, 2].
    """
    tx, ty, tz = means[..., 0], means[..., 1], means[..., 2]
    tz2 = tz * tz

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]

    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy

    # Asymmetric clamping limits (supports off-center principal point)
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    # Clamp for Jacobian stability (NOT for means2d)
    tx = tz * mx.clip(tx / tz, -lim_x_neg, lim_x_pos)
    ty = tz * mx.clip(ty / tz, -lim_y_neg, lim_y_pos)

    # Build 2x3 Jacobian: [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    O = mx.zeros_like(tx)
    J = mx.stack(
        [fx / tz, O, -fx * tx / tz2,
         O, fy / tz, -fy * ty / tz2],
        axis=-1,
    )
    J = J.reshape(J.shape[:-1] + (2, 3))

    # Project covariance: J @ covars @ J^T
    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    # Project means: K[:2,:3] @ mean_3d / z
    means2d = mx.einsum("...ij,...nj->...ni", Ks[..., :2, :3], means)
    means2d = means2d / tz[..., None]

    return means2d, cov2d
```

**Key torch-to-MLX translations**:

| torch | MLX |
|-------|-----|
| `torch.unbind(means, dim=-1)` | `means[..., 0], means[..., 1], means[..., 2]` |
| `torch.clamp(x, min=a, max=b)` | `mx.clip(x, a, b)` |
| `torch.zeros(shape, device=..., dtype=...)` | `mx.zeros_like(tx)` |
| `torch.stack([...], dim=-1)` | `mx.stack([...], axis=-1)` |
| `.reshape(...)` | `.reshape(...)` (same API) |
| `torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1,-2))` | `mx.einsum("...ij,...jk,...lk->...il", J, covars, J)` |

Note the einsum difference: upstream uses explicit `.transpose(-1,-2)` on J, while we fold it into the subscript (`...lk` instead of `...kl`). This avoids a transpose allocation.

**Important**: The upstream uses `J.transpose(-1, -2)` explicitly for `J @ covars @ J^T`. In our einsum, the third subscript `...lk` handles the transpose implicitly because `J_{lk}` with `l` as the contracted index gives `J^T`.

### 4.6 Function: `fisheye_proj` (Fisheye / Equidistant Projection)

**Purpose**: Project 3D Gaussians using the equidistant fisheye model, which maps angles linearly to pixel distances from the principal point.

#### 4.6.1 Fisheye Projection Model

The equidistant fisheye model maps a 3D point `(X, Y, Z)` to pixel `(u, v)` via:

```
r = sqrt(X^2 + Y^2)             # distance from optical axis
theta = atan2(r, Z)              # angle from optical axis
u = fx * X * theta / r + cx
v = fy * Y * theta / r + cy
```

The Jacobian is more complex due to the nonlinear `atan2` and `sqrt`:

```
a = Z / ((X^2+Y^2+Z^2) * (X^2+Y^2))      # radial derivative term
b = atan2(r, Z) / (r * (X^2+Y^2))          # angular normalization term

J = [ fx * (X^2*a + Y^2*b),    fx * X*Y*(a-b),      -fx * X / (X^2+Y^2+Z^2) ]
    [ fy * X*Y*(a-b),          fy * (Y^2*a + X^2*b), -fy * Y / (X^2+Y^2+Z^2) ]
```

#### 4.6.2 MLX Implementation

```python
def fisheye_proj(
    means: mx.array,   # [..., C, N, 3]
    covars: mx.array,  # [..., C, N, 3, 3]
    Ks: mx.array,      # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Fisheye (equidistant) projection of 3D Gaussians to 2D.

    Args:
        means: Gaussian means in camera space. [..., C, N, 3].
        covars: Gaussian covariances in camera space. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        means2d: Projected 2D means. [..., C, N, 2].
        cov2d: Projected 2D covariances. [..., C, N, 2, 2].
    """
    x, y, z = means[..., 0], means[..., 1], means[..., 2]

    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]

    eps = 1e-7
    xy_len = mx.sqrt(x * x + y * y) + eps
    theta = mx.arctan2(xy_len, z + eps)

    means2d = mx.stack(
        [x * fx * theta / xy_len + cx,
         y * fy * theta / xy_len + cy],
        axis=-1,
    )  # [..., C, N, 2]

    # Jacobian components
    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = mx.arctan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / x2y2

    J = mx.stack(
        [fx * (x2 * a + y2 * b),
         fx * xy * (a - b),
         -fx * x * x2y2z2_inv,
         fy * xy * (a - b),
         fy * (y2 * a + x2 * b),
         -fy * y * x2y2z2_inv],
        axis=-1,
    ).reshape(means.shape[:-1] + (2, 3))

    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    return means2d, cov2d
```

**Key differences from perspective**:
- Uses `mx.arctan2` instead of simple division
- More complex Jacobian with `a` and `b` terms
- Small epsilon `1e-7` to avoid division by zero at optical axis (where `xy_len = 0`)
- No FOV clamping (fisheye can handle very wide angles)

### 4.7 Function: `ortho_proj` (Orthographic Projection)

**Purpose**: Project 3D Gaussians using orthographic projection, where depth does not affect the projected position or size (no perspective foreshortening).

#### 4.7.1 Orthographic Projection Model

```
u = fx * X + cx
v = fy * Y + cy
```

The Jacobian is constant (does not depend on the 3D point):

```
J = [ fx  0  0 ]
    [  0  fy 0 ]
```

#### 4.7.2 MLX Implementation

```python
def ortho_proj(
    means: mx.array,   # [..., C, N, 3]
    covars: mx.array,  # [..., C, N, 3, 3]
    Ks: mx.array,      # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Orthographic projection of 3D Gaussians to 2D.

    Args:
        means: Gaussian means in camera space. [..., C, N, 3].
        covars: Gaussian covariances in camera space. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        means2d: Projected 2D means. [..., C, N, 2].
        cov2d: Projected 2D covariances. [..., C, N, 2, 2].
    """
    batch_dims = means.shape[:-3]
    C = means.shape[-3]
    N = means.shape[-2]

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]

    # Build constant 2x3 Jacobian: [[fx, 0, 0], [0, fy, 0]]
    O = mx.zeros(batch_dims + (C, 1), dtype=means.dtype)
    J = mx.stack([fx, O, O, O, fy, O], axis=-1)
    J = J.reshape(batch_dims + (C, 1, 2, 3))
    # Broadcast J to all N Gaussians
    J = mx.broadcast_to(J, batch_dims + (C, N, 2, 3))

    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    # means2d = [fx * X + cx, fy * Y + cy]
    means2d = (
        means[..., :2] * Ks[..., None, [0, 1], [0, 1]]
        + Ks[..., None, [0, 1], [2, 2]]
    )  # [..., C, N, 2]

    return means2d, cov2d
```

**Key properties**:
- The Jacobian is independent of `Z`, so depth does not affect the 2D covariance
- `means2d` only depends on `X` and `Y`, not `Z`
- The Jacobian is broadcast from `[..., C, 1, 2, 3]` to `[..., C, N, 2, 3]`

**torch-to-MLX note**: The upstream uses `.repeat([1]*len(batch_dims) + [1, N, 1, 1])` which maps to `mx.broadcast_to(...)` in MLX.

### 4.8 Function: `fully_fused_projection` (Main Entry Point)

**Purpose**: Orchestrate the full projection pipeline: world-to-cam, projection, regularization, conic computation, radius computation, and culling.

#### 4.8.1 Pipeline Stages

```
Stage 1: world_to_cam(means, covars, viewmats)
  Input:  means [..., N, 3], covars [..., N, 3, 3], viewmats [..., C, 4, 4]
  Output: means_c [..., C, N, 3], covars_c [..., C, N, 3, 3]

Stage 2: {persp,fisheye,ortho}_proj(means_c, covars_c, Ks, width, height)
  Input:  means_c, covars_c, Ks [..., C, 3, 3]
  Output: means2d [..., C, N, 2], covars2d [..., C, N, 2, 2]

Stage 3: Regularization
  det_orig = covars2d[0,0] * covars2d[1,1] - covars2d[0,1] * covars2d[1,0]
  covars2d += eps2d * I_2x2

Stage 4: Determinant and conics
  det = covars2d[0,0] * covars2d[1,1] - covars2d[0,1] * covars2d[1,0]
  det = clip(det, min=1e-10)
  conics = [covars2d[1,1]/det, -(covars2d[0,1]+covars2d[1,0])/2/det, covars2d[0,0]/det]

Stage 5: Compensation (optional)
  compensations = sqrt(clip(det_orig / det, min=0.0))

Stage 6: Radius computation
  radius_x = ceil(3.33 * sqrt(covars2d[0,0]))
  radius_y = ceil(3.33 * sqrt(covars2d[1,1]))

Stage 7: Frustum culling
  valid = (depths > near_plane) & (depths < far_plane)
  radius[~valid] = 0

Stage 8: Screen bounds culling
  inside = (means2d_x + radius_x > 0) & (means2d_x - radius_x < width) & ...
  radius[~inside] = 0

Stage 9: Cast to int32
  radii = radius.astype(mx.int32)
```

#### 4.8.2 The 3.33 Sigma Radius Factor

The radius is computed as `ceil(3.33 * sqrt(variance))`. This corresponds to approximately 3.33 standard deviations, which captures 99.9% of the Gaussian mass (3-sigma captures 99.7%). The ceiling ensures at least 1 pixel radius for any visible Gaussian.

#### 4.8.3 Conic Representation

The conic `[a, b, c]` represents the inverse of the 2D covariance matrix:

```
cov2d = [[sigma_xx, sigma_xy],
         [sigma_xy, sigma_yy]]

cov2d_inv = (1/det) * [[sigma_yy, -sigma_xy],
                        [-sigma_xy, sigma_xx]]

conics = [a, b, c] = [sigma_yy/det, -sigma_xy/det, sigma_xx/det]
```

So `cov2d_inv = [[a, b], [b, c]]`. The rasterizer (PRD-07) uses this to evaluate `exp(-0.5 * [dx, dy] @ [[a,b],[b,c]] @ [dx, dy]^T)` for each pixel offset `(dx, dy)`.

Note: The upstream computes `b = -(covars2d[0,1] + covars2d[1,0]) / 2 / det` rather than just `-covars2d[0,1] / det`. This averages the off-diagonal elements to enforce exact symmetry, even though the covariance should already be symmetric. This guards against floating-point asymmetry accumulated through matrix operations.

#### 4.8.4 Anti-Aliasing Compensation

When `eps2d > 0`, the regularization inflates the covariance, which would make Gaussians appear slightly larger. The compensation factor corrects for this:

```
compensation = sqrt(det_orig / det_padded)
```

Where `det_orig` is the determinant before adding `eps2d * I` and `det_padded` is the determinant after. This factor is in `[0, 1]` and is multiplied into the Gaussian weight during rasterization to maintain energy conservation.

For small Gaussians (whose covariance is dominated by the `eps2d` padding), the compensation approaches 0, effectively fading them out. This prevents aliasing from sub-pixel Gaussians.

#### 4.8.5 MLX Implementation

```python
def fully_fused_projection(
    means: mx.array,           # [..., N, 3]
    covars: mx.array,          # [..., N, 3, 3]
    viewmats: mx.array,        # [..., C, 4, 4]
    Ks: mx.array,              # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Fully fused 3D-to-2D Gaussian projection.

    Projects 3D Gaussians into 2D screen-space ellipses. This is the first
    stage of the rendering pipeline.

    Args:
        means: 3D Gaussian centers in world space. [..., N, 3].
        covars: 3D covariance matrices. [..., N, 3, 3].
        viewmats: World-to-camera 4x4 matrices. [..., C, 4, 4].
        Ks: Camera intrinsic matrices. [..., C, 3, 3].
        width: Image width in pixels.
        height: Image height in pixels.
        eps2d: Regularization added to 2D covariance diagonal.
        near_plane: Near clipping distance.
        far_plane: Far clipping distance.
        calc_compensations: Whether to compute anti-aliasing compensations.
        camera_model: One of "pinhole", "ortho", "fisheye".

    Returns:
        radii: Per-axis screen-space radii (int32). 0 = culled. [..., C, N, 2].
        means2d: 2D pixel positions. [..., C, N, 2].
        depths: Z-depth in camera frame. [..., C, N].
        conics: Inverse 2D covariance [a, b, c]. [..., C, N, 3].
        compensations: Anti-aliasing factor or None. [..., C, N].
    """
    # Stage 1: World to camera
    means_c, covars_c = world_to_cam(means, covars, viewmats)

    # Stage 2: Camera to screen (dispatch on camera model)
    if camera_model == "ortho":
        means2d, covars2d = ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = persp_proj(means_c, covars_c, Ks, width, height)
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")

    # Stage 3: Compute determinant before regularization
    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )

    # Add eps2d * I for numerical stability
    covars2d = covars2d + mx.eye(2, dtype=means.dtype) * eps2d

    # Stage 4: Determinant and conics
    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = mx.clip(det, a_min=1e-10)

    # Stage 5: Anti-aliasing compensation
    if calc_compensations:
        compensations = mx.sqrt(mx.clip(det_orig / det, a_min=0.0))
    else:
        compensations = None

    # Inverse covariance in packed form [a, b, c] where inv = [[a,b],[b,c]]
    conics = mx.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        axis=-1,
    )  # [..., C, N, 3]

    # Stage 6: Depths
    depths = means_c[..., 2]  # [..., C, N]

    # Stage 7: Radii from covariance diagonal
    radius_x = mx.ceil(3.33 * mx.sqrt(covars2d[..., 0, 0]))
    radius_y = mx.ceil(3.33 * mx.sqrt(covars2d[..., 1, 1]))
    radius = mx.stack([radius_x, radius_y], axis=-1)  # [..., C, N, 2]

    # Stage 8: Frustum culling
    valid = (depths > near_plane) & (depths < far_plane)
    # MLX does NOT support boolean masking (radius[~valid] = 0.0)
    # Use mx.where instead:
    radius = mx.where(
        mx.expand_dims(valid, axis=-1),
        radius,
        mx.zeros_like(radius),
    )

    # Stage 9: Screen bounds culling
    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius = mx.where(
        mx.expand_dims(inside, axis=-1),
        radius,
        mx.zeros_like(radius),
    )

    # Stage 10: Cast to int32
    radii = radius.astype(mx.int32)

    return radii, means2d, depths, conics, compensations
```

**Critical MLX differences from torch**:

| Pattern | torch | MLX |
|---------|-------|-----|
| Boolean masking | `radius[~valid] = 0.0` | `radius = mx.where(valid[..., None], radius, zeros)` |
| Identity matrix | `torch.eye(2, device=..., dtype=...)` | `mx.eye(2, dtype=...)` |
| Determinant clamp | `det.clamp(min=1e-10)` | `mx.clip(det, a_min=1e-10)` |
| Int cast | `radius.int()` | `radius.astype(mx.int32)` |
| Expand dims | `valid.unsqueeze(-1)` | `mx.expand_dims(valid, axis=-1)` |

### 4.9 Public API Wrapper

The public API supports two input modes:

1. **Covariance mode**: Pass `covars` directly (already computed, e.g., from PRD-03)
2. **Quaternion+scale mode**: Pass `quats` and `scales`, convert to covariance internally

```python
def projection(
    means: mx.array,                      # [..., N, 3]
    covars: Optional[mx.array] = None,    # [..., N, 3, 3] or None
    quats: Optional[mx.array] = None,     # [..., N, 4] or None
    scales: Optional[mx.array] = None,    # [..., N, 3] or None
    viewmats: mx.array = None,            # [..., C, 4, 4]
    Ks: mx.array = None,                  # [..., C, 3, 3]
    width: int = 0,
    height: int = 0,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Public API for 3D Gaussian projection.

    Supports two input modes:
    - Covariance mode: pass `covars` (and leave quats/scales as None)
    - Quaternion+scale mode: pass `quats` and `scales` (and leave covars as None)

    Args:
        means: 3D Gaussian positions in world space.
        covars: Pre-computed 3D covariance matrices (mutually exclusive with quats/scales).
        quats: Quaternion rotations [w,x,y,z] (mutually exclusive with covars).
        scales: Per-axis scales (mutually exclusive with covars).
        viewmats: World-to-camera matrices.
        Ks: Camera intrinsic matrices.
        width: Image width.
        height: Image height.
        eps2d: Regularization for 2D covariance.
        near_plane: Near clipping distance.
        far_plane: Far clipping distance.
        calc_compensations: Compute anti-aliasing compensations.
        camera_model: Camera model type.

    Returns:
        radii, means2d, depths, conics, compensations
    """
    if covars is None:
        assert quats is not None and scales is not None, (
            "Either covars or (quats, scales) must be provided"
        )
        # Convert quaternion+scale to covariance (from PRD-03)
        from .covariance import quat_scale_to_covar_preci
        covars, _ = quat_scale_to_covar_preci(
            quats, scales, compute_covar=True, compute_preci=False, triu=False
        )
    else:
        assert quats is None and scales is None, (
            "covars and (quats, scales) are mutually exclusive"
        )

    return fully_fused_projection(
        means, covars, viewmats, Ks, width, height,
        eps2d=eps2d, near_plane=near_plane, far_plane=far_plane,
        calc_compensations=calc_compensations, camera_model=camera_model,
    )
```

### 4.10 Custom VJP (Backward Pass)

#### 4.10.1 Why We Need a Custom VJP

The `fully_fused_projection` function has outputs that are NOT differentiable:

- `radii` (int32, discrete) -- no gradient
- Culling decisions (boolean masks) -- non-differentiable discontinuity

If we let MLX auto-diff through the entire function, it would:
1. Fail on `astype(mx.int32)` (not differentiable)
2. Produce zero gradients through `mx.where` at culling boundaries

The solution is to use `@mx.custom_function` to define a VJP that:
- Ignores gradients for `radii` and `compensations`
- Passes gradients through the differentiable core (means2d, depths, conics)
- Uses the pre-culling values for gradient computation (gradients flow through even for culled Gaussians -- the optimizer still sees them)

#### 4.10.2 Strategy: Auto-diff the Differentiable Core

Rather than manually deriving the backward pass (which the CUDA kernel does), we define an inner function containing only the differentiable operations and let MLX auto-diff through it.

```python
@mx.custom_function
def _fused_projection_fwd(means, covars, viewmats, Ks,
                          width, height, eps2d, camera_model_str):
    """Forward pass returning differentiable outputs + non-differentiable outputs."""
    # ... full forward computation ...
    return means2d, depths, conics  # differentiable outputs only


@_fused_projection_fwd.vjp
def _fused_projection_vjp(primals, cotangents, outputs):
    """VJP: compute gradients of means and covars from cotangents of means2d, depths, conics."""
    means, covars, viewmats, Ks, width, height, eps2d, camera_model_str = primals
    v_means2d, v_depths, v_conics = cotangents

    # Define inner differentiable function
    def differentiable_fwd(means_, covars_):
        means_c, covars_c = world_to_cam(means_, covars_, viewmats)
        if camera_model_str == "pinhole":
            means2d, covars2d = persp_proj(means_c, covars_c, Ks, width, height)
        elif camera_model_str == "fisheye":
            means2d, covars2d = fisheye_proj(means_c, covars_c, Ks, width, height)
        elif camera_model_str == "ortho":
            means2d, covars2d = ortho_proj(means_c, covars_c, Ks, width, height)

        # Regularize
        covars2d_reg = covars2d + mx.eye(2, dtype=means_.dtype) * eps2d
        det = covars2d_reg[..., 0, 0] * covars2d_reg[..., 1, 1] - \
              covars2d_reg[..., 0, 1] * covars2d_reg[..., 1, 0]
        det = mx.clip(det, a_min=1e-10)

        conics = mx.stack([
            covars2d_reg[..., 1, 1] / det,
            -(covars2d_reg[..., 0, 1] + covars2d_reg[..., 1, 0]) / 2.0 / det,
            covars2d_reg[..., 0, 0] / det,
        ], axis=-1)

        depths = means_c[..., 2]
        return means2d, depths, conics

    # Use mx.vjp to compute gradients
    (means2d, depths, conics), vjp_fn = mx.vjp(
        differentiable_fwd, [means, covars], [v_means2d, v_depths, v_conics]
    )
    v_means, v_covars = vjp_fn

    # No gradients for viewmats, Ks, width, height, eps2d, camera_model
    return (v_means, v_covars, None, None, None, None, None, None)
```

#### 4.10.3 Alternative: No Custom VJP (Simpler MVP)

For the MVP, an alternative is to split the function into:
1. A differentiable core that returns `(means2d, depths, conics)` -- let MLX auto-diff this
2. A non-differentiable post-processing step that computes `(radii, compensations)` using `mx.stop_gradient`

```python
def fully_fused_projection(...):
    # Differentiable core (MLX will auto-diff through this)
    means2d, depths, conics = _projection_core(means, covars, viewmats, Ks, ...)

    # Non-differentiable post-processing
    covars2d_diag_xx = mx.stop_gradient(covars2d[..., 0, 0])
    covars2d_diag_yy = mx.stop_gradient(covars2d[..., 1, 1])
    radius_x = mx.ceil(3.33 * mx.sqrt(covars2d_diag_xx))
    # ... culling ...
    radii = mx.stop_gradient(radius.astype(mx.int32))

    return radii, means2d, depths, conics, compensations
```

**Recommendation**: Start with the `mx.stop_gradient` approach for MVP simplicity. Add `@mx.custom_function` only if auto-diff performance is insufficient.

#### 4.10.4 Gradient Flow Summary

| Output | Gradient w.r.t. `means` | Gradient w.r.t. `covars` | Notes |
|--------|-------------------------|--------------------------|-------|
| `means2d` | Yes | No (only through conics) | Position gradient |
| `depths` | Yes | No | Depth ordering gradient |
| `conics` | Yes (through Jacobian) | Yes | Shape gradient |
| `radii` | No (int32) | No (int32) | Non-differentiable |
| `compensations` | Yes (optional) | Yes (optional) | Anti-aliasing |

---

## 5. Edge Cases and Numerical Considerations

### 5.1 Gaussian Behind the Camera (`z <= 0`)

When a Gaussian's camera-space Z-coordinate is at or behind the camera:
- **Perspective**: The Jacobian has `1/z` terms that blow up or become negative
- **Handling**: Near-plane culling sets `radii = 0`. The default `near_plane = 0.01` catches this.
- **Gradient**: Gradients still flow through depths even for culled Gaussians, pushing them forward.

### 5.2 Gaussian at Camera Origin (`z ≈ 0, x ≈ 0, y ≈ 0`)

- **Perspective**: Both `means2d` (via `1/z`) and the Jacobian (`fx/z`, `-fx*x/z^2`) become singular
- **Handling**: Near-plane culling. Additionally, the FOV clamping limits the Jacobian.
- **Fisheye**: The `eps = 1e-7` in `xy_len` prevents division by zero on the optical axis.

### 5.3 Degenerate Covariance (Zero or Near-Zero Covariance)

- **Problem**: A Gaussian with zero covariance produces a zero 2D covariance, making the conic undefined
- **Handling**: The `eps2d` regularization adds a minimum covariance of `eps2d * I_2x2 = [[0.3, 0], [0, 0.3]]` (in default config), ensuring the determinant is always positive.
- **Result**: The conic is always well-defined, and the Gaussian renders as at least a ~1 pixel dot.

### 5.4 Very Large Covariance

- **Problem**: Very large Gaussians produce very large radii, causing excessive tile intersection
- **Handling**: The radius is computed from the diagonal of the regularized covariance, and the screen-bounds culling removes Gaussians that are entirely off-screen. No explicit radius cap in MVP (upstream has `radius_clip` which we set to 0).

### 5.5 Non-Positive-Definite 2D Covariance

- **Problem**: Numerical errors in the Jacobian projection can produce a non-PD 2D covariance
- **Handling**: `det = clip(det, min=1e-10)` ensures the determinant is positive. The compensation factor `sqrt(det_orig / det)` will be close to 1 if `det_orig` is also positive, or clipped to 0 if `det_orig` is negative.

### 5.6 Off-Center Principal Point

- **Problem**: When `cx != width/2` or `cy != height/2`, the FOV clamping limits are asymmetric
- **Handling**: Separate `lim_x_pos` and `lim_x_neg` (and similarly for y) handle this correctly.

---

## 6. Torch-to-MLX Translation Reference

Complete mapping for all operations in this PRD:

| torch | MLX | Notes |
|-------|-----|-------|
| `torch.einsum("...cij,...nj->...cni", R, means)` | `mx.einsum("...cij,...nj->...cni", R, means)` | MLX supports einsum natively |
| `torch.unbind(x, dim=-1)` | `x[..., 0], x[..., 1], x[..., 2]` | Manual unbind |
| `torch.clamp(x, min=a, max=b)` | `mx.clip(x, a, b)` | Same semantics |
| `x.clamp(min=v)` | `mx.clip(x, a_min=v)` | One-sided clamp |
| `torch.zeros(shape, device=d, dtype=t)` | `mx.zeros(shape, dtype=t)` | No device in MLX |
| `torch.zeros_like(x)` | `mx.zeros_like(x)` | Same API |
| `torch.eye(2, device=d, dtype=t)` | `mx.eye(2, dtype=t)` | No device |
| `torch.stack([a,b,c], dim=-1)` | `mx.stack([a,b,c], axis=-1)` | `dim` -> `axis` |
| `x.reshape(shape)` | `x.reshape(shape)` | Same API |
| `J.transpose(-1, -2)` | `mx.swapaxes(J, -1, -2)` | Or fold into einsum |
| `x[~mask] = 0.0` (boolean indexing) | `mx.where(mask[..., None], x, mx.zeros_like(x))` | **Critical difference** |
| `radius.int()` | `radius.astype(mx.int32)` | Explicit dtype |
| `x.unsqueeze(-1)` | `mx.expand_dims(x, axis=-1)` | Different name |
| `torch.atan2(y, x)` | `mx.arctan2(y, x)` | Same semantics, different name |
| `torch.sqrt(x)` | `mx.sqrt(x)` | Same API |
| `torch.ceil(x)` | `mx.ceil(x)` | Same API |
| `x ** 2` | `x * x` | Prefer explicit multiply for clarity |
| `x.repeat([1]*n + [1, N, 1, 1])` | `mx.broadcast_to(x, target_shape)` | Broadcast instead of repeat |
| `assert_never(camera_model)` | `raise ValueError(...)` | No typing_extensions dependency |

---

## 7. Test Plan

### File: `tests/test_projection.py`

All tests use `float32` precision unless otherwise noted.

### 7.1 Test Fixtures

```python
import pytest
import mlx.core as mx
import numpy as np

@pytest.fixture
def simple_camera():
    """Single pinhole camera with standard intrinsics."""
    K = mx.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    viewmat = mx.eye(4)  # Camera at origin, looking down -Z
    return K[None], viewmat[None], 640, 480  # Add C dimension

@pytest.fixture
def simple_gaussian():
    """Single Gaussian at (0, 0, 5) with unit covariance."""
    means = mx.array([[0.0, 0.0, 5.0]])  # [N=1, 3]
    covars = mx.eye(3)[None]  # [N=1, 3, 3]
    return means, covars

@pytest.fixture
def multi_camera():
    """4 cameras at different positions."""
    viewmats = mx.zeros((4, 4, 4))
    for i in range(4):
        viewmats[i] = mx.eye(4)
        # Translate each camera
        viewmats = viewmats.at[i, 0, 3].add(float(i) * 0.5)
    Ks = mx.broadcast_to(
        mx.array([[[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]]),
        (4, 3, 3),
    )
    return Ks, viewmats, 640, 480
```

### 7.2 World-to-Camera Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_world_to_cam_identity` | `viewmat = I_4x4` | `means_c == means`, `covars_c == covars` |
| `test_world_to_cam_translation` | `viewmat` with only translation `[1, 2, 3]` | `means_c = means + [1, 2, 3]` |
| `test_world_to_cam_rotation_90z` | 90-degree rotation about Z-axis | `(x, y, z) -> (-y, x, z)` |
| `test_world_to_cam_rotation_90x` | 90-degree rotation about X-axis | `(x, y, z) -> (x, -z, y)` |
| `test_world_to_cam_covar_rotation` | Verify `covars_c = R @ covars @ R^T` element-wise | Manual computation match |
| `test_world_to_cam_batch` | C=4 cameras, N=10 Gaussians, batch dim B=2 | Shapes `[B, C, N, 3]` and `[B, C, N, 3, 3]` correct |

**Test vectors for `test_world_to_cam_rotation_90z`**:

```python
# 90-degree rotation about Z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
R = mx.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=mx.float32)
viewmat = mx.eye(4)
viewmat = viewmat.at[:3, :3].set(R)  # Note: need to handle MLX immutability

mean = mx.array([[1.0, 0.0, 5.0]])
# Expected: R @ [1, 0, 5]^T = [0, 1, 5]
```

### 7.3 Perspective Projection Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_persp_proj_on_axis` | Gaussian at `(0, 0, 5)` | `means2d = (cx, cy) = (320, 240)` |
| `test_persp_proj_off_axis` | Gaussian at `(1, 0, 5)` | `means2d = (fx/5 + cx, cy) = (420, 240)` |
| `test_persp_proj_off_axis_y` | Gaussian at `(0, 2, 5)` | `means2d = (cx, fy*2/5 + cy) = (320, 440)` |
| `test_persp_proj_depth_scaling` | Gaussian at z=5 vs z=10 with same covariance | cov2d at z=10 is ~4x smaller |
| `test_persp_proj_cov2d_symmetric` | Random Gaussians | `cov2d[0,1] == cov2d[1,0]` |
| `test_persp_proj_cov2d_psd` | Random Gaussians | Both eigenvalues >= 0 |
| `test_persp_proj_fov_clamping` | Gaussian at extreme angle | Jacobian is clamped, no NaN |

**Test vector for `test_persp_proj_on_axis`**:

```python
K = mx.array([[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]])
means_c = mx.array([[[0.0, 0.0, 5.0]]])  # [C=1, N=1, 3]
covars_c = mx.eye(3).reshape(1, 1, 3, 3)  # [C=1, N=1, 3, 3]

means2d, cov2d = persp_proj(means_c, covars_c, K, 640, 480)

# means2d = K[:2,:3] @ [0, 0, 5]^T / 5 = [500*0/5 + 320, 500*0/5 + 240] = [320, 240]
assert mx.allclose(means2d, mx.array([[[320.0, 240.0]]]), atol=1e-5)

# Jacobian at (0, 0, 5):
# J = [[500/5, 0, -500*0/25], [0, 500/5, -500*0/25]]
# J = [[100, 0, 0], [0, 100, 0]]
# cov2d = J @ I @ J^T = [[10000, 0], [0, 10000]]
assert mx.allclose(cov2d[0, 0, 0, 0], mx.array(10000.0), atol=1e-2)
```

**Test vector for `test_persp_proj_depth_scaling`**:

```python
# For a Gaussian on-axis, J = [[fx/z, 0, 0], [0, fy/z, 0]]
# cov2d = [[fx^2/z^2, 0], [0, fy^2/z^2]] (for identity covariance)
# At z=5: cov2d = [[10000, 0], [0, 10000]]
# At z=10: cov2d = [[2500, 0], [0, 2500]]
# Ratio: 10000/2500 = 4 = (z2/z1)^2
```

### 7.4 Fisheye Projection Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_fisheye_on_axis` | Gaussian at `(0, 0, 5)` | `means2d ≈ (cx, cy)` (modulo eps) |
| `test_fisheye_off_axis` | Gaussian at `(1, 0, 1)` (45 degrees) | `means2d_x = fx * pi/4 + cx` approximately |
| `test_fisheye_vs_persp_small_angle` | Gaussian at small angle (< 10 degrees) | Fisheye and perspective results within 1% |
| `test_fisheye_wide_angle` | Gaussian at 80 degrees | Does not produce NaN/Inf |

**Test vector for `test_fisheye_off_axis`**:

```python
# At (1, 0, 1): theta = atan2(1, 1) = pi/4
# xy_len = 1
# means2d_x = fx * 1 * (pi/4) / 1 + cx = 500 * 0.7854 + 320 = 712.7
# means2d_y = fy * 0 * theta / 1 + cy = 240
```

### 7.5 Orthographic Projection Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_ortho_no_perspective` | Same Gaussian at z=5 and z=50 | Same means2d and cov2d |
| `test_ortho_scaling` | Gaussian at `(1, 2, 5)` | `means2d = (fx + cx, 2*fy + cy)` |
| `test_ortho_cov2d_constant` | Same covariance at different depths | Identical cov2d |

### 7.6 Fully Fused Projection Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_fused_proj_single` | 1 Gaussian, 1 camera | All outputs have correct shapes |
| `test_fused_proj_multi` | N=100 Gaussians, C=4 cameras | Shape `[C, N, ...]` |
| `test_fused_proj_near_plane` | Gaussian at z=-1 (behind camera) | `radii == [0, 0]` |
| `test_fused_proj_far_plane` | Gaussian at z=1e12 (beyond far plane) | `radii == [0, 0]` |
| `test_fused_proj_screen_left` | Gaussian projecting to x=-1000 | `radii == [0, 0]` |
| `test_fused_proj_screen_right` | Gaussian projecting to x=width+1000 | `radii == [0, 0]` |
| `test_fused_proj_screen_edge` | Gaussian at screen edge with large radius | `radii != 0` (visible) |
| `test_fused_proj_conics_inverse` | Verify `[[a,b],[b,c]]` is inverse of `cov2d` | `cov2d @ cov2d_inv ≈ I` |
| `test_fused_proj_conics_symmetric` | Verify `conics[1]` is shared off-diagonal | `cov2d_inv[0,1] == cov2d_inv[1,0]` |
| `test_fused_proj_compensations_range` | With `calc_compensations=True` | Values in `[0, 1]` |
| `test_fused_proj_compensations_identity` | Large Gaussian (eps2d negligible) | `compensation ≈ 1.0` |
| `test_fused_proj_compensations_small` | Sub-pixel Gaussian | `compensation ≈ 0.0` |
| `test_fused_proj_eps2d_prevents_singular` | Zero covariance Gaussian | `det > 0`, no NaN |
| `test_fused_proj_camera_model_pinhole` | Pinhole model produces perspective results | Match `persp_proj` |
| `test_fused_proj_camera_model_fisheye` | Fisheye model produces fisheye results | Match `fisheye_proj` |
| `test_fused_proj_camera_model_ortho` | Ortho model produces ortho results | Match `ortho_proj` |
| `test_fused_proj_batch_dims` | Input with batch dim `[B, N, 3]` | Output `[B, C, N, ...]` |
| `test_fused_proj_radius_values` | Known covariance -> known radius | `radius = ceil(3.33 * sqrt(diag))` |

**Test vector for `test_fused_proj_conics_inverse`**:

```python
# For a Gaussian with identity covariance at (0, 0, 5) with fx=fy=500, eps2d=0.3:
# cov2d = [[10000, 0], [0, 10000]]  (from perspective Jacobian)
# cov2d + eps2d*I = [[10000.3, 0], [0, 10000.3]]
# det = 10000.3^2 = 100006000.09
# conics = [10000.3/det, 0, 10000.3/det]
# => [a, b, c] ≈ [1/10000.3, 0, 1/10000.3]
# Verify: [[a, b], [b, c]] @ [[10000.3, 0], [0, 10000.3]] ≈ I
```

### 7.7 VJP / Gradient Tests

| Test | Description | Tolerance |
|------|-------------|-----------|
| `test_vjp_means_grad` | `d(means2d) / d(means)` matches finite differences | atol=1e-3 |
| `test_vjp_covars_grad` | `d(conics) / d(covars)` matches finite differences | atol=1e-3 |
| `test_vjp_depths_grad` | `d(depths) / d(means)` matches finite differences | atol=1e-3 |
| `test_vjp_combined` | Loss = sum(means2d + conics + depths), check all grads | atol=1e-3 |
| `test_vjp_batch` | Gradients with B=2, C=2, N=50 | atol=1e-3 |
| `test_vjp_fisheye` | Gradients through fisheye projection | atol=1e-3 |
| `test_vjp_ortho` | Gradients through ortho projection | atol=1e-3 |

**Finite difference gradient check pattern**:

```python
def test_vjp_means_grad():
    means = mx.random.normal((10, 3))
    covars = mx.broadcast_to(mx.eye(3), (10, 3, 3))
    viewmats = mx.eye(4)[None]  # [C=1, 4, 4]
    Ks = mx.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]])

    def loss_fn(m):
        _, means2d, depths, conics, _ = fully_fused_projection(
            m, covars, viewmats, Ks, 640, 480
        )
        return mx.sum(means2d) + mx.sum(depths) + mx.sum(conics)

    # Analytical gradient
    grad_fn = mx.grad(loss_fn)
    grad_analytical = grad_fn(means)

    # Finite difference gradient
    eps = 1e-4
    grad_fd = mx.zeros_like(means)
    for i in range(means.shape[0]):
        for j in range(3):
            means_plus = means.at[i, j].add(eps)
            means_minus = means.at[i, j].add(-eps)
            loss_plus = loss_fn(means_plus)
            loss_minus = loss_fn(means_minus)
            grad_fd = grad_fd.at[i, j].set((loss_plus - loss_minus) / (2 * eps))

    assert mx.allclose(grad_analytical, grad_fd, atol=1e-3)
```

### 7.8 Cross-Framework Tests

These tests require torch and run with `@pytest.mark.requires_torch`.

| Test | Description | Tolerance |
|------|-------------|-----------|
| `test_cross_world_to_cam` | 500 random Gaussians, 4 cameras | atol=1e-5 |
| `test_cross_persp_proj` | 500 random camera-space Gaussians | atol=1e-4 |
| `test_cross_fisheye_proj` | 500 random Gaussians | atol=1e-4 |
| `test_cross_ortho_proj` | 500 random Gaussians | atol=1e-4 |
| `test_cross_fused_projection` | Full pipeline, all outputs | atol=1e-4 |
| `test_cross_vjp` | Gradient comparison (MLX vs torch autograd) | atol=1e-3 |

**Cross-framework comparison pattern**:

```python
@pytest.mark.requires_torch
def test_cross_fused_projection():
    import torch
    from gsplat.cuda._torch_impl import _fully_fused_projection

    N, C = 500, 2
    np.random.seed(42)
    means_np = np.random.randn(N, 3).astype(np.float32)
    means_np[:, 2] = np.abs(means_np[:, 2]) + 2.0  # Ensure positive Z
    covars_np = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0) * 0.1

    viewmats_np = np.zeros((C, 4, 4), dtype=np.float32)
    viewmats_np[:] = np.eye(4)
    viewmats_np[1, 0, 3] = 0.5  # Second camera offset

    Ks_np = np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]] * C, dtype=np.float32)

    # MLX computation
    radii_mlx, means2d_mlx, depths_mlx, conics_mlx, _ = fully_fused_projection(
        mx.array(means_np), mx.array(covars_np),
        mx.array(viewmats_np), mx.array(Ks_np), 640, 480,
    )

    # Torch computation
    radii_pt, means2d_pt, depths_pt, conics_pt, _ = _fully_fused_projection(
        torch.tensor(means_np), torch.tensor(covars_np),
        torch.tensor(viewmats_np), torch.tensor(Ks_np), 640, 480,
    )

    # Compare
    np.testing.assert_allclose(
        np.array(means2d_mlx), means2d_pt.numpy(), atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(depths_mlx), depths_pt.numpy(), atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(conics_mlx), conics_pt.numpy(), atol=1e-4
    )
    np.testing.assert_equal(
        np.array(radii_mlx), radii_pt.numpy()
    )
```

### 7.9 Input Validation Tests

| Test | Description |
|------|-------------|
| `test_quat_scale_input` | Using quats+scales via public API matches covariance input |
| `test_mutual_exclusive_inputs` | Passing both covars and quats raises AssertionError |
| `test_neither_input` | Passing neither covars nor quats raises AssertionError |
| `test_invalid_camera_model` | Passing `camera_model="ftheta"` raises ValueError |

### 7.10 Tolerance Rationale

| Check | Tolerance | Reason |
|-------|-----------|--------|
| Forward (means2d) | `atol=1e-4` | Accumulation through einsum and division |
| Forward (conics) | `atol=1e-4` | Matrix inversion via determinant |
| Forward (depths) | `atol=1e-5` | Simple linear transform |
| Forward (radii) | exact match | Integer output, ceil is deterministic |
| VJP | `atol=1e-3` | Chain rule through Jacobian, clip, and stack |
| Finite diff | `atol=1e-3` | Finite-difference approximation error |
| Cross-framework | `atol=1e-4` | float32 differences between MLX and torch backends |

---

## 8. Dependencies

### 8.1 Upstream Dependencies

| PRD | What We Need | Status |
|-----|-------------|--------|
| PRD-01 | Dev environment, package structure, conftest.py | Required |
| PRD-02 | `_quat_to_rotmat` (used indirectly via PRD-03) | Required |
| PRD-03 | `quat_scale_to_covar_preci` (for quaternion+scale input mode) | Required |

### 8.2 Downstream Dependents

| PRD | What They Need From Us |
|-----|----------------------|
| PRD-06 (Tile Intersection) | `means2d`, `radii`, `depths` |
| PRD-07 (Rasterization) | `means2d`, `conics`, `compensations` |
| PRD-09 (Rendering API) | `fully_fused_projection` / `projection` public API |

---

## 9. Acceptance Criteria

- [ ] `world_to_cam` matches torch `_world_to_cam` within `atol=1e-5` for 500+ random Gaussians
- [ ] `persp_proj` matches torch `_persp_proj` within `atol=1e-4` for 500+ random Gaussians
- [ ] `fisheye_proj` matches torch `_fisheye_proj` within `atol=1e-4` for 500+ random Gaussians
- [ ] `ortho_proj` matches torch `_ortho_proj` within `atol=1e-4` for 500+ random Gaussians
- [ ] `fully_fused_projection` matches torch `_fully_fused_projection` within `atol=1e-4` for all outputs
- [ ] All 3 camera models (pinhole, fisheye, ortho) produce correct projections
- [ ] Frustum culling correctly zeros radii for Gaussians outside `[near_plane, far_plane]`
- [ ] Screen-bounds culling correctly zeros radii for fully off-screen Gaussians
- [ ] Conics are valid inverse covariance representations: `cov2d @ [[a,b],[b,c]] approx I`
- [ ] Anti-aliasing compensations are in `[0, 1]` range when computed
- [ ] eps2d prevents singular covariances (zero covariance input does not produce NaN)
- [ ] Backward pass gradients match finite differences within `atol=1e-3`
- [ ] Gradients flow correctly through all 3 camera models
- [ ] Supports both covariance and quaternion+scale input modes via public API
- [ ] Supports arbitrary batch dimensions `[..., N, 3]` and multi-camera `[..., C, 4, 4]`
- [ ] No NaN/Inf for edge cases (behind camera, at camera origin, degenerate covariance)
- [ ] All tests pass with `pytest tests/test_projection.py -v`
- [ ] Cross-framework tests pass with `pytest tests/test_projection.py -v -m requires_torch`

---

## 10. Implementation Order

1. **`cameras.py`**: Create `CameraModel` type alias (5 min)
2. **`world_to_cam`**: Implement and test (1 hour)
3. **`persp_proj`**: Implement with Jacobian, test thoroughly (2 hours)
4. **`ortho_proj`**: Implement (simpler than perspective) and test (30 min)
5. **`fisheye_proj`**: Implement with complex Jacobian and test (1 hour)
6. **`fully_fused_projection`**: Assemble pipeline stages and test (2 hours)
7. **Public API wrapper**: `projection()` with quat/scale support (30 min)
8. **VJP / backward pass**: Implement and validate with finite differences (2 hours)
9. **Cross-framework tests**: Compare against torch reference (1 hour)
10. **Edge case hardening**: NaN/Inf guards, degenerate inputs (1 hour)

**Estimated total**: ~11 hours

---

## 11. Open Questions

1. **`mx.einsum` performance**: MLX's einsum may be slower than fused operations for the Jacobian computation. If profiling shows this is a bottleneck, consider manual matrix multiplication via `mx.matmul`.

2. **`mx.custom_function` vs auto-diff**: The MVP uses `mx.stop_gradient` for non-differentiable outputs. Should we switch to `@mx.custom_function` for the production implementation? The tradeoff is code complexity vs. efficiency (custom VJP can cache intermediates).

3. **`radius_clip` parameter**: Upstream has a `radius_clip` parameter that filters Gaussians with any radius below the clip value. This is set to 0.0 by default. Should we expose it in the MVP API?

4. **Opacity-based radius**: Upstream's CUDA kernel can adjust radius based on opacity. This is not in the `_torch_impl.py` reference. Defer to a future PRD?
