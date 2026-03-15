# PRD-12: 2D Gaussian Splatting (2DGS) -- Surfel-Based Rendering

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-12 |
| **Title** | 2D Gaussian Splatting (2DGS) -- Surfel-Based Rendering |
| **Status** | DRAFT |
| **Priority** | P1 -- High |
| **Estimated Effort** | 8--12 hours |
| **Dependencies** | PRD-01 (dev env), PRD-02 (math utils), PRD-06 (intersection/tiling), PRD-09 (rendering API patterns) |
| **Blocks** | PRD-13 (optional: 2DGS training loop) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Port the 2D Gaussian Splatting (2DGS) pipeline from the upstream gsplat `_torch_impl_2dgs.py` reference implementation to Apple MLX. 2DGS represents each Gaussian as a **2D disk (surfel) embedded in 3D space** rather than a 3D ellipsoid, yielding superior surface reconstruction and geometry accuracy compared to standard 3DGS.

After this PRD is implemented, users will be able to:

1. Project 2D Gaussian surfels to screen space via ray-surfel intersection transforms
2. Rasterize surfels with proper normal accumulation
3. Obtain rendered color, alpha, and normal maps from a single API call
4. Train 2DGS models on Apple Silicon via gradient backpropagation through the entire pipeline

---

## 2. Context & Motivation

### 2.1 What is 2DGS?

2D Gaussian Splatting (from "2D Gaussian Splatting for Geometrically Accurate Radiance Fields", Huang et al. 2024) replaces the 3D ellipsoidal Gaussians of standard 3DGS with **2D disk primitives (surfels)**. Each surfel is a flat, oriented disk defined by:

- A center point in 3D (same as 3DGS)
- A quaternion orientation defining the disk's tangent plane
- Two tangential scales (the disk's extent) plus a third scale (the normal direction, which should be near-zero for a true surfel)
- An opacity and color (or SH coefficients)

### 2.2 Key Differences from 3DGS

| Aspect | 3DGS | 2DGS |
|--------|------|------|
| **Primitive** | 3D ellipsoid | 2D disk (surfel) in 3D |
| **Parameterization** | Covariance matrix (6 DOF) | Quaternion + 2 tangential scales |
| **Projection** | EWA splatting (J * Sigma * J^T) | Ray-surfel intersection transform |
| **Screen representation** | 2D conic (3 params) | 3x3 ray_transforms matrix (9 params) |
| **Per-pixel computation** | Mahalanobis distance via conic | Cross-product ray-surfel intersection |
| **Sigma computation** | Single 2D Gaussian sigma | min(3D sigma, 2D sigma) for robustness |
| **Normal output** | Not natively supported | First-class normal rendering |
| **Geometry quality** | Blobby surfaces | Sharp, accurate surfaces |

### 2.3 Why min(sigma_3d, sigma_2d)?

The 2DGS paper introduces a dual-sigma strategy to handle numerical instability at grazing angles:

- **sigma_3d**: Computed from the true ray-surfel intersection point (u, v coordinates on the surfel). Accurate when the surfel is viewed head-on but can explode at grazing angles where the intersection becomes ill-conditioned.
- **sigma_2d**: Computed from the 2D screen-space distance between the pixel and the surfel center. Acts as a fallback that gracefully degrades at grazing angles.
- **min(sigma_3d, sigma_2d)**: Takes the smaller value, ensuring the Gaussian falloff is always well-behaved. At head-on views, sigma_3d dominates (accurate geometry). At grazing angles, sigma_2d acts as a safety net.

### 2.4 Source Reference

- **Primary**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl_2dgs.py` (335 lines)
- **High-level API**: `repositories/gsplat-upstream/gsplat/rendering.py` lines 1876--2234 (`rasterization_2dgs`)
- **Math dependency**: `repositories/gsplat-upstream/gsplat/cuda/_math.py` (`_quat_scale_to_matrix`)
- **Constants**: `MAX_ALPHA = 0.99`

---

## 3. Scope

### 3.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `fully_fused_projection_2dgs()` | Project 2D Gaussian surfels to screen space; output ray_transforms and normals |
| `accumulate_2dgs()` | Alpha compositing with ray-surfel intersection and normal accumulation |
| `rasterize_to_pixels_2dgs()` | Tile-based iterative rasterization for 2DGS |
| `rasterization_2dgs()` | High-level API orchestrating the full 2DGS pipeline |
| `quat_scale_to_matrix()` | Helper: quaternion + scale to RS matrix (if not already in PRD-02) |
| VJP for projection | Custom backward pass for `fully_fused_projection_2dgs` |
| Test suite | Comprehensive tests covering projection, normals, accumulation, rendering, gradients |

### 3.2 Out of Scope

| Item | Reason |
|------|--------|
| Packed mode | Optimization, not needed for correctness |
| Distortion loss | Advanced regularization, can be added later |
| Median depth rendering | Advanced feature, can be added later |
| Surface normals from depth | Post-processing utility, not core 2DGS |
| Sparse gradients | Memory optimization for large scenes |
| Densification strategy for 2DGS | Belongs in a separate PRD |
| Camera distortion models | Not supported in 2DGS reference impl |

---

## 4. Mathematical Foundation

This section provides the complete mathematical derivation of the 2DGS pipeline. Every equation maps directly to code.

### 4.1 Surfel Parameterization

A 2D Gaussian surfel is defined by:

- **Center**: `p_w` in world space, shape `[3]`
- **Quaternion**: `q = (w, x, y, z)`, shape `[4]`, defines the orientation of the surfel's local frame
- **Scales**: `s = (s_u, s_v, s_n)`, shape `[3]`, where `s_u, s_v` are tangential extents and `s_n` is the normal-direction scale

The rotation-scale matrix is:

```
RS = R(q) * diag(s_u, s_v, s_n)
```

where `R(q)` is the 3x3 rotation matrix from quaternion `q`. The columns of `RS` are:

- Column 0: `t_u = s_u * r_0` -- scaled tangent vector u
- Column 1: `t_v = s_v * r_1` -- scaled tangent vector v
- Column 2: `n_s = s_n * r_2` -- scaled normal vector

For a perfect surfel, `s_n -> 0`, making the Gaussian infinitely thin in the normal direction.

### 4.2 World-to-Camera Transformation

Given view matrix `V = [R_cw | t_cw]` (world-to-camera):

```
p_c = R_cw @ p_w + t_cw                    # Mean in camera space [3]
RS_c = R_cw @ RS_w                          # RS matrix in camera space [3, 3]
```

### 4.3 Normal Computation

The surfel normal in camera space is the third column of `RS_c`:

```
n_c = RS_c[:, 2]                            # Camera-space normal [3]
```

The normal must point toward the camera (i.e., toward the origin in camera space). We check this with the dot product:

```
cos_angle = dot(-n_c, p_c)                  # equivalently: -sum(n_c * p_c)
if cos_angle > 0:
    n_c = +n_c                              # already facing camera
else:
    n_c = -n_c                              # flip toward camera
```

Implementation detail: the sign flip uses `multiplier = sign(cos_angle)` applied to the normal vector, where `sign(x) = 1 if x > 0 else -1`.

**Why the dot product works**: In camera space, the camera is at the origin. The vector from the surfel center to the camera is `-p_c`. If `dot(n_c, -p_c) > 0`, the normal already points toward the camera. Otherwise, we flip it.

**Upstream reference** (lines 64-68 of `_torch_impl_2dgs.py`):

```python
cos = -normals.reshape((-1, 1, 3)) @ means_c.reshape((-1, 3, 1))
cos = cos.reshape(batch_dims + (C, N, 1))
multiplier = torch.where(cos > 0, torch.tensor(1.0), torch.tensor(-1.0))
normals *= multiplier
```

### 4.4 Ray Transform Matrix Construction

The ray transform matrix maps pixel coordinates to the surfel's local UV frame. It is constructed from the first two columns of `RS_c` (tangent vectors) and the camera-space mean:

```
T_c = [RS_c[:, 0], RS_c[:, 1], p_c]        # [3, 3] -- columns are t_u_c, t_v_c, p_c
```

Note: the normal column (`RS_c[:, 2]`) is **not** included. The transform only needs the tangent plane and the center.

Project to screen space using the intrinsics matrix `K`:

```
T_s = K @ T_c                               # [3, 3]
M = T_s^T                                   # [3, 3] -- transposed for AABB computation
```

In the upstream code, `M = T_s^T` is used internally for AABB computation, then transposed back to `T_s` before returning as `ray_transforms`. The internal AABB computation uses the transposed form so that rows (not columns) correspond to the projected axes.

**Upstream reference** (lines 71-77):

```python
T_cl = torch.cat([RS_cl[..., :2], means_c[..., None]], dim=-1)  # [..., C, N, 3, 3]
T_sl = torch.einsum("...cij,...cnjk->...cnik", Ks[..., :3, :3], T_cl)  # [..., C, N, 3, 3]
M = torch.transpose(T_sl, -1, -2)  # [..., C, N, 3, 3]
```

### 4.5 AABB (Axis-Aligned Bounding Box) Computation

The AABB determines the screen-space extent of each surfel for tile-based rasterization. Using `M = T_s^T`:

```
test = [1.0, 1.0, -1.0]
d = sum(M[2, :] * M[2, :] * test)           # scalar discriminant
valid = |d| > eps

f = test / d                                 # [3], safe division (0 if invalid)

means2d[k] = sum_j(M[k, j] * M[2, j] * f[j])   for k in {0, 1}   # screen center [2]

extents[k] = sqrt(clamp(means2d[k]^2 - sum_j(M[k, j]^2 * f[j]), min=1e-4))   # [2]
```

The `test = [1, 1, -1]` vector encodes the signature of the homogeneous quadratic form used to compute the AABB. The discriminant `d` indicates whether the surfel projects to a valid bounded region.

The bounding radius is:

```
radius = ceil(3.33 * extents)               # [2] -- per-axis radius in pixels
```

The factor 3.33 is chosen to encompass approximately 3.3 standard deviations of the Gaussian, ensuring negligible contribution outside the AABB.

**Upstream reference** (lines 79-106):

```python
test = torch.tensor([1.0, 1.0, -1.0], device=means.device)
d = (M[..., 2] * M[..., 2] * test).sum(dim=-1, keepdim=True)
valid = torch.abs(d) > eps
f = torch.where(valid, test / d, torch.zeros_like(test)).unsqueeze(-1)
means2d = (M[..., :2] * M[..., 2:3] * f).sum(dim=-2)
extents = torch.sqrt((means2d**2 - (M[..., :2] * M[..., :2] * f).sum(dim=-2)).clamp_min(1e-4))
depths = means_c[..., 2]
radius = torch.ceil(3.33 * extents)
```

### 4.6 Visibility Filtering

A surfel is visible if all three conditions hold:

1. `valid` is True (discriminant is non-degenerate)
2. `near_plane < depth < far_plane`
3. The AABB overlaps the image bounds:
   - `means2d_x + radius_x > 0` and `means2d_x - radius_x < width`
   - `means2d_y + radius_y > 0` and `means2d_y - radius_y < height`

Surfels failing any condition get `radii = 0`, which causes them to be skipped by tile intersection.

### 4.7 Ray-Surfel Intersection (Per-Pixel)

This is the core geometric computation that distinguishes 2DGS from 3DGS. For each pixel `(px, py)` and each overlapping surfel with ray_transforms matrix `M_t` (the returned `ray_transforms = T_s`, NOT the transposed M used for AABB):

**Step 1: Compute homogeneous intersection lines**

```
h_u = -M_t[0, :3] + M_t[2, :3] * px        # [3]
h_v = -M_t[1, :3] + M_t[2, :3] * py        # [3]
```

Here `px = pixel_x + 0.5` and `py = pixel_y + 0.5` (pixel center coordinates).

Geometrically, `h_u` encodes the constraint that the intersection point's projected x-coordinate equals `px`, and `h_v` encodes the y-coordinate constraint. Each is a line in the surfel's projected homogeneous space.

**Step 2: Cross product to find intersection**

```
intersection = cross(h_u, h_v)              # [3]
```

The cross product of two lines in projective 2D space (represented as 3-vectors) yields their intersection point (also as a homogeneous 3-vector). This is a standard result from projective geometry.

**Step 3: Dehomogenize to get surfel UV coordinates**

```
u = intersection[0] / intersection[2]
v = intersection[1] / intersection[2]
```

`(u, v)` are the coordinates of the ray-surfel intersection in the surfel's local tangent frame. A point at `(0, 0)` is at the surfel center; `u^2 + v^2 = 1` is at the surfel boundary (at scale 1).

**Step 4: Compute 3D sigma (geometric)**

```
sigma_3d = u^2 + v^2
```

This is the squared distance from the surfel center in the surfel's local frame, normalized by the surfel's tangential scales. It represents the true geometric Gaussian falloff based on where the viewing ray actually hits the surfel.

**Step 5: Compute 2D sigma (screen-space fallback)**

```
dx = px - means2d_x
dy = py - means2d_y
sigma_2d = 2 * (dx^2 + dy^2)
```

This is a screen-space distance metric that acts as a robust fallback. The factor of 2 normalizes it relative to sigma_3d so the two are comparable in magnitude.

**Step 6: Select minimum sigma**

```
sigma = 0.5 * min(sigma_3d, sigma_2d)
```

The 0.5 factor converts from squared distance to the Gaussian exponent convention. The minimum selection ensures:

- **Head-on viewing**: `sigma_3d` is well-conditioned and typically smaller, so the accurate geometric value is used
- **Grazing angles**: `sigma_3d` can become very large or numerically unstable as the ray nearly parallels the surfel; `sigma_2d` provides a smooth, bounded fallback
- **Transition**: The `min` creates a smooth (C0 but not C1) transition between the two regimes

**Upstream reference** (lines 170-184 of `_torch_impl_2dgs.py`):

```python
pixel_ids_x = pixel_ids % image_width + 0.5
pixel_ids_y = pixel_ids // image_width + 0.5
pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1)
deltas = pixel_coords - means2d[image_ids, gaussian_ids]

M = ray_transforms[image_ids, gaussian_ids]  # [M, 3, 3]
h_u = -M[..., 0, :3] + M[..., 2, :3] * pixel_ids_x[..., None]
h_v = -M[..., 1, :3] + M[..., 2, :3] * pixel_ids_y[..., None]
tmp = torch.cross(h_u, h_v, dim=-1)
us = tmp[..., 0] / tmp[..., 2]
vs = tmp[..., 1] / tmp[..., 2]
sigmas_3d = us**2 + vs**2
sigmas_2d = 2 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)
sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)
```

### 4.8 Alpha Compositing with Normal Accumulation

Once `sigma` is computed for each pixel-surfel pair:

```
alpha = clamp(opacity * exp(-sigma), max=MAX_ALPHA)     # MAX_ALPHA = 0.99
```

The front-to-back compositing is identical to 3DGS, but additionally accumulates surface normals:

```
# For each pixel, processing surfels front-to-back:
T = 1.0                                     # transmittance
rendered_color = [0, 0, 0]
rendered_normal = [0, 0, 0]
rendered_alpha = 0

for each surfel i (front to back):
    weight = T * alpha_i
    rendered_color += weight * color_i
    rendered_normal += weight * normal_i
    rendered_alpha += weight
    T *= (1 - alpha_i)
    if T < threshold:
        break                                # early termination
```

The accumulated normal map can be used for:
- Normal consistency loss during training
- Surface reconstruction and meshing
- Relighting applications

### 4.9 Gradient Flow Through min(sigma_3d, sigma_2d)

The `min` operation creates a piecewise-linear function with gradients:

```
if sigma_3d < sigma_2d:
    # sigma = 0.5 * sigma_3d = 0.5 * (u^2 + v^2)
    d(sigma)/d(u) = 0.5 * 2u = u
    d(sigma)/d(v) = 0.5 * 2v = v
    d(sigma)/d(dx) = 0                      # no gradient to 2D path
    d(sigma)/d(dy) = 0
else:
    # sigma = 0.5 * sigma_2d = 0.5 * 2 * (dx^2 + dy^2)
    d(sigma)/d(u) = 0                        # no gradient to 3D path
    d(sigma)/d(v) = 0
    d(sigma)/d(dx) = 0.5 * 2 * 2 * dx = 2 * dx
    d(sigma)/d(dy) = 0.5 * 2 * 2 * dy = 2 * dy
```

In MLX, `mx.minimum` supports automatic differentiation. The gradient flows through whichever input is smaller. At the boundary (`sigma_3d == sigma_2d`), the subgradient from either branch is valid.

**Impact on training**: When a surfel is viewed at a grazing angle and `sigma_2d` is selected, gradients flow through the screen-space path (means2d) rather than the geometric intersection path (u, v from cross product). This prevents gradient explosion from the ill-conditioned cross-product division and allows the optimizer to still adjust the surfel's position via `means2d`.

When `sigma_3d` is selected (head-on viewing), gradients flow through:

```
sigma -> (u, v) -> intersection -> cross(h_u, h_v) -> M (ray_transforms)
  -> T_sl -> T_cl -> {RS_cl, means_c} -> {quats, scales, means}
```

This full chain provides geometrically accurate gradients for training the surfel parameters.

### 4.10 Gradient Through Cross-Product Intersection

The cross product `intersection = cross(h_u, h_v)` and the subsequent division `u = intersection[0] / intersection[2]` are both differentiable operations in MLX. The chain rule through these operations:

```
# Forward:
h_u = -M[0, :] + M[2, :] * px               # Linear in M
h_v = -M[1, :] + M[2, :] * py               # Linear in M
c = cross(h_u, h_v)                          # Bilinear in (h_u, h_v)
u = c[0] / c[2]                             # Rational function of c
v = c[1] / c[2]                             # Rational function of c

# Cross product expansion:
c[0] = h_u[1]*h_v[2] - h_u[2]*h_v[1]
c[1] = h_u[2]*h_v[0] - h_u[0]*h_v[2]
c[2] = h_u[0]*h_v[1] - h_u[1]*h_v[0]

# Backward for u = c[0]/c[2]:
d(u)/d(c[0]) = 1/c[2]
d(u)/d(c[2]) = -c[0]/c[2]^2 = -u/c[2]

# When c[2] -> 0 (degenerate, ray parallel to surfel):
# u and v blow up, making sigma_3d very large
# The min(sigma_3d, sigma_2d) selects sigma_2d instead
# So gradients never actually flow through this degenerate path
```

MLX autograd handles the cross product and division automatically. The key insight is that `min(sigma_3d, sigma_2d)` acts as a natural gradient gate: degenerate intersections produce large `sigma_3d`, causing the `min` to select `sigma_2d`, which has well-behaved gradients.

---

## 5. Technical Design

### 5.1 File Structure

```
src/gsplat_mlx/
    core_2dgs/
        __init__.py              # Public API exports
        projection_2dgs.py       # fully_fused_projection_2dgs()
        accumulate_2dgs.py       # accumulate_2dgs()
        rasterization_2dgs.py    # rasterize_to_pixels_2dgs(), rasterization_2dgs()

tests/
    test_2dgs.py                 # All 2DGS tests
```

### 5.2 Function Signatures and Implementations

#### 5.2.1 `quat_scale_to_matrix(quats, scales) -> mx.array`

Converts quaternion orientation and scales to a 3x3 rotation-scale matrix. This may already exist in PRD-02 math utils; if not, it must be added.

```python
def quat_scale_to_matrix(
    quats: mx.array,    # [..., 4] -- (w, x, y, z) quaternion
    scales: mx.array,   # [..., 3] -- scale factors
) -> mx.array:          # [..., 3, 3] -- RS matrix
    """Compute R(q) * diag(s).

    Returns:
        RS matrix where column i = s_i * r_i (rotation column scaled by corresponding scale).
    """
    R = quat_to_rotmat(quats)       # [..., 3, 3]  (from PRD-02)
    RS = R * scales[..., None, :]   # [..., 3, 3]  broadcast: scale each column
    return RS
```

**Upstream reference** (`_math.py` lines 667-677):

```python
def _quat_scale_to_matrix(quats, scales):
    R = _quat_to_rotmat(quats)  # [..., 3, 3]
    M = R * scales[..., None, :]  # [..., 3, 3]
    return M
```

#### 5.2.2 `fully_fused_projection_2dgs(...)`

```python
def fully_fused_projection_2dgs(
    means: mx.array,        # [..., N, 3]
    quats: mx.array,        # [..., N, 4]
    scales: mx.array,       # [..., N, 3]
    viewmats: mx.array,     # [..., C, 4, 4]
    Ks: mx.array,           # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps: float = 0.0,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Project 2D Gaussian surfels to screen space.

    This function computes the ray-surfel intersection transform matrix for each
    surfel in each camera view, along with screen-space bounding boxes, depths,
    and camera-space normals.

    Args:
        means: Surfel centers in world space. [..., N, 3]
        quats: Surfel orientations as quaternions (w,x,y,z). [..., N, 4]
        scales: Surfel scales (s_u, s_v, s_n). [..., N, 3]
        viewmats: World-to-camera transforms. [..., C, 4, 4]
        Ks: Camera intrinsic matrices. [..., C, 3, 3]
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping distance.
        far_plane: Far clipping distance.
        eps: Epsilon for discriminant validity check.

    Returns:
        radii: Integer bounding radii per axis. [..., C, N, 2] int32.
            Zero indicates culled surfel.
        means2d: Screen-space surfel centers. [..., C, N, 2] float32.
        depths: Camera-space depths (z coordinate). [..., C, N] float32.
        ray_transforms: Ray-to-surfel intersection matrices. [..., C, N, 3, 3] float32.
            Used by accumulate_2dgs for per-pixel intersection computation.
        normals: Camera-space surface normals (facing camera). [..., C, N, 3] float32.
    """
```

**Full implementation**:

```python
def fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width, height,
                                 near_plane=0.01, far_plane=1e10, eps=0.0):
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]

    R_cw = viewmats[..., :3, :3]                     # [..., C, 3, 3]
    t_cw = viewmats[..., :3, 3]                       # [..., C, 3]

    # 1. Transform means to camera space
    means_c = mx.einsum("...cij,...nj->...cni", R_cw, means) + t_cw[..., None, :]
    # Shape: [..., C, N, 3]

    # 2. Compute RS matrix in world space
    RS_wl = quat_scale_to_matrix(quats, scales)        # [..., N, 3, 3]

    # 3. Transform RS to camera space
    RS_cl = mx.einsum("...cij,...njk->...cnik", R_cw, RS_wl)  # [..., C, N, 3, 3]

    # 4. Compute normals (third column of RS in camera space)
    normals = RS_cl[..., 2]                            # [..., C, N, 3]

    # 5. Flip normals to face camera: dot(-normal, mean_c) should be > 0
    cos = mx.sum(-normals * means_c, axis=-1, keepdims=True)  # [..., C, N, 1]
    multiplier = mx.where(cos > 0, mx.array(1.0), mx.array(-1.0))
    normals = normals * multiplier                     # [..., C, N, 3]

    # 6. Build ray transform matrix T_cl = [RS_cl[:, :2], means_c]
    T_cl = mx.concatenate([RS_cl[..., :2], means_c[..., None]], axis=-1)
    # Shape: [..., C, N, 3, 3]

    # 7. Project to screen space
    T_sl = mx.einsum("...cij,...cnjk->...cnik", Ks[..., :3, :3], T_cl)
    # Shape: [..., C, N, 3, 3]

    # 8. Transpose for AABB computation
    M = mx.swapaxes(T_sl, -1, -2)  # [..., C, N, 3, 3]

    # 9. AABB computation
    test = mx.broadcast_to(
        mx.array([1.0, 1.0, -1.0]),
        batch_dims + (1, 1, 3)
    )

    d = mx.sum(M[..., 2, :] * M[..., 2, :] * test, axis=-1, keepdims=True)
    # Shape: [..., C, N, 1]

    valid = mx.abs(d) > eps
    f = mx.where(valid, test / d, mx.zeros_like(test))
    f = mx.expand_dims(f, axis=-1)  # [..., C, N, 3, 1]

    means2d = mx.sum(M[..., :2, :] * M[..., 2:3, :] * f[..., 0], axis=-1)
    # Shape: [..., C, N, 2]

    extents = mx.sqrt(
        mx.clip(
            means2d ** 2 - mx.sum(M[..., :2, :] * M[..., :2, :] * f[..., 0], axis=-1),
            a_min=1e-4
        )
    )
    # Shape: [..., C, N, 2]

    # 10. Depths and radii
    depths = means_c[..., 2]                           # [..., C, N]
    radius = mx.ceil(3.33 * extents)                   # [..., C, N, 2]

    # 11. Visibility filtering
    valid_squeezed = mx.squeeze(valid, axis=-1)        # [..., C, N]
    valid_depth = valid_squeezed & (depths > near_plane) & (depths < far_plane)
    radius = mx.where(valid_depth[..., None], radius, mx.zeros_like(radius))

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0) &
        (means2d[..., 0] - radius[..., 0] < width) &
        (means2d[..., 1] + radius[..., 1] > 0) &
        (means2d[..., 1] - radius[..., 1] < height)
    )
    radius = mx.where(inside[..., None], radius, mx.zeros_like(radius))
    radii = radius.astype(mx.int32)

    # 12. Return T_sl (un-transposed M) as ray_transforms
    ray_transforms = mx.swapaxes(M, -1, -2)           # [..., C, N, 3, 3]

    return radii, means2d, depths, ray_transforms, normals
```

#### 5.2.3 `accumulate_2dgs(...)`

```python
def accumulate_2dgs(
    means2d: mx.array,          # [..., N, 2]
    ray_transforms: mx.array,   # [..., N, 3, 3]
    opacities: mx.array,        # [..., N]
    colors: mx.array,           # [..., N, channels]
    normals: mx.array,          # [..., N, 3]
    gaussian_ids: mx.array,     # [M]
    pixel_ids: mx.array,        # [M]
    image_ids: mx.array,        # [M]
    image_width: int,
    image_height: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Alpha compositing for 2D Gaussian surfels.

    Computes ray-surfel intersections per pixel, evaluates the dual-sigma
    Gaussian falloff (min of 3D and 2D sigma), and accumulates colors,
    alphas, and normals via front-to-back compositing.

    Args:
        means2d: Screen-space surfel centers. [..., N, 2]
        ray_transforms: Ray-to-surfel intersection matrices. [..., N, 3, 3]
        opacities: Per-view surfel opacities. [..., N]
        colors: Per-view surfel colors. [..., N, channels]
        normals: Per-view camera-space normals. [..., N, 3]
        gaussian_ids: Indices of surfels to rasterize. [M]
        pixel_ids: Row-major pixel indices to rasterize. [M]
        image_ids: Image indices to rasterize. [M]
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        renders: Accumulated colors. [..., image_height, image_width, channels]
        alphas: Accumulated opacities. [..., image_height, image_width, 1]
        render_normals: Accumulated normals. [..., image_height, image_width, 3]
    """
```

**Full implementation**:

```python
def accumulate_2dgs(means2d, ray_transforms, opacities, colors, normals,
                     gaussian_ids, pixel_ids, image_ids, image_width, image_height):
    MAX_ALPHA = 0.99

    image_dims = means2d.shape[:-2]
    I = 1
    for d in image_dims:
        I *= d
    N = means2d.shape[-2]
    channels = colors.shape[-1]

    # Flatten batch dims
    means2d_flat = mx.reshape(means2d, (I, N, 2))
    ray_transforms_flat = mx.reshape(ray_transforms, (I, N, 3, 3))
    opacities_flat = mx.reshape(opacities, (I, N))
    colors_flat = mx.reshape(colors, (I, N, channels))
    normals_flat = mx.reshape(normals, (I, N, 3))

    # === Per-intersection sigma computation ===

    # Pixel center coordinates
    pixel_ids_x = (pixel_ids % image_width).astype(mx.float32) + 0.5
    pixel_ids_y = (pixel_ids // image_width).astype(mx.float32) + 0.5
    pixel_coords = mx.stack([pixel_ids_x, pixel_ids_y], axis=-1)  # [M, 2]

    # Screen-space deltas for sigma_2d
    deltas = pixel_coords - means2d_flat[image_ids, gaussian_ids]  # [M, 2]

    # Gather ray transform matrices for this intersection set
    M = ray_transforms_flat[image_ids, gaussian_ids]               # [M, 3, 3]

    # Ray-surfel intersection via cross product
    h_u = -M[..., 0, :3] + M[..., 2, :3] * pixel_ids_x[..., None]  # [M, 3]
    h_v = -M[..., 1, :3] + M[..., 2, :3] * pixel_ids_y[..., None]  # [M, 3]

    tmp = mx.cross(h_u, h_v)                                        # [M, 3]

    # Dehomogenize to surfel-local UV coordinates
    us = tmp[..., 0] / tmp[..., 2]                                   # [M]
    vs = tmp[..., 1] / tmp[..., 2]                                   # [M]

    # 3D sigma: squared distance in surfel's local frame
    sigmas_3d = us ** 2 + vs ** 2                                    # [M]

    # 2D sigma: screen-space squared distance (with factor 2 for normalization)
    sigmas_2d = 2.0 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)   # [M]

    # Dual sigma selection: use whichever is smaller
    sigmas = 0.5 * mx.minimum(sigmas_3d, sigmas_2d)                  # [M]

    # Per-intersection alpha
    alphas = mx.clip(
        opacities_flat[image_ids, gaussian_ids] * mx.exp(-sigmas),
        a_max=MAX_ALPHA
    )  # [M]

    # === Alpha compositing (replacing nerfacc) ===

    indices = image_ids * image_height * image_width + pixel_ids
    total_pixels = I * image_height * image_width

    # Compute weights via front-to-back transmittance
    weights, _ = _render_weight_from_alpha(alphas, indices, total_pixels)

    # Accumulate colors
    renders = _accumulate_along_rays(
        weights, colors_flat[image_ids, gaussian_ids], indices, total_pixels
    )
    renders = mx.reshape(renders, image_dims + (image_height, image_width, channels))

    # Accumulate alphas
    alphas_out = _accumulate_along_rays(weights, None, indices, total_pixels)
    alphas_out = mx.reshape(alphas_out, image_dims + (image_height, image_width, 1))

    # Accumulate normals (key 2DGS addition)
    renders_normal = _accumulate_along_rays(
        weights, normals_flat[image_ids, gaussian_ids], indices, total_pixels
    )
    renders_normal = mx.reshape(renders_normal, image_dims + (image_height, image_width, 3))

    return renders, alphas_out, renders_normal
```

#### 5.2.4 `rasterize_to_pixels_2dgs(...)`

```python
def rasterize_to_pixels_2dgs(
    means2d: mx.array,          # [..., N, 2]
    ray_transforms: mx.array,   # [..., N, 3, 3]
    colors: mx.array,           # [..., N, channels]
    normals: mx.array,          # [..., N, 3]
    opacities: mx.array,        # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: mx.array,    # [..., tile_height, tile_width]
    flatten_ids: mx.array,      # [n_isects]
    backgrounds: Optional[mx.array] = None,  # [..., channels]
    batch_per_iter: int = 100,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Tile-based rasterization for 2D Gaussian surfels.

    Iteratively processes batches of surfels per tile, calling accumulate_2dgs
    for each batch and accumulating results with proper transmittance tracking.

    This follows the same batched iteration pattern as the 3DGS rasterize_to_pixels
    (PRD-07), but calls accumulate_2dgs and additionally tracks render_normals.

    Returns:
        render_colors: [..., image_height, image_width, channels]
        render_alphas: [..., image_height, image_width, 1]
        render_normals: [..., image_height, image_width, 3]
    """
```

**Implementation sketch** (follows upstream `_rasterize_to_pixels_2dgs` lines 215-334):

```python
def rasterize_to_pixels_2dgs(means2d, ray_transforms, colors, normals, opacities,
                              image_width, image_height, tile_size,
                              isect_offsets, flatten_ids, backgrounds=None,
                              batch_per_iter=100):
    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    N = means2d.shape[-2]
    n_isects = flatten_ids.shape[0]

    # Initialize output buffers
    render_colors = mx.zeros(image_dims + (image_height, image_width, channels))
    render_alphas = mx.zeros(image_dims + (image_height, image_width, 1))
    render_normals = mx.zeros(image_dims + (image_height, image_width, 3))

    # Compute iteration bounds
    block_size = tile_size * tile_size
    isect_offsets_fl = mx.concatenate([
        mx.reshape(isect_offsets, (-1,)),
        mx.array([n_isects])
    ])
    max_range = int(mx.max(isect_offsets_fl[1:] - isect_offsets_fl[:-1]).item())
    num_batches = (max_range + block_size - 1) // block_size

    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find intersections for this batch range
        # Uses rasterize_to_indices_in_range_2dgs (ported from _wrapper.py)
        gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range_2dgs(
            step, step + batch_per_iter,
            transmittances, means2d, ray_transforms, opacities,
            image_width, image_height, tile_size,
            isect_offsets, flatten_ids,
        )

        if len(gs_ids) == 0:
            break

        # Accumulate this batch
        renders_step, alphas_step, normals_step = accumulate_2dgs(
            means2d, ray_transforms, opacities, colors, normals,
            gs_ids, pixel_ids, image_ids,
            image_width, image_height,
        )

        # Apply transmittance and accumulate
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + alphas_step * transmittances[..., None]
        render_normals = render_normals + normals_step * transmittances[..., None]

    # Apply background
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[..., None, None, :] * (1.0 - render_alphas)

    return render_colors, render_alphas, render_normals
```

#### 5.2.5 `rasterization_2dgs(...)` -- High-Level API

```python
def rasterization_2dgs(
    means: mx.array,            # [N, 3]
    quats: mx.array,            # [N, 4]
    scales: mx.array,           # [N, 3]
    opacities: mx.array,        # [N]
    colors: mx.array,           # [N, D] or [N, K, 3] (SH) or [C, N, D]
    viewmats: mx.array,         # [C, 4, 4]
    Ks: mx.array,               # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[mx.array] = None,
    render_mode: str = "RGB",
) -> Tuple[mx.array, mx.array, mx.array, Dict]:
    """Render 2D Gaussian surfels to images.

    This is the main user-facing entry point for the 2DGS pipeline. It orchestrates:
    1. Scale activation (exp) and opacity activation (sigmoid)
    2. Projection via fully_fused_projection_2dgs
    3. Spherical harmonics evaluation (if sh_degree is set)
    4. Depth channel handling (for "D", "RGB+D" modes)
    5. Tile intersection and sorting (reuses PRD-06)
    6. Rasterization via rasterize_to_pixels_2dgs

    Args:
        means: Surfel centers in world space. [N, 3]
        quats: Quaternions (w,x,y,z). [N, 4]
        scales: Log-space scales (will be exponentiated). [N, 3]
        opacities: Logit-space opacities (will be sigmoided). [N]
        colors: Colors or SH coefficients.
        viewmats: World-to-camera transforms. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        near_plane: Near clipping distance.
        far_plane: Far clipping distance.
        eps2d: Epsilon for projection stability.
        sh_degree: SH degree (None = direct colors).
        tile_size: Tile size for rasterization.
        backgrounds: Background colors. [C, D] or None.
        render_mode: "RGB", "D", "RGB+D", etc.

    Returns:
        render_colors: Rendered image. [C, H, W, D]
        render_alphas: Rendered alpha. [C, H, W, 1]
        render_normals: Rendered normal map. [C, H, W, 3]
        info: Dict with intermediate results:
            - "means2d": [C, N, 2]
            - "depths": [C, N]
            - "radii": [C, N, 2]
            - "ray_transforms": [C, N, 3, 3]
            - "normals": [C, N, 3]
            - "opacities": [C, N]
            - "width": int
            - "height": int
            - "tile_size": int
            - "tiles_per_gauss": [C, N]
            - "isect_ids": [n_isects]
            - "flatten_ids": [n_isects]
            - "isect_offsets": [C, tile_height, tile_width]
            - "n_cameras": int
    """
```

**Pipeline flow**:

```
Input (means, quats, scales, opacities, colors)
    |
    v
[exp(scales), sigmoid(opacities)]          # Activation
    |
    v
fully_fused_projection_2dgs()              # -> radii, means2d, depths, ray_transforms, normals
    |
    v
spherical_harmonics() [if SH]              # -> colors_rast [C, N, D]  (reuse PRD-04)
    |
    v
[concatenate depth channel if needed]      # For "D", "RGB+D" modes
    |
    v
isect_tiles() + isect_offset_encode()      # Reuse from PRD-06 (same tile logic)
    |
    v
rasterize_to_pixels_2dgs()                 # -> render_colors, render_alphas, render_normals
    |
    v
Output (render_colors, render_alphas, render_normals, info)
```

### 5.3 Custom VJP Requirements

#### `fully_fused_projection_2dgs` VJP

The projection function contains operations that benefit from a custom VJP for efficiency and correctness (normal flipping via `mx.where`, AABB computation, integer radii output).

```python
@mx.custom_function
def fully_fused_projection_2dgs_fwd(means, quats, scales, viewmats, Ks, ...):
    # Forward pass as described in Section 5.2.2
    radii, means2d, depths, ray_transforms, normals = _projection_impl(...)
    return (radii, means2d, depths, ray_transforms, normals)

@fully_fused_projection_2dgs_fwd.vjp
def fully_fused_projection_2dgs_bwd(primals, outputs, grad_outputs):
    g_radii, g_means2d, g_depths, g_ray_transforms, g_normals = grad_outputs
    # g_radii is ignored (integer output, no gradient)

    # Key gradient paths:
    #   g_ray_transforms -> g_T_sl -> g_T_cl -> g_RS_cl, g_means_c
    #     -> g_means (via R_cw^T), g_quats, g_scales (via RS chain rule)
    #   g_normals -> g_RS_cl[..., 2] (with sign flip from multiplier)
    #     -> g_quats, g_scales
    #   g_means2d -> g_M (AABB computation) -> same chain as ray_transforms
    #   g_depths -> g_means_c[..., 2] -> g_means
    ...
```

**Gradient flow summary**:

| Output | Gradient flows to | Via |
|--------|-------------------|-----|
| `ray_transforms` | means, quats, scales | T_sl = K @ [RS_cl[:,:2], means_c] |
| `normals` | quats, scales | RS_cl[:, 2] with sign flip |
| `means2d` | means, quats, scales | AABB from M = T_sl^T |
| `depths` | means | means_c[:, 2] |
| `radii` | -- (integer, no gradient) | -- |

#### `accumulate_2dgs` Gradient

The accumulate function is composed of differentiable MLX operations (`mx.cross`, division, `mx.minimum`, `mx.exp`, `mx.clip`, scatter-add). MLX autograd can handle this without a custom VJP, provided:

1. `mx.cross` supports autograd (it does in MLX)
2. The `_render_weight_from_alpha` and `_accumulate_along_rays` helpers are implemented using differentiable ops

For the initial implementation, these helpers may use numpy loops (non-differentiable, matching PRD-08 pattern) for validation. A follow-up should replace them with pure MLX scatter ops for training support.

**Pure MLX approach for differentiable compositing** (preferred for training):

```python
def _render_weight_from_alpha_mlx(alphas, ray_indices, n_rays):
    """Pure MLX implementation for differentiable weight computation.

    Computes: weight[i] = alpha[i] * prod(1 - alpha[j] for j < i on same ray)

    This requires computing an exclusive cumulative product of (1 - alpha) per ray,
    which is challenging to vectorize. Options:
    1. Sequential loop (correct but slow)
    2. Segment-wise exclusive cumprod (requires ray boundary detection)
    3. Log-space: transmittance = exp(cumsum(log(1-alpha))) with segment resets
    """
    # Option 3 (vectorized, differentiable):
    log_one_minus_alpha = mx.log(mx.clip(1.0 - alphas, a_min=1e-10))

    # Detect ray boundaries (where ray_indices changes)
    # Reset cumsum at each boundary
    # ... (implementation details depend on MLX segment primitives)

    # Fallback: sequential for correctness first
    ...
```

### 5.4 Differences from 3DGS Port Summary

| Component | 3DGS (PRD-05/07/08) | 2DGS (this PRD) |
|-----------|---------------------|-----------------|
| Projection input | covars [N, 3, 3] or [N, 6] | quats [N, 4] + scales [N, 3] directly |
| Projection output | conics [C, N, 3] | ray_transforms [C, N, 3, 3] |
| Normal output | None | normals [C, N, 3] |
| Per-pixel sigma | Mahalanobis via conic: `0.5*(a*dx^2 + c*dy^2) + b*dx*dy` | Cross-product intersection + dual min sigma |
| Accumulate output | `(renders, alphas)` | `(renders, alphas, render_normals)` |
| Memory per Gaussian | 3 floats (conic) | 9 floats (ray_transforms) + 3 floats (normal) = 12 floats |
| Rendering API return | `(colors, alphas, info)` | `(colors, alphas, normals, info)` |

---

## 6. Data Flow Diagram

```
                     means [N,3]     quats [N,4]     scales [N,3]
                         |               |               |
                         v               v               v
                    +-----------------------------------------+
                    |   fully_fused_projection_2dgs()         |
                    |                                         |
                    |   1. means_c = R_cw @ means + t_cw      |
                    |   2. RS_wl = quat_scale_to_matrix()     |
                    |   3. RS_cl = R_cw @ RS_wl               |
                    |   4. normals = RS_cl[...,2] (flipped)   |
                    |   5. T_cl = [RS_cl[:,:2], means_c]      |
                    |   6. T_sl = K @ T_cl                    |
                    |   7. M = T_sl^T  (for AABB)             |
                    |   8. AABB -> means2d, extents, radii    |
                    |   9. Visibility filtering               |
                    +-----------------------------------------+
                         |         |        |         |           |
                     radii    means2d   depths   ray_transforms  normals
                    [C,N,2]  [C,N,2]   [C,N]    [C,N,3,3]      [C,N,3]
                      int32                                        |
                         |         |        |         |            |
                         v         v        v         |            |
                    +-------------------------+       |            |
                    | isect_tiles()   (PRD-06) |       |            |
                    | isect_offset_encode()    |       |            |
                    +-------------------------+       |            |
                         |              |             |            |
                    isect_offsets   flatten_ids        |            |
                    [C,TH,TW]      [n_isects]         |            |
                         |              |             |            |
                         v              v             v            v
                    +---------------------------------------------------+
                    |   rasterize_to_pixels_2dgs()                      |
                    |                                                   |
                    |   For each tile batch:                            |
                    |     accumulate_2dgs():                            |
                    |       1. px, py = pixel centers                   |
                    |       2. h_u = -M[0,:] + M[2,:]*px               |
                    |       3. h_v = -M[1,:] + M[2,:]*py               |
                    |       4. tmp = cross(h_u, h_v)                   |
                    |       5. u = tmp[0]/tmp[2], v = tmp[1]/tmp[2]    |
                    |       6. sigma_3d = u^2 + v^2                    |
                    |       7. sigma_2d = 2*(dx^2 + dy^2)              |
                    |       8. sigma = 0.5 * min(sigma_3d, sigma_2d)   |
                    |       9. alpha = opacity * exp(-sigma)           |
                    |      10. front-to-back composite: colors+normals |
                    +---------------------------------------------------+
                         |              |              |
                    render_colors  render_alphas  render_normals
                    [C,H,W,D]     [C,H,W,1]     [C,H,W,3]
```

---

## 7. Test Plan

### 7.1 Test File: `tests/test_2dgs.py`

| # | Test Case | Description | Key Assertions |
|---|-----------|-------------|----------------|
| 1 | `test_2dgs_projection_shapes` | Verify output shapes from `fully_fused_projection_2dgs` | radii: `[C,N,2]` int32; means2d: `[C,N,2]`; depths: `[C,N]`; ray_transforms: `[C,N,3,3]`; normals: `[C,N,3]` |
| 2 | `test_2dgs_projection_depth_filtering` | Surfels behind near plane have radii=0 | `radii[behind_near] == 0` for all culled surfels |
| 3 | `test_2dgs_projection_bounds_filtering` | Surfels outside image bounds have radii=0 | `radii[outside_bounds] == 0` |
| 4 | `test_2dgs_normals_face_camera` | All returned normals point toward camera origin | `dot(normal, -mean_c) > 0` for all visible surfels |
| 5 | `test_2dgs_normals_flip` | Verify normal flipping for back-facing surfels | Construct surfel with normal pointing away, verify it gets flipped in output |
| 6 | `test_2dgs_ray_transforms_structure` | Verify `ray_transforms = K @ [RS_cl[:,:2], means_c]` | Manual step-by-step computation matches function output |
| 7 | `test_2dgs_accumulate_single_surfel` | One surfel at pixel center, compute intersection | `sigma_3d ~= 0`, `alpha ~= opacity`, rendered color ~= surfel color |
| 8 | `test_2dgs_accumulate_off_center` | One surfel evaluated at offset pixel | `sigma > 0`, `alpha < opacity`, falloff matches Gaussian |
| 9 | `test_2dgs_accumulate_two_surfels` | Two overlapping surfels, verify blending order | Front surfel contributes more weight than back surfel |
| 10 | `test_2dgs_min_sigma_3d_dominates` | Head-on view: construct case where sigma_3d < sigma_2d | Verify `sigma == 0.5 * sigma_3d` by comparing both values |
| 11 | `test_2dgs_min_sigma_2d_dominates` | Grazing angle: construct case where sigma_2d < sigma_3d | Verify `sigma == 0.5 * sigma_2d` by comparing both values |
| 12 | `test_2dgs_cross_product_intersection` | Verify (u,v) from cross product matches manual calculation | Direct formula vs function output, atol=1e-6 |
| 13 | `test_2dgs_render_colors` | Full pipeline: 50 surfels at 64x64 | Output shape `[C,64,64,3]`, values in `[0,1]`, not all zero |
| 14 | `test_2dgs_render_normals` | Verify normal map output from full render | Output shape `[C,64,64,3]`, normals roughly unit-length where alpha > 0.5 |
| 15 | `test_2dgs_render_with_background` | Background visible through transparent regions | `render_colors[alpha < 0.01] ~= background` |
| 16 | `test_2dgs_render_depth_mode` | "RGB+D" mode appends depth channel | Output shape `[C,H,W,4]`, depth channel has reasonable values |
| 17 | `test_2dgs_vs_3dgs_flat` | For flat Gaussians (s_n ~= 0), 2DGS should approximate 3DGS | Rendered images within atol=0.1 (approximate, not exact match) |
| 18 | `test_2dgs_vjp_means` | Gradient w.r.t. means through full pipeline | Finite difference check, atol=1e-3 |
| 19 | `test_2dgs_vjp_quats` | Gradient w.r.t. quats through full pipeline | Finite difference check, atol=1e-3 |
| 20 | `test_2dgs_vjp_scales` | Gradient w.r.t. scales through full pipeline | Finite difference check, atol=1e-3 |
| 21 | `test_2dgs_vjp_opacities` | Gradient w.r.t. opacities through full pipeline | Finite difference check, atol=1e-3 |
| 22 | `test_2dgs_vjp_colors` | Gradient w.r.t. colors through full pipeline | Finite difference check, atol=1e-3 |
| 23 | `test_2dgs_cross_framework` | Compare MLX output with torch reference for same inputs | Forward: atol=1e-4; VJP: atol=1e-3 |
| 24 | `test_2dgs_empty_scene` | Zero visible surfels (all behind camera) | All-zero renders, alpha=0 everywhere |
| 25 | `test_2dgs_single_camera` | C=1 rendering | Correct shapes and reasonable values |
| 26 | `test_2dgs_multi_camera` | C=2 rendering from different viewpoints | Different images per camera |

### 7.2 Cross-Framework Test Detail

The `test_2dgs_cross_framework` test is the most critical validation. It should:

1. Generate random surfels (N=50) with a known seed
2. Set up a camera at a known position (e.g., z=3 looking at origin)
3. Run the torch reference `_fully_fused_projection_2dgs` and `accumulate_2dgs`
4. Run the MLX implementation with the same inputs (converted via `mx.array(tensor.numpy())`)
5. Compare outputs element-by-element

```python
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_2dgs_cross_framework():
    """Compare MLX 2DGS against torch reference implementation."""
    import torch
    from gsplat.cuda._torch_impl_2dgs import (
        _fully_fused_projection_2dgs as torch_proj,
        accumulate_2dgs as torch_accum,
    )
    from gsplat_mlx.core_2dgs.projection_2dgs import fully_fused_projection_2dgs
    from tests.conftest import check_all_close

    # Setup with fixed seed
    torch.manual_seed(42)
    N, C = 50, 1
    W, H = 64, 64

    means = torch.randn(N, 3)
    quats = torch.randn(N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3) * 0.1
    viewmats = torch.eye(4).unsqueeze(0)
    viewmats[0, 2, 3] = 3.0  # camera at z=3
    Ks = torch.tensor([
        [300., 0., 32.], [0., 300., 32.], [0., 0., 1.]
    ]).unsqueeze(0)

    # Torch forward
    t_radii, t_means2d, t_depths, t_M, t_normals = torch_proj(
        means, quats, scales, viewmats, Ks, W, H
    )

    # MLX forward
    m_radii, m_means2d, m_depths, m_M, m_normals = fully_fused_projection_2dgs(
        mx.array(means.numpy()), mx.array(quats.numpy()),
        mx.array(scales.numpy()), mx.array(viewmats.numpy()),
        mx.array(Ks.numpy()), W, H
    )

    # Compare (skip radii which may differ by 1 due to ceil rounding)
    check_all_close(m_means2d, t_means2d.numpy(), atol=1e-4)
    check_all_close(m_depths, t_depths.numpy(), atol=1e-4)
    check_all_close(m_M, t_M.numpy(), atol=1e-4)
    check_all_close(m_normals, t_normals.numpy(), atol=1e-4)
```

### 7.3 Tolerances

| Comparison | Tolerance | Rationale |
|------------|-----------|-----------|
| Forward pass (MLX vs torch) | `atol=1e-4` | float32 precision differences across frameworks |
| VJP (vs finite difference) | `atol=1e-3` | Finite difference introduces O(h) error |
| VJP (MLX vs torch autograd) | `atol=1e-3` | Accumulated float32 rounding differences |
| 2DGS vs 3DGS (flat surfels) | `atol=0.1` | Approximate equivalence only; different projection math |

---

## 8. Implementation Notes

### 8.1 MLX-Specific Considerations

1. **`mx.cross`**: MLX supports `mx.cross(a, b)` for 3D vectors. Verify it operates on the last dimension by default. If not, use `mx.cross(a, b, axis=-1)`.

2. **`mx.einsum`**: Used extensively for batched matrix operations. MLX's einsum supports the same notation as numpy/torch. Performance may vary; profile and consider `mx.matmul` alternatives if einsum is slow.

3. **`mx.where` with scalar broadcasts**: The normal flipping uses `mx.where(cos > 0, 1.0, -1.0)`. Ensure proper broadcasting with the `[..., C, N, 1]` shape against normals `[..., C, N, 3]`.

4. **`mx.minimum` autograd**: `mx.minimum(a, b)` is differentiable in MLX. The gradient flows to whichever input is smaller (subgradient at equality).

5. **Integer outputs**: `radii` is int32 and has no gradient. The custom VJP must handle this by returning zero/None for the radii gradient.

6. **Memory**: Each surfel stores a 3x3 ray_transforms matrix (9 floats = 36 bytes) vs 3 floats for a conic in 3DGS (12 bytes). Plus 3 floats for normals (12 bytes). For N=1M surfels and C=1 camera: 48MB for 2DGS vs 12MB for 3DGS. Still manageable on Apple Silicon unified memory (16GB+).

7. **`mx.concatenate` axis semantics**: When building T_cl from RS_cl columns and means_c, ensure the concatenation axis correctly forms a `[3, 3]` matrix from `[3, 2]` and `[3, 1]` parts. The `means_c[..., None]` adds the column dimension.

### 8.2 Reuse from Existing PRDs

| Component | Source PRD | Reuse Strategy |
|-----------|------------|----------------|
| `quat_to_rotmat` | PRD-02 | Direct import from `gsplat_mlx.core.math_utils` |
| `isect_tiles`, `isect_offset_encode` | PRD-06 | Direct import (identical tile logic, works with 2D radii) |
| `_render_weight_from_alpha` | PRD-08 | Import or copy (same front-to-back compositing logic) |
| `_accumulate_along_rays` | PRD-08 | Import or copy, extended to handle normals as an additional value channel |
| `spherical_harmonics` | PRD-04 | Direct import for SH color evaluation in high-level API |
| Test fixtures (`check_all_close`, camera setup) | PRD-01 | Direct import from `tests/conftest.py` |

### 8.3 Performance Considerations

The 2DGS pipeline is computationally heavier per surfel than 3DGS due to:

1. **Cross product per pixel-surfel pair**: 6 multiplies + 3 subtracts (vs 5 ops for conic evaluation in 3DGS)
2. **Division for dehomogenization**: 2 divides per intersection (not needed in 3DGS)
3. **Dual sigma + min**: Extra comparison per intersection
4. **Normal accumulation**: Additional weighted sum per pixel (3 extra channels)

For the initial port, **correctness is prioritized over performance**. Optimization opportunities for future work:

- Fuse the h_u/h_v computation with the cross product into a single vectorized op
- Use Metal shader for the per-pixel intersection kernel (similar to CUDA kernel fusion)
- Replace sequential transmittance loop with vectorized segment-cumprod
- Consider tiling normals separately to reduce memory pressure during accumulation

---

## 9. Acceptance Criteria

- [ ] `fully_fused_projection_2dgs()` produces correct ray_transforms, normals, means2d, depths, and radii for arbitrary inputs
- [ ] Normals always point toward the camera for visible surfels
- [ ] `accumulate_2dgs()` correctly computes ray-surfel intersection via cross product
- [ ] Dual sigma selection (`min(sigma_3d, sigma_2d)`) works correctly at both head-on and grazing angles
- [ ] Alpha compositing accumulates colors, alphas, AND normals (three output buffers)
- [ ] `rasterize_to_pixels_2dgs()` iterative batched rasterization produces correct images
- [ ] `rasterization_2dgs()` high-level API renders correct images from raw surfel parameters
- [ ] Normal map output is geometrically valid (roughly unit-length where alpha > 0)
- [ ] Gradient flows backward through the entire pipeline (means, quats, scales, opacities, colors)
- [ ] Forward pass matches torch reference within `atol=1e-4`
- [ ] VJP matches finite-difference / torch autograd within `atol=1e-3`
- [ ] All 26 tests pass with `pytest tests/test_2dgs.py -v`
- [ ] No dependency on `nerfacc` -- all compositing implemented natively in MLX
- [ ] Supports both single-camera (C=1) and multi-camera (C>1) rendering
- [ ] Info dict from `rasterization_2dgs()` contains all expected keys: means2d, depths, radii, ray_transforms, normals, opacities, isect_offsets, flatten_ids, tile dimensions

---

## 10. Dependencies

| PRD | What it provides | Required by |
|-----|-----------------|-------------|
| PRD-01 | Dev environment, test infrastructure, `check_all_close` | All tests |
| PRD-02 | `quat_to_rotmat`, basic math utils | `quat_scale_to_matrix`, projection |
| PRD-06 | `isect_tiles`, `isect_offset_encode` | Tile-based rasterization |
| PRD-09 | Rendering API patterns, pipeline structure | `rasterization_2dgs` API design |

### Optional dependencies (for shared helpers):

| PRD | What it provides | Used for |
|-----|-----------------|----------|
| PRD-04 | `spherical_harmonics` | SH color evaluation in high-level API |
| PRD-08 | `_render_weight_from_alpha`, `_accumulate_along_rays` | Alpha compositing helpers |

---

## 11. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `mx.cross` not supporting autograd | VJP through intersection fails | Low | Implement cross product manually: `c[0]=a[1]*b[2]-a[2]*b[1]`, etc. All ops individually differentiable. |
| Numerical instability at `tmp[2] -> 0` | NaN/Inf in dehomogenization | Medium | The `min(sigma_3d, sigma_2d)` selection handles this: sigma_2d takes over when intersection is degenerate. Add `mx.clip(tmp[2], a_min=1e-10)` as extra safety. |
| Memory pressure from 3x3 matrices | OOM on large scenes (N > 5M) | Low | 48 bytes/surfel is acceptable; 5M surfels = 240MB, well within M1/M2/M3/M4 unified memory. |
| Sequential transmittance computation | Slow pure-Python compositing | High (for training) | Acceptable for initial correctness validation. Replace with vectorized MLX ops before training benchmarks. |
| Normal flipping discontinuity at `cos = 0` | Gradient issues at exactly edge-on view | Very Low | Extremely rare in practice. `mx.where` subgradient is valid. Edge-on surfels contribute negligible alpha anyway. |
| `rasterize_to_indices_in_range_2dgs` not yet ported | `rasterize_to_pixels_2dgs` cannot function | Medium | Port this helper as part of this PRD, or implement a simplified version that iterates all intersection indices. |

---

## 12. Future Extensions (Out of Scope for PRD-12)

These are documented for future PRDs but explicitly not part of PRD-12:

1. **Distortion loss**: Regularization for better geometry (L1 version from gsplat, different from L2 in original 2DGS paper). Penalizes depth inconsistency within each pixel's surfel stack.
2. **Median depth rendering**: Alternative to expected depth for more robust depth maps.
3. **Surface normals from depth**: Compute normals from the rendered depth map via finite differences (post-processing, not per-surfel).
4. **2DGS densification strategy**: Split/clone/prune adapted for the surfel representation. May use `gradient_2dgs` from the info dict.
5. **Metal kernel fusion**: Fuse projection + intersection into a single GPU kernel for competitive performance with CUDA.
6. **Packed mode**: Memory-efficient sparse representation for large scenes with many culled surfels.
7. **Depth-to-normal consistency loss**: Enforce that rendered normals match normals derived from rendered depth.
