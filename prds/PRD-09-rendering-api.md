# PRD-09: High-Level Rendering API (`rasterization()`)

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-09 |
| **Title** | High-Level Rendering API |
| **Status** | DRAFT |
| **Priority** | P0 -- Critical Path |
| **Estimated Effort** | 8--12 hours |
| **Dependencies** | PRD-01 (environment), PRD-02 (math), PRD-03 (covariance), PRD-04 (SH), PRD-05 (projection), PRD-06 (intersection), PRD-07 (rasterization), PRD-08 (accumulate) |
| **Blocks** | PRD-10 (densification strategy), PRD-13 (training loop) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Implement the `rasterization()` function -- the single public entry point that users call to render 3D Gaussians into images. This function orchestrates the entire rendering pipeline:

1. Convert quaternions + scales to covariance matrices (PRD-03)
2. Project 3D Gaussians to 2D screen space (PRD-05)
3. Evaluate spherical harmonics for view-dependent color (PRD-04, when using SH)
4. Compute tile-Gaussian intersections (PRD-06)
5. Rasterize pixels via alpha compositing (PRD-07)

The function returns rendered images, alpha maps, and an `info` dictionary containing all intermediate results needed by training strategies (PRD-10) and the training loop (PRD-13).

After this PRD is implemented, a user should be able to:

```python
import gsplat_mlx as gsplat

render_colors, render_alphas, info = gsplat.rasterization(
    means, quats, scales, opacities, colors,
    viewmats, Ks, width, height
)
```

---

## 2. Source Reference

| Item | Location |
|------|----------|
| **Primary** | `repositories/gsplat-upstream/gsplat/rendering.py:255-1208` (`rasterization()`) |
| **Helper: RenderMode** | `rendering.py:59-92` (type alias + query functions) |
| **Helper: viewmat_to_camera_position** | `rendering.py:215-223` |
| **Helper: compute_directions** | `rendering.py:226-252` |
| **Helper: normalize_features_layout** | `rendering.py:169-212` |
| **Fallback: _rasterization** | `rendering.py:1251-1500` (PyTorch autograd version) |
| **Lines** | ~950 lines for `rasterization()`, ~250 lines helpers |

---

## 3. Scope

### 3.1 In Scope (MVP)

- `rasterization()` public API function
- Render modes: `"RGB"`, `"D"`, `"ED"`, `"RGB+D"`, `"RGB+ED"`
- Rasterize modes: `"classic"`, `"antialiased"`
- Camera models: `"pinhole"` only (MVP)
- SH color evaluation (degrees 0--4) when `sh_degree` is set
- Direct RGB/feature color pass-through when `sh_degree` is None
- Per-camera colors (`[C, N, D]` layout) and per-Gaussian colors (`[N, D]` layout)
- Background color blending
- Channel chunking for high-dimensional features (`channel_chunk`)
- Antialiased rendering with compensation factors
- The `info` dictionary with all intermediate results
- View direction computation for SH evaluation
- Input validation and shape assertions
- Full end-to-end differentiability via MLX autodiff
- Public API exports in `__init__.py`

### 3.2 Out of Scope (Deferred)

- `packed` mode (sparse/CSR format) -- defer to future PRD
- `sparse_grad` -- requires packed mode
- `distributed` rendering -- no multi-GPU on Apple Silicon
- `with_ut` (Unscented Transform) projection
- `with_eval3d` (3D evaluation mode)
- `return_normals`
- Hit distance render modes (`"d"`, `"Ed"`, `"RGB-d"`, `"RGB-Ed"`) -- require eval3d
- Fisheye, orthographic, f-theta, lidar camera models
- Distortion coefficients (radial, tangential, thin prism)
- Rolling shutter
- Extra signals (`extra_signals`, `extra_signals_sh_degree`)
- `segmented` radix sort
- `absgrad` mode (absolute gradients for densification) -- deferred to PRD-10 integration
- Batch dimensions (`batch_dims`) beyond a single batch -- MVP supports `means.shape == [N, 3]`
- Covariance matrix direct input (`covars` parameter)

### 3.3 Scope Notes

The upstream `rasterization()` is 950+ lines because it handles distributed rendering, packed mode, UT projection, eval3d, rolling shutter, lidar, and many camera models. Our MVP strips all of that away and focuses on the core path that covers >95% of use cases: pinhole camera, unpacked mode, SH or direct colors, classic or antialiased rasterization.

---

## 4. Technical Design

### 4.1 Module Structure

```
src/gsplat_mlx/
    __init__.py          # Public API exports (add rasterization)
    rendering.py         # NEW: rasterization() and helpers
```

### 4.2 Type Definitions

```python
from typing import Dict, Literal, Optional, Tuple
import mlx.core as mx

# Render mode type
RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]

# Rasterize mode type
RasterizeMode = Literal["classic", "antialiased"]
```

### 4.3 Helper Functions

#### 4.3.1 Render Mode Query Functions

These pure functions classify the render mode to control branching logic throughout the pipeline. They are ported directly from upstream `rendering.py:64-92`:

```python
def render_mode_has_color(mode: RenderMode) -> bool:
    """Returns True if the render mode includes RGB color output."""
    return mode in {"RGB", "RGB+D", "RGB+ED"}


def render_mode_has_depth(mode: RenderMode) -> bool:
    """Returns True if the render mode includes depth output."""
    return mode in {"D", "ED", "RGB+D", "RGB+ED"}


def render_mode_has_expected_depth(mode: RenderMode) -> bool:
    """Returns True if the render mode uses expected (normalized) depth."""
    return mode in {"ED", "RGB+ED"}


def render_mode_has_only_depth(mode: RenderMode) -> bool:
    """Returns True if the render mode outputs ONLY depth (no color)."""
    return mode in {"D", "ED"}


def render_mode_has_only_color(mode: RenderMode) -> bool:
    """Returns True if the render mode outputs ONLY color (no depth)."""
    return mode == "RGB"
```

#### 4.3.2 Camera Position from View Matrix

Ported from upstream `rendering.py:215-223`. Extracts the camera's world-space position from the world-to-camera 4x4 matrix without computing a full inverse.

For a view matrix `V = [R | t; 0 1]`, the inverse has translation `-R^T @ t`, so the camera position in world space is `-R^T @ t`.

```python
def viewmat_to_camera_position(viewmats: mx.array) -> mx.array:
    """Extract camera position in world coordinates from world-to-camera matrix.

    For V = [R | t; 0 1], the camera position is -R^T @ t.
    This avoids a full 4x4 matrix inverse.

    Args:
        viewmats: World-to-camera transformation matrices. [C, 4, 4]

    Returns:
        campos: Camera positions in world coordinates. [C, 3]
    """
    R = viewmats[..., :3, :3]  # [C, 3, 3]
    t = viewmats[..., :3, 3]   # [C, 3]
    # -R^T @ t
    R_T = mx.swapaxes(R, -1, -2)  # [C, 3, 3]
    result = -(R_T @ t[..., None])  # [C, 3, 1]
    return result[..., 0]  # [C, 3]
```

**MLX note**: MLX does not have a `.mT` property like PyTorch. We use `mx.swapaxes(R, -1, -2)` to transpose the last two dimensions.

#### 4.3.3 View Direction Computation

Ported and simplified from upstream `rendering.py:226-252` (the `compute_directions` function and `_compute_view_dirs_packed`). Since we do not support packed mode in the MVP, this is significantly simpler.

```python
def compute_view_directions(
    means: mx.array,      # [N, 3]
    viewmats: mx.array,   # [C, 4, 4]
) -> mx.array:
    """Compute normalized view directions from cameras to Gaussian centers.

    Computes the direction vector from each camera's world-space position to
    each Gaussian's world-space center, then normalizes to unit length.

    Args:
        means: 3D Gaussian centers. [N, 3]
        viewmats: World-to-camera matrices. [C, 4, 4]

    Returns:
        dirs: Normalized view directions. [C, N, 3]
    """
    # Camera positions in world space
    campos = viewmat_to_camera_position(viewmats)  # [C, 3]

    # Direction from camera to each Gaussian
    # means: [N, 3] -> [1, N, 3], campos: [C, 3] -> [C, 1, 3]
    dirs = means[None, :, :] - campos[:, None, :]  # [C, N, 3]

    # Normalize to unit length
    norms = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))  # [C, N, 1]
    dirs = dirs / mx.maximum(norms, mx.array(1e-8))

    return dirs  # [C, N, 3]
```

#### 4.3.4 Color Layout Normalization

Handles the two color input layouts (per-Gaussian `[N, D]` and per-camera `[C, N, D]`), broadcasting per-Gaussian colors to per-camera format. Ported from the logic in `normalize_features_layout` (upstream `rendering.py:169-212`) but simplified for our non-packed, non-batch MVP.

```python
def _normalize_colors(
    colors: mx.array,
    C: int,
    N: int,
    sh_degree: Optional[int],
) -> mx.array:
    """Normalize color tensor layout to [C, N, D] or [C, N, K, 3].

    Handles both per-Gaussian colors [N, D] and per-camera colors [C, N, D].
    When per-Gaussian, broadcasts to [C, N, D].

    Args:
        colors: Input color tensor.
            If sh_degree is None: [N, D] or [C, N, D]
            If sh_degree is set: [N, K, 3] or [C, N, K, 3]
        C: Number of cameras.
        N: Number of Gaussians.
        sh_degree: SH degree, or None for direct colors.

    Returns:
        colors: Normalized to [C, N, D] or [C, N, K, 3].
    """
    if sh_degree is None:
        # Direct colors: [N, D] or [C, N, D]
        if colors.ndim == 2:
            assert colors.shape[0] == N
            # Broadcast [N, D] -> [C, N, D]
            colors = mx.broadcast_to(colors[None, :, :], (C, N, colors.shape[-1]))
        else:
            assert colors.shape == (C, N, colors.shape[-1])
    else:
        # SH coefficients: [N, K, 3] or [C, N, K, 3]
        if colors.ndim == 3:
            assert colors.shape[0] == N
            assert colors.shape[-1] == 3
            # Broadcast [N, K, 3] -> [C, N, K, 3]
            colors = mx.broadcast_to(
                colors[None, :, :, :],
                (C, N, colors.shape[-2], 3)
            )
        else:
            assert colors.shape[:2] == (C, N)
            assert colors.shape[-1] == 3

    return colors
```

### 4.4 Main Function: `rasterization()`

This is the core of the PRD. The implementation follows the upstream structure but is simplified for our MVP scope.

```python
def rasterization(
    means: mx.array,                        # [N, 3]
    quats: mx.array,                        # [N, 4]
    scales: mx.array,                       # [N, 3]
    opacities: mx.array,                    # [N]
    colors: mx.array,                       # [N, D] or [C, N, D] or [N, K, 3] or [C, N, K, 3]
    viewmats: mx.array,                     # [C, 4, 4]
    Ks: mx.array,                           # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[mx.array] = None,  # [C, D]
    render_mode: RenderMode = "RGB",
    rasterize_mode: RasterizeMode = "classic",
    channel_chunk: int = 32,
) -> Tuple[mx.array, mx.array, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This is the main entry point for the gsplat-mlx rendering pipeline. It takes
    raw 3D Gaussian parameters and camera settings and returns rendered images.

    The pipeline internally executes:
        1. Projection: 3D Gaussians -> 2D screen-space ellipses
        2. SH Evaluation: SH coefficients -> view-dependent RGB (if sh_degree set)
        3. Tile Intersection: determine which tiles overlap each Gaussian
        4. Rasterization: alpha-composite Gaussians per pixel

    Color input modes:
        - Direct RGB/features: colors shape [N, D] or [C, N, D], sh_degree=None
        - SH coefficients: colors shape [N, K, 3] or [C, N, K, 3], sh_degree=0..4
          K must satisfy (sh_degree + 1)^2 <= K

    Render modes:
        - "RGB": Color only. Output channels = D (or 3 for SH).
        - "D": Accumulated depth only. Output channels = 1.
        - "ED": Expected depth (depth / alpha). Output channels = 1.
        - "RGB+D": Color + accumulated depth. Output channels = D+1.
        - "RGB+ED": Color + expected depth. Output channels = D+1.

    Rasterize modes:
        - "classic": Standard rasterization.
        - "antialiased": Apply Mip-Splatting compensation factor to opacities.
          Compensation = sqrt(det(Sigma) / det(Sigma + eps * I)), which reduces
          the opacity of near-degenerate Gaussians to prevent aliasing.

    Args:
        means: 3D centers of Gaussians. [N, 3]
        quats: Quaternion rotations (wxyz convention, unnormalized OK). [N, 4]
        scales: Per-axis scales. [N, 3]
        opacities: Opacity values in [0, 1]. [N]
        colors: Color data. Shape depends on mode:
            [N, D] or [C, N, D] for direct colors (sh_degree=None).
            [N, K, 3] or [C, N, K, 3] for SH coefficients (sh_degree set).
        viewmats: World-to-camera 4x4 matrices. [C, 4, 4]
        Ks: Camera intrinsics (3x3). [C, 3, 3]
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping plane distance. Default 0.01.
        far_plane: Far clipping plane distance. Default 1e10.
        radius_clip: Skip Gaussians with 2D radius <= this value (pixels).
            Useful for large scenes. Default 0.0 (disabled).
        eps2d: Epsilon added to 2D covariance eigenvalues to prevent
            degenerate (sub-pixel) Gaussians. Default 0.3.
        sh_degree: SH degree to activate. None means colors are direct values.
            When set, must satisfy (sh_degree + 1)^2 <= K. Default None.
        tile_size: Tile size for rasterization in pixels. Default 16.
        backgrounds: Background colors per camera. [C, D]. Default None (black).
        render_mode: What to render. Default "RGB".
        rasterize_mode: Rasterization variant. Default "classic".
        channel_chunk: Max channels per rasterization pass. Default 32.
            If total channels > channel_chunk, rendering is done in chunks.

    Returns:
        render_colors: Rendered output. [C, height, width, X] where X depends on
            render_mode: D for "RGB", 1 for "D"/"ED", D+1 for "RGB+D"/"RGB+ED".
        render_alphas: Rendered alpha (opacity accumulation). [C, height, width, 1].
        info: Dictionary of intermediate results for training/strategy use.
            Keys documented in section 4.7 of PRD-09.

    Examples:
        >>> import mlx.core as mx
        >>> import gsplat_mlx as gsplat
        >>> # Define Gaussians
        >>> means = mx.random.normal((100, 3))
        >>> quats = mx.random.normal((100, 4))
        >>> scales = mx.random.uniform(shape=(100, 3)) * 0.1
        >>> colors = mx.random.uniform(shape=(100, 3))
        >>> opacities = mx.sigmoid(mx.random.normal((100,)))
        >>> # Define camera
        >>> viewmats = mx.eye(4)[None, :, :]  # [1, 4, 4]
        >>> Ks = mx.array([[[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]]])
        >>> # Render
        >>> colors_out, alphas, info = gsplat.rasterization(
        ...     means, quats, scales, opacities, colors,
        ...     viewmats, Ks, width=300, height=200
        ... )
        >>> print(colors_out.shape, alphas.shape)
        (1, 200, 300, 3) (1, 200, 300, 1)
    """
```

### 4.5 Pipeline Orchestration (Complete Implementation Body)

The body of `rasterization()` is organized into 6 stages. Each stage is annotated with the PRD it depends on and the data it produces.

```python
    # =========================================================================
    # Stage 0: Input Validation
    # =========================================================================
    N = means.shape[0]
    C = viewmats.shape[0]
    info = {}

    assert means.shape == (N, 3), f"means shape {means.shape}, expected ({N}, 3)"
    assert quats.shape == (N, 4), f"quats shape {quats.shape}, expected ({N}, 4)"
    assert scales.shape == (N, 3), f"scales shape {scales.shape}, expected ({N}, 3)"
    assert opacities.shape == (N,), f"opacities shape {opacities.shape}, expected ({N},)"
    assert viewmats.shape == (C, 4, 4), f"viewmats shape {viewmats.shape}, expected ({C}, 4, 4)"
    assert Ks.shape == (C, 3, 3), f"Ks shape {Ks.shape}, expected ({C}, 3, 3)"

    # Validate color shapes based on sh_degree
    if sh_degree is not None:
        # SH mode: colors should be [N, K, 3] or [C, N, K, 3]
        if colors.ndim == 3:
            assert colors.shape[0] == N and colors.shape[-1] == 3, (
                f"SH colors shape {colors.shape}, expected [N, K, 3]"
            )
        elif colors.ndim == 4:
            assert colors.shape[:2] == (C, N) and colors.shape[-1] == 3, (
                f"SH colors shape {colors.shape}, expected [C, N, K, 3]"
            )
        else:
            raise ValueError(f"SH colors must be 3D or 4D, got {colors.ndim}D")
        K = colors.shape[-2]
        assert (sh_degree + 1) ** 2 <= K, (
            f"sh_degree={sh_degree} requires K >= {(sh_degree + 1) ** 2}, got K={K}"
        )
    else:
        # Direct color mode: [N, D] or [C, N, D]
        if colors.ndim == 2:
            assert colors.shape[0] == N, (
                f"Direct colors shape {colors.shape}, expected [N, D]"
            )
        elif colors.ndim == 3:
            assert colors.shape[:2] == (C, N), (
                f"Direct colors shape {colors.shape}, expected [C, N, D]"
            )
        else:
            raise ValueError(f"Direct colors must be 2D or 3D, got {colors.ndim}D")

    # Number of color channels (D) after potential SH evaluation
    D = colors.shape[-1] if sh_degree is None else 3

    # Validate render_mode
    valid_render_modes = {"RGB", "D", "ED", "RGB+D", "RGB+ED"}
    assert render_mode in valid_render_modes, (
        f"render_mode '{render_mode}' not in {valid_render_modes}"
    )

    # Validate rasterize_mode
    assert rasterize_mode in {"classic", "antialiased"}, (
        f"rasterize_mode '{rasterize_mode}' not in {{'classic', 'antialiased'}}"
    )

    # =========================================================================
    # Stage 1: Projection -- 3D Gaussians -> 2D Screen Space  [PRD-05]
    # =========================================================================
    # fully_fused_projection internally:
    #   a) Converts quats + scales -> covariance matrices via
    #      quat_scale_to_covar_preci (PRD-03)
    #   b) Transforms means + covars to camera space via _world_to_cam
    #   c) Projects to 2D via _persp_proj (perspective projection)
    #   d) Computes 2D radii, conics (inverse 2D covariance), depths
    #   e) Applies frustum culling (near/far plane, screen bounds)
    #   f) Optionally computes antialiasing compensation factors
    #
    # Returns unpacked format: all tensors are [C, N, ...]
    # Only elements where radii > 0 are valid (not culled).

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means,
        None,       # covars=None means use quats/scales path
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model="pinhole",
    )
    # radii:          [C, N]    int32   -- 2D bounding radius (0 = culled)
    # means2d:        [C, N, 2] float32 -- 2D projected centers in pixel coords
    # depths:         [C, N]    float32 -- depth in camera space (z-coordinate)
    # conics:         [C, N, 3] float32 -- inverse 2D covariance [a, b, c]
    # compensations:  [C, N]    float32 or None -- antialiasing factors

    # Broadcast opacities from [N] to [C, N] for per-camera rasterization
    opacities_2d = mx.broadcast_to(opacities[None, :], (C, N))  # [C, N]

    # Apply antialiasing compensation factor to opacities
    # This implements the Mip-Splatting approach:
    #   opacity_final = opacity * sqrt(det(Sigma) / det(Sigma + eps*I))
    # which reduces opacity for near-degenerate (very small) Gaussians
    if compensations is not None:
        opacities_2d = opacities_2d * compensations

    # Validity mask: radii > 0 means the Gaussian passed frustum culling
    # Used by SH evaluation to skip culled Gaussians (optimization)
    valid_mask = radii > 0  # [C, N] bool

    # Store projection results in info dict
    info.update({
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities_2d,
        "compensations": compensations,
    })

    # =========================================================================
    # Stage 2: Color Evaluation -- SH -> RGB (if needed)  [PRD-04]
    # =========================================================================
    # Normalize colors to [C, N, D] or [C, N, K, 3] (broadcast if per-Gaussian)
    colors = _normalize_colors(colors, C, N, sh_degree)

    if sh_degree is not None:
        # Compute view directions: from camera position to each Gaussian center
        # These directions are needed by the SH basis functions to produce
        # view-dependent colors.
        dirs = compute_view_directions(means, viewmats)  # [C, N, 3]

        # Evaluate spherical harmonics
        # Input:  colors [C, N, K, 3] (SH coefficients)
        # Output: colors [C, N, 3]    (evaluated RGB colors)
        #
        # The masks parameter tells SH evaluation to skip culled Gaussians
        # (their colors don't matter since they won't be rasterized).
        colors = spherical_harmonics(
            sh_degree, dirs, colors, masks=valid_mask
        )  # [C, N, 3]

        # Add 0.5 offset and clamp to match upstream behavior.
        # The Inria CUDA backend uses this convention: SH evaluates to values
        # centered around 0, so we add 0.5 and clamp to get colors in [0, 1+].
        colors = mx.maximum(colors + 0.5, mx.array(0.0))

        D = 3  # Output channels after SH evaluation

    # After this stage: colors is [C, N, D] regardless of input mode

    # =========================================================================
    # Stage 3: Depth Channel Preparation
    # =========================================================================
    # Depending on render_mode, we either:
    #   (a) Append depth as an extra channel to colors (RGB+D, RGB+ED)
    #   (b) Replace colors entirely with depth (D, ED)
    #   (c) Keep colors as-is (RGB)
    #
    # For "expected depth" modes (ED, RGB+ED), the depth is initially accumulated
    # as sum(w_i * z_i) during rasterization, then normalized by alpha in Stage 6.

    if render_mode_has_depth(render_mode) and render_mode_has_color(render_mode):
        # Combined color + depth modes: RGB+D or RGB+ED
        # Append depth as last channel: [C, N, D] + [C, N, 1] -> [C, N, D+1]
        colors = mx.concatenate(
            [colors, depths[..., None]],  # depths: [C, N] -> [C, N, 1]
            axis=-1
        )
        # Extend backgrounds with a zero depth channel
        if backgrounds is not None:
            backgrounds = mx.concatenate(
                [backgrounds, mx.zeros((C, 1))],
                axis=-1
            )
    elif render_mode_has_only_depth(render_mode):
        # Depth-only modes: D or ED
        # Replace colors entirely with depth
        colors = depths[..., None]  # [C, N, 1]
        if backgrounds is not None:
            backgrounds = mx.zeros((C, 1))
    else:
        # RGB-only mode: colors unchanged
        assert render_mode_has_only_color(render_mode)

    # =========================================================================
    # Stage 4: Tile Intersection  [PRD-06]
    # =========================================================================
    # Determine which screen tiles (tile_size x tile_size pixel blocks) each
    # projected 2D Gaussian overlaps. Then sort Gaussians within each tile
    # by depth for correct front-to-back alpha compositing.
    #
    # This stage produces:
    #   - isect_offsets: per-tile start index into flatten_ids
    #   - flatten_ids: sorted Gaussian indices for all tile intersections

    import math as _math
    tile_width = _math.ceil(width / float(tile_size))
    tile_height = _math.ceil(height / float(tile_size))

    I = C  # number of images (no batch dims in MVP)

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=I,
    )

    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)

    info.update({
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
    })

    # =========================================================================
    # Stage 5: Pixel Rasterization (with channel chunking)  [PRD-07]
    # =========================================================================
    # For each pixel, iterate over the sorted Gaussians in its tile and perform
    # front-to-back alpha compositing:
    #
    #   T = 1.0  (transmittance)
    #   color = [0, 0, ..., 0]
    #   for each Gaussian g in front-to-back order:
    #       alpha = opacity * exp(-0.5 * delta^T * conic * delta)
    #       color += T * alpha * gaussian_color
    #       T *= (1 - alpha)
    #       if T < 1e-4: break  (early termination)
    #   color += T * background  (remaining transmittance)
    #
    # When total color channels exceed channel_chunk, we split into multiple
    # passes. Each pass renders a subset of channels. The alpha map is the
    # same for all passes (since Gaussian geometry is identical).

    total_channels = colors.shape[-1]

    if total_channels > channel_chunk:
        # Render in chunks to limit per-pass memory
        n_chunks = (total_channels + channel_chunk - 1) // channel_chunk
        render_colors_chunks = []
        render_alphas_chunks = []

        for i in range(n_chunks):
            ch_start = i * channel_chunk
            ch_end = min((i + 1) * channel_chunk, total_channels)
            colors_chunk = colors[..., ch_start:ch_end]

            backgrounds_chunk = None
            if backgrounds is not None:
                backgrounds_chunk = backgrounds[..., ch_start:ch_end]

            rc, ra = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities_2d,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
            )
            render_colors_chunks.append(rc)
            render_alphas_chunks.append(ra)

        render_colors = mx.concatenate(render_colors_chunks, axis=-1)
        render_alphas = render_alphas_chunks[0]  # alpha is identical for all chunks
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities_2d,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )

    # render_colors: [C, height, width, total_channels]
    # render_alphas: [C, height, width, 1]

    # =========================================================================
    # Stage 6: Post-Processing -- Expected Depth Normalization
    # =========================================================================
    # For "expected depth" modes (ED, RGB+ED), the depth channel currently
    # contains accumulated depth: sum(w_i * z_i). To get expected depth,
    # divide by total alpha: sum(w_i * z_i) / sum(w_i).
    #
    # This is done by dividing the last channel of render_colors by render_alphas.
    # We clamp alpha to avoid division by zero in empty regions.

    if render_mode_has_expected_depth(render_mode):
        render_colors = mx.concatenate(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / mx.maximum(render_alphas, mx.array(1e-10)),
            ],
            axis=-1,
        )

    return render_colors, render_alphas, info
```

### 4.6 Pipeline Data Flow Diagram

```
INPUT
  means [N, 3]
  quats [N, 4]
  scales [N, 3]
  opacities [N]
  colors [N, D] or [N, K, 3]
  viewmats [C, 4, 4]
  Ks [C, 3, 3]

  |
  v
+--------------------------------------+
| Stage 1: fully_fused_projection      |
| (PRD-05)                             |
| internally calls:                     |
|   quat_scale_to_covar_preci (PRD-03) |
|   _world_to_cam                      |
|   _persp_proj                        |
|   frustum culling + radius           |
+--------------------------------------+
  |
  | radii [C, N], means2d [C, N, 2], depths [C, N],
  | conics [C, N, 3], compensations [C, N] or None
  |
  | opacities_2d = opacities[None,:] * compensations
  v
+--------------------------------------+
| Stage 2: spherical_harmonics         |
| (PRD-04)                             |
| (only if sh_degree is set)           |
|   compute_view_directions            |
|   SH basis evaluation                |
|   colors + 0.5, clamp >= 0           |
+--------------------------------------+
  |
  | colors [C, N, D]  (D=3 after SH, or original D)
  v
+--------------------------------------+
| Stage 3: Depth channel preparation   |
| RGB+D/RGB+ED: append depth [C,N,1]   |
| D/ED: replace colors with depth      |
| RGB: no change                        |
+--------------------------------------+
  |
  | colors [C, N, D'] where D' = D or D+1 or 1
  v
+--------------------------------------+
| Stage 4: isect_tiles +               |
| isect_offset_encode  (PRD-06)        |
+--------------------------------------+
  |
  | isect_offsets [C, tile_H, tile_W]
  | flatten_ids [n_isects]
  v
+--------------------------------------+
| Stage 5: rasterize_to_pixels         |
| (PRD-07)                             |
| Per-tile alpha compositing            |
| (with channel chunking if D' > 32)   |
+--------------------------------------+
  |
  | render_colors [C, H, W, D']
  | render_alphas [C, H, W, 1]
  v
+--------------------------------------+
| Stage 6: Post-processing             |
| ED/RGB+ED: depth /= alpha            |
+--------------------------------------+
  |
  v
OUTPUT
  render_colors [C, H, W, X]
  render_alphas [C, H, W, 1]
  info dict
```

### 4.7 The `info` Dictionary Specification

The `info` dict contains all intermediate results that downstream consumers (strategies, training loops, visualization) need. Every key is documented here:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `radii` | `[C, N]` | int32 | 2D bounding radius per Gaussian per camera. 0 = culled by frustum or radius_clip. |
| `means2d` | `[C, N, 2]` | float32 | 2D projected Gaussian centers in pixel coordinates. |
| `depths` | `[C, N]` | float32 | Depth of each Gaussian in camera space (z-coordinate). |
| `conics` | `[C, N, 3]` | float32 | Inverse 2D covariance matrix `[a, b, c]` where the full matrix is `[[a, b], [b, c]]`. |
| `opacities` | `[C, N]` | float32 | Final per-camera opacities (after antialiasing compensation if applicable). |
| `compensations` | `[C, N]` or `None` | float32 | Antialiasing compensation factors. `None` if `rasterize_mode="classic"`. |
| `tile_width` | scalar | int | Number of tiles horizontally. `ceil(width / tile_size)`. |
| `tile_height` | scalar | int | Number of tiles vertically. `ceil(height / tile_size)`. |
| `tiles_per_gauss` | `[C, N]` | int32 | Number of tiles each Gaussian overlaps. |
| `isect_ids` | `[n_isects]` | int64 | Packed intersection IDs encoding (image_id, tile_id, depth). Used for sorting. |
| `flatten_ids` | `[n_isects]` | int32 | Flattened Gaussian indices sorted by tile and depth. Index into the N Gaussians. |
| `isect_offsets` | `[C, tile_H, tile_W]` | int32 | Per-tile start offset into `flatten_ids`. `flatten_ids[isect_offsets[c,ty,tx]:isect_offsets[c,ty,tx+1]]` gives the Gaussians in tile `(ty, tx)` for camera `c`. |
| `width` | scalar | int | Image width in pixels. |
| `height` | scalar | int | Image height in pixels. |
| `tile_size` | scalar | int | Tile size in pixels (default 16). |
| `n_cameras` | scalar | int | Number of cameras (C). |

**Strategy usage (PRD-10):**
- `radii` -- identifies which Gaussians are visible across cameras
- `means2d` -- gradient magnitude of 2D positions used for split/clone decisions in MCMCStrategy and DefaultStrategy
- `opacities` -- pruning of low-opacity Gaussians
- `tiles_per_gauss` -- monitors tile coverage per Gaussian

**Training usage (PRD-13):**
- `means2d` -- requires grad for position optimization; gradient signals drive densification
- `opacities` -- monitoring convergence; used in opacity reset
- `depths` -- used in depth-supervised training (e.g., MonoGS)
- `conics` -- debugging and visualization

### 4.8 Differentiability Design

The entire pipeline must be differentiable end-to-end via MLX's autodiff. Each stage uses `@mx.custom_function` where needed:

| Stage | Differentiable? | Mechanism |
|-------|----------------|-----------|
| `fully_fused_projection` | Yes | `@mx.custom_function` with `.vjp` (PRD-05) |
| `spherical_harmonics` | Yes | `@mx.custom_function` with `.vjp` (PRD-04) |
| `isect_tiles` | No | No gradients needed (integer tile assignment) |
| `isect_offset_encode` | No | No gradients needed (prefix sum of counts) |
| `rasterize_to_pixels` | Yes | MLX autodiff or `@mx.custom_function` (PRD-07) |
| Depth normalization | Yes | Standard MLX ops (division, concatenation) |
| Color clamping | Yes | `mx.maximum` passes gradients through for `x > 0` |

The `rasterization()` function itself does NOT need `@mx.custom_function`. It composes differentiable sub-functions, and MLX will automatically chain VJPs through the composition.

**Gradient flow for key parameters:**

```
loss = f(render_colors)

d(loss)/d(means) flows through:
  render_colors
    -> rasterize_to_pixels (d/d means2d)
    -> fully_fused_projection (d/d means_cam -> d/d means_world)
    -> means

d(loss)/d(quats) flows through:
  render_colors
    -> rasterize_to_pixels (d/d conics)
    -> fully_fused_projection (d/d covars_2d -> d/d covars_3d)
    -> quat_scale_to_covar_preci (d/d quats)
    -> quats

d(loss)/d(scales) flows through:
  render_colors
    -> rasterize_to_pixels (d/d conics)
    -> fully_fused_projection (d/d covars_2d -> d/d covars_3d)
    -> quat_scale_to_covar_preci (d/d scales)
    -> scales

d(loss)/d(opacities) flows through:
  render_colors
    -> rasterize_to_pixels (d/d opacities_2d)
    -> broadcast (d/d opacities)
    [if antialiased: * compensations, but compensations has no grad w.r.t. opacities]
    -> opacities

d(loss)/d(colors) flows through:
  [if sh_degree is None:]
    render_colors
      -> rasterize_to_pixels (d/d colors)
      -> broadcast (d/d colors)
      -> colors

  [if sh_degree is set:]
    render_colors
      -> rasterize_to_pixels (d/d colors_rgb)
      -> clamp (gradient passthrough for > 0)
      -> spherical_harmonics (d/d sh_coefficients)
      -> broadcast (d/d colors)
      -> colors (SH coefficients)
```

**Important note on `mx.stop_gradient`**: The `isect_tiles` and `isect_offset_encode` stages return integer arrays that are inherently non-differentiable. No explicit `mx.stop_gradient` calls are needed because MLX does not track gradients for integer-typed arrays.

### 4.9 MLX-Specific Considerations

#### 4.9.1 No In-Place Operations

MLX arrays are immutable. All operations create new arrays. This affects:
- The `info` dict: values are snapshots, not references to mutable tensors
- Depth channel appending: uses `mx.concatenate`, not in-place append
- Color clamping: `mx.maximum(colors + 0.5, 0.0)` creates a new array

#### 4.9.2 Lazy Evaluation

MLX uses lazy evaluation. The `rasterization()` function builds a computation graph; actual computation only happens when results are evaluated (e.g., `mx.eval(render_colors)` or when values are read by the host). This means:
- The entire pipeline is fused into one computation graph
- Intermediate results in `info` are not computed until accessed/evaluated
- Users should call `mx.eval()` explicitly or let the training loop handle it
- Errors in intermediate stages may not surface until `mx.eval()` is called

#### 4.9.3 No `.requires_grad` Flag

Unlike PyTorch, MLX determines what needs gradients at `mx.grad()` call time, not at tensor creation time. The `rasterization()` function does not need to mark any tensors for gradient tracking. The caller wraps the function in `mx.grad()` or `mx.value_and_grad()`:

```python
def loss_fn(means, quats, scales, opacities, colors):
    render_colors, render_alphas, info = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height
    )
    return some_loss(render_colors, target)

# Compute loss + gradients in one pass
loss_and_grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
loss, grads = loss_and_grad_fn(means, quats, scales, opacities, colors)
```

#### 4.9.4 Memory Management

MLX shares unified memory between CPU and GPU on Apple Silicon. No explicit `.to(device)` calls are needed. However, for large scenes:
- The `channel_chunk` parameter limits per-pass memory for high-dimensional features
- Intermediate results in `info` hold references to lazy arrays; they occupy memory only when evaluated
- Users can delete `info` entries they don't need, then call `mx.eval()` on remaining tensors, to reduce peak memory

#### 4.9.5 Broadcasting Semantics

MLX follows NumPy broadcasting rules. Key broadcasting in this function:
- `opacities[None, :]` broadcasts `[N]` to `[1, N]`, then `mx.broadcast_to` expands to `[C, N]`
- `means[None, :, :]` broadcasts `[N, 3]` to `[1, N, 3]` for view direction computation
- `campos[:, None, :]` broadcasts `[C, 3]` to `[C, 1, 3]` for subtraction

---

## 5. Public API Exports

### 5.1 `src/gsplat_mlx/__init__.py`

Add the following exports to the package's `__init__.py`:

```python
from .rendering import (
    rasterization,
    RenderMode,
    RasterizeMode,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_expected_depth,
    render_mode_has_only_depth,
    render_mode_has_only_color,
    viewmat_to_camera_position,
    compute_view_directions,
)
```

### 5.2 Internal Imports in `rendering.py`

```python
import math
from typing import Dict, Literal, Optional, Tuple

import mlx.core as mx

from .core.projection import fully_fused_projection
from .core.sh import spherical_harmonics
from .core.intersection import isect_tiles, isect_offset_encode
from .core.rasterization import rasterize_to_pixels
```

**Note**: The exact import paths depend on how PRDs 03-08 organized their modules. The key contract is that these functions exist with the signatures defined in their respective PRDs. If the module structure differs, update the imports accordingly.

---

## 6. Test Plan

### File: `tests/test_rendering.py`

All tests use synthetic Gaussian scenes with known expected outputs. Tests are organized by category: end-to-end, output shape, info dict, gradients, edge cases, and validation.

#### 6.1 Test Fixture: Synthetic Scene

```python
import pytest
import mlx.core as mx


@pytest.fixture
def simple_scene():
    """A simple scene with 10 Gaussians in front of the camera.

    All Gaussians are at z~3.0 (in front of the camera at the origin),
    with small scales and moderate opacity. Suitable for basic rendering tests.
    """
    mx.random.seed(42)
    N = 10
    C = 1

    means = mx.random.normal((N, 3)) * 0.5
    # Push all Gaussians to z~3 (in front of camera looking down +z)
    means_list = []
    for i in range(N):
        means_list.append(mx.concatenate([
            means[i, :2],
            means[i, 2:3] + 3.0
        ]))
    means = mx.stack(means_list)

    quats = mx.random.normal((N, 4))
    quats = quats / mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))

    scales = mx.ones((N, 3)) * 0.1
    opacities = mx.ones((N,)) * 0.8
    colors_rgb = mx.random.uniform(shape=(N, 3))

    # SH coefficients: degree 0 (1 band, 1 coeff per color)
    colors_sh0 = mx.random.normal((N, 1, 3))
    # SH coefficients: degree 3 (16 coefficients per color)
    colors_sh3 = mx.random.normal((N, 16, 3))

    # Camera at origin looking down +z axis (identity viewmat)
    viewmats = mx.eye(4)[None, :, :]  # [1, 4, 4]

    # Camera intrinsics: focal length 200, principal point at (64, 64)
    Ks = mx.array([[[200.0, 0.0, 64.0],
                     [0.0, 200.0, 64.0],
                     [0.0, 0.0, 1.0]]])  # [1, 3, 3]
    width, height = 128, 128

    return {
        "means": means, "quats": quats, "scales": scales,
        "opacities": opacities, "colors_rgb": colors_rgb,
        "colors_sh0": colors_sh0, "colors_sh3": colors_sh3,
        "viewmats": viewmats, "Ks": Ks,
        "width": width, "height": height,
        "N": N, "C": C,
    }


@pytest.fixture
def multi_camera_scene(simple_scene):
    """Extend simple_scene to 3 cameras with different viewpoints."""
    scene = dict(simple_scene)
    C = 3

    # Camera 0: identity (looking down +z)
    vm0 = mx.eye(4)

    # Camera 1: small rotation around Y axis
    angle = 0.1
    cos_a, sin_a = float(mx.cos(mx.array(angle))), float(mx.sin(mx.array(angle)))
    vm1 = mx.array([
        [cos_a, 0.0, sin_a, 0.0],
        [0.0,   1.0, 0.0,   0.0],
        [-sin_a, 0.0, cos_a, 0.0],
        [0.0,   0.0, 0.0,   1.0],
    ])

    # Camera 2: translated 0.5 units to the right
    vm2 = mx.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    scene["viewmats"] = mx.stack([vm0, vm1, vm2], axis=0)  # [3, 4, 4]
    scene["Ks"] = mx.broadcast_to(scene["Ks"], (3, 3, 3))
    scene["C"] = C
    return scene
```

#### 6.2 End-to-End Tests

| Test | Description | Verifies |
|------|-------------|----------|
| `test_rasterization_rgb` | Basic RGB rendering with direct colors | Output shape `[C, H, W, 3]`, non-zero pixels exist, alpha in `[0, 1]` |
| `test_rasterization_sh_degree_0` | SH degree 0 (constant color per Gaussian) | Output shape correct, colors are view-independent |
| `test_rasterization_sh_degree_3` | SH degree 3 with 16 coefficients | Output shape correct, produces view-dependent colors |
| `test_rasterization_depth_only` | `render_mode="D"` | Output shape `[C, H, W, 1]`, depth values > 0 where alpha > 0 |
| `test_rasterization_expected_depth` | `render_mode="ED"` | Depth normalized by alpha, shape `[C, H, W, 1]` |
| `test_rasterization_rgb_depth` | `render_mode="RGB+D"` | Output shape `[C, H, W, 4]`, last channel is depth |
| `test_rasterization_rgb_expected_depth` | `render_mode="RGB+ED"` | Output shape `[C, H, W, 4]`, depth channel normalized |
| `test_rasterization_antialiased` | `rasterize_mode="antialiased"` | `info["compensations"]` is not None, opacities modified |
| `test_rasterization_background` | With `backgrounds` argument | Background visible where alpha < 1 |
| `test_rasterization_multi_camera` | C > 1 with different viewpoints | Each camera produces a different image |
| `test_rasterization_per_camera_colors` | Colors with shape `[C, N, D]` | Different colors per camera view |
| `test_rasterization_channel_chunk` | `channel_chunk=2` with D=8 features | Same result as `channel_chunk=32`, tests chunking logic |
| `test_rasterization_empty_scene` | N=0 Gaussians | Returns black image (or background), zero alpha |
| `test_rasterization_all_culled` | All Gaussians behind camera | Returns black image (or background), zero alpha |

#### 6.3 Output Shape Tests

```python
@pytest.mark.parametrize("render_mode,expected_channels", [
    ("RGB", 3),
    ("D", 1),
    ("ED", 1),
    ("RGB+D", 4),
    ("RGB+ED", 4),
])
def test_rasterization_output_shape(simple_scene, render_mode, expected_channels):
    """Verify output shapes for all render modes."""
    s = simple_scene
    render_colors, render_alphas, info = rasterization(
        s["means"], s["quats"], s["scales"], s["opacities"], s["colors_rgb"],
        s["viewmats"], s["Ks"], s["width"], s["height"],
        render_mode=render_mode,
    )
    mx.eval(render_colors, render_alphas)

    assert render_colors.shape == (s["C"], s["height"], s["width"], expected_channels)
    assert render_alphas.shape == (s["C"], s["height"], s["width"], 1)
```

#### 6.4 Info Dict Tests

```python
def test_rasterization_info_dict(simple_scene):
    """Verify info dict contains all expected keys with correct shapes."""
    s = simple_scene
    _, _, info = rasterization(
        s["means"], s["quats"], s["scales"], s["opacities"], s["colors_rgb"],
        s["viewmats"], s["Ks"], s["width"], s["height"],
    )

    # Required keys
    required_keys = {
        "radii", "means2d", "depths", "conics", "opacities",
        "tile_width", "tile_height", "tiles_per_gauss",
        "isect_ids", "flatten_ids", "isect_offsets",
        "width", "height", "tile_size", "n_cameras",
        "compensations",
    }
    assert required_keys.issubset(info.keys()), (
        f"Missing keys: {required_keys - info.keys()}"
    )

    # Shape checks for array values
    N, C = s["N"], s["C"]
    assert info["radii"].shape == (C, N)
    assert info["means2d"].shape == (C, N, 2)
    assert info["depths"].shape == (C, N)
    assert info["conics"].shape == (C, N, 3)
    assert info["opacities"].shape == (C, N)

    # Scalar checks
    assert info["width"] == s["width"]
    assert info["height"] == s["height"]
    assert info["tile_size"] == 16
    assert info["n_cameras"] == C

    # compensations should be None for classic mode
    assert info["compensations"] is None


def test_rasterization_info_dict_antialiased(simple_scene):
    """Verify compensations are present in antialiased mode."""
    s = simple_scene
    _, _, info = rasterization(
        s["means"], s["quats"], s["scales"], s["opacities"], s["colors_rgb"],
        s["viewmats"], s["Ks"], s["width"], s["height"],
        rasterize_mode="antialiased",
    )

    assert info["compensations"] is not None
    assert info["compensations"].shape == (s["C"], s["N"])
```

#### 6.5 Gradient Tests

These tests verify that gradients flow end-to-end through the rendering pipeline. Each test differentiates a simple scalar loss (sum of rendered pixels) with respect to one input parameter.

```python
def test_grad_means(simple_scene):
    """Gradient of rendered image w.r.t. 3D positions."""
    s = simple_scene

    def loss_fn(means):
        rc, ra, info = rasterization(
            means, s["quats"], s["scales"], s["opacities"], s["colors_rgb"],
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["means"])
    mx.eval(grads)

    assert grads.shape == s["means"].shape  # [N, 3]
    # At least some gradients should be non-zero (visible Gaussians)
    assert mx.any(grads != 0.0), "Expected non-zero gradients for visible Gaussians"


def test_grad_quats(simple_scene):
    """Gradient w.r.t. quaternion rotations."""
    s = simple_scene

    def loss_fn(quats):
        rc, ra, info = rasterization(
            s["means"], quats, s["scales"], s["opacities"], s["colors_rgb"],
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["quats"])
    mx.eval(grads)
    assert grads.shape == (s["N"], 4)
    assert mx.any(grads != 0.0)


def test_grad_scales(simple_scene):
    """Gradient w.r.t. scale parameters."""
    s = simple_scene

    def loss_fn(scales):
        rc, ra, info = rasterization(
            s["means"], s["quats"], scales, s["opacities"], s["colors_rgb"],
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["scales"])
    mx.eval(grads)
    assert grads.shape == (s["N"], 3)
    assert mx.any(grads != 0.0)


def test_grad_opacities(simple_scene):
    """Gradient w.r.t. opacity values."""
    s = simple_scene

    def loss_fn(opacities):
        rc, ra, info = rasterization(
            s["means"], s["quats"], s["scales"], opacities, s["colors_rgb"],
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["opacities"])
    mx.eval(grads)
    assert grads.shape == (s["N"],)
    assert mx.any(grads != 0.0)


def test_grad_colors_direct(simple_scene):
    """Gradient w.r.t. direct RGB colors."""
    s = simple_scene

    def loss_fn(colors):
        rc, ra, info = rasterization(
            s["means"], s["quats"], s["scales"], s["opacities"], colors,
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["colors_rgb"])
    mx.eval(grads)
    assert grads.shape == (s["N"], 3)
    assert mx.any(grads != 0.0)


def test_grad_colors_sh(simple_scene):
    """Gradient w.r.t. SH coefficients (degree 3)."""
    s = simple_scene

    def loss_fn(colors_sh):
        rc, ra, info = rasterization(
            s["means"], s["quats"], s["scales"], s["opacities"], colors_sh,
            s["viewmats"], s["Ks"], s["width"], s["height"],
            sh_degree=3,
        )
        return mx.sum(rc)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(s["colors_sh3"])
    mx.eval(grads)
    assert grads.shape == (s["N"], 16, 3)
    assert mx.any(grads != 0.0)
```

#### 6.6 Numerical Gradient Verification

Compares analytical gradients (from `mx.grad`) against numerical finite differences to catch backward pass bugs.

```python
def test_grad_numerical_vs_analytical(simple_scene):
    """Verify analytical gradients match numerical finite differences.

    Uses central differences: df/dx ~ (f(x+h) - f(x-h)) / (2h)
    Checks a few elements of the means gradient.
    """
    s = simple_scene
    eps = 1e-3

    def render_sum(means):
        rc, _, _ = rasterization(
            means, s["quats"], s["scales"], s["opacities"], s["colors_rgb"],
            s["viewmats"], s["Ks"], s["width"], s["height"],
        )
        return mx.sum(rc)

    # Analytical gradient
    grad_fn = mx.grad(render_sum)
    analytical = grad_fn(s["means"])
    mx.eval(analytical)

    # Numerical gradient (central differences) for a few elements
    import numpy as np
    means_np = np.array(s["means"])

    for i, j in [(0, 0), (0, 1), (0, 2), (1, 0)]:
        means_plus = means_np.copy()
        means_minus = means_np.copy()
        means_plus[i, j] += eps
        means_minus[i, j] -= eps

        f_plus = float(render_sum(mx.array(means_plus)))
        f_minus = float(render_sum(mx.array(means_minus)))
        numerical = (f_plus - f_minus) / (2 * eps)

        analytical_val = float(analytical[i, j])
        assert abs(analytical_val - numerical) < 0.1 * (abs(numerical) + 1e-5), (
            f"Gradient mismatch at ({i},{j}): "
            f"analytical={analytical_val:.6f}, numerical={numerical:.6f}"
        )
```

#### 6.7 Edge Case Tests

```python
def test_rasterization_single_gaussian():
    """Single Gaussian at known position produces predictable output."""
    means = mx.array([[0.0, 0.0, 3.0]])   # centered in front of camera
    quats = mx.array([[1.0, 0.0, 0.0, 0.0]])  # identity rotation
    scales = mx.array([[0.3, 0.3, 0.3]])
    opacities = mx.array([0.99])
    colors = mx.array([[1.0, 0.0, 0.0]])   # red

    viewmats = mx.eye(4)[None]
    Ks = mx.array([[[100.0, 0.0, 50.0],
                     [0.0, 100.0, 50.0],
                     [0.0, 0.0, 1.0]]])
    width, height = 100, 100

    rc, ra, info = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height,
    )
    mx.eval(rc, ra)

    # Center pixel (50, 50) should be approximately red
    center = rc[0, 50, 50]
    assert float(center[0]) > 0.5, "Center pixel red channel should be high"
    assert float(center[1]) < 0.1, "Center pixel green channel should be low"
    assert float(center[2]) < 0.1, "Center pixel blue channel should be low"

    # Alpha at center should be high
    assert float(ra[0, 50, 50, 0]) > 0.5, "Center alpha should be significant"

    # Corner pixels should have lower alpha (far from Gaussian center)
    assert float(ra[0, 0, 0, 0]) < float(ra[0, 50, 50, 0]), (
        "Corner alpha should be less than center alpha"
    )


def test_rasterization_no_gaussians():
    """Empty scene (N=0) renders as background."""
    means = mx.zeros((0, 3))
    quats = mx.zeros((0, 4))
    scales = mx.zeros((0, 3))
    opacities = mx.zeros((0,))
    colors = mx.zeros((0, 3))

    viewmats = mx.eye(4)[None]
    Ks = mx.array([[[100.0, 0.0, 50.0],
                     [0.0, 100.0, 50.0],
                     [0.0, 0.0, 1.0]]])

    bg = mx.array([[0.5, 0.5, 0.5]])
    rc, ra, info = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, 100, 100,
        backgrounds=bg,
    )
    mx.eval(rc, ra)

    # Should be background color everywhere
    assert mx.allclose(rc[0, 50, 50], mx.array([0.5, 0.5, 0.5]), atol=1e-5)
    # Alpha should be zero everywhere
    assert mx.allclose(ra, mx.zeros_like(ra), atol=1e-5)


def test_rasterization_behind_camera():
    """All Gaussians behind camera should produce empty image."""
    means = mx.array([[0.0, 0.0, -3.0]])  # behind camera (negative z)
    quats = mx.array([[1.0, 0.0, 0.0, 0.0]])
    scales = mx.array([[0.3, 0.3, 0.3]])
    opacities = mx.array([0.99])
    colors = mx.array([[1.0, 0.0, 0.0]])

    viewmats = mx.eye(4)[None]
    Ks = mx.array([[[100.0, 0.0, 50.0],
                     [0.0, 100.0, 50.0],
                     [0.0, 0.0, 1.0]]])

    rc, ra, info = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, 100, 100,
    )
    mx.eval(rc, ra)

    # Everything should be zero (black, no alpha)
    assert mx.allclose(ra, mx.zeros_like(ra), atol=1e-5)


def test_rasterization_high_dimensional_features():
    """Rendering with D=16 feature channels using channel chunking.

    Verifies that chunked rendering produces identical results to
    unchunked rendering for high-dimensional feature vectors.
    """
    mx.random.seed(123)
    N, C = 20, 1
    D = 16

    means = mx.random.normal((N, 3)) * 0.5
    means_list = []
    for i in range(N):
        means_list.append(mx.concatenate([means[i, :2], means[i, 2:3] + 3.0]))
    means = mx.stack(means_list)

    quats = mx.random.normal((N, 4))
    scales = mx.ones((N, 3)) * 0.1
    opacities = mx.ones((N,)) * 0.8
    colors = mx.random.uniform(shape=(N, D))

    viewmats = mx.eye(4)[None]
    Ks = mx.array([[[200.0, 0.0, 64.0],
                     [0.0, 200.0, 64.0],
                     [0.0, 0.0, 1.0]]])

    # Render with large chunk (no chunking)
    rc1, ra1, _ = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, 128, 128, channel_chunk=32,
    )

    # Render with small chunk (forces chunking into 4 passes)
    rc2, ra2, _ = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, 128, 128, channel_chunk=4,
    )

    mx.eval(rc1, rc2, ra1, ra2)

    assert rc1.shape == (1, 128, 128, D)
    assert rc2.shape == (1, 128, 128, D)
    assert mx.allclose(rc1, rc2, atol=1e-5), "Chunked rendering should match unchunked"
    assert mx.allclose(ra1, ra2, atol=1e-5), "Alpha should be identical across chunking"
```

#### 6.8 Validation Tests

```python
def test_rasterization_invalid_render_mode():
    """Invalid render_mode raises AssertionError."""
    with pytest.raises(AssertionError, match="render_mode"):
        rasterization(
            mx.zeros((1, 3)), mx.zeros((1, 4)), mx.zeros((1, 3)),
            mx.zeros((1,)), mx.zeros((1, 3)),
            mx.eye(4)[None], mx.eye(3)[None], 10, 10,
            render_mode="INVALID",
        )


def test_rasterization_invalid_rasterize_mode():
    """Invalid rasterize_mode raises AssertionError."""
    with pytest.raises(AssertionError, match="rasterize_mode"):
        rasterization(
            mx.zeros((1, 3)), mx.zeros((1, 4)), mx.zeros((1, 3)),
            mx.zeros((1,)), mx.zeros((1, 3)),
            mx.eye(4)[None], mx.eye(3)[None], 10, 10,
            rasterize_mode="wrong",
        )


def test_rasterization_sh_degree_mismatch():
    """SH degree requiring more bands than provided raises AssertionError."""
    N = 5
    colors_sh = mx.random.normal((N, 4, 3))  # K=4, supports up to degree 1

    with pytest.raises(AssertionError, match="sh_degree"):
        rasterization(
            mx.random.normal((N, 3)),
            mx.random.normal((N, 4)),
            mx.ones((N, 3)) * 0.1,
            mx.ones((N,)) * 0.8,
            colors_sh,
            mx.eye(4)[None],
            mx.array([[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]]),
            100, 100,
            sh_degree=3,  # requires K >= 16, but only K=4 provided
        )


def test_rasterization_shape_mismatch_means():
    """Wrong means shape raises AssertionError."""
    with pytest.raises(AssertionError, match="means shape"):
        rasterization(
            mx.zeros((5, 2)),  # should be (5, 3)
            mx.zeros((5, 4)),
            mx.zeros((5, 3)),
            mx.zeros((5,)),
            mx.zeros((5, 3)),
            mx.eye(4)[None],
            mx.eye(3)[None],
            10, 10,
        )
```

#### 6.9 Test Summary

| Category | Count | Description |
|----------|-------|-------------|
| End-to-end | 14 | Full pipeline rendering tests |
| Output shape | 5 | Parametrized shape verification |
| Info dict | 2 | Key presence and shape checks |
| Gradients | 6 | Per-parameter gradient flow |
| Numerical verification | 1 | Analytical vs finite difference |
| Edge cases | 4 | Empty scene, behind camera, single Gaussian, high-D |
| Validation | 4 | Invalid inputs raise errors |
| **Total** | **36** | |

---

## 7. Dependencies

### 7.1 Required PRDs (Must Be Complete)

| PRD | What It Provides | Used By |
|-----|-----------------|---------|
| PRD-01 | Dev environment, test harness, package structure | Package imports, pytest |
| PRD-02 | `_quat_to_rotmat`, `_quat_scale_to_matrix` | Used internally by PRD-03 |
| PRD-03 | `quat_scale_to_covar_preci` | Called internally by `fully_fused_projection` |
| PRD-04 | `spherical_harmonics` | Called directly in Stage 2 when `sh_degree` is set |
| PRD-05 | `fully_fused_projection` | Called in Stage 1 for 3D-to-2D projection |
| PRD-06 | `isect_tiles`, `isect_offset_encode` | Called in Stage 4 for tile intersection |
| PRD-07 | `rasterize_to_pixels` | Called in Stage 5 for final pixel rendering |

### 7.2 Optional PRDs (Not Required for MVP)

| PRD | What It Would Provide |
|-----|----------------------|
| PRD-08 | `accumulate` -- alternative to tile-based rasterization, useful for debugging |

### 7.3 Downstream Consumers

| PRD | How It Uses This PRD |
|-----|---------------------|
| PRD-10 | Strategy reads `info["means2d"]`, `info["radii"]`, `info["opacities"]` for densification decisions (split/clone/prune) |
| PRD-13 | Training loop calls `rasterization()` every iteration, uses `render_colors` and `render_alphas` for loss computation, uses `info` for densification |

---

## 8. Acceptance Criteria

### 8.1 Functional

- [ ] `rasterization()` can be imported from `gsplat_mlx`
- [ ] Basic RGB rendering produces a non-zero image for a simple synthetic scene
- [ ] SH-based colors (degrees 0, 1, 2, 3) produce valid rendered images
- [ ] All 5 render modes produce correct output shapes:
  - `"RGB"` -> `[C, H, W, 3]`
  - `"D"` -> `[C, H, W, 1]`
  - `"ED"` -> `[C, H, W, 1]`
  - `"RGB+D"` -> `[C, H, W, 4]`
  - `"RGB+ED"` -> `[C, H, W, 4]`
- [ ] Antialiased mode applies compensation factors and produces different opacities from classic mode
- [ ] Background colors are correctly blended where alpha < 1
- [ ] Multi-camera rendering (C > 1) produces distinct per-camera images
- [ ] Per-camera colors `[C, N, D]` are handled correctly
- [ ] Channel chunking produces identical results to non-chunked rendering
- [ ] Empty scene (N=0) and fully-culled scenes produce black/background images
- [ ] Expected depth modes (ED, RGB+ED) correctly normalize depth by alpha

### 8.2 Info Dictionary

- [ ] Info dict contains all 16 required keys: `radii`, `means2d`, `depths`, `conics`, `opacities`, `compensations`, `tile_width`, `tile_height`, `tiles_per_gauss`, `isect_ids`, `flatten_ids`, `isect_offsets`, `width`, `height`, `tile_size`, `n_cameras`
- [ ] All info dict array values have correct shapes as specified in Section 4.7
- [ ] `compensations` is `None` for classic mode, `[C, N]` for antialiased mode

### 8.3 Differentiability

- [ ] Gradients flow through to `means` (3D positions)
- [ ] Gradients flow through to `quats` (rotations)
- [ ] Gradients flow through to `scales`
- [ ] Gradients flow through to `opacities`
- [ ] Gradients flow through to `colors` (direct RGB)
- [ ] Gradients flow through to SH coefficients when `sh_degree` is set
- [ ] Analytical gradients approximately match numerical finite differences (within 10% relative error or absolute tolerance 1e-3)

### 8.4 Validation

- [ ] Invalid `render_mode` raises `AssertionError` with informative message
- [ ] Invalid `rasterize_mode` raises `AssertionError` with informative message
- [ ] SH degree exceeding available bands raises `AssertionError`
- [ ] Mismatched input shapes raise `AssertionError` with shape details

### 8.5 Tests

- [ ] All 36 tests in `tests/test_rendering.py` pass with `pytest tests/test_rendering.py -v`
- [ ] No individual test takes longer than 30 seconds on Apple Silicon (M1 or later)

---

## 9. Implementation Notes

### 9.1 Implementation Order

1. **Helpers first**: Implement `render_mode_*` functions, `viewmat_to_camera_position`, `compute_view_directions`, `_normalize_colors`. These are pure functions with no upstream dependencies beyond MLX.
2. **Core function skeleton**: Implement `rasterization()` with input validation and the info dict. Use placeholder values (zeros) for outputs to verify the function structure.
3. **Wire up projection**: Replace the projection stub with `fully_fused_projection()` from PRD-05. Verify that `radii`, `means2d`, `depths`, `conics` have correct shapes.
4. **Wire up SH**: Add the SH evaluation branch. Verify that SH degree 0 produces view-independent colors.
5. **Wire up intersection**: Add `isect_tiles` and `isect_offset_encode` from PRD-06. Verify that `isect_offsets` shape matches expected tile grid.
6. **Wire up rasterization**: Add `rasterize_to_pixels` from PRD-07. Verify that a single red Gaussian at image center produces the expected image.
7. **Add depth modes**: Implement depth channel preparation (Stage 3) and expected depth normalization (Stage 6).
8. **Add channel chunking**: Implement the chunked rendering loop. Verify identical results with and without chunking.
9. **Add gradient tests**: Verify end-to-end differentiability for all parameters.
10. **Polish**: Error messages, docstrings, edge cases, empty scene handling.

### 9.2 Debugging Tips

- Use `render_mode="D"` to debug projection issues -- depth should increase with distance from camera
- Use a single Gaussian at a known position to verify projection + rasterization alignment
- Compare intermediate results (`means2d`, `depths`, `conics`) against upstream torch values
- If gradients are zero, check that `rasterize_to_pixels` backward is correctly implemented (PRD-07)
- Use `mx.eval()` liberally during debugging to force lazy computation and surface errors early
- Print `info["radii"]` to verify that Gaussians are not being culled unexpectedly
- For SH debugging, start with `sh_degree=0` (constant color) before testing higher degrees

### 9.3 Known Differences from Upstream

| Aspect | Upstream (gsplat) | Our Implementation (MVP) |
|--------|-------------------|--------------------------|
| Packed mode | Supported (default=True) | Not supported |
| Batch dims | Arbitrary leading dims `[...]` | None (means is `[N, 3]`) |
| Camera models | pinhole, ortho, fisheye, ftheta, lidar | pinhole only |
| Distributed | Multi-GPU via `torch.distributed` | Not applicable (single-device Apple Silicon) |
| absgrad | Supported via custom autograd | Deferred to PRD-10 integration |
| Sparse grad | With packed mode | Not supported |
| `covars` input | Direct covariance matrix bypass | Not supported (use quats + scales) |
| Activation | Caller's responsibility | Caller's responsibility (same) |
| SH offset | `colors + 0.5`, then `clamp_min(0)` | Same behavior |
| Extra signals | Separate rendering path | Not supported |

**Note on activations**: Unlike the simplified pipeline flow in the existing PRD-09, this implementation does NOT apply `exp(scales)` or `sigmoid(opacities)` internally. The caller is responsible for providing pre-activated values, matching upstream gsplat behavior where these activations are applied in the training loop before calling `rasterization()`.

### 9.4 Performance Expectations (MVP)

The MVP implementation prioritizes correctness over speed. Expected performance on M1/M2 Pro:

| Scene Size | Resolution | Expected Time (Forward) | Notes |
|-----------|------------|------------------------|-------|
| N=1,000 | 256x256 | < 1 second | Small scene, fast enough for iteration |
| N=10,000 | 512x512 | < 10 seconds | Medium scene |
| N=100,000 | 800x800 | < 60 seconds | Large scene, PRD-14 will optimize |

PRD-14 (Metal shaders) will replace the Python rasterization loop with GPU compute shaders, bringing performance to real-time speeds. The Python reference implementation serves as a correctness baseline.

---

## 10. Future Extensions (Post-MVP)

These features can be added incrementally without changing the core function signature:

1. **Packed mode** (`packed=True`): CSR-format intermediate tensors for memory efficiency with large scenes where each camera sees a small subset of Gaussians. Add `packed`, `sparse_grad` parameters.

2. **Batch dimensions**: Support leading batch dims on `means` (e.g., `[B, N, 3]`) for batched training scenarios.

3. **Covariance input**: Support `covars` parameter as alternative to `quats`/`scales` for when covariances are pre-computed.

4. **absgrad**: Add `absgrad=True` to compute absolute gradients of `means2d` for densification strategy (PRD-10). This requires wrapping `means2d` in a custom function that computes `abs(grad)` during backward.

5. **Additional camera models**: `"ortho"` (orthographic), `"fisheye"` (equidistant) via new projection paths in PRD-05.

6. **Hit distance modes**: `"d"`, `"Ed"`, `"RGB-d"`, `"RGB-Ed"` for along-ray distance computation (requires eval3d).

7. **Extra signals**: Additional per-Gaussian features rendered separately and returned in `info["render_extra_signals"]`.

8. **Metal shaders** (PRD-14): Replace Python rasterization loop with Metal compute shaders for real-time performance.

9. **2DGS support** (PRD-12): Alternative projection and rasterization for 2D Gaussian Splatting.
