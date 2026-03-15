"""High-level rendering API for gsplat-mlx.

Provides the ``rasterization()`` function -- the single public entry point
that users call to render 3D Gaussians into images.  This function
orchestrates the full pipeline:

1. Convert quaternions + scales to covariance matrices
2. Project 3D Gaussians to 2D screen space
3. Evaluate spherical harmonics for view-dependent color (when using SH)
4. Compute tile-Gaussian intersections
5. Rasterize pixels via alpha compositing

Port of ``gsplat/rendering.py:rasterization()`` from PyTorch to Apple MLX.
See PRD-09 for details.
"""

import math
from typing import Any, Dict, Literal, Optional, Tuple

import mlx.core as mx

from gsplat_mlx.core.cameras import CameraModel
from gsplat_mlx.core.covariance import quat_scale_to_covar_preci
from gsplat_mlx.core.intersection import isect_offset_encode, isect_tiles
from gsplat_mlx.core.projection import fully_fused_projection
from gsplat_mlx.core.rasterization import rasterize_to_pixels
from gsplat_mlx.core.rasterization_mlx import rasterize_to_pixels_mlx
from gsplat_mlx.core.spherical_harmonics import spherical_harmonics
from gsplat_mlx.core_2dgs.projection_2dgs import fully_fused_projection_2dgs
from gsplat_mlx.core_2dgs.rasterization_2dgs import rasterize_to_pixels_2dgs

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RenderMode = Literal[
    "RGB", "D", "ED", "RGB+D", "RGB+ED",
    "d", "Ed", "RGB+d", "RGB+Ed",
]
"""Supported rendering modes.

- ``"RGB"``: Render only colour channels.
- ``"D"``: Render accumulated projection depth (z-depth) only.
- ``"ED"``: Render expected (normalised) projection depth only.
- ``"RGB+D"``: Render colour with accumulated projection depth as an extra channel.
- ``"RGB+ED"``: Render colour with expected projection depth as an extra channel.
- ``"d"``: Render accumulated hit distance (ray distance) only.
- ``"Ed"``: Render expected (normalised) hit distance only.
- ``"RGB+d"``: Render colour with accumulated hit distance as an extra channel.
- ``"RGB+Ed"``: Render colour with expected hit distance as an extra channel.
"""

RasterizeMode = Literal["classic", "antialiased"]
"""Rasterization strategy.

- ``"classic"``: Standard alpha compositing.
- ``"antialiased"``: Multiply opacities by compensation factors from
  Mip-Splatting for alias-free rendering.
"""


# ---------------------------------------------------------------------------
# Render-mode helpers
# ---------------------------------------------------------------------------


def render_mode_has_color(mode: str) -> bool:
    """Return True if the render mode includes colour channels.

    Args:
        mode: A render mode string (e.g. ``"RGB"``, ``"D"``, ``"RGB+D"``).

    Returns:
        Whether the mode produces colour output.

    Examples:
        >>> render_mode_has_color("RGB")
        True
        >>> render_mode_has_color("D")
        False
    """
    return mode in {"RGB", "RGB+D", "RGB+ED", "RGB+d", "RGB+Ed"}


def render_mode_has_depth(mode: str) -> bool:
    """Return True if the render mode includes a projection depth (z-depth) channel.

    Args:
        mode: A render mode string.

    Returns:
        Whether the mode produces projection depth output.

    Examples:
        >>> render_mode_has_depth("RGB+D")
        True
        >>> render_mode_has_depth("RGB")
        False
    """
    return mode in {"D", "ED", "RGB+D", "RGB+ED"}


def render_mode_has_expected_depth(mode: str) -> bool:
    """Return True if the render mode uses expected (normalised) depth or hit distance.

    Args:
        mode: A render mode string.

    Returns:
        Whether the depth/hit-distance channel should be normalised by accumulated alpha.

    Examples:
        >>> render_mode_has_expected_depth("ED")
        True
        >>> render_mode_has_expected_depth("D")
        False
    """
    return mode in {"ED", "RGB+ED", "Ed", "RGB+Ed"}


def render_mode_has_hit_distance(mode: str) -> bool:
    """Return True if the render mode includes a hit-distance channel.

    Hit distance is the Euclidean (ray) distance from the camera origin
    to each Gaussian, as opposed to projection depth which is the
    z-component in camera space.

    Args:
        mode: A render mode string.

    Returns:
        Whether the mode produces hit-distance output.

    Examples:
        >>> render_mode_has_hit_distance("d")
        True
        >>> render_mode_has_hit_distance("D")
        False
        >>> render_mode_has_hit_distance("RGB+Ed")
        True
    """
    return mode in {"d", "Ed", "RGB+d", "RGB+Ed"}


def render_mode_has_only_depth_channel(mode: str) -> bool:
    """Return True if the render mode produces only a depth/hit-distance channel.

    These are the modes that do *not* include RGB colour, outputting only
    a single depth or hit-distance channel.

    Args:
        mode: A render mode string.

    Returns:
        Whether the mode has only a depth/hit-distance channel (no colour).

    Examples:
        >>> render_mode_has_only_depth_channel("D")
        True
        >>> render_mode_has_only_depth_channel("d")
        True
        >>> render_mode_has_only_depth_channel("RGB+D")
        False
    """
    return mode in {"D", "ED", "d", "Ed"}


# ---------------------------------------------------------------------------
# View-direction computation
# ---------------------------------------------------------------------------


def _viewmat_to_campos(viewmats: mx.array) -> mx.array:
    """Extract camera position in world coords from a world-to-camera matrix.

    For ``V = [R | t; 0 1]``, camera position is ``-R^T t``.

    Args:
        viewmats: World-to-camera matrices. Shape ``[..., 4, 4]``.

    Returns:
        Camera positions. Shape ``[..., 3]``.
    """
    R = viewmats[..., :3, :3]  # [..., 3, 3]
    t = viewmats[..., :3, 3]  # [..., 3]
    # campos = -R^T @ t
    return -mx.einsum("...ji,...j->...i", R, t)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def rasterization(
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[mx.array] = None,
    render_mode: RenderMode = "RGB",
    rasterize_mode: RasterizeMode = "classic",
    camera_model: "CameraModel" = "pinhole",
    differentiable: bool = True,
) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
    """Render 3D Gaussians to images.

    The main entry point for Gaussian Splatting rendering.  Orchestrates:

    1. Covariance computation from quaternion + scale parameters
    2. 3D-to-2D projection with camera model support
    3. Spherical harmonics evaluation (if SH coefficients provided)
    4. Tile-Gaussian intersection and depth sorting
    5. Per-pixel alpha compositing

    Args:
        means: 3D Gaussian centres. Shape ``[N, 3]``.
        quats: Quaternion rotations ``(w, x, y, z)``. Shape ``[N, 4]``.
        scales: Scale factors. Shape ``[N, 3]``.
        opacities: Per-Gaussian opacity in ``[0, 1]``. Shape ``[N]``.
        colors: Either SH coefficients ``[N, K, 3]`` (when ``sh_degree``
            is not ``None``) or direct RGB values ``[C, N, 3]`` (when
            ``sh_degree is None``).
        viewmats: World-to-camera transformation matrices. ``[C, 4, 4]``.
        Ks: Camera intrinsic matrices. ``[C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping plane distance. Default ``0.01``.
        far_plane: Far clipping plane distance. Default ``1e10``.
        eps2d: Regularisation added to the 2D covariance diagonal.
            Default ``0.3``.
        sh_degree: SH degree to activate.  ``None`` means direct colour
            mode (no SH evaluation).
        tile_size: Tile side length for rasterisation. Default ``16``.
        backgrounds: Per-camera background colour. ``[C, 3]`` or ``None``.
        render_mode: One of ``"RGB"``, ``"D"``, ``"ED"``, ``"RGB+D"``,
            ``"RGB+ED"``, ``"d"``, ``"Ed"``, ``"RGB+d"``, ``"RGB+Ed"``.
            Lowercase ``d`` modes use hit distance (ray distance) instead
            of projection depth (z-depth).  Default ``"RGB"``.
        rasterize_mode: ``"classic"`` or ``"antialiased"``.
            Default ``"classic"``.
        camera_model: ``"pinhole"``, ``"ortho"``, or ``"fisheye"``.
            Default ``"pinhole"``.
        differentiable: If ``True`` (default), use the Tier-2 pure-MLX
            differentiable rasterizer that preserves the computation graph
            for ``mx.grad()``.  If ``False``, use the Tier-1 NumPy
            reference rasterizer (faster for validation but NOT
            differentiable).

    Returns:
        A tuple ``(render_colors, render_alphas, info)``:

        - **render_colors**: Rendered images.
          Shape ``[C, H, W, D]`` where ``D`` depends on ``render_mode``:
          3 for ``"RGB"``, 1 for ``"D"``/``"ED"``, 4 for ``"RGB+D"``/``"RGB+ED"``.
        - **render_alphas**: Accumulated per-pixel opacity.
          Shape ``[C, H, W, 1]``.
        - **info**: Dictionary of intermediate results for densification
          strategies and debugging.

    Raises:
        ValueError: If ``render_mode``, ``rasterize_mode``, or
            ``camera_model`` is invalid.

    Examples:
        >>> import mlx.core as mx
        >>> import gsplat_mlx as gsplat
        >>> N, C = 100, 1
        >>> means = mx.random.normal((N, 3))
        >>> quats = mx.random.normal((N, 4))
        >>> scales = mx.random.uniform(shape=(N, 3)) * 0.1
        >>> opacities = mx.sigmoid(mx.random.normal((N,)))
        >>> colors = mx.random.normal((N, 1, 3)) * 0.1  # SH degree 0
        >>> viewmats = mx.eye(4)[None, ...]  # [1, 4, 4]
        >>> Ks = mx.array([[[50., 0., 32.],
        ...                  [0., 50., 32.],
        ...                  [0., 0., 1.]]])  # [1, 3, 3]
        >>> imgs, alphas, info = gsplat.rasterization(
        ...     means, quats, scales, opacities, colors,
        ...     viewmats, Ks, width=64, height=64, sh_degree=0,
        ... )
    """
    # ---- Input validation ----
    valid_render_modes = {
        "RGB", "D", "ED", "RGB+D", "RGB+ED",
        "d", "Ed", "RGB+d", "RGB+Ed",
    }
    if render_mode not in valid_render_modes:
        raise ValueError(
            f"Invalid render_mode '{render_mode}'. "
            f"Must be one of {valid_render_modes}."
        )
    valid_rasterize_modes = {"classic", "antialiased"}
    if rasterize_mode not in valid_rasterize_modes:
        raise ValueError(
            f"Invalid rasterize_mode '{rasterize_mode}'. "
            f"Must be one of {valid_rasterize_modes}."
        )
    valid_camera_models = {"pinhole", "ortho", "fisheye"}
    if camera_model not in valid_camera_models:
        raise ValueError(
            f"Invalid camera_model '{camera_model}'. "
            f"Must be one of {valid_camera_models}."
        )

    N = means.shape[0]
    C = viewmats.shape[0]

    # ---- Step 1: Covariance from quaternions + scales ----
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False,
    )  # [N, 3, 3]

    # ---- Step 2: Fully-fused projection ----
    calc_compensations = rasterize_mode == "antialiased"
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )
    # radii: [C, N, 2], means2d: [C, N, 2], depths: [C, N], conics: [C, N, 3]

    # ---- Step 3: Opacities (broadcast to [C, N]) ----
    # Broadcast scalar per-Gaussian opacities to per-camera
    opacities_cn = mx.broadcast_to(
        mx.expand_dims(opacities, axis=0), (C, N)
    )  # [C, N]

    # Apply antialiased compensation
    if compensations is not None:
        opacities_cn = opacities_cn * compensations  # [C, N]

    # ---- Step 4: Colour computation ----
    if sh_degree is not None:
        # SH mode: colors shape [N, K, 3]
        # Compute view directions: mean_position - camera_position
        campos = _viewmat_to_campos(viewmats)  # [C, 3]

        # dirs: [C, N, 3]
        dirs = means[None, :, :] - campos[:, None, :]

        # Normalise directions
        dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
        dirs = dirs / mx.maximum(dirs_norm, mx.array(1e-8))

        # Evaluate SH for each camera-Gaussian pair
        # coeffs: [N, K, 3] -> broadcast to [C, N, K, 3]
        coeffs_broadcast = mx.broadcast_to(
            mx.expand_dims(colors, axis=0),
            (C, N, colors.shape[-2], 3),
        )  # [C, N, K, 3]

        # Compute colours per camera-Gaussian pair
        rgb = spherical_harmonics(sh_degree, dirs, coeffs_broadcast)  # [C, N, 3]

        # Clamp: make sure colors >= 0 (add 0.5 bias like upstream)
        rgb = mx.maximum(rgb + 0.5, 0.0)  # [C, N, 3]
        colors_for_raster = rgb
    else:
        # Direct colour mode: colors shape [C, N, D] or [N, D]
        if colors.ndim == 2:
            # [N, D] -> [C, N, D]
            colors_for_raster = mx.broadcast_to(
                mx.expand_dims(colors, axis=0), (C, N, colors.shape[-1])
            )
        else:
            colors_for_raster = colors  # already [C, N, D]

    # ---- Step 5: Handle depth / hit-distance channels ----
    has_color = render_mode_has_color(render_mode)
    has_depth = render_mode_has_depth(render_mode)
    has_hit_dist = render_mode_has_hit_distance(render_mode)

    # Compute hit distances when needed: Euclidean distance from camera to
    # each Gaussian in camera space.  means_c is [C, N, 3] and the camera
    # origin in camera space is (0,0,0), so hit_dists = ||means_c||.
    if has_hit_dist:
        # Re-derive camera-space means (cheaper than modifying projection API)
        R = viewmats[:, :3, :3]  # [C, 3, 3]
        t = viewmats[:, :3, 3]   # [C, 3]
        means_c = mx.einsum("cij,nj->cni", R, means) + t[:, None, :]  # [C, N, 3]
        hit_dists = mx.sqrt(mx.sum(means_c * means_c, axis=-1))  # [C, N]

    # Select the distance channel to use for rasterization
    if has_depth:
        dist_channel = depths  # z-depth
    elif has_hit_dist:
        dist_channel = hit_dists  # ray distance
    else:
        dist_channel = None

    if has_color and dist_channel is not None:
        # Append depth/hit-distance as extra channel: [C, N, D+1]
        colors_for_raster = mx.concatenate(
            [colors_for_raster, dist_channel[..., None]], axis=-1
        )
        if backgrounds is not None:
            backgrounds = mx.concatenate(
                [backgrounds, mx.zeros((C, 1), dtype=backgrounds.dtype)],
                axis=-1,
            )
    elif dist_channel is not None and not has_color:
        # Depth/hit-distance only: [C, N, 1]
        colors_for_raster = dist_channel[..., None]
        if backgrounds is not None:
            backgrounds = mx.zeros((C, 1), dtype=backgrounds.dtype)

    # ---- Step 6: Tile-Gaussian intersection ----
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )

    # ---- Step 7: Offset encoding ----
    I = C  # number of images (no batch dims for MVP)
    isect_offsets = isect_offset_encode(
        isect_ids, I, tile_width, tile_height,
    )  # [C, tile_height, tile_width]

    # ---- Step 8: Rasterise to pixels ----
    _rasterize_fn = rasterize_to_pixels_mlx if differentiable else rasterize_to_pixels
    render_colors, render_alphas = _rasterize_fn(
        means2d,
        conics,
        colors_for_raster,
        opacities_cn,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    # render_colors: [C, H, W, D], render_alphas: [C, H, W, 1]

    # ---- Step 9: Post-process depth / hit-distance ----
    if render_mode_has_expected_depth(render_mode):
        # Normalise accumulated depth/hit-distance by alpha to get expected value
        if has_color:
            # Depth is the last channel
            color_channels = render_colors[..., :-1]
            depth_channel = render_colors[..., -1:]
            depth_channel = depth_channel / mx.maximum(render_alphas, mx.array(1e-10))
            render_colors = mx.concatenate([color_channels, depth_channel], axis=-1)
        else:
            # All channels are depth
            render_colors = render_colors / mx.maximum(render_alphas, mx.array(1e-10))

    # ---- Step 10: Build info dict ----
    info: Dict[str, Any] = {
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "radii": radii,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_offsets": isect_offsets,
        "flatten_ids": flatten_ids,
    }

    return render_colors, render_alphas, info


# ---------------------------------------------------------------------------
# 2D Gaussian Splatting (surfel) rendering entry point
# ---------------------------------------------------------------------------


def rasterization_2dgs(
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array,
    viewmats: mx.array,
    Ks: mx.array,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    tile_size: int = 16,
    backgrounds: Optional[mx.array] = None,
    sh_degree: Optional[int] = None,
) -> Tuple[mx.array, mx.array, mx.array, Dict[str, Any]]:
    """Render 2D Gaussian surfels to images.

    High-level entry point for 2D Gaussian Splatting. Orchestrates:

    1. Surfel projection (ray-transform matrices, normals, screen bounds)
    2. Optional spherical harmonics evaluation for view-dependent colour
    3. Tile-surfel intersection and depth sorting
    4. Per-pixel rasterization with normal accumulation

    Args:
        means: 3D surfel centres. Shape ``[N, 3]``.
        quats: Quaternion orientations ``(w, x, y, z)``. Shape ``[N, 4]``.
        scales: Scale factors. Shape ``[N, 3]``.
        opacities: Per-surfel opacity in ``[0, 1]``. Shape ``[N]``.
        colors: Either SH coefficients ``[N, K, 3]`` (when ``sh_degree``
            is not ``None``) or direct RGB values ``[C, N, 3]`` (when
            ``sh_degree is None``).
        viewmats: World-to-camera matrices. ``[C, 4, 4]``.
        Ks: Camera intrinsic matrices. ``[C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping plane distance. Default ``0.01``.
        far_plane: Far clipping plane distance. Default ``1e10``.
        tile_size: Tile side length for rasterisation. Default ``16``.
        backgrounds: Per-camera background colour. ``[C, 3]`` or ``None``.
        sh_degree: SH degree to activate. ``None`` means direct colour mode.

    Returns:
        A tuple ``(render_colors, render_alphas, render_normals, info)``:

        - **render_colors**: Rendered colour images. ``[C, H, W, channels]``.
        - **render_alphas**: Accumulated opacity. ``[C, H, W, 1]``.
        - **render_normals**: Accumulated normal map. ``[C, H, W, 3]``.
        - **info**: Dictionary of intermediate results for debugging.
    """
    N = means.shape[0]
    C = viewmats.shape[0]

    # ---- Step 1: Surfel projection ----
    radii, means2d, depths, ray_transforms, normals = (
        fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
        )
    )

    # ---- Step 2: Opacities (broadcast to [C, N]) ----
    opacities_cn = mx.broadcast_to(
        mx.expand_dims(opacities, axis=0), (C, N)
    )

    # ---- Step 3: Colour computation ----
    if sh_degree is not None:
        campos = _viewmat_to_campos(viewmats)
        dirs = means[None, :, :] - campos[:, None, :]
        dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
        dirs = dirs / mx.maximum(dirs_norm, mx.array(1e-8))

        coeffs_broadcast = mx.broadcast_to(
            mx.expand_dims(colors, axis=0),
            (C, N, colors.shape[-2], 3),
        )
        rgb = spherical_harmonics(sh_degree, dirs, coeffs_broadcast)
        rgb = mx.maximum(rgb + 0.5, 0.0)
        colors_for_raster = rgb
    else:
        if colors.ndim == 2:
            colors_for_raster = mx.broadcast_to(
                mx.expand_dims(colors, axis=0), (C, N, colors.shape[-1])
            )
        else:
            colors_for_raster = colors

    # ---- Step 4: Tile-surfel intersection ----
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )

    # ---- Step 5: Offset encoding ----
    isect_offsets = isect_offset_encode(
        isect_ids, C, tile_width, tile_height,
    )

    # ---- Step 6: Rasterise to pixels ----
    render_colors, render_alphas, render_normals = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors_for_raster,
        normals,
        opacities_cn,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    # ---- Step 7: Build info dict ----
    info: Dict[str, Any] = {
        "means2d": means2d,
        "depths": depths,
        "radii": radii,
        "ray_transforms": ray_transforms,
        "normals": normals,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_offsets": isect_offsets,
        "flatten_ids": flatten_ids,
    }

    return render_colors, render_alphas, render_normals, info
