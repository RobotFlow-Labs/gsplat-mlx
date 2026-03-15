"""3D-to-2D Gaussian projection pipeline for gsplat-mlx.

Port of gsplat/cuda/_torch_impl.py projection functions to Apple MLX.
Implements world-to-camera transform, perspective/fisheye/orthographic projection,
and the fully-fused projection entry point.

Upstream reference: gsplat/cuda/_torch_impl.py lines 31-327
See PRD-05 for details.
"""

from typing import Optional, Tuple

import mlx.core as mx

from gsplat_mlx.core.cameras import CameraModel


# ---------------------------------------------------------------------------
# World-to-Camera Transform
# ---------------------------------------------------------------------------


def world_to_cam(
    means: mx.array,      # [..., N, 3]
    covars: mx.array,     # [..., N, 3, 3]
    viewmats: mx.array,   # [..., C, 4, 4]
) -> Tuple[mx.array, mx.array]:
    """Transform Gaussian means and covariances from world to camera coordinates.

    Args:
        means: Gaussian means in world coordinates. Shape ``[..., N, 3]``.
        covars: Gaussian covariances in world coordinates. Shape ``[..., N, 3, 3]``.
        viewmats: World-to-camera transformation matrices. Shape ``[..., C, 4, 4]``.

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinates. ``[..., C, N, 3]``.
        - **covars_c**: Gaussian covariances in camera coordinates. ``[..., C, N, 3, 3]``.
    """
    R = viewmats[..., :3, :3]   # [..., C, 3, 3]
    t = viewmats[..., :3, 3]    # [..., C, 3]

    # means_c = R @ means^T + t  =>  [..., C, N, 3]
    means_c = (
        mx.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )

    # covars_c = R @ covars @ R^T  =>  [..., C, N, 3, 3]
    covars_c = mx.einsum(
        "...cij,...njk,...clk->...cnil", R, covars, R
    )

    return means_c, covars_c


# ---------------------------------------------------------------------------
# Perspective (Pinhole) Projection
# ---------------------------------------------------------------------------


def persp_proj(
    means: mx.array,      # [..., C, N, 3]
    covars: mx.array,     # [..., C, N, 3, 3]
    Ks: mx.array,         # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Perspective projection of 3D Gaussians using the EWA Jacobian.

    Args:
        means: Gaussian means in camera coordinates. Shape ``[..., C, N, 3]``.
        covars: Gaussian covariances in camera coordinates. Shape ``[..., C, N, 3, 3]``.
        Ks: Camera intrinsic matrices. Shape ``[..., C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A tuple:

        - **means2d**: Projected 2D means. ``[..., C, N, 2]``.
        - **cov2d**: Projected 2D covariances. ``[..., C, N, 2, 2]``.
    """
    tx, ty, tz = means[..., 0], means[..., 1], means[..., 2]  # [..., C, N]
    tz2 = tz ** 2

    fx = Ks[..., 0, 0, None]   # [..., C, 1]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]

    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    tx = tz * mx.clip(tx / tz, -lim_x_neg, lim_x_pos)
    ty = tz * mx.clip(ty / tz, -lim_y_neg, lim_y_pos)

    batch_dims = means.shape[:-3]
    C, N = means.shape[-3], means.shape[-2]
    O = mx.zeros(batch_dims + (C, N), dtype=means.dtype)

    J = mx.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], axis=-1
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    means2d = mx.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], means
    )  # [..., C, N, 2]
    means2d = means2d / tz[..., None]

    return means2d, cov2d


# ---------------------------------------------------------------------------
# Fisheye (Equidistant) Projection
# ---------------------------------------------------------------------------


def fisheye_proj(
    means: mx.array,      # [..., C, N, 3]
    covars: mx.array,     # [..., C, N, 3, 3]
    Ks: mx.array,         # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Fisheye equidistant projection of 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinates. Shape ``[..., C, N, 3]``.
        covars: Gaussian covariances in camera coordinates. Shape ``[..., C, N, 3, 3]``.
        Ks: Camera intrinsic matrices. Shape ``[..., C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A tuple:

        - **means2d**: Projected 2D means. ``[..., C, N, 2]``.
        - **cov2d**: Projected 2D covariances. ``[..., C, N, 2, 2]``.
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3], means.shape[-2]

    x, y, z = means[..., 0], means[..., 1], means[..., 2]

    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]

    eps = 1e-7
    xy_len = (x ** 2 + y ** 2) ** 0.5 + eps
    theta = mx.arctan2(xy_len, z + eps)

    means2d = mx.stack(
        [
            x * fx * theta / xy_len + cx,
            y * fy * theta / xy_len + cy,
        ],
        axis=-1,
    )

    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = mx.arctan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / x2y2

    J = mx.stack(
        [
            fx * (x2 * a + y2 * b),
            fx * xy * (a - b),
            -fx * x * x2y2z2_inv,
            fy * xy * (a - b),
            fy * (y2 * a + x2 * b),
            -fy * y * x2y2z2_inv,
        ],
        axis=-1,
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    return means2d, cov2d


# ---------------------------------------------------------------------------
# Orthographic Projection
# ---------------------------------------------------------------------------


def ortho_proj(
    means: mx.array,      # [..., C, N, 3]
    covars: mx.array,     # [..., C, N, 3, 3]
    Ks: mx.array,         # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[mx.array, mx.array]:
    """Orthographic projection of 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinates. Shape ``[..., C, N, 3]``.
        covars: Gaussian covariances in camera coordinates. Shape ``[..., C, N, 3, 3]``.
        Ks: Camera intrinsic matrices. Shape ``[..., C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A tuple:

        - **means2d**: Projected 2D means. ``[..., C, N, 2]``.
        - **cov2d**: Projected 2D covariances. ``[..., C, N, 2, 2]``.
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3], means.shape[-2]

    fx = Ks[..., 0, 0, None]   # [..., C, 1]
    fy = Ks[..., 1, 1, None]

    O = mx.zeros(batch_dims + (C, 1), dtype=means.dtype)
    J = (
        mx.stack([fx, O, O, O, fy, O], axis=-1)
        .reshape(batch_dims + (C, 1, 2, 3))
    )
    # Broadcast J to [..., C, N, 2, 3]
    J = mx.broadcast_to(J, batch_dims + (C, N, 2, 3))

    cov2d = mx.einsum("...ij,...jk,...lk->...il", J, covars, J)

    # means2d = means_xy * diag(fx, fy) + (cx, cy)
    diag_f = mx.stack(
        [Ks[..., 0, 0], Ks[..., 1, 1]], axis=-1
    )  # [..., C, 2]
    offset = mx.stack(
        [Ks[..., 0, 2], Ks[..., 1, 2]], axis=-1
    )  # [..., C, 2]
    means2d = means[..., :2] * diag_f[..., None, :] + offset[..., None, :]

    return means2d, cov2d


# ---------------------------------------------------------------------------
# Fully-Fused Projection (main entry point)
# ---------------------------------------------------------------------------


def fully_fused_projection(
    means: mx.array,          # [..., N, 3]
    covars: mx.array,         # [..., N, 3, 3]
    viewmats: mx.array,       # [..., C, 4, 4]
    Ks: mx.array,             # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Fully-fused 3D-to-2D Gaussian projection.

    Chains world-to-camera transform, camera projection (pinhole/fisheye/ortho),
    covariance regularization, conic computation, radius computation, and culling.

    Args:
        means: Gaussian means in world coordinates. Shape ``[..., N, 3]``.
        covars: Gaussian 3x3 covariances in world coordinates. Shape ``[..., N, 3, 3]``.
        viewmats: World-to-camera matrices. Shape ``[..., C, 4, 4]``.
        Ks: Camera intrinsic matrices. Shape ``[..., C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.
        eps2d: Regularization added to 2D covariance diagonal. Default ``0.3``.
        near_plane: Near clipping plane. Default ``0.01``.
        far_plane: Far clipping plane. Default ``1e10``.
        calc_compensations: Whether to compute opacity compensation factors.
        camera_model: One of ``"pinhole"``, ``"fisheye"``, ``"ortho"``.

    Returns:
        A tuple of:

        - **radii**: Per-axis integer radii ``[..., C, N, 2]`` (int32). Zero for culled.
        - **means2d**: Projected 2D means ``[..., C, N, 2]``.
        - **depths**: Z-depths in camera space ``[..., C, N]``.
        - **conics**: Inverse covariance conics ``[..., C, N, 3]``.
        - **compensations**: Compensation factors ``[..., C, N]`` or ``None``.
    """
    # 1. World to camera
    means_c, covars_c = world_to_cam(means, covars, viewmats)

    # 2. Camera to screen projection
    if camera_model == "ortho":
        means2d, covars2d = ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = persp_proj(means_c, covars_c, Ks, width, height)
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")

    # 3. Compute original determinant (before regularization)
    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )

    # 4. Add eps2d regularization to diagonal
    covars2d = covars2d + mx.eye(2, dtype=means.dtype) * eps2d

    # 5. Compute regularized determinant
    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = mx.clip(det, 1e-10, None)

    # 6. Compensation factors
    if calc_compensations:
        compensations = mx.sqrt(mx.clip(det_orig / det, 0.0, None))
    else:
        compensations = None

    # 7. Conics (inverse of 2D covariance, stored as 3 elements)
    conics = mx.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        axis=-1,
    )  # [..., C, N, 3]

    # 8. Depths
    depths = means_c[..., 2]  # [..., C, N]

    # 9. Radius computation
    radius_x = mx.ceil(3.33 * mx.sqrt(covars2d[..., 0, 0]))
    radius_y = mx.ceil(3.33 * mx.sqrt(covars2d[..., 1, 1]))
    radius = mx.stack([radius_x, radius_y], axis=-1)  # [..., C, N, 2]

    # 10. Near/far plane culling (no boolean indexing — use mx.where)
    valid = (depths > near_plane) & (depths < far_plane)
    radius = mx.where(
        mx.expand_dims(valid, axis=-1),
        radius,
        mx.zeros_like(radius),
    )

    # 11. Screen bounds culling
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

    # 12. Convert to integer radii
    radii = radius.astype(mx.int32)

    return radii, means2d, depths, conics, compensations
