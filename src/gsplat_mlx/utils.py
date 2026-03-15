"""Utility functions for gsplat-mlx.

Port of ``gsplat/utils.py`` depth-to-normal and projection matrix utilities
from PyTorch to Apple MLX.
"""

import math

import mlx.core as mx
import numpy as np


def depth_to_points(
    depths: mx.array,
    camtoworlds: mx.array,
    Ks: mx.array,
    z_depth: bool = True,
) -> mx.array:
    """Convert depth maps to 3D world-space points.

    Args:
        depths: Depth maps. Shape ``[..., H, W, 1]``.
        camtoworlds: Camera-to-world transformation matrices.
            Shape ``[..., 4, 4]``.
        Ks: Camera intrinsic matrices. Shape ``[..., 3, 3]``.
        z_depth: If ``True``, treat depth as z-depth; otherwise as ray
            distance.

    Returns:
        World-space points. Shape ``[..., H, W, 3]``.
    """
    batch_shape = depths.shape[:-3]
    H, W = depths.shape[-3], depths.shape[-2]

    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    # Pixel grids
    x_coords = np.arange(W, dtype=np.float32)
    y_coords = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    grid_x = mx.array(grid_x)  # [H, W]
    grid_y = mx.array(grid_y)  # [H, W]

    # Build camera directions: (u - cx + 0.5) / fx, (v - cy + 0.5) / fy, 1
    # Expand intrinsics for broadcasting
    fx_e = fx[..., None, None]
    fy_e = fy[..., None, None]
    cx_e = cx[..., None, None]
    cy_e = cy[..., None, None]

    dir_x = (grid_x - cx_e + 0.5) / fx_e  # [..., H, W]
    dir_y = (grid_y - cy_e + 0.5) / fy_e
    dir_z = mx.ones_like(dir_x)

    camera_dirs = mx.stack([dir_x, dir_y, dir_z], axis=-1)  # [..., H, W, 3]

    # Rotate to world
    R = camtoworlds[..., :3, :3]  # [..., 3, 3]
    directions = mx.einsum("...ij,...hwj->...hwi", R, camera_dirs)

    if not z_depth:
        # Normalise for ray-depth mode
        norm = mx.sqrt(mx.sum(directions * directions, axis=-1, keepdims=True))
        directions = directions / mx.maximum(norm, mx.array(1e-8))

    # Origin
    origins = camtoworlds[..., :3, 3]  # [..., 3]

    # Broadcast origin to [..., H, W, 3]
    for _ in range(2):
        origins = mx.expand_dims(origins, axis=-2)

    points = origins + depths * directions  # [..., H, W, 3]
    return points


def depth_to_normal(
    depths: mx.array,
    camtoworlds: mx.array,
    Ks: mx.array,
    z_depth: bool = True,
) -> mx.array:
    """Convert depth maps to surface normals via finite differences.

    Computes normals by taking cross products of horizontal and vertical
    finite differences of the back-projected 3D points.  Border pixels
    are set to zero.

    Args:
        depths: Depth maps. Shape ``[..., H, W, 1]``.
        camtoworlds: Camera-to-world transformation matrices.
            Shape ``[..., 4, 4]``.
        Ks: Camera intrinsic matrices. Shape ``[..., 3, 3]``.
        z_depth: If ``True``, treat depth as z-depth; otherwise as ray
            distance.

    Returns:
        Surface normals in world coordinates. Shape ``[..., H, W, 3]``.
        Normals at the image border are zero.
    """
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)

    # Finite differences
    dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]  # [..., H-2, W-2, 3]
    dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]  # [..., H-2, W-2, 3]

    # Cross product
    normals_inner = mx.cross(dx, dy)  # [..., H-2, W-2, 3]

    # Normalise
    norm = mx.sqrt(mx.sum(normals_inner * normals_inner, axis=-1, keepdims=True))
    normals_inner = normals_inner / mx.maximum(norm, mx.array(1e-8))

    # Pad with zeros to restore original spatial size
    batch_shape = normals_inner.shape[:-3]
    H, W = depths.shape[-3], depths.shape[-2]
    H_inner, W_inner = normals_inner.shape[-3], normals_inner.shape[-2]

    normals = mx.zeros(batch_shape + (H, W, 3), dtype=normals_inner.dtype)
    # Place inner normals at [1:-1, 1:-1]
    normals = normals.at[..., 1:-1, 1:-1, :].add(normals_inner)

    return normals


def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
) -> mx.array:
    """Create an OpenGL-style perspective projection matrix.

    Args:
        znear: Near clipping plane distance.
        zfar: Far clipping plane distance.
        fovX: Horizontal field of view in radians.
        fovY: Vertical field of view in radians.

    Returns:
        4x4 projection matrix as ``mx.array`` with float32 dtype.

    Examples:
        >>> P = get_projection_matrix(0.01, 100.0, 1.0, 0.8)
        >>> P.shape
        (4, 4)
    """
    tan_half_fovY = math.tan(fovY / 2.0)
    tan_half_fovX = math.tan(fovX / 2.0)

    top = tan_half_fovY * znear
    bottom = -top
    right = tan_half_fovX * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return mx.array(P)
