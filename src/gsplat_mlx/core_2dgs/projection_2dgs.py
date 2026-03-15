"""2D Gaussian Splatting (surfel) projection to screen space.

Port of ``_fully_fused_projection_2dgs`` from upstream
``gsplat/cuda/_torch_impl_2dgs.py`` to Apple MLX.

Each 2D Gaussian is a flat disk (surfel) defined by a center, orientation
quaternion, and scale. This module transforms surfels from world space to
screen space, computing the ray-surfel intersection transform matrix (M),
screen-space bounding boxes, and camera-space normals.
"""

from typing import Tuple

import mlx.core as mx

from gsplat_mlx.core.math_utils import _quat_scale_to_matrix


def fully_fused_projection_2dgs(
    means: mx.array,  # [N, 3]
    quats: mx.array,  # [N, 4]
    scales: mx.array,  # [N, 3]
    viewmats: mx.array,  # [C, 4, 4]
    Ks: mx.array,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps: float = 0.0,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Project 2D Gaussian surfels to screen space.

    Transforms surfel primitives from world space into screen-space
    parameters needed for rasterization: 2D means, bounding radii,
    ray-surfel intersection matrices, and camera-space normals.

    Args:
        means: 3D surfel centres. Shape ``[N, 3]``.
        quats: Quaternion orientations ``(w, x, y, z)``. Shape ``[N, 4]``.
        scales: Scale factors for each surfel. Shape ``[N, 3]``.
        viewmats: World-to-camera matrices. Shape ``[C, 4, 4]``.
        Ks: Camera intrinsic matrices. Shape ``[C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping distance. Default ``0.01``.
        far_plane: Far clipping distance. Default ``1e10``.
        eps: Epsilon for numerical stability in AABB computation.
            Default ``0.0``.

    Returns:
        A tuple of five arrays:

        - **radii**: Integer bounding radii per axis. ``[C, N, 2]`` int32.
          Zero for culled surfels.
        - **means2d**: Screen-space surfel centres. ``[C, N, 2]``.
        - **depths**: Camera-space z-depth. ``[C, N]``.
        - **ray_transforms**: Ray-surfel intersection matrix (M^T in paper).
          ``[C, N, 3, 3]``.
        - **normals**: Camera-space surfel normals (flipped toward camera).
          ``[C, N, 3]``.
    """
    N = means.shape[0]
    C = viewmats.shape[0]

    # --- Step 1: Transform means to camera space ---
    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]

    # means_c = R_cw @ means + t_cw  -> [C, N, 3]
    means_c = mx.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]

    # --- Step 2: Build RS matrix in world space and transform to camera ---
    RS_wl = _quat_scale_to_matrix(quats, scales)  # [N, 3, 3]
    RS_cl = mx.einsum("cij,njk->cnik", R_cw, RS_wl)  # [C, N, 3, 3]

    # --- Step 3: Compute normals (third column of RS_cl) ---
    normals = RS_cl[..., 2]  # [C, N, 3]

    # Flip normals toward camera: dot(normal, means_c) should be < 0
    # cos = -normal . means_c (per surfel)
    cos = -mx.sum(normals * means_c, axis=-1, keepdims=True)  # [C, N, 1]
    multiplier = mx.where(cos > 0, mx.array(1.0), mx.array(-1.0))  # [C, N, 1]
    normals = normals * multiplier  # [C, N, 3]

    # --- Step 4: Build ray transform matrix ---
    # T_cl = [RS_cl[..., :2], means_c[..., None]]  -> [C, N, 3, 3]
    T_cl = mx.concatenate(
        [RS_cl[..., :2], means_c[..., None]], axis=-1
    )  # [C, N, 3, 3]

    # T_sl = K @ T_cl  -> [C, N, 3, 3]
    T_sl = mx.einsum("cij,cnjk->cnik", Ks[:, :3, :3], T_cl)  # [C, N, 3, 3]

    # M = T_sl^T (in paper notation M = (WH)^T)
    M = mx.transpose(T_sl, (0, 1, 3, 2))  # [C, N, 3, 3]

    # --- Step 5: Compute AABB from M matrix ---
    # M is [C, N, 3, 3]. In upstream torch: M[..., i] indexes last dim (columns).
    # M[..., 2] = column 2 -> [C, N, 3]
    # M[..., :2] = columns 0,1 -> [C, N, 3, 2]
    # M[..., 2:3] = column 2 kept -> [C, N, 3, 1]
    test = mx.array([1.0, 1.0, -1.0])  # [3]
    test_broadcast = mx.reshape(test, (1, 1, 3))

    # d = sum(M[..., 2] * M[..., 2] * test, dim=-1, keepdim=True)
    # M[..., 2] is column 2: [C, N, 3]
    M_col2 = M[..., 2]  # [C, N, 3]
    d = mx.sum(
        M_col2 * M_col2 * test_broadcast, axis=-1, keepdims=True
    )  # [C, N, 1]

    valid = mx.abs(d) > eps  # [C, N, 1]

    # f = test / d where valid, else 0  -> [C, N, 3] -> unsqueeze to [C, N, 3, 1]
    f = mx.where(
        valid, test_broadcast / d, mx.zeros_like(test_broadcast)
    )  # [C, N, 3]
    f = mx.expand_dims(f, axis=-1)  # [C, N, 3, 1]

    # means2d = sum(M[..., :2] * M[..., 2:3] * f, dim=-2)
    # M[..., :2] is [C, N, 3, 2], M[..., 2:3] is [C, N, 3, 1], f is [C, N, 3, 1]
    # product is [C, N, 3, 2], sum over dim=-2 (the 3) gives [C, N, 2]
    means2d = mx.sum(
        M[..., :2] * M[..., 2:3] * f, axis=-2
    )  # [C, N, 2]

    # extents = sqrt(clamp(means2d^2 - sum(M[..., :2] * M[..., :2] * f, dim=-2), min=1e-4))
    extents_sq = means2d ** 2 - mx.sum(
        M[..., :2] * M[..., :2] * f, axis=-2
    )  # [C, N, 2]
    extents = mx.sqrt(mx.clip(extents_sq, a_min=1e-4, a_max=None))  # [C, N, 2]

    # --- Step 6: Compute depths and radii ---
    depths = means_c[..., 2]  # [C, N]
    radius = mx.ceil(3.33 * extents)  # [C, N, 2]

    # --- Step 7: Cull invalid surfels ---
    valid_squeeze = valid[..., 0]  # [C, N]
    depth_valid = (depths > near_plane) & (depths < far_plane)  # [C, N]
    combined_valid = valid_squeeze & depth_valid  # [C, N]

    # radius[~valid] = 0
    combined_valid_2 = mx.expand_dims(combined_valid, axis=-1)  # [C, N, 1]
    radius = mx.where(
        mx.broadcast_to(combined_valid_2, radius.shape),
        radius,
        mx.zeros_like(radius),
    )  # [C, N, 2]

    # Check inside image bounds
    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )  # [C, N]

    inside_2 = mx.expand_dims(inside, axis=-1)  # [C, N, 1]
    radius = mx.where(
        mx.broadcast_to(inside_2, radius.shape),
        radius,
        mx.zeros_like(radius),
    )  # [C, N, 2]

    radii = radius.astype(mx.int32)  # [C, N, 2]

    # Transpose M back for output (matching upstream final transpose)
    ray_transforms = mx.transpose(M, (0, 1, 3, 2))  # [C, N, 3, 3]

    return radii, means2d, depths, ray_transforms, normals
