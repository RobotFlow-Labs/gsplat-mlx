"""Tier-1 (NumPy reference) rasterization for 2D Gaussian Splatting (surfels).

This is the surfel-specific rasterizer that performs ray-surfel intersection
per pixel using the M matrix from projection. It accumulates colour, alpha,
and normal maps via front-to-back alpha compositing.

Algorithm per pixel:
    For each surfel g in depth order within the pixel's tile:
        h_u = -M[0,:] + M[2,:] * px
        h_v = -M[1,:] + M[2,:] * py
        tmp = cross(h_u, h_v)
        us, vs = tmp[0]/tmp[2], tmp[1]/tmp[2]
        sigma_3d = us^2 + vs^2
        sigma_2d = 2 * (dx^2 + dy^2)
        sigma = 0.5 * min(sigma_3d, sigma_2d)
        alpha = clamp(opacity * exp(-sigma), max=MAX_ALPHA)
        Composite colour AND normals with front-to-back blending.

This implementation is correct but slow (Python loops). It serves as a
reference for validating faster MLX/Metal implementations.
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    TRANSMITTANCE_THRESHOLD,
)


def rasterize_to_pixels_2dgs(
    means2d: mx.array,  # [C, N, 2]
    ray_transforms: mx.array,  # [C, N, 3, 3]
    colors: mx.array,  # [C, N, channels]
    normals: mx.array,  # [C, N, 3]
    opacities: mx.array,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: mx.array,  # [C, tile_H, tile_W]
    flatten_ids: mx.array,  # [n_isects]
    backgrounds: Optional[mx.array] = None,  # [C, channels]
) -> Tuple[mx.array, mx.array, mx.array]:
    """Rasterize 2D Gaussian surfels to pixels via tile-based compositing.

    Tier-1 NumPy reference implementation. Correct but slow and NOT
    differentiable. Use for unit-testing and validation only.

    Args:
        means2d: Projected 2D surfel centres. ``[C, N, 2]``.
        ray_transforms: Ray-surfel intersection matrices from projection.
            ``[C, N, 3, 3]``.
        colors: Per-surfel colour features. ``[C, N, channels]``.
        normals: Per-surfel camera-space normals. ``[C, N, 3]``.
        opacities: Per-surfel opacity. ``[C, N]``.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        tile_size: Side length of square tiles (e.g. 16).
        isect_offsets: Cumulative intersection counts per tile.
            ``[C, tile_H, tile_W]``.
        flatten_ids: Flattened sorted surfel indices. ``[n_isects]``.
        backgrounds: Optional per-camera background colour.
            ``[C, channels]`` or ``None``.

    Returns:
        A tuple of three arrays:

        - **render_colors**: Composited colour image. ``[C, H, W, channels]``.
        - **render_alphas**: Accumulated opacity. ``[C, H, W, 1]``.
        - **render_normals**: Accumulated normal map. ``[C, H, W, 3]``.
    """
    C = means2d.shape[0]
    N = means2d.shape[1]
    channels = colors.shape[-1]
    tile_height = isect_offsets.shape[1]
    tile_width = isect_offsets.shape[2]

    # Materialise to numpy for the reference loop
    mx.eval(
        means2d, ray_transforms, colors, normals, opacities,
        isect_offsets, flatten_ids,
    )
    means2d_np = np.array(means2d, dtype=np.float32)
    M_np = np.array(ray_transforms, dtype=np.float32)
    colors_np = np.array(colors, dtype=np.float32)
    normals_np = np.array(normals, dtype=np.float32)
    opacities_np = np.array(opacities, dtype=np.float32)
    isect_offsets_np = np.array(isect_offsets, dtype=np.int32)
    flatten_ids_np = np.array(flatten_ids, dtype=np.int32)

    render_colors = np.zeros(
        (C, image_height, image_width, channels), dtype=np.float32
    )
    render_alphas = np.zeros(
        (C, image_height, image_width, 1), dtype=np.float32
    )
    render_normals = np.zeros(
        (C, image_height, image_width, 3), dtype=np.float32
    )

    n_isects_total = int(flatten_ids_np.shape[0])

    for c in range(C):
        offsets_flat = isect_offsets_np[c].ravel()
        offsets_with_end = np.append(offsets_flat, n_isects_total)

        for tile_y in range(tile_height):
            for tile_x in range(tile_width):
                tile_idx = tile_y * tile_width + tile_x
                start = int(offsets_with_end[tile_idx])
                end = int(offsets_with_end[tile_idx + 1])

                if start >= end:
                    continue

                # flatten_ids = image_id * N + gauss_id
                # Extract local gaussian ids for this camera
                gauss_ids = flatten_ids_np[start:end] % N

                py_start = tile_y * tile_size
                py_end = min(py_start + tile_size, image_height)
                px_start = tile_x * tile_size
                px_end = min(px_start + tile_size, image_width)

                for py in range(py_start, py_end):
                    for px in range(px_start, px_end):
                        pxf = float(px) + 0.5
                        pyf = float(py) + 0.5
                        T = 1.0

                        for gid in gauss_ids:
                            # Ray-surfel intersection
                            Mg = M_np[c, gid]  # [3, 3]

                            h_u = -Mg[0, :3] + Mg[2, :3] * pxf  # [3]
                            h_v = -Mg[1, :3] + Mg[2, :3] * pyf  # [3]

                            # Cross product
                            tmp = np.cross(h_u, h_v)  # [3]

                            # Guard against degenerate cross product
                            if abs(tmp[2]) < 1e-10:
                                continue

                            us = tmp[0] / tmp[2]
                            vs = tmp[1] / tmp[2]

                            # 3D sigma from ray-surfel intersection
                            sigma_3d = us * us + vs * vs

                            # 2D fallback sigma
                            mu = means2d_np[c, gid]
                            dx = pxf - mu[0]
                            dy = pyf - mu[1]
                            sigma_2d = 2.0 * (dx * dx + dy * dy)

                            # Take the tighter bound
                            sigma = 0.5 * min(sigma_3d, sigma_2d)

                            if sigma < 0.0:
                                continue

                            alpha = min(
                                float(opacities_np[c, gid]) * float(
                                    np.exp(-sigma)
                                ),
                                MAX_ALPHA,
                            )
                            if alpha < ALPHA_THRESHOLD:
                                continue
                            if T < TRANSMITTANCE_THRESHOLD:
                                break

                            weight = T * alpha
                            render_colors[c, py, px] += (
                                weight * colors_np[c, gid]
                            )
                            render_normals[c, py, px] += (
                                weight * normals_np[c, gid]
                            )
                            T *= 1.0 - alpha

                        render_alphas[c, py, px, 0] = 1.0 - T

    # Background blending
    if backgrounds is not None:
        mx.eval(backgrounds)
        bg_np = np.array(backgrounds, dtype=np.float32)
        for c in range(C):
            render_colors[c] += (1.0 - render_alphas[c]) * bg_np[c]

    return (
        mx.array(render_colors),
        mx.array(render_alphas),
        mx.array(render_normals),
    )
