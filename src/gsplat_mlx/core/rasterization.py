"""Tier-1 (NumPy reference) pixel rasterization via tile-based alpha compositing.

This is a *correctness-first* implementation that loops over tiles and pixels
in Python/NumPy.  It is NOT differentiable and NOT fast -- its sole purpose is
to validate the alpha-compositing algorithm against upstream gsplat CUDA
kernels before we build the Tier-2 (pure-MLX) and Tier-3 (Metal shader)
versions.

Algorithm (front-to-back compositing per pixel):
    For each Gaussian g in depth order that overlaps the pixel's tile:
        dx, dy  = pixel_center - mean2d[g]
        sigma   = 0.5*(a*dx^2 + c*dy^2) + b*dx*dy   (Mahalanobis distance)
        alpha   = clamp(opacity[g] * exp(-sigma), 0, MAX_ALPHA)
        if alpha < ALPHA_THRESHOLD: skip
        if T < TRANSMITTANCE_THRESHOLD: break  (early termination)
        color  += T * alpha * color[g]
        T      *= (1 - alpha)
    render_alpha = 1 - T
    if background: color += T * background
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    TRANSMITTANCE_THRESHOLD,
)


def rasterize_to_pixels(
    means2d: mx.array,  # [C, N, 2]
    conics: mx.array,  # [C, N, 3]  (a, b, c of inverse 2D covariance)
    colors: mx.array,  # [C, N, channels]
    opacities: mx.array,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: mx.array,  # [C, tile_H, tile_W]
    flatten_ids: mx.array,  # [n_isects]
    backgrounds: Optional[mx.array] = None,  # [C, channels] or None
) -> Tuple[mx.array, mx.array]:
    """Rasterize 2D Gaussians to pixels via tile-based alpha compositing.

    This is the Tier-1 NumPy reference implementation.  It is correct but slow
    and **not differentiable**.  Use it only for unit-testing and validation.

    Args:
        means2d: Projected 2D means, shape ``[C, N, 2]``.
        conics: Upper-triangle entries ``(a, b, c)`` of the inverse 2D
            covariance for each Gaussian, shape ``[C, N, 3]``.
        colors: Per-Gaussian color features, shape ``[C, N, channels]``.
        opacities: Per-Gaussian scalar opacity (after sigmoid), ``[C, N]``.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        tile_size: Side length of square tiles (e.g. 16).
        isect_offsets: Cumulative intersection counts per tile,
            shape ``[C, tile_H, tile_W]``.  ``isect_offsets[c, ty, tx]``
            gives the *starting* index into ``flatten_ids`` for camera ``c``,
            tile ``(ty, tx)``.
        flatten_ids: Flattened array of Gaussian indices that intersect tiles,
            shape ``[n_isects]``.  Indices are *local* Gaussian ids in
            ``[0, N)``.
        backgrounds: Optional per-camera background colour,
            shape ``[C, channels]``.

    Returns:
        render_colors: ``[C, H, W, channels]`` composited colour image.
        render_alphas: ``[C, H, W, 1]`` accumulated opacity.
    """
    C = means2d.shape[0]
    N = means2d.shape[1]
    channels = colors.shape[-1]
    tile_height = isect_offsets.shape[1]
    tile_width = isect_offsets.shape[2]

    # Materialise MLX arrays and convert to numpy for the reference loop.
    mx.eval(means2d, conics, colors, opacities, isect_offsets, flatten_ids)
    means2d_np = np.array(means2d, dtype=np.float32)
    conics_np = np.array(conics, dtype=np.float32)
    colors_np = np.array(colors, dtype=np.float32)
    opacities_np = np.array(opacities, dtype=np.float32)
    isect_offsets_np = np.array(isect_offsets, dtype=np.int32)
    flatten_ids_np = np.array(flatten_ids, dtype=np.int32)

    render_colors = np.zeros(
        (C, image_height, image_width, channels), dtype=np.float32
    )
    render_alphas = np.zeros(
        (C, image_height, image_width, 1), dtype=np.float32
    )

    n_isects_total = int(flatten_ids_np.shape[0])

    for c in range(C):
        # Flatten the per-camera tile offsets to a 1-D array and append the
        # total intersection count so that ``end`` for the last tile is well
        # defined.
        offsets_flat = isect_offsets_np[c].ravel()  # [tile_H * tile_W]
        offsets_with_end = np.append(offsets_flat, n_isects_total)

        for tile_y in range(tile_height):
            for tile_x in range(tile_width):
                tile_idx = tile_y * tile_width + tile_x
                start = int(offsets_with_end[tile_idx])
                end = int(offsets_with_end[tile_idx + 1])

                if start >= end:
                    continue

                gauss_ids = flatten_ids_np[start:end]

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
                            mu = means2d_np[c, gid]
                            con = conics_np[c, gid]
                            dx = pxf - mu[0]
                            dy = pyf - mu[1]
                            sigma = (
                                0.5 * (con[0] * dx * dx + con[2] * dy * dy)
                                + con[1] * dx * dy
                            )

                            if sigma < 0.0:
                                continue

                            alpha = min(
                                float(opacities_np[c, gid])
                                * float(np.exp(-sigma)),
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
                            T *= 1.0 - alpha

                        render_alphas[c, py, px, 0] = 1.0 - T

    # Background blending: color += T * background  (T = 1 - alpha)
    if backgrounds is not None:
        mx.eval(backgrounds)
        bg_np = np.array(backgrounds, dtype=np.float32)
        for c in range(C):
            # (1 - alpha) broadcast over [H, W, channels]
            render_colors[c] += (1.0 - render_alphas[c]) * bg_np[c]

    return mx.array(render_colors), mx.array(render_alphas)
