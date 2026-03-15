"""Differentiable tile-based rasterizer in pure MLX.

This is the Tier-2 rasterizer: correct, differentiable, and GPU-accelerated
via MLX's lazy evaluation, but not as fast as a Metal shader (Tier-3).

The key insight: we loop over sorted Gaussians (sequential for correct
alpha compositing) but vectorize ALL pixel computations within each
Gaussian's contribution. For each Gaussian, we compute its effect on
ALL H*W pixels simultaneously using standard MLX ops, keeping the
computation graph intact for mx.grad().

Algorithm (front-to-back compositing):
    For each camera c:
        T = ones(H, W)           -- transmittance per pixel
        accum = zeros(H, W, ch)  -- accumulated colour per pixel
        For each Gaussian g in depth order (from intersection list):
            dx, dy = pixel_grid - mean2d[g]
            sigma  = 0.5*(a*dx^2 + c*dy^2) + b*dx*dy
            alpha  = clamp(opacity[g] * exp(-sigma), 0, MAX_ALPHA)
            alpha  *= mask(sigma >= 0) * mask(alpha >= ALPHA_THRESHOLD)
            weight = T * alpha
            accum += weight[..., None] * color[g]
            T     *= (1 - alpha)
        render_alpha = 1 - T
        if background: accum += T[..., None] * background
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    TRANSMITTANCE_THRESHOLD,
)


def rasterize_to_pixels_mlx(
    means2d: mx.array,        # [C, N, 2]
    conics: mx.array,         # [C, N, 3]  (a, b, c of inverse 2D covariance)
    colors: mx.array,         # [C, N, channels]
    opacities: mx.array,      # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: mx.array,  # [C, tile_H, tile_W]
    flatten_ids: mx.array,    # [n_isects]
    backgrounds: Optional[mx.array] = None,  # [C, channels] or None
) -> Tuple[mx.array, mx.array]:
    """Differentiable rasterizer using pure MLX operations.

    Unlike rasterize_to_pixels (NumPy Tier-1), this function preserves
    the MLX computation graph so gradients flow through mx.grad().

    The design loops over Gaussians in depth order (required for correct
    alpha compositing) but vectorizes all H*W pixel computations per
    Gaussian. Each Gaussian's contribution is a single fused MLX kernel
    operating on the full pixel grid.

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
            shape ``[C, tile_H, tile_W]``.
        flatten_ids: Flattened array of Gaussian indices that intersect
            tiles, shape ``[n_isects]``.
        backgrounds: Optional per-camera background colour,
            shape ``[C, channels]``.

    Returns:
        render_colors: ``[C, H, W, channels]`` composited colour image.
        render_alphas: ``[C, H, W, 1]`` accumulated opacity.
    """
    C = means2d.shape[0]
    N = means2d.shape[1]
    channels = colors.shape[-1]

    # Handle empty case
    if N == 0:
        rc = mx.zeros((C, image_height, image_width, channels))
        ra = mx.zeros((C, image_height, image_width, 1))
        if backgrounds is not None:
            for c in range(C):
                rc = rc.at[c].add(
                    mx.broadcast_to(backgrounds[c], (image_height, image_width, channels))
                )
        return rc, ra

    # Build pixel coordinate grid [H, W] -- these are integer ops, non-diff is fine
    px = mx.arange(image_width).astype(mx.float32) + 0.5    # [W]
    py = mx.arange(image_height).astype(mx.float32) + 0.5   # [H]
    # grid_x[h, w] = px[w], grid_y[h, w] = py[h]
    grid_x = mx.broadcast_to(px[None, :], (image_height, image_width))   # [H, W]
    grid_y = mx.broadcast_to(py[:, None], (image_height, image_width))   # [H, W]

    # Materialise integer intersection data (non-differentiable, index-only)
    mx.eval(isect_offsets, flatten_ids)
    offsets_np = np.array(isect_offsets, dtype=np.int32)
    flatten_np = np.array(flatten_ids, dtype=np.int32)
    n_isects_total = int(flatten_np.shape[0])

    # Determine the unique Gaussian ordering per camera.
    # We need a global depth-sorted list of Gaussian IDs.
    # Strategy: collect all unique Gaussian IDs across all tiles in their
    # depth-sorted order. Since flatten_ids is sorted by (tile_id, depth),
    # we need to merge the per-tile lists by the original depth order.
    #
    # Simpler approach: just iterate through ALL entries in flatten_ids for
    # this camera, tracking which Gaussians we've already processed. The
    # flatten_ids are sorted by (camera, tile, depth). To get correct
    # compositing we process each Gaussian once globally in depth order,
    # applying it to ALL pixels (not just its tile).
    #
    # But this loses the tile structure (a Gaussian far from a pixel
    # contributes negligibly due to exp(-sigma) -> 0).
    # That's fine for correctness -- the exponential decay handles it.
    #
    # However, the flatten_ids interleave Gaussians from different tiles,
    # so the same Gaussian appears multiple times (once per tile it overlaps).
    # We need to deduplicate and sort by depth.

    all_camera_colors = []
    all_camera_alphas = []

    for c in range(C):
        # Collect all Gaussian IDs for this camera from all tiles, deduplicate
        offsets_flat = offsets_np[c].ravel()
        offsets_with_end = np.append(offsets_flat, n_isects_total)
        tile_H = isect_offsets.shape[1]
        tile_W = isect_offsets.shape[2]

        # Collect unique Gaussian IDs preserving first-occurrence order
        # (which corresponds to depth order within tiles).
        # Since we need global depth order, we extract from tile (0,0) first,
        # then add any new ones from subsequent tiles. However, this doesn't
        # guarantee correct global ordering.
        #
        # Better approach: collect ALL (gid, first_tile_idx) pairs, then
        # sort by the index in flatten_ids where each gid first appears
        # (which correlates with depth since tiles are sorted by depth).
        #
        # Actually, the correct approach is to NOT deduplicate at all, but
        # instead process tile-by-tile and only apply each Gaussian to its
        # tile's pixels. This preserves the correct per-tile depth ordering
        # and limits each Gaussian's spatial extent.

        # Process tile-by-tile, accumulating into full-image buffers
        T = mx.ones((image_height, image_width))
        accum = mx.zeros((image_height, image_width, channels))

        for ty in range(tile_H):
            for tx in range(tile_W):
                tile_idx = ty * tile_W + tx
                start = int(offsets_with_end[tile_idx])
                end = int(offsets_with_end[tile_idx + 1])

                if start >= end:
                    continue

                # Pixel range for this tile
                py_s = ty * tile_size
                py_e = min(py_s + tile_size, image_height)
                px_s = tx * tile_size
                px_e = min(px_s + tile_size, image_width)
                th = py_e - py_s
                tw = px_e - px_s

                # Tile pixel grids
                tile_gx = grid_x[py_s:py_e, px_s:px_e]  # [th, tw]
                tile_gy = grid_y[py_s:py_e, px_s:px_e]  # [th, tw]

                # Extract tile transmittance and accumulated color
                tile_T = T[py_s:py_e, px_s:px_e]        # [th, tw]
                tile_accum = accum[py_s:py_e, px_s:px_e] # [th, tw, ch]

                # Get sorted Gaussian IDs for this tile
                gauss_ids = flatten_np[start:end] % N

                # Loop over Gaussians in depth order (sequential for compositing)
                for gid in gauss_ids:
                    gid = int(gid)

                    mu = means2d[c, gid]       # [2]
                    con = conics[c, gid]        # [3]
                    opa = opacities[c, gid]     # scalar
                    col = colors[c, gid]        # [channels]

                    # Compute Mahalanobis distance for all tile pixels
                    dx = tile_gx - mu[0]  # [th, tw]
                    dy = tile_gy - mu[1]  # [th, tw]
                    sigma = 0.5 * (con[0] * dx * dx + con[2] * dy * dy) + con[1] * dx * dy

                    # Compute alpha with masks (differentiable via mx.where)
                    raw_alpha = opa * mx.exp(-sigma)
                    alpha = mx.clip(raw_alpha, a_min=0.0, a_max=MAX_ALPHA)

                    # Mask: sigma must be >= 0 (valid Gaussian region)
                    alpha = mx.where(sigma >= 0, alpha, mx.zeros_like(alpha))
                    # Mask: alpha must be >= ALPHA_THRESHOLD
                    alpha = mx.where(alpha >= ALPHA_THRESHOLD, alpha, mx.zeros_like(alpha))

                    # Early termination: zero out alpha for saturated pixels
                    # (transmittance near zero). Using mx.where preserves
                    # differentiability -- do NOT call mx.eval() on any
                    # differentiable arrays in this loop.
                    alpha = mx.where(tile_T >= TRANSMITTANCE_THRESHOLD, alpha, mx.zeros_like(alpha))

                    # Compositing
                    weight = tile_T * alpha                    # [th, tw]
                    tile_accum = tile_accum + weight[:, :, None] * col  # broadcast col over [th, tw]
                    tile_T = tile_T * (1.0 - alpha)

                # Write tile results back into full image arrays.
                # We use a differentiable pattern: subtract old value, add new value.
                # This is equivalent to assignment but works with MLX's computation graph.
                old_accum = accum[py_s:py_e, px_s:px_e]
                accum = accum.at[py_s:py_e, px_s:px_e].add(tile_accum - old_accum)
                old_T = T[py_s:py_e, px_s:px_e]
                T = T.at[py_s:py_e, px_s:px_e].add(tile_T - old_T)

        all_camera_colors.append(accum)
        all_camera_alphas.append(1.0 - T)

    # Stack cameras into [C, H, W, channels] and [C, H, W, 1]
    render_colors = mx.stack(all_camera_colors, axis=0)  # [C, H, W, ch]
    render_alphas = mx.stack(all_camera_alphas, axis=0)[:, :, :, None]  # [C, H, W, 1]

    # Background blending
    if backgrounds is not None:
        remaining = 1.0 - render_alphas  # [C, H, W, 1]
        # backgrounds: [C, channels] -> [C, 1, 1, channels]
        bg = backgrounds[:, None, None, :]
        render_colors = render_colors + remaining * bg

    return render_colors, render_alphas
