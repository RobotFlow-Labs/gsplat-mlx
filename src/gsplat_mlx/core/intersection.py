"""Tile-Gaussian intersection and depth sorting for 3D Gaussian Splatting.

This module implements the acceleration structure that maps projected 2D Gaussians
to the screen tiles they overlap, sorted by depth within each tile. This enables
the rasterizer to process each tile independently with only its relevant Gaussians.

All operations are non-differentiable (no gradients needed).

Port of ``_isect_tiles`` and ``_isect_offset_encode`` from
``gsplat/cuda/_torch_impl.py`` (lines 330-456).
"""

import math

import mlx.core as mx
import numpy as np


def isect_tiles(
    means2d: mx.array,  # [..., N, 2] float32
    radii: mx.array,  # [..., N, 2] int32
    depths: mx.array,  # [..., N]    float32
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute tile-Gaussian intersections with depth-sorted keys.

    For each projected 2D Gaussian, determines which screen tiles it overlaps
    and creates a sorted intersection list keyed by (image_id, tile_id, depth).

    Args:
        means2d: Projected 2D Gaussian centers. [..., N, 2]
        radii: Maximum pixel radii of projected Gaussians. [..., N, 2] int32.
               Gaussians with any radius component <= 0 are skipped.
        depths: Camera-space z-depth of each Gaussian. [..., N]
        tile_size: Pixel size of each tile (typically 16).
        tile_width: Number of tile columns (ceil(image_width / tile_size)).
        tile_height: Number of tile rows (ceil(image_height / tile_size)).
        sort: Whether to sort intersections by the 64-bit key. Default True.

    Returns:
        tiles_per_gauss: Number of tiles each Gaussian intersects. [..., N] int32
        isect_ids: Packed 64-bit sort keys. [n_isects] int64
            Layout: (image_id << (tile_n_bits + 32)) | (tile_id << 32) | depth_bits
        flatten_ids: Flattened Gaussian indices (image_id * N + gauss_id). [n_isects] int32
    """
    # --- Phase 0: Shape bookkeeping ---
    image_dims = means2d.shape[:-2]
    N = means2d.shape[-2]
    I = math.prod(image_dims) if image_dims else 1

    # --- Phase 1: Convert to numpy, compute tile bounds (vectorized) ---
    means2d_np = np.array(means2d, dtype=np.float32).reshape(I, N, 2)
    radii_np = np.array(radii, dtype=np.float32).reshape(I, N, 2)
    depths_np = np.array(depths, dtype=np.float32).reshape(I, N)

    tile_means = means2d_np / tile_size  # [I, N, 2]
    tile_radii = radii_np / tile_size  # [I, N, 2]
    tile_mins = np.floor(tile_means - tile_radii).astype(np.int32)  # [I, N, 2]
    tile_maxs = np.ceil(tile_means + tile_radii).astype(np.int32)  # [I, N, 2]

    tile_mins[..., 0] = np.clip(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = np.clip(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = np.clip(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = np.clip(tile_maxs[..., 1], 0, tile_height)

    tile_ranges = tile_maxs - tile_mins  # [I, N, 2]
    tiles_per_gauss = tile_ranges.prod(axis=-1)  # [I, N]
    valid_mask = (radii_np > 0).all(axis=-1)  # [I, N]
    tiles_per_gauss *= valid_mask  # [I, N]

    n_isects = int(tiles_per_gauss.sum())

    # --- Phase 2: Enumerate intersections (semi-vectorized) ---
    if n_isects == 0:
        tiles_per_gauss_out = tiles_per_gauss.reshape(image_dims + (N,)).astype(
            np.int32
        )
        return (
            mx.array(tiles_per_gauss_out),
            mx.array(np.empty(0, dtype=np.int64)),
            mx.array(np.empty(0, dtype=np.int32)),
        )

    flat_tiles_per_gauss = tiles_per_gauss.flatten().astype(np.int64)  # [I*N]

    # Replicate metadata for each intersection
    flat_indices = np.arange(I * N)
    image_ids_flat = flat_indices // N  # [I*N]
    gauss_ids_flat = flat_indices % N  # [I*N]
    tile_min_x_flat = tile_mins[..., 0].flatten()  # [I*N]
    tile_min_y_flat = tile_mins[..., 1].flatten()  # [I*N]
    tile_range_x_flat = tile_ranges[..., 0].flatten()  # [I*N]

    # np.repeat expands each entry by its tile count
    rep_image_ids = np.repeat(image_ids_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_gauss_ids = np.repeat(gauss_ids_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_tile_min_x = np.repeat(tile_min_x_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_tile_min_y = np.repeat(tile_min_y_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_tile_range_x = np.repeat(
        tile_range_x_flat, flat_tiles_per_gauss
    )  # [n_isects]

    # Compute local index within each Gaussian's tile range
    cum_tiles = np.cumsum(flat_tiles_per_gauss)  # [I*N]
    offsets = np.zeros(I * N, dtype=np.int64)
    offsets[1:] = cum_tiles[:-1]
    rep_offsets = np.repeat(offsets, flat_tiles_per_gauss)  # [n_isects]
    local_idx = np.arange(n_isects) - rep_offsets  # [n_isects]

    # Handle case where rep_tile_range_x could be 0 (shouldn't happen for valid
    # Gaussians since tiles_per_gauss > 0 implies range_x > 0, but guard anyway)
    safe_range_x = np.maximum(rep_tile_range_x, 1)

    # Convert local index to (tile_x, tile_y) within the Gaussian's tile range
    local_x = (local_idx % safe_range_x).astype(np.int32)
    local_y = (local_idx // safe_range_x).astype(np.int32)
    tile_x = rep_tile_min_x + local_x  # [n_isects]
    tile_y = rep_tile_min_y + local_y  # [n_isects]
    tile_id = (tile_y * tile_width + tile_x).astype(np.int32)  # [n_isects]

    # --- Phase 3: Build sort keys (vectorized) ---
    tile_n_bits = (tile_width * tile_height).bit_length()

    isect_ids_hi = (rep_image_ids.astype(np.int32) << tile_n_bits) | tile_id

    # Float-to-uint32 bit reinterpretation for depth
    rep_depths = depths_np[rep_image_ids, rep_gauss_ids].astype(np.float32)
    isect_ids_lo = rep_depths.view(np.uint32)  # [n_isects]

    # Combine into 64-bit sort key
    isect_ids = (isect_ids_hi.astype(np.int64) << 32) | (
        isect_ids_lo.astype(np.int64) & 0xFFFFFFFF
    )

    # Build flatten_ids
    flatten_ids = (rep_image_ids * N + rep_gauss_ids).astype(np.int32)

    # --- Phase 4: Sort ---
    if sort:
        sort_indices = np.argsort(isect_ids, kind="stable")
        isect_ids = isect_ids[sort_indices]
        flatten_ids = flatten_ids[sort_indices]

    # --- Phase 5: Convert to MLX ---
    tiles_per_gauss_out = tiles_per_gauss.reshape(image_dims + (N,)).astype(np.int32)
    tiles_per_gauss_mlx = mx.array(tiles_per_gauss_out)
    isect_ids_mlx = mx.array(isect_ids)  # int64
    flatten_ids_mlx = mx.array(flatten_ids)  # int32

    return tiles_per_gauss_mlx, isect_ids_mlx, flatten_ids_mlx


def isect_offset_encode(
    isect_ids: mx.array,  # [n_isects] int64
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> mx.array:
    """Encode tile offsets as prefix sums for O(1) per-tile lookup.

    Given the sorted intersection IDs from isect_tiles, builds an offset table
    where offsets[i, y, x] gives the start index into the sorted flatten_ids
    for tile (x, y) of image i.

    Args:
        isect_ids: Sorted 64-bit intersection keys from isect_tiles. [n_isects]
        n_images: Number of images (I).
        tile_width: Number of tile columns.
        tile_height: Number of tile rows.

    Returns:
        offsets: Start indices per tile. [I, tile_height, tile_width] int32
    """
    isect_ids_np = np.array(isect_ids)
    tile_n_bits = (tile_width * tile_height).bit_length()

    tile_counts = np.zeros(
        (n_images, tile_height, tile_width), dtype=np.int64
    )

    if len(isect_ids_np) > 0:
        hi_bits = (isect_ids_np >> 32).astype(np.int64)
        unique_hi, counts = np.unique(hi_bits, return_counts=True)

        image_ids = (unique_hi >> tile_n_bits).astype(np.int64)
        tile_ids = unique_hi & ((1 << tile_n_bits) - 1)
        tile_x = (tile_ids % tile_width).astype(np.int64)
        tile_y = (tile_ids // tile_width).astype(np.int64)

        tile_counts[image_ids, tile_y, tile_x] = counts

    cum_counts = np.cumsum(tile_counts.flatten()).reshape(tile_counts.shape)
    offsets = cum_counts - tile_counts

    return mx.array(offsets.astype(np.int32))
