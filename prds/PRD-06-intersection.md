# PRD-06: Tile-Gaussian Intersection and Depth Sorting

## Overview

Port the tile-based intersection and depth-sorting system from `_torch_impl.py`. The screen is divided into tiles (typically 16x16 pixels). For each tile, we determine which Gaussians overlap it, then sort those Gaussians by depth (front-to-back). This is the key acceleration structure for rasterization -- without it, every pixel would need to evaluate every Gaussian. The intersection system produces a sorted, flattened list of Gaussian indices per tile, plus an offset table for O(1) lookup into that list.

This module is the bridge between projection (PRD-05) and rasterization (PRD-07). All operations are non-differentiable (`@torch.no_grad()` in upstream), so no backward pass or `@mx.custom_function` is needed.

## Source Reference

- **Primary**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:330-456`
  - `_isect_tiles` (lines 330-426): Compute tile-Gaussian intersections with depth-sorted keys
  - `_isect_offset_encode` (lines 429-456): Build prefix-sum offset table for tile lookup
- **Public API**: `_wrapper.py:622-697` (`isect_tiles`) -- includes packed mode (out of scope)
- **Public API**: `_wrapper.py:773-793` (`isect_offset_encode`)
- **CUDA kernel**: `csrc/IntersectTile.cu` (reference only, we port the Python reference)
- **Lines**: ~130 lines total

## Scope

### In Scope
- `isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, sort)` -- compute all tile-Gaussian intersections with 64-bit sort keys
- `isect_offset_encode(isect_ids, n_images, tile_width, tile_height)` -- encode prefix-sum offsets for per-tile lookup
- Depth-based sorting within tiles using the 64-bit key bit-packing scheme: `image_id | tile_id | depth_bits`
- Float-to-int bit reinterpretation for IEEE 754-correct depth sorting
- Vectorized MLX/numpy implementation (replacing upstream's nested Python loops)
- int64 workaround for MLX Metal backend limitations

### Out of Scope
- Packed input mode (nnz format with `image_ids`, `gaussian_ids`)
- Segmented radix sort (`segmented=True` argument)
- Lidar tile intersection (`isect_tiles_lidar`)
- CUDA kernel (`IntersectTile.cu`) -- we port the Python reference

## Technical Design

### Architecture Decision: CPU-Side NumPy Computation

Since intersection is a non-differentiable, no-grad operation, and the upstream Python reference already runs on CPU, our implementation uses NumPy for the core computation and converts results back to MLX arrays. This sidesteps all MLX Metal limitations (no int64, no advanced sorting) while maintaining correctness. The computation is embarrassingly parallel in principle but the upstream reference uses sequential loops -- we vectorize with NumPy broadcasting.

### Key Functions to Port

| Function | Upstream Location | MLX Approach |
|----------|-------------------|-------------|
| `_isect_tiles` | `_torch_impl.py:330-426` | Vectorized NumPy tile-bounds + expansion; NumPy int64 sort; return MLX arrays |
| `_isect_offset_encode` | `_torch_impl.py:429-456` | NumPy `unique` + scatter into count array + `cumsum` prefix-sum; return MLX array |

### MLX Port Challenges and Solutions

#### Challenge 1: No int64 on MLX Metal GPU

The upstream packs a 64-bit sort key as `(image_id << (tile_n_bits + 32)) | (tile_id << 32) | depth_bits`. MLX's Metal backend does not support int64 for GPU operations.

**Solution**: Perform the sort in NumPy (which fully supports int64). Since this is a no-grad CPU-side operation, there is zero performance penalty from using NumPy. The upstream torch code already stores `isect_ids_hi` and `isect_ids_lo` as separate int32 arrays before combining into int64, confirming this split is natural.

**Alternative considered**: Lexicographic sort with two int32 keys (sort by depth first, then stable-sort by tile_id). Rejected because NumPy int64 sort is simpler, faster, and matches upstream exactly.

**Return format**: We return `isect_ids` as two separate int32 MLX arrays (`isect_ids_hi`, `isect_ids_lo`) instead of a single int64 array. Downstream consumers (PRD-07 rasterization, PRD-08 accumulate) only need `flatten_ids` and the offset table, so the int64 keys are only used internally for sorting. However, `isect_offset_encode` needs the hi bits, so we also provide a combined int64 numpy path for that function.

**Final return format decision**: Return `isect_ids` as a single int64 MLX array. MLX *does* support int64 dtype for storage and basic indexing -- the limitation is only for Metal GPU compute kernels (reductions, custom ops). Since `isect_ids` is only used as input to `isect_offset_encode` (which we also implement via NumPy), this works. If MLX raises on int64 creation, fall back to returning `(isect_ids_hi, isect_ids_lo)` tuple.

#### Challenge 2: Nested Python Loops (O(I * N) iterations)

The upstream `_isect_tiles` uses:
```python
for image_id in range(I):
    for gauss_id in range(N):
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                # fill intersection entry
```

This is O(I * N * avg_tiles_per_gauss) with Python loop overhead. For I=1, N=100k, avg 4 tiles per Gaussian, that is 400k Python iterations.

**Solution**: Vectorize with NumPy in three phases:
1. **Tile bounds** (fully vectorized): compute tile_mins, tile_maxs, tiles_per_gauss for all Gaussians at once
2. **Intersection enumeration** (semi-vectorized): use `np.repeat` + `np.arange` to expand each Gaussian into its tile list without Python inner loops
3. **Sort key construction** (fully vectorized): build int64 keys with vectorized bit operations

See detailed algorithm below.

#### Challenge 3: Float-to-Int Bit Reinterpretation for Depth Sorting

The upstream uses `struct.pack("f", depth) -> struct.unpack("i", ...)` to reinterpret float32 bits as int32. This ensures IEEE 754 positive floats sort correctly when compared as integers (the bit pattern of positive floats is monotonically increasing with value).

**Solution**: Use `np.ndarray.view(np.uint32)` for zero-copy bit reinterpretation:
```python
depth_as_uint32 = depths_np.view(np.uint32)
```

This is equivalent to the struct.pack/unpack but operates on entire arrays at once.

**Important**: This only works correctly for non-negative depths. Negative depths would have the sign bit set and sort incorrectly as unsigned integers. However, Gaussians with negative depth are behind the camera and should have `radii=0` (filtered by PRD-05 frustum culling), so they produce zero tile intersections.

#### Challenge 4: `unique_consecutive` (used in offset encode)

The upstream uses `torch.unique_consecutive(isect_ids >> 32, return_counts=True)` which only deduplicates adjacent equal elements (O(n), not O(n log n) like full unique).

**Solution**: Since our isect_ids are already sorted, `np.unique` with `return_counts=True` produces the same result. The sorted invariant guarantees that equal elements are adjacent, making `unique_consecutive` equivalent to `unique` on sorted input.

For maximum fidelity, we can also implement the diff-based approach:
```python
hi_bits = isect_ids_np >> 32
changes = np.concatenate([[True], hi_bits[1:] != hi_bits[:-1]])
unique_vals = hi_bits[changes]
change_indices = np.concatenate([np.where(changes)[0], [len(hi_bits)]])
counts = np.diff(change_indices)
```

### Detailed Vectorized Algorithm for `isect_tiles`

#### Phase 1: Tile Bounds Computation (Fully Vectorized)

```
Input shapes:
  means2d: [I, N, 2]  float32
  radii:   [I, N, 2]  int32
  depths:  [I, N]      float32

Step 1.1: Convert to tile coordinates
  tile_means = means2d / tile_size          # [I, N, 2] float32
  tile_radii = radii / tile_size            # [I, N, 2] float32

Step 1.2: Compute tile extents
  tile_mins = floor(tile_means - tile_radii).astype(int32)   # [I, N, 2]
  tile_maxs = ceil(tile_means + tile_radii).astype(int32)    # [I, N, 2]

Step 1.3: Clamp to grid bounds
  tile_mins[..., 0] = clip(tile_mins[..., 0], 0, tile_width)
  tile_mins[..., 1] = clip(tile_mins[..., 1], 0, tile_height)
  tile_maxs[..., 0] = clip(tile_maxs[..., 0], 0, tile_width)
  tile_maxs[..., 1] = clip(tile_maxs[..., 1], 0, tile_height)

Step 1.4: Count tiles per Gaussian
  tile_ranges = tile_maxs - tile_mins                         # [I, N, 2]
  tiles_per_gauss = tile_ranges.prod(axis=-1)                 # [I, N]
  valid_mask = (radii > 0).all(axis=-1)                       # [I, N] bool
  tiles_per_gauss *= valid_mask                                # [I, N]

Step 1.5: Total intersection count
  n_isects = tiles_per_gauss.sum()                             # scalar
```

#### Phase 2: Intersection Enumeration (Semi-Vectorized)

This is the trickiest part. We need to expand each Gaussian into its list of (tile_x, tile_y) pairs. The challenge is that each Gaussian covers a variable number of tiles.

**Strategy**: Use `np.repeat` to replicate Gaussian metadata, then compute tile coordinates from local indices.

```
Step 2.1: Flatten and compute cumulative sums
  flat_tiles_per_gauss = tiles_per_gauss.flatten()    # [I*N]
  cum_tiles = np.cumsum(flat_tiles_per_gauss)          # [I*N]

Step 2.2: Create per-intersection metadata via np.repeat
  # For each Gaussian, we know:
  #   - image_id = flat_index // N
  #   - gauss_id = flat_index % N
  #   - tile_min_x, tile_min_y, tile_width_range, tile_height_range

  flat_indices = np.arange(I * N)                     # [I*N]
  image_ids = flat_indices // N                        # [I*N]
  gauss_ids = flat_indices % N                         # [I*N]

  # Repeat each entry by its tile count
  rep_image_ids = np.repeat(image_ids, flat_tiles_per_gauss)       # [n_isects]
  rep_gauss_ids = np.repeat(gauss_ids, flat_tiles_per_gauss)       # [n_isects]
  rep_tile_min_x = np.repeat(tile_mins[:,:,0].flatten(), flat_tiles_per_gauss)  # [n_isects]
  rep_tile_min_y = np.repeat(tile_mins[:,:,1].flatten(), flat_tiles_per_gauss)  # [n_isects]
  rep_tile_w = np.repeat(tile_ranges[:,:,0].flatten(), flat_tiles_per_gauss)    # [n_isects]

Step 2.3: Compute local tile offsets within each Gaussian's tile range
  # For each intersection entry, compute its local index within that Gaussian's tiles
  # local_idx goes 0, 1, ..., tiles_per_gauss[i]-1 for each Gaussian i
  local_idx = np.arange(n_isects) - np.repeat(
      np.concatenate([[0], cum_tiles[:-1]]), flat_tiles_per_gauss
  )                                                     # [n_isects]

  # Convert local_idx to (local_x, local_y) within the Gaussian's tile range
  local_x = local_idx % rep_tile_w                      # [n_isects]
  local_y = local_idx // rep_tile_w                     # [n_isects]

  tile_x = rep_tile_min_x + local_x                     # [n_isects]
  tile_y = rep_tile_min_y + local_y                     # [n_isects]

Step 2.4: Compute tile IDs
  tile_id = tile_y * tile_width + tile_x                # [n_isects]
```

#### Phase 3: Sort Key Construction and Sorting (Fully Vectorized)

```
Step 3.1: Compute bit layout
  tile_n_bits = (tile_width * tile_height).bit_length()

Step 3.2: Build high 32 bits
  isect_ids_hi = (rep_image_ids << tile_n_bits) | tile_id    # [n_isects] int32

Step 3.3: Build low 32 bits (depth as uint32)
  rep_depths = depths[rep_image_ids, rep_gauss_ids]           # [n_isects] float32
  isect_ids_lo = rep_depths.view(np.uint32)                   # [n_isects] uint32

Step 3.4: Combine into 64-bit keys
  isect_ids = (isect_ids_hi.astype(np.int64) << 32) | (isect_ids_lo.astype(np.int64) & 0xFFFFFFFF)
  # Shape: [n_isects] int64

Step 3.5: Build flatten_ids
  flatten_ids = rep_image_ids * N + rep_gauss_ids             # [n_isects] int32

Step 3.6: Sort by 64-bit key
  if sort:
      sort_indices = np.argsort(isect_ids, kind='stable')
      isect_ids = isect_ids[sort_indices]
      flatten_ids = flatten_ids[sort_indices]
```

#### Phase 4: Convert Back to MLX

```
Step 4.1: Reshape tiles_per_gauss to original batch dims
  tiles_per_gauss_mlx = mx.array(tiles_per_gauss.reshape(image_dims + (N,)), dtype=mx.int32)

Step 4.2: Convert intersection arrays
  # Option A: Single int64 array (if MLX supports it)
  isect_ids_mlx = mx.array(isect_ids)     # [n_isects] int64

  # Option B: Fallback to (hi, lo) pair
  # isect_ids_hi_mlx = mx.array(isect_ids_hi, dtype=mx.int32)
  # isect_ids_lo_mlx = mx.array(isect_ids_lo, dtype=mx.int32)

  flatten_ids_mlx = mx.array(flatten_ids, dtype=mx.int32)     # [n_isects] int32
```

### Detailed Algorithm for `isect_offset_encode`

```
Input:
  isect_ids: [n_isects] int64 (sorted)
  n_images: int (I)
  tile_width: int
  tile_height: int

Step 1: Extract high 32 bits (image_id << tile_n_bits | tile_id)
  tile_n_bits = (tile_width * tile_height).bit_length()
  hi_bits = isect_ids >> 32                               # [n_isects] int64

Step 2: Find unique (image, tile) pairs and their counts
  unique_hi, counts = np.unique(hi_bits, return_counts=True)
  # Since isect_ids is sorted, unique_consecutive == unique

Step 3: Decode image_id and tile coordinates
  image_ids = (unique_hi >> tile_n_bits).astype(np.int64)
  tile_ids = unique_hi & ((1 << tile_n_bits) - 1)
  tile_x = (tile_ids % tile_width).astype(np.int64)
  tile_y = (tile_ids // tile_width).astype(np.int64)

Step 4: Scatter counts into 3D array
  tile_counts = np.zeros((n_images, tile_height, tile_width), dtype=np.int64)
  tile_counts[image_ids, tile_y, tile_x] = counts

Step 5: Prefix sum to compute offsets
  cum_counts = np.cumsum(tile_counts.flatten()).reshape(tile_counts.shape)
  offsets = cum_counts - tile_counts
  # offsets[i, y, x] = number of intersections before tile (i, y, x) in the sorted list

Step 6: Convert to MLX
  return mx.array(offsets.astype(np.int32))    # [I, tile_height, tile_width] int32
```

### Data Flow

#### `isect_tiles` Input/Output

**Inputs** (from PRD-05 projection):
| Tensor | Shape | Dtype | Source |
|--------|-------|-------|--------|
| `means2d` | `[..., N, 2]` | float32 | `fully_fused_projection` |
| `radii` | `[..., N, 2]` | int32 | `fully_fused_projection` (0 = culled) |
| `depths` | `[..., N]` | float32 | `fully_fused_projection` (camera-space z) |
| `tile_size` | scalar | int | Typically 16 |
| `tile_width` | scalar | int | `ceil(image_width / tile_size)` |
| `tile_height` | scalar | int | `ceil(image_height / tile_size)` |
| `sort` | scalar | bool | Default True |

**Outputs** (consumed by PRD-07 rasterization):
| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `tiles_per_gauss` | `[..., N]` | int32 | Number of tiles each Gaussian intersects |
| `isect_ids` | `[n_isects]` | int64 | Packed sort keys: `(image_id << (tile_n_bits + 32)) \| (tile_id << 32) \| depth_bits` |
| `flatten_ids` | `[n_isects]` | int32 | Flattened Gaussian index `image_id * N + gauss_id` |

#### `isect_offset_encode` Input/Output

**Inputs**:
| Tensor | Shape | Dtype | Source |
|--------|-------|-------|--------|
| `isect_ids` | `[n_isects]` | int64 | From `isect_tiles` (must be sorted) |
| `n_images` | scalar | int | Number of images (I) |
| `tile_width` | scalar | int | Same as used in `isect_tiles` |
| `tile_height` | scalar | int | Same as used in `isect_tiles` |

**Outputs**:
| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `offsets` | `[I, tile_height, tile_width]` | int32 | Start index into sorted `flatten_ids` for each tile |

#### How Downstream Uses These Outputs

The rasterizer (PRD-07) processes each tile independently:
```python
for tile_y in range(tile_height):
    for tile_x in range(tile_width):
        start = offsets[image_id, tile_y, tile_x]
        end = offsets[image_id, tile_y, tile_x + 1]  # or next tile in flat order
        gauss_indices = flatten_ids[start:end]
        # gauss_indices are sorted front-to-back by depth
        # Composite these Gaussians for all pixels in this tile
```

### Typical Data Sizes

| Scenario | I | N | tile_size | tile_width | tile_height | n_tiles | avg_tiles/gauss | n_isects |
|----------|---|---|-----------|------------|-------------|---------|-----------------|----------|
| Small | 1 | 1,000 | 16 | 40 | 30 | 1,200 | 4 | ~4,000 |
| Medium | 1 | 50,000 | 16 | 40 | 30 | 1,200 | 4 | ~200,000 |
| Large | 1 | 500,000 | 16 | 120 | 68 | 8,160 | 6 | ~3,000,000 |
| Multi-cam | 4 | 100,000 | 16 | 40 | 30 | 1,200 | 4 | ~1,600,000 |

### 64-Bit Sort Key Layout

```
Bit 63                                                    Bit 0
|<--- image_id --->|<------- tile_id ------->|<------ depth_bits ------>|
|   variable bits  |     tile_n_bits         |        32 bits           |
|<----------- isect_ids_hi (32 bits) ------->|<-- isect_ids_lo (32) -->|
```

- `tile_n_bits = (tile_width * tile_height).bit_length()` -- e.g., for 1200 tiles, `bit_length() = 11`
- `image_n_bits = I.bit_length()` -- e.g., for I=4, `bit_length() = 3`
- Constraint: `image_n_bits + tile_n_bits + 32 <= 64` (always true for practical sizes)
- The high 32 bits encode `(image_id << tile_n_bits) | tile_id`
- The low 32 bits encode `float32_depth` reinterpreted as `uint32`
- Sorting this 64-bit key sorts by image first, then tile within image, then depth within tile

### IEEE 754 Depth Bit Reinterpretation

Positive IEEE 754 float32 values have the property that their bit patterns, when interpreted as unsigned integers, maintain the same ordering. This means:

```
depth=0.1  -> bits=0x3DCCCCCD -> uint32=1036831949
depth=0.5  -> bits=0x3F000000 -> uint32=1056964608
depth=1.0  -> bits=0x3F800000 -> uint32=1065353216
depth=10.0 -> bits=0x41200000 -> uint32=1092616192
```

Sorting these uint32 values produces the correct depth ordering (0.1 < 0.5 < 1.0 < 10.0). This trick avoids needing separate float sorting and lets us pack depth into the integer sort key.

**Negative depth handling**: Negative-depth Gaussians have `radii=0` from frustum culling (PRD-05) and are excluded by the `valid_mask = (radii > 0).all()` check. No special handling needed.

## File Structure

### `src/gsplat_mlx/core/intersection.py`

```python
"""Tile-Gaussian intersection and depth sorting for 3D Gaussian Splatting.

This module implements the acceleration structure that maps projected 2D Gaussians
to the screen tiles they overlap, sorted by depth within each tile. This enables
the rasterizer to process each tile independently with only its relevant Gaussians.

All operations are non-differentiable (no gradients needed).
"""

import math
import numpy as np
import mlx.core as mx


def isect_tiles(
    means2d: mx.array,    # [..., N, 2] float32
    radii: mx.array,      # [..., N, 2] int32
    depths: mx.array,     # [..., N]    float32
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
    means2d_np = np.array(means2d).reshape(I, N, 2)
    radii_np = np.array(radii).reshape(I, N, 2)
    depths_np = np.array(depths).reshape(I, N)

    tile_means = means2d_np / tile_size                       # [I, N, 2]
    tile_radii = radii_np / tile_size                         # [I, N, 2]
    tile_mins = np.floor(tile_means - tile_radii).astype(np.int32)  # [I, N, 2]
    tile_maxs = np.ceil(tile_means + tile_radii).astype(np.int32)   # [I, N, 2]

    tile_mins[..., 0] = np.clip(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = np.clip(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = np.clip(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = np.clip(tile_maxs[..., 1], 0, tile_height)

    tile_ranges = tile_maxs - tile_mins                       # [I, N, 2]
    tiles_per_gauss = tile_ranges.prod(axis=-1)               # [I, N]
    valid_mask = (radii_np > 0).all(axis=-1)                  # [I, N]
    tiles_per_gauss *= valid_mask                              # [I, N]

    n_isects = int(tiles_per_gauss.sum())

    # --- Phase 2: Enumerate intersections (semi-vectorized) ---
    if n_isects == 0:
        tiles_per_gauss_mlx = mx.array(
            tiles_per_gauss.reshape(image_dims + (N,)), dtype=mx.int32
        )
        isect_ids_mlx = mx.array(np.empty(0, dtype=np.int64))
        flatten_ids_mlx = mx.array(np.empty(0, dtype=np.int32))
        return tiles_per_gauss_mlx, isect_ids_mlx, flatten_ids_mlx

    flat_tiles_per_gauss = tiles_per_gauss.flatten()           # [I*N]

    # Replicate metadata for each intersection
    flat_indices = np.arange(I * N)
    image_ids_flat = flat_indices // N                         # [I*N]
    gauss_ids_flat = flat_indices % N                          # [I*N]
    tile_min_x_flat = tile_mins[..., 0].flatten()              # [I*N]
    tile_min_y_flat = tile_mins[..., 1].flatten()              # [I*N]
    tile_range_x_flat = tile_ranges[..., 0].flatten()          # [I*N]

    # np.repeat expands each entry by its tile count
    rep_image_ids = np.repeat(image_ids_flat, flat_tiles_per_gauss)     # [n_isects]
    rep_gauss_ids = np.repeat(gauss_ids_flat, flat_tiles_per_gauss)     # [n_isects]
    rep_tile_min_x = np.repeat(tile_min_x_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_tile_min_y = np.repeat(tile_min_y_flat, flat_tiles_per_gauss)  # [n_isects]
    rep_tile_range_x = np.repeat(tile_range_x_flat, flat_tiles_per_gauss)  # [n_isects]

    # Compute local index within each Gaussian's tile range
    cum_tiles = np.cumsum(flat_tiles_per_gauss)                # [I*N]
    offsets = np.zeros(I * N, dtype=np.int64)
    offsets[1:] = cum_tiles[:-1]
    rep_offsets = np.repeat(offsets, flat_tiles_per_gauss)      # [n_isects]
    local_idx = np.arange(n_isects) - rep_offsets              # [n_isects]

    # Convert local index to (tile_x, tile_y) within the Gaussian's tile range
    local_x = (local_idx % rep_tile_range_x).astype(np.int32)
    local_y = (local_idx // rep_tile_range_x).astype(np.int32)
    tile_x = rep_tile_min_x + local_x                         # [n_isects]
    tile_y = rep_tile_min_y + local_y                          # [n_isects]
    tile_id = tile_y * tile_width + tile_x                     # [n_isects]

    # --- Phase 3: Build sort keys (vectorized) ---
    tile_n_bits = (tile_width * tile_height).bit_length()

    isect_ids_hi = (rep_image_ids.astype(np.int32) << tile_n_bits) | tile_id.astype(np.int32)

    # Float-to-uint32 bit reinterpretation for depth
    rep_depths = depths_np[rep_image_ids, rep_gauss_ids].astype(np.float32)
    isect_ids_lo = rep_depths.view(np.uint32)                  # [n_isects]

    # Combine into 64-bit sort key
    isect_ids = (
        (isect_ids_hi.astype(np.int64) << 32)
        | (isect_ids_lo.astype(np.int64) & 0xFFFFFFFF)
    )

    # Build flatten_ids
    flatten_ids = (rep_image_ids * N + rep_gauss_ids).astype(np.int32)

    # --- Phase 4: Sort ---
    if sort:
        sort_indices = np.argsort(isect_ids, kind='stable')
        isect_ids = isect_ids[sort_indices]
        flatten_ids = flatten_ids[sort_indices]

    # --- Phase 5: Convert to MLX ---
    tiles_per_gauss_mlx = mx.array(
        tiles_per_gauss.reshape(image_dims + (N,)).astype(np.int32)
    )
    isect_ids_mlx = mx.array(isect_ids)        # int64
    flatten_ids_mlx = mx.array(flatten_ids)    # int32

    return tiles_per_gauss_mlx, isect_ids_mlx, flatten_ids_mlx


def isect_offset_encode(
    isect_ids: mx.array,   # [n_isects] int64
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
        hi_bits = isect_ids_np >> 32
        unique_hi, counts = np.unique(hi_bits, return_counts=True)

        image_ids = (unique_hi >> tile_n_bits).astype(np.int64)
        tile_ids = unique_hi & ((1 << tile_n_bits) - 1)
        tile_x = (tile_ids % tile_width).astype(np.int64)
        tile_y = (tile_ids // tile_width).astype(np.int64)

        tile_counts[image_ids, tile_y, tile_x] = counts

    cum_counts = np.cumsum(tile_counts.flatten()).reshape(tile_counts.shape)
    offsets = cum_counts - tile_counts

    return mx.array(offsets.astype(np.int32))
```

## Test Plan

### File: `tests/test_intersection.py`

#### Basic Intersection Tests

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_single_gaussian_single_tile` | One Gaussian at center of tile (0,0), small radius covering exactly 1 tile | `tiles_per_gauss == 1`, `n_isects == 1`, `flatten_ids == [0]` |
| `test_single_gaussian_multi_tile` | One Gaussian spanning a 2x2 tile area | `tiles_per_gauss == 4`, `n_isects == 4` |
| `test_multi_gaussian_overlap` | Two Gaussians both covering the same tile, at different depths | Both appear in intersection list, sorted by depth |
| `test_empty_tiles` | Gaussians only in top-left quadrant; bottom-right tiles empty | Offsets for empty tiles equal next tile's offset |
| `test_out_of_bounds_radii_zero` | Gaussian with `radii=[0, 0]` (culled by projection) | `tiles_per_gauss == 0`, not in intersection list |

#### Depth Sorting Tests

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_depth_sorting` | 5 Gaussians in same tile at depths [5.0, 1.0, 3.0, 2.0, 4.0] | After sorting, `flatten_ids` order corresponds to depths [1.0, 2.0, 3.0, 4.0, 5.0] |
| `test_depth_sort_float_bits` | Verify float-to-uint32 reinterpretation gives correct ordering | `depth.view(uint32)` is monotonically increasing for positive depths |
| `test_multi_image_sorting` | I=2 images, verify image_id boundary in sort keys | All intersections for image 0 come before image 1 |
| `test_sort_disabled` | `sort=False`, verify intersections still computed but not sorted | `n_isects` is same as sorted, order may differ |

#### Offset Encode Tests

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_offset_encode_basic` | Known tile layout -> verify offset values match expected | Exact integer match |
| `test_offset_encode_empty_tiles` | Grid with some empty tiles | Empty tiles have `offsets[tile] == offsets[next_tile]` (zero-width range) |
| `test_offset_encode_consistency` | For each tile: `offsets[tile] + count[tile] == offsets[next_tile]` | Prefix sum invariant holds |
| `test_offset_encode_monotonic` | Flattened offsets are monotonically non-decreasing | `np.all(np.diff(offsets.flatten()) >= 0)` |
| `test_offset_encode_total` | Last offset + last count == n_isects | Sum of all tile counts equals total intersections |

#### Edge Cases

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_zero_gaussians` | N=0, no Gaussians at all | `n_isects == 0`, empty arrays returned |
| `test_large_tile_size` | `tile_size > image_width` -> single tile covers entire image | `tile_width == 1, tile_height == 1` |
| `test_all_behind_camera` | All Gaussians have `radii=0` (behind near plane) | `n_isects == 0`, all `tiles_per_gauss == 0` |
| `test_gaussian_at_tile_boundary` | Gaussian centered exactly on tile edge | Correct assignment to adjacent tiles |
| `test_gaussian_covers_entire_screen` | Very large radius covering all tiles | `tiles_per_gauss == tile_width * tile_height` |
| `test_many_gaussians` | N=10,000 random Gaussians | `n_isects > 0`, no crashes, reasonable total count |

#### Cross-Framework Tests (requires torch)

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_cross_framework_isect_tiles` | 500 random Gaussians, compare MLX vs torch `_isect_tiles` | Exact match: `tiles_per_gauss`, sort-stable `flatten_ids` per tile |
| `test_cross_framework_isect_offset` | Compare offset tables for same input | Exact match: `offsets` |
| `test_cross_framework_multi_image` | I=3 images, 200 Gaussians each | Exact match on all outputs |

#### Tolerances

All outputs are integers. All assertions use **exact match** (`==`), no floating-point tolerance needed:
- `tiles_per_gauss`: exact int32 match
- `isect_ids`: exact int64 match (same bit patterns)
- `flatten_ids`: exact int32 match (after sorting)
- `offsets`: exact int32 match

The only floating-point operation is the depth bit reinterpretation, which is exact by definition (same bits, different type interpretation).

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `src/gsplat_mlx/core/intersection.py`
- [ ] Implement `isect_tiles` with vectorized NumPy algorithm
- [ ] Implement `isect_offset_encode` with prefix-sum approach
- [ ] Handle n_isects=0 edge case (empty arrays)
- [ ] Verify int64 MLX array creation works (add fallback if needed)

### Phase 2: Testing
- [ ] Create `tests/test_intersection.py`
- [ ] Implement all basic intersection tests
- [ ] Implement all depth sorting tests
- [ ] Implement all offset encode tests
- [ ] Implement edge case tests
- [ ] Implement cross-framework tests (mark with `@pytest.mark.requires_torch`)

### Phase 3: Integration
- [ ] Export `isect_tiles` and `isect_offset_encode` from `core/__init__.py`
- [ ] Verify PRD-05 projection outputs feed correctly into intersection inputs
- [ ] Document expected input ranges and constraints

## Dependencies

- **PRD-01**: Dev environment, test infrastructure, `conftest.py`
- **PRD-05**: Projection produces `means2d`, `radii`, `depths` that are the inputs to intersection

## Blocks

- **PRD-07**: Rasterization uses `flatten_ids` and `offsets` to iterate sorted Gaussians per tile
- **PRD-08**: Accumulate uses similar tile-based indexing
- **PRD-09**: Top-level rendering API orchestrates projection -> intersection -> rasterization

## Acceptance Criteria

- [ ] `isect_tiles` produces correct tile-Gaussian intersection lists for all test cases
- [ ] Intersections are correctly sorted by (image_id, tile_id, depth) when `sort=True`
- [ ] Float-to-uint32 depth encoding preserves correct depth ordering for positive depths
- [ ] `isect_offset_encode` produces valid prefix-sum offsets (monotonically non-decreasing)
- [ ] Offsets correctly index into sorted `flatten_ids` for per-tile Gaussian lookup
- [ ] Handles edge cases: N=0, all culled, single tile, entire-screen coverage
- [ ] No gradient computation needed (verified: no `@mx.custom_function`, no backward pass)
- [ ] Vectorized NumPy implementation (no Python-level per-Gaussian loops)
- [ ] Matches torch `_isect_tiles` and `_isect_offset_encode` exactly for random inputs
- [ ] All tests pass with `pytest tests/test_intersection.py -v`
