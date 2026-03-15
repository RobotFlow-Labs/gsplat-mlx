# PRD-14: Metal Compute Shaders for Interactive Rasterization

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-14 |
| **Title** | Metal Compute Shaders for Interactive Rasterization |
| **Status** | DRAFT |
| **Priority** | P2 -- Future Performance Optimization (Phase 6) |
| **Estimated Effort** | 4--8 weeks |
| **Dependencies** | PRD-01 through PRD-13 (complete Python pipeline as reference) |
| **Blocks** | Nothing (this is the performance optimization phase) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Replace the Python-loop-based rasterization pipeline (PRD-07) with Metal compute shaders running natively on Apple Silicon GPUs. The Python implementation is correct but fundamentally bottlenecked by interpreter overhead -- nested Python loops over tiles, pixels, and Gaussians cannot achieve interactive framerates regardless of NumPy/MLX vectorization tricks.

After this PRD is implemented, the gsplat-mlx renderer should achieve:

- **30+ FPS** at 512x512 resolution with 100K Gaussians on M3 Pro
- **60+ FPS** at 256x256 resolution with 50K Gaussians on M3 Pro
- **Real-time preview** during training at reduced resolution
- **Full backward pass** on GPU for training loops without CPU roundtrips

This PRD is NOT required for MVP correctness. The Python reference (PRD-07) establishes correctness; this PRD establishes performance.

---

## 2. Context & Motivation

### 2.1 Why Python Is the Bottleneck

The PRD-07 rasterization implementation uses nested Python loops:

```
for each image (I):
    for each tile (tile_height * tile_width):
        for each pixel in tile (tile_size^2):
            for each Gaussian in tile (K):
                compute sigma, alpha, accumulate color
```

For a 512x512 image with 16x16 tiles and 100K Gaussians (average 50 per tile), this is:

- 1024 tiles x 256 pixels x 50 Gaussians = ~13 million Python-level iterations
- Each iteration involves float math, conditionals, array indexing
- At ~100ns per Python iteration: ~1.3 seconds per frame
- Target: 33ms per frame (30 FPS) -- need **40x** improvement minimum

Even vectorized MLX approaches (computing all pixels in a tile simultaneously) still require a sequential loop over the Gaussian dimension for front-to-back alpha compositing, and a Python loop over tiles.

### 2.2 Why Metal Specifically

The CUDA equivalent (`RasterizeToPixels3DGSFwd.cu`) achieves sub-millisecond rasterization on NVIDIA hardware by:

1. Launching one threadgroup (block) per tile -- all tiles run in parallel
2. Using shared memory to batch-load Gaussian data cooperatively
3. Each thread composites one pixel sequentially over Gaussians
4. Early termination per-thread when transmittance drops below threshold

Metal on Apple Silicon provides the same primitives:

| CUDA Concept | Metal Equivalent |
|--------------|-----------------|
| Block | Threadgroup |
| Thread | Thread |
| `__shared__` memory | `threadgroup` memory |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__syncthreads_count()` | Manual reduction via `threadgroup` atomic or simdgroup vote |
| `__expf()` | `metal::exp()` or `metal::fast::exp()` |
| `gpuAtomicAdd` | `atomic_fetch_add_explicit` |
| Warp shuffle/reduce | `simd_sum`, `simd_max`, simdgroup operations |
| `cooperative_groups` | Simdgroups (32 threads) |

### 2.3 MLX Integration Paths

MLX provides two ways to integrate custom Metal kernels:

**Path A: `mx.fast.metal_kernel()` (Python-inline MSL)**
- Kernel source is a string embedded in Python
- MLX auto-generates the function signature from input/output names
- Supports `atomic_outputs`, `init_value`, templates
- Grid/threadgroup dispatch via `dispatchThreads`
- Automatically includes `mlx/backend/metal/kernels/utils.h` for helper functions
- Automatically adds shape/strides/ndim for each input if referenced in source
- Best for: prototyping, simpler kernels, rapid iteration

**Path B: C++ Extension with `mlx::core::Primitive`**
- Full control over Metal compute command encoding
- Can set threadgroup memory size explicitly via `setThreadgroupMemoryLength:atIndex:`
- Can use Metal library precompilation
- Can encode multiple dispatch calls
- Best for: production kernels, complex shared memory patterns, maximum performance

**Recommendation**: Start with Path A for the forward kernel prototype. Move to Path B if threadgroup memory limitations or performance tuning requires it. The backward kernel (which needs atomics + shared memory for colors) should use Path B from the start.

### 2.4 Upstream CUDA Architecture Reference

The upstream gsplat CUDA implementation (from `RasterizeToPixels3DGSFwd.cu`) uses this pattern:

```
Grid:  (I, tile_height, tile_width)     -- one block per (image, tile)
Block: (tile_size, tile_size, 1)         -- one thread per pixel in tile (16x16 = 256)

Shared memory per block:
  id_batch:         [block_size] int32           -- Gaussian IDs
  xy_opacity_batch: [block_size] vec3 (float3)   -- (mean2d.x, mean2d.y, opacity)
  conic_batch:      [block_size] vec3 (float3)   -- (conic.x, conic.y, conic.z)
  Total: block_size * (4 + 12 + 12) = 256 * 28 = 7168 bytes

Algorithm:
  1. Determine pixel (i, j) and tile_id from thread/block indices
  2. Look up range [start, end) in flatten_ids for this tile via tile_offsets
  3. Process Gaussians in batches of block_size:
     a. Each thread cooperatively loads ONE Gaussian into shared memory
     b. threadgroup_barrier sync
     c. Each thread iterates over all block_size Gaussians in shared memory
     d. For each Gaussian: compute sigma, alpha, accumulate color
     e. Early exit if __syncthreads_count(done) == block_size (all pixels done)
  4. Write final color + alpha + last_ids to output
```

The backward kernel (`RasterizeToPixels3DGSBwd.cu`) iterates in **reverse order** (back-to-front), recomputing transmittance from `T_final` and accumulating gradients with atomic adds. It uses warp-level reductions (`warpSum` via cooperative_groups) before atomics to reduce contention by 32x.

### 2.5 Upstream Constants

From `gsplat/cuda/include/Common.h`:

```c
#define ALPHA_THRESHOLD        (1.f / 255.f)   // Minimum alpha to process a Gaussian
#define MAX_ALPHA              0.99f            // Clamp alpha to prevent fully opaque
#define TRANSMITTANCE_THRESHOLD 1e-4f           // Early termination when pixel is ~opaque
```

---

## 3. Scope

### 3.1 In Scope

| Priority | Component | CUDA Equivalent | Effort |
|----------|-----------|-----------------|--------|
| **P0** | Forward rasterization kernel | `RasterizeToPixels3DGSFwd.cu` | 2 weeks |
| **P0** | Backward rasterization kernel | `RasterizeToPixels3DGSBwd.cu` | 2 weeks |
| **P1** | Gaussian projection kernel | `ProjectionEWA3DGSFused.cu` | 1 week |
| **P1** | Tile intersection + sort kernel | `IntersectTile.cu` | 1 week |
| **P2** | Performance benchmarking harness | N/A | 3 days |
| **P2** | Fallback dispatcher (Metal vs Python) | N/A | 2 days |

### 3.2 Out of Scope

- 2DGS rasterization kernels (`RasterizeToPixels2DGS*.cu`)
- Packed input format support
- Lidar camera model kernels
- Multi-GPU / distributed rendering
- MPS (Metal Performance Shaders) library integration for sorting (evaluate but not required)
- Unscented transform projection kernel
- Tile masks for selective rendering (add later as incremental optimization)
- `absgrad` mode (absolute gradient for densification -- add after base backward works)

---

## 4. Technical Design: Forward Rasterization Kernel

This is the highest-priority kernel and the core of this PRD. The forward rasterization kernel is the single biggest bottleneck in the rendering pipeline.

### 4.1 Threadgroup Sizing Analysis

| Tile Size | Threads/Threadgroup | Threadgroup Memory (FWD) | Threadgroup Memory (BWD, C=3) | Notes |
|-----------|--------------------|--------------------------|-----------------------------|-------|
| 8x8 | 64 | 1,792 B | 2,560 B | Low occupancy. Too few threads to hide memory latency. |
| **16x16** | **256** | **7,168 B** | **10,240 B** | **Optimal.** Matches CUDA default. Good occupancy. |
| 32x32 | 1024 | 28,672 B | 40,960 B (exceeds 32 KB for C>2!) | Max Metal threadgroup size. Risky for backward. |

**Recommendation: 16x16 (256 threads per threadgroup)**

Rationale:
- Apple Silicon GPUs have 32 KB of threadgroup memory per threadgroup. At 7,168 bytes (forward) we use only 22%, leaving headroom for the backward kernel.
- 256 threads = 8 simdgroups (Apple GPU simdgroup width = 32). M1/M2/M3 can execute multiple simdgroups concurrently per execution unit.
- 16x16 tiles match the upstream CUDA default, ensuring comparable tile-to-Gaussian ratios and making correctness validation easier.
- A 512x512 image yields 1024 tiles = 1024 threadgroups, sufficient to saturate even M3 Max.

**Apple Silicon GPU Specs (relevant to occupancy):**

| Chip | GPU Cores | Max Concurrent Threadgroups | Threadgroup Mem | Peak TFLOPS (FP32) |
|------|-----------|---------------------------|-----------------|---------------------|
| M1 | 7--8 | ~1024 | 32 KB | 2.6 |
| M2 | 8--10 | ~1024 | 32 KB | 3.6 |
| M3 | 10 | ~1024 | 32 KB | 4.1 |
| M3 Pro | 14--18 | ~2048 | 32 KB | 7.4 |
| M3 Max | 30--40 | ~4096 | 32 KB | 14.2 |
| M4 | 10 | ~1024 | 32 KB | 4.6 |

At 16x16 tiles with 7 KB threadgroup memory, we can have `floor(32768 / 7168) = 4` threadgroups resident per compute unit (limited by threadgroup memory). With 256 threads each, that is 1024 threads per compute unit -- near maximum occupancy on Apple Silicon.

### 4.2 Shared Memory Layout (Threadgroup Memory)

```
Forward kernel threadgroup memory (7168 bytes for block_size=256):
┌─────────────────────────────────────────────────┐
│ id_batch[256]        : int32   = 1024 bytes     │  Gaussian flat IDs for color lookup
├─────────────────────────────────────────────────┤
│ xy_opacity_batch[256]: float3  = 3072 bytes     │  (mean2d.x, mean2d.y, opacity)
├─────────────────────────────────────────────────┤
│ conic_batch[256]     : float3  = 3072 bytes     │  (conic.x, conic.y, conic.z)
└─────────────────────────────────────────────────┘
Total: 7168 bytes  (22% of 32 KB limit)
```

**Why shared memory matters**: Without it, each of the 256 threads loads the same Gaussian data independently from global memory. With shared memory, each thread loads ONE Gaussian, then all 256 threads read from fast threadgroup memory. For a tile with 256 Gaussians processed in one batch, this is a **256x reduction** in global memory bandwidth.

**Metal-specific notes:**
- Metal `threadgroup` memory is statically or dynamically allocated per-threadgroup.
- For the C++ extension path, set via `setThreadgroupMemoryLength:atIndex:` on the compute command encoder.
- For `mx.fast.metal_kernel()`, threadgroup memory can be declared inline in the shader source using `threadgroup` storage qualifier with fixed sizes.

### 4.3 Complete Forward Metal Shader

This is a direct translation of `RasterizeToPixels3DGSFwd.cu`, adapted for Metal Shading Language.

```metal
// ============================================================================
// rasterize_fwd.metal
// Metal compute shader for 3DGS tile-based rasterization (forward pass)
//
// Translates from: gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
// Architecture: One threadgroup per tile. One thread per pixel.
//               Cooperative Gaussian loading into threadgroup memory.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ── Constants (matching gsplat/cuda/include/Common.h) ──
constant float ALPHA_THRESHOLD         = 1.0f / 255.0f;
constant float MAX_ALPHA               = 0.99f;
constant float TRANSMITTANCE_THRESHOLD = 1e-4f;

// ── Maximum supported channels (compile-time constant for stack arrays) ──
// For RGB: 3. For feature rendering: up to 32.
// If more channels are needed, increase this or template the kernel.
constant uint MAX_CHANNELS = 32;

// ── Tile size (compile-time constant for threadgroup memory sizing) ──
// This must match the threadgroup dimensions used at dispatch time.
constant uint TILE_SIZE  = 16;
constant uint BLOCK_SIZE = TILE_SIZE * TILE_SIZE;  // 256

// ── Kernel parameters (passed as constant buffer) ──
struct RasterizeParams {
    uint image_width;
    uint image_height;
    uint tile_width;      // ceil(image_width / TILE_SIZE)
    uint tile_height;     // ceil(image_height / TILE_SIZE)
    uint n_channels;      // color channels (typically 3)
    uint N;               // Gaussians per image
    uint I;               // number of images in batch
    uint n_isects;        // total entries in flatten_ids
};

// ============================================================================
// Forward rasterization kernel
// ============================================================================
//
// Grid dispatch:
//   grid         = (tile_width, tile_height, I)  -- one threadgroup per tile per image
//   threadgroup  = (TILE_SIZE, TILE_SIZE, 1)     -- one thread per pixel in tile
//
// Each threadgroup processes all Gaussians intersecting its tile,
// loading them cooperatively in batches of BLOCK_SIZE into threadgroup memory.
// Each thread composites its pixel front-to-back with early termination.

kernel void rasterize_to_pixels_3dgs_fwd(
    // ── Input buffers ──
    device const float2*  means2d         [[buffer(0)]],   // [I*N, 2]  2D centers
    device const float3*  conics          [[buffer(1)]],   // [I*N, 3]  inverse covariance (a,b,c)
    device const float*   colors          [[buffer(2)]],   // [I*N, C]  per-Gaussian color
    device const float*   opacities       [[buffer(3)]],   // [I*N]     per-Gaussian opacity
    device const int*     tile_offsets    [[buffer(4)]],   // [I * tile_H * tile_W]  prefix-sum offsets
    device const int*     flatten_ids     [[buffer(5)]],   // [n_isects]  sorted Gaussian indices
    device const float*   backgrounds     [[buffer(6)]],   // [I * C]   background color per image
    // ── Output buffers ──
    device float*         render_colors   [[buffer(7)]],   // [I * H * W * C]
    device float*         render_alphas   [[buffer(8)]],   // [I * H * W]
    device int*           last_ids        [[buffer(9)]],   // [I * H * W]  index of last contributing Gaussian
    // ── Parameters ──
    constant RasterizeParams& params      [[buffer(10)]],
    constant uint& has_background         [[buffer(11)]],  // 1 if backgrounds buffer is valid, 0 otherwise
    // ── Thread indexing ──
    uint3 threadgroup_pos   [[threadgroup_position_in_grid]],     // (tile_x, tile_y, image_id)
    uint3 thread_pos        [[thread_position_in_threadgroup]],   // (local_x, local_y, 0)
    uint  thread_rank       [[thread_index_in_threadgroup]]       // linear index within threadgroup
) {
    // ────────────────────────────────────────────────
    // 1. Determine pixel coordinates and tile identity
    // ────────────────────────────────────────────────
    const uint tile_x   = threadgroup_pos.x;
    const uint tile_y   = threadgroup_pos.y;
    const uint image_id = threadgroup_pos.z;
    const uint local_x  = thread_pos.x;
    const uint local_y  = thread_pos.y;

    const uint px = tile_x * TILE_SIZE + local_x;
    const uint py = tile_y * TILE_SIZE + local_y;
    const bool inside = (px < params.image_width) && (py < params.image_height);

    const float pixel_x = float(px) + 0.5f;
    const float pixel_y = float(py) + 0.5f;

    const uint tile_id = tile_y * params.tile_width + tile_x;

    // ────────────────────────────────────────────────
    // 2. Compute Gaussian range for this tile
    // ────────────────────────────────────────────────
    // tile_offsets is a prefix-sum table: [I, tile_height, tile_width]
    // The range of flatten_ids for this tile is [start, end).
    const uint flat_tile = image_id * params.tile_height * params.tile_width + tile_id;

    const int range_start = tile_offsets[flat_tile];
    int range_end;
    // Last tile of last image: end is n_isects
    if ((image_id == params.I - 1) &&
        (tile_id == params.tile_width * params.tile_height - 1)) {
        range_end = int(params.n_isects);
    } else {
        range_end = tile_offsets[flat_tile + 1];
    }

    const uint num_gaussians_in_tile = uint(max(0, range_end - range_start));
    const uint num_batches = (num_gaussians_in_tile + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ────────────────────────────────────────────────
    // 3. Threadgroup (shared) memory declaration
    // ────────────────────────────────────────────────
    // Layout: contiguous arrays for cooperative loading
    threadgroup int    tg_id_batch[BLOCK_SIZE];          // 1024 bytes
    threadgroup float3 tg_xy_opacity_batch[BLOCK_SIZE];  // 3072 bytes (mean2d.x, mean2d.y, opacity)
    threadgroup float3 tg_conic_batch[BLOCK_SIZE];       // 3072 bytes (conic.x, conic.y, conic.z)
    // Total: 7168 bytes per threadgroup

    // ────────────────────────────────────────────────
    // 4. Per-pixel state
    // ────────────────────────────────────────────────
    float T = 1.0f;                 // transmittance (starts fully transparent)
    uint  cur_idx = 0;              // index of most recent contributing Gaussian
    bool  done = !inside;           // threads outside image boundary are pre-done

    // Per-pixel color accumulator (stack-allocated)
    float pix_out[MAX_CHANNELS];
    for (uint c = 0; c < params.n_channels && c < MAX_CHANNELS; c++) {
        pix_out[c] = 0.0f;
    }

    // ────────────────────────────────────────────────
    // 5. Main rasterization loop: batched cooperative loading
    // ────────────────────────────────────────────────
    // Process Gaussians in batches of BLOCK_SIZE.
    // In each batch:
    //   a) Every thread loads ONE Gaussian into threadgroup memory (cooperative)
    //   b) Barrier sync
    //   c) Every thread iterates over all loaded Gaussians for its pixel
    //   d) Barrier sync before next batch

    for (uint b = 0; b < num_batches; b++) {
        // ── Sync before loading next batch ──
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Early exit optimization ──
        // CUDA uses __syncthreads_count(done) to exit when all threads are done.
        // Metal lacks this intrinsic. We use a threadgroup atomic counter.
        // For the initial implementation, we skip this optimization and always
        // process all batches. The per-thread `done` flag still prevents
        // unnecessary computation; we just don't skip the shared memory loads.
        //
        // TODO: Implement threadgroup-level early exit via:
        //   threadgroup atomic_uint done_count;
        //   atomic_fetch_add_explicit(&done_count, done ? 1 : 0, memory_order_relaxed);
        //   threadgroup_barrier(mem_flags::mem_threadgroup);
        //   if (atomic_load_explicit(&done_count, memory_order_relaxed) >= BLOCK_SIZE) break;
        //   atomic_store_explicit(&done_count, 0, memory_order_relaxed);

        // ── Each thread cooperatively loads ONE Gaussian ──
        const uint batch_start = uint(range_start) + BLOCK_SIZE * b;
        const uint idx = batch_start + thread_rank;
        if (idx < uint(range_end)) {
            const int g = flatten_ids[idx];
            tg_id_batch[thread_rank] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            tg_xy_opacity_batch[thread_rank] = float3(xy.x, xy.y, opac);
            tg_conic_batch[thread_rank] = conics[g];
        }

        // ── Wait for all threads to finish loading ──
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Process loaded Gaussians ──
        const uint batch_size = min(BLOCK_SIZE, uint(range_end) - batch_start);
        for (uint t = 0; t < batch_size && !done; t++) {
            const float3 con     = tg_conic_batch[t];
            const float3 xy_opac = tg_xy_opacity_batch[t];
            const float  opac    = xy_opac.z;

            const float dx = xy_opac.x - pixel_x;
            const float dy = xy_opac.y - pixel_y;

            // Compute Mahalanobis distance (exponent of 2D Gaussian)
            const float sigma = 0.5f * (con.x * dx * dx + con.z * dy * dy)
                              + con.y * dx * dy;

            // Skip if sigma is negative (numerical issue with covariance)
            if (sigma < 0.0f) continue;

            // Compute alpha: opacity * Gaussian response, clamped
            float alpha = min(MAX_ALPHA, opac * metal::fast::exp(-sigma));

            // Skip near-zero alpha (below visibility threshold)
            if (alpha < ALPHA_THRESHOLD) continue;

            // Check transmittance: if pixel is effectively opaque, stop
            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) {
                // This pixel is done (exclusive: this Gaussian does NOT contribute)
                done = true;
                break;
            }

            // ── Accumulate color ──
            const int g = tg_id_batch[t];
            const float vis = alpha * T;
            const uint color_base = uint(g) * params.n_channels;
            for (uint c = 0; c < params.n_channels && c < MAX_CHANNELS; c++) {
                pix_out[c] += vis * colors[color_base + c];
            }

            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    // ────────────────────────────────────────────────
    // 6. Write output
    // ────────────────────────────────────────────────
    if (inside) {
        const uint pix_id = py * params.image_width + px;
        const uint pix_offset = image_id * params.image_height * params.image_width + pix_id;

        // Write per-channel color with background blending
        const uint color_out_base = pix_offset * params.n_channels;
        for (uint c = 0; c < params.n_channels && c < MAX_CHANNELS; c++) {
            float bg = 0.0f;
            if (has_background != 0) {
                bg = backgrounds[image_id * params.n_channels + c];
            }
            render_colors[color_out_base + c] = pix_out[c] + T * bg;
        }

        // Write alpha (1 - final transmittance)
        render_alphas[pix_offset] = 1.0f - T;

        // Write last contributing Gaussian index (used by backward pass)
        last_ids[pix_offset] = int(cur_idx);
    }
}
```

### 4.4 MLX Python Integration (Forward)

Two integration approaches, from simplest to most performant:

#### 4.4.1 Approach A: `mx.fast.metal_kernel()` (No Shared Memory)

This version trades shared memory for simplicity. Each thread loads Gaussians independently from global memory. Suitable for prototyping and small scenes.

```python
# src/gsplat_mlx/metal/rasterize_fwd_inline.py

import mlx.core as mx
import numpy as np

_RASTERIZE_FWD_SIMPLE_SOURCE = """
    // ── Thread positions ──
    uint image_id = threadgroup_position_in_grid.z;
    uint tile_x   = threadgroup_position_in_grid.x;
    uint tile_y   = threadgroup_position_in_grid.y;

    uint px = tile_x * tile_size[0] + thread_position_in_threadgroup.x;
    uint py = tile_y * tile_size[0] + thread_position_in_threadgroup.y;
    bool inside = (px < image_width[0]) && (py < image_height[0]);

    float pixel_x = float(px) + 0.5f;
    float pixel_y = float(py) + 0.5f;

    // ── Tile range lookup ──
    uint tile_id = tile_y * tile_w[0] + tile_x;
    uint flat_tile = image_id * tile_h[0] * tile_w[0] + tile_id;
    int range_start = tile_offsets[flat_tile];
    int range_end;
    if ((image_id == num_images[0] - 1) &&
        (tile_id == tile_w[0] * tile_h[0] - 1)) {
        range_end = int(total_isects[0]);
    } else {
        range_end = tile_offsets[flat_tile + 1];
    }

    // ── Per-pixel compositing (no shared memory) ──
    float T = 1.0f;
    float color_r = 0.0f, color_g = 0.0f, color_b = 0.0f;

    for (int i = range_start; i < range_end; i++) {
        int g = flatten_ids[i];

        float dx = means2d[g * 2 + 0] - pixel_x;
        float dy = means2d[g * 2 + 1] - pixel_y;

        float con_a = conics[g * 3 + 0];
        float con_b = conics[g * 3 + 1];
        float con_c = conics[g * 3 + 2];

        float sigma = 0.5f * (con_a * dx * dx + con_c * dy * dy)
                    + con_b * dx * dy;
        if (sigma < 0.0f) continue;

        float alpha = min(0.99f, opacities[g] * metal::fast::exp(-sigma));
        if (alpha < 1.0f / 255.0f) continue;

        float next_T = T * (1.0f - alpha);
        if (next_T <= 1e-4f) break;

        float vis = alpha * T;
        color_r += vis * colors[g * 3 + 0];
        color_g += vis * colors[g * 3 + 1];
        color_b += vis * colors[g * 3 + 2];
        T = next_T;
    }

    // ── Write output ──
    if (inside) {
        uint pix_id = py * image_width[0] + px;
        uint pix_offset = image_id * image_height[0] * image_width[0] + pix_id;

        render_colors[pix_offset * 3 + 0] = color_r + T * bg_color[0];
        render_colors[pix_offset * 3 + 1] = color_g + T * bg_color[1];
        render_colors[pix_offset * 3 + 2] = color_b + T * bg_color[2];
        render_alphas[pix_offset] = 1.0f - T;
    }
"""

# Build kernel once, reuse many times
_fwd_kernel_simple = None

def _get_fwd_kernel_simple():
    global _fwd_kernel_simple
    if _fwd_kernel_simple is None:
        _fwd_kernel_simple = mx.fast.metal_kernel(
            name="rasterize_fwd_simple",
            input_names=[
                "means2d", "conics", "colors", "opacities",
                "tile_offsets", "flatten_ids",
                "image_width", "image_height", "tile_size",
                "tile_w", "tile_h",
                "num_images", "total_isects",
                "bg_color",
            ],
            output_names=["render_colors", "render_alphas"],
            source=_RASTERIZE_FWD_SIMPLE_SOURCE,
        )
    return _fwd_kernel_simple


def rasterize_to_pixels_metal_simple(
    means2d, conics, colors, opacities,
    image_width, image_height, tile_size,
    isect_offsets, flatten_ids,
    backgrounds=None,
):
    """Metal forward rasterization (simple version, no shared memory).

    Drop-in replacement for Python rasterize_to_pixels() from PRD-07.
    """
    kernel = _get_fwd_kernel_simple()

    I = means2d.shape[0] if means2d.ndim == 3 else 1
    tile_h = (image_height + tile_size - 1) // tile_size
    tile_w = (image_width + tile_size - 1) // tile_size
    n_isects = flatten_ids.shape[0]

    # Flatten inputs
    means2d_flat = means2d.reshape(-1).astype(mx.float32)    # [I*N*2]
    conics_flat = conics.reshape(-1).astype(mx.float32)       # [I*N*3]
    colors_flat = colors.reshape(-1).astype(mx.float32)       # [I*N*3]
    opacities_flat = opacities.reshape(-1).astype(mx.float32) # [I*N]
    offsets_flat = isect_offsets.reshape(-1).astype(mx.int32)
    flatten_ids_i32 = flatten_ids.astype(mx.int32)

    bg = mx.zeros((3,), dtype=mx.float32)
    if backgrounds is not None:
        bg = backgrounds.reshape(-1)[:3].astype(mx.float32)

    outputs = kernel(
        inputs=[
            means2d_flat, conics_flat, colors_flat, opacities_flat,
            offsets_flat, flatten_ids_i32,
            mx.array([image_width], mx.int32),
            mx.array([image_height], mx.int32),
            mx.array([tile_size], mx.int32),
            mx.array([tile_w], mx.int32),
            mx.array([tile_h], mx.int32),
            mx.array([I], mx.int32),
            mx.array([n_isects], mx.int32),
            bg,
        ],
        output_shapes=[
            (I * image_height * image_width * 3,),
            (I * image_height * image_width,),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(tile_w * tile_size, tile_h * tile_size, I),
        threadgroup=(tile_size, tile_size, 1),
    )

    render_colors = outputs[0].reshape(I, image_height, image_width, 3)
    render_alphas = outputs[1].reshape(I, image_height, image_width, 1)
    return render_colors, render_alphas
```

#### 4.4.2 Approach B: C++ Extension (With Shared Memory)

For the production kernel with threadgroup memory:

```cpp
// src/gsplat_mlx/metal/rasterize_ext.cpp
// MLX C++ extension for Metal rasterization with shared memory

#include "mlx/mlx.h"
#include <Metal/Metal.h>

class RasterizeFwd : public mlx::core::Primitive {
public:
    explicit RasterizeFwd(mlx::core::Stream stream)
        : mlx::core::Primitive(stream) {}

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs
    ) override {
        // Extract inputs
        auto& means2d      = inputs[0];  // [I*N, 2]
        auto& conics        = inputs[1];  // [I*N, 3]
        auto& colors        = inputs[2];  // [I*N, C]
        auto& opacities     = inputs[3];  // [I*N]
        auto& tile_offsets  = inputs[4];  // [I*tile_H*tile_W]
        auto& flatten_ids   = inputs[5];  // [n_isects]
        auto& backgrounds   = inputs[6];  // [I*C]

        // Get Metal device and command buffer
        auto& s = stream();
        auto& d = mlx::core::metal::device(s.device);

        // Load pre-compiled Metal library
        auto lib = d.get_library("gsplat_rasterize");
        auto kernel = d.get_kernel("rasterize_to_pixels_3dgs_fwd", lib);

        // Set up compute command encoder
        auto compute_encoder = d.get_command_encoder(s.index);
        compute_encoder->setComputePipelineState(kernel);

        // Bind buffers
        compute_encoder->setBuffer(means2d, 0);
        compute_encoder->setBuffer(conics, 1);
        compute_encoder->setBuffer(colors, 2);
        compute_encoder->setBuffer(opacities, 3);
        compute_encoder->setBuffer(tile_offsets, 4);
        compute_encoder->setBuffer(flatten_ids, 5);
        compute_encoder->setBuffer(backgrounds, 6);
        compute_encoder->setBuffer(outputs[0], 7);  // render_colors
        compute_encoder->setBuffer(outputs[1], 8);  // render_alphas
        compute_encoder->setBuffer(outputs[2], 9);  // last_ids

        // Set params buffer
        RasterizeParams params = { ... };
        compute_encoder->setBytes(&params, sizeof(params), 10);

        // Set threadgroup memory size
        // 256 * (4 + 12 + 12) = 7168 bytes
        compute_encoder->setThreadgroupMemoryLength(7168, 0);

        // Dispatch
        MTLSize grid(tile_width, tile_height, I);
        MTLSize threadgroup(TILE_SIZE, TILE_SIZE, 1);
        compute_encoder->dispatchThreadgroups(grid, threadgroup);
    }
};
```

---

## 5. Technical Design: Backward Rasterization Kernel

### 5.1 Algorithm Design

The backward pass is more complex than the forward because it must:

1. **Iterate Gaussians in reverse order** (back-to-front, opposite of forward)
2. **Recompute transmittance** from `T_final` stored during forward pass
3. **Accumulate per-Gaussian gradients** from multiple pixels using atomic adds
4. **Compute gradients for five quantities**: `v_colors`, `v_opacities`, `v_means2d`, `v_conics`, and optionally `v_means2d_abs`

The key insight from the CUDA backward kernel: iterating back-to-front allows recovering per-Gaussian transmittance via `T = T / (1 - alpha)` without storing it.

### 5.2 Per-Pixel Gradient Derivation

Given the forward equations:
```
color_pixel = sum_k( T_k * alpha_k * color_k )  +  T_final * background
alpha_pixel = 1 - T_final

where:
  T_0 = 1
  T_{k+1} = T_k * (1 - alpha_k)
  T_final = T_{last+1}
  alpha_k = min(MAX_ALPHA, opacity_k * exp(-sigma_k))
  sigma_k = 0.5 * (a_k*dx^2 + c_k*dy^2) + b_k*dx*dy
```

The gradients (back-to-front iteration, starting from `T = T_final`):
```
For Gaussian k (from last to first):
    ra = 1 / (1 - alpha_k)
    T *= ra                   // recover T_k (transmittance BEFORE this Gaussian)
    fac = alpha_k * T         // = weight used in forward

    // d(loss)/d(color_k)
    v_color_k[c] = fac * v_render_color[c]

    // d(loss)/d(alpha_k)  (chain rule through color AND transmittance)
    v_alpha_k = sum_c( (color_k[c] * T - buffer[c] * ra) * v_render_color[c] )
              + T_final * ra * v_render_alpha
              - T_final * ra * sum_c(bg[c] * v_render_color[c])   // if background

    // d(loss)/d(sigma_k)  (chain rule through alpha, only if not clamped)
    if (opacity_k * exp(-sigma_k) <= MAX_ALPHA):
        v_sigma_k = -opacity_k * exp(-sigma_k) * v_alpha_k

    // d(loss)/d(conic_k)  (chain rule through sigma)
    v_conic_k = (0.5 * v_sigma * dx^2, v_sigma * dx*dy, 0.5 * v_sigma * dy^2)

    // d(loss)/d(mean2d_k)  (chain rule through sigma)
    v_mean2d_k = (v_sigma * (a*dx + b*dy), v_sigma * (b*dx + c*dy))

    // d(loss)/d(opacity_k)
    v_opacity_k = exp(-sigma_k) * v_alpha_k

    // Update buffer for next Gaussian
    buffer[c] += color_k[c] * fac
```

### 5.3 Backward Threadgroup Memory Analysis

The backward kernel needs additional storage for colors in shared memory (to avoid redundant global memory reads during gradient computation):

```
Backward kernel threadgroup memory:
┌─────────────────────────────────────────────────┐
│ tg_id_batch[256]        : int32   = 1024 bytes  │
├─────────────────────────────────────────────────┤
│ tg_xy_opacity_batch[256]: float3  = 3072 bytes  │
├─────────────────────────────────────────────────┤
│ tg_conic_batch[256]     : float3  = 3072 bytes  │
├─────────────────────────────────────────────────┤
│ tg_rgbs_batch[256 * C]  : float   = variable    │
└─────────────────────────────────────────────────┘

Memory vs channels:
  C=3:   10,240 bytes  (31% of 32 KB) -- fits easily
  C=8:   15,360 bytes  (47% of 32 KB) -- fits
  C=16:  23,552 bytes  (72% of 32 KB) -- fits but tight
  C=24:  31,744 bytes  (97% of 32 KB) -- barely fits
  C=32:  39,936 bytes  (122% of 32 KB) -- DOES NOT FIT
```

**Important constraint**: For C > 24, threadgroup memory exceeds the 32 KB limit with block_size=256.

**Mitigations for high channel counts:**
1. **Reduce block_size**: Use 128 (8x16 or 16x8 tiles). Halves memory, doubles batch count.
2. **Channel batching**: Process channels in groups (e.g., 16 at a time). Requires multiple passes.
3. **Skip shared colors**: Load colors from global memory per-thread. Slower but no memory limit.

For the typical RGB case (C=3), 10 KB is well within limits.

### 5.4 Complete Backward Metal Shader

```metal
// ============================================================================
// rasterize_bwd.metal
// Metal compute shader for 3DGS tile-based rasterization (backward pass)
//
// Translates from: gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu
// Iterates Gaussians in REVERSE order (back-to-front).
// Uses atomic float adds to scatter gradients to per-Gaussian arrays.
// Uses simdgroup reductions to minimize atomic contention.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float ALPHA_THRESHOLD = 1.0f / 255.0f;
constant float MAX_ALPHA       = 0.99f;
constant uint  MAX_CHANNELS    = 32;
constant uint  TILE_SIZE       = 16;
constant uint  BLOCK_SIZE      = TILE_SIZE * TILE_SIZE;

struct RasterizeBwdParams {
    uint image_width;
    uint image_height;
    uint tile_width;
    uint tile_height;
    uint n_channels;
    uint N;
    uint I;
    uint n_isects;
};

kernel void rasterize_to_pixels_3dgs_bwd(
    // ── Forward inputs (read-only, needed for recomputation) ──
    device const float2*  means2d         [[buffer(0)]],
    device const float3*  conics          [[buffer(1)]],
    device const float*   colors          [[buffer(2)]],
    device const float*   opacities       [[buffer(3)]],
    device const float*   backgrounds     [[buffer(4)]],
    device const int*     tile_offsets    [[buffer(5)]],
    device const int*     flatten_ids     [[buffer(6)]],
    // ── Forward outputs (read-only, saved for backward) ──
    device const float*   render_alphas   [[buffer(7)]],    // [I*H*W]
    device const int*     last_ids        [[buffer(8)]],    // [I*H*W]
    // ── Upstream gradients (read-only) ──
    device const float*   v_render_colors [[buffer(9)]],    // [I*H*W*C]
    device const float*   v_render_alphas [[buffer(10)]],   // [I*H*W]
    // ── Output gradients (written via atomics) ──
    device atomic_float*  v_means2d       [[buffer(11)]],   // [I*N, 2]
    device atomic_float*  v_conics        [[buffer(12)]],   // [I*N, 3]
    device atomic_float*  v_colors        [[buffer(13)]],   // [I*N, C]
    device atomic_float*  v_opacities     [[buffer(14)]],   // [I*N]
    // ── Parameters ──
    constant RasterizeBwdParams& params   [[buffer(15)]],
    constant uint& has_background         [[buffer(16)]],
    // ── Thread indexing ──
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos      [[thread_position_in_threadgroup]],
    uint  thread_rank     [[thread_index_in_threadgroup]],
    uint  simd_lane       [[thread_index_in_simdgroup]]
) {
    // ── 1. Compute indices ──
    const uint tile_x   = threadgroup_pos.x;
    const uint tile_y   = threadgroup_pos.y;
    const uint image_id = threadgroup_pos.z;

    const uint px = tile_x * TILE_SIZE + thread_pos.x;
    const uint py = tile_y * TILE_SIZE + thread_pos.y;
    const bool inside = (px < params.image_width) && (py < params.image_height);

    const float pixel_x = float(px) + 0.5f;
    const float pixel_y = float(py) + 0.5f;

    // Clamp pix_id to valid range (out-of-bounds threads still participate in loads)
    const uint pix_id = min(py * params.image_width + px,
                           params.image_width * params.image_height - 1);
    const uint pix_offset = image_id * params.image_height * params.image_width + pix_id;

    // ── 2. Tile range ──
    const uint tile_id = tile_y * params.tile_width + tile_x;
    const uint flat_tile = image_id * params.tile_height * params.tile_width + tile_id;

    const int range_start = tile_offsets[flat_tile];
    int range_end;
    if ((image_id == params.I - 1) &&
        (tile_id == params.tile_width * params.tile_height - 1)) {
        range_end = int(params.n_isects);
    } else {
        range_end = tile_offsets[flat_tile + 1];
    }

    const uint num_gaussians = uint(max(0, range_end - range_start));
    const uint num_batches = (num_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── 3. Threadgroup memory ──
    threadgroup int    tg_id_batch[BLOCK_SIZE];
    threadgroup float3 tg_xy_opacity_batch[BLOCK_SIZE];
    threadgroup float3 tg_conic_batch[BLOCK_SIZE];
    // For C=3: 256 * 3 * 4 = 3072 bytes. Total with above: 10240 bytes.
    threadgroup float  tg_rgbs_batch[BLOCK_SIZE * 3];  // Hardcoded C=3 for this version
    // TODO: Template on n_channels or use dynamic sizing

    // ── 4. Per-pixel backward state ──
    const float T_final = 1.0f - render_alphas[pix_offset];
    float T = T_final;

    // Buffer: accumulated contribution from Gaussians BEHIND the current one
    float buffer[MAX_CHANNELS];
    for (uint c = 0; c < params.n_channels && c < MAX_CHANNELS; c++) {
        buffer[c] = 0.0f;
    }

    // Index of last Gaussian that contributed to this pixel (from forward)
    const int bin_final = inside ? last_ids[pix_offset] : 0;

    // Read upstream gradients for this pixel
    float v_render_c[MAX_CHANNELS];
    for (uint c = 0; c < params.n_channels && c < MAX_CHANNELS; c++) {
        v_render_c[c] = v_render_colors[pix_offset * params.n_channels + c];
    }
    const float v_render_a = v_render_alphas[pix_offset];

    // ── 5. Main backward loop: REVERSE iteration ──
    for (uint b = 0; b < num_batches; b++) {
        // Sync before loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Batch end is the LAST index in this batch (iterating from end)
        const int batch_end = range_end - 1 - int(BLOCK_SIZE * b);
        const int batch_size_i = min(int(BLOCK_SIZE), batch_end + 1 - range_start);
        if (batch_size_i <= 0) break;
        const uint batch_size = uint(batch_size_i);

        // ── Load Gaussians from BACK to FRONT ──
        // Thread 0 loads the furthest-back Gaussian in this batch
        const int idx = batch_end - int(thread_rank);
        if (idx >= range_start) {
            const int g = flatten_ids[idx];
            tg_id_batch[thread_rank] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            tg_xy_opacity_batch[thread_rank] = float3(xy.x, xy.y, opac);
            tg_conic_batch[thread_rank] = conics[g];
            for (uint c = 0; c < params.n_channels && c < 3; c++) {
                tg_rgbs_batch[thread_rank * params.n_channels + c] =
                    colors[g * params.n_channels + c];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Process Gaussians in this batch ──
        // Index 0 is the furthest-back Gaussian
        for (uint t = 0; t < batch_size; t++) {
            // Skip Gaussians that are past this pixel's last contributor
            bool valid = inside && (int(batch_end) - int(t) <= bin_final);

            float alpha = 0.0f;
            float opac  = 0.0f;
            float2 delta;
            float3 con;
            float vis = 0.0f;

            if (valid) {
                con = tg_conic_batch[t];
                const float3 xy_opac = tg_xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = float2(xy_opac.x - pixel_x, xy_opac.y - pixel_y);

                const float sigma = 0.5f * (con.x * delta.x * delta.x
                                           + con.z * delta.y * delta.y)
                                  + con.y * delta.x * delta.y;
                vis = metal::fast::exp(-sigma);
                alpha = min(MAX_ALPHA, opac * vis);

                if (sigma < 0.0f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            if (!valid) continue;

            // ── Recover transmittance before this Gaussian ──
            const float ra = 1.0f / (1.0f - alpha);
            T *= ra;  // Now T = transmittance BEFORE Gaussian k
            const float fac = alpha * T;

            // ── Compute per-pixel gradients ──
            const int g = tg_id_batch[t];
            float v_alpha = 0.0f;

            // Gradient for colors + accumulate v_alpha
            float v_rgb_local[3] = {0.0f, 0.0f, 0.0f};
            for (uint c = 0; c < params.n_channels && c < 3; c++) {
                v_rgb_local[c] = fac * v_render_c[c];
                v_alpha += (tg_rgbs_batch[t * params.n_channels + c] * T
                           - buffer[c] * ra) * v_render_c[c];
            }

            // Contribution from alpha gradient
            v_alpha += T_final * ra * v_render_a;

            // Background contribution to v_alpha
            if (has_background != 0) {
                float bg_accum = 0.0f;
                for (uint c = 0; c < params.n_channels && c < 3; c++) {
                    bg_accum += backgrounds[image_id * params.n_channels + c]
                              * v_render_c[c];
                }
                v_alpha -= T_final * ra * bg_accum;
            }

            // ── Compute parameter gradients (chain rule through sigma) ──
            float v_conic_local[3] = {0.0f, 0.0f, 0.0f};
            float v_xy_local[2]    = {0.0f, 0.0f};
            float v_opacity_local  = 0.0f;

            if (opac * vis <= MAX_ALPHA) {
                const float v_sigma = -opac * vis * v_alpha;

                v_conic_local[0] = 0.5f * v_sigma * delta.x * delta.x;
                v_conic_local[1] = v_sigma * delta.x * delta.y;
                v_conic_local[2] = 0.5f * v_sigma * delta.y * delta.y;

                v_xy_local[0] = v_sigma * (con.x * delta.x + con.y * delta.y);
                v_xy_local[1] = v_sigma * (con.y * delta.x + con.z * delta.y);

                v_opacity_local = vis * v_alpha;
            }

            // ── Simdgroup reduction before atomic write ──
            // Reduces atomic contention by 32x (simdgroup width on Apple Silicon)
            for (uint c = 0; c < 3; c++) {
                v_rgb_local[c] = simd_sum(v_rgb_local[c]);
            }
            v_conic_local[0] = simd_sum(v_conic_local[0]);
            v_conic_local[1] = simd_sum(v_conic_local[1]);
            v_conic_local[2] = simd_sum(v_conic_local[2]);
            v_xy_local[0]    = simd_sum(v_xy_local[0]);
            v_xy_local[1]    = simd_sum(v_xy_local[1]);
            v_opacity_local  = simd_sum(v_opacity_local);

            // Only lane 0 of each simdgroup writes atomics
            if (simd_lane == 0) {
                for (uint c = 0; c < params.n_channels && c < 3; c++) {
                    atomic_fetch_add_explicit(
                        &v_colors[g * params.n_channels + c],
                        v_rgb_local[c], memory_order_relaxed);
                }
                atomic_fetch_add_explicit(&v_conics[g * 3 + 0],
                    v_conic_local[0], memory_order_relaxed);
                atomic_fetch_add_explicit(&v_conics[g * 3 + 1],
                    v_conic_local[1], memory_order_relaxed);
                atomic_fetch_add_explicit(&v_conics[g * 3 + 2],
                    v_conic_local[2], memory_order_relaxed);
                atomic_fetch_add_explicit(&v_means2d[g * 2 + 0],
                    v_xy_local[0], memory_order_relaxed);
                atomic_fetch_add_explicit(&v_means2d[g * 2 + 1],
                    v_xy_local[1], memory_order_relaxed);
                atomic_fetch_add_explicit(&v_opacities[g],
                    v_opacity_local, memory_order_relaxed);
            }

            // ── Update buffer for next (further forward) Gaussian ──
            for (uint c = 0; c < params.n_channels && c < 3; c++) {
                buffer[c] += tg_rgbs_batch[t * params.n_channels + c] * fac;
            }
        }
    }
}
```

### 5.5 Backward Integration with `@mx.custom_function`

```python
# src/gsplat_mlx/core/rasterization_metal.py

import mlx.core as mx

@mx.custom_function
def rasterize_to_pixels_metal(means2d, conics, colors, opacities,
                                image_width, image_height, tile_size,
                                isect_offsets, flatten_ids, backgrounds):
    """Metal-accelerated rasterization with custom backward pass."""
    render_colors, render_alphas, last_ids = _metal_forward(
        means2d, conics, colors, opacities,
        image_width, image_height, tile_size,
        isect_offsets, flatten_ids, backgrounds
    )
    return render_colors, render_alphas

@rasterize_to_pixels_metal.vjp
def rasterize_vjp(primals, cotangents, outputs):
    means2d, conics, colors, opacities = primals[:4]
    image_width, image_height, tile_size = primals[4:7]
    isect_offsets, flatten_ids, backgrounds = primals[7:]

    render_colors, render_alphas = outputs
    v_render_colors, v_render_alphas = cotangents

    # Retrieve last_ids from forward (saved in closure or recomputed)
    # ...

    v_means2d, v_conics, v_colors, v_opacities = _metal_backward(
        means2d, conics, colors, opacities, backgrounds,
        isect_offsets, flatten_ids,
        render_alphas, last_ids,
        v_render_colors, v_render_alphas,
        image_width, image_height, tile_size
    )

    return (v_means2d, v_conics, v_colors, v_opacities,
            None, None, None, None, None, None)  # no grad for discrete params
```

### 5.6 Backward Atomic Operations: Detailed Analysis

**Contention pattern**: Multiple pixels contribute gradients to the same Gaussian. A Gaussian visible in K tiles, each with 256 pixels, could receive up to 256*K atomic writes.

**Mitigation hierarchy** (from CUDA, adapted for Metal):

1. **Simdgroup reduction** (implemented above): Each simdgroup (32 threads) reduces locally before one atomic. Reduces contention 32x.

2. **Threadgroup reduction** (future optimization): Accumulate all per-threadgroup gradients into threadgroup memory, then one thread writes to global. Requires additional threadgroup memory for gradient accumulators.

3. **Deterministic mode**: Sort atomic writes by Gaussian ID. Slower but bit-reproducible. Useful for debugging.

---

## 6. Technical Design: Gaussian Projection Kernel

### 6.1 Design Overview

The projection kernel is embarrassingly parallel: each Gaussian is independently projected from 3D world space to 2D screen space. No inter-Gaussian communication or shared memory is needed.

**CUDA equivalent**: `ProjectionEWA3DGSFused.cu`

**Grid dispatch**: `(N * C_cams, 1, 1)` with threadgroup size `(256, 1, 1)`.

### 6.2 Projection Kernel Outline

```metal
// projection.metal
// One thread per (Gaussian, camera) pair. No threadgroup memory.

kernel void projection_ewa_3dgs_fwd(
    device const float*   means3d      [[buffer(0)]],   // [N, 3]
    device const float*   covars       [[buffer(1)]],   // [N, 6] upper-tri (xx,xy,xz,yy,yz,zz)
    device const float*   viewmats     [[buffer(2)]],   // [C, 4, 4]
    device const float*   Ks           [[buffer(3)]],   // [C, 3, 3]
    device const float*   opacities    [[buffer(4)]],   // [N]
    device float2*        means2d      [[buffer(5)]],   // [C, N, 2]
    device float3*        conics       [[buffer(6)]],   // [C, N, 3]
    device float*         compensations [[buffer(7)]],  // [C, N]
    device int*           radii        [[buffer(8)]],   // [C, N]
    device float*         depths       [[buffer(9)]],   // [C, N]
    constant uint&        N            [[buffer(10)]],
    constant uint&        C_cams       [[buffer(11)]],
    constant uint&        image_width  [[buffer(12)]],
    constant uint&        image_height [[buffer(13)]],
    constant float&       near_plane   [[buffer(14)]],
    constant float&       far_plane    [[buffer(15)]],
    constant float&       radius_clip  [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    // Thread maps to one (camera, Gaussian) pair
    uint cam_id   = gid / N;
    uint gauss_id = gid % N;
    if (cam_id >= C_cams) return;

    // ── 1. Load 3D Gaussian mean ──
    float3 mean3d = float3(
        means3d[gauss_id * 3 + 0],
        means3d[gauss_id * 3 + 1],
        means3d[gauss_id * 3 + 2]
    );

    // ── 2. Load view matrix [4x4] for this camera ──
    uint vm_base = cam_id * 16;
    float4x4 viewmat;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            viewmat[c][r] = viewmats[vm_base + r * 4 + c];

    // ── 3. World-to-camera transform ──
    float4 mean_cam4 = viewmat * float4(mean3d, 1.0f);
    float3 mean_cam = mean_cam4.xyz;

    // ── 4. Frustum culling ──
    if (mean_cam.z <= near_plane || mean_cam.z >= far_plane) {
        uint out_idx = cam_id * N + gauss_id;
        radii[out_idx] = 0;
        return;
    }

    // ── 5. Load intrinsics [3x3] ──
    uint k_base = cam_id * 9;
    float fx = Ks[k_base + 0], fy = Ks[k_base + 4];
    float cx = Ks[k_base + 2], cy = Ks[k_base + 5];

    // ── 6. Perspective projection ──
    float z_inv = 1.0f / mean_cam.z;
    float2 mean2d_val = float2(
        fx * mean_cam.x * z_inv + cx,
        fy * mean_cam.y * z_inv + cy
    );

    // ── 7. Load 3D covariance (upper triangular: xx,xy,xz,yy,yz,zz) ──
    uint cv_base = gauss_id * 6;
    float3x3 Sigma = float3x3(
        float3(covars[cv_base+0], covars[cv_base+1], covars[cv_base+2]),
        float3(covars[cv_base+1], covars[cv_base+3], covars[cv_base+4]),
        float3(covars[cv_base+2], covars[cv_base+4], covars[cv_base+5])
    );

    // ── 8. Compute 2D covariance: J * R * Sigma * R^T * J^T ──
    // R = rotation part of viewmat (3x3)
    float3x3 R;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R[c][r] = viewmat[c][r];

    // Jacobian of perspective projection
    float z2 = mean_cam.z * mean_cam.z;
    float3x3 J = float3x3(0.0f);
    J[0][0] = fx * z_inv;
    J[1][1] = fy * z_inv;
    J[0][2] = -fx * mean_cam.x / z2;
    J[1][2] = -fy * mean_cam.y / z2;

    // Sigma_cam = R * Sigma * R^T
    float3x3 Sigma_cam = R * Sigma * transpose(R);

    // Sigma_2d = J * Sigma_cam * J^T  (2x2 result, but computed in 3x3)
    float3x3 cov2d_full = J * Sigma_cam * transpose(J);

    // Extract 2x2 covariance
    float cov_xx = cov2d_full[0][0];
    float cov_xy = cov2d_full[0][1];
    float cov_yy = cov2d_full[1][1];

    // ── 9. Low-pass filter (anti-aliasing) ──
    float det_before = cov_xx * cov_yy - cov_xy * cov_xy;
    cov_xx += 0.3f;  // Low-pass filter constant
    cov_yy += 0.3f;
    float det_after = cov_xx * cov_yy - cov_xy * cov_xy;
    float compensation = metal::sqrt(metal::max(0.0f, det_before / det_after));

    // ── 10. Invert 2x2 covariance -> conics ──
    float det_inv = 1.0f / det_after;
    float3 conic_val = float3(
        cov_yy * det_inv,     // a
        -cov_xy * det_inv,    // b
        cov_xx * det_inv      // c
    );

    // ── 11. Compute radius from eigenvalues ──
    float opacity = opacities[gauss_id];
    float mid = 0.5f * (cov_xx + cov_yy);
    float lambda_max = mid + metal::sqrt(metal::max(0.1f, mid * mid - det_after));
    float radius_f = metal::ceil(3.0f * metal::sqrt(lambda_max));

    // Clip radius by opacity-dependent extent
    if (opacity >= ALPHA_THRESHOLD) {
        float extend = metal::sqrt(2.0f * metal::log(opacity / ALPHA_THRESHOLD));
        radius_f = min(radius_f, extend * metal::sqrt(lambda_max));
    }

    int radius_i = int(radius_f);
    if (radius_i <= int(radius_clip)) {
        uint out_idx = cam_id * N + gauss_id;
        radii[out_idx] = 0;
        return;
    }

    // ── 12. Screen-space culling ──
    if (mean2d_val.x + radius_f < 0 || mean2d_val.x - radius_f >= float(image_width) ||
        mean2d_val.y + radius_f < 0 || mean2d_val.y - radius_f >= float(image_height)) {
        uint out_idx = cam_id * N + gauss_id;
        radii[out_idx] = 0;
        return;
    }

    // ── 13. Write outputs ──
    uint out_idx = cam_id * N + gauss_id;
    means2d[out_idx]       = mean2d_val;
    conics[out_idx]        = conic_val;
    compensations[out_idx] = compensation;
    radii[out_idx]         = radius_i;
    depths[out_idx]        = mean_cam.z;
}
```

---

## 7. Technical Design: Tile Intersection and Sort Kernel

### 7.1 Design Overview

**CUDA equivalent**: `IntersectTile.cu`

Tile intersection has three phases:

### 7.2 Phase 1: Count Intersections Per Tile

Each Gaussian's screen-space bounding box (from `means2d` and `radii`) determines which tiles it overlaps. One thread per Gaussian, atomically incrementing per-tile counters.

```metal
kernel void count_tile_intersections(
    device const float2*    means2d    [[buffer(0)]],
    device const int*       radii      [[buffer(1)]],
    device atomic_uint*     tile_counts [[buffer(2)]],
    constant uint&          N          [[buffer(3)]],
    constant uint&          tile_size  [[buffer(4)]],
    constant uint&          tile_width [[buffer(5)]],
    constant uint&          tile_height [[buffer(6)]],
    constant uint&          image_width [[buffer(7)]],
    constant uint&          image_height [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    int radius = radii[gid];
    if (radius <= 0) return;

    float2 center = means2d[gid];
    float r = float(radius);

    int tile_x_min = max(0, int(center.x - r) / int(tile_size));
    int tile_x_max = min(int(tile_width) - 1, int(center.x + r) / int(tile_size));
    int tile_y_min = max(0, int(center.y - r) / int(tile_size));
    int tile_y_max = min(int(tile_height) - 1, int(center.y + r) / int(tile_size));

    for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
        for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
            uint tile_id = uint(ty) * tile_width + uint(tx);
            atomic_fetch_add_explicit(&tile_counts[tile_id], 1u, memory_order_relaxed);
        }
    }
}
```

### 7.3 Phase 2: Prefix Sum

Convert per-tile counts into offsets (exclusive prefix sum). Options:
- **MPS**: `MPSParallelExclusivePrefixSum` (if available via Metal Performance Shaders)
- **CPU**: `np.cumsum` (fast enough for < 100K tiles)
- **Metal kernel**: Implement work-efficient parallel scan (Blelloch algorithm)

**Recommendation**: Start with CPU (`mx.cumsum` or `np.cumsum`), replace with Metal scan only if profiling shows it as a bottleneck.

### 7.4 Phase 3: Populate and Sort

Each Gaussian writes its entries with depth-encoded sort keys, then sort per-tile by depth.

**Sorting strategy options:**

| Strategy | Latency (100K isects) | Complexity | Notes |
|----------|----------------------|------------|-------|
| CPU `np.argsort` | ~5ms | Low | Current PRD-06 approach. Works. |
| Metal bitonic sort | ~0.5ms | High | Custom kernel. O(n log^2 n). |
| MPS radix sort | ~0.3ms | Medium | If available. O(n). |
| Hybrid: Metal count + CPU sort | ~3ms | Low | Best of both worlds for now. |

**Recommendation**: Keep CPU sort (PRD-06 approach) for the initial Metal implementation. Add Metal parallel sort only after profiling confirms sorting as a top-3 bottleneck.

---

## 8. Fallback Dispatcher

```python
# src/gsplat_mlx/core/dispatch.py

import mlx.core as mx
import os

def _metal_available():
    """Check if Metal compute shaders are available."""
    try:
        # Check MLX has GPU backend
        if not mx.metal.is_available():
            return False
        # Check our Metal kernels are compiled
        # ... (kernel compilation check)
        return True
    except Exception:
        return False

# Environment variable override
_USE_METAL = os.environ.get("GSPLAT_MLX_BACKEND", "auto")

def rasterize_to_pixels(means2d, conics, colors, opacities,
                        image_width, image_height, tile_size,
                        isect_offsets, flatten_ids, backgrounds=None):
    """Dispatch to Metal or Python rasterization.

    Backend selection:
      GSPLAT_MLX_BACKEND=metal   -> force Metal (error if unavailable)
      GSPLAT_MLX_BACKEND=python  -> force Python
      GSPLAT_MLX_BACKEND=auto    -> Metal if available, else Python (default)
    """
    use_metal = False
    if _USE_METAL == "metal":
        use_metal = True
    elif _USE_METAL == "python":
        use_metal = False
    else:  # auto
        use_metal = _metal_available()

    if use_metal:
        from .rasterization_metal import rasterize_to_pixels_metal
        return rasterize_to_pixels_metal(
            means2d, conics, colors, opacities,
            image_width, image_height, tile_size,
            isect_offsets, flatten_ids, backgrounds
        )
    else:
        from .rasterization import rasterize_to_pixels
        return rasterize_to_pixels(
            means2d, conics, colors, opacities,
            image_width, image_height, tile_size,
            isect_offsets, flatten_ids, backgrounds
        )
```

---

## 9. Performance Targets and Benchmarking

### 9.1 Target Performance

| Operation | Python (PRD-07) | Metal Target | Expected Speedup |
|-----------|----------------|--------------|------------------|
| Projection (10K GS, 1 cam) | ~50 ms | ~0.3 ms | 150x |
| Projection (100K GS, 1 cam) | ~500 ms | ~1.5 ms | 300x |
| Intersection + Sort (10K GS) | ~100 ms | ~1.5 ms | 65x |
| Intersection + Sort (100K GS) | ~1000 ms | ~5 ms | 200x |
| Rasterize FWD (256x256, 10K GS) | ~500 ms | ~1.5 ms | 300x |
| Rasterize FWD (512x512, 10K GS) | ~2000 ms | ~3 ms | 650x |
| Rasterize FWD (512x512, 100K GS) | ~10 sec | ~10 ms | 1000x |
| Rasterize BWD (512x512, 100K GS) | ~15 sec | ~20 ms | 750x |
| **Full render (512x512, 100K GS)** | **~12 sec** | **~15 ms** | **~800x** |
| **Full train step (512x512, 100K GS)** | **~30 sec** | **~50 ms** | **~600x** |

**Target milestones:**
1. Forward rendering at 30 FPS for 256x256 with 10K GS on M1
2. Forward rendering at 30 FPS for 512x512 with 100K GS on M3 Pro
3. Full training loop at 10 iterations/sec for 512x512 with 100K GS on M3 Pro

### 9.2 Benchmarking Strategy

#### 9.2.1 Micro-benchmarks

Each kernel benchmarked independently with controlled inputs:

```python
# benchmarks/bench_metal_rasterize.py

import mlx.core as mx
import time

def bench_rasterize_fwd(n_gaussians, image_width, image_height,
                        tile_size=16, avg_per_tile=50,
                        n_warmup=5, n_iters=100):
    """Benchmark forward rasterization kernel."""
    # Generate synthetic scene
    means2d = mx.random.uniform(shape=(1, n_gaussians, 2))
    means2d = means2d * mx.array([image_width, image_height])
    conics = mx.random.normal(shape=(1, n_gaussians, 3)) * 0.01
    conics = conics.at[..., 0].add(0.1)  # positive definite
    conics = conics.at[..., 2].add(0.1)
    colors = mx.random.uniform(shape=(1, n_gaussians, 3))
    opacities = mx.random.uniform(shape=(1, n_gaussians)) * 0.5 + 0.3

    # Synthetic intersections
    tile_h = (image_height + tile_size - 1) // tile_size
    tile_w = (image_width + tile_size - 1) // tile_size
    n_isects = tile_h * tile_w * avg_per_tile
    offsets = mx.arange(0, tile_h * tile_w + 1) * avg_per_tile
    offsets = offsets[:-1].reshape(1, tile_h, tile_w)
    flatten_ids = mx.random.randint(0, n_gaussians, shape=(n_isects,))

    mx.eval(means2d, conics, colors, opacities, offsets, flatten_ids)

    # Warmup
    for _ in range(n_warmup):
        out = rasterize_to_pixels_metal(
            means2d, conics, colors, opacities,
            image_width, image_height, tile_size,
            offsets, flatten_ids
        )
        mx.eval(out)

    # Measure
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = rasterize_to_pixels_metal(
            means2d, conics, colors, opacities,
            image_width, image_height, tile_size,
            offsets, flatten_ids
        )
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    fps = 1000.0 / avg_ms
    print(f"  {n_gaussians:>7} GS @ {image_width}x{image_height}: "
          f"{avg_ms:.2f} ms ({fps:.1f} FPS)")
    return avg_ms, fps
```

#### 9.2.2 End-to-end Pipeline Benchmark

```python
def bench_full_pipeline(n_gaussians, image_width, image_height):
    """Benchmark: projection -> intersection -> rasterize -> total."""
    results = {}

    # Stage 1: Projection
    t0 = time.perf_counter()
    means2d, conics, depths, radii = project_gaussians_metal(...)
    mx.eval(means2d, conics, depths, radii)
    results['projection_ms'] = (time.perf_counter() - t0) * 1000

    # Stage 2: Intersection + Sort
    t0 = time.perf_counter()
    isect_offsets, flatten_ids = intersect_tiles(...)
    mx.eval(isect_offsets, flatten_ids)
    results['intersection_ms'] = (time.perf_counter() - t0) * 1000

    # Stage 3: Rasterization
    t0 = time.perf_counter()
    colors, alphas = rasterize_to_pixels_metal(...)
    mx.eval(colors, alphas)
    results['rasterize_ms'] = (time.perf_counter() - t0) * 1000

    results['total_ms'] = sum(results.values())
    results['fps'] = 1000.0 / results['total_ms']
    return results
```

#### 9.2.3 Comparison Framework

```python
def bench_comparison(n_gaussians, image_size):
    """Side-by-side: Python vs Metal, with speedup ratio."""
    print(f"\n{'='*60}")
    print(f"Scene: {n_gaussians} Gaussians @ {image_size}x{image_size}")
    print(f"{'='*60}")

    py_ms, py_fps = bench_rasterize_python(n_gaussians, image_size, image_size)
    mt_ms, mt_fps = bench_rasterize_fwd(n_gaussians, image_size, image_size)

    speedup = py_ms / mt_ms
    print(f"\n  Python:  {py_ms:>8.2f} ms ({py_fps:>6.1f} FPS)")
    print(f"  Metal:   {mt_ms:>8.2f} ms ({mt_fps:>6.1f} FPS)")
    print(f"  Speedup: {speedup:>8.1f}x")
```

### 9.3 Profiling Tools

```bash
# Apple Instruments (Metal System Trace)
xcrun xctrace record --template "Metal System Trace" --launch -- python bench.py

# MLX Metal capture (programmatic)
# In Python:
#   mx.metal.start_capture("rasterize_profile")
#   ... run kernel ...
#   mx.metal.stop_capture()

# Environment variables for Metal debugging
export MTL_CAPTURE_ENABLED=1
export MTL_DEBUG_LAYER=1
export MTL_SHADER_VALIDATION=1
```

Key metrics to track:
- **GPU utilization**: target > 80% during kernel execution
- **Occupancy**: threads in flight / maximum threads
- **Memory bandwidth**: should approach peak for memory-bound kernels (projection)
- **Threadgroup memory bank conflicts**: monitor via GPU profiler
- **Atomic contention**: monitor in backward kernel profiling

---

## 10. File Structure

```
src/gsplat_mlx/
├── metal/
│   ├── __init__.py
│   ├── rasterize_fwd.metal           # Forward rasterization shader (standalone .metal file)
│   ├── rasterize_bwd.metal           # Backward rasterization shader
│   ├── projection.metal              # Gaussian projection shader
│   ├── intersect.metal               # Tile intersection shader
│   ├── rasterize_fwd_inline.py       # Forward shader source as Python string (for mx.fast.metal_kernel)
│   ├── rasterize_bwd_inline.py       # Backward shader source as Python string
│   └── constants.py                  # Shared constants (ALPHA_THRESHOLD, MAX_ALPHA, etc.)
├── core/
│   ├── rasterization_metal.py        # Metal-backed rasterize_to_pixels() + @mx.custom_function
│   ├── projection_metal.py           # Metal-backed fully_fused_projection()
│   ├── intersection_metal.py         # Metal-backed isect_tiles()
│   └── dispatch.py                   # Auto-dispatch: Metal if available, Python fallback

tests/
├── test_metal_rasterize_fwd.py       # Forward: Metal matches Python reference
├── test_metal_rasterize_bwd.py       # Backward: Metal matches Python VJP
├── test_metal_projection.py          # Projection: Metal matches Python reference
├── test_metal_intersection.py        # Intersection: Metal matches Python reference
├── test_metal_e2e.py                # End-to-end: full pipeline correctness
├── test_metal_numerical.py           # Numerical stability edge cases

benchmarks/
├── bench_metal_rasterize.py          # Rasterization microbenchmarks
├── bench_metal_projection.py         # Projection microbenchmarks
├── bench_metal_pipeline.py           # Full pipeline benchmarks
├── bench_comparison.py               # Python vs Metal comparison
└── bench_hardware_matrix.py          # Cross-hardware benchmark suite
```

---

## 11. Test Plan

### 11.1 Correctness Tests

| Test | Description | Tolerance |
|------|-------------|-----------|
| `test_metal_fwd_single_gaussian` | Single Gaussian at center, compare Metal vs Python output | atol=1e-4 |
| `test_metal_fwd_multi_gaussian` | 10 overlapping Gaussians, verify compositing order | atol=1e-4 |
| `test_metal_fwd_background` | Background color correctly blended | atol=1e-4 |
| `test_metal_fwd_early_termination` | 100 dense Gaussians, verify early termination produces same result | atol=1e-4 |
| `test_metal_fwd_empty_tiles` | Mix of empty and populated tiles | exact match |
| `test_metal_fwd_edge_pixels` | Pixels on tile/image boundaries | atol=1e-4 |
| `test_metal_fwd_large_scene` | 100K Gaussians, 512x512 | atol=1e-3 |
| `test_metal_fwd_multi_image` | I=4 batch rendering | atol=1e-4 |
| `test_metal_fwd_multi_channel` | C=16 feature channels | atol=1e-4 |
| `test_metal_fwd_alpha_clamp` | MAX_ALPHA (0.99) clamping | atol=1e-6 |
| `test_metal_fwd_subthreshold_skip` | Very transparent Gaussians skipped | atol=1e-4 |
| `test_metal_bwd_gradient_colors` | d(loss)/d(colors) matches Python VJP | atol=1e-3 |
| `test_metal_bwd_gradient_opacities` | d(loss)/d(opacities) matches Python VJP | atol=1e-3 |
| `test_metal_bwd_gradient_means2d` | d(loss)/d(means2d) matches Python VJP | atol=1e-3 |
| `test_metal_bwd_gradient_conics` | d(loss)/d(conics) matches Python VJP | atol=1e-3 |
| `test_metal_bwd_with_background` | Backward pass with non-zero background | atol=1e-3 |
| `test_metal_proj_matches_python` | Metal projection vs Python reference | atol=1e-4 |
| `test_metal_isect_matches_python` | Metal intersection vs Python reference | exact match |

### 11.2 Numerical Stability Tests

| Test | Description |
|------|-------------|
| `test_metal_bwd_small_alpha` | Backward with alpha values near ALPHA_THRESHOLD |
| `test_metal_bwd_small_transmittance` | Backward where T_final ~ 1e-6 |
| `test_metal_bwd_atomic_precision` | Verify atomic float precision under high contention |
| `test_metal_bwd_numerical_gradients` | Finite-difference gradient check (slow but thorough) |
| `test_metal_exp_precision` | `metal::fast::exp` vs `metal::exp` vs Python `exp` |

### 11.3 Performance Tests

| Test | Description |
|------|-------------|
| `test_metal_bench_fwd_256` | FWD 10K GS @ 256x256, report ms/FPS |
| `test_metal_bench_fwd_512` | FWD 100K GS @ 512x512, report ms/FPS |
| `test_metal_bench_bwd_512` | BWD 100K GS @ 512x512, report ms/FPS |
| `test_metal_bench_pipeline` | Full pipeline 100K GS @ 512x512 |
| `test_metal_speedup_vs_python` | Assert Metal >= 50x faster for > 10K GS |

### 11.4 Edge Case Tests

| Test | Description |
|------|-------------|
| `test_metal_zero_gaussians` | No Gaussians -- output is background only |
| `test_metal_one_pixel_image` | 1x1 image |
| `test_metal_non_square_image` | 640x480 image |
| `test_metal_non_aligned_image` | Image size not divisible by tile_size (e.g., 500x500) |
| `test_metal_max_gaussians_per_tile` | Tile with 10K Gaussians (stress test) |
| `test_metal_all_culled` | All Gaussians outside image bounds |
| `test_metal_deterministic` | Same inputs produce bitwise-identical outputs |

---

## 12. Implementation Plan

### Phase 1: Forward Kernel Prototype (Week 1--2)

1. Implement forward rasterization using `mx.fast.metal_kernel()` WITHOUT shared memory (Section 4.4.1)
2. Verify correctness against Python reference for all forward tests
3. Establish baseline Metal performance without shared memory optimization
4. Profile with Metal GPU profiler to identify bottlenecks (global memory bandwidth vs compute)

### Phase 2: Forward Kernel with Shared Memory (Week 2--3)

1. Implement C++ extension (`mlx::core::Primitive`) for forward kernel (Section 4.3 + 4.4.2)
2. Add threadgroup memory allocation and cooperative loading
3. Verify correctness (same tests as Phase 1, must be identical)
4. Benchmark: measure speedup from shared memory (expected 3--5x over Phase 1)
5. Tune threadgroup size: evaluate tile_size=16 vs tile_size=8 empirically

### Phase 3: Backward Kernel (Week 3--5)

1. Implement backward kernel via C++ extension (Section 5)
2. Wire up to `@mx.custom_function` VJP (Section 5.5)
3. Verify gradient correctness against Python reference
4. Implement simdgroup reduction optimization for atomics (Section 5.4, already in shader)
5. Benchmark forward + backward together
6. Test training convergence: 100 iterations, compare loss curve with Python-only

### Phase 4: Projection Kernel (Week 5--6)

1. Implement projection kernel using `mx.fast.metal_kernel()` (Section 6)
2. Verify against Python projection reference (PRD-05)
3. Benchmark; this kernel is memory-bandwidth-bound, expect near-peak bandwidth utilization

### Phase 5: Intersection Kernel (Week 6)

1. Implement tile counting kernel in Metal (Section 7.2)
2. Use CPU prefix sum initially (Section 7.3)
3. Keep CPU sort from PRD-06 (Section 7.4)
4. Verify correctness
5. Profile: determine if sorting is a top-3 bottleneck; if so, plan Metal sort

### Phase 6: Integration and Polish (Week 6--7)

1. Implement dispatch layer (auto-select Metal vs Python) (Section 8)
2. End-to-end pipeline test: 3D Gaussians -> rendered image
3. Full benchmark suite across configurations
4. Profile and optimize remaining hotspots
5. Ensure `GSPLAT_MLX_BACKEND=python` still works for all tests

### Phase 7: Training Loop Validation (Week 7--8)

1. Run complete training loop: forward + loss + backward + optimizer step
2. Verify training convergence matches Python-only pipeline (+/- 1% loss at 1000 iters)
3. Measure training speed (iterations/second)
4. Profile memory usage patterns; check for leaks over 1000+ frames
5. Document results with hardware/OS/MLX versions

---

## 13. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `mx.fast.metal_kernel()` cannot express threadgroup memory | Medium | Medium | Fall back to C++ extension (Path B). Phase 1 prototype works without shared memory. |
| Metal `atomic_float` precision differs from CUDA | Low | High | Use simdgroup reductions to minimize atomics. Test with finite-difference gradients. If needed, use double-precision buffer or compensated summation. |
| Backward gradient numerical drift at scale | Medium | High | Extensive finite-difference testing. Higher tolerance (1e-3) for large scenes. Compare loss curves rather than per-element gradients. |
| Apple Silicon threadgroup memory bank conflicts | Low | Medium | Profile bank conflicts. Adjust memory layout (pad to avoid conflicts). Current layout is naturally aligned. |
| MLX graph compilation overhead amortization | Medium | Medium | Profile end-to-end including first call. Use `mx.compile()` to pre-compile. Ensure kernel is built once and reused. |
| C++ extension build complexity | Medium | Medium | Provide pre-built wheels for major macOS versions. Clear build instructions. CI/CD build matrix. |
| `metal::fast::exp` precision issues | Low | Low | Compare with `metal::precise::exp`. Use precise version if fast version causes gradient issues. |

---

## 14. Dependencies

| Dependency | Type | Status |
|-----------|------|--------|
| PRD-01: Dev Environment | Hard | Must be complete |
| PRD-02: Math Utils | Hard | Must be complete |
| PRD-03: Covariance | Hard | Must be complete |
| PRD-04: Spherical Harmonics | Hard | Must be complete |
| PRD-05: Projection | Hard | Must be complete (Python reference for projection kernel validation) |
| PRD-06: Intersection | Hard | Must be complete (Python reference for intersection kernel validation) |
| PRD-07: Rasterization | Hard | Must be complete (Python reference = correctness oracle for all Metal kernels) |
| PRD-08: Accumulate | Hard | Must be complete |
| PRD-09: Rendering API | Hard | Must be complete |
| MLX >= 0.25 | External | Required for `mx.fast.metal_kernel()` API |
| Xcode Command Line Tools | External | Metal compiler toolchain for C++ extension |
| macOS >= 14.0 (Sonoma) | External | Metal 3.1+ for latest simdgroup features |

---

## 15. Acceptance Criteria

- [ ] Forward Metal kernel produces output matching Python reference within atol=1e-4 for all test scenes
- [ ] Backward Metal kernel produces gradients matching Python VJP within atol=1e-3
- [ ] Forward rasterization achieves >= 30 FPS at 512x512 with 100K Gaussians on M3 Pro
- [ ] Full pipeline (projection + intersection + rasterization) runs end-to-end on GPU
- [ ] Training loop converges to same loss as Python-only pipeline (+/- 1% after 1000 iterations)
- [ ] Fallback to Python implementation works when Metal kernels are unavailable
- [ ] All benchmark results documented with hardware, OS version, and MLX version
- [ ] No memory leaks during extended rendering (1000+ frames verified)
- [ ] Handles tile_size values: 8, 16, 32
- [ ] Handles channel counts: 1, 3, 16
- [ ] All tests pass with `pytest tests/test_metal_*.py -v`

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Threadgroup** | Metal equivalent of CUDA block. Group of threads that can share threadgroup memory and synchronize via barriers. |
| **Simdgroup** | Metal equivalent of CUDA warp. 32 threads executing in lockstep on Apple Silicon. Supports `simd_sum`, `simd_max`, and other cross-lane operations. |
| **Threadgroup memory** | Metal equivalent of CUDA shared memory. Fast on-chip SRAM (32 KB per threadgroup on Apple Silicon). |
| **MSL** | Metal Shading Language. C++14-based GPU programming language for Apple GPUs. |
| **Transmittance (T)** | Product of (1 - alpha) for all Gaussians in front of current point. T=1 means fully transparent (nothing in front), T=0 means fully occluded. |
| **Conics** | Inverse 2x2 covariance matrix of a projected Gaussian, stored as (a, b, c) where the matrix is [[a, b], [b, c]]. |
| **Alpha compositing** | Front-to-back blending: `color += T * alpha * gaussian_color; T *= (1 - alpha)`. |
| **Early termination** | Stop compositing when T < TRANSMITTANCE_THRESHOLD (1e-4). The pixel is effectively opaque. |
| **Cooperative loading** | All threads in a threadgroup collaboratively load data into shared memory, then all threads read from shared memory. Reduces global memory bandwidth by block_size factor. |
| **MPS** | Metal Performance Shaders. Apple's library of optimized GPU primitives (sorting, matrix ops, image processing). |
| **`mx.fast.metal_kernel()`** | MLX Python API for defining and dispatching custom Metal compute kernels with auto-generated function signatures. |
| **`atomic_fetch_add_explicit`** | Metal atomic float addition. Used in backward pass to scatter gradients from multiple pixels to shared Gaussians. |
| **`simd_sum`** | Metal simdgroup intrinsic that sums a value across all 32 lanes of a simdgroup. Used to reduce atomic contention. |
