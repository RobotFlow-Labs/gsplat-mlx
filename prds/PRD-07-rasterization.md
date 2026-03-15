# PRD-07: Pixel Rasterization (Core Alpha Compositing)

## Overview

Port the core pixel-level rasterization -- the heart of 3D Gaussian Splatting rendering. For each pixel, iterate through the sorted Gaussians that overlap its tile, compute per-pixel Gaussian weights, and alpha-composite them front-to-back to produce the final image.

This is the most computationally intensive operation in the entire pipeline. Every pixel in the output image must evaluate potentially hundreds of Gaussians and accumulate their contributions sequentially. The backward pass is equally critical -- it flows gradients from the rendered image loss back through the compositing chain to every Gaussian's position, shape, color, and opacity.

In upstream gsplat, the forward and backward are fully fused CUDA kernels (`RasterizeToPixels3DGSFwd.cu` and `RasterizeToPixels3DGSBwd.cu`). The Python reference in `_torch_impl.py` depends on `nerfacc` and the `rasterize_to_indices_in_range` CUDA kernel, so we cannot use it directly. Instead, we port the **algorithm** from the CUDA kernels into pure MLX Python with a vectorized-per-tile strategy.

## Source Reference

- **CUDA forward kernel**: `repositories/gsplat-upstream/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` (lines 40-211) -- `rasterize_to_pixels_3dgs_fwd_kernel`
- **CUDA backward kernel**: `repositories/gsplat-upstream/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` (lines 38-301) -- `rasterize_to_pixels_3dgs_bwd_kernel`
- **Python reference**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:560-670` -- `_rasterize_to_pixels` (uses nerfacc, NOT portable)
- **autograd.Function**: `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:1676-1803` -- `_RasterizeToPixels`
- **Public API**: `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py:796-928` -- `rasterize_to_pixels`
- **Constants**: `repositories/gsplat-upstream/gsplat/cuda/_constants.py` -- `ALPHA_THRESHOLD`, `MAX_ALPHA`, `TRANSMITTANCE_THRESHOLD`

## Scope

### In Scope
- Forward pass: tile-based alpha compositing producing rendered image + alpha map
- Backward pass: gradient computation for means2d, conics, colors, opacities via `@mx.custom_function`
- Background color blending with remaining transmittance
- Early termination when transmittance drops below threshold
- Alpha clamping to MAX_ALPHA
- Sub-threshold alpha skipping (alpha < ALPHA_THRESHOLD)
- Negative sigma rejection
- Multi-channel rendering (C = 1, 3, 16, or arbitrary)
- Batch rendering (multiple images)
- `last_ids` tracking for backward pass efficiency

### Out of Scope
- `absgrad` mode (absolute gradient computation for densification -- add in a follow-up)
- Packed input format (`nnz` mode)
- Tile masks for selective rendering
- Channel padding logic (handle at API layer in PRD-09)
- Metal shader implementation (PRD-14)
- `rasterize_to_indices_in_range` (not needed -- we implement compositing directly)
- `accumulate` / nerfacc dependency (not needed)

---

## Constants

From `_constants.py`:

```python
ALPHA_THRESHOLD = 1.0 / 255.0      # ~0.00392 -- skip Gaussians with alpha below this
MAX_ALPHA = 0.99                     # clamp alpha to prevent fully opaque single Gaussian
TRANSMITTANCE_THRESHOLD = 1e-4       # stop when pixel is saturated
# Relationship: TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2 = 0.01^2 = 1e-4
# This means a maximally opaque Gaussian must be rasterized TWICE to reach the threshold.
```

---

## Mathematical Foundation

### Alpha Compositing (Front-to-Back)

For each pixel at position $(p_x, p_y)$, given $K$ sorted Gaussians (front-to-back):

**Step 1: Gaussian evaluation**

For Gaussian $k$ with 2D mean $\mu_k = (\mu_x, \mu_y)$ and inverse covariance (conic) $\Sigma^{-1}_k = (a_k, b_k, c_k)$:

$$\delta_k = (p_x - \mu_x,\ p_y - \mu_y)$$

$$\sigma_k = \frac{1}{2}(a_k \cdot \delta_x^2 + c_k \cdot \delta_y^2) + b_k \cdot \delta_x \cdot \delta_y$$

$$\alpha_k = \min\left(o_k \cdot e^{-\sigma_k},\ \text{MAX\_ALPHA}\right)$$

where $o_k$ is the opacity and the conic encodes the upper-triangular inverse 2D covariance: $\Sigma^{-1} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$.

**Step 2: Alpha compositing**

$$T_0 = 1.0$$

$$\text{color}_{pixel} = \sum_{k=0}^{K-1} T_k \cdot \alpha_k \cdot c_k$$

$$T_{k+1} = T_k \cdot (1 - \alpha_k)$$

$$\text{alpha}_{pixel} = 1 - T_K$$

**Step 3: Background blending**

$$\text{color}_{final} = \text{color}_{pixel} + T_K \cdot \text{background}$$

### Backward Pass Derivation

Given upstream gradient $\frac{\partial L}{\partial \text{color}_{pixel}}$ (denoted $v_{\text{color}}$) and $\frac{\partial L}{\partial \text{alpha}_{pixel}}$ (denoted $v_{\text{alpha}}$).

The backward pass processes Gaussians **back-to-front** (reverse order). Starting from $T = T_{\text{final}}$ (the transmittance after the last Gaussian), we recover each Gaussian's transmittance by dividing: $T_k = T / (1 - \alpha_k)$ as we walk backward.

For each Gaussian $k$ (back-to-front):

**Recover transmittance before this Gaussian:**

$$r_k = \frac{1}{1 - \alpha_k}$$

$$T \leftarrow T \cdot r_k \quad \text{(this gives } T_k \text{, the transmittance BEFORE Gaussian } k \text{)}$$

**Gradient of color (color channel $c$):**

$$\frac{\partial L}{\partial c_{k,c}} = \alpha_k \cdot T_k \cdot v_{\text{color},c}$$

**Gradient of alpha:**

$$\frac{\partial L}{\partial \alpha_k} = \sum_c \left(c_{k,c} \cdot T_k - \text{buffer}_c \cdot r_k\right) \cdot v_{\text{color},c} + T_{\text{final}} \cdot r_k \cdot v_{\text{alpha}}$$

where $\text{buffer}_c$ accumulates the weighted color contributions from Gaussians BEHIND the current one. It starts at 0 and accumulates: $\text{buffer}_c \mathrel{+}= c_{k,c} \cdot \alpha_k \cdot T_k$.

**Background contribution to alpha gradient (if background is provided):**

$$\frac{\partial L}{\partial \alpha_k} \mathrel{+}= -T_{\text{final}} \cdot r_k \cdot \sum_c \text{bg}_c \cdot v_{\text{color},c}$$

**Gradient through sigma (only when $o_k \cdot e^{-\sigma_k} \leq \text{MAX\_ALPHA}$, i.e., alpha was not clamped):**

$$v_{\sigma_k} = -o_k \cdot e^{-\sigma_k} \cdot \frac{\partial L}{\partial \alpha_k}$$

**Gradient of opacity:**

$$\frac{\partial L}{\partial o_k} = e^{-\sigma_k} \cdot \frac{\partial L}{\partial \alpha_k}$$

**Gradient of conic:**

$$\frac{\partial L}{\partial a_k} = \frac{1}{2} \cdot v_{\sigma_k} \cdot \delta_x^2$$

$$\frac{\partial L}{\partial b_k} = v_{\sigma_k} \cdot \delta_x \cdot \delta_y$$

$$\frac{\partial L}{\partial c_k} = \frac{1}{2} \cdot v_{\sigma_k} \cdot \delta_y^2$$

**Gradient of means2d:**

$$\frac{\partial L}{\partial \mu_x} = v_{\sigma_k} \cdot (a_k \cdot \delta_x + b_k \cdot \delta_y) \cdot (-1)$$

$$\frac{\partial L}{\partial \mu_y} = v_{\sigma_k} \cdot (b_k \cdot \delta_x + c_k \cdot \delta_y) \cdot (-1)$$

Note: The negative sign comes from $\delta = p - \mu$, so $\partial\sigma/\partial\mu_x = -(a \cdot \delta_x + b \cdot \delta_y)$. In the CUDA kernel, the gradient is computed as $v_\sigma \cdot (a \cdot \delta_x + b \cdot \delta_y)$ which gives $\partial L / \partial p_x$; the sign for $\mu$ is absorbed because $\partial\sigma/\partial\mu = -\partial\sigma/\partial p$. However, looking at the CUDA kernel code, it stores:

```
v_xy_local.x = v_sigma * (conic.x * delta.x + conic.y * delta.y)
v_xy_local.y = v_sigma * (conic.y * delta.x + conic.z * delta.y)
```

This is because $v_\sigma$ already carries the negative sign from $v_\sigma = -\text{opac} \cdot \text{vis} \cdot v_\alpha$, and the chain rule through $\delta = p - \mu$ contributes another negative to get $\partial/\partial\mu$. The net effect: the signs work out to give the correct gradient w.r.t. means2d.

---

## MLX Implementation Strategy

### The Core Challenge

Alpha compositing is **inherently sequential** per pixel -- each step depends on the previous transmittance. CUDA solves this with massive parallelism (one thread per pixel, all pixels run simultaneously, each doing its own sequential loop). MLX cannot launch per-pixel threads from Python.

### Chosen Approach: Loop over Gaussians, Vectorize over Pixels (Option C)

We iterate over Gaussians sequentially (the sequential compositing axis), but for each Gaussian iteration, we compute contributions to **all pixels simultaneously** using MLX vectorized operations.

```
for each Gaussian k in sorted order (SEQUENTIAL):
    compute sigma for ALL pixels at once          (VECTORIZED: [H, W])
    compute alpha for ALL pixels at once           (VECTORIZED: [H, W])
    accumulate color += T * alpha * color_k        (VECTORIZED: [H, W, C])
    update T *= (1 - alpha)                        (VECTORIZED: [H, W])
```

Within each tile, we know exactly which Gaussians to process (from `isect_offsets` and `flatten_ids`). The tile structure provides data locality.

### Implementation Tiers

**Tier 1 (MVP -- this PRD):** Pure Python with numpy fallback for tile iteration, MLX for the math. Correct but slow. Suitable for testing and validation.

**Tier 2 (Optimization pass):** Fully vectorized MLX with padded tile processing. Loop only over max-Gaussians-per-tile dimension.

**Tier 3 (PRD-14):** Metal shader that mirrors the CUDA kernel 1:1 (one thread per pixel, shared memory per tile).

---

## Detailed Forward Algorithm

### Function Signature

```python
def rasterize_to_pixels(
    means2d: mx.array,        # [I, N, 2]   float32 -- 2D Gaussian centers
    conics: mx.array,         # [I, N, 3]   float32 -- inverse covariance [a, b, c]
    colors: mx.array,         # [I, N, C]   float32 -- per-Gaussian colors/features
    opacities: mx.array,      # [I, N]      float32 -- per-Gaussian opacity in [0, 1]
    image_width: int,          #             pixel width of output image
    image_height: int,         #             pixel height of output image
    tile_size: int,            #             tile side length (typically 16)
    isect_offsets: mx.array,  # [I, tile_H, tile_W]  int32 -- start offset per tile
    flatten_ids: mx.array,    # [n_isects]  int32 -- sorted Gaussian indices
    backgrounds: Optional[mx.array] = None,  # [I, C]  float32 -- background color
) -> Tuple[mx.array, mx.array]:
    """
    Returns:
        render_colors: [I, image_height, image_width, C]  float32
        render_alphas: [I, image_height, image_width, 1]  float32
    """
```

Note: The `[...]` batch dimensions from upstream are flattened to `[I, ...]` internally. The public API (PRD-09) handles reshaping.

### Forward Pseudocode (Tier 1 -- Reference Implementation)

```python
import numpy as np
import mlx.core as mx

ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4

def _rasterize_to_pixels_fwd(
    means2d, conics, colors, opacities,
    image_width, image_height, tile_size,
    isect_offsets, flatten_ids, backgrounds
):
    I = means2d.shape[0]       # number of images
    N = means2d.shape[1]       # number of Gaussians
    C = colors.shape[2]        # number of color channels
    tile_H = isect_offsets.shape[1]
    tile_W = isect_offsets.shape[2]
    n_isects = flatten_ids.shape[0]

    # Convert to numpy for the reference loop implementation
    means2d_np = np.array(means2d)    # [I, N, 2]
    conics_np  = np.array(conics)     # [I, N, 3]
    colors_np  = np.array(colors)     # [I, N, C]
    opac_np    = np.array(opacities)  # [I, N]
    offsets_np = np.array(isect_offsets)  # [I, tile_H, tile_W]
    fids_np    = np.array(flatten_ids)    # [n_isects]

    # Output buffers
    out_colors = np.zeros((I, image_height, image_width, C), dtype=np.float32)
    out_alphas = np.zeros((I, image_height, image_width, 1), dtype=np.float32)
    last_ids   = np.full((I, image_height, image_width), 0, dtype=np.int32)

    for img_id in range(I):
        for ty in range(tile_H):
            for tx in range(tile_W):
                # --- Determine Gaussian range for this tile ---
                tile_id = ty * tile_W + tx
                start = int(offsets_np[img_id, ty, tx])

                # End offset: next tile's start, or n_isects for the very last tile
                if img_id == I - 1 and tile_id == tile_H * tile_W - 1:
                    end = n_isects
                else:
                    # Linear index into flattened [I, tile_H, tile_W]
                    flat_tile = img_id * tile_H * tile_W + tile_id
                    # Next tile in flattened order
                    next_flat = flat_tile + 1
                    next_img = next_flat // (tile_H * tile_W)
                    next_tile = next_flat % (tile_H * tile_W)
                    next_ty = next_tile // tile_W
                    next_tx = next_tile % tile_W
                    end = int(offsets_np[next_img, next_ty, next_tx])

                if start >= end:
                    continue

                # --- Pixel range for this tile ---
                px_start = tx * tile_size
                py_start = ty * tile_size
                px_end = min(px_start + tile_size, image_width)
                py_end = min(py_start + tile_size, image_height)

                # --- Per-pixel compositing ---
                for py in range(py_start, py_end):
                    for px in range(px_start, px_end):
                        pixel_x = px + 0.5  # pixel center
                        pixel_y = py + 0.5

                        T = 1.0                        # transmittance
                        pixel_color = np.zeros(C)      # accumulated color
                        cur_idx = 0                    # last contributing Gaussian

                        for k in range(start, end):
                            g = int(fids_np[k])  # global flatten index in [I * N]
                            gid = g % N          # Gaussian index
                            iid = g // N         # image index

                            # Safety: skip if wrong image (should not happen with correct isect)
                            if iid != img_id:
                                continue

                            # Compute delta
                            dx = pixel_x - means2d_np[img_id, gid, 0]
                            dy = pixel_y - means2d_np[img_id, gid, 1]

                            # Compute sigma (Mahalanobis-like distance)
                            a, b, c_val = conics_np[img_id, gid]
                            sigma = 0.5 * (a * dx * dx + c_val * dy * dy) + b * dx * dy

                            # Skip if sigma is negative (numerical issue)
                            if sigma < 0.0:
                                continue

                            # Compute alpha
                            vis = np.exp(-sigma)
                            alpha = min(opac_np[img_id, gid] * vis, MAX_ALPHA)

                            # Skip if alpha too small
                            if alpha < ALPHA_THRESHOLD:
                                continue

                            # Check transmittance BEFORE accumulating
                            next_T = T * (1.0 - alpha)
                            if next_T <= TRANSMITTANCE_THRESHOLD:
                                # This pixel is done (EXCLUSIVE: don't include this Gaussian)
                                break

                            # Accumulate
                            pixel_color += T * alpha * colors_np[img_id, gid]
                            cur_idx = k
                            T = next_T

                        out_colors[img_id, py, px] = pixel_color
                        out_alphas[img_id, py, px, 0] = 1.0 - T
                        last_ids[img_id, py, px] = cur_idx

    # Apply background
    if backgrounds is not None:
        bg_np = np.array(backgrounds)  # [I, C]
        for img_id in range(I):
            remaining_T = 1.0 - out_alphas[img_id, :, :, 0]  # [H, W]
            out_colors[img_id] += remaining_T[:, :, np.newaxis] * bg_np[img_id]

    render_colors = mx.array(out_colors)  # [I, H, W, C]
    render_alphas = mx.array(out_alphas)  # [I, H, W, 1]
    last_ids_mx   = mx.array(last_ids)    # [I, H, W]

    return render_colors, render_alphas, last_ids_mx
```

### Tensor Shapes at Each Step (Forward)

```
Input:
  means2d       : [I, N, 2]
  conics        : [I, N, 3]
  colors        : [I, N, C]
  opacities     : [I, N]
  isect_offsets  : [I, tile_H, tile_W]      where tile_H = ceil(H / tile_size), tile_W = ceil(W / tile_size)
  flatten_ids    : [n_isects]

Per-tile (conceptual):
  gauss_range    : [start, end)              indices into flatten_ids for this tile
  pixel_grid     : [tile_h, tile_w, 2]       where tile_h <= tile_size, tile_w <= tile_size (edge tiles may be smaller)

Per-pixel per-Gaussian:
  delta          : [2]                        pixel_center - mean2d
  sigma          : scalar                     0.5 * (a*dx^2 + c*dy^2) + b*dx*dy
  vis            : scalar                     exp(-sigma)
  alpha          : scalar                     min(opacity * vis, MAX_ALPHA)
  contribution   : [C]                        T * alpha * color

Output:
  render_colors  : [I, H, W, C]
  render_alphas  : [I, H, W, 1]
  last_ids       : [I, H, W]                 index in flatten_ids of last contributing Gaussian per pixel
```

### Forward Pseudocode (Tier 2 -- Vectorized MLX)

```python
def _rasterize_to_pixels_fwd_vectorized(
    means2d, conics, colors, opacities,
    image_width, image_height, tile_size,
    isect_offsets, flatten_ids, backgrounds
):
    I, N, _ = means2d.shape
    C = colors.shape[2]
    tile_H = (image_height + tile_size - 1) // tile_size
    tile_W = (image_width + tile_size - 1) // tile_size
    n_isects = flatten_ids.shape[0]

    # Build pixel coordinate grids: [H, W]
    pixel_x = mx.arange(image_width).astype(mx.float32) + 0.5    # [W]
    pixel_y = mx.arange(image_height).astype(mx.float32) + 0.5   # [H]

    # Initialize output
    render_colors = mx.zeros((I, image_height, image_width, C))   # [I, H, W, C]
    transmittance = mx.ones((I, image_height, image_width))       # [I, H, W]

    # Compute the offset table with an appended sentinel
    # offsets_flat: [I * tile_H * tile_W + 1] where the last entry = n_isects
    offsets_flat = mx.reshape(isect_offsets, (I * tile_H * tile_W,))
    offsets_flat = mx.concatenate([offsets_flat, mx.array([n_isects])])

    # Compute max Gaussians per tile (determines loop count)
    tile_counts = offsets_flat[1:] - offsets_flat[:-1]             # [I * tile_H * tile_W]
    max_per_tile = int(mx.max(tile_counts).item())

    # For each Gaussian slot k (0..max_per_tile-1), process ALL tiles simultaneously
    for k in range(max_per_tile):
        # For each tile, compute the global index of the k-th Gaussian
        # tile_start[t] + k, valid only if k < tile_count[t]
        tile_starts = offsets_flat[:-1]                            # [I * tile_H * tile_W]
        gauss_idx_in_flat = tile_starts + k                        # [I * tile_H * tile_W]
        valid_tile = (k < tile_counts)                             # [I * tile_H * tile_W]  bool

        # Clamp to valid range and gather
        gauss_idx_clamped = mx.clip(gauss_idx_in_flat, 0, n_isects - 1)
        fid = flatten_ids[gauss_idx_clamped]                       # [I * tile_H * tile_W]
        gid = fid % N                                              # Gaussian index
        iid = fid // N                                             # image index

        # Gather Gaussian properties
        # Need to gather from [I, N, ...] using (iid, gid)
        # Flatten for gather: flat_idx = iid * N + gid
        flat_g = iid * N + gid                                    # [I * tile_H * tile_W]

        means_flat = mx.reshape(means2d, (I * N, 2))
        conics_flat = mx.reshape(conics, (I * N, 3))
        colors_flat = mx.reshape(colors, (I * N, C))
        opac_flat = mx.reshape(opacities, (I * N,))

        g_mean = means_flat[flat_g]                                # [n_tiles, 2]
        g_conic = conics_flat[flat_g]                              # [n_tiles, 3]
        g_color = colors_flat[flat_g]                              # [n_tiles, C]
        g_opac = opac_flat[flat_g]                                 # [n_tiles]

        # Reshape to tile grid: [I, tile_H, tile_W, ...]
        g_mean = mx.reshape(g_mean, (I, tile_H, tile_W, 2))
        g_conic = mx.reshape(g_conic, (I, tile_H, tile_W, 3))
        g_color = mx.reshape(g_color, (I, tile_H, tile_W, C))
        g_opac = mx.reshape(g_opac, (I, tile_H, tile_W))
        valid_tile = mx.reshape(valid_tile, (I, tile_H, tile_W))

        # Expand to pixel level: each tile covers tile_size x tile_size pixels
        # Tile (ty, tx) covers pixels [ty*T : (ty+1)*T, tx*T : (tx+1)*T]
        # We need to broadcast tile-level Gaussian data to pixel level

        # Create per-pixel tile indices
        # pixel (py, px) belongs to tile (py // tile_size, px // tile_size)
        # Use repeat/expand to broadcast [I, tile_H, tile_W, ...] -> [I, H, W, ...]
        # via: tile_data[:, ty, tx, :] -> pixel_data[:, ty*T:(ty+1)*T, tx*T:(tx+1)*T, :]

        # For each pixel, compute delta, sigma, alpha
        # pixel_x: [W], pixel_y: [H]
        # g_mean at pixel (py, px): g_mean[:, py//T, px//T, :]

        # Broadcast approach: expand tile data to pixel grid
        # g_mean_px[i, py, px, :] = g_mean[i, py//T, px//T, :]
        tile_y_idx = mx.arange(image_height) // tile_size          # [H]  -> tile row
        tile_x_idx = mx.arange(image_width) // tile_size           # [W]  -> tile col

        # Gather: [I, H, W, 2]
        # g_mean_px = g_mean[:, tile_y_idx, tile_x_idx, :]
        # This requires advanced indexing; in practice, use mx.take or reshape tricks

        # ... (detailed indexing implementation)

        # delta_x = pixel_x[px] - g_mean_px[..., 0]               # [I, H, W]
        # delta_y = pixel_y[py] - g_mean_px[..., 1]               # [I, H, W]

        # sigma = 0.5 * (a * dx^2 + c * dy^2) + b * dx * dy       # [I, H, W]
        # vis = exp(-sigma)                                         # [I, H, W]
        # alpha = min(opac * vis, MAX_ALPHA)                        # [I, H, W]

        # Apply masks: invalid tiles, sigma < 0, alpha < threshold
        # alpha = where(valid & sigma >= 0 & alpha >= ALPHA_THRESH, alpha, 0)

        # Check transmittance cutoff
        # next_T = transmittance * (1 - alpha)
        # saturated = next_T <= TRANSMITTANCE_THRESHOLD
        # alpha = where(~saturated, alpha, 0)  # zero out saturated pixels

        # Accumulate
        # render_colors += transmittance[..., None] * alpha[..., None] * g_color_px
        # transmittance *= (1 - alpha)

    # Background
    # if backgrounds is not None:
    #     render_colors += transmittance[..., None] * backgrounds[:, None, None, :]

    # render_alphas = 1.0 - transmittance[..., None]
    # return render_colors, render_alphas
```

**Key insight for Tier 2:** The loop count is `max_per_tile` (the maximum number of Gaussians in any single tile). For typical scenes this is 50-500. Each iteration is fully vectorized over all `I * H * W` pixels.

### Tensor Shapes at Each Step (Tier 2 Vectorized Forward)

```
Setup:
  pixel_x                 : [W]
  pixel_y                 : [H]
  offsets_flat             : [I * tile_H * tile_W + 1]
  tile_counts              : [I * tile_H * tile_W]
  max_per_tile             : scalar (int)

Per iteration k:
  gauss_idx_in_flat        : [I * tile_H * tile_W]
  valid_tile               : [I * tile_H * tile_W]   bool
  fid                      : [I * tile_H * tile_W]
  flat_g                   : [I * tile_H * tile_W]

  g_mean                   : [I, tile_H, tile_W, 2]
  g_conic                  : [I, tile_H, tile_W, 3]
  g_color                  : [I, tile_H, tile_W, C]
  g_opac                   : [I, tile_H, tile_W]

  After broadcast to pixel grid:
  g_mean_px                : [I, H, W, 2]
  delta                    : [I, H, W, 2]
  sigma                    : [I, H, W]
  alpha                    : [I, H, W]

  Accumulation:
  render_colors            : [I, H, W, C]   (in-place +=)
  transmittance            : [I, H, W]      (in-place *=)

Final:
  render_colors            : [I, H, W, C]
  render_alphas            : [I, H, W, 1]
```

---

## Detailed Backward Algorithm

### Function Signature

```python
def _rasterize_to_pixels_bwd(
    # Forward inputs (saved for backward)
    means2d: mx.array,        # [I, N, 2]
    conics: mx.array,         # [I, N, 3]
    colors: mx.array,         # [I, N, C]
    opacities: mx.array,      # [I, N]
    backgrounds: Optional[mx.array],  # [I, C] or None
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: mx.array,  # [I, tile_H, tile_W]
    flatten_ids: mx.array,    # [n_isects]
    # Forward outputs (saved for backward)
    render_alphas: mx.array,  # [I, H, W, 1]
    last_ids: mx.array,       # [I, H, W]
    # Upstream gradients
    v_render_colors: mx.array,  # [I, H, W, C]
    v_render_alphas: mx.array,  # [I, H, W, 1]
) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """
    Returns:
        v_means2d:    [I, N, 2]
        v_conics:     [I, N, 3]
        v_colors:     [I, N, C]
        v_opacities:  [I, N]
        v_backgrounds: [I, C] or None
    """
```

### Backward Pseudocode (Tier 1 -- Reference)

```python
def _rasterize_to_pixels_bwd_ref(
    means2d_np, conics_np, colors_np, opac_np, backgrounds_np,
    image_width, image_height, tile_size,
    offsets_np, fids_np,
    render_alphas_np, last_ids_np,
    v_render_colors_np, v_render_alphas_np,
    I, N, C, tile_H, tile_W, n_isects
):
    # Initialize gradient accumulators
    v_means2d = np.zeros((I, N, 2), dtype=np.float32)
    v_conics  = np.zeros((I, N, 3), dtype=np.float32)
    v_colors  = np.zeros((I, N, C), dtype=np.float32)
    v_opac    = np.zeros((I, N), dtype=np.float32)

    for img_id in range(I):
        for ty in range(tile_H):
            for tx in range(tile_W):
                tile_id = ty * tile_W + tx
                start = int(offsets_np[img_id, ty, tx])

                # Compute end (same logic as forward)
                if img_id == I - 1 and tile_id == tile_H * tile_W - 1:
                    end = n_isects
                else:
                    flat_tile = img_id * tile_H * tile_W + tile_id
                    next_flat = flat_tile + 1
                    next_img = next_flat // (tile_H * tile_W)
                    next_tile = next_flat % (tile_H * tile_W)
                    next_ty = next_tile // tile_W
                    next_tx = next_tile % tile_W
                    end = int(offsets_np[next_img, next_ty, next_tx])

                if start >= end:
                    continue

                px_start = tx * tile_size
                py_start = ty * tile_size
                px_end = min(px_start + tile_size, image_width)
                py_end = min(py_start + tile_size, image_height)

                for py in range(py_start, py_end):
                    for px in range(px_start, px_end):
                        pixel_x = px + 0.5
                        pixel_y = py + 0.5

                        # T_final: transmittance AFTER the last Gaussian
                        T_final = 1.0 - render_alphas_np[img_id, py, px, 0]
                        T = T_final

                        # Index of last contributing Gaussian for this pixel
                        bin_final = int(last_ids_np[img_id, py, px])

                        # Upstream gradients for this pixel
                        v_rc = v_render_colors_np[img_id, py, px]  # [C]
                        v_ra = v_render_alphas_np[img_id, py, px, 0]  # scalar

                        # Buffer: accumulates weighted color from BEHIND current Gaussian
                        buffer = np.zeros(C, dtype=np.float32)

                        # --- Back-to-front iteration ---
                        for k in range(end - 1, start - 1, -1):
                            # Skip Gaussians beyond the last contributing one
                            if k > bin_final:
                                continue

                            g = int(fids_np[k])
                            gid = g % N
                            iid = g // N
                            if iid != img_id:
                                continue

                            # Recompute forward quantities
                            dx = pixel_x - means2d_np[img_id, gid, 0]
                            dy = pixel_y - means2d_np[img_id, gid, 1]
                            a, b, c_val = conics_np[img_id, gid]
                            sigma = 0.5 * (a * dx*dx + c_val * dy*dy) + b * dx * dy

                            if sigma < 0.0:
                                continue

                            vis = np.exp(-sigma)
                            alpha = min(opac_np[img_id, gid] * vis, MAX_ALPHA)

                            if alpha < ALPHA_THRESHOLD:
                                continue

                            # Recover T_k (transmittance BEFORE this Gaussian)
                            ra = 1.0 / (1.0 - alpha)
                            T = T * ra  # now T = T_k

                            fac = alpha * T

                            # grad_color
                            v_colors[img_id, gid] += fac * v_rc

                            # grad_alpha
                            v_alpha = 0.0
                            for c_idx in range(C):
                                v_alpha += (
                                    colors_np[img_id, gid, c_idx] * T
                                    - buffer[c_idx] * ra
                                ) * v_rc[c_idx]

                            v_alpha += T_final * ra * v_ra

                            # Background contribution to grad_alpha
                            if backgrounds_np is not None:
                                bg_dot = 0.0
                                for c_idx in range(C):
                                    bg_dot += backgrounds_np[img_id, c_idx] * v_rc[c_idx]
                                v_alpha += -T_final * ra * bg_dot

                            # Gradient through sigma (only if alpha was not clamped)
                            if opac_np[img_id, gid] * vis <= MAX_ALPHA:
                                v_sigma = -opac_np[img_id, gid] * vis * v_alpha

                                # grad_conics
                                v_conics[img_id, gid, 0] += 0.5 * v_sigma * dx * dx
                                v_conics[img_id, gid, 1] += v_sigma * dx * dy
                                v_conics[img_id, gid, 2] += 0.5 * v_sigma * dy * dy

                                # grad_means2d
                                v_means2d[img_id, gid, 0] += v_sigma * (a * dx + b * dy)
                                v_means2d[img_id, gid, 1] += v_sigma * (b * dx + c_val * dy)

                                # grad_opacity
                                v_opac[img_id, gid] += vis * v_alpha

                            # Update buffer
                            for c_idx in range(C):
                                buffer[c_idx] += colors_np[img_id, gid, c_idx] * fac

    # grad_backgrounds
    v_backgrounds = None
    if backgrounds_np is not None:
        # v_bg = sum over pixels of v_render_colors * (1 - render_alphas)
        v_backgrounds = np.zeros((I, C), dtype=np.float32)
        for img_id in range(I):
            remaining = 1.0 - render_alphas_np[img_id, :, :, 0]  # [H, W]
            for c_idx in range(C):
                v_backgrounds[img_id, c_idx] = np.sum(
                    v_render_colors_np[img_id, :, :, c_idx] * remaining
                )

    return v_means2d, v_conics, v_colors, v_opac, v_backgrounds
```

### Tensor Shapes at Each Step (Backward)

```
Saved from forward:
  means2d           : [I, N, 2]
  conics            : [I, N, 3]
  colors            : [I, N, C]
  opacities         : [I, N]
  render_alphas     : [I, H, W, 1]
  last_ids          : [I, H, W]

Upstream gradients:
  v_render_colors   : [I, H, W, C]
  v_render_alphas   : [I, H, W, 1]

Per-pixel per-Gaussian (backward, back-to-front):
  T                 : scalar       (recovered transmittance before this Gaussian)
  ra                : scalar       1 / (1 - alpha)
  fac               : scalar       alpha * T
  v_alpha           : scalar       gradient w.r.t. alpha
  v_sigma           : scalar       gradient w.r.t. sigma
  buffer            : [C]          accumulated weighted colors from behind

Output gradients:
  v_means2d         : [I, N, 2]    (accumulated via atomic-add-like pattern)
  v_conics          : [I, N, 3]    (accumulated via atomic-add-like pattern)
  v_colors          : [I, N, C]    (accumulated via atomic-add-like pattern)
  v_opacities       : [I, N]      (accumulated via atomic-add-like pattern)
  v_backgrounds     : [I, C]      (sum over all pixels)
```

### Key Backward Implementation Notes

1. **Gradient accumulation**: Multiple pixels contribute gradients to the same Gaussian. In CUDA, this uses `gpuAtomicAdd`. In MLX, we accumulate into per-Gaussian buffers and use `mx.scatter_add` or equivalent.

2. **Transmittance recovery**: The backward walks back-to-front, recovering $T_k$ by multiplying by $1/(1-\alpha_k)$. This requires recomputing $\alpha_k$ from scratch (recomputing sigma, vis, alpha). The CUDA kernel uses `last_ids` to skip Gaussians beyond the last contributor.

3. **Buffer accumulation**: The `buffer` array tracks $\sum_{j>k} c_j \cdot \alpha_j \cdot T_j$ -- the total color contribution from Gaussians behind the current one. This is needed for the $v_\alpha$ computation.

4. **Alpha clamping guard**: When $\alpha$ was clamped to `MAX_ALPHA`, the gradient through sigma/opacity is zero (the `if opac * vis <= MAX_ALPHA` check). This prevents incorrect gradients when clamping was active.

---

## `@mx.custom_function` Integration

```python
@mx.custom_function
def rasterize_to_pixels(means2d, conics, colors, opacities, ...):
    render_colors, render_alphas, last_ids = _rasterize_to_pixels_fwd(...)
    return render_colors, render_alphas

@rasterize_to_pixels.vjp
def rasterize_to_pixels_vjp(primals, cotangents, output):
    means2d, conics, colors, opacities, ... = primals
    v_render_colors, v_render_alphas = cotangents

    # last_ids was saved during forward (need to use side channel or recompute)
    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = _rasterize_to_pixels_bwd(...)

    return v_means2d, v_conics, v_colors, v_opacities, ...
```

**Challenge with `last_ids`**: The forward pass computes `last_ids` as a side effect. MLX's `@mx.custom_function` VJP receives `primals` and `output` (the returned arrays). We have two options:

- **Option A**: Return `last_ids` as a third output from the forward pass, then ignore it in the training loss but use it in the VJP. The VJP receives it via `output`.
- **Option B**: Recompute `last_ids` in the backward pass (wasteful but simpler).
- **Option C**: Use a closure or module-level cache to pass `last_ids` from forward to backward.

**Recommendation**: Use **Option A** -- return `(render_colors, render_alphas, last_ids)` from the custom function. The caller extracts only the first two for loss computation. The VJP accesses all three via `output`.

```python
@mx.custom_function
def rasterize_to_pixels_fwd(means2d, conics, colors, opacities, ...):
    render_colors, render_alphas, last_ids = _fwd_impl(...)
    return render_colors, render_alphas, last_ids

@rasterize_to_pixels_fwd.vjp
def rasterize_to_pixels_vjp(primals, cotangents, output):
    render_colors, render_alphas, last_ids = output
    v_render_colors, v_render_alphas, _ = cotangents  # last_ids grad is None/zero

    v_means2d, v_conics, v_colors, v_opacities, v_bg = _bwd_impl(
        *primals, render_alphas, last_ids, v_render_colors, v_render_alphas
    )
    # Return cotangents for each primal
    return (v_means2d, v_conics, v_colors, v_opacities, ...)
```

---

## Performance Considerations

### Memory

| Item | Size (640x480, N=100k, C=3) | Notes |
|------|---------------------------|-------|
| `render_colors` | 640 * 480 * 3 * 4 = 3.5 MB | Output |
| `render_alphas` | 640 * 480 * 1 * 4 = 1.2 MB | Output |
| `transmittance` | 640 * 480 * 4 = 1.2 MB | Intermediate (Tier 2) |
| `last_ids` | 640 * 480 * 4 = 1.2 MB | Saved for backward |
| Per-iteration broadcast | 640 * 480 * C * 4 = 3.5 MB | Tier 2: per-Gaussian-slot |

Tier 2 peak memory per iteration is O(I * H * W * C) -- roughly the same as the output. The loop runs `max_per_tile` times, but each iteration is independent (no accumulation of intermediate per-Gaussian data).

### Compute

| Approach | Time complexity | Practical speed |
|----------|----------------|-----------------|
| Tier 1 (Python loops) | O(n_isects * tile_size^2) | Very slow (~seconds for 640x480) |
| Tier 2 (Vectorized MLX) | O(max_per_tile * I * H * W) | Moderate (~100ms for 640x480) |
| Tier 3 (Metal shader) | O(n_isects) parallel | Fast (~5ms for 640x480) |

### Optimization Opportunities (for Tier 2)

1. **Early exit from the Gaussian loop**: Track a global "all pixels saturated" flag. If all pixels have T < threshold, break the loop early.

2. **Tile-level padding**: Pad all tiles to `max_per_tile` Gaussians. Unused slots get opacity=0. This eliminates conditional logic.

3. **Chunked processing**: If `max_per_tile` is very large (>1000), process in chunks to limit memory.

4. **Avoid repeated gather**: Pre-gather all Gaussian data for each tile into contiguous buffers before the compositing loop.

5. **Float16 for colors**: Use half precision for color channels to halve memory bandwidth (forward only; backward needs float32).

---

## File Structure

### Source: `src/gsplat_mlx/core/rasterization.py`

```python
"""
Pixel-level rasterization: alpha-composite sorted Gaussians into a final image.

This module implements the core rendering loop of 3D Gaussian Splatting.
For each pixel, it evaluates overlapping Gaussians and composites them
front-to-back to produce the rendered image.

Forward: tile-based alpha compositing
Backward: reverse-order gradient accumulation via @mx.custom_function VJP
"""

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple

# Constants (matching upstream gsplat)
ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4


def rasterize_to_pixels(...) -> Tuple[mx.array, mx.array]:
    """Public API: rasterize Gaussians to pixels with differentiable backward."""
    ...

def _rasterize_to_pixels_fwd(...) -> Tuple[mx.array, mx.array, mx.array]:
    """Forward pass: alpha compositing. Returns (colors, alphas, last_ids)."""
    ...

def _rasterize_to_pixels_bwd(...) -> Tuple[mx.array, mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Backward pass: reverse-order gradient accumulation."""
    ...
```

### Tests: `tests/test_rasterization.py`

---

## Test Plan

### Forward Pass Tests

| Test | Description | Key assertion |
|------|-------------|---------------|
| `test_single_gaussian_render` | One Gaussian at image center, verify Gaussian blob shape | Peak pixel color matches `opacity * color`, falloff follows exp(-sigma) |
| `test_single_gaussian_falloff` | Check specific pixels at known distances from Gaussian center | Pixel values match analytic `T * alpha * color` within atol=1e-4 |
| `test_two_gaussian_overlap` | Two overlapping Gaussians with different colors | Front Gaussian's contribution is `alpha1 * color1`, back is `(1-alpha1) * alpha2 * color2` |
| `test_depth_ordering` | Two Gaussians at same position, different depths | Closer Gaussian dominates; swap depth order and verify output changes |
| `test_background_color` | Single semi-transparent Gaussian with blue background | Verify `pixel = T*alpha*color + (1-alpha)*background` |
| `test_empty_tile` | No Gaussians in a tile region | Output is background (or black if no background) |
| `test_full_opacity` | Gaussian with opacity=1.0 (clamped to MAX_ALPHA=0.99) | Background bleeds through slightly (1% transmittance) |
| `test_transmittance_cutoff` | 100 overlapping Gaussians, all with opacity=0.5 | After ~20 Gaussians, transmittance < 1e-4, remaining are skipped |
| `test_alpha_threshold_skip` | Gaussian far from all pixels (sigma very large) | alpha < 1/255, Gaussian has zero contribution |
| `test_negative_sigma_skip` | Malformed conic producing sigma < 0 | Gaussian is skipped, no contribution |
| `test_max_alpha_clamp` | Gaussian with very high opacity and small sigma | Alpha clamped to 0.99, not 1.0 |
| `test_multi_channel` | C=16 feature rendering | All 16 channels rendered correctly |
| `test_batch_render` | I=2 images with different Gaussian configurations | Each image rendered independently |
| `test_pixel_centers` | Verify pixel coordinates are at (px+0.5, py+0.5) | Gaussian at (0.5, 0.5) has maximum response at pixel (0, 0) |
| `test_last_ids` | Verify `last_ids` correctly tracks the last contributing Gaussian | Match against manual computation |
| `test_tile_boundary_gaussian` | Gaussian straddling two tiles | Both tiles render it correctly |

### VJP (Backward Pass) Tests

| Test | Description | Key assertion |
|------|-------------|---------------|
| `test_vjp_colors` | Gradient of loss w.r.t. Gaussian colors | Finite-difference verification, atol=1e-3 |
| `test_vjp_opacities` | Gradient of loss w.r.t. Gaussian opacities | Finite-difference verification, atol=1e-3 |
| `test_vjp_means2d` | Gradient of loss w.r.t. 2D positions | Finite-difference verification, atol=1e-3 |
| `test_vjp_conics` | Gradient of loss w.r.t. conic parameters | Finite-difference verification, atol=1e-3 |
| `test_vjp_backgrounds` | Gradient of loss w.r.t. background color | Finite-difference verification, atol=1e-3 |
| `test_vjp_numerical_full` | Full numerical Jacobian check for all parameters | Perturb each input element, verify gradient matches |
| `test_vjp_zero_grad` | Loss independent of some Gaussians (outside view) | Gradients for those Gaussians are zero |
| `test_vjp_clamped_alpha` | Gaussian whose alpha hits MAX_ALPHA | grad_sigma and grad_means2d are zero (clamped) |
| `test_vjp_single_pixel` | Gradient through a single pixel with one Gaussian | Analytically verify all partial derivatives |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_rasterize_with_projection` | Chain: projection (PRD-05) -> intersection (PRD-06) -> rasterization (PRD-07) |
| `test_rasterize_multi_camera` | Two cameras viewing same scene, independent renders |
| `test_end_to_end_gradient` | Full backward through rasterize -> intersect -> project chain |

### Cross-Framework Tests

| Test | Description | Tolerance |
|------|-------------|-----------|
| `test_cross_framework_forward` | Compare rendered image against PyTorch `_rasterize_to_pixels` on identical inputs | atol=1e-4 per pixel |
| `test_cross_framework_backward` | Compare all gradients against PyTorch backward pass | atol=1e-3 per gradient element |
| `test_cross_framework_scene` | Known synthetic scene (e.g., 5 Gaussians, 64x64 image) with reference output saved as fixture | Exact pixel match within tolerance |

### Tolerances

| Quantity | Absolute tolerance | Rationale |
|----------|-------------------|-----------|
| Forward pixel values | atol=1e-4 | Single exp + multiply chain |
| Forward alpha values | atol=1e-5 | Simple product chain |
| VJP grad_colors | atol=1e-3 | Accumulated through compositing chain |
| VJP grad_opacities | atol=1e-3 | Chain through exp and product |
| VJP grad_means2d | atol=1e-3 | Chain through sigma computation |
| VJP grad_conics | atol=1e-3 | Chain through sigma computation |
| Finite difference step | eps=1e-4 | Balance between truncation and roundoff |

---

## Edge Cases and Numerical Considerations

1. **Division by (1 - alpha) in backward**: When alpha is close to 1.0 (clamped to 0.99), `1/(1-alpha) = 100`. This amplifies numerical errors. The CUDA kernel notes this and considered using double precision for transmittance (but opted for float32 for speed). We should monitor this.

2. **Very small transmittance**: When T approaches `TRANSMITTANCE_THRESHOLD` (1e-4), the backward division `T *= 1/(1-alpha)` recovers larger T values. This is numerically stable in the backward direction.

3. **Gradient accumulation from many pixels**: A single Gaussian may contribute to thousands of pixels. Its gradient is the sum of all pixel-level gradients. In CUDA, this uses atomicAdd. In MLX, we use scatter-add or index-based accumulation.

4. **Empty tiles**: Tiles with no Gaussians should produce zero color and zero alpha (or background). The offset start == end check handles this.

5. **Edge tiles**: Tiles at the image boundary may be smaller than tile_size. Pixels outside [0, image_width) x [0, image_height) must not be written.

6. **Gaussian contributing to multiple tiles**: The same Gaussian may appear in flatten_ids multiple times (once per tile it overlaps). Gradients from all tile instances must be accumulated.

---

## Dependencies

- **PRD-01**: Development environment (MLX, pytest, project structure)
- **PRD-06**: Tile intersection (`isect_tiles`, `isect_offset_encode`) provides `isect_offsets` and `flatten_ids` inputs

## Blocks

- **PRD-09**: Rendering API (wraps rasterization with projection + intersection into a single call)
- **PRD-13**: Training loop (requires backward pass for optimization)
- **PRD-14**: Metal shader optimization (replaces Tier 1/2 with GPU kernel)

## Acceptance Criteria

- [ ] Single Gaussian renders correct pixel values (within atol=1e-4 of analytic computation)
- [ ] Multi-Gaussian alpha compositing matches expected front-to-back blending
- [ ] Background color correctly blended where transmittance > 0
- [ ] Early termination works (transmittance < 1e-4 stops accumulation)
- [ ] MAX_ALPHA clamping prevents fully opaque single Gaussians (alpha capped at 0.99)
- [ ] Alpha threshold skipping works (alpha < 1/255 produces no contribution)
- [ ] Negative sigma values are rejected
- [ ] Supports arbitrary number of color channels (C=1, 3, 16, etc.)
- [ ] Supports batch rendering (I > 1 images)
- [ ] `last_ids` correctly tracks the last contributing Gaussian index per pixel
- [ ] Backward pass produces correct gradients for colors (verified by finite differences)
- [ ] Backward pass produces correct gradients for opacities (verified by finite differences)
- [ ] Backward pass produces correct gradients for means2d (verified by finite differences)
- [ ] Backward pass produces correct gradients for conics (verified by finite differences)
- [ ] Backward pass produces correct gradients for backgrounds (verified by finite differences)
- [ ] Rendered image matches torch `_RasterizeToPixels` forward for synthetic test scenes
- [ ] All gradients match torch `_RasterizeToPixels` backward for synthetic test scenes
- [ ] All tests pass with `pytest tests/test_rasterization.py -v`
