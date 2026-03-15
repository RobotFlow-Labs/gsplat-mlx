# PRD-08: Alpha Compositing with Explicit Intersections (accumulate)

## Overview

Port the `accumulate()` function from `_torch_impl.py` to MLX. This function provides a pure-Python alpha compositing path that works with explicit (gaussian_id, pixel_id, image_id) intersection pairs rather than the tile-based fused rasterizer (PRD-07). In the upstream gsplat, it depends on the `nerfacc` library for `render_weight_from_alpha` and `accumulate_along_rays`. For the MLX port, we implement these two nerfacc functions from scratch using pure MLX operations, eliminating the external dependency entirely.

This function is useful as a flexible compositing backend: it lets callers specify arbitrary subsets of Gaussian-pixel intersections (e.g., from `rasterize_to_indices_in_range`), enabling per-pixel compositing without re-running the full tile-based pipeline. Because it uses standard differentiable operations (no custom CUDA kernels), the backward pass comes for free from MLX's automatic differentiation.

## Source Reference

- **Primary**: `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py:459-557` (`accumulate`)
- **External dependency**: `nerfacc.render_weight_from_alpha`, `nerfacc.accumulate_along_rays`
- **Public API**: `_wrapper.py` (called from the public `accumulate` wrapper)
- **Lines**: ~100 lines for `accumulate`, plus ~50 lines of nerfacc logic to reimplement
- **Constant**: `MAX_ALPHA = 0.999` (defined earlier in `_torch_impl.py`)

## Scope

### In Scope
- `render_weight_from_alpha(alphas, ray_indices, n_rays)` -- segment-wise exclusive cumprod to compute front-to-back compositing weights (replaces nerfacc)
- `accumulate_along_rays(weights, values, ray_indices, n_rays)` -- weighted scatter-add grouped by ray (replaces nerfacc)
- `accumulate(means2d, conics, opacities, colors, gaussian_ids, pixel_ids, image_ids, image_width, image_height)` -- the main compositing function
- Forward pass producing rendered image + alpha map
- Backward pass via MLX's automatic differentiation (all ops are natively differentiable)

### Out of Scope
- The `nerfacc` Python package itself -- we do NOT import or wrap it
- Packed input format
- `absgrad` mode (absolute gradient computation for densification)
- 2DGS variant (`accumulate_2dgs`) -- separate PRD
- `accumulate_eval3d` -- separate function with different semantics
- Custom Metal kernels for the scatter operations (future optimization)
- The `rasterize_to_indices_in_range` function that produces the intersection pairs (handled in PRD-07 or a future PRD)

## Technical Design

### Key Functions to Implement

| Function | Origin | MLX Approach |
|----------|--------|-------------|
| `render_weight_from_alpha` | nerfacc | Segment-wise exclusive cumprod via log-space cumsum with segment boundary correction |
| `accumulate_along_rays` | nerfacc | `mx.zeros(...).at[ray_indices].add(weights * values)` scatter-add |
| `accumulate` | `_torch_impl.py:459-557` | Direct port replacing nerfacc calls with our implementations |

### File: `src/gsplat_mlx/core/accumulate.py`

### Constants

```python
MAX_ALPHA = 0.999  # Maximum alpha value, prevents fully opaque single Gaussians
```

This constant must match the upstream value exactly. It is used in both `accumulate` (this PRD) and `rasterize_to_pixels` (PRD-07).

---

### Function 1: `render_weight_from_alpha`

#### Semantics

Given a flat array of alpha values and corresponding ray indices (which ray each alpha belongs to), compute the front-to-back compositing weight for each intersection. Intersections belonging to the same ray must be contiguous and ordered front-to-back.

For each ray, the compositing follows the standard volume rendering equation:

```
transmittance[0] = 1.0
transmittance[i] = prod(1 - alpha[0], 1 - alpha[1], ..., 1 - alpha[i-1])
weight[i] = transmittance[i] * alpha[i]
```

This is equivalent to a **segment-wise exclusive cumprod** of `(1 - alpha)`.

#### Signature

```python
def render_weight_from_alpha(
    alphas: mx.array,       # [M] float32, alpha values in [0, 1]
    ray_indices: mx.array,  # [M] int32, which ray each alpha belongs to
    n_rays: int,            # total number of rays
) -> Tuple[mx.array, mx.array]:
    """Compute front-to-back compositing weights from per-intersection alphas.

    Intersections belonging to the same ray MUST be contiguous in the input
    arrays and sorted front-to-back (nearest first).

    Args:
        alphas: Per-intersection opacity values. Shape [M].
        ray_indices: Ray index for each intersection. Shape [M]. Values in [0, n_rays).
        n_rays: Total number of rays (used for output sizing, not directly in computation).

    Returns:
        weights: Per-intersection compositing weights. Shape [M].
        transmittances: Per-intersection transmittance (exclusive cumprod of 1-alpha). Shape [M].
    """
```

#### Algorithm: Segment-Wise Exclusive Cumprod

The core challenge is computing an exclusive cumprod within each segment (ray) without a dedicated `segment_cumprod` primitive. MLX provides `mx.cumsum` but not `mx.cumprod`. We use the **log-space approach**:

```
cumprod(1 - alpha) = exp(cumsum(log(1 - alpha)))
```

But a global `cumsum` bleeds across ray boundaries. We correct this by subtracting the accumulated log at each segment start.

**Complete algorithm (step by step):**

1. Compute `log(1 - alpha)` for each intersection, clamping `(1 - alpha)` to `[1e-10, inf)` to avoid `log(0)`.
2. Compute the global inclusive cumulative sum of these log values.
3. Convert from inclusive to exclusive cumsum by shifting right by one position (insert 0 at front).
4. Identify segment boundaries: index 0 and any index where `ray_indices[i] != ray_indices[i-1]`.
5. At each segment start, the exclusive cumsum should be 0 (transmittance = 1.0). Compute the correction by extracting the exclusive cumsum value at each segment start and subtracting it from all elements in that segment.
6. Exponentiate the corrected exclusive cumsum to get transmittance.
7. Multiply transmittance by alpha to get weights.

#### Reference Implementation (NumPy fallback -- for testing only)

This implementation uses a simple Python loop and serves as a correctness baseline. It is NOT differentiable through MLX.

```python
def _render_weight_from_alpha_numpy(alphas, ray_indices, n_rays):
    """Reference implementation using numpy loop. NOT differentiable."""
    import numpy as np

    alphas_np = np.array(alphas)
    ray_indices_np = np.array(ray_indices)
    M = len(alphas_np)

    weights = np.zeros(M, dtype=np.float32)
    transmittances = np.zeros(M, dtype=np.float32)

    # Track transmittance per ray
    T = np.ones(n_rays, dtype=np.float32)

    for i in range(M):
        ray_id = int(ray_indices_np[i])
        transmittances[i] = T[ray_id]
        weights[i] = T[ray_id] * alphas_np[i]
        T[ray_id] *= (1.0 - alphas_np[i])

    return mx.array(weights), mx.array(transmittances)
```

#### Production Implementation (Fully Differentiable, Pure MLX)

For the production implementation that supports `mx.grad()`, all operations must stay in MLX:

```python
def render_weight_from_alpha(alphas, ray_indices, n_rays):
    M = alphas.shape[0]
    if M == 0:
        return mx.array([], dtype=mx.float32), mx.array([], dtype=mx.float32)

    # Step 1: Compute log(1 - alpha) with clamping to avoid log(0)
    # When alpha = MAX_ALPHA = 0.999, (1 - alpha) = 0.001, which is safe.
    # Clamp to 1e-10 as a safety net for any alpha that slips through.
    one_minus_alpha = mx.clip(1.0 - alphas, a_min=1e-10, a_max=None)
    log_oma = mx.log(one_minus_alpha)  # [M]

    # Step 2: Global inclusive cumulative sum in log space
    log_cumsum = mx.cumsum(log_oma)  # [M]

    # Step 3: Convert inclusive cumsum to EXCLUSIVE cumsum
    # exclusive[i] = sum of log(1-alpha) for indices 0..i-1
    # exclusive[0] = 0 (no prior contributions)
    log_exclusive = mx.concatenate([mx.zeros(1), log_cumsum[:-1]])  # [M]

    # Step 4: Identify segment boundaries
    # A new segment starts at index 0 and wherever ray_indices changes.
    shifted_rays = mx.concatenate([ray_indices[:1] - 1, ray_indices[:-1]])  # [M]
    is_start = (ray_indices != shifted_rays)  # [M] bool

    # Step 5: Compute per-segment correction
    # At each segment start, log_exclusive should reset to 0.
    # The global cumsum includes contributions from previous segments.
    # We subtract log_exclusive[segment_start] from all elements in that segment.

    # Assign each element a 0-indexed segment ID via cumsum of is_start
    segment_ids = mx.cumsum(is_start.astype(mx.int32)) - 1  # [M]
    n_segments = mx.max(segment_ids).item() + 1 if M > 0 else 0

    # Scatter: store each segment start's cumsum value into a compact array.
    # At segment starts, is_start=1 so the value passes through; elsewhere is_start=0.
    start_mask = is_start.astype(mx.float32)  # [M], 1.0 at starts, 0.0 elsewhere
    corrections_compact = mx.zeros(n_segments, dtype=mx.float32)
    corrections_compact = corrections_compact.at[segment_ids].add(
        log_exclusive * start_mask
    )
    # Since only segment starts contribute (mask=1), and each segment_id appears
    # exactly once as a start, this is a clean scatter with no races.

    # Step 6: Gather correction for each element
    corrections = corrections_compact[segment_ids]  # [M]

    # Corrected exclusive cumsum: now segment-local
    log_transmittance = log_exclusive - corrections  # [M]

    # Step 7: Exponentiate to get transmittance
    transmittances = mx.exp(log_transmittance)  # [M]

    # Step 8: Compute weights = transmittance * alpha
    weights = transmittances * alphas  # [M]

    return weights, transmittances
```

**Why this is fully differentiable:** Every operation (`mx.clip`, `mx.log`, `mx.cumsum`, `mx.concatenate`, comparison, `mx.cumsum` on int cast, `.at[].add()`, indexing, `mx.exp`, multiply) is a standard MLX op with defined gradients. The `is_start` mask and `segment_ids` are derived from integer comparisons on `ray_indices` -- these are treated as constants (no gradient needed with respect to ray indices) and do not break the computation graph for the actual differentiable inputs (`alphas`).

**Important correctness notes:**

1. The `exclusive` cumsum means `transmittance[0] = exp(0) = 1.0` for each segment, which is correct: the first intersection along a ray sees full transmittance.
2. The clamping of `(1 - alpha)` to `1e-10` prevents `-inf` in log space but introduces a tiny error for fully opaque Gaussians (`alpha = 1.0`). This is acceptable because `MAX_ALPHA = 0.999` already prevents alpha from reaching exactly 1.0.
3. The `start_mask` multiplication ensures that only segment-start values are scattered, avoiding race conditions in the scatter-add.
4. Float32 precision in log space: for typical 3DGS scenes with 10-50 Gaussians per ray, the cumulative rounding error is bounded by `M * eps` where `eps ~ 1e-7`, well within tolerance.

---

### Function 2: `accumulate_along_rays`

#### Semantics

Perform a weighted scatter-add: for each intersection, add `weight[i] * value[i]` to the output at position `ray_indices[i]`.

When `values` is `None`, accumulate just the weights (used for computing total alpha per ray).

#### Signature

```python
def accumulate_along_rays(
    weights: mx.array,      # [M] float32
    values: mx.array,       # [M, C] float32 or None
    ray_indices: mx.array,  # [M] int32
    n_rays: int,            # total number of output rays
) -> mx.array:
    """Accumulate weighted values along rays via scatter-add.

    For each intersection i, adds weights[i] * values[i] to output[ray_indices[i]].
    If values is None, accumulates weights only (equivalent to values = ones).

    Args:
        weights: Per-intersection weights. Shape [M].
        values: Per-intersection feature vectors. Shape [M, C] or None.
        ray_indices: Ray index for each intersection. Shape [M].
        n_rays: Total number of output rays.

    Returns:
        accumulated: Shape [n_rays, C] if values provided, [n_rays, 1] if values is None.
    """
```

#### Algorithm

```python
def accumulate_along_rays(weights, values, ray_indices, n_rays):
    if values is not None:
        # values: [M, C], weights: [M] -> weighted: [M, C]
        C = values.shape[-1]
        weighted = weights[:, None] * values  # [M, C]
        output = mx.zeros((n_rays, C), dtype=mx.float32)
        output = output.at[ray_indices].add(weighted)
    else:
        # Accumulate weights only -> output shape [n_rays, 1]
        output = mx.zeros((n_rays, 1), dtype=mx.float32)
        output = output.at[ray_indices].add(weights[:, None])

    return output
```

**Differentiability note:** The `mx.array.at[].add()` operation is differentiable in MLX. The gradient of scatter-add with respect to the values being scattered is a gather: `grad_values[i] = grad_output[ray_indices[i]]`. MLX handles this automatically.

**Edge case:** When multiple intersections share the same `ray_indices` value, their contributions are summed. This is the desired behavior for compositing.

---

### Function 3: `accumulate` (Main Compositing Function)

#### Signature

```python
def accumulate(
    means2d: mx.array,      # [..., N, 2]
    conics: mx.array,       # [..., N, 3]
    opacities: mx.array,    # [..., N]
    colors: mx.array,       # [..., N, channels]
    gaussian_ids: mx.array, # [M] int32
    pixel_ids: mx.array,    # [M] int32
    image_ids: mx.array,    # [M] int32
    image_width: int,
    image_height: int,
) -> Tuple[mx.array, mx.array]:
    """Alpha compositing of 2D Gaussians with explicit intersection pairs.

    Given explicit (gaussian_id, pixel_id, image_id) tuples specifying which
    Gaussians contribute to which pixels, compute front-to-back alpha compositing.
    The intersections must be sorted front-to-back within each ray (pixel).

    This is a flexible alternative to the tile-based fused rasterizer (PRD-07).
    It is slower but allows arbitrary intersection patterns and relies on
    MLX's automatic differentiation for the backward pass.

    Args:
        means2d: 2D Gaussian centers. Shape [..., N, 2].
        conics: Inverse 2D covariance (upper triangle: a, b, c). Shape [..., N, 3].
        opacities: Per-Gaussian opacity. Shape [..., N].
        colors: Per-Gaussian color/features. Shape [..., N, channels].
        gaussian_ids: Which Gaussian each intersection refers to. Shape [M].
        pixel_ids: Which pixel (row-major index) each intersection refers to. Shape [M].
        image_ids: Which image (batch index) each intersection refers to. Shape [M].
        image_width: Width of the output image in pixels.
        image_height: Height of the output image in pixels.

    Returns:
        renders: Composited colors. Shape [..., image_height, image_width, channels].
        alphas: Composited opacity. Shape [..., image_height, image_width, 1].
    """
```

#### Algorithm (Step by Step)

```python
import math
import mlx.core as mx

MAX_ALPHA = 0.999

def accumulate(means2d, conics, opacities, colors, gaussian_ids, pixel_ids,
               image_ids, image_width, image_height):
    # ---- Step 0: Shape bookkeeping ----
    image_dims = means2d.shape[:-2]       # e.g., () or (I,) or (B, V)
    I = math.prod(image_dims) if image_dims else 1
    N = means2d.shape[-2]
    channels = colors.shape[-1]

    # Validate shapes
    assert means2d.shape == image_dims + (N, 2), f"means2d shape {means2d.shape}"
    assert conics.shape == image_dims + (N, 3), f"conics shape {conics.shape}"
    assert opacities.shape == image_dims + (N,), f"opacities shape {opacities.shape}"
    assert colors.shape == image_dims + (N, channels), f"colors shape {colors.shape}"

    # Flatten batch dimensions
    means2d_flat = means2d.reshape(I, N, 2)      # [I, N, 2]
    conics_flat  = conics.reshape(I, N, 3)        # [I, N, 3]
    opacities_flat = opacities.reshape(I, N)      # [I, N]
    colors_flat  = colors.reshape(I, N, channels) # [I, N, C]

    # ---- Step 1: Compute pixel coordinates ----
    # pixel_ids is a row-major pixel index: pixel_id = py * image_width + px
    pixel_ids_x = pixel_ids % image_width          # [M]
    pixel_ids_y = pixel_ids // image_width          # [M]
    # Pixel centers are at (px + 0.5, py + 0.5)
    pixel_coords = mx.stack(
        [pixel_ids_x, pixel_ids_y], axis=-1
    ).astype(mx.float32) + 0.5  # [M, 2]

    # ---- Step 2: Gather Gaussian parameters for each intersection ----
    # Index with (image_ids, gaussian_ids) to get per-intersection values.
    means2d_selected = means2d_flat[image_ids, gaussian_ids]    # [M, 2]
    conics_selected  = conics_flat[image_ids, gaussian_ids]     # [M, 3]
    opacities_selected = opacities_flat[image_ids, gaussian_ids]  # [M]
    colors_selected  = colors_flat[image_ids, gaussian_ids]     # [M, C]

    # ---- Step 3: Compute sigma (Gaussian exponent) ----
    deltas = pixel_coords - means2d_selected  # [M, 2]
    sigmas = (
        0.5 * (conics_selected[:, 0] * deltas[:, 0] ** 2
             + conics_selected[:, 2] * deltas[:, 1] ** 2)
        + conics_selected[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]

    # ---- Step 4: Compute per-intersection alpha ----
    alphas_per_isect = mx.minimum(
        opacities_selected * mx.exp(-sigmas),
        MAX_ALPHA
    )  # [M]

    # ---- Step 5: Compute ray indices ----
    # Each unique (image_id, pixel_id) pair is a unique ray.
    # ray_index = image_id * H * W + pixel_id
    ray_indices = image_ids * (image_height * image_width) + pixel_ids  # [M]
    total_pixels = I * image_height * image_width

    # ---- Step 6: Compute front-to-back compositing weights ----
    weights, transmittances = render_weight_from_alpha(
        alphas_per_isect, ray_indices, total_pixels
    )  # [M], [M]

    # ---- Step 7: Accumulate weighted colors ----
    renders = accumulate_along_rays(
        weights, colors_selected, ray_indices, total_pixels
    )  # [total_pixels, C]

    # ---- Step 8: Accumulate alpha (weights only) ----
    alphas_out = accumulate_along_rays(
        weights, None, ray_indices, total_pixels
    )  # [total_pixels, 1]

    # ---- Step 9: Reshape to image ----
    renders = renders.reshape(image_dims + (image_height, image_width, channels))
    alphas_out = alphas_out.reshape(image_dims + (image_height, image_width, 1))

    return renders, alphas_out
```

### Data Flow

```
Inputs                              Processing                        Outputs
--------                            ----------                        -------
means2d [..., N, 2]  ─┐
conics  [..., N, 3]  ─┤
opacities [..., N]   ─┤─→ Gather by (image_ids, gaussian_ids) ─→ deltas, sigmas, alphas [M]
colors  [..., N, C]  ─┘                                              │
                                                                      │
gaussian_ids [M] ─┐                                                   │
pixel_ids    [M] ─┤─→ pixel_coords [M, 2]                           │
image_ids    [M] ─┘   ray_indices  [M]                               │
                          │                                           │
                          ▼                                           ▼
                   render_weight_from_alpha(alphas, ray_indices, n_rays)
                          │
                          ▼
                   weights [M], transmittances [M]
                          │
                   ┌──────┴──────┐
                   ▼              ▼
        accumulate_along_rays   accumulate_along_rays
        (weights, colors, ...)  (weights, None, ...)
                   │              │
                   ▼              ▼
        renders [...,H,W,C]   alphas [...,H,W,1]
```

### Differentiation Strategy

All operations in the forward pass are composed of standard differentiable MLX primitives:

| Operation | MLX Primitive | Gradient Support |
|-----------|--------------|-----------------|
| Array indexing `a[idx1, idx2]` | Gather | Yes (scatter in backward) |
| `mx.exp(-sigmas)` | Elementwise | Yes |
| `mx.minimum(x, MAX_ALPHA)` | Elementwise | Yes (passes gradient where x < MAX_ALPHA) |
| `mx.log(1 - alpha)` | Elementwise | Yes |
| `mx.cumsum(x)` | Scan | Yes |
| `array.at[idx].add(val)` | Scatter-add | Yes (gather in backward) |
| `mx.stack`, `mx.reshape` | View/layout | Yes |

Because every operation is natively differentiable, **no `@mx.custom_function` is needed**. MLX's automatic differentiation will produce correct gradients for `means2d`, `conics`, `opacities`, and `colors` through the full compositing pipeline.

**Important caveat:** The numpy-based reference implementation of `render_weight_from_alpha` breaks the computation graph. For gradient support, the **fully vectorized MLX implementation** (the production version above) must be used. The numpy version is suitable only as a correctness baseline for forward-pass testing.

---

## Edge Cases and Numerical Considerations

### Empty Intersections
When `M = 0` (no intersections), all functions must return correctly shaped empty/zero arrays. The output images should be all zeros.

### Single Intersection Per Ray
When each ray has exactly one Gaussian, `transmittance = 1.0` and `weight = alpha`. The output is simply `alpha * color` for each pixel.

### Fully Opaque First Gaussian
When `alpha[0] = MAX_ALPHA = 0.999` for the first intersection on a ray, the transmittance for subsequent intersections drops to `0.001`, making them near-invisible. The `MAX_ALPHA` clamp prevents numerical issues with `log(0)`.

### Negative Sigma
When sigma < 0 (pixel is inside the Gaussian but the conic computation yields a negative value due to numerical issues), `exp(-sigma)` with negative sigma gives `exp(positive) > 1`, but the `mx.minimum(..., MAX_ALPHA)` clamp prevents the alpha from exceeding `MAX_ALPHA`. This matches upstream behavior.

### Unsorted Intersections
The algorithm assumes intersections are sorted front-to-back within each ray. If intersections are unsorted, the compositing result will be incorrect. The function does NOT sort internally -- callers are responsible for providing sorted inputs (typically from `rasterize_to_indices_in_range` or the tile-based intersection pipeline from PRD-06).

### Float32 Precision in Log Space
The log-space cumsum approach can accumulate rounding errors over long rays (many intersections). For typical 3DGS scenes with 10-50 Gaussians per ray, float32 precision is sufficient. For extreme cases (hundreds of intersections per ray), the error is bounded by `M_per_ray * eps` where `eps ~ 1e-7`.

---

## Test Plan

### File: `tests/test_accumulate.py`

---

#### `render_weight_from_alpha` Tests

| Test Case | Description | Key Assertions |
|-----------|-------------|---------------|
| `test_rwfa_single_ray_basic` | One ray with 3 intersections, alpha = [0.5, 0.3, 0.8] | trans = [1.0, 0.5, 0.35], weights = [0.5, 0.15, 0.28], sum(weights) <= 1.0 |
| `test_rwfa_opaque_first` | alpha = [0.999, 0.5, 0.5] (first is MAX_ALPHA) | weight[0] ~= 0.999, weight[1] ~= 0.0005, weight[2] ~= 0.00025 |
| `test_rwfa_all_transparent` | alpha = [0.1, 0.1, 0.1] for one ray | weights decrease exponentially: [0.1, 0.09, 0.081] |
| `test_rwfa_multi_ray` | Two rays with different alphas, verify independence | Each ray's weights computed independently, no cross-contamination |
| `test_rwfa_single_intersection` | One intersection per ray | weight = alpha, transmittance = 1.0 |
| `test_rwfa_empty` | M = 0, no intersections | Returns empty arrays with shape [0] |
| `test_rwfa_weight_sum_leq_one` | Random alphas, many rays | For each ray, sum(weights) <= 1.0 (with tolerance for float32) |
| `test_rwfa_transmittance_monotonic` | Random alphas per ray | Transmittance is non-increasing within each ray |

**Example test implementation:**

```python
def test_rwfa_single_ray_basic():
    """One ray with 3 intersections, verify weight computation."""
    from gsplat_mlx.core.accumulate import render_weight_from_alpha

    alphas = mx.array([0.5, 0.3, 0.8])
    ray_indices = mx.array([0, 0, 0], dtype=mx.int32)

    weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=1)

    # transmittance[0] = 1.0
    # transmittance[1] = 1.0 * (1 - 0.5) = 0.5
    # transmittance[2] = 0.5 * (1 - 0.3) = 0.35
    expected_trans = mx.array([1.0, 0.5, 0.35])
    expected_weights = mx.array([0.5, 0.15, 0.28])

    np.testing.assert_allclose(np.array(trans), np.array(expected_trans), atol=1e-5)
    np.testing.assert_allclose(np.array(weights), np.array(expected_weights), atol=1e-5)
    assert np.array(weights).sum() <= 1.0 + 1e-6


def test_rwfa_multi_ray():
    """Two rays, verify independence."""
    from gsplat_mlx.core.accumulate import render_weight_from_alpha

    # Ray 0: alpha = [0.5, 0.3]
    # Ray 1: alpha = [0.8, 0.2]
    alphas = mx.array([0.5, 0.3, 0.8, 0.2])
    ray_indices = mx.array([0, 0, 1, 1], dtype=mx.int32)

    weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=2)

    # Ray 0: trans = [1.0, 0.5], weights = [0.5, 0.15]
    # Ray 1: trans = [1.0, 0.2], weights = [0.8, 0.04]
    expected_trans = mx.array([1.0, 0.5, 1.0, 0.2])
    expected_weights = mx.array([0.5, 0.15, 0.8, 0.04])

    np.testing.assert_allclose(np.array(trans), np.array(expected_trans), atol=1e-5)
    np.testing.assert_allclose(np.array(weights), np.array(expected_weights), atol=1e-5)
```

---

#### `accumulate_along_rays` Tests

| Test Case | Description | Key Assertions |
|-----------|-------------|---------------|
| `test_aar_scatter_basic` | 3 intersections, 2 rays, 1 channel | output[ray_0] = sum of weighted values for ray 0 |
| `test_aar_scatter_multi_channel` | C = 3 channels (RGB) | Each channel accumulated independently |
| `test_aar_no_values` | values = None, accumulate weights only | output shape [n_rays, 1], values are summed weights |
| `test_aar_empty` | M = 0 | Returns zeros with correct shape |
| `test_aar_single_ray_all_intersections` | All intersections belong to ray 0 | output[0] = sum of all weighted values, others = 0 |
| `test_aar_no_overlap` | Each intersection maps to a unique ray | output[i] = weight[i] * value[i] exactly |

**Example test implementation:**

```python
def test_aar_scatter_basic():
    """Verify weighted scatter-add for simple case."""
    from gsplat_mlx.core.accumulate import accumulate_along_rays

    weights = mx.array([0.5, 0.3, 0.8])
    values = mx.array([[1.0], [2.0], [3.0]])  # [3, 1]
    ray_indices = mx.array([0, 0, 1], dtype=mx.int32)

    result = accumulate_along_rays(weights, values, ray_indices, n_rays=2)

    # ray 0: 0.5 * 1.0 + 0.3 * 2.0 = 1.1
    # ray 1: 0.8 * 3.0 = 2.4
    expected = mx.array([[1.1], [2.4]])
    np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-5)
```

---

#### `accumulate` Tests (Integration)

| Test Case | Description | Key Assertions |
|-----------|-------------|---------------|
| `test_single_gaussian_single_pixel` | One Gaussian at pixel center, one intersection | render = alpha * color, alpha_out = alpha |
| `test_single_gaussian_offset_pixel` | Gaussian offset from pixel center | render = alpha(sigma) * color where sigma depends on offset |
| `test_two_gaussians_front_to_back` | Two Gaussians at same pixel, sorted by depth | Front Gaussian contributes more than back one |
| `test_full_occlusion` | Front Gaussian with alpha ~= MAX_ALPHA | Back Gaussian contributes almost nothing |
| `test_multiple_pixels` | One Gaussian visible from 4 pixels | Each pixel gets correct alpha based on distance |
| `test_multi_channel` | channels = 16 (feature rendering) | All channels composited correctly |
| `test_batch_images` | I = 2 images, different Gaussians per image | Images rendered independently |
| `test_empty_intersections` | No intersections provided | Output is all zeros |
| `test_output_shape` | Verify output shapes for various image_dims | renders: [..., H, W, C], alphas: [..., H, W, 1] |
| `test_alpha_sum_leq_one` | Random scene | All output alpha values in [0, 1] |

**Example test implementation:**

```python
def test_single_gaussian_single_pixel():
    """One Gaussian centered on a pixel, one intersection."""
    from gsplat_mlx.core.accumulate import accumulate

    H, W, C = 4, 4, 3
    # Gaussian at pixel (2, 2) center = (2.5, 2.5)
    means2d = mx.array([[[2.5, 2.5]]])   # [1, 1, 2]
    conics = mx.array([[[1.0, 0.0, 1.0]]])  # [1, 1, 3] circular
    opacities = mx.array([[0.8]])          # [1, 1]
    colors = mx.array([[[1.0, 0.0, 0.0]]])  # [1, 1, 3] red

    gaussian_ids = mx.array([0], dtype=mx.int32)
    pixel_ids = mx.array([2 * W + 2], dtype=mx.int32)  # pixel (2,2) row-major
    image_ids = mx.array([0], dtype=mx.int32)

    renders, alphas = accumulate(
        means2d, conics, opacities, colors,
        gaussian_ids, pixel_ids, image_ids, W, H
    )

    assert renders.shape == (1, H, W, C)
    assert alphas.shape == (1, H, W, 1)

    # At pixel (2,2): delta = (0, 0), sigma = 0, alpha = 0.8
    # render = 0.8 * [1, 0, 0] = [0.8, 0, 0]
    pixel_color = np.array(renders[0, 2, 2])
    np.testing.assert_allclose(pixel_color, [0.8, 0.0, 0.0], atol=1e-5)

    pixel_alpha = np.array(alphas[0, 2, 2, 0])
    np.testing.assert_allclose(pixel_alpha, 0.8, atol=1e-5)

    # All other pixels should be 0
    renders_np = np.array(renders[0])
    renders_np[2, 2] = 0
    np.testing.assert_allclose(renders_np, 0.0, atol=1e-8)


def test_two_gaussians_front_to_back():
    """Two Gaussians at same pixel, front-to-back compositing."""
    from gsplat_mlx.core.accumulate import accumulate

    H, W, C = 4, 4, 3
    means2d = mx.array([[[2.5, 2.5], [2.5, 2.5]]])  # [1, 2, 2] both at same location
    conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, 2, 3))
    opacities = mx.array([[0.5, 0.5]])  # [1, 2]
    colors = mx.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # red, green

    # Two intersections at pixel (2,2), front Gaussian first
    gaussian_ids = mx.array([0, 1], dtype=mx.int32)
    pixel_ids = mx.array([2 * W + 2, 2 * W + 2], dtype=mx.int32)
    image_ids = mx.array([0, 0], dtype=mx.int32)

    renders, alphas = accumulate(
        means2d, conics, opacities, colors,
        gaussian_ids, pixel_ids, image_ids, W, H
    )

    # delta = (0, 0), sigma = 0 for both
    # alpha_0 = 0.5, alpha_1 = 0.5
    # weight_0 = 1.0 * 0.5 = 0.5
    # weight_1 = (1 - 0.5) * 0.5 = 0.25
    # render = 0.5 * [1,0,0] + 0.25 * [0,1,0] = [0.5, 0.25, 0]
    # alpha = 0.5 + 0.25 = 0.75
    pixel_color = np.array(renders[0, 2, 2])
    np.testing.assert_allclose(pixel_color, [0.5, 0.25, 0.0], atol=1e-4)
    np.testing.assert_allclose(np.array(alphas[0, 2, 2, 0]), 0.75, atol=1e-4)
```

---

#### Cross-Framework Tests

| Test Case | Description | Key Assertions |
|-----------|-------------|---------------|
| `test_cross_framework_accumulate` | Compare MLX `accumulate` against torch `accumulate` (with nerfacc) for identical inputs | Forward output atol=1e-4 |
| `test_cross_framework_rwfa` | Compare MLX `render_weight_from_alpha` against nerfacc's | weights and transmittances atol=1e-5 |
| `test_cross_framework_aar` | Compare MLX `accumulate_along_rays` against nerfacc's | accumulated values atol=1e-5 |

**Cross-framework test setup:**

```python
import pytest
import numpy as np

torch = pytest.importorskip("torch")
nerfacc = pytest.importorskip("nerfacc")

def test_cross_framework_accumulate():
    """Compare MLX accumulate against torch+nerfacc for identical random inputs."""
    from gsplat_mlx.core.accumulate import accumulate as mlx_accumulate
    from gsplat.cuda._torch_impl import accumulate as torch_accumulate

    np.random.seed(42)
    N, M, C = 50, 200, 3
    H, W = 32, 32

    # Generate random inputs as numpy, convert to both frameworks
    means2d_np = np.random.randn(1, N, 2).astype(np.float32) * 10 + 16
    conics_np = np.zeros((1, N, 3), dtype=np.float32)
    conics_np[..., 0] = 1.0  # a = 1 (circular Gaussians)
    conics_np[..., 2] = 1.0  # c = 1
    opacities_np = np.random.uniform(0.1, 0.9, (1, N)).astype(np.float32)
    colors_np = np.random.randn(1, N, C).astype(np.float32)

    # Generate sorted intersection pairs
    gaussian_ids_np = np.random.randint(0, N, M).astype(np.int32)
    pixel_ids_np = np.random.randint(0, H * W, M).astype(np.int32)
    image_ids_np = np.zeros(M, dtype=np.int32)

    # Sort by ray index for front-to-back ordering
    ray_indices = image_ids_np * H * W + pixel_ids_np
    sort_order = np.argsort(ray_indices, kind='stable')
    gaussian_ids_np = gaussian_ids_np[sort_order]
    pixel_ids_np = pixel_ids_np[sort_order]
    image_ids_np = image_ids_np[sort_order]

    # MLX
    import mlx.core as mx
    renders_mlx, alphas_mlx = mlx_accumulate(
        mx.array(means2d_np), mx.array(conics_np),
        mx.array(opacities_np), mx.array(colors_np),
        mx.array(gaussian_ids_np), mx.array(pixel_ids_np),
        mx.array(image_ids_np), W, H,
    )

    # Torch
    renders_torch, alphas_torch = torch_accumulate(
        torch.tensor(means2d_np), torch.tensor(conics_np),
        torch.tensor(opacities_np), torch.tensor(colors_np),
        torch.tensor(gaussian_ids_np.astype(np.int64)),
        torch.tensor(pixel_ids_np.astype(np.int64)),
        torch.tensor(image_ids_np.astype(np.int64)),
        W, H,
    )

    np.testing.assert_allclose(
        np.array(renders_mlx), renders_torch.numpy(), atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        np.array(alphas_mlx), alphas_torch.numpy(), atol=1e-4, rtol=1e-4
    )
```

---

#### Gradient Tests

| Test Case | Description | Key Assertions |
|-----------|-------------|---------------|
| `test_gradient_colors` | Gradient of sum(renders) w.r.t. colors | Non-zero gradients for visible Gaussians, atol=1e-3 vs finite differences |
| `test_gradient_opacities` | Gradient of sum(renders) w.r.t. opacities | Non-zero gradients, correct sign |
| `test_gradient_means2d` | Gradient of sum(renders) w.r.t. means2d | Gradients push means toward/away from pixel centers |
| `test_gradient_finite_diff` | Compare analytic gradient against finite differences | atol=1e-3 for all differentiable inputs |

**Gradient test pattern:**

```python
def test_gradient_colors():
    """Verify gradient of accumulate w.r.t. colors."""
    import mlx.core as mx
    from gsplat_mlx.core.accumulate import accumulate

    N, C, H, W = 5, 3, 8, 8
    means2d = mx.array([[4.0, 4.0], [2.0, 2.0], [6.0, 6.0], [4.0, 6.0], [6.0, 4.0]])
    means2d = means2d[None]  # [1, N, 2]
    conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, N, 3))
    opacities = mx.full((1, N), 0.5)
    colors = mx.ones((1, N, C))

    # Each Gaussian hits its nearest pixel
    gaussian_ids = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
    pixel_ids = mx.array([4*W+4, 2*W+2, 6*W+6, 6*W+4, 4*W+6], dtype=mx.int32)
    image_ids = mx.zeros(5, dtype=mx.int32)

    def loss_fn(c):
        renders, _ = accumulate(means2d, conics, opacities, c,
                                gaussian_ids, pixel_ids, image_ids, W, H)
        return mx.sum(renders)

    grad = mx.grad(loss_fn)(colors)
    # Gradients should be non-zero for the 5 visible Gaussians
    assert mx.any(grad != 0).item(), "Expected non-zero gradients"


def test_gradient_finite_diff():
    """Compare analytic gradient against finite differences for colors."""
    import mlx.core as mx
    from gsplat_mlx.core.accumulate import accumulate

    N, C, H, W = 3, 2, 4, 4
    means2d = mx.array([[[2.5, 2.5], [1.5, 1.5], [3.5, 3.5]]])
    conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, N, 3))
    opacities = mx.array([[0.5, 0.5, 0.5]])
    colors = mx.array([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])

    gaussian_ids = mx.array([0, 1, 2], dtype=mx.int32)
    pixel_ids = mx.array([2*W+2, 1*W+1, 3*W+3], dtype=mx.int32)
    image_ids = mx.zeros(3, dtype=mx.int32)

    def loss_fn(c):
        renders, _ = accumulate(means2d, conics, opacities, c,
                                gaussian_ids, pixel_ids, image_ids, W, H)
        return mx.sum(renders)

    # Analytic gradient
    grad_analytic = mx.grad(loss_fn)(colors)

    # Finite differences
    eps = 1e-4
    colors_np = np.array(colors)
    grad_fd = np.zeros_like(colors_np)
    for i in range(colors_np.shape[1]):
        for j in range(colors_np.shape[2]):
            c_plus = colors_np.copy()
            c_plus[0, i, j] += eps
            c_minus = colors_np.copy()
            c_minus[0, i, j] -= eps
            loss_plus = loss_fn(mx.array(c_plus)).item()
            loss_minus = loss_fn(mx.array(c_minus)).item()
            grad_fd[0, i, j] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(
        np.array(grad_analytic), grad_fd, atol=1e-3, rtol=1e-3
    )
```

---

## Tolerances

| Comparison | `atol` | `rtol` | Rationale |
|------------|--------|--------|-----------|
| Forward (MLX vs analytic) | 1e-4 | 1e-4 | Log-space cumsum introduces small float32 errors |
| Forward (MLX vs torch+nerfacc) | 1e-4 | 1e-4 | Different computation paths (log-space vs direct cumprod) |
| Gradients (analytic vs finite diff) | 1e-3 | 1e-3 | Finite differences inherently less precise |
| `render_weight_from_alpha` (MLX vs nerfacc) | 1e-5 | 1e-5 | Tighter tolerance for this isolated function |
| `accumulate_along_rays` (MLX vs nerfacc) | 1e-5 | 1e-5 | Scatter-add is exact when not racing |

---

## Dependencies

- **PRD-01**: Dev environment, package structure, test infrastructure
- **No dependency on PRD-06 or PRD-07**: This function takes explicit intersection pairs, not tile-based structures. The intersection pairs may come from PRD-07's `rasterize_to_indices_in_range`, but `accumulate` itself is independent.

## Blocks

- **PRD-09** (Rendering API): The high-level rendering API can use `accumulate` as an alternative compositing backend alongside the fused rasterizer from PRD-07.
- **PRD-12** (2DGS): `accumulate_2dgs` follows the same pattern but with different Gaussian evaluation.

---

## Acceptance Criteria

- [ ] `render_weight_from_alpha` produces correct segment-wise exclusive cumprod weights
- [ ] `render_weight_from_alpha` handles multi-ray inputs with no cross-contamination between rays
- [ ] `render_weight_from_alpha` returns transmittance = 1.0 at the start of each ray segment
- [ ] `accumulate_along_rays` correctly scatter-adds weighted values by ray index
- [ ] `accumulate_along_rays` supports both valued (shape `[n_rays, C]`) and None/weight-only (shape `[n_rays, 1]`) modes
- [ ] `accumulate` produces correct rendered images matching upstream torch+nerfacc output (atol=1e-4)
- [ ] All output shapes are correct: renders `[..., H, W, C]`, alphas `[..., H, W, 1]`
- [ ] MAX_ALPHA clamping matches upstream behavior (0.999)
- [ ] Supports arbitrary batch dimensions `[...]` in input shapes
- [ ] Supports arbitrary channel count (C = 1, 3, 16, etc.)
- [ ] Empty intersection inputs (M = 0) produce zero-filled outputs with correct shapes
- [ ] Gradients flow correctly through `mx.grad()` for colors, opacities, means2d, conics
- [ ] Gradient test against finite differences passes at atol=1e-3
- [ ] No dependency on `nerfacc` -- pure MLX implementation
- [ ] All tests pass with `pytest tests/test_accumulate.py -v`
