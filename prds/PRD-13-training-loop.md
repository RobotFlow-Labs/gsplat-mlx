# PRD-13: End-to-End Training Loop (Capstone)

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-13 |
| **Title** | End-to-End Training Loop — Single-Image Gaussian Optimization |
| **Status** | DRAFT |
| **Priority** | P0 — Capstone Validation |
| **Estimated Effort** | 8--12 hours |
| **Dependencies** | PRD-01 through PRD-09 (all core modules), PRD-10 (strategy), PRD-11 (optimizer) |
| **Blocks** | Nothing (this is the capstone) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Build the end-to-end training loop that optimizes a set of 3D Gaussians to reproduce a single target image. This is the **capstone validation** of the entire gsplat-mlx port. If this training loop converges (loss decreases monotonically on average), it proves:

1. **Forward pass** produces correct images (`rasterization()` from PRD-09)
2. **Backward pass** produces correct gradients (`mx.grad()` flows through all operations)
3. **Optimizer** updates parameters correctly (SelectiveAdam from PRD-11)
4. **Densification strategy** manages Gaussian lifecycle (split/clone/prune from PRD-10)
5. **All components** compose without numerical instability

A user should be able to run `python examples/simple_trainer.py` and watch loss converge from ~0.3 to <0.05 on a synthetic scene in under 2 minutes on an M1/M2/M3/M4 Mac.

---

## 2. Context & Motivation

### 2.1 Why a training loop PRD?

Individual module tests (PRD-01 through PRD-09) verify correctness in isolation. But Gaussian Splatting training is a complex pipeline where subtle bugs compound:

- A slightly wrong gradient in covariance computation might not fail a unit test but causes training to diverge after 200 steps.
- Incorrect opacity clamping might look fine in a single forward pass but leads to NaN after sigmoid/logit round-trips.
- Memory layout mismatches between modules might silently produce wrong results.

The training loop is the only test that exercises the full pipeline under gradient descent pressure.

### 2.2 Upstream reference

The upstream `gsplat` project provides `examples/simple_trainer.py` (~1800 lines), a full-featured trainer with multi-view dataset loading, COLMAP integration, viser viewer, and distributed training. Our MLX trainer is a **radically simplified** version:

- Single image (not a multi-view dataset)
- Single camera (not COLMAP poses)
- No viewer, no logging framework
- No distributed training
- Focus on proving convergence, not production quality

### 2.3 MLX vs PyTorch training paradigm

| Aspect | PyTorch (upstream) | MLX (our port) |
|--------|-------------------|----------------|
| Gradient computation | `loss.backward()` | `mx.grad(loss_fn)(...)` |
| Gradient accumulation | Accumulated in `.grad` | Returned as values |
| Zero gradients | `optimizer.zero_grad()` | Not needed (functional) |
| Materialization | Eager by default | Lazy — must call `mx.eval()` |
| Parameter storage | `torch.nn.Parameter` | Plain `mx.array` in a dict |
| Optimizer state | Per-parameter state dict | Separate state arrays |
| Gradient retention | `tensor.retain_grad()` | Captured via `mx.grad` argnums |
| Custom backward | `torch.autograd.Function` | `@mx.custom_function` + `.vjp` |

---

## 3. Scope

### 3.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `src/gsplat_mlx/losses.py` | L1 loss, SSIM loss, combined loss function |
| `src/gsplat_mlx/scenes.py` | Synthetic test scene generators |
| `examples/simple_trainer.py` | Standalone single-image training script |
| `tests/test_training.py` | End-to-end convergence and integration tests |
| `tests/test_losses.py` | Unit tests for loss functions |

### 3.2 Out of Scope

| Item | Reason |
|------|--------|
| Multi-view dataset loading | Not needed for single-image validation |
| COLMAP integration | No external dataset dependency |
| Viser viewer | Visualization is optional |
| Distributed training | Apple Silicon is single-GPU |
| MCMC strategy | DefaultStrategy is sufficient for validation |
| TensorBoard logging | Print-based logging is sufficient |
| Image I/O (PNG/JPG loading) | Use synthetic scenes only |
| Color correction (affine/quadratic) | Post-processing, not core training |
| Learning rate scheduling | Fixed LR is sufficient for convergence proof |
| PLY model loading/saving | No model persistence needed |

---

## 4. Technical Design

### 4.1 File Layout

```
src/gsplat_mlx/
├── losses.py                          # L1, SSIM, combined loss
├── scenes.py                          # Synthetic test scene generators
├── optimizers/                        # (from PRD-11)
│   ├── __init__.py
│   └── selective_adam.py
├── strategy/                          # (from PRD-10)
│   ├── __init__.py
│   ├── base.py
│   ├── default.py
│   └── ops.py
examples/
└── simple_trainer.py                  # Standalone training script
tests/
├── test_training.py                   # End-to-end convergence tests
└── test_losses.py                     # Loss function unit tests
```

---

### 4.2 Loss Functions — `src/gsplat_mlx/losses.py`

#### 4.2.1 L1 Loss

```python
import mlx.core as mx


def l1_loss(rendered: mx.array, target: mx.array) -> mx.array:
    """Mean absolute error between rendered and target images.

    Args:
        rendered: Rendered image [H, W, C] or [B, H, W, C], float32 in [0, 1].
        target: Target image, same shape as rendered, float32 in [0, 1].

    Returns:
        Scalar loss value.
    """
    return mx.mean(mx.abs(rendered - target))
```

#### 4.2.2 SSIM Loss

SSIM (Structural Similarity Index Measure) is critical for perceptual quality. The upstream uses `fused_ssim` from a separate package. We implement a pure MLX version using separable Gaussian convolution.

```python
def _fspecial_gauss_1d(size: int = 11, sigma: float = 1.5) -> mx.array:
    """Create 1D Gaussian kernel.

    Args:
        size: Kernel size (must be odd).
        sigma: Gaussian standard deviation.

    Returns:
        1D Gaussian kernel of shape [size], normalized to sum to 1.
    """
    coords = mx.arange(size, dtype=mx.float32) - size // 2
    g = mx.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / mx.sum(g)
    return g


def _gaussian_filter_2d(img: mx.array, kernel_1d: mx.array) -> mx.array:
    """Apply separable 2D Gaussian filter using two 1D convolutions.

    Uses depthwise convolution via mx.conv2d with groups=C to filter each
    channel independently. Two passes: vertical then horizontal.

    MLX conv2d expects NHWC layout by default:
    - Input: [N, H, W, C_in]
    - Weight: [C_out, kH, kW, C_in / groups]

    Args:
        img: Input image [B, H, W, C] (MLX NHWC format).
        kernel_1d: 1D Gaussian kernel [K].

    Returns:
        Filtered image, same spatial size as input (edge-padded).
    """
    C = img.shape[-1]
    K = kernel_1d.shape[0]
    pad = K // 2

    # --- Vertical pass (filter along H dimension) ---
    # Pad height only
    img_padded = mx.pad(img, [(0, 0), (pad, pad), (0, 0), (0, 0)], mode="edge")

    # Depthwise conv: weight shape [C_out, kH, kW, C_in/groups]
    # For depthwise: groups=C, C_out=C, C_in/groups=1
    w_v = kernel_1d.reshape(1, K, 1, 1)   # [1, K, 1, 1]
    w_v = mx.tile(w_v, (C, 1, 1, 1))      # [C, K, 1, 1]
    out = mx.conv2d(img_padded, w_v, stride=1, padding=0, groups=C)

    # --- Horizontal pass (filter along W dimension) ---
    out_padded = mx.pad(out, [(0, 0), (0, 0), (pad, pad), (0, 0)], mode="edge")
    w_h = kernel_1d.reshape(1, 1, K, 1)   # [1, 1, K, 1]
    w_h = mx.tile(w_h, (C, 1, 1, 1))      # [C, 1, K, 1]
    out = mx.conv2d(out_padded, w_h, stride=1, padding=0, groups=C)

    return out


def ssim(
    img1: mx.array,
    img2: mx.array,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    reduction: str = "mean",
) -> mx.array:
    """Compute Structural Similarity Index (SSIM).

    Implements the SSIM formula from Wang et al. 2004:

        SSIM(x, y) = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                     -------------------------------------------
                     (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2)

    Where mu and sigma are computed via Gaussian-weighted local statistics,
    and C1, C2 are stability constants.

    Args:
        img1: First image [H, W, C] or [B, H, W, C], float32.
        img2: Second image, same shape as img1.
        window_size: Size of the Gaussian window (default 11).
        sigma: Standard deviation of the Gaussian window (default 1.5).
        max_val: Dynamic range of the images (default 1.0).
        reduction: "mean" returns scalar, "none" returns per-pixel SSIM map.

    Returns:
        SSIM value (scalar if reduction="mean", else [B, H, W, C] map).
        Value range: [-1, 1] where 1.0 = identical images.
    """
    # Ensure 4D: [B, H, W, C]
    squeeze = False
    if img1.ndim == 3:
        img1 = img1[None]
        img2 = img2[None]
        squeeze = True

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    kernel = _fspecial_gauss_1d(window_size, sigma)

    # Local means
    mu1 = _gaussian_filter_2d(img1, kernel)
    mu2 = _gaussian_filter_2d(img2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = _gaussian_filter_2d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = _gaussian_filter_2d(img2 * img2, kernel) - mu2_sq
    sigma12 = _gaussian_filter_2d(img1 * img2, kernel) - mu1_mu2

    # Clamp variances to avoid negative values from numerical imprecision
    sigma1_sq = mx.maximum(sigma1_sq, mx.array(0.0))
    sigma2_sq = mx.maximum(sigma2_sq, mx.array(0.0))

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    if squeeze:
        ssim_map = ssim_map[0]

    if reduction == "mean":
        return mx.mean(ssim_map)
    return ssim_map


def ssim_loss(rendered: mx.array, target: mx.array, **kwargs) -> mx.array:
    """SSIM loss: 1 - SSIM.

    Returns a value in [0, 2] where 0 means perfect structural similarity.

    Args:
        rendered: Rendered image [H, W, C] or [B, H, W, C].
        target: Target image, same shape.

    Returns:
        Scalar SSIM loss (1 - SSIM).
    """
    return 1.0 - ssim(rendered, target, **kwargs)


def combined_loss(
    rendered: mx.array,
    target: mx.array,
    lambda_ssim: float = 0.2,
) -> mx.array:
    """Combined L1 + SSIM loss as used in the original 3DGS paper.

        loss = (1 - lambda_ssim) * L1 + lambda_ssim * (1 - SSIM)

    This is the standard loss function from Kerbl et al. 2023. The L1 term
    drives pixel-level accuracy while the SSIM term improves perceptual
    quality and structural coherence.

    Args:
        rendered: Rendered image [H, W, C] or [B, H, W, C].
        target: Target image, same shape.
        lambda_ssim: Weight for SSIM loss term (default 0.2, per 3DGS paper).

    Returns:
        Scalar combined loss.
    """
    loss_l1 = l1_loss(rendered, target)
    loss_ssim = ssim_loss(rendered, target)
    return (1.0 - lambda_ssim) * loss_l1 + lambda_ssim * loss_ssim
```

---

### 4.3 Synthetic Test Scenes — `src/gsplat_mlx/scenes.py`

Four synthetic scenes of increasing difficulty for validation. All return a tuple of (target image, camera dict) with consistent conventions.

```python
"""Synthetic test scene generators for training loop validation.

Each generator returns:
    target: Target image [1, H, W, 3] float32 in [0, 1].
    camera: Dict with:
        - viewmat: [1, 4, 4] world-to-camera matrix
        - K: [1, 3, 3] camera intrinsic matrix
        - width: int
        - height: int

Camera convention: camera at z=+3 looking toward origin (negative z direction
in camera frame), matching the standard OpenGL/3DGS convention where objects
are at positive z in world space and the camera looks down -z in its local frame.
"""

import mlx.core as mx
from typing import Dict, Tuple


def make_solid_color_scene(
    width: int = 256,
    height: int = 256,
    color: Tuple[float, float, float] = (0.8, 0.2, 0.1),
) -> Tuple[mx.array, Dict]:
    """Create a solid-color target image and matching camera.

    The simplest possible scene -- a uniform color rectangle. Any set of
    Gaussians that covers the image with the right color will converge.
    This is the "hello world" of 3DGS training: if this doesn't converge,
    something is fundamentally broken.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB color tuple in [0, 1].

    Returns:
        target: Target image [1, H, W, 3].
        camera: Camera parameter dict.
    """
    target = mx.ones((1, height, width, 3), dtype=mx.float32) * mx.array(
        color, dtype=mx.float32
    ).reshape(1, 1, 1, 3)

    fx = fy = float(width)  # focal length in pixels
    cx, cy = width / 2.0, height / 2.0
    K = mx.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=mx.float32)

    # Camera at z=+3, looking toward origin
    viewmat = mx.array([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]], dtype=mx.float32)

    camera = {"viewmat": viewmat, "K": K, "width": width, "height": height}
    return target, camera


def make_gradient_scene(
    width: int = 256,
    height: int = 256,
    direction: str = "horizontal",
) -> Tuple[mx.array, Dict]:
    """Create a smooth gradient target image.

    A gradient from black to white. Tests the model's ability to represent
    smooth spatial variation in color. Requires Gaussians to tile the image
    with smoothly varying colors -- harder than solid but still tractable.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        direction: "horizontal" (left-to-right) or "vertical" (top-to-bottom).

    Returns:
        target: Target image [1, H, W, 3].
        camera: Camera parameter dict.
    """
    if direction == "horizontal":
        grad = mx.linspace(0, 1, width).reshape(1, 1, width, 1)
        grad = mx.broadcast_to(grad, (1, height, width, 1))
    else:
        grad = mx.linspace(0, 1, height).reshape(1, height, 1, 1)
        grad = mx.broadcast_to(grad, (1, height, width, 1))

    target = mx.broadcast_to(grad, (1, height, width, 3)).astype(mx.float32)

    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=mx.float32)

    viewmat = mx.array([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]], dtype=mx.float32)

    camera = {"viewmat": viewmat, "K": K, "width": width, "height": height}
    return target, camera


def make_checkerboard_scene(
    width: int = 256,
    height: int = 256,
    squares: int = 8,
    color1: Tuple[float, float, float] = (0.9, 0.9, 0.9),
    color2: Tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> Tuple[mx.array, Dict]:
    """Create a checkerboard target image.

    The hardest synthetic test. A checkerboard has sharp edges and high-
    frequency content, requiring many well-placed Gaussians. Convergence
    here proves the model has sufficient capacity and the optimization
    landscape is navigable.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        squares: Number of squares per row/column.
        color1, color2: Alternating colors.

    Returns:
        target: Target image [1, H, W, 3].
        camera: Camera parameter dict.
    """
    rows = mx.arange(height).reshape(height, 1) // (height // squares)
    cols = mx.arange(width).reshape(1, width) // (width // squares)
    checker = ((rows + cols) % 2).astype(mx.float32)  # [H, W]

    c1 = mx.array(color1, dtype=mx.float32).reshape(1, 1, 3)
    c2 = mx.array(color2, dtype=mx.float32).reshape(1, 1, 3)
    target = checker[..., None] * c1 + (1.0 - checker[..., None]) * c2  # [H, W, 3]
    target = target[None]  # [1, H, W, 3]

    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=mx.float32)

    viewmat = mx.array([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]], dtype=mx.float32)

    camera = {"viewmat": viewmat, "K": K, "width": width, "height": height}
    return target, camera


def make_colored_circles_scene(
    width: int = 256,
    height: int = 256,
    n_circles: int = 5,
    seed: int = 42,
) -> Tuple[mx.array, Dict]:
    """Create a scene with soft colored circles on a dark background.

    Colored circles with Gaussian falloff are natural targets for 3DGS:
    each circle looks like a single Gaussian splat. This tests both
    color and spatial reconstruction with a scene that is "Gaussian-friendly."

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        n_circles: Number of circles to generate.
        seed: Random seed for reproducibility.

    Returns:
        target: Target image [1, H, W, 3].
        camera: Camera parameter dict.
    """
    mx.random.seed(seed)

    target = mx.zeros((height, width, 3), dtype=mx.float32)

    # Coordinate grids
    yy = mx.arange(height, dtype=mx.float32).reshape(height, 1)
    xx = mx.arange(width, dtype=mx.float32).reshape(1, width)

    for _ in range(n_circles):
        cx = mx.random.uniform(low=0.2, high=0.8).item() * width
        cy = mx.random.uniform(low=0.2, high=0.8).item() * height
        r = mx.random.uniform(low=0.05, high=0.15).item() * min(width, height)
        color = mx.random.uniform(low=0.3, high=1.0, shape=(3,))

        # Soft circle (Gaussian falloff)
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        alpha = mx.exp(-dist_sq / (2 * r * r))  # [H, W]
        target = target + alpha[..., None] * color.reshape(1, 1, 3)

    target = mx.clip(target, 0.0, 1.0)[None]  # [1, H, W, 3]

    fx = fy = float(width)
    cx_cam, cy_cam = width / 2.0, height / 2.0
    K = mx.array([[[fx, 0, cx_cam], [0, fy, cy_cam], [0, 0, 1]]], dtype=mx.float32)

    viewmat = mx.array([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]], dtype=mx.float32)

    camera = {"viewmat": viewmat, "K": K, "width": width, "height": height}
    return target, camera


def make_synthetic_gaussians_scene(
    width: int = 128,
    height: int = 128,
) -> Tuple[mx.array, Dict, Dict[str, mx.array]]:
    """Create a synthetic target by rendering known Gaussians.

    This is the ultimate roundtrip test: render a known set of Gaussians,
    then try to recover similar parameters from scratch. Returns the
    ground truth parameters along with the target image.

    Uses 5 Gaussians with distinct colors at known positions:
    - Center: red
    - Right: green
    - Left: blue
    - Top: yellow
    - Bottom: magenta

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        target: Target image [1, H, W, 3].
        camera: Camera parameter dict.
        gt_params: Ground truth parameter dict.
    """
    import numpy as np
    from gsplat_mlx.rendering import rasterization

    means = mx.array([
        [0.0, 0.0, 0.0],     # center (will be at z=3 from camera)
        [0.8, 0.0, 0.5],     # right, slightly farther
        [-0.8, 0.0, 0.3],    # left
        [0.0, 0.8, 0.0],     # top
        [0.0, -0.8, 0.0],    # bottom
    ], dtype=mx.float32)

    quats = mx.broadcast_to(
        mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32), (5, 4)
    ).copy()

    scales = mx.full((5, 3), float(np.log(0.3)), dtype=mx.float32)
    opacities = mx.full((5,), 2.0, dtype=mx.float32)  # sigmoid(2) ~ 0.88

    sh_coeffs = mx.array([
        [[1.0, 0.0, 0.0]],   # red
        [[0.0, 1.0, 0.0]],   # green
        [[0.0, 0.0, 1.0]],   # blue
        [[1.0, 1.0, 0.0]],   # yellow
        [[1.0, 0.0, 1.0]],   # magenta
    ], dtype=mx.float32)

    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=mx.float32)

    viewmat = mx.array([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]], dtype=mx.float32)

    target, alpha, _ = rasterization(
        means, quats, scales, mx.sigmoid(opacities), sh_coeffs,
        viewmat, K, width, height, sh_degree=0,
    )

    camera = {"viewmat": viewmat, "K": K, "width": width, "height": height}
    gt_params = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }

    return target, camera, gt_params
```

---

### 4.4 Training Loop Architecture

#### 4.4.1 Gaussian Parameter Initialization

Parameters are stored in their **optimizable form** using standard reparameterizations:

| Parameter | Stored As | Activated By | Actual Range |
|-----------|-----------|-------------|-------------|
| `means` | raw positions [N, 3] | identity | (-inf, +inf) |
| `scales` | log-space [N, 3] | `mx.exp(scales)` | (0, +inf) |
| `quats` | unnormalized [N, 4] | normalize to unit quaternion | unit sphere |
| `opacities` | logit-space [N] | `mx.sigmoid(opacities)` | (0, 1) |
| `sh_coeffs` | raw coefficients [N, K, 3] | identity (SH evaluation) | (-inf, +inf) |

```python
def init_gaussians(
    n: int,
    scene_scale: float = 1.0,
    sh_degree: int = 0,
    seed: int = 42,
) -> Dict[str, mx.array]:
    """Initialize N random Gaussians.

    Initialization strategy:
    - Positions: uniform random in [-scene_scale, scene_scale]^3.
      These will be in front of the camera (camera at z=+3 looking toward origin).
    - Scales: log(0.05 * scene_scale). Small initial Gaussians that grow as needed.
    - Quaternions: identity rotation [1, 0, 0, 0]. All Gaussians start axis-aligned.
    - Opacities: logit(-2.0) -> sigmoid(-2) ~ 0.12. Low initial opacity so
      Gaussians must earn their contribution via gradient descent.
    - SH coefficients: small random normal. DC term will represent base color.

    Args:
        n: Number of Gaussians.
        scene_scale: Scale of the scene (affects position and scale initialization).
        sh_degree: SH degree (0 = constant color, 1 = linear, 2 = quadratic, 3 = cubic).
        seed: Random seed for reproducibility.

    Returns:
        Dict of parameter arrays keyed by name.
    """
    mx.random.seed(seed)
    K = (sh_degree + 1) ** 2  # number of SH coefficients per channel

    params = {
        "means": mx.random.uniform(
            low=-scene_scale, high=scene_scale, shape=(n, 3)
        ),
        "scales": mx.ones((n, 3), dtype=mx.float32) * mx.log(
            mx.array(0.05 * scene_scale)
        ),
        "quats": mx.concatenate([
            mx.ones((n, 1), dtype=mx.float32),
            mx.zeros((n, 3), dtype=mx.float32),
        ], axis=-1),
        "opacities": mx.ones(n, dtype=mx.float32) * (-2.0),
        "sh_coeffs": mx.random.normal((n, K, 3)) * 0.1,
    }

    return params
```

#### 4.4.2 Per-Parameter Learning Rates

Following the 3DGS paper (Kerbl et al. 2023), different parameters use different learning rates. This is critical for stable convergence because the gradient scales differ wildly between position updates (small steps in 3D) and opacity updates (large logit-space changes).

| Parameter | Learning Rate | Rationale |
|-----------|--------------|-----------|
| `means` | 1.6e-4 | Positions need small updates to avoid oscillation |
| `scales` | 5e-3 | Log-space scales are less sensitive |
| `quats` | 1e-3 | Rotations are moderately sensitive |
| `opacities` | 5e-2 | Logit-space opacities need aggressive updates to become visible/invisible |
| `sh_coeffs` | 2.5e-3 | SH coefficients are moderately sensitive |

```python
# Learning rate configuration
LR_CONFIG = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh_coeffs": 2.5e-3,
}
```

#### 4.4.3 Loss Function Closure

The core challenge in MLX is that `mx.grad` requires a pure function `f(params) -> scalar`. But we also need the `info` dict from `rasterization()` for densification. We handle this with two approaches:

**Approach A: `mx.value_and_grad` with auxiliary outputs** (if MLX supports it via tuple returns):

```python
def _train_step(params, target, K, viewmat, W, H, sh_degree, tile_size):
    """Single training step: forward + loss + gradients.

    Uses mx.value_and_grad to compute loss and gradients simultaneously.
    The loss function returns (loss, info) as a tuple so we can capture
    both the scalar loss for differentiation and the info dict for
    densification decisions.
    """
    from gsplat_mlx.rendering import rasterization

    def loss_fn(means, quats, scales, opacities, sh_coeffs):
        rendered, alpha, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=mx.sigmoid(opacities),
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            tile_size=tile_size,
            render_mode="RGB",
            rasterize_mode="classic",
        )

        # L1 loss
        loss_l1 = mx.mean(mx.abs(rendered - target))

        # Return loss as first element, info as auxiliary
        return loss_l1, info

    # Compute loss, info, and gradients for all 5 parameter groups
    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    (loss, info), grads_tuple = grad_fn(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["sh_coeffs"],
    )

    grads = {
        "means": grads_tuple[0],
        "quats": grads_tuple[1],
        "scales": grads_tuple[2],
        "opacities": grads_tuple[3],
        "sh_coeffs": grads_tuple[4],
    }

    return loss, grads, info
```

**Approach B: Separate forward pass for info** (fallback if Approach A has issues):

```python
def _train_step_separate(params, target, K, viewmat, W, H, sh_degree, tile_size):
    """Training step with separate forward pass for info dict.

    Cleaner but ~50% more compute. Use this if mx.value_and_grad
    doesn't handle auxiliary outputs correctly.
    """
    from gsplat_mlx.rendering import rasterization

    def loss_fn(means, quats, scales, opacities, sh_coeffs):
        rendered, alpha, info = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=mx.sigmoid(opacities), colors=sh_coeffs,
            viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=sh_degree, tile_size=tile_size,
        )
        return mx.mean(mx.abs(rendered - target))

    # Gradients
    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    loss, grads_tuple = grad_fn(
        params["means"], params["quats"], params["scales"],
        params["opacities"], params["sh_coeffs"],
    )

    # Separate forward for info dict (no gradient tracking)
    rendered, alpha, info = rasterization(
        params["means"], params["quats"], params["scales"],
        mx.sigmoid(params["opacities"]), params["sh_coeffs"],
        viewmat, K, W, H, sh_degree=sh_degree, tile_size=tile_size,
    )

    grads = dict(zip(
        ["means", "quats", "scales", "opacities", "sh_coeffs"],
        grads_tuple,
    ))

    return loss, grads, info
```

---

### 4.5 Complete Training Loop — `examples/simple_trainer.py`

```python
#!/usr/bin/env python3
"""gsplat-mlx Simple Trainer -- Single-Image Gaussian Optimization.

Optimizes a set of 3D Gaussians to reproduce a single target image.
This is the capstone validation for the gsplat-mlx port.

Usage:
    python examples/simple_trainer.py --scene gradient --n_gaussians 2000 --steps 1000
    python examples/simple_trainer.py --scene checkerboard --n_gaussians 5000 --steps 2000
    python examples/simple_trainer.py --scene circles --n_gaussians 3000 --steps 1500

Expected output:
    Step     0 | loss=0.312456 | N= 2000 | time=0.1s
    Step    50 | loss=0.198234 | N= 2000 | time=2.3s
    Step   100 | loss=0.142567 | N= 2000 | time=4.5s
    ...
    Step   950 | loss=0.023456 | N= 2000 | time=45.2s
    ============================================================
    Training complete.
      Final loss:       0.023456
      Initial loss:     0.312456
      Loss reduction:   92.5%
      Final Gaussians:  2000
      Total time:       45.2s
    ============================================================
"""

import argparse
import time
from typing import Dict, Tuple

import mlx.core as mx

from gsplat_mlx.rendering import rasterization
from gsplat_mlx.losses import l1_loss, ssim_loss, combined_loss
from gsplat_mlx.optimizers.selective_adam import SelectiveAdam
from gsplat_mlx.scenes import (
    make_solid_color_scene,
    make_gradient_scene,
    make_checkerboard_scene,
    make_colored_circles_scene,
)


# ---------------------------------------------------------------------------
# Learning rate configuration (per 3DGS paper)
# ---------------------------------------------------------------------------

LR_CONFIG = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh_coeffs": 2.5e-3,
}


# ---------------------------------------------------------------------------
# Gaussian parameter initialization
# ---------------------------------------------------------------------------

def init_gaussians(
    n: int,
    scene_scale: float = 1.0,
    sh_degree: int = 0,
    seed: int = 42,
) -> Dict[str, mx.array]:
    """Initialize N random Gaussians.

    See Section 4.4.1 of PRD-13 for detailed documentation.
    """
    mx.random.seed(seed)
    K = (sh_degree + 1) ** 2

    params = {
        "means": mx.random.uniform(
            low=-scene_scale, high=scene_scale, shape=(n, 3)
        ),
        "scales": mx.ones((n, 3), dtype=mx.float32) * mx.log(
            mx.array(0.05 * scene_scale)
        ),
        "quats": mx.concatenate([
            mx.ones((n, 1), dtype=mx.float32),
            mx.zeros((n, 3), dtype=mx.float32),
        ], axis=-1),
        "opacities": mx.ones(n, dtype=mx.float32) * (-2.0),
        "sh_coeffs": mx.random.normal((n, K, 3)) * 0.1,
    }

    return params


# ---------------------------------------------------------------------------
# Loss function (creates differentiable closure)
# ---------------------------------------------------------------------------

def make_loss_fn(
    target: mx.array,
    viewmat: mx.array,
    K: mx.array,
    width: int,
    height: int,
    sh_degree: int = 0,
    lambda_ssim: float = 0.2,
    use_ssim: bool = True,
):
    """Create a closure that computes the training loss.

    Returns a function suitable for mx.grad() / mx.value_and_grad().
    The function takes the 5 trainable parameter arrays and returns
    a scalar loss.

    Args:
        target: Target image [C, H, W, 3].
        viewmat: View matrix [C, 4, 4].
        K: Intrinsic matrix [C, 3, 3].
        width, height: Image dimensions.
        sh_degree: SH degree for color evaluation.
        lambda_ssim: Weight for SSIM loss (default 0.2).
        use_ssim: Whether to include SSIM in the loss.

    Returns:
        Callable (means, quats, scales, opacities, sh_coeffs) -> scalar loss.
    """

    def loss_fn(means, quats, scales, opacities, sh_coeffs):
        rendered, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=mx.sigmoid(opacities),
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            rasterize_mode="classic",
        )

        if use_ssim:
            loss = combined_loss(rendered, target, lambda_ssim=lambda_ssim)
        else:
            loss = l1_loss(rendered, target)

        return loss

    return loss_fn


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    params: Dict[str, mx.array],
    target: mx.array,
    viewmat: mx.array,
    K: mx.array,
    width: int,
    height: int,
    sh_degree: int = 0,
    lambda_ssim: float = 0.2,
    use_ssim: bool = True,
) -> Tuple[mx.array, Dict[str, mx.array], Dict]:
    """Execute a single training step: forward + loss + backward.

    Uses mx.value_and_grad with auxiliary outputs to capture both the
    scalar loss (for differentiation) and the info dict (for densification).

    Args:
        params: Current parameter dict.
        target: Target image [1, H, W, 3].
        viewmat: View matrix [1, 4, 4].
        K: Intrinsic matrix [1, 3, 3].
        width, height: Image dimensions.
        sh_degree: SH degree.
        lambda_ssim: SSIM loss weight.
        use_ssim: Whether to use SSIM loss.

    Returns:
        loss: Scalar loss value.
        grads: Dict of gradient arrays keyed by parameter name.
        info: Rasterization info dict for densification.
    """

    def loss_fn(means, quats, scales, opacities, sh_coeffs):
        rendered, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=mx.sigmoid(opacities),
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            rasterize_mode="classic",
        )

        if use_ssim:
            loss = combined_loss(rendered, target, lambda_ssim=lambda_ssim)
        else:
            loss = l1_loss(rendered, target)

        return loss, info

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    (loss, info), grads_tuple = grad_fn(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["sh_coeffs"],
    )

    grads = {
        "means": grads_tuple[0],
        "quats": grads_tuple[1],
        "scales": grads_tuple[2],
        "opacities": grads_tuple[3],
        "sh_coeffs": grads_tuple[4],
    }

    return loss, grads, info


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    scene: str = "gradient",
    n_gaussians: int = 2000,
    steps: int = 1000,
    sh_degree: int = 0,
    width: int = 256,
    height: int = 256,
    lambda_ssim: float = 0.2,
    use_ssim: bool = True,
    densify: bool = False,
    log_every: int = 50,
    seed: int = 42,
) -> Dict[str, list]:
    """Run the single-image training loop.

    This is the main entry point. It:
    1. Generates a synthetic target image
    2. Initializes random Gaussians
    3. Optimizes via gradient descent (SelectiveAdam)
    4. Optionally densifies (split/clone/prune) via DefaultStrategy
    5. Logs loss and Gaussian count at regular intervals

    Args:
        scene: Scene type ("solid", "gradient", "checkerboard", "circles").
        n_gaussians: Initial number of Gaussians.
        steps: Number of training steps.
        sh_degree: SH degree for color (0 = constant, 3 = full).
        width, height: Image dimensions.
        lambda_ssim: SSIM loss weight (0 = pure L1).
        use_ssim: Whether to use SSIM loss at all.
        densify: Whether to use densification strategy.
        log_every: Print loss every N steps (0 = silent).
        seed: Random seed.

    Returns:
        Dict with training history:
        - "losses": list of float loss values, one per step
        - "n_gaussians": list of int Gaussian counts, one per step
        - "times": list of float elapsed times in seconds, one per step
    """

    # ---------- Scene setup ----------
    scene_generators = {
        "solid": make_solid_color_scene,
        "gradient": make_gradient_scene,
        "checkerboard": make_checkerboard_scene,
        "circles": make_colored_circles_scene,
    }

    assert scene in scene_generators, (
        f"Unknown scene '{scene}'. Choose from {list(scene_generators.keys())}"
    )

    target, camera = scene_generators[scene](width=width, height=height)
    mx.eval(target)

    if log_every > 0:
        print(f"Scene: {scene}")
        print(f"Target shape: {target.shape}")
        print(f"Image size: {width}x{height}")
        print(f"Initial Gaussians: {n_gaussians}")
        print(f"SH degree: {sh_degree}")
        print(f"Steps: {steps}")
        print(f"SSIM: {'ON (lambda={lambda_ssim})' if use_ssim else 'OFF'}")
        print(f"Densification: {'ON' if densify else 'OFF'}")
        print()

    # ---------- Initialize Gaussians ----------
    params = init_gaussians(
        n=n_gaussians,
        scene_scale=1.0,
        sh_degree=sh_degree,
        seed=seed,
    )
    mx.eval(*params.values())

    # ---------- Initialize optimizers (one per parameter) ----------
    optimizers = {}
    for name, lr in LR_CONFIG.items():
        opt = SelectiveAdam(lr=lr, betas=(0.9, 0.999), eps=1e-15)
        opt.init_param(name, params[name])
        optimizers[name] = opt

    # ---------- Initialize densification strategy ----------
    if densify:
        from gsplat_mlx.strategy.default import DefaultStrategy
        strategy = DefaultStrategy(
            refine_start_iter=200,
            refine_stop_iter=steps,
            refine_every=100,
            reset_every=1000,
            verbose=(log_every > 0),
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=1.0)
    else:
        strategy = None
        strategy_state = None

    # ---------- Training loop ----------
    history = {"losses": [], "n_gaussians": [], "times": []}
    t0 = time.time()

    for step in range(steps):
        # --- Forward + backward ---
        loss, grads, info = train_step(
            params=params,
            target=target,
            viewmat=camera["viewmat"],
            K=camera["K"],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            lambda_ssim=lambda_ssim,
            use_ssim=use_ssim,
        )

        # Materialize gradients (MLX is lazy)
        mx.eval(loss, *grads.values())

        loss_val = loss.item()
        history["losses"].append(loss_val)
        history["n_gaussians"].append(params["means"].shape[0])
        history["times"].append(time.time() - t0)

        # --- NaN guard ---
        if loss_val != loss_val:  # NaN check (NaN != NaN)
            print(f"ERROR: NaN loss at step {step}. Aborting.")
            break

        # --- Densification strategy ---
        if strategy is not None:
            strategy.step_pre_backward(
                params, optimizers, strategy_state, step, info
            )
            # Note: grad_means2d is None here; simplified densification
            # uses opacity-based pruning and radii-based decisions only.
            strategy.step_post_backward(
                params, optimizers, strategy_state, step, info,
                grad_means2d=None,
            )

        # --- Compute visibility mask for selective optimizer ---
        if "radii" in info:
            radii = info["radii"]  # [C, N, 2]
            # Visible = radii > 0 in both dimensions for at least one camera
            visibility = mx.any(mx.all(radii > 0, axis=-1), axis=0)  # [N]
            mx.eval(visibility)
        else:
            visibility = None

        # --- Optimizer step (per-parameter) ---
        for name in params:
            params[name] = optimizers[name].step(
                name, params[name], grads[name], visibility
            )

        # Materialize updated parameters
        mx.eval(*params.values())

        # --- Logging ---
        if log_every > 0 and (step % log_every == 0 or step == steps - 1):
            elapsed = time.time() - t0
            n = params["means"].shape[0]
            print(
                f"Step {step:5d} | loss={loss_val:.6f} | "
                f"N={n:5d} | time={elapsed:.1f}s"
            )

    # ---------- Summary ----------
    if log_every > 0 and len(history["losses"]) > 0:
        print()
        print("=" * 60)
        print("Training complete.")
        print(f"  Final loss:       {history['losses'][-1]:.6f}")
        print(f"  Initial loss:     {history['losses'][0]:.6f}")
        reduction = (1 - history['losses'][-1] / history['losses'][0]) * 100
        print(f"  Loss reduction:   {reduction:.1f}%")
        print(f"  Final Gaussians:  {history['n_gaussians'][-1]}")
        print(f"  Total time:       {history['times'][-1]:.1f}s")
        print("=" * 60)

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="gsplat-mlx single-image trainer (capstone validation)"
    )
    parser.add_argument(
        "--scene", type=str, default="gradient",
        choices=["solid", "gradient", "checkerboard", "circles"],
        help="Synthetic scene type (default: gradient)",
    )
    parser.add_argument(
        "--n_gaussians", type=int, default=2000,
        help="Initial number of Gaussians (default: 2000)",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of training steps (default: 1000)",
    )
    parser.add_argument(
        "--sh_degree", type=int, default=0,
        help="SH degree for color (default: 0 = constant color)",
    )
    parser.add_argument(
        "--width", type=int, default=256,
        help="Image width (default: 256)",
    )
    parser.add_argument(
        "--height", type=int, default=256,
        help="Image height (default: 256)",
    )
    parser.add_argument(
        "--lambda_ssim", type=float, default=0.2,
        help="SSIM loss weight (default: 0.2)",
    )
    parser.add_argument(
        "--no_ssim", action="store_true",
        help="Disable SSIM loss (use L1 only)",
    )
    parser.add_argument(
        "--densify", action="store_true",
        help="Enable densification strategy (split/clone/prune)",
    )
    parser.add_argument(
        "--log_every", type=int, default=50,
        help="Print loss every N steps (default: 50, 0 = silent)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    train(
        scene=args.scene,
        n_gaussians=args.n_gaussians,
        steps=args.steps,
        sh_degree=args.sh_degree,
        width=args.width,
        height=args.height,
        lambda_ssim=args.lambda_ssim,
        use_ssim=not args.no_ssim,
        densify=args.densify,
        log_every=args.log_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

---

### 4.6 MLX-Specific Training Patterns

#### 4.6.1 Functional Gradients vs Imperative Gradients

The most fundamental difference from PyTorch:

```python
# ---- PyTorch (upstream) ----
optimizer.zero_grad()
loss = compute_loss(params)
loss.backward()                    # Gradients stored in param.grad
optimizer.step()                   # Uses param.grad internally

# ---- MLX (our port) ----
loss_fn = lambda means, ...: compute_loss(means, ...)
grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
(loss, aux), grads = grad_fn(means, ...)  # Gradients returned as values
mx.eval(loss, *grads)                     # Materialize lazy computation
params["means"] = optimizer.step("means", means, grads[0], visibility)
# No zero_grad needed -- each call returns fresh gradients
```

Key implications:
- **No gradient accumulation bugs**: Each `grad_fn` call returns independent gradients.
- **No `.retain_grad()` needed**: All intermediate gradients are available via `argnums`.
- **Must explicitly pass gradients to optimizer**: No implicit `.grad` attribute.

#### 4.6.2 Lazy Evaluation Discipline

MLX is lazy: operations build a computation graph and only execute when `mx.eval()` is called. Critical eval points in the training loop:

```python
# MUST eval after gradient computation (before using .item() or reading values)
mx.eval(loss, *grads.values())

# MUST eval after optimizer step (before next forward pass uses updated params)
mx.eval(*params.values())

# SHOULD eval after info dict computation (before strategy reads radii/means2d)
mx.eval(visibility)

# DON'T eval unnecessarily inside the forward pass -- let the graph build up
# for maximum fusion opportunity
```

**Anti-pattern** (causes excessive kernel launches):
```python
# BAD: eval inside the forward pass
rendered = rasterization(...)
mx.eval(rendered)  # unnecessary -- let it fuse with the loss computation
loss = l1_loss(rendered, target)
mx.eval(loss)  # now eval
```

**Correct pattern** (one eval per logical phase):
```python
# GOOD: let the graph build, eval once
(loss, info), grads = grad_fn(params...)
mx.eval(loss, *grads.values())  # single eval materializes everything
```

#### 4.6.3 Handling `info` Dict Through `mx.grad`

`mx.grad` only differentiates a function that returns a scalar. But we need the `info` dict from `rasterization()` for densification. The solution uses `mx.value_and_grad` which supports auxiliary outputs when the loss function returns a tuple `(loss, aux)`:

```python
def loss_fn(means, quats, scales, opacities, sh_coeffs):
    rendered, alpha, info = rasterization(...)
    loss = l1_loss(rendered, target)
    return loss, info   # <-- tuple: (scalar, auxiliary)

# mx.value_and_grad differentiates w.r.t. the first element of the tuple
grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
(loss, info), grads = grad_fn(means, quats, scales, opacities, sh_coeffs)
```

If `mx.value_and_grad` does not support tuple returns, fall back to a separate forward pass:

```python
# Fallback: separate forward pass for info
loss, grads = loss_and_grad_fn(params...)         # get loss + gradients
rendered, alphas, info = rasterize_fn(params...)  # get info dict (extra cost)
```

#### 4.6.4 Gradient for `means2d` (Densification)

The upstream strategy needs `d(loss)/d(means2d)` for the grow/split decision. In PyTorch, this is captured via `means2d.retain_grad()`. In MLX, `means2d` is an intermediate variable inside `rasterization()`, not a direct argument to `loss_fn`.

**Solution for capstone**: Use a simplified densification approach that does not require `means2d` gradients:

1. **Opacity-based pruning** works without gradients (just check `sigmoid(opacities) < threshold`).
2. **Scale-based splitting** works without gradients (check `exp(scales) > threshold`).
3. **Gradient-based growing** (the full upstream approach) is deferred to post-capstone refinement.

To implement full gradient-based densification later, the approach would be:

```python
# Make means2d an explicit input to the loss function
def loss_fn_with_means2d(means, quats, scales, opacities, sh_coeffs, means2d_override):
    # ... rasterization that uses means2d_override instead of computing it ...
    pass

# Include means2d in argnums to get its gradient
grad_fn = mx.value_and_grad(loss_fn_with_means2d, argnums=(0, 1, 2, 3, 4, 5))
```

---

## 5. Data Flow Diagram

```
                    init_gaussians(n=2000, seed=42)
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │  params: Dict[str, mx.array]          │
           │  ├── means       [N, 3]   float32     │
           │  ├── quats       [N, 4]   float32     │
           │  ├── scales      [N, 3]   float32     │
           │  ├── opacities   [N]      float32     │
           │  └── sh_coeffs   [N, K, 3] float32    │
           └───────────────────┬──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   TRAINING LOOP     │
                    │   for step in range │
                    │      (num_steps):   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────────────────────┐
                    │                                      │
                    ▼                                      ▼
       ┌───────────────────┐              ┌──────────────────────┐
       │  loss_fn(params)  │              │  target image        │
       │  ┌─────────────┐  │              │  [1, H, W, 3]        │
       │  │ rasterize() │  │              │  from scene generator │
       │  │ PRD-09      │  │              └──────────────────────┘
       │  │  ┌────────┐ │  │
       │  │  │PRD-03  │ │  │  quat_scale_to_covar_preci
       │  │  │PRD-05  │ │  │  fully_fused_projection
       │  │  │PRD-04  │ │  │  spherical_harmonics
       │  │  │PRD-06  │ │  │  isect_tiles
       │  │  │PRD-07  │ │  │  rasterize_to_pixels
       │  │  └────────┘ │  │
       │  └──────┬──────┘  │
       │         ▼         │
       │  ┌─────────────┐  │
       │  │ L1 + SSIM   │  │
       │  │ losses.py   │  │
       │  └──────┬──────┘  │
       │         ▼         │
       │  scalar loss      │
       └─────────┬─────────┘
                 │
                 ▼
       mx.value_and_grad()
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  loss (scalar)     grads (tuple of 5 arrays)
        │                 │
        │                 ▼
        │        ┌──────────────────┐
        │        │ mx.eval(...)     │  ← materialize lazy graph
        │        └────────┬─────────┘
        │                 │
        ▼                 ▼
  history.append   ┌──────────────────┐
                   │ Strategy         │  (optional)
                   │ step_pre/post    │
                   │ → split/clone/   │
                   │   prune Gaussians│
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ SelectiveAdam    │
                   │ .step(name,      │
                   │   param, grad,   │
                   │   visibility)    │
                   │ for each of 5    │
                   │ parameter groups │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ mx.eval(         │  ← materialize updated params
                   │   *params.values │
                   │ )                │
                   └────────┬─────────┘
                            │
                            ▼
                   params ────────────► next iteration
```

---

## 6. Test Plan

### 6.1 Loss Function Tests — `tests/test_losses.py`

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_l1_identical` | L1 of identical images = 0 | `loss == 0.0` exactly |
| `test_l1_different` | L1 of different images > 0 | `loss > 0.0` |
| `test_l1_known_value` | L1 of all-zeros vs all-ones = 1.0 | `atol=1e-7` |
| `test_l1_range` | L1 of images in [0,1] is in [0,1] | `0 <= loss <= 1` |
| `test_l1_symmetric` | L1(a,b) == L1(b,a) | `atol=1e-7` |
| `test_l1_gradient_nonzero` | Gradient of L1 w.r.t. rendered is non-zero | All grads != 0 |
| `test_l1_gradient_sign` | Gradient points toward target | grad[rendered > target] < 0 |
| `test_ssim_identical` | SSIM of identical images = 1.0 | `atol=1e-5` |
| `test_ssim_different` | SSIM of random images < 1.0 | `ssim < 0.99` |
| `test_ssim_range` | SSIM is in [-1, 1] | `-1 <= ssim <= 1` |
| `test_ssim_symmetric` | SSIM(a,b) == SSIM(b,a) | `atol=1e-5` |
| `test_ssim_gradient_flows` | Gradient of SSIM loss w.r.t. image is non-zero | All grads != 0 |
| `test_ssim_black_vs_white` | SSIM of all-black vs all-white is very low | `ssim < 0.1` |
| `test_ssim_noise_vs_clean` | SSIM of clean vs noisy image < clean vs clean | Strict inequality |
| `test_combined_loss_lambda_zero` | lambda_ssim=0 gives pure L1 | `atol=1e-7` |
| `test_combined_loss_lambda_one` | lambda_ssim=1 gives pure SSIM loss | `atol=1e-7` |
| `test_combined_loss_gradient_flows` | Combined loss gradient is non-zero | All grads != 0 |
| `test_gaussian_kernel_sums_to_one` | Gaussian kernel normalization | `sum == 1.0` within `atol=1e-6` |
| `test_gaussian_kernel_symmetric` | Kernel values are symmetric around center | `atol=1e-7` |
| `test_gaussian_filter_constant_image` | Constant image unchanged by filter | `atol=1e-4` |
| `test_gaussian_filter_preserves_shape` | Output shape matches input shape | Exact match |

### 6.2 Optimizer Tests (extends PRD-11 tests)

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_adam_basic_update` | Parameter changes after step | `param_new != param_old` |
| `test_adam_reduces_quadratic` | Adam minimizes x^2 in 100 steps | final `|x| < 0.01` |
| `test_adam_visibility_mask_visible` | Visible params are updated | values change |
| `test_adam_visibility_mask_hidden` | Hidden params unchanged | `atol=0` (exact) |
| `test_adam_momentum_effect` | Step 2 update differs from step 1 | Different update magnitudes |
| `test_adam_bias_correction_step1` | First step uses bias-corrected moments | Formula validation |
| `test_adam_reset_clears_state` | Reset zeroes exp_avg and exp_avg_sq | All zeros |
| `test_adam_resize_preserves_existing` | Resize keeps existing state for kept indices | Values match |
| `test_adam_resize_zeros_new` | New entries after resize are zero | All zeros |
| `test_adam_lr_scaling` | Different lr produces proportionally different updates | Ratio check |

### 6.3 Convergence Tests — `tests/test_training.py`

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_convergence_solid_color` | N=500 Gaussians on solid color, 300 steps, L1 only | Loss reduction >= 50% |
| `test_convergence_gradient` | N=1000 Gaussians on gradient, 500 steps, L1 only | Loss reduction >= 40% |
| `test_convergence_circles` | N=2000 Gaussians on circles, 500 steps, L1 only | Loss reduction >= 30% |
| `test_loss_decreases_first_100` | Average loss steps 80-100 < average steps 0-20 | Strict inequality |
| `test_loss_not_nan` | No NaN in loss for 200 steps | All finite |
| `test_loss_not_inf` | No Inf in loss for 200 steps | All finite |

### 6.4 Gradient Flow Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_grad_flow_means` | means gradient is non-zero after 1 step | `mx.any(grad != 0)` |
| `test_grad_flow_quats` | quats gradient is non-zero after 1 step | `mx.any(grad != 0)` |
| `test_grad_flow_scales` | scales gradient is non-zero after 1 step | `mx.any(grad != 0)` |
| `test_grad_flow_opacities` | opacities gradient is non-zero after 1 step | `mx.any(grad != 0)` |
| `test_grad_flow_sh_coeffs` | sh_coeffs gradient is non-zero after 1 step | `mx.any(grad != 0)` |
| `test_grad_magnitude_means` | means gradient max abs < 1e4 | No explosion |
| `test_grad_magnitude_scales` | scales gradient max abs < 1e4 | No explosion |
| `test_grad_magnitude_opacities` | opacities gradient max abs < 1e4 | No explosion |
| `test_no_nan_grads_all_params` | No NaN/Inf in any gradient after 50 steps | `mx.all(mx.isfinite(...))` |
| `test_no_nan_params_after_100_steps` | No NaN/Inf in any parameter after 100 steps | `mx.all(mx.isfinite(...))` |

### 6.5 Parameter Validity Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_opacity_range_after_training` | sigmoid(opacities) in (0, 1) after 100 steps | Strict bounds |
| `test_scales_positive_after_training` | exp(scales) > 0 after 100 steps | All positive |
| `test_quats_normalizable` | Quaternion norms > 1e-6 (not degenerate) after 100 steps | All > threshold |
| `test_means_finite` | All means are finite after 100 steps | No NaN/Inf |
| `test_sh_coeffs_bounded` | SH coefficients don't explode after 100 steps | max abs < 100 |

### 6.6 Integration Tests

| Test | Description | Acceptance |
|------|-------------|------------|
| `test_forward_backward_roundtrip` | Forward + mx.grad produces valid output | No crash, shapes match |
| `test_rasterization_differentiable` | mx.grad of rasterization loss returns non-None gradients | Non-None |
| `test_optimizer_changes_params` | At least one param differs after optimizer step | Not all identical |
| `test_training_loop_10_steps` | Full loop runs 10 steps without error | No exception |
| `test_training_loop_deterministic` | Same seed produces same loss at each step | `atol=1e-6` |
| `test_training_loop_different_seeds` | Different seeds produce different losses | Not identical |
| `test_param_shapes_consistent` | All param shapes self-consistent after training | N dimension matches |
| `test_densification_changes_n` | N increases or decreases when densify=True | N != initial N |
| `test_optimizer_state_matches_params` | Optimizer state shapes match param shapes after densification | Same first dim |

### 6.7 Test Implementation Examples

```python
"""End-to-end training loop tests for gsplat-mlx.

These tests validate that the full pipeline -- forward rendering,
loss computation, backward gradients, and optimizer updates -- works
together correctly. They are the capstone validation of the port.

All tests use small image sizes (32x32 or 64x64) and few Gaussians
(100-500) to run quickly in CI. The convergence criteria are relaxed
compared to production training.
"""

import pytest
import mlx.core as mx


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def solid_scene():
    """64x64 solid color target scene."""
    from gsplat_mlx.scenes import make_solid_color_scene
    target, camera = make_solid_color_scene(width=64, height=64)
    mx.eval(target)
    return target, camera


@pytest.fixture
def gradient_scene():
    """64x64 gradient target scene."""
    from gsplat_mlx.scenes import make_gradient_scene
    target, camera = make_gradient_scene(width=64, height=64)
    mx.eval(target)
    return target, camera


@pytest.fixture
def small_params():
    """100 random Gaussians for fast tests."""
    from examples.simple_trainer import init_gaussians
    params = init_gaussians(n=100, sh_degree=0, seed=42)
    mx.eval(*params.values())
    return params


# ===================================================================
# Convergence Tests
# ===================================================================

class TestConvergence:
    """Tests that the training loop actually converges (loss decreases)."""

    def test_convergence_solid_color(self):
        """Solid color is the easiest target. Must converge reliably.

        If this fails, something is fundamentally broken:
        - Gradients not flowing
        - Optimizer not updating
        - Rasterization producing wrong output
        """
        from examples.simple_trainer import train

        history = train(
            scene="solid",
            n_gaussians=500,
            steps=300,
            width=64,
            height=64,
            use_ssim=False,
            densify=False,
            log_every=0,  # silent
            seed=42,
        )

        initial = history["losses"][0]
        final = history["losses"][-1]
        reduction = 1.0 - final / initial

        assert reduction >= 0.5, (
            f"Loss only reduced by {reduction*100:.1f}% "
            f"({initial:.4f} -> {final:.4f}). Expected >= 50%."
        )

    def test_convergence_gradient_scene(self):
        """Gradient scene requires spatial color variation."""
        from examples.simple_trainer import train

        history = train(
            scene="gradient",
            n_gaussians=1000,
            steps=500,
            width=64,
            height=64,
            use_ssim=False,
            densify=False,
            log_every=0,
            seed=42,
        )

        initial = history["losses"][0]
        final = history["losses"][-1]
        reduction = 1.0 - final / initial

        assert reduction >= 0.4, (
            f"Loss only reduced by {reduction*100:.1f}%. Expected >= 40%."
        )

    def test_loss_decreases_early(self):
        """Loss should decrease in the first 100 steps (averaging over windows)."""
        from examples.simple_trainer import train

        history = train(
            scene="solid",
            n_gaussians=500,
            steps=100,
            width=64,
            height=64,
            use_ssim=False,
            densify=False,
            log_every=0,
            seed=42,
        )

        early_avg = sum(history["losses"][:20]) / 20
        late_avg = sum(history["losses"][80:100]) / 20

        assert late_avg < early_avg, (
            f"Loss not decreasing: early avg={early_avg:.4f}, "
            f"late avg={late_avg:.4f}"
        )

    def test_no_nan_loss(self):
        """Loss should never be NaN during training."""
        from examples.simple_trainer import train

        history = train(
            scene="solid",
            n_gaussians=200,
            steps=200,
            width=32,
            height=32,
            use_ssim=False,
            densify=False,
            log_every=0,
            seed=42,
        )

        for i, loss in enumerate(history["losses"]):
            assert loss == loss, f"NaN loss at step {i}"
            assert abs(loss) < float("inf"), f"Inf loss at step {i}"


# ===================================================================
# Gradient Flow Tests
# ===================================================================

class TestGradientFlow:
    """Tests that gradients flow through all parameters correctly."""

    def test_all_params_get_gradients(self, solid_scene, small_params):
        """Every parameter type must receive a non-zero gradient.

        If any gradient is all-zero, that parameter cannot be optimized,
        meaning the training loop is broken for that component.
        """
        from examples.simple_trainer import make_loss_fn

        target, camera = solid_scene
        loss_fn = make_loss_fn(
            target=target,
            viewmat=camera["viewmat"],
            K=camera["K"],
            width=camera["width"],
            height=camera["height"],
            use_ssim=False,
        )

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))
        grads = grad_fn(
            small_params["means"],
            small_params["quats"],
            small_params["scales"],
            small_params["opacities"],
            small_params["sh_coeffs"],
        )
        mx.eval(*grads)

        names = ["means", "quats", "scales", "opacities", "sh_coeffs"]
        for name, g in zip(names, grads):
            has_nonzero = mx.any(g != 0).item()
            assert has_nonzero, f"Gradient for '{name}' is entirely zero."

    def test_no_nan_in_gradients(self, solid_scene, small_params):
        """No NaN or Inf in any gradient tensor."""
        from examples.simple_trainer import make_loss_fn

        target, camera = solid_scene
        loss_fn = make_loss_fn(
            target=target,
            viewmat=camera["viewmat"],
            K=camera["K"],
            width=camera["width"],
            height=camera["height"],
            use_ssim=False,
        )

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))
        grads = grad_fn(
            small_params["means"],
            small_params["quats"],
            small_params["scales"],
            small_params["opacities"],
            small_params["sh_coeffs"],
        )
        mx.eval(*grads)

        names = ["means", "quats", "scales", "opacities", "sh_coeffs"]
        for name, g in zip(names, grads):
            assert mx.all(mx.isfinite(g)).item(), (
                f"NaN/Inf in gradient for '{name}'."
            )

    def test_gradient_magnitudes_reasonable(self, solid_scene, small_params):
        """Gradients should not be exploding (max |grad| < 1e4)."""
        from examples.simple_trainer import make_loss_fn

        target, camera = solid_scene
        loss_fn = make_loss_fn(
            target=target,
            viewmat=camera["viewmat"],
            K=camera["K"],
            width=camera["width"],
            height=camera["height"],
            use_ssim=False,
        )

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))
        grads = grad_fn(
            small_params["means"],
            small_params["quats"],
            small_params["scales"],
            small_params["opacities"],
            small_params["sh_coeffs"],
        )
        mx.eval(*grads)

        names = ["means", "quats", "scales", "opacities", "sh_coeffs"]
        for name, g in zip(names, grads):
            max_abs = mx.max(mx.abs(g)).item()
            assert max_abs < 1e4, (
                f"Gradient for '{name}' is exploding: max|grad| = {max_abs:.2e}"
            )


# ===================================================================
# Sanity Tests
# ===================================================================

class TestSanity:
    """Basic sanity checks for parameter validity and determinism."""

    def test_opacity_valid_range(self):
        """sigmoid(opacities) must be in (0, 1) before and after training."""
        from examples.simple_trainer import init_gaussians
        params = init_gaussians(n=200, seed=42)
        opa = mx.sigmoid(params["opacities"])
        mx.eval(opa)
        assert mx.all(opa > 0).item(), "Opacity <= 0 detected"
        assert mx.all(opa < 1).item(), "Opacity >= 1 detected"

    def test_scales_positive(self):
        """exp(scales) must always be positive (exp never returns <= 0)."""
        from examples.simple_trainer import init_gaussians
        params = init_gaussians(n=200, seed=42)
        actual_scales = mx.exp(params["scales"])
        mx.eval(actual_scales)
        assert mx.all(actual_scales > 0).item(), "Non-positive scale detected"

    def test_deterministic_training(self):
        """Same seed must produce identical loss sequences."""
        from examples.simple_trainer import train

        kwargs = dict(
            scene="solid", n_gaussians=100, steps=20,
            width=32, height=32, use_ssim=False,
            densify=False, log_every=0, seed=123,
        )
        h1 = train(**kwargs)
        h2 = train(**kwargs)

        for i, (l1, l2) in enumerate(zip(h1["losses"], h2["losses"])):
            assert abs(l1 - l2) < 1e-6, (
                f"Non-deterministic at step {i}: {l1:.8f} vs {l2:.8f}"
            )

    def test_training_loop_runs_10_steps(self):
        """Full training loop completes 10 steps without error."""
        from examples.simple_trainer import train

        history = train(
            scene="solid", n_gaussians=50, steps=10,
            width=32, height=32, use_ssim=False,
            densify=False, log_every=0, seed=42,
        )

        assert len(history["losses"]) == 10
        assert all(isinstance(l, float) for l in history["losses"])

    def test_param_shapes_consistent(self):
        """All parameters must have matching N dimension throughout training."""
        from examples.simple_trainer import init_gaussians, train_step
        from gsplat_mlx.scenes import make_solid_color_scene

        params = init_gaussians(n=100, seed=42)
        target, camera = make_solid_color_scene(width=32, height=32)
        mx.eval(target, *params.values())

        for _ in range(5):
            loss, grads, info = train_step(
                params, target, camera["viewmat"], camera["K"],
                camera["width"], camera["height"],
            )
            mx.eval(loss, *grads.values())

            # Check N consistency
            N = params["means"].shape[0]
            assert params["quats"].shape[0] == N
            assert params["scales"].shape[0] == N
            assert params["opacities"].shape[0] == N
            assert params["sh_coeffs"].shape[0] == N
```

---

## 7. Performance Considerations

### 7.1 Expected Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Training speed | 50-200 steps/sec at 64x64, N=1000 | M1/M2 baseline |
| Training speed | 10-50 steps/sec at 256x256, N=2000 | M1/M2 baseline |
| Memory usage | < 2 GB for N=5000 at 256x256 | Fits any Apple Silicon Mac |
| Convergence (solid) | Loss < 0.05 in 300 steps | Easiest scene |
| Convergence (gradient) | Loss < 0.10 in 500 steps | Medium difficulty |
| Convergence (circles) | Loss < 0.15 in 500 steps | Harder scene |

### 7.2 Optimization Opportunities (Post-Capstone)

1. **Fuse forward + gradient**: Capture `info` inside the loss closure to avoid a second forward pass.
2. **Batch `mx.eval()` calls**: Group all materializations to minimize kernel launch overhead.
3. **Vectorize strategy loops**: Replace per-camera Python loops with batched MLX operations.
4. **Use `mx.compile()`**: Compile the training step function for kernel fusion.
5. **Profile hot paths**: Use `time.time()` per-step to identify bottlenecks.

### 7.3 Memory Management

MLX manages memory automatically, but be aware of:

- **Graph retention**: Large lazy computation graphs consume memory. Call `mx.eval()` after every training step to prevent graph growth.
- **Parameter copies during densification**: `split()` and `duplicate()` create temporary arrays. Old arrays are freed when no Python references remain.
- **Gradient computation**: `mx.grad()` builds a new computation graph each call. No persistent autograd tape to clear.

---

## 8. Acceptance Criteria

### 8.1 Must-Have (P0)

- [ ] `losses.py` implements L1 loss with correct gradient (verified by test)
- [ ] `losses.py` implements SSIM loss with correct gradient (verified by test)
- [ ] `losses.py` implements combined L1+SSIM loss with configurable lambda
- [ ] `scenes.py` generates all 4 synthetic scenes (solid, gradient, checkerboard, circles)
- [ ] Training loop runs for 500 steps on solid-color scene without crash, NaN, or Inf
- [ ] Loss decreases by at least 50% on solid-color scene (500 Gaussians, 300 steps, L1 only)
- [ ] Loss decreases by at least 40% on gradient scene (1000 Gaussians, 500 steps, L1 only)
- [ ] All 5 parameter types (means, quats, scales, opacities, sh_coeffs) receive non-zero gradients
- [ ] No NaN/Inf in any gradient or parameter throughout 200 steps of training
- [ ] `python examples/simple_trainer.py --scene solid --steps 300` runs end-to-end successfully
- [ ] All tests in `tests/test_training.py` pass
- [ ] All tests in `tests/test_losses.py` pass

### 8.2 Should-Have (P1)

- [ ] DefaultStrategy densification (split/clone/prune) runs without crash
- [ ] Gaussian count changes during training with `--densify` flag
- [ ] Training is deterministic (same seed = same loss sequence)
- [ ] SSIM loss gradient flows correctly (verified by test)
- [ ] Combined loss (L1 + SSIM) produces better perceptual quality than L1-only
- [ ] Optimizer state resizes correctly after densification

### 8.3 Nice-to-Have (P2)

- [ ] Checkerboard scene converges (loss reduction >= 20% in 1000 steps)
- [ ] Circles scene converges (loss reduction >= 30% in 500 steps)
- [ ] Training speed meets performance targets (section 7.1)
- [ ] Rendered images are visually recognizable (manual inspection)
- [ ] `--save_result` flag writes final rendered image to disk as PNG

---

## 9. Dependencies

| Dependency | PRD | What We Need |
|------------|-----|-------------|
| Dev environment | PRD-01 | Package structure, test infrastructure, `pyproject.toml` |
| Math utilities | PRD-02 | Basic array operations, helper functions |
| Covariance | PRD-03 | `quat_scale_to_covar_preci` |
| Spherical harmonics | PRD-04 | `spherical_harmonics` |
| Projection | PRD-05 | `fully_fused_projection` |
| Intersection | PRD-06 | `isect_tiles`, `isect_offset_encode` |
| Rasterization | PRD-07 | `rasterize_to_pixels` |
| Accumulate | PRD-08 | `accumulate` (used internally by PRD-07) |
| Rendering API | PRD-09 | `rasterization()` — the main user-facing entry point |
| Strategy | PRD-10 | `DefaultStrategy`, `duplicate`, `split`, `remove`, `reset_opa` |
| Optimizer | PRD-11 | `SelectiveAdam` |

**Critical path**: PRD-01 through PRD-09 must be fully implemented and passing all tests before this PRD can begin. PRD-10 and PRD-11 are needed for densification and optimization but can be implemented in parallel with this PRD if necessary (the training loop can use basic `mlx.optimizers.Adam` and skip densification as a fallback).

---

## 10. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Gradients don't flow through rasterization | **Critical** — training cannot converge | Medium | Test gradient flow through each module independently (PRD-07 backward tests). If `@mx.custom_function` VJP is broken, debug with finite differences against PyTorch reference. |
| Loss does not converge on solid color | **Critical** — capstone fails completely | Low | Start with L1-only, no SSIM, no densification. If this fails, compare per-component outputs against PyTorch reference. Check: are Gaussians in front of camera? Is the camera matrix correct? |
| NaN/Inf during training | **High** — silent corruption | Medium | Add NaN checks after every `mx.eval()`. Common sources: `log(0)` in scale parameterization, division by zero in conic inverse, overflow in `exp()`. Add epsilon guards everywhere. |
| SSIM implementation incorrect | **Medium** — wrong perceptual quality | Medium | Validate SSIM against scipy/skimage reference on identical-image and random-image cases. SSIM(img, img) must be exactly 1.0. |
| `mx.value_and_grad` doesn't support tuple returns | **Medium** — need fallback approach | Low | Fall back to separate forward pass for info dict (Approach B in section 4.4.3). Costs ~50% more compute but still correct. |
| Densification changes param shapes, breaking optimizer | **Medium** — training crashes at densification steps | Medium | Implement optimizer state resizing (SelectiveAdam.resize_param). Test shape consistency after every densification step. |
| MLX lazy eval causes subtle ordering bugs | **Medium** — hard to debug | Medium | Call `mx.eval()` after every mutation. Never store pre-eval array references across iterations. Add assertions on array shapes at every step. |
| Training too slow for useful iteration | **Low** — correctness comes first | Low | Profile with `time.time()` per step. Optimize only after correctness is proven. 64x64 images should be fast enough for rapid iteration. |

---

## 11. Implementation Order

```
Phase 1: Loss Functions (2 hours)
├── Implement l1_loss in losses.py
├── Implement _fspecial_gauss_1d (Gaussian kernel)
├── Implement _gaussian_filter_2d (separable 2D convolution)
├── Implement ssim (full SSIM formula)
├── Implement ssim_loss and combined_loss
├── Write tests/test_losses.py (all 20 tests)
└── Gate: SSIM(img, img) == 1.0, gradient flows through both losses

Phase 2: Synthetic Scenes (1 hour)
├── Implement make_solid_color_scene
├── Implement make_gradient_scene
├── Implement make_checkerboard_scene
├── Implement make_colored_circles_scene
├── Implement make_synthetic_gaussians_scene (roundtrip test)
└── Gate: All scenes generate valid [1, H, W, 3] images in [0, 1]

Phase 3: Minimal Training Loop (3 hours)
├── Implement init_gaussians
├── Implement make_loss_fn (closure for mx.grad)
├── Implement train_step (forward + backward)
├── Implement train() with L1-only, no densification, no SSIM
├── Test: solid color converges (loss decreases by 50%)
├── Test: gradient flows to all 5 parameter types
├── Test: no NaN/Inf in gradients
└── Gate: "python examples/simple_trainer.py --scene solid" converges

Phase 4: Full Loss + SSIM (1.5 hours)
├── Integrate SSIM into train_step
├── Add --lambda_ssim and --no_ssim CLI flags
├── Test: combined loss converges
├── Test: SSIM loss gradient flows
└── Gate: L1+SSIM converges on gradient scene

Phase 5: Densification Integration (2.5 hours)
├── Integrate DefaultStrategy (PRD-10) into training loop
├── Add --densify CLI flag
├── Handle optimizer state resizing on split/clone/prune
├── Test: Gaussian count changes with densify=True
├── Test: no crash during densification steps
├── Test: optimizer state shapes match after densification
└── Gate: Training with densification runs 500 steps without crash

Phase 6: Polish & Final Tests (2 hours)
├── Write remaining tests (determinism, param validity, etc.)
├── Add CLI argument parsing in simple_trainer.py
├── Performance profiling (time per step)
├── Fix any remaining edge cases
├── Final acceptance: run all 4 scenes, verify convergence
└── Gate: All tests pass, all scenes converge, no warnings
```

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Capstone** | A final integration test that validates the entire system end-to-end |
| **Convergence** | Loss decreasing monotonically (on average) during training |
| **Densification** | Adding/removing Gaussians during training to improve scene coverage |
| **Split** | Replace one large Gaussian with two smaller ones offset along principal axes |
| **Clone/Duplicate** | Copy a small Gaussian to a nearby position |
| **Prune** | Remove a Gaussian that has low opacity or is too large |
| **Logit-space** | Parameters stored as `logit(x)` so `sigmoid(param)` gives the actual value in (0,1) |
| **Log-space** | Scales stored as `log(s)` so `exp(param)` gives positive actual scale |
| **SH degree 0** | Constant color: 1 coefficient per channel, view-independent |
| **Visibility mask** | Boolean [N] array indicating which Gaussians were rendered this frame |
| **SelectiveAdam** | Adam optimizer that only updates momentum for visible Gaussians |
| **Scene scale** | Normalization factor for position/scale thresholds in densification |
| **Lazy evaluation** | MLX defers computation until `mx.eval()` is called |
| **`mx.value_and_grad`** | Returns both the function value and gradients in one call |
| **argnums** | Which function arguments to differentiate with respect to |
| **SSIM** | Structural Similarity Index Measure — perceptual image quality metric |
| **L1 loss** | Mean Absolute Error between rendered and target images |
