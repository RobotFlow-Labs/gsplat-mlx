"""Loss functions for 3D Gaussian Splatting training.

All functions are differentiable through MLX's automatic differentiation.
No NumPy is used in any differentiable path.

Provides:
- l1_loss: Mean absolute error
- ssim / ssim_loss: Structural Similarity Index (SSIM) via separable Gaussian blur
- combined_loss: Weighted L1 + SSIM as used in the original 3DGS paper
"""

import mlx.core as mx


# ---------------------------------------------------------------------------
# L1 Loss
# ---------------------------------------------------------------------------


def l1_loss(rendered: mx.array, target: mx.array) -> mx.array:
    """Mean absolute error between rendered and target images.

    Args:
        rendered: Rendered image [H, W, C] or [B, H, W, C], float32.
        target: Target image, same shape as rendered, float32.

    Returns:
        Scalar loss value.
    """
    return mx.mean(mx.abs(rendered - target))


# ---------------------------------------------------------------------------
# Gaussian Blur Helpers (for SSIM)
# ---------------------------------------------------------------------------


def _fspecial_gauss_1d(size: int = 11, sigma: float = 1.5) -> mx.array:
    """Create a 1D Gaussian kernel.

    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation.

    Returns:
        Normalized 1D Gaussian kernel of shape [size], summing to 1.
    """
    coords = mx.arange(size, dtype=mx.float32) - size // 2
    g = mx.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / mx.sum(g)


def _gaussian_filter_2d(img: mx.array, kernel_1d: mx.array) -> mx.array:
    """Apply separable 2D Gaussian filter using two 1D convolutions.

    Uses depthwise convolution via mx.conv2d with groups=C to filter each
    channel independently. Two passes: vertical then horizontal.

    MLX conv2d expects:
    - Input: (N, H, W, C_in)
    - Weight: (C_out, kH, kW, C_in / groups)

    For depthwise with groups=C: C_out=C, C_in/groups=1,
    so weight shape is (C, kH, kW, 1).

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
    img_padded = mx.pad(img, [(0, 0), (pad, pad), (0, 0), (0, 0)], mode="edge")
    # Weight: (C_out=C, kH=K, kW=1, C_in/groups=1)
    w_v = kernel_1d.reshape(1, K, 1, 1)
    w_v = mx.broadcast_to(w_v, (C, K, 1, 1))
    out = mx.conv2d(img_padded, w_v, stride=1, padding=0, groups=C)

    # --- Horizontal pass (filter along W dimension) ---
    out_padded = mx.pad(out, [(0, 0), (0, 0), (pad, pad), (0, 0)], mode="edge")
    # Weight: (C_out=C, kH=1, kW=K, C_in/groups=1)
    w_h = kernel_1d.reshape(1, 1, K, 1)
    w_h = mx.broadcast_to(w_h, (C, 1, K, 1))
    out = mx.conv2d(out_padded, w_h, stride=1, padding=0, groups=C)

    return out


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


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
        **kwargs: Additional arguments passed to ssim().

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

    Args:
        rendered: Rendered image [H, W, C] or [B, H, W, C].
        target: Target image, same shape.
        lambda_ssim: Weight for the SSIM component. Default: 0.2.

    Returns:
        Scalar combined loss.
    """
    return (1.0 - lambda_ssim) * l1_loss(rendered, target) + \
           lambda_ssim * ssim_loss(rendered, target)
