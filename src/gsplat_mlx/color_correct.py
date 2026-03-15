"""Color correction utilities for post-processing evaluation.

Port of ``gsplat/color_correct.py`` from PyTorch to MLX / NumPy.
The least-squares solve uses NumPy (``np.linalg.lstsq``); inputs and
outputs are ``mx.array``.
"""

import mlx.core as mx
import numpy as np


def color_correct_quadratic(
    img: mx.array,
    ref: mx.array,
    num_iters: int = 5,
    eps: float = 0.5 / 255,
) -> mx.array:
    """Warp *img* to match the colors in *ref* using iterative quadratic fitting.

    For each channel the algorithm constructs a design matrix containing
    quadratic, linear, and bias terms of the current image pixels, then
    solves a least-squares system against the reference channel.  Saturated
    pixels (outside ``[eps, 1-eps]``) are excluded from the fit.  The
    process repeats for *num_iters* iterations since the set of saturated
    pixels may change after each correction.

    Adapted from the multinerf implementation by Google Research.

    Args:
        img: Input image. Shape ``[..., C]`` (typically ``[H, W, 3]``).
        ref: Reference image to match. Shape ``[..., C]``.
        num_iters: Number of iterations. Default ``5``.
        eps: Clipping threshold. Default ``0.5 / 255``.

    Returns:
        Color-corrected image with the same shape as *img*.
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"Channel mismatch: img has {img.shape[-1]}, ref has {ref.shape[-1]}"
        )
    num_channels = img.shape[-1]
    original_shape = img.shape

    # Work in NumPy for lstsq
    img_np = np.array(img).reshape(-1, num_channels).astype(np.float64)
    ref_np = np.array(ref).reshape(-1, num_channels).astype(np.float64)

    def is_unclipped(z: np.ndarray) -> np.ndarray:
        return (z >= eps) & (z <= 1 - eps)

    mask0 = is_unclipped(img_np)

    for _ in range(num_iters):
        # Build design matrix: quadratic + linear + bias
        a_parts = []
        for c in range(num_channels):
            # Quadratic: img[:,c] * img[:,c:] -> columns for upper-triangle products
            a_parts.append(img_np[:, c : c + 1] * img_np[:, c:])
        a_parts.append(img_np)  # linear
        a_parts.append(np.ones((img_np.shape[0], 1), dtype=np.float64))  # bias
        a_mat = np.concatenate(a_parts, axis=-1)

        warp_cols = []
        for c in range(num_channels):
            b = ref_np[:, c]
            mask = mask0[:, c] & is_unclipped(img_np[:, c]) & is_unclipped(b)
            ma = np.where(mask[:, None], a_mat, 0.0)
            mb = np.where(mask, b, 0.0)
            w, _, _, _ = np.linalg.lstsq(ma, mb, rcond=None)
            warp_cols.append(w)
        warp = np.stack(warp_cols, axis=-1)  # (D, C)
        img_np = np.clip(a_mat @ warp, 0, 1)

    corrected = mx.array(img_np.astype(np.float32).reshape(original_shape))
    return corrected


def color_correct_affine(
    img: mx.array,
    ref: mx.array,
    num_iters: int = 5,
    eps: float = 0.5 / 255,
) -> mx.array:
    """Warp *img* to match the colors in *ref* using per-channel affine correction.

    Computes the best-fit affine mapping ``a * ref + b = img`` per channel,
    then applies the inverse ``(img - b) / a`` to produce a corrected image
    that should approximate *ref*.

    Args:
        img: Input image. Shape ``[..., C]``.
        ref: Reference image. Shape ``[..., C]``.
        num_iters: Unused (kept for API compatibility). Default ``5``.
        eps: Unused (kept for API compatibility). Default ``0.5 / 255``.

    Returns:
        Color-corrected image with the same shape as *img*.
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"Channel mismatch: img has {img.shape[-1]}, ref has {ref.shape[-1]}"
        )
    num_channels = img.shape[-1]
    original_shape = img.shape

    img_np = np.array(img).reshape(-1, num_channels).astype(np.float64)
    ref_np = np.array(ref).reshape(-1, num_channels).astype(np.float64)

    ref_mean = ref_np.mean(axis=0)
    img_mean = img_np.mean(axis=0)
    ref_img_mean = (ref_np * img_np).mean(axis=0)
    ref_ref_mean = (ref_np * ref_np).mean(axis=0)

    var_ref = ref_ref_mean - ref_mean * ref_mean
    var_ref = np.maximum(var_ref, 1e-8)

    a = (ref_img_mean - ref_mean * img_mean) / var_ref
    b = img_mean - a * ref_mean

    # Inverse mapping
    a = np.where(np.abs(a) < 1e-8, np.ones_like(a), a)
    corrected = np.clip((img_np - b) / a, 0, 1)

    return mx.array(corrected.astype(np.float32).reshape(original_shape))
