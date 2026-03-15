"""Alpha compositing with explicit intersection pairs.

Replaces nerfacc dependency with pure MLX implementations of
``render_weight_from_alpha`` and ``accumulate_along_rays``, then
provides the main ``accumulate`` compositing function that mirrors
upstream ``gsplat.cuda._torch_impl.accumulate``.

All operations use standard differentiable MLX primitives so that
``mx.grad`` works through the full compositing pipeline without any
custom backward pass.
"""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx

from gsplat_mlx.core.constants import MAX_ALPHA


# ---------------------------------------------------------------------------
# render_weight_from_alpha  (replaces nerfacc)
# ---------------------------------------------------------------------------


def render_weight_from_alpha(
    alphas: mx.array,
    ray_indices: mx.array,
    n_rays: int,
) -> Tuple[mx.array, mx.array]:
    """Compute front-to-back compositing weights from per-intersection alphas.

    Intersections belonging to the same ray **must** be contiguous in the
    input arrays and sorted front-to-back (nearest first).

    This implementation is fully differentiable through MLX's autodiff.

    Args:
        alphas: Per-intersection opacity values.  Shape ``[M]``.
        ray_indices: Ray index for each intersection.  Shape ``[M]``.
            Values in ``[0, n_rays)``.
        n_rays: Total number of rays (used for output sizing).

    Returns:
        weights: Per-intersection compositing weights.  Shape ``[M]``.
        transmittances: Per-intersection transmittance (exclusive cumprod
            of ``1 - alpha``).  Shape ``[M]``.
    """
    M = alphas.shape[0]
    if M == 0:
        return mx.array([], dtype=mx.float32), mx.array([], dtype=mx.float32)

    # Step 1: log(1 - alpha), clamped to avoid log(0)
    one_minus_alpha = mx.clip(1.0 - alphas, a_min=1e-10, a_max=None)
    log_oma = mx.log(one_minus_alpha)  # [M]

    # Step 2: Global inclusive cumulative sum in log space
    log_cumsum = mx.cumsum(log_oma)  # [M]

    # Step 3: Inclusive -> exclusive cumsum (shift right, prepend 0)
    log_exclusive = mx.concatenate([mx.zeros(1), log_cumsum[:-1]])  # [M]

    # Step 4: Identify segment boundaries (new ray starts)
    shifted_rays = mx.concatenate([ray_indices[:1] - 1, ray_indices[:-1]])
    is_start = ray_indices != shifted_rays  # [M] bool

    # Step 5: Compute per-segment correction
    # At each segment start the exclusive cumsum should be 0 (T=1).
    # We subtract the boundary's cumsum value from the whole segment.
    segment_ids = mx.cumsum(is_start.astype(mx.int32)) - 1  # [M]

    # Use n_rays as upper bound for segment count to avoid breaking
    # the lazy computation graph with mx.eval / .item() calls.
    n_segments = n_rays

    start_mask = is_start.astype(mx.float32)  # 1.0 at starts
    corrections_compact = mx.zeros(n_segments, dtype=mx.float32)
    corrections_compact = corrections_compact.at[segment_ids].add(
        log_exclusive * start_mask
    )

    # Step 6: Gather correction per element and correct
    corrections = corrections_compact[segment_ids]  # [M]
    log_transmittance = log_exclusive - corrections  # [M]

    # Step 7: Exponentiate to get transmittance
    transmittances = mx.exp(log_transmittance)  # [M]

    # Step 8: Compute weights = transmittance * alpha
    weights = transmittances * alphas  # [M]

    return weights, transmittances


# ---------------------------------------------------------------------------
# accumulate_along_rays  (replaces nerfacc)
# ---------------------------------------------------------------------------


def accumulate_along_rays(
    weights: mx.array,
    values: mx.array | None,
    ray_indices: mx.array,
    n_rays: int,
) -> mx.array:
    """Accumulate weighted values along rays via scatter-add.

    For each intersection *i*, adds ``weights[i] * values[i]`` to
    ``output[ray_indices[i]]``.  If *values* is ``None``, accumulates
    weights only (equivalent to ``values = ones``).

    Args:
        weights: Per-intersection weights.  Shape ``[M]``.
        values: Per-intersection feature vectors.  Shape ``[M, C]`` or
            ``None``.
        ray_indices: Ray index for each intersection.  Shape ``[M]``.
        n_rays: Total number of output rays.

    Returns:
        accumulated: Shape ``[n_rays, C]`` if *values* provided,
            ``[n_rays, 1]`` if *values* is ``None``.
    """
    if values is not None:
        C = values.shape[-1]
        weighted = weights[:, None] * values  # [M, C]
        output = mx.zeros((n_rays, C), dtype=mx.float32)
        output = output.at[ray_indices].add(weighted)
    else:
        output = mx.zeros((n_rays, 1), dtype=mx.float32)
        output = output.at[ray_indices].add(weights[:, None])
    return output


# ---------------------------------------------------------------------------
# accumulate  (main compositing function)
# ---------------------------------------------------------------------------


def accumulate(
    means2d: mx.array,
    conics: mx.array,
    opacities: mx.array,
    colors: mx.array,
    gaussian_ids: mx.array,
    pixel_ids: mx.array,
    image_ids: mx.array,
    image_width: int,
    image_height: int,
) -> Tuple[mx.array, mx.array]:
    """Alpha compositing of 2D Gaussians with explicit intersection pairs.

    Given explicit ``(gaussian_id, pixel_id, image_id)`` tuples specifying
    which Gaussians contribute to which pixels, compute front-to-back alpha
    compositing.  The intersections must be sorted front-to-back within each
    ray (pixel).

    This is a flexible alternative to the tile-based fused rasteriser.  It
    is slower but allows arbitrary intersection patterns and relies on
    MLX's automatic differentiation for the backward pass.

    Args:
        means2d: 2D Gaussian centres.  Shape ``[..., N, 2]``.
        conics: Inverse 2D covariance (upper triangle: a, b, c).
            Shape ``[..., N, 3]``.
        opacities: Per-Gaussian opacity.  Shape ``[..., N]``.
        colors: Per-Gaussian colour / features.  Shape ``[..., N, C]``.
        gaussian_ids: Which Gaussian each intersection refers to.
            Shape ``[M]``.
        pixel_ids: Which pixel (row-major index) each intersection refers
            to.  Shape ``[M]``.
        image_ids: Which image (batch index) each intersection refers to.
            Shape ``[M]``.
        image_width: Width of the output image in pixels.
        image_height: Height of the output image in pixels.

    Returns:
        renders: Composited colours.
            Shape ``[..., image_height, image_width, C]``.
        alphas: Composited opacity.
            Shape ``[..., image_height, image_width, 1]``.
    """
    # ---- Step 0: Shape bookkeeping ----
    image_dims = means2d.shape[:-2]
    I = math.prod(image_dims) if image_dims else 1
    N = means2d.shape[-2]
    channels = colors.shape[-1]

    assert means2d.shape == image_dims + (N, 2), f"means2d shape {means2d.shape}"
    assert conics.shape == image_dims + (N, 3), f"conics shape {conics.shape}"
    assert opacities.shape == image_dims + (N,), f"opacities shape {opacities.shape}"
    assert colors.shape == image_dims + (N, channels), f"colors shape {colors.shape}"

    # Flatten batch dimensions
    means2d_flat = means2d.reshape(I, N, 2)
    conics_flat = conics.reshape(I, N, 3)
    opacities_flat = opacities.reshape(I, N)
    colors_flat = colors.reshape(I, N, channels)

    # ---- Step 1: Compute pixel coordinates ----
    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = (
        mx.stack([pixel_ids_x, pixel_ids_y], axis=-1).astype(mx.float32) + 0.5
    )  # [M, 2]

    # ---- Step 2: Gather Gaussian parameters per intersection ----
    means2d_selected = means2d_flat[image_ids, gaussian_ids]  # [M, 2]
    conics_selected = conics_flat[image_ids, gaussian_ids]  # [M, 3]
    opacities_selected = opacities_flat[image_ids, gaussian_ids]  # [M]
    colors_selected = colors_flat[image_ids, gaussian_ids]  # [M, C]

    # ---- Step 3: Compute sigma (Gaussian exponent) ----
    deltas = pixel_coords - means2d_selected  # [M, 2]
    sigmas = (
        0.5
        * (
            conics_selected[:, 0] * deltas[:, 0] ** 2
            + conics_selected[:, 2] * deltas[:, 1] ** 2
        )
        + conics_selected[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]

    # ---- Step 4: Compute per-intersection alpha ----
    alphas_per_isect = mx.minimum(
        opacities_selected * mx.exp(-sigmas), MAX_ALPHA
    )  # [M]

    # ---- Step 5: Compute ray indices ----
    ray_indices = image_ids * (image_height * image_width) + pixel_ids  # [M]
    total_pixels = I * image_height * image_width

    # ---- Step 6: Front-to-back compositing weights ----
    weights, _transmittances = render_weight_from_alpha(
        alphas_per_isect, ray_indices, total_pixels
    )

    # ---- Step 7: Accumulate weighted colours ----
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
