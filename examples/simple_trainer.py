#!/usr/bin/env python3
"""Simple single-image 3DGS trainer for gsplat-mlx.

Trains a set of 3D Gaussians to reproduce a synthetic gradient image.
Uses the differentiable ``accumulate`` compositing path (not the tile-based
NumPy reference rasteriser, which is non-differentiable).

The pipeline:
  1. Project 3D Gaussians to 2D (differentiable via MLX autodiff)
  2. Evaluate SH degree-0 colours (differentiable)
  3. Build explicit intersection pairs for all Gaussian-pixel combinations
  4. Composite via ``accumulate()`` (differentiable)

Usage:
    python examples/simple_trainer.py [--num-steps 300] [--num-gaussians 200] [--lr 1e-2]

NOTE: The full tile-based ``rasterization()`` pipeline uses a NumPy reference
rasteriser (PRD-07) that is NOT differentiable.  Once a Tier-2 pure-MLX or
Tier-3 Metal rasteriser lands, the training loop can switch to the standard
``rasterization()`` entry point with no other changes.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import mlx.core as mx

from gsplat_mlx.core.covariance import quat_scale_to_covar_preci
from gsplat_mlx.core.projection import fully_fused_projection
from gsplat_mlx.core.spherical_harmonics import spherical_harmonics
from gsplat_mlx.core.accumulate import accumulate


# ---------------------------------------------------------------------------
# Gaussian initialisation
# ---------------------------------------------------------------------------


def create_random_gaussians(N: int, seed: int = 42) -> Dict[str, mx.array]:
    """Initialise *N* random Gaussians in a unit cube.

    Returns a dict with keys ``means``, ``quats``, ``scales``, ``opacities``,
    ``sh_coeffs`` -- the canonical 3DGS parameter set.
    """
    mx.random.seed(seed)
    means = mx.random.uniform(-1, 1, (N, 3))
    # Identity quaternions (w, x, y, z)
    quats = mx.concatenate([mx.ones((N, 1)), mx.zeros((N, 3))], axis=1)
    # Log-space scales: exp(-3) ~ 0.05
    scales = mx.full((N, 3), -3.0)
    # Sigmoid(0) = 0.5
    opacities = mx.full((N,), 0.0)
    # SH degree 0: one coefficient per channel
    sh_coeffs = mx.random.normal((N, 1, 3)) * 0.5
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }


# ---------------------------------------------------------------------------
# Synthetic target
# ---------------------------------------------------------------------------


def create_target_image(width: int, height: int) -> mx.array:
    """Create a simple gradient target image of shape ``[H, W, 3]``."""
    x = mx.linspace(0, 1, width)
    y = mx.linspace(0, 1, height)
    gx = mx.broadcast_to(mx.expand_dims(x, 0), (height, width))
    gy = mx.broadcast_to(mx.expand_dims(y, 1), (height, width))
    return mx.stack([gx, gy, mx.ones_like(gx) * 0.3], axis=-1)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def create_camera(
    width: int, height: int
) -> Tuple[mx.array, mx.array]:
    """Create a pinhole camera at z=3 looking at the origin.

    Returns ``(Ks, viewmats)`` each with a leading camera dimension of 1.
    """
    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=mx.float32
    )
    # Camera at z=3 looking along -z (identity rotation, translation in z).
    viewmat = mx.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
        dtype=mx.float32,
    )
    return K[None], viewmat[None]  # [1, 3, 3], [1, 4, 4]


# ---------------------------------------------------------------------------
# Differentiable rendering via accumulate
# ---------------------------------------------------------------------------


def _differentiable_render(
    means: mx.array,
    quats: mx.array,
    scales_exp: mx.array,
    opacities_sig: mx.array,
    sh_coeffs: mx.array,
    viewmat: mx.array,
    K: mx.array,
    width: int,
    height: int,
) -> mx.array:
    """Render Gaussians to an image using the differentiable accumulate path.

    This bypasses the non-differentiable tile-based rasteriser and instead:
      1. Projects all Gaussians (differentiable).
      2. Evaluates SH colours (differentiable).
      3. Builds explicit intersection lists for *all* projected Gaussians
         against *all* pixels (brute force -- feasible for small images).
      4. Composites via ``accumulate()`` which uses standard MLX ops.

    Args:
        means: Gaussian centres ``[N, 3]``.
        quats: Quaternions ``[N, 4]``.
        scales_exp: Already-exponentiated scales ``[N, 3]``.
        opacities_sig: Already-sigmoided opacities ``[N]``.
        sh_coeffs: SH coefficients ``[N, 1, 3]``.
        viewmat: World-to-camera matrix ``[1, 4, 4]``.
        K: Intrinsic matrix ``[1, 3, 3]``.
        width: Image width.
        height: Image height.

    Returns:
        Rendered image ``[H, W, 3]``.
    """
    N = means.shape[0]
    C = 1  # single camera

    # --- Step 1: Covariance from quats + scales ---
    covars, _ = quat_scale_to_covar_preci(
        quats, scales_exp, compute_covar=True, compute_preci=False, triu=False,
    )  # [N, 3, 3]

    # --- Step 2: Project ---
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, covars, viewmat, K, width, height,
        eps2d=0.3, near_plane=0.01, far_plane=1e10,
        calc_compensations=False, camera_model="pinhole",
    )
    # radii: [C, N, 2], means2d: [C, N, 2], depths: [C, N], conics: [C, N, 3]

    # --- Step 3: SH colours ---
    # Camera position from viewmat: -R^T @ t
    R = viewmat[0, :3, :3]
    t = viewmat[0, :3, 3]
    campos = -mx.einsum("ji,j->i", R, t)  # [3]

    dirs = means - campos[None, :]  # [N, 3]
    dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
    dirs = dirs / mx.maximum(dirs_norm, mx.array(1e-8))

    # spherical_harmonics expects [C, N, K, 3] coeffs and [C, N, 3] dirs
    coeffs_b = mx.expand_dims(sh_coeffs, 0)  # [1, N, 1, 3]
    dirs_b = mx.expand_dims(dirs, 0)  # [1, N, 3]
    rgb = spherical_harmonics(0, dirs_b, coeffs_b)  # [1, N, 3]
    rgb = mx.maximum(rgb + 0.5, 0.0)  # bias + clamp

    # --- Step 4: Build brute-force intersection lists ---
    # For small images this is tractable: N*H*W pairs per camera.
    # Filter to Gaussians with positive depth and non-zero radii.
    valid_mask = (depths[0] > 0.0)  # [N]
    valid_r = mx.minimum(radii[0, :, 0], radii[0, :, 1])  # [N]
    valid_mask = valid_mask & (valid_r > 0)

    valid_indices = mx.arange(N)
    # We need to sort Gaussians by depth for correct front-to-back compositing.
    depth_vals = depths[0]  # [N]

    # Build all (gaussian, pixel) pairs for valid Gaussians
    n_pixels = height * width
    pixel_ids_all = mx.arange(n_pixels)  # [H*W]

    # Sort valid Gaussians by depth
    sorted_order = mx.argsort(depth_vals)  # [N]

    # For the intersection list, we iterate depth-sorted Gaussians
    # and for each one emit pairs with all pixels.
    # Result: gaussian_ids = [g0,g0,...,g1,g1,...], pixel_ids = [0,1,...,0,1,...]
    # This is done via repeat/tile.

    # Tile pixel ids N times and repeat gaussian ids n_pixels times
    gaussian_ids = mx.repeat(sorted_order, n_pixels)  # [N * n_pixels]
    pixel_ids = mx.tile(pixel_ids_all, (N,))  # [N * n_pixels]
    image_ids = mx.zeros_like(gaussian_ids)  # all camera 0

    # --- Step 5: Accumulate ---
    renders, alphas = accumulate(
        means2d,       # [C, N, 2]
        conics,        # [C, N, 3]
        opacities_sig[None, :],  # [C, N] -- broadcast camera dim
        rgb,           # [C, N, 3]
        gaussian_ids,
        pixel_ids,
        image_ids,
        width,
        height,
    )
    # renders: [C, H, W, 3], alphas: [C, H, W, 1]
    return renders[0]  # [H, W, 3]


# ---------------------------------------------------------------------------
# L1 loss (inline to avoid dependency on losses.py being built)
# ---------------------------------------------------------------------------


def _l1_loss(pred: mx.array, target: mx.array) -> mx.array:
    """Element-wise L1 loss, averaged over all elements."""
    return mx.mean(mx.abs(pred - target))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    num_steps: int = 300,
    num_gaussians: int = 200,
    width: int = 32,
    height: int = 32,
    lr: float = 1e-2,
) -> list[float]:
    """Train Gaussians to reproduce a synthetic gradient image.

    Uses plain ``mx.optim.Adam`` (not ``SelectiveAdam``) because all
    Gaussians are visible in every frame of this single-image setup.

    Args:
        num_steps: Number of optimisation steps.
        num_gaussians: Number of Gaussians.
        width: Target image width.
        height: Target image height.
        lr: Base learning rate.

    Returns:
        List of per-step loss values.
    """
    params = create_random_gaussians(num_gaussians)
    target = create_target_image(width, height)
    K, viewmat = create_camera(width, height)

    # Flatten params into a single list for mx.value_and_grad
    param_names = ["means", "quats", "scales", "opacities", "sh_coeffs"]

    # Per-parameter learning rates (standard 3DGS practice)
    param_lrs = {
        "means": lr,
        "quats": lr * 0.1,
        "scales": lr * 0.5,
        "opacities": lr * 0.5,
        "sh_coeffs": lr * 0.5,
    }

    # Use simple Adam state dicts (no SelectiveAdam needed -- all visible)
    adam_state: Dict[str, Dict[str, mx.array]] = {}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(
        f"Training {num_gaussians} Gaussians on {width}x{height} image "
        f"for {num_steps} steps (differentiable accumulate path)"
    )

    losses: list[float] = []
    t0 = time.time()

    for step in range(num_steps):
        def loss_fn(
            means: mx.array,
            quats: mx.array,
            scales: mx.array,
            opacities: mx.array,
            sh_coeffs: mx.array,
        ) -> mx.array:
            rendered = _differentiable_render(
                means=means,
                quats=quats,
                scales_exp=mx.exp(scales),
                opacities_sig=mx.sigmoid(opacities),
                sh_coeffs=sh_coeffs,
                viewmat=viewmat,
                K=K,
                width=width,
                height=height,
            )
            return _l1_loss(rendered, target)

        loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
            params["means"],
            params["quats"],
            params["scales"],
            params["opacities"],
            params["sh_coeffs"],
        )
        mx.eval(loss)

        grad_dict = dict(zip(param_names, grads))

        # Manual Adam update per parameter
        for name in param_names:
            p = params[name]
            g = grad_dict[name]
            plr = param_lrs[name]

            if name not in adam_state:
                adam_state[name] = {
                    "step": 0,
                    "m": mx.zeros_like(p),
                    "v": mx.zeros_like(p),
                }

            s = adam_state[name]
            s["step"] += 1
            t_step = s["step"]

            s["m"] = beta1 * s["m"] + (1 - beta1) * g
            s["v"] = beta2 * s["v"] + (1 - beta2) * g * g

            m_hat = s["m"] / (1 - beta1**t_step)
            v_hat = s["v"] / (1 - beta2**t_step)

            params[name] = p - plr * m_hat / (mx.sqrt(v_hat) + eps)

        # Force evaluation of updated params and state
        mx.eval(*[params[n] for n in param_names])
        mx.eval(
            *[adam_state[n]["m"] for n in param_names],
            *[adam_state[n]["v"] for n in param_names],
        )

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 50 == 0 or step == num_steps - 1:
            elapsed = time.time() - t0
            print(
                f"  Step {step:4d}/{num_steps} | "
                f"Loss: {loss_val:.4f} | Time: {elapsed:.1f}s"
            )

    print(f"\nFinal loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
    if losses[0] > 0:
        print(f"Loss reduction: {(1 - losses[-1] / losses[0]) * 100:.1f}%")
    return losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple single-image 3DGS trainer for gsplat-mlx."
    )
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-gaussians", type=int, default=200)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    train(args.num_steps, args.num_gaussians, args.width, args.height, args.lr)
