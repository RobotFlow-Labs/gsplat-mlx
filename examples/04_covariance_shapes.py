#!/usr/bin/env python3
"""Visualize Gaussian shapes: spheres, ellipsoids, disks, needles.

Demonstrates how quaternion rotation and anisotropic scale parameters
control the shape of 3D Gaussians. Creates 4 Gaussians with distinct
geometries and renders them from a single viewpoint.

Usage:
    python examples/04_covariance_shapes.py
"""

import math
import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization
from gsplat_mlx.core.covariance import quat_scale_to_covar_preci


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------


def _axis_angle_to_quat(axis: list, angle_deg: float) -> list:
    """Convert axis-angle to quaternion (w, x, y, z)."""
    angle = math.radians(angle_deg)
    norm = math.sqrt(sum(a * a for a in axis))
    ax = [a / norm for a in axis]
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    return [c, ax[0] * s, ax[1] * s, ax[2] * s]


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------


def render_covariance_shapes(
    width: int = 384,
    height: int = 256,
) -> tuple:
    """Create 4 Gaussians with different shapes and render them.

    Shapes:
        1. Sphere: equal scales in all axes.
        2. Flat disk: one scale near zero (pancake shape).
        3. Needle: two scales near zero (elongated cigar shape).
        4. Rotated ellipsoid: 45-degree rotation with anisotropic scales.

    Returns:
        (image [H, W, 3], alphas [H, W, 1])
    """
    # -- Gaussian parameters --
    means = mx.array(
        [
            [-1.5, 0.0, 0.0],    # sphere
            [-0.5, 0.0, 0.0],    # flat disk
            [0.5, 0.0, 0.0],     # needle
            [1.5, 0.0, 0.0],     # rotated ellipsoid
        ],
        dtype=mx.float32,
    )
    N = means.shape[0]

    # Quaternions
    identity_q = [1.0, 0.0, 0.0, 0.0]
    rotated_q = _axis_angle_to_quat([1, 1, 0], 45)
    quats = mx.array(
        [
            identity_q,     # sphere: no rotation needed
            identity_q,     # disk: flat in Z
            identity_q,     # needle: elongated in X
            rotated_q,      # rotated 45 degrees
        ],
        dtype=mx.float32,
    )

    # Scales (actual, not log-space)
    scales = mx.array(
        [
            [0.3, 0.3, 0.3],       # sphere
            [0.4, 0.4, 0.02],      # disk (flat in Z)
            [0.5, 0.03, 0.03],     # needle (thin in Y and Z)
            [0.4, 0.15, 0.08],     # anisotropic ellipsoid
        ],
        dtype=mx.float32,
    )

    opacities = mx.full((N,), 0.95, dtype=mx.float32)

    # Direct colors [1, N, 3]
    colors = mx.array(
        [
            [0.9, 0.3, 0.2],   # red sphere
            [0.2, 0.8, 0.3],   # green disk
            [0.3, 0.3, 0.9],   # blue needle
            [0.9, 0.7, 0.1],   # yellow ellipsoid
        ],
        dtype=mx.float32,
    )[None, :, :]

    # -- Camera --
    viewmat = mx.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]],
        dtype=mx.float32,
    )[None]

    focal = float(width) * 0.7
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=mx.float32
    )[None]

    backgrounds = mx.ones((1, 3), dtype=mx.float32) * 0.12

    # -- Print covariance info --
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False,
    )
    mx.eval(covars)
    labels = ["Sphere", "Flat disk", "Needle", "Rotated ellipsoid"]
    print("Covariance matrices (diagonal elements):")
    for i, label in enumerate(labels):
        c = np.array(covars[i])
        diag = np.diag(c)
        print(f"  {label:20s}: diag = [{diag[0]:.4f}, {diag[1]:.4f}, {diag[2]:.4f}]")

    # -- Render --
    t0 = time.perf_counter()
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        backgrounds=backgrounds,
        render_mode="RGB",
        sh_degree=None,
    )
    mx.eval(render_colors, render_alphas)
    elapsed = time.perf_counter() - t0

    img = render_colors[0]
    alphas = render_alphas[0]

    print(f"\nRendered 4 shape variants at {width}x{height} in {elapsed*1000:.1f} ms")
    print(f"  Shapes: {', '.join(labels)}")

    return img, alphas


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_image(img: mx.array, path: str) -> None:
    """Save an MLX image array [H, W, 3] as PNG."""
    from PIL import Image

    arr = np.array(img)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    img, alphas = render_covariance_shapes()
    save_image(img, os.path.join(out_dir, "04_covariance_shapes.png"))
