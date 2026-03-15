#!/usr/bin/env python3
"""Hello Gaussians -- Render your first 3D Gaussians on Apple Silicon.

This is the simplest gsplat-mlx example. It creates a handful of
colored Gaussians, renders them to an image, and saves the result.

Usage:
    python examples/01_hello_gaussians.py
"""

import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization


# ---------------------------------------------------------------------------
# Core rendering function (importable for tests)
# ---------------------------------------------------------------------------


def render_hello_gaussians(
    width: int = 256,
    height: int = 256,
) -> tuple:
    """Create a small set of colored Gaussians and render them.

    Returns:
        (rendered_image, alphas) where rendered_image is [H, W, 3] float32
        and alphas is [H, W, 1] float32.
    """
    # -- Define 7 Gaussians at known positions with known colors --
    means = mx.array(
        [
            [0.0, 0.0, 0.0],     # center -- red
            [-1.0, 0.8, 0.2],    # top-left -- green
            [1.0, 0.8, -0.1],    # top-right -- blue
            [-1.0, -0.8, 0.1],   # bottom-left -- yellow
            [1.0, -0.8, -0.2],   # bottom-right -- magenta
            [0.0, 1.2, 0.0],     # top-center -- cyan
            [0.0, -1.2, 0.0],    # bottom-center -- white
        ],
        dtype=mx.float32,
    )
    N = means.shape[0]

    # Identity quaternions (no rotation): (w, x, y, z) = (1, 0, 0, 0)
    quats = mx.concatenate(
        [mx.ones((N, 1)), mx.zeros((N, 3))], axis=1
    )

    # Uniform moderate scales
    scales = mx.full((N, 3), 0.3, dtype=mx.float32)

    # Full opacity
    opacities = mx.ones((N,), dtype=mx.float32)

    # Direct RGB colors (no SH) -- shape [C, N, 3] where C=1
    colors_rgb = mx.array(
        [
            [1.0, 0.2, 0.1],   # red
            [0.1, 0.9, 0.2],   # green
            [0.1, 0.3, 1.0],   # blue
            [1.0, 1.0, 0.1],   # yellow
            [1.0, 0.1, 0.9],   # magenta
            [0.1, 0.9, 0.9],   # cyan
            [0.9, 0.9, 0.9],   # white
        ],
        dtype=mx.float32,
    )
    # Add camera dimension: [1, N, 3]
    colors_rgb = colors_rgb[None, :, :]

    # -- Camera setup: pinhole at z=5 looking at origin --
    viewmat = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=mx.float32,
    )[None, :, :]  # [1, 4, 4]

    focal = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=mx.float32,
    )[None, :, :]  # [1, 3, 3]

    # White background
    backgrounds = mx.ones((1, 3), dtype=mx.float32)

    # -- Render --
    t0 = time.perf_counter()
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_rgb,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        backgrounds=backgrounds,
        render_mode="RGB",
        sh_degree=None,  # direct color mode
    )
    mx.eval(render_colors, render_alphas)
    elapsed = time.perf_counter() - t0

    # Remove camera dimension: [1, H, W, 3] -> [H, W, 3]
    img = render_colors[0]
    alphas = render_alphas[0]

    print(f"Rendered {N} Gaussians at {width}x{height} in {elapsed*1000:.1f} ms")
    print(f"  Output shape: {img.shape}")
    print(f"  Value range: [{float(mx.min(img)):.3f}, {float(mx.max(img)):.3f}]")
    print(f"  Alpha range: [{float(mx.min(alphas)):.3f}, {float(mx.max(alphas)):.3f}]")

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

    img, alphas = render_hello_gaussians()
    save_image(img, os.path.join(out_dir, "01_hello_gaussians.png"))
