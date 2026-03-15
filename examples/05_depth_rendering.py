#!/usr/bin/env python3
"""Render depth maps and surface normals from Gaussians.

Creates a simple scene of Gaussians at varying depths and renders:
  - RGB color image
  - Depth map (accumulated depth channel)
  - Surface normals derived from the depth map

Usage:
    python examples/05_depth_rendering.py
"""

import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization
from gsplat_mlx.utils import depth_to_normal


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------


def render_depth_and_normals(
    width: int = 256,
    height: int = 256,
) -> dict:
    """Render RGB, depth, and normals for a scene with varying depths.

    Returns:
        Dict with keys 'rgb' [H,W,3], 'depth' [H,W,1], 'normals' [H,W,3],
        'alphas' [H,W,1].
    """
    # -- Create Gaussians at different depths --
    positions = []
    colors_list = []
    np.random.seed(123)

    # A grid of Gaussians with varying Z positions
    for row in np.linspace(-1.2, 1.2, 5):
        for col in np.linspace(-1.2, 1.2, 5):
            # Depth varies: closer in center, farther at edges
            z = 0.3 * (row**2 + col**2)
            positions.append([float(col), float(row), float(z)])
            # Color based on depth: near=warm, far=cool
            t = z / 1.0
            colors_list.append([
                float(1.0 - 0.5 * t),
                float(0.3 + 0.4 * t),
                float(0.2 + 0.6 * t),
            ])

    N = len(positions)
    means = mx.array(positions, dtype=mx.float32)
    quats = mx.concatenate([mx.ones((N, 1)), mx.zeros((N, 3))], axis=1)
    scales = mx.full((N, 3), 0.2, dtype=mx.float32)
    opacities = mx.full((N,), 0.9, dtype=mx.float32)
    colors = mx.array(colors_list, dtype=mx.float32)[None, :, :]  # [1, N, 3]

    # -- Camera --
    viewmat = mx.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]],
        dtype=mx.float32,
    )[None]

    focal = float(width)
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=mx.float32
    )[None]

    backgrounds = mx.zeros((1, 3), dtype=mx.float32)  # black bg (RGB only; depth channel added internally)

    # -- Render RGB + Depth --
    t0 = time.perf_counter()
    render_out, render_alphas, info = rasterization(
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
        render_mode="RGB+D",
        sh_degree=None,
    )
    mx.eval(render_out, render_alphas)
    elapsed_render = time.perf_counter() - t0

    # Split RGB and depth channels
    rgb = render_out[0, :, :, :3]       # [H, W, 3]
    depth = render_out[0, :, :, 3:4]    # [H, W, 1]
    alphas = render_alphas[0]           # [H, W, 1]

    print(f"Rendered RGB+D at {width}x{height} in {elapsed_render*1000:.1f} ms")
    print(f"  Depth range: [{float(mx.min(depth)):.3f}, {float(mx.max(depth)):.3f}]")
    print(f"  Alpha range: [{float(mx.min(alphas)):.3f}, {float(mx.max(alphas)):.3f}]")

    # -- Compute normals from depth --
    t1 = time.perf_counter()

    # Need camera-to-world matrix (inverse of viewmat)
    # For our simple camera: viewmat = [I | t], so camtoworld = [I | -t]
    camtoworld = mx.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5], [0, 0, 0, 1]],
        dtype=mx.float32,
    )

    normals = depth_to_normal(
        depths=depth[None],        # [1, H, W, 1]
        camtoworlds=camtoworld,    # [4, 4]
        Ks=K[0],                   # [3, 3]
        z_depth=True,
    )
    mx.eval(normals)
    elapsed_normals = time.perf_counter() - t1

    # Remove batch dim if present
    if normals.ndim == 4:
        normals = normals[0]

    print(f"  Computed normals in {elapsed_normals*1000:.1f} ms")
    print(f"  Normal range: [{float(mx.min(normals)):.3f}, {float(mx.max(normals)):.3f}]")

    return {
        "rgb": rgb,
        "depth": depth,
        "normals": normals,
        "alphas": alphas,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def depth_to_colormap(depth: mx.array) -> np.ndarray:
    """Convert a depth map [H, W, 1] to a viridis-like color image.

    Uses a simple blue-green-yellow colormap.
    """
    d = np.array(depth[:, :, 0])
    # Normalize to [0, 1]
    d_min, d_max = d[d > 0].min() if (d > 0).any() else 0, d.max()
    if d_max - d_min > 1e-6:
        d_norm = (d - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(d)

    # Simple blue-green-yellow colormap
    r = np.clip(d_norm * 2 - 0.5, 0, 1)
    g = np.clip(1.0 - abs(d_norm - 0.5) * 2, 0, 1)
    b = np.clip(1.0 - d_norm * 2, 0, 1)

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def normal_to_rgb(normals: mx.array) -> np.ndarray:
    """Convert normal map [H, W, 3] to RGB visualization.

    Maps normal components from [-1, 1] to [0, 1].
    """
    n = np.array(normals)
    return ((n * 0.5) + 0.5).clip(0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_image(arr: np.ndarray, path: str) -> None:
    """Save a numpy float32 image [H, W, 3] as PNG."""
    from PIL import Image

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    results = render_depth_and_normals()

    # Save individual images
    save_image(np.array(results["rgb"]), os.path.join(out_dir, "05_rgb.png"))
    save_image(
        depth_to_colormap(results["depth"]),
        os.path.join(out_dir, "05_depth.png"),
    )
    save_image(
        normal_to_rgb(results["normals"]),
        os.path.join(out_dir, "05_normals.png"),
    )

    # Combined side-by-side
    from PIL import Image

    rgb_np = np.clip(np.array(results["rgb"]) * 255, 0, 255).astype(np.uint8)
    depth_np = np.clip(depth_to_colormap(results["depth"]) * 255, 0, 255).astype(np.uint8)
    normal_np = np.clip(normal_to_rgb(results["normals"]) * 255, 0, 255).astype(np.uint8)
    combined = np.concatenate([rgb_np, depth_np, normal_np], axis=1)
    combined_path = os.path.join(out_dir, "05_rgb_depth_normals.png")
    Image.fromarray(combined).save(combined_path)
    print(f"  Saved combined: {combined_path}")
