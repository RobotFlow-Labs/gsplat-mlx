#!/usr/bin/env python3
"""Compare pinhole, fisheye, and orthographic camera models.

Renders the same scene of colored Gaussians with each of the three
camera models supported by gsplat-mlx and saves them side by side.

Usage:
    python examples/02_camera_models.py
"""

import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization


# ---------------------------------------------------------------------------
# Shared scene
# ---------------------------------------------------------------------------


def _create_scene(width: int, height: int):
    """Create a grid of colored Gaussians and camera parameters.

    Returns:
        Dict with means, quats, scales, opacities, colors, viewmat, K,
        backgrounds, width, height.
    """
    # 3x3 grid of Gaussians on the XY plane
    positions = []
    colors_list = []
    palette = [
        [1.0, 0.2, 0.1],
        [0.1, 0.8, 0.2],
        [0.2, 0.3, 1.0],
        [1.0, 1.0, 0.1],
        [1.0, 0.1, 0.8],
        [0.1, 0.9, 0.9],
        [0.9, 0.5, 0.1],
        [0.5, 0.1, 0.9],
        [0.9, 0.9, 0.9],
    ]
    idx = 0
    for row in [-1.0, 0.0, 1.0]:
        for col in [-1.0, 0.0, 1.0]:
            positions.append([col, row, 0.0])
            colors_list.append(palette[idx])
            idx += 1

    N = len(positions)
    means = mx.array(positions, dtype=mx.float32)
    quats = mx.concatenate([mx.ones((N, 1)), mx.zeros((N, 3))], axis=1)
    scales = mx.full((N, 3), 0.25, dtype=mx.float32)
    opacities = mx.ones((N,), dtype=mx.float32)
    colors = mx.array(colors_list, dtype=mx.float32)[None, :, :]  # [1, N, 3]

    viewmat = mx.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]],
        dtype=mx.float32,
    )[None]

    focal = float(width) * 0.8
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=mx.float32
    )[None]

    backgrounds = mx.ones((1, 3), dtype=mx.float32) * 0.15

    return dict(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmat=viewmat, K=K, backgrounds=backgrounds,
        width=width, height=height,
    )


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------


def render_camera_models(
    width: int = 256,
    height: int = 256,
) -> dict:
    """Render the same scene with pinhole, fisheye, and orthographic cameras.

    Returns:
        Dict mapping camera model name to (image [H,W,3], alphas [H,W,1]).
    """
    scene = _create_scene(width, height)
    results = {}

    for model in ("pinhole", "fisheye", "ortho"):
        t0 = time.perf_counter()
        render_colors, render_alphas, info = rasterization(
            means=scene["means"],
            quats=scene["quats"],
            scales=scene["scales"],
            opacities=scene["opacities"],
            colors=scene["colors"],
            viewmats=scene["viewmat"],
            Ks=scene["K"],
            width=width,
            height=height,
            backgrounds=scene["backgrounds"],
            render_mode="RGB",
            sh_degree=None,
            camera_model=model,
        )
        mx.eval(render_colors, render_alphas)
        elapsed = time.perf_counter() - t0

        img = render_colors[0]
        alphas = render_alphas[0]
        results[model] = (img, alphas)

        print(f"  {model:10s}: rendered in {elapsed*1000:.1f} ms, "
              f"alpha range [{float(mx.min(alphas)):.3f}, {float(mx.max(alphas)):.3f}]")

    return results


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_image(img: mx.array, path: str) -> None:
    """Save an MLX image array [H, W, 3] as PNG."""
    from PIL import Image

    arr = np.array(img)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    print("Rendering scene with 3 camera models...")
    results = render_camera_models()

    for model, (img, _) in results.items():
        path = os.path.join(out_dir, f"02_camera_{model}.png")
        save_image(img, path)
        print(f"  Saved: {path}")

    # Also create a combined side-by-side image
    from PIL import Image

    images = []
    for model in ("pinhole", "fisheye", "ortho"):
        arr = np.clip(np.array(results[model][0]) * 255, 0, 255).astype(np.uint8)
        images.append(arr)
    combined = np.concatenate(images, axis=1)
    combined_path = os.path.join(out_dir, "02_camera_comparison.png")
    Image.fromarray(combined).save(combined_path)
    print(f"  Saved combined: {combined_path}")
