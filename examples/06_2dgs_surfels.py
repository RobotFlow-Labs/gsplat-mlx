#!/usr/bin/env python3
"""Compare 3DGS (ellipsoids) vs 2DGS (surfels) rendering.

Renders the same set of Gaussians using both the classic 3DGS pipeline
and the 2DGS surfel pipeline, showing the visual difference and the
normal map output unique to 2DGS.

Usage:
    python examples/06_2dgs_surfels.py
"""

import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization, rasterization_2dgs


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------


def render_3dgs_vs_2dgs(
    width: int = 256,
    height: int = 256,
) -> dict:
    """Render a scene with both 3DGS and 2DGS pipelines.

    Returns:
        Dict with keys:
            '3dgs_rgb' [H,W,3], '3dgs_alphas' [H,W,1],
            '2dgs_rgb' [H,W,3], '2dgs_alphas' [H,W,1],
            '2dgs_normals' [H,W,3].
    """
    # -- Scene: a ring of Gaussians --
    import math

    positions = []
    colors_list = []
    quats_list = []
    scales_list = []
    N_ring = 12

    for i in range(N_ring):
        angle = 2 * math.pi * i / N_ring
        x = 1.0 * math.cos(angle)
        y = 1.0 * math.sin(angle)
        z = 0.0
        positions.append([x, y, z])

        # Color varies around the ring
        t = i / N_ring
        colors_list.append([
            0.5 + 0.5 * math.cos(2 * math.pi * t),
            0.5 + 0.5 * math.cos(2 * math.pi * (t + 1/3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (t + 2/3)),
        ])

        # Orient each Gaussian to face outward from center
        # Rotation around Z by the ring angle
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        quats_list.append([qw, 0.0, 0.0, qz])

        # Slight anisotropy for visual interest
        scales_list.append([0.25, 0.15, 0.15])

    # Add a center Gaussian
    positions.append([0.0, 0.0, 0.0])
    colors_list.append([0.9, 0.9, 0.9])
    quats_list.append([1.0, 0.0, 0.0, 0.0])
    scales_list.append([0.3, 0.3, 0.3])

    N = len(positions)
    means = mx.array(positions, dtype=mx.float32)
    quats = mx.array(quats_list, dtype=mx.float32)
    scales = mx.array(scales_list, dtype=mx.float32)
    opacities = mx.full((N,), 0.9, dtype=mx.float32)
    colors = mx.array(colors_list, dtype=mx.float32)[None, :, :]  # [1, N, 3]

    # -- Camera --
    viewmat = mx.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]],
        dtype=mx.float32,
    )[None]

    focal = float(width) * 0.9
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=mx.float32
    )[None]

    backgrounds_rgb = mx.ones((1, 3), dtype=mx.float32) * 0.1

    # -- Render 3DGS --
    print("Rendering with 3DGS (classic ellipsoids)...")
    t0 = time.perf_counter()
    render_3d, alphas_3d, info_3d = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        backgrounds=backgrounds_rgb,
        render_mode="RGB",
        sh_degree=None,
    )
    mx.eval(render_3d, alphas_3d)
    elapsed_3d = time.perf_counter() - t0
    print(f"  3DGS: {elapsed_3d*1000:.1f} ms")

    # -- Render 2DGS --
    print("Rendering with 2DGS (surfels)...")
    t1 = time.perf_counter()
    render_2d, alphas_2d, normals_2d, info_2d = rasterization_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        backgrounds=backgrounds_rgb,
        sh_degree=None,
    )
    mx.eval(render_2d, alphas_2d, normals_2d)
    elapsed_2d = time.perf_counter() - t1
    print(f"  2DGS: {elapsed_2d*1000:.1f} ms")

    rgb_3d = render_3d[0]
    rgb_2d = render_2d[0]
    normals = normals_2d[0]

    print(f"\n  3DGS color range: [{float(mx.min(rgb_3d)):.3f}, {float(mx.max(rgb_3d)):.3f}]")
    print(f"  2DGS color range: [{float(mx.min(rgb_2d)):.3f}, {float(mx.max(rgb_2d)):.3f}]")
    print(f"  2DGS normal range: [{float(mx.min(normals)):.3f}, {float(mx.max(normals)):.3f}]")

    return {
        "3dgs_rgb": rgb_3d,
        "3dgs_alphas": alphas_3d[0],
        "2dgs_rgb": rgb_2d,
        "2dgs_alphas": alphas_2d[0],
        "2dgs_normals": normals,
    }


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


def normal_to_rgb(normals: mx.array) -> np.ndarray:
    """Convert normal map [H,W,3] to RGB visualization."""
    n = np.array(normals)
    return ((n * 0.5) + 0.5).clip(0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    results = render_3dgs_vs_2dgs()

    save_image(results["3dgs_rgb"], os.path.join(out_dir, "06_3dgs_rgb.png"))
    save_image(results["2dgs_rgb"], os.path.join(out_dir, "06_2dgs_rgb.png"))

    # Normal map from 2DGS
    normal_rgb = normal_to_rgb(results["2dgs_normals"])
    from PIL import Image
    arr = np.clip(normal_rgb * 255, 0, 255).astype(np.uint8)
    path = os.path.join(out_dir, "06_2dgs_normals.png")
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")

    # Combined comparison
    rgb_3d = np.clip(np.array(results["3dgs_rgb"]) * 255, 0, 255).astype(np.uint8)
    rgb_2d = np.clip(np.array(results["2dgs_rgb"]) * 255, 0, 255).astype(np.uint8)
    normal_img = np.clip(normal_rgb * 255, 0, 255).astype(np.uint8)
    combined = np.concatenate([rgb_3d, rgb_2d, normal_img], axis=1)
    combined_path = os.path.join(out_dir, "06_3dgs_vs_2dgs.png")
    Image.fromarray(combined).save(combined_path)
    print(f"  Saved combined: {combined_path}")
