#!/usr/bin/env python3
"""Visualize how Spherical Harmonics produce view-dependent colors.

Creates a single large Gaussian with SH degree-3 coefficients and
renders it from 4 different viewpoints to show how color changes
with viewing angle.

Usage:
    python examples/03_spherical_harmonics.py
"""

import math
import os
import time

import mlx.core as mx
import numpy as np

from gsplat_mlx.rendering import rasterization


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------


def render_sh_views(
    width: int = 256,
    height: int = 256,
    sh_degree: int = 3,
) -> list:
    """Render a Gaussian with SH coefficients from 4 viewpoints.

    Returns:
        List of 4 tuples: (label, image [H,W,3], alphas [H,W,1]).
    """
    N = 1
    K_sh = (sh_degree + 1) ** 2  # number of SH basis functions

    # Single Gaussian at the origin
    means = mx.zeros((N, 3), dtype=mx.float32)

    # Identity quaternion
    quats = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)

    # Large scale so it fills the view
    scales = mx.full((N, 3), 0.8, dtype=mx.float32)

    # Full opacity
    opacities = mx.ones((N,), dtype=mx.float32)

    # SH coefficients: [N, K, 3] -- set strong view-dependent components
    # Degree 0 (base color): warm gray
    mx.random.seed(42)
    sh_coeffs = mx.zeros((N, K_sh, 3), dtype=mx.float32)

    # Manually set interesting SH coefficients for visible view-dependence
    # Degree 0: base color (gray-ish)
    sh_coeffs_np = np.zeros((N, K_sh, 3), dtype=np.float32)
    sh_coeffs_np[0, 0, :] = [0.5, 0.4, 0.3]   # warm base
    # Degree 1: directional color variation
    sh_coeffs_np[0, 1, :] = [0.0, 0.3, 0.0]    # green from one side
    sh_coeffs_np[0, 2, :] = [0.0, 0.0, 0.4]    # blue from top/bottom
    sh_coeffs_np[0, 3, :] = [0.4, 0.0, 0.0]    # red from another side
    # Degree 2: more complex variation
    sh_coeffs_np[0, 4, :] = [0.2, -0.1, 0.1]
    sh_coeffs_np[0, 5, :] = [-0.1, 0.2, -0.1]
    sh_coeffs_np[0, 6, :] = [0.1, 0.1, 0.2]
    sh_coeffs_np[0, 7, :] = [-0.1, 0.1, 0.2]
    sh_coeffs_np[0, 8, :] = [0.2, 0.1, -0.1]
    # Degree 3: high-frequency variation
    if K_sh >= 16:
        sh_coeffs_np[0, 9, :] = [0.15, -0.1, 0.05]
        sh_coeffs_np[0, 10, :] = [-0.05, 0.15, -0.1]
        sh_coeffs_np[0, 11, :] = [0.1, -0.05, 0.15]

    sh_coeffs = mx.array(sh_coeffs_np)

    # Camera intrinsics
    focal = float(width) * 0.8
    cx, cy = width / 2.0, height / 2.0
    K = mx.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=mx.float32
    )[None]

    background = mx.array([[0.1, 0.1, 0.15]], dtype=mx.float32)

    # 4 viewpoints: front, right, top, and 45-degree diagonal
    cam_dist = 3.0
    viewpoints = [
        ("front", _viewmat_at(0, 0, cam_dist)),
        ("right", _viewmat_at(0, math.pi / 2, cam_dist)),
        ("top", _viewmat_at(math.pi / 2 - 0.01, 0, cam_dist)),
        ("diagonal", _viewmat_at(math.pi / 4, math.pi / 4, cam_dist)),
    ]

    results = []
    for label, viewmat in viewpoints:
        t0 = time.perf_counter()
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            backgrounds=background,
            render_mode="RGB",
            sh_degree=sh_degree,
        )
        mx.eval(render_colors, render_alphas)
        elapsed = time.perf_counter() - t0

        img = render_colors[0]
        alphas = render_alphas[0]
        results.append((label, img, alphas))

        print(f"  {label:10s}: {elapsed*1000:.1f} ms, "
              f"color range [{float(mx.min(img)):.3f}, {float(mx.max(img)):.3f}]")

    return results


def _viewmat_at(elevation: float, azimuth: float, distance: float) -> mx.array:
    """Create a view matrix for a camera orbiting the origin.

    Args:
        elevation: Angle from the XZ plane (radians).
        azimuth: Angle around Y axis (radians).
        distance: Distance from origin.

    Returns:
        View matrix [1, 4, 4].
    """
    # Camera position in world space
    ce, se = math.cos(elevation), math.sin(elevation)
    ca, sa = math.cos(azimuth), math.sin(azimuth)

    cam_x = distance * ce * sa
    cam_y = distance * se
    cam_z = distance * ce * ca

    # Look-at matrix construction
    forward = np.array([-cam_x, -cam_y, -cam_z], dtype=np.float32)
    forward /= np.linalg.norm(forward) + 1e-8

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Handle case when looking straight down
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)

    # World-to-camera rotation: rows are right, up, -forward (OpenGL convention
    # adapted for gsplat where +Z goes into the scene)
    R = np.array([right, -up, -forward], dtype=np.float32)
    t = R @ np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    return mx.array(viewmat[None])


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

    print(f"Rendering SH degree-3 Gaussian from 4 viewpoints...")
    results = render_sh_views()

    for label, img, _ in results:
        path = os.path.join(out_dir, f"03_sh_{label}.png")
        save_image(img, path)
        print(f"  Saved: {path}")

    # Combined 2x2 grid
    from PIL import Image

    grid_images = []
    for _, img, _ in results:
        arr = np.clip(np.array(img) * 255, 0, 255).astype(np.uint8)
        grid_images.append(arr)

    top = np.concatenate(grid_images[:2], axis=1)
    bottom = np.concatenate(grid_images[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    grid_path = os.path.join(out_dir, "03_sh_comparison.png")
    Image.fromarray(grid).save(grid_path)
    print(f"  Saved grid: {grid_path}")
