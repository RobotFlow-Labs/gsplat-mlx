"""Synthetic test scene generators for training validation.

Each generator returns a dict containing:
- Gaussian parameters as mx.arrays (means, quats, scales, opacities, sh_coeffs)
- target_image: [H, W, 3] the image to reconstruct
- viewmat: [1, 4, 4] camera view matrix
- K: [1, 3, 3] camera intrinsics
- width, height: image dimensions

These scenes are used to validate end-to-end training convergence.
Gaussians are initialized in front of the camera with reasonable parameters
so that training can start immediately.
"""

import math

import mlx.core as mx
import numpy as np


def _make_camera(
    width: int,
    height: int,
    focal: float = 50.0,
    cam_z: float = 5.0,
) -> tuple:
    """Create a simple camera looking down the -Z axis.

    Returns (viewmat, K) as mx.arrays with batch dimension [1, ...].
    """
    # Intrinsics
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[focal, 0.0, cx],
         [0.0, focal, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # View matrix: camera at (0, 0, cam_z) looking at origin
    # world-to-camera: translate by -cam_z in Z
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[2, 3] = cam_z  # camera at +Z, objects at origin

    return mx.array(viewmat[None]), mx.array(K[None])


def _init_gaussians(
    N: int,
    spread: float,
    scale_val: float,
    seed: int,
) -> dict:
    """Initialize N Gaussians centered around the origin on the XY plane.

    Args:
        N: Number of Gaussians.
        spread: Spatial spread of Gaussian centers in XY.
        scale_val: Log-scale value for all Gaussians.
        seed: Random seed.

    Returns:
        Dict with means, quats, scales, opacities, sh_coeffs.
    """
    np.random.seed(seed)

    # Means: distributed on XY plane near origin, small Z variation
    means_np = np.zeros((N, 3), dtype=np.float32)
    means_np[:, 0] = np.random.uniform(-spread, spread, N)
    means_np[:, 1] = np.random.uniform(-spread, spread, N)
    means_np[:, 2] = np.random.uniform(-0.1, 0.1, N)

    # Quaternions: identity (no rotation)
    quats_np = np.zeros((N, 4), dtype=np.float32)
    quats_np[:, 0] = 1.0  # w=1, x=y=z=0

    # Scales: uniform in log space
    scales_np = np.full((N, 3), scale_val, dtype=np.float32)

    # Opacities: sigmoid pre-activations, start at ~0.5 opacity (pre-sigmoid = 0)
    opacities_np = np.zeros((N,), dtype=np.float32)

    # SH coefficients: degree 0 only (1 coefficient per channel)
    # Initialize with small random values -- training will optimize these
    sh_coeffs_np = np.random.randn(N, 1, 3).astype(np.float32) * 0.1

    return {
        "means": mx.array(means_np),
        "quats": mx.array(quats_np),
        "scales": mx.array(scales_np),
        "opacities": mx.array(opacities_np),
        "sh_coeffs": mx.array(sh_coeffs_np),
    }


def create_solid_color_scene(
    N: int = 500,
    width: int = 64,
    height: int = 64,
    color: tuple = (0.8, 0.2, 0.1),
    seed: int = 42,
) -> dict:
    """Create a scene of Gaussians forming a solid-color blob.

    The target image is a uniform color field. Gaussians are initialized
    spread across the field of view.

    Args:
        N: Number of Gaussians.
        width: Image width.
        height: Image height.
        color: Target RGB color (each in [0, 1]).
        seed: Random seed.

    Returns:
        Dict with Gaussian parameters, target_image, viewmat, K, width, height.
    """
    viewmat, K = _make_camera(width, height)

    # Target: uniform color image
    target = mx.ones((height, width, 3), dtype=mx.float32)
    target = target * mx.array(list(color), dtype=mx.float32)

    params = _init_gaussians(N, spread=2.0, scale_val=-2.0, seed=seed)

    return {
        **params,
        "target_image": target,
        "viewmat": viewmat,
        "K": K,
        "width": width,
        "height": height,
    }


def create_gradient_scene(
    N: int = 500,
    width: int = 64,
    height: int = 64,
    seed: int = 42,
) -> dict:
    """Create a scene targeting a horizontal gradient image.

    Left side is dark, right side is bright. This tests whether the
    optimizer can produce spatially varying color.

    Args:
        N: Number of Gaussians.
        width: Image width.
        height: Image height.
        seed: Random seed.

    Returns:
        Dict with Gaussian parameters, target_image, viewmat, K, width, height.
    """
    viewmat, K = _make_camera(width, height)

    # Target: horizontal gradient from 0 to 1
    gradient = mx.arange(width, dtype=mx.float32) / max(width - 1, 1)
    # Shape: [1, W, 1] -> broadcast to [H, W, 3]
    gradient = mx.broadcast_to(
        gradient.reshape(1, width, 1),
        (height, width, 3),
    )
    target = gradient

    params = _init_gaussians(N, spread=2.5, scale_val=-2.0, seed=seed)

    return {
        **params,
        "target_image": target,
        "viewmat": viewmat,
        "K": K,
        "width": width,
        "height": height,
    }


def create_checkerboard_scene(
    N: int = 1000,
    width: int = 64,
    height: int = 64,
    tile_size: int = 8,
    seed: int = 42,
) -> dict:
    """Create a scene targeting a checkerboard pattern.

    Alternating black and white squares. This is a harder optimization
    target that tests spatial precision.

    Args:
        N: Number of Gaussians.
        width: Image width.
        height: Image height.
        tile_size: Size of each checker square in pixels.
        seed: Random seed.

    Returns:
        Dict with Gaussian parameters, target_image, viewmat, K, width, height.
    """
    viewmat, K = _make_camera(width, height)

    # Target: checkerboard
    # Use numpy for index computation (not in differentiable path)
    rows = np.arange(height)
    cols = np.arange(width)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    checker = ((row_grid // tile_size) + (col_grid // tile_size)) % 2
    checker_img = checker.astype(np.float32)
    # Expand to 3 channels
    checker_img = np.stack([checker_img] * 3, axis=-1)
    target = mx.array(checker_img)

    params = _init_gaussians(N, spread=3.0, scale_val=-2.5, seed=seed)

    return {
        **params,
        "target_image": target,
        "viewmat": viewmat,
        "K": K,
        "width": width,
        "height": height,
    }
