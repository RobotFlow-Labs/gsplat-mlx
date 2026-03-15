"""Shared test infrastructure for gsplat-mlx.

Provides:
- check_all_close: compare mx.array / np.ndarray / torch.Tensor with diagnostics
- make_gaussians: generate synthetic Gaussian parameters
- make_camera_intrinsics: generate camera intrinsic matrix K
- make_view_matrix: generate camera extrinsic (view) matrix
- Fixtures for common test configurations
"""

import math

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def check_all_close(
    mlx_result,
    reference,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: str = "",
):
    """Compare an MLX result against a reference (mx.array, np.ndarray, or torch.Tensor).

    Converts everything to numpy for comparison and provides detailed
    diagnostics on failure including max absolute error, location, and shapes.

    Args:
        mlx_result: The MLX computation result.
        reference: The reference value to compare against.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Optional message prefix for assertion errors.

    Raises:
        AssertionError: If arrays are not close within tolerances.
    """
    # Convert mlx_result to numpy
    if isinstance(mlx_result, mx.array):
        mx.eval(mlx_result)
        a = np.array(mlx_result)
    elif isinstance(mlx_result, np.ndarray):
        a = mlx_result
    else:
        # Assume torch.Tensor
        a = mlx_result.detach().cpu().numpy()

    # Convert reference to numpy
    if isinstance(reference, mx.array):
        mx.eval(reference)
        b = np.array(reference)
    elif isinstance(reference, np.ndarray):
        b = reference
    else:
        # Assume torch.Tensor
        b = reference.detach().cpu().numpy()

    # Shape check
    assert a.shape == b.shape, (
        f"{msg}Shape mismatch: mlx_result.shape={a.shape} vs reference.shape={b.shape}"
    )

    # Compute errors
    abs_diff = np.abs(a - b)
    max_abs_err = float(np.max(abs_diff))
    max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    mean_abs_err = float(np.mean(abs_diff))

    close = np.allclose(a, b, atol=atol, rtol=rtol)
    if not close:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}Arrays not close (atol={atol}, rtol={rtol}).\n"
            f"  Shape: {a.shape}\n"
            f"  Max abs error: {max_abs_err:.2e} at index {max_abs_idx}\n"
            f"  Mean abs error: {mean_abs_err:.2e}\n"
            f"  mlx_result[{max_abs_idx}] = {a[max_abs_idx]}\n"
            f"  reference[{max_abs_idx}] = {b[max_abs_idx]}"
        )


# ---------------------------------------------------------------------------
# Gaussian parameter generators
# ---------------------------------------------------------------------------

def make_gaussians(N: int = 100, sh_degree: int = 3, seed: int = 42) -> dict:
    """Generate synthetic 3D Gaussian parameters as mx.arrays.

    Args:
        N: Number of Gaussians.
        sh_degree: Degree of spherical harmonics (0-3).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - means: (N, 3) float32 — Gaussian centers
            - quats: (N, 4) float32 — unit quaternions (wxyz)
            - scales: (N, 3) float32 — log-scales
            - opacities: (N,) float32 — sigmoid pre-activations
            - sh_coeffs: (N, K, 3) float32 — SH coefficients
              where K = (sh_degree + 1)^2
    """
    np.random.seed(seed)
    K = (sh_degree + 1) ** 2

    # Means: random positions in [-5, 5]
    means = mx.array(np.random.uniform(-5.0, 5.0, (N, 3)).astype(np.float32))

    # Quaternions: random unit quaternions (wxyz convention)
    raw_quats = np.random.randn(N, 4).astype(np.float32)
    raw_quats /= np.linalg.norm(raw_quats, axis=-1, keepdims=True)
    quats = mx.array(raw_quats)

    # Scales: log-space scales
    scales = mx.array(np.random.uniform(-3.0, 1.0, (N, 3)).astype(np.float32))

    # Opacities: sigmoid pre-activations (0 maps to 0.5 opacity)
    opacities = mx.array(np.random.uniform(-2.0, 2.0, (N,)).astype(np.float32))

    # SH coefficients
    sh_coeffs = mx.array(np.random.randn(N, K, 3).astype(np.float32) * 0.1)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def make_camera_intrinsics(
    width: int = 640,
    height: int = 480,
    fx: float = 500.0,
    fy: float = 500.0,
) -> mx.array:
    """Create a 3x3 camera intrinsic matrix K.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).

    Returns:
        mx.array of shape (3, 3) with float32 dtype.
    """
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return mx.array(K)


def make_view_matrix(
    eye: tuple = (0.0, 0.0, 5.0),
    target: tuple = (0.0, 0.0, 0.0),
    up: tuple = (0.0, 1.0, 0.0),
) -> mx.array:
    """Create a 4x4 view (world-to-camera) matrix using look-at convention.

    Args:
        eye: Camera position in world coordinates.
        target: Point the camera looks at.
        up: Up direction vector.

    Returns:
        mx.array of shape (4, 4) with float32 dtype.
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    # Build 4x4 view matrix (OpenGL convention: camera looks down -Z)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)

    return mx.array(view)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussians_small():
    """10 Gaussians for quick unit tests."""
    return make_gaussians(N=10, sh_degree=3, seed=42)


@pytest.fixture
def gaussians_medium():
    """100 Gaussians for standard tests."""
    return make_gaussians(N=100, sh_degree=3, seed=42)


@pytest.fixture
def gaussians_large():
    """10,000 Gaussians for stress / benchmark tests."""
    return make_gaussians(N=10000, sh_degree=3, seed=42)


@pytest.fixture
def camera_64x64():
    """Small 64x64 camera for fast tests."""
    return make_camera_intrinsics(width=64, height=64, fx=50.0, fy=50.0)


@pytest.fixture
def camera_640x480():
    """Standard 640x480 camera."""
    return make_camera_intrinsics(width=640, height=480, fx=500.0, fy=500.0)
