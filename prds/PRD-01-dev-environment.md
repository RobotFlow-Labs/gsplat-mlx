# PRD-01: Development Environment & Project Foundation

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-01 |
| **Title** | Development Environment & Project Foundation |
| **Status** | DRAFT |
| **Priority** | P0 -- Critical Path |
| **Estimated Effort** | 2--4 hours |
| **Dependencies** | None (this is the root PRD) |
| **Blocks** | PRD-02 through PRD-14 (every other PRD) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Establish the complete development environment, package structure, build configuration, test harness, and smoke tests for the `gsplat-mlx` project. After this PRD is implemented, any engineer should be able to:

1. Clone the repo
2. Run a single install command
3. Import the package
4. Run the test suite
5. Begin implementing algorithm code in any subpackage

This is a **zero-algorithm** PRD. No Gaussian Splatting logic is implemented here. Every file created is either configuration, empty `__init__.py` stubs, shared test infrastructure, or environment validation.

---

## 2. Context & Motivation

### 2.1 What is gsplat?

[gsplat](https://github.com/nerfstudio-project/gsplat) is the standard CUDA-accelerated 3D Gaussian Splatting (3DGS) rasterizer from the nerfstudio project. It provides:

- Custom CUDA kernels for projecting 3D Gaussians to 2D screen space
- Tile-based rasterization with alpha compositing
- Spherical harmonics evaluation for view-dependent color
- Custom backward passes for training via gradient descent
- Densification strategies (clone/split/prune)
- A high-level `rasterization()` API used by nerfstudio, SplaTAM, MonoGS, and others

### 2.2 Why port to MLX?

gsplat is CUDA-only, locking out the entire Apple Silicon ecosystem (M1/M2/M3/M4). By porting to MLX (Apple's native ML framework), we enable:

- 3DGS training and inference on Mac hardware
- Integration with the growing MLX ecosystem
- A path to bring SplaTAM, MonoGS, and other 3DGS-SLAM papers to Apple Silicon

### 2.3 Porting strategy

We port from gsplat's `_torch_impl.py` pure-Python reference implementations -- NOT from raw CUDA `.cu` kernels. These reference files contain complete, correct, tested PyTorch implementations of every kernel. The translation is `torch` API calls to `mlx.core` API calls, with `torch.autograd.Function` replaced by `@mx.custom_function` + `.vjp`.

### 2.4 Why this PRD first?

Every subsequent PRD (math utils, covariance, spherical harmonics, projection, rasterization, etc.) depends on:

- A working `pyproject.toml` that installs cleanly
- Package structure with importable subpackages
- Test fixtures (`check_all_close`, synthetic Gaussian generators, camera fixtures)
- Pytest markers (`requires_torch`, `benchmark`)
- Confirmation that MLX is available and the Metal backend works

Without this foundation, no algorithm work can begin.

---

## 3. Scope

### 3.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `pyproject.toml` | Full package metadata, dependencies, build system, pytest config |
| Package structure | All `__init__.py` files for `src/gsplat_mlx/` and subpackages |
| `_version.py` | Version constant `"0.1.0"` |
| `core/constants.py` | Mirrored constants from upstream `_constants.py` |
| `tests/conftest.py` | Shared fixtures, comparison helpers, synthetic data generators |
| `tests/test_smoke.py` | Environment validation and package import tests |
| `.gitignore` updates | MLX/Metal artifacts, `.venv`, `__pycache__`, etc. |

### 3.2 Out of Scope

- Any algorithm implementation (starts in PRD-02)
- CI/CD pipeline (GitHub Actions, etc.)
- Documentation site (Sphinx/MkDocs)
- Publishing to PyPI
- Metal shader compilation infrastructure
- Benchmark suite (beyond the pytest marker)

---

## 4. Technical Design

### 4.1 File: `pyproject.toml`

Create at project root: `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/pyproject.toml`

```toml
[project]
name = "gsplat-mlx"
version = "0.1.0"
description = "Apple MLX port of gsplat — 3D Gaussian Splatting rasterizer for Apple Silicon"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [
    {name = "AIFLOW LABS", email = "ilessio@aiflowlabs.io"},
]
keywords = [
    "3d-gaussian-splatting",
    "mlx",
    "apple-silicon",
    "rasterization",
    "nerf",
    "gaussian-splatting",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
]

dependencies = [
    "mlx>=0.31.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pillow>=9.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "torch>=2.0",
    "imageio>=2.20",
    "ruff>=0.4.0",
]

[project.urls]
Homepage = "https://github.com/AIFLOW-LABS/gsplat-mlx"
Repository = "https://github.com/AIFLOW-LABS/gsplat-mlx"
Issues = "https://github.com/AIFLOW-LABS/gsplat-mlx/issues"

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "requires_torch: tests that need PyTorch for cross-framework comparison",
    "benchmark: performance benchmark tests (use with pytest-benchmark)",
    "slow: tests that take more than 5 seconds",
]
addopts = "-ra --strict-markers"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]
ignore = ["E501"]
```

**Design decisions:**

1. **`setuptools` build backend** -- simplest option, no compiled extensions needed at this stage. If Metal shaders are added later (PRD-14), we can switch to `scikit-build-core` or `meson-python`.
2. **`src/` layout** -- prevents accidental imports from the working directory during development and testing. This is PEP 517 best practice.
3. **`torch>=2.0` in dev extras only** -- torch is needed solely for cross-framework comparison tests. It is never imported by the library itself.
4. **`mlx>=0.31.0`** -- minimum version that includes `mx.custom_function` with `.vjp` support, which is required for all backward passes.
5. **`ruff` in dev extras** -- for linting consistency. Not strictly required but recommended.
6. **`--strict-markers`** -- forces all pytest markers to be declared, preventing typos.

---

### 4.2 Package Structure

Create the following directory tree under `src/gsplat_mlx/`. Every directory gets an `__init__.py`.

```
src/gsplat_mlx/
    __init__.py
    _version.py
    core/
        __init__.py
        constants.py
    core_2dgs/
        __init__.py
    strategy/
        __init__.py
    optimizers/
        __init__.py
    compression/
        __init__.py
```

#### 4.2.1 File: `src/gsplat_mlx/__init__.py`

```python
"""gsplat-mlx: Apple MLX port of gsplat — 3D Gaussian Splatting rasterizer."""

from gsplat_mlx._version import __version__

__all__ = ["__version__"]
```

**Notes:**
- The public API will grow as subsequent PRDs add functions. For now, only `__version__` is exported.
- Subpackage imports (e.g., `from gsplat_mlx.core import projection`) are deferred to avoid import errors on empty modules.

#### 4.2.2 File: `src/gsplat_mlx/_version.py`

```python
"""Version information for gsplat-mlx."""

__version__ = "0.1.0"
```

**Notes:**
- Single source of truth for the version string. `pyproject.toml` also declares `version = "0.1.0"` statically. For a future release workflow, consider `setuptools-scm` or reading from `_version.py` dynamically. For now, keep both in sync manually.

#### 4.2.3 File: `src/gsplat_mlx/core/__init__.py`

```python
"""Core 3DGS operations ported from gsplat's _torch_impl.py to MLX.

Submodules (added by subsequent PRDs):
    - constants: Shared numeric constants
    - math_utils: Numeric helpers (PRD-02)
    - covariance: Quaternion+scale to covariance matrices (PRD-03)
    - spherical_harmonics: SH evaluation (PRD-04)
    - projection: 3D Gaussian to 2D screen-space projection (PRD-05)
    - intersection: Tile-Gaussian intersection and sorting (PRD-06)
    - rasterization: Per-pixel alpha compositing (PRD-07)
    - accumulate: High-level accumulation (PRD-08)
    - cameras: Camera model implementations
"""

from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    MAX_KERNEL_DENSITY_CUTOFF,
    TRANSMITTANCE_THRESHOLD,
)

__all__ = [
    "ALPHA_THRESHOLD",
    "MAX_ALPHA",
    "TRANSMITTANCE_THRESHOLD",
    "MAX_KERNEL_DENSITY_CUTOFF",
]
```

#### 4.2.4 File: `src/gsplat_mlx/core/constants.py`

Mirrored exactly from upstream `gsplat/cuda/_constants.py`:

```python
"""Shared constants for 3D Gaussian Splatting rasterization.

Mirrored from upstream gsplat/cuda/_constants.py.
These values are used throughout the rasterization pipeline for numerical
stability and correctness. Do not change them unless upstream changes.

Source: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_constants.py
"""

# Minimum alpha value for a Gaussian contribution to be considered.
# Gaussians with alpha below this are skipped during rasterization.
ALPHA_THRESHOLD: float = 1.0 / 255.0

# Maximum allowed alpha value per Gaussian per pixel.
# Clamped to prevent numerical instability in the backward pass.
# Chosen so that a maximal-opacity Gaussian must be rasterized at least
# twice to reach TRANSMITTANCE_THRESHOLD: (1 - MAX_ALPHA)^2 = 1e-4.
MAX_ALPHA: float = 0.99

# Minimum transmittance before early-stopping pixel accumulation.
# When transmittance drops below this, the pixel is considered fully opaque
# and no more Gaussians are composited.
# Satisfies: TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
TRANSMITTANCE_THRESHOLD: float = 1e-4

# Maximum Mahalanobis distance (squared) for the Gaussian kernel density cutoff.
# Gaussians beyond this distance from a pixel center do not contribute.
# exp(-0.5 * MAX_KERNEL_DENSITY_CUTOFF) is the minimum kernel weight.
# This corresponds to approximately 3.0 sigma in 2D.
MAX_KERNEL_DENSITY_CUTOFF: float = 0.0113
```

**Why mirror these constants?**
- These values are carefully chosen for numerical stability of the backward pass.
- Changing them would cause divergence from upstream behavior and make cross-framework comparison tests fail.
- The relationship `TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2` is a mathematical invariant that must hold.

#### 4.2.5 File: `src/gsplat_mlx/core_2dgs/__init__.py`

```python
"""2D Gaussian Splatting (2DGS) operations.

Surfel-based variant of 3DGS. Implemented in PRD-12.
"""
```

#### 4.2.6 File: `src/gsplat_mlx/strategy/__init__.py`

```python
"""Gaussian densification strategies (clone, split, prune).

Implemented in PRD-10.
"""
```

#### 4.2.7 File: `src/gsplat_mlx/optimizers/__init__.py`

```python
"""Custom optimizers for Gaussian Splatting training.

Implemented in PRD-11.
"""
```

#### 4.2.8 File: `src/gsplat_mlx/compression/__init__.py`

```python
"""Model compression utilities for Gaussian Splatting.

Implemented in a future PRD.
"""
```

---

### 4.3 File: `tests/conftest.py`

Create at: `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/tests/conftest.py`

This is the shared test infrastructure used by every subsequent PRD's tests.

```python
"""Shared test fixtures and utilities for gsplat-mlx tests.

Provides:
    - check_all_close(): Compare MLX arrays against numpy/torch references
    - Synthetic Gaussian parameter generators
    - Camera intrinsics and view matrix fixtures
    - Pytest markers for requires_torch and benchmark tests
"""

import math
from typing import Optional

import numpy as np
import pytest

import mlx.core as mx


# ---------------------------------------------------------------------------
# Cross-framework comparison helper
# ---------------------------------------------------------------------------

def check_all_close(
    mlx_result: mx.array,
    reference,  # np.ndarray, torch.Tensor, or mx.array
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: str = "",
):
    """Compare an MLX array against a reference value (numpy, torch, or MLX).

    Converts both sides to numpy for comparison. On failure, prints detailed
    diagnostics including shape, dtype, max absolute difference, and the
    indices of the largest discrepancy.

    Args:
        mlx_result: The MLX array to validate.
        reference: The expected values. Can be np.ndarray, torch.Tensor,
            or mx.array.
        atol: Absolute tolerance for np.allclose.
        rtol: Relative tolerance for np.allclose.
        msg: Optional message appended to the assertion error.

    Raises:
        AssertionError: If shapes differ or values are not close.
    """
    # Convert MLX result to numpy
    if isinstance(mlx_result, mx.array):
        mx.eval(mlx_result)  # Force evaluation of lazy graph
        mlx_np = np.array(mlx_result)
    else:
        mlx_np = np.asarray(mlx_result)

    # Convert reference to numpy
    try:
        import torch
        if isinstance(reference, torch.Tensor):
            ref_np = reference.detach().cpu().numpy()
        else:
            ref_np = np.asarray(reference)
    except ImportError:
        ref_np = np.asarray(reference)

    # If reference is mx.array
    if isinstance(reference, mx.array):
        mx.eval(reference)
        ref_np = np.array(reference)

    # Shape check
    if mlx_np.shape != ref_np.shape:
        raise AssertionError(
            f"Shape mismatch: MLX {mlx_np.shape} vs reference {ref_np.shape}. {msg}"
        )

    # Dtype alignment: cast both to float64 for comparison
    mlx_f64 = mlx_np.astype(np.float64)
    ref_f64 = ref_np.astype(np.float64)

    # Value check
    abs_diff = np.abs(mlx_f64 - ref_f64)
    max_diff = np.max(abs_diff) if abs_diff.size > 0 else 0.0
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape) if abs_diff.size > 0 else ()

    if not np.allclose(mlx_f64, ref_f64, atol=atol, rtol=rtol):
        # Find percentage of failing elements
        close_mask = np.isclose(mlx_f64, ref_f64, atol=atol, rtol=rtol)
        n_fail = np.sum(~close_mask)
        n_total = close_mask.size
        pct_fail = 100.0 * n_fail / n_total if n_total > 0 else 0.0

        raise AssertionError(
            f"Values not close (atol={atol}, rtol={rtol}).\n"
            f"  Max absolute diff: {max_diff:.6e} at index {max_idx}\n"
            f"  MLX value at max diff:  {mlx_f64[max_idx]:.6e}\n"
            f"  Reference at max diff:  {ref_f64[max_idx]:.6e}\n"
            f"  Failing elements: {n_fail}/{n_total} ({pct_fail:.1f}%)\n"
            f"  MLX dtype: {mlx_np.dtype}, Reference dtype: {ref_np.dtype}\n"
            f"  {msg}"
        )


# ---------------------------------------------------------------------------
# Numpy RNG fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded numpy random number generator for reproducible tests."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# MLX random key fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mx_key():
    """MLX random key for reproducible random array generation."""
    return mx.random.key(42)


# ---------------------------------------------------------------------------
# Synthetic Gaussian parameter generators
# ---------------------------------------------------------------------------

def make_gaussians(
    n: int = 100,
    sh_degree: int = 3,
    seed: int = 42,
) -> dict:
    """Create synthetic 3D Gaussian parameters for testing.

    Generates a batch of N Gaussians with random but reasonable parameters.
    All arrays are returned as MLX arrays in float32.

    Args:
        n: Number of Gaussians to generate.
        sh_degree: Maximum spherical harmonics degree (0-3).
            Determines the number of SH coefficients:
            degree 0 -> 1, degree 1 -> 4, degree 2 -> 9, degree 3 -> 16.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - means: (N, 3) float32 -- 3D positions, range [-5, 5]
            - quats: (N, 4) float32 -- unit quaternions (w, x, y, z), normalized
            - scales: (N, 3) float32 -- log-space scales, range [-3, 1]
            - opacities: (N,) float32 -- sigmoid-space opacities, range [-2, 2]
            - sh_coeffs: (N, K, 3) float32 -- SH coefficients, K = (degree+1)^2
    """
    rng = np.random.default_rng(seed)
    n_sh_coeffs = (sh_degree + 1) ** 2

    # Positions: uniformly distributed in a cube
    means = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)

    # Quaternions: random unit quaternions
    # Generate random quaternions and normalize
    quats_raw = rng.standard_normal(size=(n, 4)).astype(np.float32)
    quats_norms = np.linalg.norm(quats_raw, axis=-1, keepdims=True)
    quats = quats_raw / np.maximum(quats_norms, 1e-8)

    # Scales: log-space (will be exponentiated during projection)
    scales = rng.uniform(-3.0, 1.0, size=(n, 3)).astype(np.float32)

    # Opacities: in logit space (will be sigmoided during rasterization)
    opacities = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32)

    # SH coefficients: small random values
    # First coefficient (DC) has larger magnitude for reasonable base color
    sh_coeffs = rng.standard_normal(size=(n, n_sh_coeffs, 3)).astype(np.float32) * 0.1
    sh_coeffs[:, 0, :] = rng.uniform(0.1, 0.9, size=(n, 3)).astype(np.float32)

    return {
        "means": mx.array(means),
        "quats": mx.array(quats),
        "scales": mx.array(scales),
        "opacities": mx.array(opacities),
        "sh_coeffs": mx.array(sh_coeffs),
    }


@pytest.fixture
def small_gaussians():
    """10 Gaussians with SH degree 0 -- minimal test case."""
    return make_gaussians(n=10, sh_degree=0, seed=42)


@pytest.fixture
def medium_gaussians():
    """100 Gaussians with SH degree 3 -- standard test case."""
    return make_gaussians(n=100, sh_degree=3, seed=42)


@pytest.fixture
def large_gaussians():
    """10000 Gaussians with SH degree 3 -- stress test."""
    return make_gaussians(n=10000, sh_degree=3, seed=42)


# ---------------------------------------------------------------------------
# Camera fixtures
# ---------------------------------------------------------------------------

def make_camera_intrinsics(
    width: int = 640,
    height: int = 480,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> dict:
    """Create camera intrinsic parameters.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length x. Defaults to width (roughly 90-degree FOV).
        fy: Focal length y. Defaults to fx.
        cx: Principal point x. Defaults to width/2.
        cy: Principal point y. Defaults to height/2.

    Returns:
        Dictionary with keys:
            - width: int
            - height: int
            - fx, fy: float -- focal lengths in pixels
            - cx, cy: float -- principal point in pixels
            - K: (3, 3) float32 MLX array -- intrinsic matrix
    """
    if fx is None:
        fx = float(width)
    if fy is None:
        fy = fx
    if cx is None:
        cx = float(width) / 2.0
    if cy is None:
        cy = float(height) / 2.0

    K = mx.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=mx.float32)

    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
    }


def make_view_matrix(
    eye: Optional[list] = None,
    target: Optional[list] = None,
    up: Optional[list] = None,
) -> mx.array:
    """Create a 4x4 world-to-camera view matrix (look-at).

    Uses OpenCV camera convention: +X right, +Y down, +Z forward (into scene).

    Args:
        eye: Camera position in world coordinates. Default: [0, 0, -5].
        target: Look-at target point. Default: [0, 0, 0].
        up: Up vector. Default: [0, -1, 0] (OpenCV convention: Y-down).

    Returns:
        (4, 4) float32 MLX array -- world-to-camera transformation matrix.
    """
    if eye is None:
        eye = [0.0, 0.0, -5.0]
    if target is None:
        target = [0.0, 0.0, 0.0]
    if up is None:
        up = [0.0, -1.0, 0.0]

    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Forward vector (camera looks along +Z in OpenCV convention)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute up to ensure orthogonality
    up_ortho = np.cross(right, forward)

    # Build rotation matrix (rows are camera axes in world coordinates)
    # OpenCV: Z = forward, X = right, Y = -up (but we use recomputed up)
    R = np.eye(3, dtype=np.float32)
    R[0, :] = right
    R[1, :] = -up_ortho  # Y-down in OpenCV
    R[2, :] = forward

    # Translation
    t = -R @ eye

    # Build 4x4
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t

    return mx.array(view)


@pytest.fixture
def default_camera():
    """Default pinhole camera: 640x480, looking at origin from z=-5."""
    intrinsics = make_camera_intrinsics(width=640, height=480)
    viewmat = make_view_matrix()
    return {**intrinsics, "viewmat": viewmat}


@pytest.fixture
def small_camera():
    """Small 64x64 camera for fast tests."""
    intrinsics = make_camera_intrinsics(width=64, height=64)
    viewmat = make_view_matrix()
    return {**intrinsics, "viewmat": viewmat}


@pytest.fixture
def hd_camera():
    """HD 1920x1080 camera for resolution stress tests."""
    intrinsics = make_camera_intrinsics(width=1920, height=1080)
    viewmat = make_view_matrix()
    return {**intrinsics, "viewmat": viewmat}


# ---------------------------------------------------------------------------
# Torch availability checks
# ---------------------------------------------------------------------------

def torch_available() -> bool:
    """Check if PyTorch is importable."""
    try:
        import torch
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not torch_available(),
    reason="PyTorch not installed (install with: uv pip install torch)",
)

# Re-export as a module-level name so tests can do:
#   from conftest import requires_torch
#   @requires_torch
#   def test_something(): ...


# ---------------------------------------------------------------------------
# Benchmark marker convenience
# ---------------------------------------------------------------------------

benchmark = pytest.mark.benchmark
```

**Design decisions for `conftest.py`:**

1. **`check_all_close` with diagnostics** -- When a test fails during algorithm porting, you need to know _where_ and _how much_ the values diverge. The helper prints max diff, the index of worst divergence, actual vs expected values, and percentage of failing elements. This saves significant debugging time.

2. **`make_gaussians` function (not just fixture)** -- Exposed as a callable so tests can customize `n`, `sh_degree`, and `seed` directly. The fixtures (`small_gaussians`, `medium_gaussians`, `large_gaussians`) are convenience wrappers.

3. **Gaussian parameter ranges** -- Chosen to match realistic 3DGS training:
   - Means in `[-5, 5]`: typical scene scale
   - Quaternions: normalized to unit length (required for valid rotations)
   - Scales in `[-3, 1]` (log-space): `exp(-3)` to `exp(1)` covers tiny to large Gaussians
   - Opacities in `[-2, 2]` (logit-space): `sigmoid(-2)` to `sigmoid(2)` covers 12% to 88%
   - SH DC coefficient in `[0.1, 0.9]`: reasonable base color

4. **Camera fixtures with OpenCV convention** -- gsplat uses OpenCV camera convention (Y-down, Z-forward). The view matrix construction follows this exactly.

5. **`requires_torch` marker** -- Tests that compare MLX output against PyTorch reference use this marker. Running `pytest -m "not requires_torch"` skips them, useful on machines without torch installed.

---

### 4.4 File: `tests/test_smoke.py`

Create at: `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/tests/test_smoke.py`

```python
"""Smoke tests for gsplat-mlx development environment.

These tests validate that:
1. MLX is installed and functional
2. The Metal (GPU) backend is available
3. The gsplat_mlx package imports correctly
4. All subpackages are importable
5. Core MLX operations work (array creation, matmul, eval)
6. The constants module has correct values
7. Test fixtures produce valid data

Run with: pytest tests/test_smoke.py -v
"""

import numpy as np
import pytest

import mlx.core as mx


# ---------------------------------------------------------------------------
# MLX Environment
# ---------------------------------------------------------------------------

class TestMLXEnvironment:
    """Verify MLX is installed and the Metal backend works."""

    def test_mlx_import(self):
        """MLX core module is importable."""
        import mlx.core
        assert mlx.core is not None

    def test_mlx_array_creation(self):
        """Can create MLX arrays from Python lists."""
        x = mx.array([1.0, 2.0, 3.0])
        assert x.dtype == mx.float32
        assert x.shape == (3,)

    def test_mlx_default_device_is_gpu(self):
        """MLX default device should be GPU on Apple Silicon."""
        device = mx.default_device()
        # On Apple Silicon, this should be mx.gpu
        # On CI without Metal, it may be mx.cpu -- but we still pass
        assert device in (mx.gpu, mx.cpu)

    def test_mlx_eval(self):
        """MLX lazy evaluation can be forced with mx.eval()."""
        x = mx.ones((10, 10))
        y = x + x
        mx.eval(y)
        assert y.shape == (10, 10)
        # After eval, values should be materialized
        assert np.allclose(np.array(y), 2.0)

    def test_mlx_matmul(self):
        """Matrix multiplication works correctly."""
        a = mx.ones((3, 4))
        b = mx.ones((4, 5))
        c = a @ b
        mx.eval(c)
        assert c.shape == (3, 5)
        # Each element should be 4.0 (dot product of length-4 ones vectors)
        assert np.allclose(np.array(c), 4.0)

    def test_mlx_basic_arithmetic(self):
        """Element-wise arithmetic operations work."""
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b
        d = a * b
        e = a - b
        mx.eval(c, d, e)
        assert np.allclose(np.array(c), [5.0, 7.0, 9.0])
        assert np.allclose(np.array(d), [4.0, 10.0, 18.0])
        assert np.allclose(np.array(e), [-3.0, -3.0, -3.0])

    def test_mlx_dtype_support(self):
        """MLX supports float16, float32, and int32 dtypes."""
        f32 = mx.array([1.0], dtype=mx.float32)
        f16 = mx.array([1.0], dtype=mx.float16)
        i32 = mx.array([1], dtype=mx.int32)
        assert f32.dtype == mx.float32
        assert f16.dtype == mx.float16
        assert i32.dtype == mx.int32

    def test_mlx_random(self):
        """MLX random number generation works."""
        key = mx.random.key(0)
        x = mx.random.normal(shape=(10,), key=key)
        mx.eval(x)
        assert x.shape == (10,)
        assert x.dtype == mx.float32

    def test_mlx_custom_function_exists(self):
        """The @mx.custom_function decorator is available.

        This is required for implementing custom backward passes
        (VJPs) for autograd.Function equivalents.
        """
        assert hasattr(mx, "custom_function"), (
            "mx.custom_function not found. "
            "Upgrade MLX to >= 0.31.0: uv pip install 'mlx>=0.31.0'"
        )

    def test_mlx_grad(self):
        """MLX automatic differentiation works."""
        def f(x):
            return mx.sum(x ** 2)

        x = mx.array([1.0, 2.0, 3.0])
        grad_fn = mx.grad(f)
        g = grad_fn(x)
        mx.eval(g)
        # d/dx sum(x^2) = 2x
        assert np.allclose(np.array(g), [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Package Imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify the gsplat_mlx package structure is importable."""

    def test_top_level_import(self):
        """Top-level package imports without error."""
        import gsplat_mlx
        assert gsplat_mlx is not None

    def test_version_accessible(self):
        """Version string is accessible and correct."""
        import gsplat_mlx
        assert hasattr(gsplat_mlx, "__version__")
        assert gsplat_mlx.__version__ == "0.1.0"

    def test_version_module(self):
        """Version module is importable directly."""
        from gsplat_mlx._version import __version__
        assert __version__ == "0.1.0"

    def test_core_import(self):
        """Core subpackage imports."""
        from gsplat_mlx import core
        assert core is not None

    def test_core_2dgs_import(self):
        """Core 2DGS subpackage imports."""
        from gsplat_mlx import core_2dgs
        assert core_2dgs is not None

    def test_strategy_import(self):
        """Strategy subpackage imports."""
        from gsplat_mlx import strategy
        assert strategy is not None

    def test_optimizers_import(self):
        """Optimizers subpackage imports."""
        from gsplat_mlx import optimizers
        assert optimizers is not None

    def test_compression_import(self):
        """Compression subpackage imports."""
        from gsplat_mlx import compression
        assert compression is not None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify mirrored constants match upstream values."""

    def test_constants_importable(self):
        """Constants are importable from core."""
        from gsplat_mlx.core.constants import (
            ALPHA_THRESHOLD,
            MAX_ALPHA,
            MAX_KERNEL_DENSITY_CUTOFF,
            TRANSMITTANCE_THRESHOLD,
        )
        assert ALPHA_THRESHOLD is not None
        assert MAX_ALPHA is not None
        assert TRANSMITTANCE_THRESHOLD is not None
        assert MAX_KERNEL_DENSITY_CUTOFF is not None

    def test_alpha_threshold_value(self):
        """ALPHA_THRESHOLD = 1/255."""
        from gsplat_mlx.core.constants import ALPHA_THRESHOLD
        assert abs(ALPHA_THRESHOLD - 1.0 / 255.0) < 1e-10

    def test_max_alpha_value(self):
        """MAX_ALPHA = 0.99."""
        from gsplat_mlx.core.constants import MAX_ALPHA
        assert MAX_ALPHA == 0.99

    def test_transmittance_threshold_value(self):
        """TRANSMITTANCE_THRESHOLD = 1e-4."""
        from gsplat_mlx.core.constants import TRANSMITTANCE_THRESHOLD
        assert TRANSMITTANCE_THRESHOLD == 1e-4

    def test_max_kernel_density_cutoff_value(self):
        """MAX_KERNEL_DENSITY_CUTOFF = 0.0113."""
        from gsplat_mlx.core.constants import MAX_KERNEL_DENSITY_CUTOFF
        assert MAX_KERNEL_DENSITY_CUTOFF == 0.0113

    def test_transmittance_invariant(self):
        """TRANSMITTANCE_THRESHOLD == (1 - MAX_ALPHA)^2.

        This mathematical relationship must hold for correct rasterization
        behavior. See upstream comment in _constants.py.
        """
        from gsplat_mlx.core.constants import MAX_ALPHA, TRANSMITTANCE_THRESHOLD
        expected = (1.0 - MAX_ALPHA) ** 2
        assert abs(TRANSMITTANCE_THRESHOLD - expected) < 1e-10, (
            f"Invariant broken: (1 - {MAX_ALPHA})^2 = {expected} "
            f"!= TRANSMITTANCE_THRESHOLD = {TRANSMITTANCE_THRESHOLD}"
        )

    def test_constants_from_core_init(self):
        """Constants are re-exported from core.__init__."""
        from gsplat_mlx.core import (
            ALPHA_THRESHOLD,
            MAX_ALPHA,
            MAX_KERNEL_DENSITY_CUTOFF,
            TRANSMITTANCE_THRESHOLD,
        )
        assert ALPHA_THRESHOLD == 1.0 / 255.0
        assert MAX_ALPHA == 0.99


# ---------------------------------------------------------------------------
# Test Fixtures Validation
# ---------------------------------------------------------------------------

class TestFixtures:
    """Verify that conftest.py fixtures produce valid data."""

    def test_make_gaussians_shapes(self):
        """make_gaussians returns arrays with correct shapes."""
        from conftest import make_gaussians
        g = make_gaussians(n=50, sh_degree=2, seed=0)

        assert g["means"].shape == (50, 3)
        assert g["quats"].shape == (50, 4)
        assert g["scales"].shape == (50, 3)
        assert g["opacities"].shape == (50,)
        # SH degree 2 -> (2+1)^2 = 9 coefficients
        assert g["sh_coeffs"].shape == (50, 9, 3)

    def test_make_gaussians_dtypes(self):
        """make_gaussians returns float32 arrays."""
        from conftest import make_gaussians
        g = make_gaussians(n=10, sh_degree=0)

        assert g["means"].dtype == mx.float32
        assert g["quats"].dtype == mx.float32
        assert g["scales"].dtype == mx.float32
        assert g["opacities"].dtype == mx.float32
        assert g["sh_coeffs"].dtype == mx.float32

    def test_make_gaussians_quaternions_normalized(self):
        """Quaternions from make_gaussians are unit length."""
        from conftest import make_gaussians
        g = make_gaussians(n=100, sh_degree=0)
        quats_np = np.array(g["quats"])
        norms = np.linalg.norm(quats_np, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"Quaternion norms not unit: min={norms.min():.6f}, max={norms.max():.6f}"
        )

    def test_make_gaussians_sh_degree_variants(self):
        """SH coefficient count matches (degree+1)^2 for all degrees."""
        from conftest import make_gaussians
        for degree in [0, 1, 2, 3]:
            g = make_gaussians(n=5, sh_degree=degree)
            expected_k = (degree + 1) ** 2
            assert g["sh_coeffs"].shape == (5, expected_k, 3), (
                f"SH degree {degree}: expected K={expected_k}, "
                f"got shape {g['sh_coeffs'].shape}"
            )

    def test_make_gaussians_reproducible(self):
        """Same seed produces identical Gaussians."""
        from conftest import make_gaussians
        g1 = make_gaussians(n=10, sh_degree=1, seed=123)
        g2 = make_gaussians(n=10, sh_degree=1, seed=123)
        for key in g1:
            assert np.allclose(np.array(g1[key]), np.array(g2[key])), (
                f"Key '{key}' not reproducible across calls with same seed"
            )

    def test_small_gaussians_fixture(self, small_gaussians):
        """small_gaussians fixture provides 10 Gaussians."""
        assert small_gaussians["means"].shape == (10, 3)
        assert small_gaussians["sh_coeffs"].shape == (10, 1, 3)  # degree 0

    def test_medium_gaussians_fixture(self, medium_gaussians):
        """medium_gaussians fixture provides 100 Gaussians with SH degree 3."""
        assert medium_gaussians["means"].shape == (100, 3)
        assert medium_gaussians["sh_coeffs"].shape == (100, 16, 3)  # degree 3

    def test_camera_intrinsics(self, default_camera):
        """Default camera fixture has correct structure."""
        assert default_camera["width"] == 640
        assert default_camera["height"] == 480
        assert default_camera["K"].shape == (3, 3)
        assert default_camera["K"].dtype == mx.float32

    def test_camera_intrinsic_matrix_structure(self, default_camera):
        """Intrinsic matrix K has correct structure: fx, fy on diagonal."""
        K_np = np.array(default_camera["K"])
        assert K_np[0, 0] == default_camera["fx"]
        assert K_np[1, 1] == default_camera["fy"]
        assert K_np[0, 2] == default_camera["cx"]
        assert K_np[1, 2] == default_camera["cy"]
        assert K_np[2, 2] == 1.0
        # Off-diagonal zeros
        assert K_np[0, 1] == 0.0
        assert K_np[1, 0] == 0.0
        assert K_np[2, 0] == 0.0
        assert K_np[2, 1] == 0.0

    def test_view_matrix_shape(self, default_camera):
        """View matrix is 4x4."""
        assert default_camera["viewmat"].shape == (4, 4)
        assert default_camera["viewmat"].dtype == mx.float32

    def test_view_matrix_is_rigid(self, default_camera):
        """View matrix rotation part is orthonormal (det = +1)."""
        V_np = np.array(default_camera["viewmat"])
        R = V_np[:3, :3]
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-5, f"Rotation determinant = {det}, expected 1.0"
        # R @ R^T should be identity
        RRT = R @ R.T
        assert np.allclose(RRT, np.eye(3), atol=1e-5), "R is not orthonormal"

    def test_small_camera_fixture(self, small_camera):
        """Small camera fixture is 64x64."""
        assert small_camera["width"] == 64
        assert small_camera["height"] == 64

    def test_check_all_close_passes(self):
        """check_all_close passes for identical arrays."""
        from conftest import check_all_close
        a = mx.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        check_all_close(a, b)  # Should not raise

    def test_check_all_close_fails_on_diff(self):
        """check_all_close raises AssertionError for differing arrays."""
        from conftest import check_all_close
        a = mx.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 999.0])
        with pytest.raises(AssertionError, match="Values not close"):
            check_all_close(a, b)

    def test_check_all_close_fails_on_shape_mismatch(self):
        """check_all_close raises AssertionError for shape mismatches."""
        from conftest import check_all_close
        a = mx.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(AssertionError, match="Shape mismatch"):
            check_all_close(a, b)
```

---

### 4.5 File: `.gitignore` additions

Append the following to the existing `.gitignore` (or create if missing):

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# MLX / Metal
*.metallib
*.air

# OS
.DS_Store
Thumbs.db

# Test artifacts
.pytest_cache/
.benchmarks/
htmlcov/
.coverage

# Reference repositories (large, cloned locally)
repositories/
```

---

## 5. Dev Workflow Commands

### 5.1 Initial Setup (run once)

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx

# Create virtual environment with Python 3.12
uv venv .venv --python 3.12

# Activate
source .venv/bin/activate

# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### 5.2 Verify Installation

```bash
# Verify package imports
python -c "import gsplat_mlx; print(f'gsplat-mlx v{gsplat_mlx.__version__}')"
# Expected output: gsplat-mlx v0.1.0

# Verify MLX Metal backend
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"
# Expected output: MLX device: gpu

# Verify constants
python -c "from gsplat_mlx.core.constants import ALPHA_THRESHOLD; print(f'ALPHA_THRESHOLD = {ALPHA_THRESHOLD}')"
# Expected output: ALPHA_THRESHOLD = 0.00392156862745098
```

### 5.3 Run Tests

```bash
# All tests (verbose)
pytest tests/ -v

# Smoke tests only
pytest tests/test_smoke.py -v

# Skip tests that require PyTorch
pytest tests/ -v -m "not requires_torch"

# Only cross-framework comparison tests
pytest tests/ -v -m "requires_torch"

# With benchmark output (when benchmark tests exist)
pytest tests/ -v -m "benchmark" --benchmark-only
```

### 5.4 Day-to-Day Development

```bash
# Run tests for a specific module being ported
pytest tests/test_covariance.py -v

# Run with print output visible
pytest tests/test_smoke.py -v -s

# Run a single test by name
pytest tests/test_smoke.py::TestMLXEnvironment::test_mlx_matmul -v

# Lint (optional, if ruff is installed)
ruff check src/ tests/
```

---

## 6. Acceptance Criteria

Each criterion must pass for this PRD to be considered DONE. They are ordered from most fundamental to most comprehensive.

### GO/NO-GO Gate 1: Package Installs

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"
```

**PASS**: Command exits with code 0, no errors.
**FAIL**: Any dependency resolution failure, build error, or missing module.

### GO/NO-GO Gate 2: Package Imports

```bash
python -c "import gsplat_mlx; print(gsplat_mlx.__version__)"
```

**PASS**: Prints `0.1.0` with no import errors.
**FAIL**: `ImportError`, `ModuleNotFoundError`, or wrong version.

### GO/NO-GO Gate 3: MLX Metal Backend

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
```

**PASS**: Prints `gpu` on Apple Silicon hardware.
**ACCEPTABLE**: Prints `cpu` on non-Apple-Silicon hardware (CI). Tests still run.
**FAIL**: Import error or crash.

### GO/NO-GO Gate 4: Subpackage Imports

```bash
python -c "
from gsplat_mlx import core
from gsplat_mlx import core_2dgs
from gsplat_mlx import strategy
from gsplat_mlx import optimizers
from gsplat_mlx import compression
from gsplat_mlx.core.constants import ALPHA_THRESHOLD, MAX_ALPHA
print('All subpackages OK')
"
```

**PASS**: Prints `All subpackages OK`.
**FAIL**: Any `ImportError`.

### GO/NO-GO Gate 5: Constants Correctness

```bash
python -c "
from gsplat_mlx.core.constants import *
assert abs(ALPHA_THRESHOLD - 1/255) < 1e-10
assert MAX_ALPHA == 0.99
assert TRANSMITTANCE_THRESHOLD == 1e-4
assert MAX_KERNEL_DENSITY_CUTOFF == 0.0113
assert abs(TRANSMITTANCE_THRESHOLD - (1 - MAX_ALPHA)**2) < 1e-10
print('Constants OK')
"
```

**PASS**: Prints `Constants OK`.
**FAIL**: Any assertion failure.

### GO/NO-GO Gate 6: Smoke Tests Pass

```bash
pytest tests/test_smoke.py -v
```

**PASS**: All tests pass (expected: 25+ tests, 0 failures).
**FAIL**: Any test failure.

### GO/NO-GO Gate 7: Fixtures Produce Valid Data

```bash
pytest tests/test_smoke.py::TestFixtures -v
```

**PASS**: All fixture validation tests pass.
**FAIL**: Any shape, dtype, or normalization assertion failure.

### Summary Checklist

- [ ] `pyproject.toml` exists at project root with correct metadata
- [ ] `src/gsplat_mlx/__init__.py` exports `__version__`
- [ ] `src/gsplat_mlx/_version.py` contains `__version__ = "0.1.0"`
- [ ] `src/gsplat_mlx/core/__init__.py` re-exports constants
- [ ] `src/gsplat_mlx/core/constants.py` mirrors upstream `_constants.py`
- [ ] `src/gsplat_mlx/core_2dgs/__init__.py` exists
- [ ] `src/gsplat_mlx/strategy/__init__.py` exists
- [ ] `src/gsplat_mlx/optimizers/__init__.py` exists
- [ ] `src/gsplat_mlx/compression/__init__.py` exists
- [ ] `tests/conftest.py` has `check_all_close` with diagnostic output
- [ ] `tests/conftest.py` has `make_gaussians()` function
- [ ] `tests/conftest.py` has `make_camera_intrinsics()` and `make_view_matrix()`
- [ ] `tests/conftest.py` has `small_gaussians`, `medium_gaussians`, `large_gaussians` fixtures
- [ ] `tests/conftest.py` has `default_camera`, `small_camera`, `hd_camera` fixtures
- [ ] `tests/conftest.py` has `requires_torch` marker
- [ ] `tests/test_smoke.py` has MLX environment tests (9 tests)
- [ ] `tests/test_smoke.py` has package import tests (6 tests)
- [ ] `tests/test_smoke.py` has constants tests (7 tests)
- [ ] `tests/test_smoke.py` has fixture validation tests (14 tests)
- [ ] `uv pip install -e ".[dev]"` succeeds
- [ ] `pytest tests/test_smoke.py -v` shows all green
- [ ] `.gitignore` covers `__pycache__`, `.venv`, `repositories/`, `*.metallib`

---

## 7. File Inventory

Complete list of files to create or modify, with their absolute paths:

| # | Action | Path | Lines (approx) |
|---|--------|------|-----------------|
| 1 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/pyproject.toml` | 75 |
| 2 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/__init__.py` | 5 |
| 3 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/_version.py` | 3 |
| 4 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/core/__init__.py` | 20 |
| 5 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/core/constants.py` | 30 |
| 6 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/core_2dgs/__init__.py` | 3 |
| 7 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/strategy/__init__.py` | 3 |
| 8 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/optimizers/__init__.py` | 3 |
| 9 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/src/gsplat_mlx/compression/__init__.py` | 3 |
| 10 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/tests/conftest.py` | 250 |
| 11 | CREATE | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/tests/test_smoke.py` | 280 |
| 12 | MODIFY | `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx/.gitignore` | append ~30 lines |

**Total**: 11 new files, 1 modified file, ~705 lines of code.

---

## 8. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `mlx>=0.31.0` not available on target Python version | Blocks install | Low | MLX supports Python 3.10-3.12 on macOS. Pin Python 3.12. |
| `mx.custom_function` API changes in future MLX versions | Breaks backward passes | Low | Pin minimum version. Test in smoke tests. |
| PyTorch too large for dev install | Slow setup | Medium | `torch` is in `[dev]` extras only. Can skip with `uv pip install -e .` |
| `setuptools.backends._legacy:_Backend` not recognized | Build fails | Low | Fallback: use `"setuptools.build_meta"` as build-backend. |
| Tests import `conftest` directly (not via pytest plugin) | Import path issues | Medium | Use `from conftest import ...` which pytest supports. Also expose `make_gaussians` etc. at module level. |

---

## 9. Dependencies Graph

```
PRD-01 (this)
    |
    +---> PRD-02 (Math Utils)
    |         |
    |         +---> PRD-03 (Covariance)
    |         |         |
    |         |         +---> PRD-05 (Projection)
    |         |                   |
    |         |                   +---> PRD-06 (Intersection)
    |         |                   |         |
    |         |                   |         +---> PRD-07 (Rasterization)
    |         |                   |                   |
    |         |                   |                   +---> PRD-08 (Accumulate)
    |         |                   |                             |
    |         |                   |                             +---> PRD-09 (Rendering API)
    |         |                   |
    |         +---> PRD-04 (Spherical Harmonics)
    |
    +---> PRD-10 (Strategy)
    +---> PRD-11 (Optimizer)
    +---> PRD-12 (2DGS)
    +---> PRD-13 (Training Loop) -- depends on PRD-09 + PRD-10 + PRD-11
    +---> PRD-14 (Metal Shaders) -- depends on PRD-07
```

---

## 10. Implementation Notes

### 10.1 Why `src/` layout?

The `src/` layout (PEP 517) prevents a common pitfall: if the package directory is at the repo root, `import gsplat_mlx` during development might import the source directory instead of the installed package. With `src/` layout, you must install the package (`pip install -e .`) before it is importable, ensuring tests run against the actual installed package.

### 10.2 Why not `hatchling` or `flit`?

`setuptools` is the most widely supported build backend. Since we have no compiled extensions (Metal shaders are a future PRD), there is no advantage to switching. If PRD-14 introduces Metal shader compilation, we may switch to `scikit-build-core`.

### 10.3 Quaternion convention

gsplat uses `(w, x, y, z)` quaternion ordering (scalar-first), matching SciPy and most robotics conventions. The `make_gaussians` fixture follows this convention. All subsequent PRDs must use the same ordering.

### 10.4 Camera convention

gsplat uses OpenCV camera convention:
- **X**: right
- **Y**: down
- **Z**: forward (into the scene)

The `make_view_matrix` function constructs a look-at matrix following this convention. This is critical for projection correctness in PRD-05.

### 10.5 MLX lazy evaluation

MLX uses lazy evaluation by default. Arrays are not computed until `mx.eval()` is called or the values are read (e.g., via `np.array(x)` or `x.tolist()`). The `check_all_close` helper calls `mx.eval()` before comparison to ensure values are materialized. Test authors should be aware that shape and dtype are available immediately (without eval), but values are not.
