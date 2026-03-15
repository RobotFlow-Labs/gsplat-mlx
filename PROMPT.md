# gsplat-MLX — Master Build Prompt

> Copy this entire file and paste it as a prompt when working in the `/Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx` directory.

---

## Mission

Port **gsplat** (nerfstudio's CUDA-accelerated 3D Gaussian Splatting rasterizer) from **PyTorch + CUDA C++** to **Apple MLX**, creating the first native Apple Silicon Gaussian Splatting rasterization library.

This unlocks the **entire 3DGS ecosystem** on Mac — SplaTAM, MonoGS, WildGS-SLAM, nerfstudio, and every 3DGS-SLAM paper depends on gsplat's CUDA kernels.

**Built by [AIFLOW LABS](https://aiflowlabs.io) / [RobotFlow Labs](https://robotflowlabs.com)**

---

## Context & Prior Art

We already ported **PointCNN++ (CVPR 2026)** from PyTorch+Triton+CUDA to MLX in `pointelligence-mlx`. That port involved 5 custom sparse convolution kernels (MVMR, VVOR, Indexed Distance, Segment Reduce) rewritten using `@mx.custom_function` with `.vjp` for backward passes. 344 tests, all passing.

**gsplat is the same class of problem** — custom CUDA kernels that need MLX equivalents — but the domain is rasterization instead of sparse convolution. The core patterns are:
- Tile-based sorting and binning → MLX argsort + scatter
- Per-pixel alpha compositing → MLX sequential reduce
- Gaussian projection (3D→2D) → MLX matrix ops
- Spherical harmonics evaluation → MLX vectorized math
- Custom backward passes via `torch.autograd.Function` → `@mx.custom_function` + `.vjp`

**Key advantage**: gsplat ships `_torch_impl.py` — pure-Python/PyTorch reference implementations of every CUDA kernel. These are our **exact porting blueprint**. We translate those from torch → mlx.

---

## Reference Repositories

Already cloned into `repositories/` (gitignored, never pushed):

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx

# If not already cloned:
mkdir -p repositories
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git repositories/gsplat-upstream
git clone --depth 1 https://github.com/ml-explore/mlx.git repositories/mlx-ref
git clone --depth 1 https://github.com/ml-explore/mlx-examples.git repositories/mlx-examples-ref

# Also useful for reference:
git clone --depth 1 https://github.com/RobotFlow-Labs/pointelligence-mlx.git repositories/pointelligence-mlx-ref
```

---

## Upstream gsplat Architecture

### Source Map

```
gsplat/                          ← Python package
  __init__.py                    #  Public API exports
  rendering.py                   #  High-level rasterization() entry point (2381 lines)
  utils.py                       #  depth_to_normal, projection matrices
  strategy/                      #  Gaussian densification strategies
    base.py                      #  Abstract Strategy
    default.py                   #  Clone/split/prune
    mcmc.py                      #  MCMC-based densification
    ops.py                       #  Low-level ops (duplicate, remove)
  optimizers/
    selective_adam.py             #  Sparse Adam for Gaussians
  compression/
    png_compression.py           #  Model compression
    sort.py                      #  Morton sort
  color_correct.py               #  Affine/quadratic color correction
  exporter.py                    #  PLY export
  distributed.py                 #  Multi-GPU (not relevant for MLX)
  relocation.py                  #  Gaussian relocation

  cuda/                          ← THE CORE: CUDA kernels + Python wrappers
    _wrapper.py                  #  PyTorch autograd.Function wrappers (3109 lines)
    _torch_impl.py               #  ★ Pure-Python 3DGS reference impl (775 lines)
    _torch_impl_2dgs.py          #  ★ Pure-Python 2DGS reference impl (334 lines)
    _torch_impl_eval3d.py        #  ★ Pure-Python eval3d reference impl (835 lines)
    _torch_impl_ut.py            #  ★ Pure-Python unscented transform impl (598 lines)
    _torch_impl_lidar.py         #  ★ Pure-Python lidar rasterization (388 lines)
    _math.py                     #  Math utilities (782 lines)
    _torch_cameras.py            #  Camera model implementations
    _torch_lidars.py             #  Lidar model implementations
    _backend.py                  #  JIT compilation of CUDA extensions
    _constants.py                #  Shared constants
    _lidar.py                    #  Lidar data structures
    build.py                     #  CUDA build system

    csrc/                        ← Raw CUDA C++ kernels
      # Projection kernels
      Projection.cpp/h           #  C++ dispatch
      ProjectionEWA3DGSFused.cu  #  3DGS EWA projection (forward+backward)
      ProjectionEWA3DGSPacked.cu #  Packed variant
      ProjectionEWASimple.cu     #  Simplified projection
      Projection2DGS.cuh         #  2DGS projection header
      Projection2DGSFused.cu     #  2DGS fused projection
      Projection2DGSPacked.cu    #  2DGS packed variant
      ProjectionUT3DGSFused.cu   #  Unscented transform projection

      # Rasterization kernels
      Rasterization.cpp/h        #  C++ dispatch
      RasterizeToPixels3DGSFwd.cu   #  ★ Core forward rasterizer
      RasterizeToPixels3DGSBwd.cu   #  ★ Core backward rasterizer
      RasterizeToPixels2DGSFwd.cu   #  2DGS forward
      RasterizeToPixels2DGSBwd.cu   #  2DGS backward
      RasterizeToIndices3DGS.cu     #  Tile intersection indices
      RasterizeToIndices2DGS.cu     #  2DGS tile indices
      RasterizeToPixelsFromWorld3DGSFwd.cu  # World-space rasterization
      RasterizeToPixelsFromWorld3DGSBwd.cu  # World-space backward

      # Intersection kernels
      Intersect.cpp/h            #  C++ dispatch
      IntersectTile.cu           #  Tile-Gaussian intersection test
      IntersectTileLidar.cu      #  Lidar tile intersection

      # Spherical harmonics
      SphericalHarmonics.cpp/h   #  C++ dispatch
      SphericalHarmonicsCUDA.cu  #  SH evaluation + backward

      # Covariance computation
      QuatScaleToCovar.cpp/h     #  C++ dispatch
      QuatScaleToCovarCUDA.cu    #  Quaternion+scale → covariance matrix

      # Utilities
      Adam.cpp/h                 #  Selective Adam optimizer
      AdamCUDA.cu                #  CUDA Adam
      Relocation.cpp/h           #  Gaussian relocation
      RelocationCUDA.cu          #  CUDA relocation
      Null.cpp/h                 #  No-op kernel
      NullCUDA.cu                #  CUDA no-op
      CameraWrappers.cu          #  Camera model dispatch
```

### The 10 Core autograd.Functions (what we port)

Each wraps CUDA kernels with custom forward/backward. We replace with `@mx.custom_function`:

| # | autograd.Function | CUDA Kernels | What It Does | Priority |
|---|-------------------|--------------|-------------|----------|
| 1 | `_FullyFusedProjection` | `ProjectionEWA3DGSFused.cu` | Project 3D Gaussians → 2D screen space | P0 |
| 2 | `_RasterizeToPixels` | `RasterizeToPixels3DGS{Fwd,Bwd}.cu` | Alpha-composite sorted Gaussians into image | P0 |
| 3 | `_SphericalHarmonics` | `SphericalHarmonicsCUDA.cu` | Evaluate SH coefficients → RGB color | P0 |
| 4 | `_QuatScaleToCovarPreci` | `QuatScaleToCovarCUDA.cu` | Quaternion + scale → covariance matrix | P0 |
| 5 | `_Proj` | (uses projection internally) | Basic point projection | P1 |
| 6 | `_FullyFusedProjection2DGS` | `Projection2DGSFused.cu` | 2DGS variant of projection | P1 |
| 7 | `_RasterizeToPixels2DGS` | `RasterizeToPixels2DGS{Fwd,Bwd}.cu` | 2DGS alpha compositing | P1 |
| 8 | `_RasterizeToPixelsEval3D` | `RasterizeToPixelsFromWorld3DGS*.cu` | World-space rasterization | P2 |
| 9 | `_FullyFusedProjectionPacked` | (packed variants) | Memory-efficient projection | P2 |
| 10 | `_FullyFusedProjectionPacked2DGS` | (packed 2DGS) | Memory-efficient 2DGS | P2 |

### Supporting Functions (pure Python, easier to port)

| Function | File | Lines | What |
|----------|------|-------|------|
| `isect_tiles` | `_torch_impl.py` | ~80 | Tile-Gaussian intersection (sorting + binning) |
| `isect_offset_encode` | `_torch_impl.py` | ~20 | Encode tile offsets |
| `accumulate` | `_torch_impl.py` | ~100 | Alpha compositing accumulation |
| `_fully_fused_projection` | `_torch_impl.py` | ~200 | Reference projection implementation |
| `_rasterize_to_pixels` | `_torch_impl.py` | ~150 | Reference rasterization |
| `_spherical_harmonics` | `_torch_impl.py` | ~50 | Reference SH evaluation |
| `_eval_sh_bases_fast` | `_torch_impl.py` | ~100 | SH basis functions |

---

## Architecture Design

### Strategy: Port the torch reference impls, NOT the CUDA C++

gsplat's `_torch_impl.py` files contain complete, correct, pure-PyTorch implementations of every kernel. These are used for:
- Testing (comparing CUDA output against Python reference)
- CPU fallback
- Understanding the algorithm

**We port these Python files from torch→mlx**, NOT the raw CUDA `.cu` files. This gives us:
1. Correct algorithms (already tested against CUDA)
2. Pure Python (no Metal shaders needed for MVP)
3. Readable, maintainable code
4. Custom backward passes via `@mx.custom_function` matching the torch autograd

Later (Phase 3+), we can write Metal shaders for performance-critical paths.

### Project Structure

```
gsplat-mlx/
├── .gitignore
├── pyproject.toml
├── PROMPT.md                          # This file
├── UPSTREAM_VERSION.md
├── repositories/                      # (gitignored)
│   ├── gsplat-upstream/
│   ├── mlx-ref/
│   └── mlx-examples-ref/
│
├── src/
│   └── gsplat_mlx/
│       ├── __init__.py                # Public API (mirror upstream exports)
│       ├── _version.py                # "0.1.0"
│       │
│       ├── core/                      # ★ Ported kernels (from _torch_impl.py → MLX)
│       │   ├── __init__.py
│       │   ├── projection.py          # _fully_fused_projection → MLX
│       │   ├── rasterization.py       # _rasterize_to_pixels → MLX
│       │   ├── spherical_harmonics.py # _spherical_harmonics → MLX
│       │   ├── covariance.py          # quat_scale_to_covar → MLX
│       │   ├── intersection.py        # isect_tiles, isect_offset_encode → MLX
│       │   ├── accumulate.py          # alpha compositing accumulate → MLX
│       │   ├── math_utils.py          # _math.py utilities → MLX
│       │   └── cameras.py            # Camera models → MLX
│       │
│       ├── core_2dgs/                 # 2DGS variants
│       │   ├── __init__.py
│       │   ├── projection_2dgs.py
│       │   ├── rasterization_2dgs.py
│       │   └── accumulate_2dgs.py
│       │
│       ├── rendering.py               # High-level rasterization() API
│       ├── utils.py                   # depth_to_normal, projection matrices
│       │
│       ├── strategy/                  # Densification strategies
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── default.py
│       │   └── mcmc.py
│       │
│       ├── optimizers/
│       │   └── selective_adam.py       # Selective Adam → MLX optimizer
│       │
│       ├── compression/
│       │   ├── png_compression.py
│       │   └── sort.py
│       │
│       ├── exporter.py                # PLY export (numpy-based, easy)
│       ├── color_correct.py
│       └── relocation.py
│
├── tests/
│   ├── conftest.py                    # Shared fixtures, torch comparison helpers
│   ├── test_smoke.py                  # Package imports, MLX environment
│   ├── test_math_utils.py             # Math utilities correctness
│   ├── test_covariance.py             # Quat+scale → covariance
│   ├── test_spherical_harmonics.py    # SH evaluation forward + backward
│   ├── test_projection.py            # 3D→2D projection forward + backward
│   ├── test_intersection.py           # Tile intersection and sorting
│   ├── test_rasterization.py          # Full pixel rasterization
│   ├── test_accumulate.py             # Alpha compositing
│   ├── test_rendering.py             # High-level rasterization() API
│   ├── test_2dgs.py                   # 2DGS variants
│   ├── test_strategy.py              # Densification strategies
│   └── test_training.py              # End-to-end optimization loop
│
└── prds/
    ├── PRD-01-dev-environment.md
    ├── PRD-02-math-utils.md
    ├── PRD-03-covariance.md
    ├── PRD-04-spherical-harmonics.md
    ├── PRD-05-projection.md
    ├── PRD-06-intersection.md
    ├── PRD-07-rasterization.md
    ├── PRD-08-accumulate.md
    ├── PRD-09-rendering-api.md
    ├── PRD-10-strategy.md
    ├── PRD-11-optimizer.md
    ├── PRD-12-2dgs.md
    ├── PRD-13-training-loop.md
    └── PRD-14-metal-shaders.md        # Future: performance optimization
```

---

## Build Order (PRD Sequence)

### Phase 1: Foundation (Week 1)

**PRD-01: Dev Environment**
- `pyproject.toml` with `mlx>=0.31.0`, `numpy`, `scipy`, `pillow`
- Package structure under `src/gsplat_mlx/`
- Smoke tests verifying MLX available, package imports
- Test harness with `check_all_close()` comparing MLX vs torch reference
- Dev install: `uv pip install -e ".[dev]"`

**PRD-02: Math Utilities**
- Port `_math.py` (782 lines) → `core/math_utils.py`
- `_numerically_stable_norm2`, polynomial evaluation, camera math
- All pure math — straightforward torch→mlx translation
- Tests: compare each function output against torch reference

### Phase 2: Core Primitives (Week 2)

**PRD-03: Quaternion-Scale to Covariance**
- Port `_QuatScaleToCovarPreci` autograd.Function → `@mx.custom_function`
- Forward: quaternion + scale → 3x3 covariance + precision matrices
- Backward: gradient through quaternion/scale
- Tests: forward correctness, backward VJP, batch sizes

**PRD-04: Spherical Harmonics**
- Port `_spherical_harmonics` and `_eval_sh_bases_fast` → MLX
- SH degree 0-3 (1, 4, 9, 16 coefficients)
- Forward: SH coefficients + view direction → RGB
- Backward: gradients w.r.t. SH coefficients and directions
- Tests: compare against torch reference per-degree

**PRD-05: 3D Gaussian Projection**
- Port `_fully_fused_projection` (200 lines) → `core/projection.py`
- 3D Gaussian parameters → 2D screen-space ellipse
- Camera models: pinhole, fisheye, ortho
- Uses covariance (PRD-03) internally
- Custom backward via `@mx.custom_function`
- Tests: single Gaussian, batch, different cameras, backward gradients

### Phase 3: Rasterization Pipeline (Week 3)

**PRD-06: Tile Intersection**
- Port `_isect_tiles` → `core/intersection.py`
- Compute which Gaussians overlap which screen tiles (16x16 pixels)
- Depth-based sorting within each tile
- `isect_offset_encode`: prefix sum for tile→Gaussian mapping
- Tests: known geometry, edge cases (empty tiles, overlapping)

**PRD-07: Pixel Rasterization**
- Port `_rasterize_to_pixels` → `core/rasterization.py`
- Per-pixel alpha compositing of sorted Gaussians
- Front-to-back blending with early termination
- Forward: produces rendered image + alpha + depth
- Backward: gradient through compositing chain
- Tests: single Gaussian → known pixel values, multi-Gaussian blending

**PRD-08: Accumulate**
- Port `accumulate` function → `core/accumulate.py`
- High-level alpha-composite accumulation
- Supports RGB, depth, expected depth modes
- Tests: compare rendered images against torch reference

### Phase 4: High-Level API (Week 4)

**PRD-09: Rendering API**
- Port `rendering.py` → `rendering.py`
- `rasterization()` function — the main entry point
- Orchestrates: projection → intersection → sort → rasterize
- RenderMode support: RGB, D, ED, RGB+D, etc.
- RasterizeMode: classic, antialiased
- Tests: full pipeline from Gaussians → rendered image

**PRD-10: Densification Strategy**
- Port `strategy/default.py` and `strategy/mcmc.py`
- Clone, split, prune operations on Gaussians
- Gradient-based densification decisions
- Tests: split/clone logic, pruning thresholds

**PRD-11: Selective Adam Optimizer**
- Port `optimizers/selective_adam.py` → MLX optimizer
- Sparse updates: only optimize Gaussians that were visible
- Tests: parameter update correctness, sparsity

### Phase 5: End-to-End (Week 5)

**PRD-12: 2DGS Support**
- Port `_torch_impl_2dgs.py` → `core_2dgs/`
- 2D Gaussian Splatting variants (surfel-based)
- Tests: 2DGS projection + rasterization

**PRD-13: Training Loop**
- End-to-end Gaussian Splatting optimization on a single image
- Load/create Gaussians → render → L1+SSIM loss → backward → update
- Verify loss convergence (should decrease monotonically)
- Tests: synthetic scene optimization

### Phase 6: Performance (Future)

**PRD-14: Metal Shaders**
- Replace Python rasterization with Metal compute shaders
- Tile-based parallel rasterization on GPU
- Focus on `RasterizeToPixelsFwd` and `RasterizeToPixelsBwd`
- Benchmark against Python implementation

---

## torch→mlx Mapping for gsplat

| PyTorch | MLX |
|---------|-----|
| `torch.tensor()` | `mx.array()` |
| `torch.zeros()` | `mx.zeros()` |
| `torch.ones()` | `mx.ones()` |
| `torch.cat()` | `mx.concatenate()` |
| `torch.stack()` | `mx.stack()` |
| `torch.clamp()` | `mx.clip()` |
| `torch.where()` | `mx.where()` |
| `torch.einsum()` | `mx.einsum()` (supported!) |
| `torch.sort()` | `mx.argsort()` + gather |
| `torch.cumsum()` | `mx.cumsum()` |
| `torch.bincount()` | Custom scatter-add |
| `torch.searchsorted()` | Custom binary search |
| `tensor.unsqueeze(d)` | `mx.expand_dims(arr, d)` |
| `tensor.squeeze()` | `mx.squeeze()` |
| `tensor.permute()` | `mx.transpose()` |
| `tensor.contiguous()` | No-op (MLX is lazy) |
| `tensor.to(device)` | No-op (unified memory) |
| `tensor.detach()` | `mx.stop_gradient()` |
| `torch.autograd.Function` | `@mx.custom_function` + `.vjp` |
| `F.relu()` | `mx.maximum(x, 0)` |
| `F.sigmoid()` | `mx.sigmoid()` |
| `torch.no_grad()` | Not needed |
| `torch.amp.autocast()` | Not needed |
| `torch.distributed.*` | Not applicable |

### Key Pattern: autograd.Function → @mx.custom_function

```python
# PyTorch (upstream):
class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree, dirs, coeffs, masks):
        colors = _eval_sh(degree, dirs, coeffs)
        ctx.save_for_backward(dirs, coeffs)
        return colors

    @staticmethod
    def backward(ctx, grad_colors):
        dirs, coeffs = ctx.saved_tensors
        # ... compute gradients
        return None, grad_dirs, grad_coeffs, None

# MLX (our port):
@mx.custom_function
def spherical_harmonics(degree, dirs, coeffs, masks):
    colors = _eval_sh(degree, dirs, coeffs)
    return colors

@spherical_harmonics.vjp
def sh_vjp(primals, cotangent, output):
    degree, dirs, coeffs, masks = primals
    # ... compute gradients using mlx ops
    return (None, grad_dirs, grad_coeffs, None)
```

---

## Upstream Sync Protocol

gsplat is more stable than LeRobot (rasterization math doesn't change often). Sync when:
- New camera models added
- New rasterization modes
- Bug fixes in `_torch_impl.py`

```bash
cd repositories/gsplat-upstream
git fetch origin
# Diff the reference implementations (our porting source):
git diff HEAD..origin/main -- gsplat/cuda/_torch_impl.py
git diff HEAD..origin/main -- gsplat/cuda/_torch_impl_2dgs.py
git diff HEAD..origin/main -- gsplat/rendering.py
```

---

## Dev Commands

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx

# Setup
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
pytest tests/ -v
pytest tests/test_projection.py -v          # Specific kernel
pytest tests/ -v -m "requires_torch"        # Cross-framework comparison
pytest tests/ -v -m "benchmark"             # Performance
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | Apple Silicon (M1/M2/M3/M4) |
| Python | >= 3.10 |
| MLX | >= 0.31.0 |
| NumPy | >= 1.24.0 |
| SciPy | >= 1.10.0 |
| Pillow | >= 9.0.0 |

**Dev extras**: pytest, pytest-benchmark, torch (cross-framework), imageio

---

## Success Criteria

### Phase 1 (MVP — Render a single image)
- [ ] Projection: 3D Gaussians → 2D screen-space, matches torch within atol=1e-4
- [ ] Spherical harmonics: SH → RGB, all degrees 0-3
- [ ] Tile intersection + sorting: correct binning
- [ ] Pixel rasterization: rendered image matches torch reference
- [ ] `rasterization()` API works end-to-end
- [ ] 150+ tests, all passing

### Phase 2 (Trainable)
- [ ] All backward passes produce correct gradients
- [ ] Selective Adam optimizer works
- [ ] Densification strategy (clone/split/prune)
- [ ] Single-image optimization converges
- [ ] 250+ tests

### Phase 3 (Production)
- [ ] 2DGS support
- [ ] Load pretrained .ply Gaussian scenes
- [ ] Render at interactive framerates on M3+
- [ ] Metal shaders for rasterization hot path
- [ ] nerfstudio integration guide

---

## Start Building

Begin with **PRD-01: Dev Environment**. Then **PRD-02: Math Utilities** — port `_math.py`. Then **PRD-03: Covariance** and **PRD-04: Spherical Harmonics** in parallel — these are independent leaf kernels.

Use the `/port-to-mlx` skill when translating torch→mlx. Reference `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py` as the source of truth for every algorithm. Use `repositories/pointelligence-mlx-ref/` for examples of `@mx.custom_function` VJP patterns.
