# gsplat-mlx — MLX Port of gsplat

## Overview
Port of [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) from CUDA/PyTorch to Apple's [MLX](https://github.com/ml-explore/mlx) framework. Goal: native 3D Gaussian Splatting on Apple Silicon (M1/M2/M3/M4) without CUDA dependency.

**Strategy**: Port the `_torch_impl.py` pure-Python reference implementations (NOT raw CUDA kernels) to MLX. Use `@mx.custom_function` + `.vjp` for backward passes.

**Built by [AIFLOW LABS](https://aiflowlabs.io) / [RobotFlow Labs](https://robotflowlabs.com)**

## Structure

| Directory | Purpose |
|-----------|---------|
| `src/gsplat_mlx/` | MLX port source code |
| `src/gsplat_mlx/core/` | Ported kernels (projection, rasterization, SH, etc.) |
| `src/gsplat_mlx/core_2dgs/` | 2DGS surfel-based variants |
| `src/gsplat_mlx/strategy/` | Gaussian densification (clone/split/prune) |
| `src/gsplat_mlx/optimizers/` | Selective Adam optimizer |
| `src/gsplat_mlx/compression/` | Model compression |
| `tests/` | Test suite |
| `prds/` | Product Requirements Documents (14 PRDs) |
| `repositories/` | Reference repos — **read-only, gitignored** |

## PRD Build Order

| Phase | PRD | Title | Dependencies |
|-------|-----|-------|-------------|
| 1 | PRD-01 | Dev Environment & Foundation | — |
| 1 | PRD-02 | Math Utilities (782 LOC) | PRD-01 |
| 2 | PRD-03 | Quaternion-Scale → Covariance | PRD-01, 02 |
| 2 | PRD-04 | Spherical Harmonics | PRD-01, 02 |
| 2 | PRD-05 | 3D→2D Gaussian Projection | PRD-01, 02, 03 |
| 3 | PRD-06 | Tile-Gaussian Intersection | PRD-01, 05 |
| 3 | PRD-07 | Pixel Rasterization | PRD-01, 06 |
| 3 | PRD-08 | Alpha Compositing Accumulate | PRD-01 |
| 4 | PRD-09 | High-Level Rendering API | PRD-01–08 |
| 4 | PRD-10 | Densification Strategy | PRD-01, 09 |
| 4 | PRD-11 | Selective Adam Optimizer | PRD-01 |
| 5 | PRD-12 | 2DGS Surfel Support | PRD-01, 02, 06, 09 |
| 5 | PRD-13 | End-to-End Training Loop | PRD-01–11 |
| 6 | PRD-14 | Metal Shaders (performance) | PRD-01–13 |

## Git Remotes

- `origin` → `github.com/RobotFlow-Labs/gsplat-mlx.git`
- `upstream` → `github.com/nerfstudio-project/gsplat.git`

## Dev Commands

```bash
# Setup
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
pytest tests/ -v
pytest tests/ -v -m "not requires_torch"
pytest tests/test_projection.py -v   # specific kernel
```

## Conventions

- Package manager: `uv`
- `mlx.core` aliased as `mx`
- `torch.autograd.Function` → `@mx.custom_function` + `.vjp`
- No boolean indexing in MLX — use `mx.where`
- Call `mx.eval()` to materialize lazy computation
- Use `rg` (ripgrep) for searching
- Keep function signatures compatible with upstream

## Key torch→mlx Mappings

| PyTorch | MLX |
|---------|-----|
| `torch.tensor()` | `mx.array()` |
| `torch.clamp()` | `mx.clip()` |
| `torch.einsum()` | `mx.einsum()` |
| `torch.cat()` | `mx.concatenate()` |
| `tensor.to(device)` | No-op (unified memory) |
| `tensor.detach()` | `mx.stop_gradient()` |
| `torch.autograd.Function` | `@mx.custom_function` + `.vjp` |

# currentDate
Today's date is 2026-03-15.
