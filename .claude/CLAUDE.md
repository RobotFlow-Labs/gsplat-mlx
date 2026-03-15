# gsplat-MLX — Claude Code Project Config

## Project Overview
Port of nerfstudio's gsplat (CUDA 3D Gaussian Splatting rasterizer) to Apple MLX. Enables the entire 3DGS-SLAM ecosystem on Apple Silicon.

## Key Architecture
- `src/gsplat_mlx/core/` — Ported kernels: projection, rasterization, SH, covariance, intersection
- `src/gsplat_mlx/core_2dgs/` — 2DGS variants
- `src/gsplat_mlx/rendering.py` — High-level rasterization() API
- `src/gsplat_mlx/strategy/` — Gaussian densification (clone/split/prune)
- `repositories/gsplat-upstream/` — Reference upstream (gitignored)

## Porting Strategy
Port from `gsplat/cuda/_torch_impl.py` (pure-Python reference) → MLX, NOT from raw CUDA C++.
Custom backward passes: `torch.autograd.Function` → `@mx.custom_function` + `.vjp`

## Critical Design Rules
1. **Port the _torch_impl.py reference implementations** — they are the algorithm source of truth
2. **Use @mx.custom_function for every backward pass** — same pattern as pointelligence-mlx
3. **Mirror upstream API exactly** — same function names, same arguments
4. **PRD-driven** — one PRD per kernel/component in prds/
5. **Cross-framework tests** — compare MLX output vs PyTorch _torch_impl reference

## Dev Commands
```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
```

## Build Order
PRD-01 Dev Env → PRD-02 Math → PRD-03 Covariance → PRD-04 SH → PRD-05 Projection → PRD-06 Intersection → PRD-07 Rasterization → PRD-08 Accumulate → PRD-09 Rendering API

## Reference Files (most important)
- `repositories/gsplat-upstream/gsplat/cuda/_torch_impl.py` — 3DGS reference
- `repositories/gsplat-upstream/gsplat/cuda/_torch_impl_2dgs.py` — 2DGS reference
- `repositories/gsplat-upstream/gsplat/cuda/_wrapper.py` — autograd.Function definitions
- `repositories/gsplat-upstream/gsplat/rendering.py` — high-level API

# currentDate
Today's date is 2026-03-15.
