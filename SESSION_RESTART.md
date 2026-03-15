# Session Restart Guide

How to resume development of gsplat-mlx after this build session.

## What Was Built

This project was built in a single Claude Code session on **2026-03-15** by
AIFLOW LABS using Claude Opus 4.6 with the `/port-to-mlx` skill.

### Session Stats

| Metric | Value |
|--------|-------|
| Duration | ~4 hours |
| Sub-agents used | ~30 (parallel build + review) |
| Source files created | 33 |
| Test files created | 25 |
| Example files created | 7 |
| PRD files created | 14 |
| Total LOC | ~17,000 |
| Tests | 405 passing |
| Code reviews | 3 passes, all critical/high issues resolved |
| Commits | 10 |

### Build Sequence

```
1. Read PROMPT.md (master build specification)
2. Load /port-to-mlx skill (MLX porting patterns from prior projects)
3. 14 PRD agents launched in parallel (PRD-01 through PRD-14)
4. PRD-01 through PRD-04: Foundation + core primitives (4 parallel agents)
5. PRD-05 through PRD-08: Projection + rasterization (4 parallel agents)
6. Code review #1 → fix critical issues (duplicated code, mx.eval leaks)
7. PRD-09 through PRD-11: Rendering API + strategy + optimizer (4 agents)
8. Code review #2 → fix blocking issues (pure MLX in utils.py)
9. PRD-12 + PRD-13: 2DGS + training loop (3 agents)
10. Gap analysis vs upstream gsplat → close all gaps (4 agents)
11. Code review #3 → fix transmittance termination, add e2e gradient test
12. README with Mermaid diagrams + 6 examples + benchmarks
```

## How to Restart

### 1. Open the project

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/gsplat-mlx
```

### 2. Activate environment

```bash
source .venv/bin/activate
# Or recreate if needed:
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

### 3. Verify everything works

```bash
# Run all tests
.venv/bin/pytest tests/ -v

# Run an example
.venv/bin/python examples/01_hello_gaussians.py

# Check GPU
.venv/bin/python -c "import mlx.core as mx; print(mx.default_device())"
```

### 4. Load context for Claude Code

Tell Claude:

```
Read PROMPT.md, CLAUDE.md, and load /port-to-mlx skill.
The project has 13/14 PRDs implemented (PRD-14 Metal shaders deferred).
405 tests passing. 3 code reviews done.
```

Key files for context:
- `PROMPT.md` — master build specification with upstream architecture
- `CLAUDE.md` — project instructions and build order
- `prds/` — all 14 PRD specifications (18K lines of detail)
- `SESSION_RESTART.md` — this file

### 5. Key skills/commands

```
/port-to-mlx          — MLX porting patterns, torch→mlx mappings, gotchas
/code-review          — Run production code review
/simplify             — Review changed code for quality
```

## What Remains

### PRD-14: Metal Shaders (deferred — performance optimization)

The Tier-2 differentiable rasterizer works but uses Python loops over
Gaussians. A Metal compute shader (Tier-3) would give 10-100x speedup
on the rasterization hot path. The full PRD spec is at
`prds/PRD-14-metal-shaders.md` including complete MSL shader code.

### Nice-to-have gaps (not blocking)

| Feature | Priority | Effort |
|---------|----------|--------|
| Metal rasterization shader (Tier-3) | HIGH | ~2 weeks |
| 2DGS differentiable rasterizer (Tier-2) | MEDIUM | ~1 day |
| F-theta camera model (needs UT) | LOW | ~1 day |
| LiDAR model + tiling | LOW | ~2 days |
| Rolling shutter support | LOW | ~1 day |
| MCMCStrategy full implementation | LOW | ~2 days |
| K-means SH compression (needs torchpq) | LOW | ~1 day |

## Git Remotes

```
origin   → github.com/RobotFlow-Labs/gsplat-mlx.git (our fork)
upstream → github.com/nerfstudio-project/gsplat.git  (CUDA original)
```

### Syncing with upstream

```bash
cd repositories/gsplat-upstream
git fetch origin
# Check what changed in the reference implementations:
git diff HEAD..origin/main -- gsplat/cuda/_torch_impl.py
git diff HEAD..origin/main -- gsplat/cuda/_torch_impl_2dgs.py
git diff HEAD..origin/main -- gsplat/rendering.py
```

## Architecture Quick Reference

```
rasterization()                    # User entry point (rendering.py)
  |
  +-- quat_scale_to_covar_preci()  # core/covariance.py     [GPU, differentiable]
  +-- fully_fused_projection()     # core/projection.py     [GPU, differentiable]
  +-- spherical_harmonics()        # core/spherical_harmonics.py [GPU, differentiable]
  +-- isect_tiles()                # core/intersection.py   [CPU, non-diff]
  +-- isect_offset_encode()        # core/intersection.py   [CPU, non-diff]
  +-- rasterize_to_pixels_mlx()   # core/rasterization_mlx.py [GPU, differentiable]
```

## Memory Layout

All Claude Code memory for this project is stored at:
```
~/.claude/projects/-Users-ilessio-Development-AIFLOWLABS-R-D-gsplat-mlx/memory/
```

The `/port-to-mlx` skill is at:
```
~/.claude/skills/port-to-mlx/
```
