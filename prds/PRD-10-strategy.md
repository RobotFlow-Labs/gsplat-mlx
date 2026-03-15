# PRD-10: Gaussian Densification Strategies

## Overview

Port the Gaussian densification strategy system from gsplat to MLX. During 3DGS training, the set of Gaussians evolves dynamically: Gaussians are cloned, split, pruned, and reset based on gradient statistics and geometric criteria. This PRD covers the abstract `Strategy` base class, the complete `DefaultStrategy` (original 3DGS paper algorithm), a stub for `MCMCStrategy` (deferred to P2), and all low-level parameter manipulation ops (`duplicate`, `remove`, `split`, `reset_opa`).

## Source Reference

- **`strategy/base.py`**: `repositories/gsplat-upstream/gsplat/strategy/base.py` (~66 lines) -- Abstract `Strategy` dataclass
- **`strategy/default.py`**: `repositories/gsplat-upstream/gsplat/strategy/default.py` (~355 lines) -- `DefaultStrategy` with full grow/prune/reset lifecycle
- **`strategy/mcmc.py`**: `repositories/gsplat-upstream/gsplat/strategy/mcmc.py` (~215 lines) -- `MCMCStrategy` (P2 stub only)
- **`strategy/ops.py`**: `repositories/gsplat-upstream/gsplat/strategy/ops.py` (~385 lines) -- Low-level ops: `duplicate`, `remove`, `split`, `reset_opa`, `relocate`, `sample_add`, `inject_noise_to_position`
- **`strategy/__init__.py`**: `repositories/gsplat-upstream/gsplat/strategy/__init__.py` -- Re-exports

## Scope

### In Scope (P1)
- `Strategy` abstract base class
- `DefaultStrategy` with complete grow/prune/reset algorithm
- Low-level ops: `duplicate`, `remove`, `split`, `reset_opa`
- Helper: `_update_param_with_optimizer` (core param+optimizer state update pattern)
- All gradient accumulation and state tracking logic
- Full parameter and optimizer state synchronization

### In Scope (P2 stub)
- `MCMCStrategy` class definition with `NotImplementedError` for all methods
- `relocate`, `sample_add`, `inject_noise_to_position` ops (stub only)

### Out of Scope
- Full MCMCStrategy implementation (separate PRD)
- `compute_relocation` from `gsplat.relocation` (MCMC dependency)
- `quat_scale_to_covar_preci` usage in noise injection (MCMC dependency)
- Distributed/multi-GPU strategy coordination

## Technical Design

### Architecture

```
strategy/
  __init__.py          # Re-exports Strategy, DefaultStrategy, MCMCStrategy
  base.py              # Strategy ABC
  default.py           # DefaultStrategy (3DGS paper)
  mcmc.py              # MCMCStrategy (P2 stub)
  ops.py               # Low-level ops: duplicate, remove, split, reset_opa
```

### MLX vs PyTorch: Fundamental Differences

The strategy system is deeply coupled to PyTorch's parameter and optimizer model. The MLX port requires rethinking several patterns:

| Concept | PyTorch | MLX Port |
|---------|---------|----------|
| Parameters | `torch.nn.Parameter` with `requires_grad` | Plain `mx.array`; all params assumed trainable if in optimizer dict |
| Optimizer state | `optimizer.state[param]` keyed by Parameter identity | Dict-based: `optimizer_state[param_name]` keyed by string name |
| Gradient retention | `tensor.retain_grad()` | MLX uses functional `mx.grad()` -- gradients passed explicitly via `info` dict |
| In-place update | Mutate param in optimizer's `param_groups` | Replace arrays in params dict and optimizer state dict |
| Boolean masking | `params[mask]` | `params[mask]` works in MLX, or `mx.take(params, mx.where(mask)[0])` |
| `@torch.no_grad()` | Context manager | Not needed -- MLX doesn't track gradients by default; only `mx.grad()` does |
| Device management | `tensor.device`, `.to(device)` | MLX is unified memory -- no device management needed |
| `torch.cuda.empty_cache()` | GPU memory cleanup | No-op in MLX (unified memory) |

### Optimizer State Convention

In the MLX port, optimizer state follows this convention:

```python
# params: Dict[str, mx.array]
# e.g., {"means": mx.array([N, 3]), "scales": mx.array([N, 3]), ...}

# optimizers: Dict[str, Dict[str, mx.array]]
# where optimizers[param_name] = {"exp_avg": mx.array, "exp_avg_sq": mx.array, "step": int}
#
# This mirrors PyTorch's optimizer.state[param] dict with keys like
# 'exp_avg' and 'exp_avg_sq', but keyed by parameter name (string)
# rather than parameter identity (object ref).
```

The strategy ops accept:

```python
params: Dict[str, mx.array]
optimizers: Dict[str, Dict[str, mx.array]]
# where optimizers[param_name] = {"exp_avg": ..., "exp_avg_sq": ..., "step": ...}
state: Dict[str, mx.array]
```

---

## Detailed Implementation

### 1. `strategy/ops.py` -- Low-Level Operations

#### `_update_param_with_optimizer`

The core helper that applies a transformation to both parameters and their optimizer state. This is the most critical function to port correctly.

```python
import mlx.core as mx
from typing import Callable, Dict, List, Optional, Union


def _update_param_with_optimizer(
    param_fn: Callable[[str, mx.array], mx.array],
    optimizer_fn: Callable[[str, mx.array], mx.array],
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    names: Optional[List[str]] = None,
):
    """Update parameters and their optimizer state using provided transform functions.

    This is the central mechanism for structural changes to the Gaussian set.
    When Gaussians are added, removed, or split, BOTH the parameter arrays AND
    their associated optimizer state (exp_avg, exp_avg_sq for Adam) must be
    resized/reorganized consistently.

    Args:
        param_fn: Takes (param_name, param_array) -> new_param_array.
            Defines how each parameter is transformed (e.g., concatenate for clone,
            index-select for removal).
        optimizer_fn: Takes (state_key, state_array) -> new_state_array.
            Defines how each optimizer state entry is transformed (typically:
            append zeros for new Gaussians, index-select for removals).
        params: Mutable dict of parameter name -> mx.array.
        optimizers: Mutable dict of param_name -> {state_key: mx.array}.
            The "step" key (int) is preserved as-is.
        names: If provided, only update these parameter names. Otherwise update all.
    """
    if names is None:
        names = list(params.keys())

    for name in names:
        p = params[name]
        new_p = param_fn(name, p)
        params[name] = new_p

        if name not in optimizers:
            # Non-trainable parameter (e.g., frozen features) -- skip
            continue

        opt_state = optimizers[name]
        for key in list(opt_state.keys()):
            if key == "step":
                continue  # step count is a scalar, not resized
            v = opt_state[key]
            opt_state[key] = optimizer_fn(key, v)
```

**Key differences from PyTorch version:**
1. No `torch.nn.Parameter` wrapping -- we just replace the `mx.array` in the dict.
2. Optimizer state is keyed by string name, not by Parameter object identity.
3. No `requires_grad` check -- if a param has no optimizer entry, it's assumed non-trainable.
4. No `param_groups` manipulation -- MLX optimizers don't use that pattern.

#### `duplicate`

Clone Gaussians selected by a boolean mask. The selected Gaussians are appended to the end of each parameter array.

```python
def duplicate(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
):
    """In-place duplicate Gaussians where mask=True.

    After this operation:
    - params[k].shape[0] increases by mask.sum()
    - The duplicated Gaussians have the SAME parameter values as the originals
    - The duplicated Gaussians have ZERO optimizer state (fresh start for Adam)
    - Running state (grad accumulators, counts) is also duplicated

    Args:
        params: Parameter dict. Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state (grad2d, count, etc.). Modified in-place.
        mask: Boolean array [N]. True = duplicate this Gaussian.
    """
    sel = mx.where(mask)[0]  # indices of True entries
    n_sel = sel.shape[0]

    if n_sel == 0:
        return

    def param_fn(name: str, p: mx.array) -> mx.array:
        return mx.concatenate([p, p[sel]], axis=0)

    def optimizer_fn(key: str, v: mx.array) -> mx.array:
        # New Gaussians get zero optimizer state
        zeros = mx.zeros((n_sel,) + v.shape[1:], dtype=v.dtype)
        return mx.concatenate([v, zeros], axis=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    # Update running state
    for k, v in state.items():
        if isinstance(v, mx.array) and v.ndim >= 1:
            state[k] = mx.concatenate([v, v[sel]], axis=0)
```

#### `remove`

Remove Gaussians selected by a boolean mask.

```python
def remove(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
):
    """In-place remove Gaussians where mask=True.

    After this operation:
    - params[k].shape[0] decreases by mask.sum()
    - Optimizer state entries are correspondingly removed
    - Running state entries are correspondingly removed

    Args:
        params: Parameter dict. Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Modified in-place.
        mask: Boolean array [N]. True = REMOVE this Gaussian.
    """
    sel = mx.where(~mask)[0]  # indices to KEEP

    if sel.shape[0] == mask.shape[0]:
        return  # nothing to remove

    def param_fn(name: str, p: mx.array) -> mx.array:
        return p[sel]

    def optimizer_fn(key: str, v: mx.array) -> mx.array:
        return v[sel]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    # Update running state
    for k, v in state.items():
        if isinstance(v, mx.array) and v.ndim >= 1:
            state[k] = v[sel]
```

#### `split`

Split large Gaussians into two smaller ones. This is the most complex operation:

1. Selected Gaussians are removed from their original positions.
2. Two new, smaller Gaussians are inserted at the end.
3. New positions are offset from the original by sampling from the Gaussian's covariance.
4. New scales are reduced by a factor of 1.6.

```python
def split(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
    revised_opacity: bool = False,
):
    """In-place split Gaussians where mask=True into 2 smaller Gaussians each.

    For each split Gaussian:
    - Two new Gaussians are created
    - Positions are offset by samples from the original's covariance ellipsoid
    - Scales are reduced: new_scale = old_scale / 1.6
    - If revised_opacity=True, opacity follows: new_opa = 1 - sqrt(1 - sigmoid(old_opa))
    - All other parameters (colors, SH, quats, etc.) are copied
    - Optimizer state is zeroed for all new Gaussians

    The original split Gaussians are REMOVED (replaced by 2 new ones).
    Net change: +mask.sum() Gaussians (remove K, add 2K).

    Args:
        params: Parameter dict. Must contain "means", "scales", "quats".
            Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Modified in-place.
        mask: Boolean array [N]. True = split this Gaussian.
        revised_opacity: Use revised opacity from arXiv:2404.06109. Default False.
    """
    sel = mx.where(mask)[0]    # indices to split
    rest = mx.where(~mask)[0]  # indices to keep as-is
    n_sel = sel.shape[0]

    if n_sel == 0:
        return

    # Compute displacement samples from the Gaussian covariance
    scales = mx.exp(params["scales"][sel])  # [n_sel, 3]

    # Normalize quaternions and convert to rotation matrices
    quats = params["quats"][sel]  # [n_sel, 4]
    quats = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)  # normalize

    # Build rotation matrices from quaternions (wxyz convention)
    # rotmats: [n_sel, 3, 3]
    rotmats = _normalized_quat_to_rotmat(quats)

    # Sample 2 random offsets per Gaussian: [2, n_sel, 3]
    noise = mx.random.normal(shape=(2, n_sel, 3))

    # Scale the noise by the Gaussian's scale, then rotate:
    # samples = R @ diag(s) @ noise
    # scaled_noise[b, n, j] = noise[b, n, j] * scales[n, j]
    scaled_noise = noise * mx.expand_dims(scales, axis=0)  # [2, n_sel, 3]

    # Rotate: rotmats[n, i, j] * scaled_noise[b, n, j] -> samples[b, n, i]
    # Use batched matmul: expand rotmats to [1, n_sel, 3, 3],
    # scaled_noise to [2, n_sel, 3, 1] -> result [2, n_sel, 3, 1]
    rotmats_exp = mx.expand_dims(rotmats, axis=0)  # [1, n_sel, 3, 3]
    scaled_noise_exp = mx.expand_dims(scaled_noise, axis=-1)  # [2, n_sel, 3, 1]
    samples = mx.squeeze(
        mx.matmul(
            mx.broadcast_to(rotmats_exp, (2, n_sel, 3, 3)),
            scaled_noise_exp
        ),
        axis=-1,
    )  # [2, n_sel, 3]

    n_before = mask.shape[0]

    # Build new parameter arrays
    def param_fn(name: str, p: mx.array) -> mx.array:
        if name == "means":
            # Offset positions by covariance samples
            p_sel = p[sel]  # [n_sel, 3]
            p_split = (mx.expand_dims(p_sel, axis=0) + samples).reshape(-1, 3)  # [2*n_sel, 3]
        elif name == "scales":
            # Reduce scale by factor 1.6 (in log space: subtract log(1.6))
            new_log_scale = mx.log(scales / 1.6)  # [n_sel, 3]
            p_split = mx.concatenate([new_log_scale, new_log_scale], axis=0)  # [2*n_sel, 3]
        elif name == "opacities" and revised_opacity:
            # Revised opacity: new_opa = 1 - sqrt(1 - sigmoid(old_opa))
            old_opa = mx.sigmoid(p[sel])
            new_opa = 1.0 - mx.sqrt(1.0 - old_opa)
            new_logit = _logit(new_opa)
            # Determine repeat shape
            repeats = [2] + [1] * (p[sel].ndim - 1) if p[sel].ndim > 0 else [2]
            p_split = mx.tile(new_logit, repeats)
        else:
            # All other params: just duplicate
            p_sel = p[sel]
            repeats = [2] + [1] * (p_sel.ndim - 1) if p_sel.ndim > 0 else [2]
            p_split = mx.tile(p_sel, repeats)

        # Combine: keep non-split Gaussians, append split results
        return mx.concatenate([p[rest], p_split], axis=0)

    def optimizer_fn(key: str, v: mx.array) -> mx.array:
        # Zero optimizer state for all new Gaussians from splits
        v_new = mx.zeros((2 * n_sel,) + v.shape[1:], dtype=v.dtype)
        return mx.concatenate([v[rest], v_new], axis=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    # Update running state
    for k, v in state.items():
        if isinstance(v, mx.array) and v.ndim >= 1 and v.shape[0] == n_before:
            v_sel = v[sel]
            repeats = [2] + [1] * (v_sel.ndim - 1) if v_sel.ndim > 0 else [2]
            v_split = mx.tile(v_sel, repeats)
            state[k] = mx.concatenate([v[rest], v_split], axis=0)
```

#### Helper: `_normalized_quat_to_rotmat`

```python
def _normalized_quat_to_rotmat(quat: mx.array) -> mx.array:
    """Convert normalized quaternion (wxyz) to rotation matrix.

    Args:
        quat: [..., 4] quaternion in wxyz convention (already normalized).

    Returns:
        Rotation matrix [..., 3, 3].
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    mat = mx.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),        1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),        2*(y*z + w*x),       1 - 2*(x**2 + y**2),
    ], axis=-1)

    return mat.reshape(quat.shape[:-1] + (3, 3))
```

#### Helper: `_logit`

```python
def _logit(x: mx.array) -> mx.array:
    """Inverse sigmoid: logit(x) = log(x / (1 - x)).

    Numerically: clamp to avoid log(0).
    """
    eps = 1e-7
    x = mx.clip(x, eps, 1.0 - eps)
    return mx.log(x / (1.0 - x))
```

#### `reset_opa`

```python
def reset_opa(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    value: float,
):
    """In-place reset all opacities to at most the given post-sigmoid value.

    Opacities are stored as logits (pre-sigmoid). This function clamps them
    so that sigmoid(logit) <= value. The optimizer state for opacities is zeroed.

    Args:
        params: Parameter dict. Must contain "opacities". Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Not modified by this op.
        value: Maximum post-sigmoid opacity value to clamp to.
    """
    logit_cap = _logit(mx.array(value)).item()

    def param_fn(name: str, p: mx.array) -> mx.array:
        if name == "opacities":
            return mx.minimum(p, logit_cap)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: mx.array) -> mx.array:
        return mx.zeros_like(v)

    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )
```

#### Scatter helpers: `_scatter_add` and `_scatter_max`

These are critical operations that accumulate values by index. MLX does not have `index_add_` like PyTorch.

```python
def _scatter_add(
    target: mx.array,  # [N]
    indices: mx.array,  # [M] int32
    values: mx.array,   # [M]
) -> mx.array:
    """Scatter-add: target[indices[i]] += values[i] for all i.

    Handles duplicate indices correctly (accumulates, not overwrites).
    """
    # MLX supports target.at[indices].add(values) for scatter operations
    return target.at[indices].add(values)


def _scatter_max(
    target: mx.array,  # [N]
    indices: mx.array,  # [M] int32
    values: mx.array,   # [M]
) -> mx.array:
    """Scatter-max: target[indices[i]] = max(target[indices[i]], values[i]).

    Handles duplicate indices correctly (takes max, not overwrites).
    """
    # MLX has target.at[indices].maximum(values) for scatter-max
    return target.at[indices].maximum(values)
```

**MLX `.at[]` availability note:** As of MLX 0.17+, `mx.array.at[indices].add(values)` is supported and handles duplicate indices correctly (unlike direct indexing assignment). If the target MLX version does not support `.at[]`, a numpy fallback is needed:

```python
def _scatter_add_fallback(target, indices, values):
    """Numpy fallback for scatter-add."""
    import numpy as np
    t = np.array(target)
    idx = np.array(indices)
    val = np.array(values)
    np.add.at(t, idx, val)
    return mx.array(t)
```

---

### 2. `strategy/base.py` -- Abstract Base Class

```python
from dataclasses import dataclass
from typing import Dict, Any
import mlx.core as mx


@dataclass
class Strategy:
    """Base class for Gaussian densification strategies.

    Defines the interface that all strategies must implement.
    Strategies are called at specific points in the training loop:
    - step_pre_backward: before loss.backward() (for gradient retention, etc.)
    - step_post_backward: after loss.backward() (for grow/prune/reset decisions)
    """

    def check_sanity(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
    ):
        """Verify that params and optimizers are consistent.

        Checks:
        - Every optimizer entry has a corresponding parameter.

        Note: In MLX, there is no `requires_grad` flag. We treat the presence of
        an optimizer entry as the indicator that a parameter is trainable.
        Unlike upstream which asserts exact equality between trainable params and
        optimizer keys, we only assert that optimizer keys are a subset of param keys.
        """
        param_names = set(params.keys())
        opt_names = set(optimizers.keys())
        assert opt_names.issubset(param_names), (
            "Optimizer keys must be a subset of parameter keys, "
            f"but got optimizer keys {opt_names - param_names} not in params."
        )

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        Returns:
            State dict to pass to step_pre_backward / step_post_backward.
        """
        return {}

    def step_pre_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Called before loss.backward(). Default: no-op."""
        pass

    def step_post_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        **kwargs,
    ):
        """Called after loss.backward(). Default: no-op."""
        pass
```

---

### 3. `strategy/default.py` -- DefaultStrategy (Complete Algorithm)

This is the core densification logic from the original 3DGS paper. The algorithm works in phases per refinement step:

**Phase 1 -- Accumulate (every step within refine window):**
1. For each visible Gaussian, accumulate the norm of its 2D projected gradient.
2. Track how many times each Gaussian was visible (`count`).
3. Optionally track maximum 2D screen-space radius (`radii`).

**Phase 2 -- Grow (every `refine_every` steps):**
1. Compute average 2D gradient magnitude per Gaussian: `avg_grad = grad2d / max(count, 1)`.
2. `is_grad_high = avg_grad > grow_grad2d`
3. `is_small = max(exp(scales), dim=-1) <= grow_scale3d * scene_scale`
4. Clone: `is_grad_high AND is_small` (under-reconstruction: need more coverage)
5. Split: `is_grad_high AND NOT is_small` (over-reconstruction: too large, break apart)
6. Optionally also split if 2D screen-space radius exceeds `grow_scale2d`.
7. Execute clone first, then split (newly cloned Gaussians are NOT split).

**Phase 3 -- Prune (every `refine_every` steps, after grow):**
1. `is_prune = sigmoid(opacities) < prune_opa`
2. After first reset (`step > reset_every`): also prune if `max(exp(scales)) > prune_scale3d * scene_scale`
3. Optionally also prune if screen-space radius > `prune_scale2d`.

**Phase 4 -- Reset (every `reset_every` steps):**
1. Clamp all opacities so that `sigmoid(opacity) <= prune_opa * 2.0`.
2. Zero the optimizer state for opacities.

```python
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union
import mlx.core as mx

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split, _scatter_add, _scatter_max


@dataclass
class DefaultStrategy(Strategy):
    """Default densification strategy from the original 3DGS paper.

    '3D Gaussian Splatting for Real-Time Radiance Field Rendering'
    (Kerbl et al., 2023, arXiv:2308.04079)

    Lifecycle during training:
    1. step_pre_backward: Validate that info dict contains gradient keys.
       In MLX, since we use functional gradients, the info dict must contain
       pre-computed gradient information from the rasterization forward pass.
    2. step_post_backward: Accumulate gradient stats. Periodically grow/prune/reset.

    If absgrad=True, uses absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:
    'AbsGS: Recovering Fine Details for 3D Gaussian Splatting'
    (arXiv:2404.10484). Typically leads to better results but requires
    setting grow_grad2d to a higher value (~0.0008). The rasterization
    call must also use absgrad=True.
    """

    # --- Pruning thresholds ---
    prune_opa: float = 0.005
    """Gaussians with sigmoid(opacity) < prune_opa are pruned."""

    prune_scale3d: float = 0.1
    """Gaussians with max 3D scale > prune_scale3d * scene_scale are pruned
    (only after the first opacity reset, i.e., step > reset_every)."""

    prune_scale2d: float = 0.15
    """Gaussians with max 2D radius (normalized by image resolution) > prune_scale2d
    are pruned. Only active when step < refine_scale2d_stop_iter."""

    # --- Growing thresholds ---
    grow_grad2d: float = 0.0002
    """Gaussians with avg 2D gradient norm > grow_grad2d are candidates for clone/split.
    When using absgrad=True, increase to ~0.0008."""

    grow_scale3d: float = 0.01
    """Threshold to distinguish clone vs split. Gaussians with max 3D scale
    (normalized by scene_scale) <= grow_scale3d are cloned (under-reconstruction);
    those above are split (over-reconstruction)."""

    grow_scale2d: float = 0.05
    """Gaussians with 2D radius (normalized by image resolution) > grow_scale2d
    are also split. Only active when step < refine_scale2d_stop_iter."""

    # --- Iteration schedule ---
    refine_start_iter: int = 500
    """First iteration at which grow/prune can occur."""

    refine_stop_iter: int = 15_000
    """Last iteration at which grow/prune/accumulate occurs. After this,
    the Gaussian set is frozen and no more densification happens."""

    refine_every: int = 100
    """Grow/prune fires every this many steps (within the refine window)."""

    reset_every: int = 3000
    """Reset all opacities every this many steps."""

    pause_refine_after_reset: int = 0
    """Number of steps to pause refinement after an opacity reset.
    Useful to let opacities recover before the next grow/prune cycle.
    Set to num_training_images for best results. Default 0 = no pause.
    Checked via: step % reset_every >= pause_refine_after_reset."""

    refine_scale2d_stop_iter: int = 0
    """Stop using 2D screen-space scale criteria after this iteration.
    Default 0 = disabled (2D scale criteria never used).
    Set to a positive value to enable screen-space size pruning/splitting.
    Note: the original 3DGS code has a bug that prevents this from working;
    we implement it correctly here but disable by default for compatibility."""

    # --- Algorithm variants ---
    absgrad: bool = False
    """Use absolute gradients (AbsGS, arXiv:2404.10484) instead of average.
    Typically produces better results but requires higher grow_grad2d (~0.0008).
    The rasterization call must also use absgrad=True."""

    revised_opacity: bool = False
    """Use revised opacity heuristic from arXiv:2404.06109 during splits.
    Formula: new_opa = 1 - sqrt(1 - old_opa) instead of copying opacity.
    Experimental."""

    verbose: bool = False
    """Print grow/prune/reset statistics to stdout."""

    key_for_gradient: str = "means2d"
    """Key in info dict for the 2D gradient tensor.
    3DGS uses 'means2d', 2DGS uses 'gradient_2dgs'."""

    # ---- Initialization ----

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize running state. Actual tensor allocation is deferred to first step.

        Args:
            scene_scale: Scale factor for the scene. Used to normalize 3D scale
                thresholds (grow_scale3d, prune_scale3d are multiplied by this).

        Returns:
            State dict with:
            - grad2d: None (will become [N] float32 -- accumulated gradient norms)
            - count: None (will become [N] float32 -- per-Gaussian visibility count)
            - scene_scale: float -- used to normalize 3D scale thresholds
            - radii: None (optional, only if refine_scale2d_stop_iter > 0)
        """
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
    ):
        """Check params contain required keys: means, scales, quats, opacities."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    # ---- Training loop callbacks ----

    def step_pre_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Pre-backward hook.

        In PyTorch upstream, this calls retain_grad() on means2d so gradients
        are available after backward. In MLX, gradients are computed functionally
        via mx.value_and_grad(), so this is effectively a validation step.

        The caller must ensure that after the backward pass, the info dict
        contains gradient information keyed as '{key_for_gradient}_grad'.

        The info dict must contain (populated by rasterization):
        - self.key_for_gradient: the 2D projected means
        - "radii": per-Gaussian radii [C, N, 2] or [nnz]
        - "gaussian_ids": [nnz] (if packed mode)
        - "width", "height": image dimensions
        - "n_cameras": number of cameras in batch
        """
        assert self.key_for_gradient in info, (
            f"info must contain '{self.key_for_gradient}' for gradient tracking."
        )

    def step_post_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Post-backward hook. Accumulates gradient stats and periodically refines.

        Called every training step. The refinement (grow/prune/reset) only fires
        on specific iterations based on the schedule parameters.

        Args:
            packed: If True, info contains packed format (sparse intersection lists)
                with grads shape [nnz, 2]. If False, info contains dense [C, N, 2] arrays.

        The info dict must contain:
        - "{key_for_gradient}_grad": mx.array -- the 2D gradient of the loss
          w.r.t. the projected 2D means. Shape [C, N, 2] (dense) or [nnz, 2] (packed).
          If absgrad=True, this should be the absolute gradient.
        - "radii": [C, N, 2] or [nnz] -- 2D radii of each Gaussian in pixels
        - "gaussian_ids": [nnz] int32 (packed mode only)
        - "width", "height", "n_cameras": ints
        """
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # Grow: clone under-reconstructed, split over-reconstructed
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                n_total = params["means"].shape[0]
                print(
                    f"Step {step}: {n_dupli} duplicated, {n_split} split. "
                    f"Total: {n_total} Gaussians."
                )

            # Prune: remove low-opacity and oversized
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                n_total = params["means"].shape[0]
                print(
                    f"Step {step}: {n_prune} pruned. Total: {n_total} Gaussians."
                )

            # Reset gradient accumulators for next window
            state["grad2d"] = mx.zeros_like(state["grad2d"])
            state["count"] = mx.zeros_like(state["count"])
            if self.refine_scale2d_stop_iter > 0 and "radii" in state:
                state["radii"] = mx.zeros_like(state["radii"])

        # Periodic opacity reset
        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    # ---- Internal methods ----

    def _update_state(
        self,
        params: Dict[str, mx.array],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Accumulate per-Gaussian gradient statistics from the current training step.

        This tracks:
        - grad2d[i]: sum of 2D gradient norms for Gaussian i across all pixels/cameras
        - count[i]: number of times Gaussian i was visible (contributed to a pixel)
        - radii[i]: maximum 2D radius of Gaussian i (optional, for screen-size criteria)

        These are accumulated across steps within a refine_every window and averaged
        when grow/prune decisions are made.
        """
        required_keys = ["width", "height", "n_cameras", "radii", "gaussian_ids"]
        grad_key = f"{self.key_for_gradient}_grad"
        required_keys.append(grad_key)
        for key in required_keys:
            assert key in info, f"{key} is required in info but missing."

        # Get gradients
        if self.absgrad:
            grads = mx.abs(info[grad_key])
        else:
            grads = info[grad_key]

        # Normalize gradients to [-1, 1] screen space
        scale_x = info["width"] / 2.0 * info["n_cameras"]
        scale_y = info["height"] / 2.0 * info["n_cameras"]
        grads_x = grads[..., 0:1] * scale_x
        grads_y = grads[..., 1:2] * scale_y
        grads_scaled = mx.concatenate([grads_x, grads_y], axis=-1)

        # Initialize state tensors on first call (deferred allocation)
        n_gaussian = params[list(params.keys())[0]].shape[0]

        if state["grad2d"] is None:
            state["grad2d"] = mx.zeros((n_gaussian,))
        if state["count"] is None:
            state["count"] = mx.zeros((n_gaussian,))
        if self.refine_scale2d_stop_iter > 0 and state.get("radii") is None:
            state["radii"] = mx.zeros((n_gaussian,))

        # Accumulate based on format
        if packed:
            # Packed format: grads is [nnz, 2], gaussian_ids is [nnz]
            gs_ids = info["gaussian_ids"]  # [nnz]
            grad_norms = mx.sqrt(mx.sum(grads_scaled ** 2, axis=-1))  # [nnz]
            radii = info["radii"]
            if radii.ndim > 1:
                radii = mx.max(radii, axis=-1)  # [nnz]

            state["grad2d"] = _scatter_add(state["grad2d"], gs_ids, grad_norms)
            state["count"] = _scatter_add(
                state["count"], gs_ids, mx.ones_like(grad_norms)
            )

            if self.refine_scale2d_stop_iter > 0:
                norm_radii = radii / float(max(info["width"], info["height"]))
                state["radii"] = _scatter_max(state["radii"], gs_ids, norm_radii)
        else:
            # Dense format: grads is [C, N, 2], radii is [C, N, 2]
            # Select visible Gaussians: those with all radii > 0
            visible = mx.all(info["radii"] > 0.0, axis=-1)  # [C, N] bool
            # Get the N-dimension indices of visible Gaussians
            gs_ids_2d = mx.where(visible)
            gs_ids = gs_ids_2d[1]  # [nnz] -- the Gaussian indices
            grads_visible = grads_scaled[visible]  # [nnz, 2]
            grad_norms = mx.sqrt(mx.sum(grads_visible ** 2, axis=-1))  # [nnz]
            radii_visible = mx.max(info["radii"][visible], axis=-1)  # [nnz]

            state["grad2d"] = _scatter_add(state["grad2d"], gs_ids, grad_norms)
            state["count"] = _scatter_add(
                state["count"], gs_ids, mx.ones_like(grad_norms)
            )

            if self.refine_scale2d_stop_iter > 0:
                norm_radii = radii_visible / float(max(info["width"], info["height"]))
                state["radii"] = _scatter_max(state["radii"], gs_ids, norm_radii)

    def _grow_gs(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """Grow Gaussians by cloning small high-gradient ones and splitting large ones.

        Algorithm:
        1. Compute average gradient: grad2d / max(count, 1)
        2. is_grad_high = avg_grad > grow_grad2d
        3. is_small = max(exp(scales)) <= grow_scale3d * scene_scale
        4. Clone mask = is_grad_high AND is_small
        5. Split mask = is_grad_high AND NOT is_small
        6. Optionally: split |= (radii > grow_scale2d) if before refine_scale2d_stop_iter
        7. Execute clone, then extend split mask with zeros for new clones, then split

        Returns:
            (n_duplicated, n_split)
        """
        count = state["count"]
        # Average gradient over accumulation window
        grads = state["grad2d"] / mx.maximum(count, mx.array(1.0))

        # Thresholds
        is_grad_high = grads > self.grow_grad2d

        # 3D scale criterion: max scale across xyz
        scales_3d = mx.max(mx.exp(params["scales"]), axis=-1)  # [N]
        scale_threshold = self.grow_scale3d * state["scene_scale"]
        is_small = scales_3d <= scale_threshold

        # Clone: high gradient + small scale (under-reconstruction)
        is_dupli = is_grad_high & is_small
        n_dupli = int(mx.sum(is_dupli).item())

        # Split: high gradient + large scale (over-reconstruction)
        is_large = ~is_small
        is_split = is_grad_high & is_large

        # Optional: also split based on 2D screen-space size
        if step < self.refine_scale2d_stop_iter:
            is_split = is_split | (state["radii"] > self.grow_scale2d)

        n_split = int(mx.sum(is_split).item())

        # Execute clone first
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # Extend the split mask to account for newly cloned Gaussians
        # Cloned Gaussians (appended at end) should NOT be split
        is_split = mx.concatenate([
            is_split,
            mx.zeros((n_dupli,), dtype=mx.bool_),
        ])

        # Execute split (operates on the extended array including clones)
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )

        return n_dupli, n_split

    def _prune_gs(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Prune Gaussians with low opacity or extreme scale.

        Algorithm:
        1. is_prune = sigmoid(opacities) < prune_opa
        2. If step > reset_every (after first opacity reset):
           a. is_too_big = max(exp(scales)) > prune_scale3d * scene_scale
           b. Optionally: is_too_big |= (radii > prune_scale2d) if before refine_scale2d_stop_iter
           c. is_prune |= is_too_big
        3. Remove all pruned Gaussians

        Returns:
            Number of Gaussians pruned.
        """
        # Low opacity criterion
        opacities = mx.sigmoid(params["opacities"].reshape(-1))
        is_prune = opacities < self.prune_opa

        # After the first opacity reset, also prune oversized Gaussians
        if step > self.reset_every:
            scales_3d = mx.max(mx.exp(params["scales"]), axis=-1)
            is_too_big = scales_3d > self.prune_scale3d * state["scene_scale"]

            # Optional screen-size pruning
            # Note: disabled by default (refine_scale2d_stop_iter=0) because
            # the original 3DGS code has a bug that prevents it from working.
            # See: https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            if step < self.refine_scale2d_stop_iter:
                is_too_big = is_too_big | (state["radii"] > self.prune_scale2d)

            is_prune = is_prune | is_too_big

        n_prune = int(mx.sum(is_prune).item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
```

---

### 4. `strategy/mcmc.py` -- MCMCStrategy (P2 Stub)

```python
from dataclasses import dataclass
from typing import Any, Dict
import mlx.core as mx

from .base import Strategy


@dataclass
class MCMCStrategy(Strategy):
    """MCMC-based densification strategy (stub -- not yet implemented).

    From: '3D Gaussian Splatting as Markov Chain Monte Carlo'
    (Kheradmand et al., 2024, arXiv:2404.09591)

    This strategy:
    - Periodically teleports low-opacity Gaussians to high-opacity regions
    - Periodically adds new Gaussians sampled from the opacity distribution
    - Periodically injects noise into positions for MCMC exploration

    Full implementation deferred to a separate PRD.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    noise_injection_stop_iter: int = -1
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        raise NotImplementedError("MCMCStrategy is not yet ported to MLX.")

    def step_post_backward(self, *args, **kwargs):
        raise NotImplementedError("MCMCStrategy is not yet ported to MLX.")
```

---

### 5. `strategy/__init__.py`

```python
from .base import Strategy
from .default import DefaultStrategy
from .mcmc import MCMCStrategy

__all__ = ["Strategy", "DefaultStrategy", "MCMCStrategy"]
```

---

## Data Flow

### Training Loop Integration

```
for step in range(max_steps):
    # Forward pass
    renders, alphas, info = rasterization(params, cameras, ...)

    # Strategy pre-backward (validation only in MLX)
    strategy.step_pre_backward(params, optimizers, state, step, info)

    # Compute loss
    loss = compute_loss(renders, gt_images)

    # Backward pass (MLX functional style)
    loss_val, grads = mx.value_and_grad(loss_fn)(params)

    # Inject gradient info into info dict for strategy
    info["means2d_grad"] = grads["means2d"]  # or from rasterization backward

    # Strategy post-backward (may grow/prune/reset Gaussians)
    strategy.step_post_backward(params, optimizers, state, step, info)

    # Optimizer step (must handle potentially resized params)
    optimizer.update(params, grads)
```

### Gradient Flow for Strategy

Unlike PyTorch where `retain_grad()` captures intermediate gradients during autograd, MLX computes gradients functionally. The rasterization backward pass must explicitly provide the gradient of the loss w.r.t. `means2d` in the `info` dict, keyed as `"{key_for_gradient}_grad"`.

```
rasterization forward
    |
    v
info["means2d"] = projected 2D means  (from forward pass)
    |
    v
mx.value_and_grad(loss_fn)(params)
    |
    v
info["means2d_grad"] = d(loss)/d(means2d)  (from backward pass)
    |
    v
strategy.step_post_backward()
    -> _update_state(): accumulates gradient norms per Gaussian
    -> _grow_gs(): clone/split decisions every refine_every steps
    -> _prune_gs(): remove low-quality Gaussians
    -> reset_opa(): periodic opacity reset
```

---

## Edge Cases and Correctness Constraints

### 1. Empty masks
All ops must handle the case where `mask.sum() == 0` (no Gaussians selected). The functions return early without modifying anything.

### 2. All-True masks in remove
If `remove` is called with all Gaussians masked for removal, the result is empty parameter arrays (shape `[0, ...]`). This should not crash, but the training loop should check for zero-Gaussian edge cases.

### 3. State tensor size tracking
After `duplicate` or `split`, the state tensors (`grad2d`, `count`, `radii`) are resized internally by the ops. The state dict is modified in-place, so callers always see the updated sizes.

### 4. Order of operations in grow
Clone MUST happen before split. After cloning, the split mask is extended with `n_dupli` False entries so that newly cloned Gaussians (appended at the end) are not immediately split. This matches upstream behavior exactly.

### 5. Scale factor in split
New scales after split are `log(exp(old_scales) / 1.6)` = `old_scales - log(1.6)`. The constant 1.6 comes from the original 3DGS paper and controls how much smaller the children are.

### 6. Opacity reset value
Reset clamps to `prune_opa * 2.0` in post-sigmoid space (= `logit(prune_opa * 2.0)` in logit space). This ensures recently reset Gaussians are not immediately pruned in the next refinement step, since they remain above the prune threshold.

### 7. Gradient normalization
Gradients are scaled by `width/2 * n_cameras` and `height/2 * n_cameras` to normalize to [-1, 1] screen space. The `n_cameras` factor accounts for multi-view batching.

### 8. Pause after reset
If `pause_refine_after_reset > 0`, refinement is paused for that many steps after each opacity reset, checked via `step % reset_every >= pause_refine_after_reset`. This gives opacities time to recover from the reset before the next grow/prune decision.

### 9. refine_scale2d_stop_iter semantics
When set to 0 (default), all 2D scale-based criteria are disabled -- no `radii` state is tracked, and neither split nor prune use screen-space size. When positive, 2D screen-size criteria for both splitting and pruning are active only for steps before this iteration.

### 10. mx.eval() placement
MLX uses lazy evaluation. After structural changes (duplicate/split/remove), callers should call `mx.eval()` on modified arrays to materialize results before the next operation that depends on array shapes. The ops themselves do NOT call `mx.eval()` -- that responsibility belongs to the training loop.

### 11. Gaussian count arithmetic
- `duplicate(mask)`: N -> N + sum(mask)
- `split(mask)`: N -> N - sum(mask) + 2*sum(mask) = N + sum(mask)
- `remove(mask)`: N -> N - sum(mask)
- Combined grow step: N -> N + n_dupli + n_split (clone adds n_dupli, split adds n_split net)

### 12. State consistency after combined operations
When `_grow_gs` runs clone then split sequentially, the state arrays are resized by each operation. The split mask must be extended to match the post-clone array size. The state arrays after both operations have the correct size for the pruning step that follows.

### 13. Numerical stability in logit
The `_logit` function clamps inputs to `[eps, 1-eps]` to avoid `log(0)` or `log(inf)`. The upstream uses `torch.logit` which has similar internal clamping.

---

## Files to Create

| File | Description | Lines (est.) |
|------|-------------|-------------|
| `src/gsplat_mlx/strategy/__init__.py` | Re-exports | ~5 |
| `src/gsplat_mlx/strategy/base.py` | Strategy ABC | ~60 |
| `src/gsplat_mlx/strategy/default.py` | DefaultStrategy (full implementation) | ~320 |
| `src/gsplat_mlx/strategy/mcmc.py` | MCMCStrategy (P2 stub) | ~40 |
| `src/gsplat_mlx/strategy/ops.py` | Low-level ops + helpers | ~280 |
| `tests/test_strategy.py` | Comprehensive tests | ~450 |

---

## Test Plan

### File: `tests/test_strategy.py`

| Test Case | Description | Validates |
|-----------|-------------|-----------|
| **Ops: duplicate** | | |
| `test_duplicate_increases_count` | 100 Gaussians, duplicate 10 -> 110 total | Array shapes, count |
| `test_duplicate_copies_values` | Duplicated Gaussians have identical param values to originals | Value correctness |
| `test_duplicate_zeros_optimizer_state` | New Gaussians have zero exp_avg / exp_avg_sq | Optimizer state update |
| `test_duplicate_updates_running_state` | grad2d, count arrays are extended with duplicated values | State consistency |
| `test_duplicate_empty_mask` | All-False mask -> no change to any arrays | Edge case |
| **Ops: remove** | | |
| `test_remove_decreases_count` | 100 Gaussians, remove 20 -> 80 total | Array shapes |
| `test_remove_preserves_kept` | Remaining Gaussians have correct values and order | Value correctness |
| `test_remove_updates_optimizer_state` | Optimizer state arrays are correctly reduced | Optimizer state |
| `test_remove_all` | Remove all Gaussians -> shape [0, ...] arrays | Edge case |
| `test_remove_none` | All-False mask -> no change | Edge case |
| **Ops: split** | | |
| `test_split_produces_two_per` | Split 5 from 100 -> 105 total (remove 5, add 10) | Count arithmetic |
| `test_split_reduces_scale` | New log-scales = old_log_scales - log(1.6) | Scale reduction |
| `test_split_offsets_means` | New means differ from original (covariance samples) | Position displacement |
| `test_split_zeros_optimizer_state` | All split children have zeroed optimizer state | Optimizer state |
| `test_split_revised_opacity` | With revised_opacity=True, verify new_opa = 1 - sqrt(1 - old_opa) | Opacity formula |
| `test_split_preserves_other_params` | Non-means/scales/opacities params (quats, sh) are copied | Copy correctness |
| `test_split_keeps_unsplit_intact` | Non-split Gaussians are unchanged (just reindexed) | Non-interference |
| **Ops: reset_opa** | | |
| `test_reset_opa_clamps` | After reset(0.01), all sigmoid(opacities) <= 0.01 | Clamping |
| `test_reset_opa_zeros_optimizer` | Opacity optimizer state (exp_avg, exp_avg_sq) is zeroed | Optimizer state |
| `test_reset_opa_preserves_low` | Opacities already below threshold are unchanged | Conservative |
| `test_reset_opa_leaves_other_params` | Non-opacity params and their optimizer state unchanged | Non-interference |
| **Helpers** | | |
| `test_logit_roundtrip` | sigmoid(logit(x)) == x for x in [0.01, 0.5, 0.99] | Numerical helper |
| `test_quat_to_rotmat_identity` | Identity quaternion [1,0,0,0] -> identity matrix | Rotation helper |
| `test_quat_to_rotmat_90deg` | Known 90-degree rotation quaternion -> correct matrix | Rotation helper |
| `test_scatter_add_basic` | Simple scatter-add matches numpy reference | Scatter op |
| `test_scatter_add_duplicates` | Duplicate indices accumulate correctly | Scatter op |
| `test_scatter_max_basic` | Simple scatter-max matches numpy reference | Scatter op |
| **DefaultStrategy: initialization** | | |
| `test_initialize_state` | State has grad2d=None, count=None, scene_scale | Initialization |
| `test_initialize_state_with_radii` | refine_scale2d_stop_iter > 0 -> state has "radii" | Conditional init |
| `test_check_sanity_pass` | Valid params+optimizers pass | Sanity check |
| `test_check_sanity_missing_param` | Missing "means" raises AssertionError | Sanity check |
| `test_check_sanity_extra_optimizer` | Optimizer key not in params raises AssertionError | Sanity check |
| **DefaultStrategy: grow** | | |
| `test_grow_clone` | High gradient + small scale -> Gaussians cloned | Clone logic |
| `test_grow_split` | High gradient + large scale -> Gaussians split | Split logic |
| `test_grow_both` | Mix of clone and split candidates, correct ordering | Combined logic |
| `test_grow_no_action` | Low gradient -> no grow | No false positives |
| `test_grow_clones_not_split` | Newly cloned Gaussians are not immediately split | Clone-split ordering |
| **DefaultStrategy: prune** | | |
| `test_prune_low_opacity` | sigmoid(opa) < 0.005 -> removed | Opacity prune |
| `test_prune_large_scale` | Oversized Gaussians pruned after first reset | Scale prune |
| `test_prune_no_scale_before_reset` | Scale pruning inactive before step > reset_every | Schedule guard |
| **DefaultStrategy: schedule** | | |
| `test_no_refine_before_start` | No grow/prune before refine_start_iter=500 | Schedule guard |
| `test_no_refine_after_stop` | No grow/prune after refine_stop_iter=15000 | Schedule guard |
| `test_refine_every` | Grow/prune only fires on steps divisible by refine_every | Schedule period |
| `test_reset_every` | Opacity reset fires on steps divisible by reset_every | Reset schedule |
| `test_pause_after_reset` | Refinement pauses for pause_refine_after_reset steps | Pause logic |
| **DefaultStrategy: integration** | | |
| `test_lifecycle_1000_steps` | Full 1000-step loop with periodic grow/prune/reset | End-to-end |
| `test_state_accumulation` | Gradient norms and counts accumulate correctly across steps | Accumulator |
| `test_state_reset_after_refine` | grad2d and count are zeroed after grow/prune | State lifecycle |
| **MCMCStrategy** | | |
| `test_mcmc_stub_initialize_raises` | initialize_state raises NotImplementedError | Stub behavior |
| `test_mcmc_stub_step_raises` | step_post_backward raises NotImplementedError | Stub behavior |
| `test_mcmc_has_all_params` | All 8 dataclass fields present with defaults | API surface |

### Test Utilities

```python
def make_test_params(n: int = 100) -> Dict[str, mx.array]:
    """Create a minimal set of Gaussian parameters for testing."""
    return {
        "means": mx.random.normal(shape=(n, 3)),
        "scales": mx.random.normal(shape=(n, 3)) - 2.0,  # log-space, small
        "quats": mx.concatenate([
            mx.ones((n, 1)),
            mx.zeros((n, 3)),
        ], axis=1),  # identity quaternions
        "opacities": mx.zeros((n,)),  # logit(0.5) = 0
    }

def make_test_optimizers(params: Dict[str, mx.array]) -> Dict[str, Dict[str, mx.array]]:
    """Create mock Adam optimizer state for each parameter."""
    optimizers = {}
    for name, p in params.items():
        optimizers[name] = {
            "exp_avg": mx.zeros_like(p),
            "exp_avg_sq": mx.zeros_like(p),
            "step": 0,
        }
    return optimizers

def make_test_info(
    n: int = 100,
    width: int = 800,
    height: int = 600,
    n_cameras: int = 1,
) -> Dict[str, Any]:
    """Create mock rasterization info dict for testing."""
    return {
        "means2d": mx.random.normal(shape=(n_cameras, n, 2)),
        "means2d_grad": mx.random.normal(shape=(n_cameras, n, 2)) * 0.001,
        "radii": mx.ones((n_cameras, n, 2)) * 10.0,
        "gaussian_ids": mx.arange(n),
        "width": width,
        "height": height,
        "n_cameras": n_cameras,
    }
```

---

## Dependencies

| Dependency | PRD | What it provides |
|------------|-----|-----------------|
| Dev environment | PRD-01 | MLX, pytest, project structure |
| Math utils | PRD-02 | Quaternion operations (may share `normalized_quat_to_rotmat`) |
| Rendering API | PRD-09 | `info` dict format from rasterization (means2d, radii, gaussian_ids) |

---

## Blocks

| Downstream | PRD | What it needs from this PRD |
|------------|-----|---------------------------|
| Training loop | PRD-13 | `DefaultStrategy` for full training pipeline |
| MCMC strategy | Future PRD | `Strategy` base class, `ops.py` infrastructure |

---

## Acceptance Criteria

- [ ] `Strategy` base class defines check_sanity, initialize_state, step_pre_backward, step_post_backward
- [ ] `DefaultStrategy` implements complete grow/prune/reset lifecycle matching upstream behavior
- [ ] All 16 dataclass parameters from upstream `DefaultStrategy` are present with correct defaults
- [ ] `duplicate` correctly appends cloned Gaussians and zeros optimizer state
- [ ] `remove` correctly filters out masked Gaussians from params + optimizer state + running state
- [ ] `split` produces 2 new Gaussians per split with reduced scale, offset means, zeroed optimizer state
- [ ] `reset_opa` clamps opacities to `logit(value)` and zeros opacity optimizer state only
- [ ] `_update_param_with_optimizer` correctly updates both params and all optimizer state entries
- [ ] `_update_state` correctly accumulates gradient norms via scatter-add and handles both packed and dense formats
- [ ] `_grow_gs` implements correct clone-then-split ordering with mask extension for newly cloned Gaussians
- [ ] `_prune_gs` implements opacity + scale + optional screen-size pruning with correct schedule guards
- [ ] Opacity reset fires every `reset_every` steps with value `prune_opa * 2.0`
- [ ] `MCMCStrategy` stub exists with `NotImplementedError` and all 8 dataclass parameters
- [ ] All edge cases handled: empty masks, all-true masks, zero Gaussians
- [ ] Scatter helpers (`_scatter_add`, `_scatter_max`) handle duplicate indices correctly
- [ ] `_logit` and `_normalized_quat_to_rotmat` helpers are numerically stable
- [ ] All tests in `tests/test_strategy.py` pass with `pytest tests/test_strategy.py -v`
- [ ] No PyTorch or CUDA dependencies in any strategy file
