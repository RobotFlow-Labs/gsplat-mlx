# PRD-11: SelectiveAdam Optimizer

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-11 |
| **Title** | SelectiveAdam Optimizer |
| **Status** | DRAFT |
| **Priority** | P1 -- Training Path |
| **Estimated Effort** | 3--5 hours |
| **Dependencies** | PRD-01 (project foundation) |
| **Blocks** | PRD-13 (training loop) |
| **Owner** | AIFLOW LABS |
| **Created** | 2026-03-15 |

---

## 1. Objective

Implement a **SelectiveAdam** optimizer for MLX that only updates the Adam state (first moment, second moment) and parameters for Gaussians that were actually rendered (visible) in the current frame. This is a direct port of gsplat's `SelectiveAdam` class from `gsplat/optimizers/selective_adam.py`, which uses a fused CUDA kernel (`AdamCUDA.cu`) for the selective update.

After this PRD is implemented, an engineer should be able to:

1. Create a `SelectiveAdam` optimizer with standard Adam hyperparameters
2. Pass a per-Gaussian visibility mask to each optimization step
3. Have only the visible Gaussians' parameters and optimizer state updated
4. Use the optimizer with parameters of any shape (`[N]`, `[N, 3]`, `[N, 4]`, `[N, K, 3]`)
5. Resize optimizer state during densification (clone/split/prune)
6. Integrate seamlessly into a 3DGS training loop (PRD-13)

---

## 2. Context & Motivation

### 2.1 Why Selective Updates?

In 3D Gaussian Splatting, a scene is represented by N Gaussians (typically 100K--5M). For any given camera viewpoint, only a subset of Gaussians are visible -- they fall within the camera frustum, pass depth culling, and contribute non-negligible alpha to at least one pixel. Gaussians behind the camera, occluded, or outside the frustum are invisible and receive zero gradients.

Standard Adam would still update the moving averages for invisible Gaussians:

```
# Standard Adam: invisible Gaussians have g_t = 0
m_t = beta1 * m_{t-1} + (1 - beta1) * 0  = beta1 * m_{t-1}  # momentum decays!
v_t = beta2 * v_{t-1} + (1 - beta2) * 0  = beta2 * v_{t-1}  # variance decays!
```

This is problematic because:

1. **Momentum decay**: The first moment `m_t` decays toward zero for invisible Gaussians, effectively "forgetting" the gradient history. When the Gaussian becomes visible again, Adam treats it as if it had near-zero gradient history, leading to large initial updates.
2. **Variance decay**: The second moment `v_t` also decays, which reduces the adaptive learning rate denominator, causing even larger updates when the Gaussian reappears.
3. **Wasted compute**: Updating state for invisible Gaussians wastes memory bandwidth and compute.
4. **Training instability**: The combined effect of decayed moments causes visible "popping" artifacts when Gaussians transition from invisible to visible across training iterations.

SelectiveAdam solves this by simply skipping the update entirely for invisible Gaussians, preserving their optimizer state frozen until they become visible again.

### 2.2 The Taming 3DGS Connection

SelectiveAdam is one of the two optimizers introduced in the "Taming 3D Gaussian Splatting" paper (Mallick et al., 2024). The paper demonstrates that selective updates significantly improve training stability and final quality, particularly for large scenes where Gaussians frequently enter and leave the view frustum.

### 2.3 Upstream Implementation

The upstream gsplat implementation consists of:

- **Python class**: `gsplat/optimizers/selective_adam.py` -- `SelectiveAdam(torch.optim.Adam)` that overrides `step(visibility)`.
- **CUDA kernel**: `gsplat/cuda/csrc/AdamCUDA.cu` -- a fused kernel that computes the Adam update with an early-return for invisible Gaussians (line 53: `if (valid != nullptr && !valid[g_idx]) return;`).
- **Wrapper**: `gsplat/cuda/_wrapper.py:adam()` -- the Python-to-CUDA bridge.

The CUDA kernel operates on flattened parameter tensors. Each thread handles one scalar element. The visibility mask is indexed by `g_idx = p_idx / D` where `D` is the number of elements per Gaussian (e.g., 3 for means, 4 for quaternions). This means visibility is per-Gaussian, not per-element.

Key observation: **the upstream does NOT use bias correction**. The step counter is not incremented and the raw `m_t / (sqrt(v_t) + eps)` is used directly. This matches the common practice in 3DGS training where bias correction is typically omitted.

### 2.4 Algorithm: Selective Adam

Standard Adam update:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # first moment
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # second moment
m_hat_t = m_t / (1 - beta1^t)                        # bias correction (optional)
v_hat_t = v_t / (1 - beta2^t)                        # bias correction (optional)
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)  # parameter update
```

Selective modification:

```
# Only update where visibility=True:
m_t = where(visible, beta1*m_{t-1} + (1-beta1)*g_t, m_{t-1})
v_t = where(visible, beta2*v_{t-1} + (1-beta2)*g_t^2, v_{t-1})
# Only apply parameter update where visible
theta_t = where(visible, theta_{t-1} - lr * m_t / (sqrt(v_t) + eps), theta_{t-1})
```

---

## 3. Scope

### 3.1 In Scope

| Deliverable | Description |
|-------------|-------------|
| `src/gsplat_mlx/optimizers/__init__.py` | Public exports: `SelectiveAdam`, `adam` |
| `src/gsplat_mlx/optimizers/selective_adam.py` | Full `SelectiveAdam` class + standalone `adam()` function |
| `tests/test_optimizer.py` | Comprehensive test suite (15 tests across 7 test classes) |

### 3.2 Out of Scope

- Integration with `mlx.optimizers.Optimizer` base class (we implement standalone for clarity and control)
- Learning rate scheduling (can be added later by wrapping)
- Weight decay / AdamW variant (not used in upstream SelectiveAdam)
- Metal shader implementation (pure MLX array ops for MVP)
- Gradient clipping (handled externally by the training loop)
- Multi-GPU / distributed training

---

## 4. Technical Design

### 4.1 Architecture Decision: Standalone vs. Subclass

Three options were considered:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| 1 | Subclass `mlx.optimizers.Adam` and override `apply_single` | Reuses MLX infrastructure | `apply_single` signature has no visibility param; MLX Adam uses `tree_map` which doesn't support per-step masks |
| 2 | Implement from scratch as standalone class | Full control, clear code, easy to test | Must implement Adam from scratch |
| 3 | Use `mlx.optimizers.Adam` and post-process with mask | Minimal code | Two passes over data; state management is fragmented; cannot prevent momentum decay for invisible params |

**Decision: Option 2 -- standalone implementation.** Rationale:

- The visibility mask fundamentally changes the update semantics (not just a post-filter)
- MLX's `apply_single` operates per-parameter via `tree_map`, but we need to broadcast the per-Gaussian visibility mask across parameter dimensions
- A standalone implementation is ~100 lines and is easy to understand, test, and maintain
- The upstream is also standalone (subclasses `torch.optim.Adam` for state management only, not for the actual update logic -- the CUDA kernel does all the work)

### 4.2 API Design

```python
class SelectiveAdam:
    """Adam optimizer with selective per-Gaussian updates.

    Args:
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages of gradient
               and its square. Default: (0.9, 0.999).
        eps: Term added to denominator for numerical stability. Default: 1e-8.
        bias_correction: Whether to apply bias correction to moment estimates.
                        Default: False (matches upstream gsplat behavior).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        bias_correction: bool = False,
    ) -> None: ...

    def step(
        self,
        params: dict[str, mx.array],
        grads: dict[str, mx.array],
        visibility: mx.array,
    ) -> dict[str, mx.array]: ...

    @property
    def state(self) -> dict[str, dict]: ...

    def reset_state(self, name: str | None = None) -> None: ...

    def resize_state(
        self,
        name: str,
        new_n: int,
        indices: mx.array | None = None,
    ) -> None: ...
```

### 4.3 Parameter and State Layout

In 3DGS, each Gaussian has multiple associated parameters with different shapes:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `means` | `[N, 3]` | 3D position |
| `quats` | `[N, 4]` | Rotation quaternion |
| `scales` | `[N, 3]` | Log-scale in each axis |
| `opacities` | `[N]` | Sigmoid-inverse opacity |
| `sh0` | `[N, 1, 3]` | DC spherical harmonic coefficient |
| `shN` | `[N, K, 3]` | Higher-order SH coefficients (K depends on degree) |

The visibility mask is always `[N]` (one boolean per Gaussian). For each parameter, we need to broadcast visibility to match the parameter's shape:

| Param shape | Visibility broadcast | Method |
|-------------|---------------------|--------|
| `[N]` | `[N]` | No broadcast needed |
| `[N, D]` | `[N, 1]` | `visibility[:, None]` |
| `[N, K, D]` | `[N, 1, 1]` | `visibility[:, None, None]` |

General rule: `visibility.reshape([N] + [1] * (param.ndim - 1))`

Per-parameter optimizer state mirrors the parameter shape:

```python
state[name] = {
    "step": mx.array(0),           # scalar step counter
    "exp_avg": mx.zeros_like(p),    # first moment, same shape as param
    "exp_avg_sq": mx.zeros_like(p), # second moment, same shape as param
}
```

### 4.4 The Selective Update Algorithm

For each parameter `name` with value `p`, gradient `g`, and visibility mask `vis`:

```
1. Initialize state lazily if first call for this parameter
2. Increment step counter
3. Compute full Adam moment updates:
     new_m = beta1 * m + (1 - beta1) * g
     new_v = beta2 * v + (1 - beta2) * g^2
4. Apply selective mask:
     m = where(visible, new_m, m)    -- preserve old moments for invisible
     v = where(visible, new_v, v)    -- preserve old moments for invisible
5. Compute parameter update (with optional bias correction):
     update = lr * m / (sqrt(v) + eps)
6. Apply selective parameter update:
     p = where(visible, p - update, p)
```

**Key correctness properties:**

1. If `visibility[i] = False`, then `exp_avg[i]`, `exp_avg_sq[i]`, and `param[i]` are unchanged.
2. If `visibility[i] = True`, the standard Adam update is applied.
3. The step counter increments regardless of visibility (consistent with the upstream behavior where the step counter is global, not per-Gaussian).

**Matching upstream behavior**: The CUDA kernel does NOT use bias correction and does NOT maintain a step counter. It applies `param += -lr * exp_avg / (sqrt(exp_avg_sq) + eps)` directly. Our default (`bias_correction=False`) matches this.

### 4.5 Standalone `adam()` Function

For compatibility with the upstream API pattern and for use in lower-level code, we also provide a standalone function matching the signature of `gsplat.cuda._wrapper.adam()`:

```python
def adam(
    param: mx.array,       # [N, ...]
    param_grad: mx.array,  # [N, ...]
    exp_avg: mx.array,     # [N, ...]
    exp_avg_sq: mx.array,  # [N, ...]
    valid: mx.array,       # [N] bool
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Selective Adam update matching upstream gsplat API.

    Note: Unlike the upstream CUDA kernel which modifies arrays in-place,
    this returns new arrays since MLX arrays are immutable.

    Returns:
        (updated_param, updated_exp_avg, updated_exp_avg_sq)
    """
```

### 4.6 State Resizing for Densification

During 3DGS training, Gaussians are cloned, split, and pruned, changing N. The optimizer state must be resized to match. Two modes:

1. **Without indices** (full reset): zero-initialize all new state, preserve step counter
2. **With indices**: map new positions to old positions via an index array; entries with index `-1` get zero-initialized state (for newly created Gaussians from split/clone)

```python
def resize_state(self, name: str, new_n: int, indices: mx.array | None = None):
    """Resize optimizer state after densification.

    Args:
        name: Parameter name.
        new_n: New number of Gaussians.
        indices: [new_n] int array mapping new -> old indices.
                 -1 means zero-initialize (new Gaussian).
                 None means zero-initialize everything.
    """
```

---

## 5. Complete Implementation

### 5.1 File: `src/gsplat_mlx/optimizers/__init__.py`

```python
"""Optimizers for 3D Gaussian Splatting training."""

from .selective_adam import SelectiveAdam, adam

__all__ = ["SelectiveAdam", "adam"]
```

### 5.2 File: `src/gsplat_mlx/optimizers/selective_adam.py`

```python
"""SelectiveAdam optimizer -- Adam with per-Gaussian visibility masking.

This is the MLX port of gsplat's SelectiveAdam optimizer. In 3D Gaussian Splatting,
not all Gaussians are visible in every frame. Standard Adam would decay the momentum
and variance estimates for invisible Gaussians (since their gradients are zero),
causing training instability when they become visible again.

SelectiveAdam solves this by only updating the Adam state (exp_avg, exp_avg_sq) and
the parameters for Gaussians that are marked as visible.

Upstream reference:
    - gsplat/optimizers/selective_adam.py (Python class)
    - gsplat/cuda/csrc/AdamCUDA.cu (CUDA kernel)

The upstream CUDA kernel does NOT use bias correction. Our default matches this.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


class SelectiveAdam:
    """Adam optimizer with selective per-Gaussian updates.

    Only updates parameters and optimizer state for Gaussians marked as visible
    in the current frame. This prevents momentum/variance decay for non-visible
    Gaussians, improving training stability in 3D Gaussian Splatting.

    This is one of the two optimizers from the Taming3DGS paper.

    Args:
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients (beta1, beta2) for computing running averages of
               the gradient and its square. Default: (0.9, 0.999).
        eps: Term added to the denominator for numerical stability. Default: 1e-8.
        bias_correction: Whether to apply bias correction to moment estimates.
                        Default: False (matches upstream gsplat CUDA kernel).

    Example:
        >>> import mlx.core as mx
        >>> from gsplat_mlx.optimizers import SelectiveAdam
        >>>
        >>> N = 1000  # number of Gaussians
        >>> params = {
        ...     "means": mx.random.normal((N, 3)),
        ...     "scales": mx.random.normal((N, 3)),
        ...     "opacities": mx.random.normal((N,)),
        ... }
        >>> grads = {
        ...     "means": mx.random.normal((N, 3)),
        ...     "scales": mx.random.normal((N, 3)),
        ...     "opacities": mx.random.normal((N,)),
        ... }
        >>> visibility = mx.array([True] * 500 + [False] * 500)
        >>>
        >>> optimizer = SelectiveAdam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        >>> updated_params = optimizer.step(params, grads, visibility)
        >>> # Only the first 500 Gaussians' params are updated
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        bias_correction: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.bias_correction = bias_correction
        self._state: dict[str, dict[str, mx.array]] = {}

    @property
    def state(self) -> dict[str, dict[str, mx.array]]:
        """Read-only access to the optimizer state.

        Returns a dict mapping parameter names to their state dicts, each
        containing 'step', 'exp_avg', and 'exp_avg_sq'.
        """
        return self._state

    def reset_state(self, name: Optional[str] = None) -> None:
        """Reset optimizer state for a specific parameter or all parameters.

        This is useful during Gaussian densification (clone/split/prune) when
        the number of Gaussians changes and the optimizer state must be rebuilt.

        Args:
            name: If provided, reset only this parameter's state.
                  If None, reset all state.
        """
        if name is not None:
            if name in self._state:
                del self._state[name]
        else:
            self._state.clear()

    def resize_state(
        self,
        name: str,
        new_n: int,
        indices: Optional[mx.array] = None,
    ) -> None:
        """Resize optimizer state for a parameter after densification.

        During 3DGS training, Gaussians are cloned, split, and pruned, changing
        N. This method rebuilds the optimizer state for the new N. If indices
        are provided, the existing state at those indices is preserved and
        new entries are zero-initialized.

        Args:
            name: Parameter name whose state to resize.
            new_n: New number of Gaussians.
            indices: Optional [new_n] int array mapping new indices to old
                     indices. Entries with value -1 get zero-initialized state.
                     If None, all state is zero-initialized.
        """
        if name not in self._state:
            return

        old_state = self._state[name]
        old_m = old_state["exp_avg"]
        old_v = old_state["exp_avg_sq"]

        if indices is None:
            # Full reset: preserve step, zero out moments
            new_shape_m = (new_n,) + old_m.shape[1:]
            new_shape_v = (new_n,) + old_v.shape[1:]
            self._state[name] = {
                "step": old_state["step"],
                "exp_avg": mx.zeros(new_shape_m, dtype=old_m.dtype),
                "exp_avg_sq": mx.zeros(new_shape_v, dtype=old_v.dtype),
            }
        else:
            # Selective copy: gather from old state at given indices
            # indices[i] = -1 means zero-init (new Gaussian from split/clone)
            valid = indices >= 0
            safe_indices = mx.where(valid, indices, mx.zeros_like(indices))

            new_m = old_m[safe_indices]
            new_v = old_v[safe_indices]

            # Zero out entries where indices == -1
            vis_broadcast = _broadcast_visibility(valid, new_m)
            new_m = mx.where(vis_broadcast, new_m, mx.zeros_like(new_m))
            new_v = mx.where(vis_broadcast, new_v, mx.zeros_like(new_v))

            self._state[name] = {
                "step": old_state["step"],
                "exp_avg": new_m,
                "exp_avg_sq": new_v,
            }

    def step(
        self,
        params: dict[str, mx.array],
        grads: dict[str, mx.array],
        visibility: mx.array,
    ) -> dict[str, mx.array]:
        """Perform a selective Adam update step.

        For each parameter, the Adam state and parameter value are only updated
        for Gaussians where visibility is True. Non-visible Gaussians retain
        their previous parameter values and optimizer state.

        Args:
            params: Dict mapping parameter names to mx.array values.
                    Each array has shape [N, ...] where N is the number of
                    Gaussians.
            grads: Dict mapping parameter names to gradient mx.arrays.
                   Keys must be a subset of params keys. Params without
                   a gradient are returned unchanged.
            visibility: Boolean mask of shape [N] indicating which Gaussians
                       were visible in the current frame.

        Returns:
            Dict mapping parameter names to updated mx.array values.
            Same structure and shapes as the input params.

        Raises:
            ValueError: If a gradient key is not in params, or shapes mismatch.
        """
        updated_params = {}

        for name, grad in grads.items():
            if name not in params:
                raise ValueError(
                    f"Gradient key '{name}' not found in params. "
                    f"Available: {list(params.keys())}"
                )

            p = params[name]
            if p.shape != grad.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"param {p.shape} vs grad {grad.shape}"
                )

            vis_broadcast = _broadcast_visibility(visibility, p)
            updated_params[name] = self._update_single(name, p, grad, vis_broadcast)

        # Include any params that had no gradient (unchanged)
        for name, p in params.items():
            if name not in updated_params:
                updated_params[name] = p

        # Evaluate lazily -- force computation of params and state
        mx.eval(*[v for v in updated_params.values()])
        state_arrays = []
        for s in self._state.values():
            state_arrays.extend([s["exp_avg"], s["exp_avg_sq"], s["step"]])
        if state_arrays:
            mx.eval(*state_arrays)

        return updated_params

    def _update_single(
        self,
        name: str,
        p: mx.array,
        g: mx.array,
        vis_broadcast: mx.array,
    ) -> mx.array:
        """Apply selective Adam update to a single parameter.

        This implements the core algorithm:
        1. Compute full Adam moment updates
        2. Selectively apply via mx.where (only where visible)
        3. Compute parameter update
        4. Selectively apply parameter update (only where visible)

        Args:
            name: Parameter name (for state dictionary lookup).
            p: Parameter array of shape [N, ...].
            g: Gradient array of shape [N, ...], same shape as p.
            vis_broadcast: Visibility mask broadcast to match p's shape.

        Returns:
            Updated parameter array, same shape as p.
        """
        # Lazy state initialization
        if name not in self._state:
            self._state[name] = {
                "step": mx.array(0),
                "exp_avg": mx.zeros_like(p),
                "exp_avg_sq": mx.zeros_like(p),
            }

        state = self._state[name]
        state["step"] = state["step"] + 1

        m = state["exp_avg"]
        v = state["exp_avg_sq"]

        # Compute full Adam moment updates
        new_m = self.beta1 * m + (1 - self.beta1) * g
        new_v = self.beta2 * v + (1 - self.beta2) * (g * g)

        # Selective moment update: only where visible
        m = mx.where(vis_broadcast, new_m, m)
        v = mx.where(vis_broadcast, new_v, v)

        state["exp_avg"] = m
        state["exp_avg_sq"] = v

        # Compute parameter update
        if self.bias_correction:
            t = state["step"].astype(mx.float32)
            bc1 = 1 - self.beta1 ** t
            bc2 = 1 - self.beta2 ** t
            m_hat = m / bc1
            v_hat = v / bc2
            update = self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
        else:
            # No bias correction (matches upstream CUDA kernel)
            update = self.lr * m / (mx.sqrt(v) + self.eps)

        # Selective parameter update: only where visible
        new_p = p - update
        p = mx.where(vis_broadcast, new_p, p)

        return p


def _broadcast_visibility(visibility: mx.array, param: mx.array) -> mx.array:
    """Broadcast [N] visibility mask to match parameter shape.

    Args:
        visibility: Boolean mask of shape [N].
        param: Parameter array of shape [N, ...].

    Returns:
        Visibility mask reshaped to [N, 1, 1, ...] with the same
        number of dimensions as param.

    Examples:
        param [N]       -> vis [N]         (no reshape needed)
        param [N, 3]    -> vis [N, 1]      (one trailing dim)
        param [N, 4]    -> vis [N, 1]      (one trailing dim)
        param [N, K, 3] -> vis [N, 1, 1]   (two trailing dims)
    """
    if param.ndim <= 1:
        return visibility

    shape = [visibility.shape[0]] + [1] * (param.ndim - 1)
    return visibility.reshape(shape)


def adam(
    param: mx.array,
    param_grad: mx.array,
    exp_avg: mx.array,
    exp_avg_sq: mx.array,
    valid: mx.array,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Selective Adam update matching upstream gsplat API.

    This is the standalone function equivalent of the upstream
    gsplat.cuda._wrapper.adam() CUDA kernel call. Unlike the upstream
    which modifies arrays in-place, this returns new arrays since MLX
    arrays are immutable.

    Args:
        param: Parameter array [N, ...].
        param_grad: Gradient array [N, ...], same shape as param.
        exp_avg: First moment estimate [N, ...], same shape as param.
        exp_avg_sq: Second moment estimate [N, ...], same shape as param.
        valid: Boolean visibility mask [N].
        lr: Learning rate.
        b1: Beta1 coefficient for first moment.
        b2: Beta2 coefficient for second moment.
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of (updated_param, updated_exp_avg, updated_exp_avg_sq).
        All have the same shape as the inputs.
    """
    N = valid.shape[0]

    # Broadcast visibility mask to match parameter shape
    mask = _broadcast_visibility(valid, param)

    # Compute full Adam updates
    new_exp_avg = b1 * exp_avg + (1 - b1) * param_grad
    new_exp_avg_sq = b2 * exp_avg_sq + (1 - b2) * param_grad * param_grad

    # Apply selectively
    exp_avg_out = mx.where(mask, new_exp_avg, exp_avg)
    exp_avg_sq_out = mx.where(mask, new_exp_avg_sq, exp_avg_sq)

    # Compute parameter update (no bias correction, matching upstream)
    update = exp_avg_out / (mx.sqrt(exp_avg_sq_out) + eps)
    new_param = param - lr * update
    param_out = mx.where(mask, new_param, param)

    return param_out, exp_avg_out, exp_avg_sq_out
```

---

## 6. Upstream Correspondence

| Upstream (gsplat) | MLX (this PRD) | Notes |
|-------------------|----------------|-------|
| `SelectiveAdam.__init__(params, eps, betas)` | `SelectiveAdam.__init__(lr, betas, eps, bias_correction)` | MLX version takes lr at init, not via param groups |
| `SelectiveAdam.step(visibility)` | `SelectiveAdam.step(params, grads, visibility)` | MLX uses functional style (params in/out), not in-place mutation |
| `adam()` CUDA kernel | `adam()` standalone + `_update_single()` | Pure MLX `mx.where` replaces CUDA early-return |
| `torch.optim.Adam.state` | `SelectiveAdam._state` dict | Same structure: step, exp_avg, exp_avg_sq per param |
| Per-param-group lr | Single lr at init | 3DGS uses per-param lr via separate optimizer instances |
| `param.numel() // N` implicit D | `_broadcast_visibility()` explicit | MLX version is explicit about dimension handling |
| `resize_state` (not in upstream) | `resize_state(name, new_n, indices)` | Upstream handles state resize in the strategy class |

---

## 7. Data Flow

```
Training Iteration:
                                                 +-----------+
  Camera + Gaussians  ------>  Rasterizer  ----> | visibility|  [N] bool
                                    |            +-----------+
                                    v                  |
                              rendered_image           |
                                    |                  |
                                    v                  |
                               Loss fn                 |
                                    |                  |
                                    v                  |
                                 grads                 |
                                    |                  |
                                    v                  v
                            +---------------------------+
                            |     SelectiveAdam.step()   |
                            |                           |
                            |  for each param:          |
                            |    vis = broadcast(vis)   |
                            |    m = where(vis, ...)    |
                            |    v = where(vis, ...)    |
                            |    p = where(vis, ...)    |
                            +---------------------------+
                                    |
                                    v
                             updated_params
```

---

## 8. Test Plan

### 8.1 File: `tests/test_optimizer.py`

All tests use `mx.array` operations. No PyTorch dependency.

```python
"""Tests for SelectiveAdam optimizer.

Tests cover:
1. Basic Adam update (no visibility mask / all visible)
2. Selective update (partial visibility)
3. Selective state preservation (invisible state frozen)
4. Momentum accumulation over multiple steps
5. Bias correction mode
6. Multi-shape parameters ([N], [N,3], [N,4], [N,K,3])
7. State persistence across steps
8. Zero gradients with momentum decay
9. All-invisible mask produces no change
10. Params without gradients returned unchanged
11. Convergence on quadratic (full visibility)
12. Selective convergence (partial visibility)
13. State resize without indices
14. State resize with indices
15. Input validation (6 sub-tests)
16. Standalone adam() function
"""

import mlx.core as mx
import pytest

from gsplat_mlx.optimizers import SelectiveAdam
from gsplat_mlx.optimizers.selective_adam import adam


# ---------------------------------------------------------------------------
# Test Class 1: Basic Adam Behavior (all visible)
# ---------------------------------------------------------------------------
class TestSelectiveAdamBasic:
    """Test basic Adam behavior with all Gaussians visible."""

    def test_adam_basic(self):
        """Parameters should update when all Gaussians are visible.

        Verifies the exact numerical output of one Adam step with known inputs.
        """
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # All params should have changed
        diff = mx.abs(updated["means"] - params["means"])
        assert mx.all(diff > 0).item(), "All params should update when all visible"

        # After one step with constant gradient 0.5:
        # m = (1 - 0.9) * 0.5 = 0.05
        # v = (1 - 0.999) * 0.25 = 0.00025
        # update = 0.01 * 0.05 / (sqrt(0.00025) + 1e-8)
        #        = 0.01 * 0.05 / 0.015811...
        #        = 0.01 * 3.1623...
        #        ≈ 0.031623
        # new_param = 1.0 - 0.031623 ≈ 0.968377
        expected_m = (1 - 0.9) * 0.5  # = 0.05
        expected_v = (1 - 0.999) * 0.25  # = 0.00025
        expected_update = 0.01 * expected_m / (mx.sqrt(mx.array(expected_v)) + 1e-8)
        expected_param = 1.0 - expected_update.item()

        assert mx.allclose(
            updated["means"],
            mx.full((N, 3), expected_param, dtype=mx.float32),
            atol=1e-6,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 2: Selective Update with Partial Visibility
# ---------------------------------------------------------------------------
class TestSelectiveAdamVisibility:
    """Test selective update with partial visibility."""

    def test_adam_selective(self):
        """Only visible Gaussians should be updated."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}

        # First 5 visible, last 5 invisible
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # Visible Gaussians should have changed
        visible_diff = mx.abs(updated["means"][:5] - params["means"][:5])
        assert mx.all(visible_diff > 0).item(), "Visible params should update"

        # Invisible Gaussians should be unchanged
        invisible_diff = mx.abs(updated["means"][5:] - params["means"][5:])
        assert mx.all(invisible_diff == 0).item(), (
            "Invisible params should not update"
        )

    def test_adam_selective_state_preserved(self):
        """Optimizer state for invisible Gaussians should be frozen."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        _ = optimizer.step(params, grads, visibility)

        state = optimizer.state["means"]

        # Invisible Gaussians should have zero exp_avg and exp_avg_sq
        assert mx.all(state["exp_avg"][5:] == 0).item(), (
            "Invisible exp_avg should be zero"
        )
        assert mx.all(state["exp_avg_sq"][5:] == 0).item(), (
            "Invisible exp_avg_sq should be zero"
        )

        # Visible Gaussians should have non-zero state
        assert mx.all(state["exp_avg"][:5] != 0).item(), (
            "Visible exp_avg should be non-zero"
        )
        assert mx.all(state["exp_avg_sq"][:5] != 0).item(), (
            "Visible exp_avg_sq should be non-zero"
        )


# ---------------------------------------------------------------------------
# Test Class 3: Momentum Accumulation
# ---------------------------------------------------------------------------
class TestSelectiveAdamMomentum:
    """Test momentum accumulation over multiple steps."""

    def test_adam_momentum(self):
        """Verify exp_avg accumulates correctly over multiple steps
        with a constant gradient.
        """
        N = 4
        params = {"x": mx.ones((N,), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)
        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        g_val = 1.0

        # Step 1: m = (1-0.9)*1.0 = 0.1
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m1 = 0.1 * g_val  # = 0.1
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m1, dtype=mx.float32),
            atol=1e-6,
        ).item()

        # Step 2: m = 0.9*0.1 + 0.1*1.0 = 0.19
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m2 = 0.9 * m1 + 0.1 * g_val  # = 0.19
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m2, dtype=mx.float32),
            atol=1e-6,
        ).item()

        # Step 3: m = 0.9*0.19 + 0.1*1.0 = 0.271
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m3 = 0.9 * m2 + 0.1 * g_val  # = 0.271
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m3, dtype=mx.float32),
            atol=1e-6,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 4: Bias Correction
# ---------------------------------------------------------------------------
class TestSelectiveAdamBiasCorrection:
    """Test bias correction mode."""

    def test_adam_bias_correction(self):
        """With bias_correction=True, early steps should have larger updates
        due to the 1/(1-beta^t) correction factor.
        """
        N = 4
        params_no_bc = {"x": mx.ones((N,), dtype=mx.float32)}
        params_bc = {"x": mx.ones((N,), dtype=mx.float32)}
        grads = {"x": mx.full((N,), 0.5, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        opt_no_bc = SelectiveAdam(
            lr=0.01, betas=(0.9, 0.999), eps=1e-8, bias_correction=False
        )
        opt_bc = SelectiveAdam(
            lr=0.01, betas=(0.9, 0.999), eps=1e-8, bias_correction=True
        )

        updated_no_bc = opt_no_bc.step(params_no_bc, grads, visibility)
        updated_bc = opt_bc.step(params_bc, grads, visibility)

        # Bias correction should produce a larger update in early steps
        diff_no_bc = mx.abs(updated_no_bc["x"] - 1.0)
        diff_bc = mx.abs(updated_bc["x"] - 1.0)

        assert mx.all(diff_bc > diff_no_bc).item(), (
            "Bias-corrected updates should be larger in early steps"
        )

        # Verify exact bias-corrected values at step 1:
        # m = 0.05, v = 0.00025
        # m_hat = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
        # v_hat = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
        # update = 0.01 * 0.5 / (sqrt(0.25) + 1e-8) = 0.01 * 0.5 / 0.5 = 0.01
        # new_param = 1.0 - 0.01 = 0.99
        expected_bc = 1.0 - 0.01
        assert mx.allclose(
            updated_bc["x"],
            mx.full((N,), expected_bc, dtype=mx.float32),
            atol=1e-5,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 5: Multi-Shape Parameters
# ---------------------------------------------------------------------------
class TestSelectiveAdamMultiShape:
    """Test with parameters of different shapes."""

    def test_adam_multi_shape(self):
        """Optimizer should handle [N], [N,3], [N,4], [N,K,3] params."""
        N = 8
        K = 9  # SH degree 2: 9 coefficients

        params = {
            "opacities": mx.ones((N,), dtype=mx.float32),
            "means": mx.ones((N, 3), dtype=mx.float32),
            "quats": mx.ones((N, 4), dtype=mx.float32),
            "sh_coeffs": mx.ones((N, K, 3), dtype=mx.float32),
        }

        grads = {
            "opacities": mx.full((N,), 0.1, dtype=mx.float32),
            "means": mx.full((N, 3), 0.2, dtype=mx.float32),
            "quats": mx.full((N, 4), 0.3, dtype=mx.float32),
            "sh_coeffs": mx.full((N, K, 3), 0.05, dtype=mx.float32),
        }

        # Half visible
        visibility = mx.array(
            [True] * 4 + [False] * 4, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        for name in params:
            # Visible should change
            vis_diff = mx.max(mx.abs(
                updated[name][:4] - params[name][:4]
            )).item()
            assert vis_diff > 0, f"Visible {name} should update"

            # Invisible should not change
            invis_diff = mx.max(mx.abs(
                updated[name][4:] - params[name][4:]
            )).item()
            assert invis_diff == 0, f"Invisible {name} should not update"

        # Verify state shapes match param shapes
        for name in params:
            assert optimizer.state[name]["exp_avg"].shape == params[name].shape
            assert optimizer.state[name]["exp_avg_sq"].shape == params[name].shape


# ---------------------------------------------------------------------------
# Test Class 6: State Persistence
# ---------------------------------------------------------------------------
class TestSelectiveAdamStatePersistence:
    """Test that state persists correctly across steps."""

    def test_adam_state_persistence(self):
        """State should carry over between steps and accumulate correctly."""
        N = 4
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # Run 5 steps with constant gradient
        for i in range(5):
            grads = {"x": mx.full((N, 3), 1.0, dtype=mx.float32)}
            params = optimizer.step(params, grads, visibility)

        # Step counter should be 5
        assert optimizer.state["x"]["step"].item() == 5

        # exp_avg after 5 steps with constant grad=1.0:
        # m = 1.0 * (1 - 0.9^5) = 1.0 * 0.40951 = 0.40951
        # (geometric series: sum_{k=0}^{4} 0.1 * 0.9^k = 0.1 * (1-0.9^5)/(1-0.9))
        # = 0.1 * (1 - 0.59049) / 0.1 = 1.0 * 0.40951 = 0.40951
        expected_m = 1.0 * (1 - 0.9**5)
        actual_m = optimizer.state["x"]["exp_avg"][0, 0].item()
        assert abs(actual_m - expected_m) < 1e-5, (
            f"exp_avg after 5 steps: expected {expected_m}, got {actual_m}"
        )


# ---------------------------------------------------------------------------
# Test Class 7: Edge Cases
# ---------------------------------------------------------------------------
class TestSelectiveAdamEdgeCases:
    """Test edge cases."""

    def test_adam_zero_grad(self):
        """Zero gradients with all visible: momentum should decay
        (m_new = beta1 * m_old + 0 = beta1 * m_old).
        """
        N = 4
        params = {"x": mx.ones((N,), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)
        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Step with non-zero gradient to build up state
        grads = {"x": mx.full((N,), 1.0, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m_after_step1 = optimizer.state["x"]["exp_avg"][0].item()

        # Step with zero gradient -- momentum should decay by beta1
        grads = {"x": mx.zeros((N,), dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m_after_step2 = optimizer.state["x"]["exp_avg"][0].item()

        # m2 = 0.9 * m1 + 0.1 * 0 = 0.9 * m1
        assert abs(m_after_step2 - 0.9 * m_after_step1) < 1e-6

    def test_adam_all_invisible(self):
        """When no Gaussians are visible, nothing should change."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32) * 42.0}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 100.0}
        visibility = mx.zeros((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # No parameters should change
        assert mx.allclose(updated["means"], params["means"]).item(), (
            "No params should update when all invisible"
        )

        # State should be initialized but all zeros (invisible -> no update)
        assert mx.all(optimizer.state["means"]["exp_avg"] == 0).item()
        assert mx.all(optimizer.state["means"]["exp_avg_sq"] == 0).item()

    def test_adam_no_grad_for_param(self):
        """Params without gradients should be returned unchanged."""
        N = 4
        params = {
            "means": mx.ones((N, 3), dtype=mx.float32),
            "colors": mx.ones((N, 3), dtype=mx.float32) * 0.5,
        }
        grads = {
            "means": mx.full((N, 3), 0.1, dtype=mx.float32),
            # No gradient for "colors"
        }
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        updated = optimizer.step(params, grads, visibility)

        # "means" should update
        assert not mx.allclose(updated["means"], params["means"]).item()
        # "colors" should be unchanged
        assert mx.allclose(updated["colors"], params["colors"]).item()


# ---------------------------------------------------------------------------
# Test Class 8: Convergence
# ---------------------------------------------------------------------------
class TestSelectiveAdamConvergence:
    """Test convergence on simple optimization problems."""

    def test_adam_convergence(self):
        """Optimize f(x) = sum((x - target)^2) with full visibility.

        Verifies that SelectiveAdam can actually minimize a quadratic.
        """
        N = 10
        target = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        target = mx.broadcast_to(target, (N, 3))

        # Start far from target
        params = {"x": mx.zeros((N, 3), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(300):
            # Gradient of f(x) = (x - target)^2 is 2*(x - target)
            grads = {"x": 2.0 * (params["x"] - target)}
            params = optimizer.step(params, grads, visibility)

        # Should converge close to target
        error = mx.max(mx.abs(params["x"] - target)).item()
        assert error < 0.05, f"Should converge to target, max error: {error}"

    def test_adam_selective_convergence(self):
        """Only visible Gaussians should converge; invisible should stay put."""
        N = 10
        target = mx.full((N, 3), 5.0, dtype=mx.float32)

        params = {"x": mx.zeros((N, 3), dtype=mx.float32)}
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(300):
            grads = {"x": 2.0 * (params["x"] - target)}
            params = optimizer.step(params, grads, visibility)

        # Visible Gaussians should converge
        visible_error = mx.max(mx.abs(params["x"][:5] - target[:5])).item()
        assert visible_error < 0.05, (
            f"Visible Gaussians should converge, max error: {visible_error}"
        )

        # Invisible Gaussians should remain at initial value (0.0)
        invisible_diff = mx.max(mx.abs(params["x"][5:])).item()
        assert invisible_diff == 0.0, (
            f"Invisible Gaussians should not move, max diff: {invisible_diff}"
        )


# ---------------------------------------------------------------------------
# Test Class 9: State Resize for Densification
# ---------------------------------------------------------------------------
class TestSelectiveAdamResize:
    """Test state resizing for densification."""

    def test_resize_state_no_indices(self):
        """Resizing without indices should zero-initialize new state."""
        N = 10
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"x": mx.full((N, 3), 0.5, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        _ = optimizer.step(params, grads, visibility)

        # State should exist
        assert "x" in optimizer.state
        old_step = optimizer.state["x"]["step"].item()

        # Resize to larger N
        optimizer.resize_state("x", new_n=20)

        # State should be resized with zeros
        assert optimizer.state["x"]["exp_avg"].shape == (20, 3)
        assert optimizer.state["x"]["exp_avg_sq"].shape == (20, 3)
        assert mx.all(optimizer.state["x"]["exp_avg"] == 0).item()
        assert mx.all(optimizer.state["x"]["exp_avg_sq"] == 0).item()
        # Step should be preserved
        assert optimizer.state["x"]["step"].item() == old_step

    def test_resize_state_with_indices(self):
        """Resizing with indices should preserve selected state."""
        N = 4
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"x": mx.full((N, 3), 1.0, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        _ = optimizer.step(params, grads, visibility)

        old_m = optimizer.state["x"]["exp_avg"]
        mx.eval(old_m)

        # Resize: keep Gaussians 0,1,2,3 and add 2 new ones (index -1)
        indices = mx.array([0, 1, 2, 3, -1, -1])
        optimizer.resize_state("x", new_n=6, indices=indices)

        new_m = optimizer.state["x"]["exp_avg"]
        assert new_m.shape == (6, 3)

        # First 4 should match old state
        assert mx.allclose(new_m[:4], old_m, atol=1e-7).item()
        # Last 2 should be zero
        assert mx.all(new_m[4:] == 0).item()

    def test_resize_nonexistent_param(self):
        """Resizing state for a parameter that has no state should be a no-op."""
        optimizer = SelectiveAdam(lr=0.01)
        # Should not raise
        optimizer.resize_state("nonexistent", new_n=10)
        assert "nonexistent" not in optimizer.state


# ---------------------------------------------------------------------------
# Test Class 10: Input Validation
# ---------------------------------------------------------------------------
class TestSelectiveAdamValidation:
    """Test input validation."""

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="learning rate"):
            SelectiveAdam(lr=-0.01)

    def test_invalid_beta1(self):
        with pytest.raises(ValueError, match="beta1"):
            SelectiveAdam(betas=(1.0, 0.999))

    def test_invalid_beta2(self):
        with pytest.raises(ValueError, match="beta2"):
            SelectiveAdam(betas=(0.9, 1.0))

    def test_invalid_eps(self):
        with pytest.raises(ValueError, match="epsilon"):
            SelectiveAdam(eps=-1e-8)

    def test_shape_mismatch(self):
        optimizer = SelectiveAdam()
        params = {"x": mx.ones((4, 3))}
        grads = {"x": mx.ones((4, 2))}  # wrong shape
        visibility = mx.ones((4,), dtype=mx.bool_)

        with pytest.raises(ValueError, match="Shape mismatch"):
            optimizer.step(params, grads, visibility)

    def test_unknown_grad_key(self):
        optimizer = SelectiveAdam()
        params = {"x": mx.ones((4, 3))}
        grads = {"y": mx.ones((4, 3))}  # key not in params
        visibility = mx.ones((4,), dtype=mx.bool_)

        with pytest.raises(ValueError, match="not found in params"):
            optimizer.step(params, grads, visibility)


# ---------------------------------------------------------------------------
# Test Class 11: Standalone adam() Function
# ---------------------------------------------------------------------------
class TestStandaloneAdam:
    """Test the standalone adam() function."""

    def test_standalone_adam_matches_class(self):
        """Standalone adam() should produce identical results to SelectiveAdam.step()."""
        N = 8
        param = mx.ones((N, 3), dtype=mx.float32)
        grad = mx.full((N, 3), 0.5, dtype=mx.float32)
        exp_avg = mx.zeros((N, 3), dtype=mx.float32)
        exp_avg_sq = mx.zeros((N, 3), dtype=mx.float32)
        valid = mx.array([True] * 4 + [False] * 4, dtype=mx.bool_)

        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8

        # Standalone function
        p_out, m_out, v_out = adam(param, grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps)
        mx.eval(p_out, m_out, v_out)

        # Class-based
        optimizer = SelectiveAdam(lr=lr, betas=(b1, b2), eps=eps)
        updated = optimizer.step({"x": param}, {"x": grad}, valid)

        # Should match
        assert mx.allclose(p_out, updated["x"], atol=1e-7).item(), (
            "Standalone adam() should match SelectiveAdam.step()"
        )
        assert mx.allclose(m_out, optimizer.state["x"]["exp_avg"], atol=1e-7).item()
        assert mx.allclose(v_out, optimizer.state["x"]["exp_avg_sq"], atol=1e-7).item()

    def test_standalone_adam_selective(self):
        """Standalone adam() should respect visibility mask."""
        N = 6
        param = mx.ones((N,), dtype=mx.float32) * 10.0
        grad = mx.ones((N,), dtype=mx.float32)
        exp_avg = mx.zeros((N,), dtype=mx.float32)
        exp_avg_sq = mx.zeros((N,), dtype=mx.float32)
        valid = mx.array([True, True, True, False, False, False], dtype=mx.bool_)

        p_out, m_out, v_out = adam(param, grad, exp_avg, exp_avg_sq, valid, 0.01, 0.9, 0.999, 1e-8)
        mx.eval(p_out, m_out, v_out)

        # Visible should change
        assert mx.all(mx.abs(p_out[:3] - 10.0) > 0).item()
        # Invisible should not change
        assert mx.allclose(p_out[3:], mx.full((3,), 10.0)).item()
        # Invisible state should be zero
        assert mx.all(m_out[3:] == 0).item()
        assert mx.all(v_out[3:] == 0).item()
```

### 8.2 Test Summary Table

| Test | Class | What it verifies |
|------|-------|-----------------|
| `test_adam_basic` | `TestSelectiveAdamBasic` | All-visible update with exact numerical check |
| `test_adam_selective` | `TestSelectiveAdamVisibility` | Partial visibility: visible update, invisible unchanged |
| `test_adam_selective_state_preserved` | `TestSelectiveAdamVisibility` | Invisible Gaussians have zero optimizer state |
| `test_adam_momentum` | `TestSelectiveAdamMomentum` | exp_avg accumulation over 3 steps with exact values |
| `test_adam_bias_correction` | `TestSelectiveAdamBiasCorrection` | Bias-corrected updates are larger in early steps |
| `test_adam_multi_shape` | `TestSelectiveAdamMultiShape` | [N], [N,3], [N,4], [N,K,3] all work correctly |
| `test_adam_state_persistence` | `TestSelectiveAdamStatePersistence` | State accumulates across 5 steps, step counter correct |
| `test_adam_zero_grad` | `TestSelectiveAdamEdgeCases` | Zero gradient causes momentum decay by beta1 |
| `test_adam_all_invisible` | `TestSelectiveAdamEdgeCases` | All-invisible: params and state unchanged |
| `test_adam_no_grad_for_param` | `TestSelectiveAdamEdgeCases` | Params without gradients returned unchanged |
| `test_adam_convergence` | `TestSelectiveAdamConvergence` | Quadratic minimization converges in 300 steps |
| `test_adam_selective_convergence` | `TestSelectiveAdamConvergence` | Visible converge, invisible stay at initial |
| `test_resize_state_no_indices` | `TestSelectiveAdamResize` | Zero-init resize preserves step counter |
| `test_resize_state_with_indices` | `TestSelectiveAdamResize` | Index-based resize preserves selected state |
| `test_resize_nonexistent_param` | `TestSelectiveAdamResize` | Resize no-op for unknown params |
| `test_invalid_lr` | `TestSelectiveAdamValidation` | Rejects negative learning rate |
| `test_invalid_beta1` | `TestSelectiveAdamValidation` | Rejects beta1 >= 1.0 |
| `test_invalid_beta2` | `TestSelectiveAdamValidation` | Rejects beta2 >= 1.0 |
| `test_invalid_eps` | `TestSelectiveAdamValidation` | Rejects negative epsilon |
| `test_shape_mismatch` | `TestSelectiveAdamValidation` | Rejects param/grad shape mismatch |
| `test_unknown_grad_key` | `TestSelectiveAdamValidation` | Rejects gradient keys not in params |
| `test_standalone_adam_matches_class` | `TestStandaloneAdam` | `adam()` matches `SelectiveAdam.step()` |
| `test_standalone_adam_selective` | `TestStandaloneAdam` | `adam()` respects visibility mask |

---

## 9. Integration Notes

### 9.1 With Training Loop (PRD-13)

The training loop will use SelectiveAdam as follows:

```python
# In the training loop:
optimizer = SelectiveAdam(lr=1e-3, betas=(0.9, 0.999), eps=1e-15)

for iteration in range(max_iterations):
    # Forward pass produces visibility mask
    rendered, visibility = rasterize(params, camera)

    # Compute loss
    loss = compute_loss(rendered, ground_truth)

    # Backward pass produces gradients
    grads = compute_gradients(loss, params)

    # Selective update -- only visible Gaussians are touched
    params = optimizer.step(params, grads, visibility)
```

### 9.2 Per-Parameter Learning Rates

The upstream gsplat uses PyTorch's param_groups to set different learning rates for different parameters (e.g., higher lr for means, lower for SH coefficients). With our standalone optimizer, this is achieved by using multiple SelectiveAdam instances:

```python
optimizer_means = SelectiveAdam(lr=1.6e-4, betas=(0.9, 0.999), eps=1e-15)
optimizer_scales = SelectiveAdam(lr=5e-3, betas=(0.9, 0.999), eps=1e-15)
optimizer_quats = SelectiveAdam(lr=1e-3, betas=(0.9, 0.999), eps=1e-15)
optimizer_opacities = SelectiveAdam(lr=5e-2, betas=(0.9, 0.999), eps=1e-15)
optimizer_sh = SelectiveAdam(lr=2.5e-3, betas=(0.9, 0.999), eps=1e-15)

# Each optimizer handles one parameter
params["means"] = optimizer_means.step(
    {"means": params["means"]},
    {"means": grads["means"]},
    visibility,
)["means"]
```

Alternatively, a future enhancement could add per-parameter lr support to a single optimizer instance.

### 9.3 Densification (Clone/Split/Prune) with PRD-10 Strategy

When Gaussians are cloned, split, or pruned during training, N changes. The optimizer state must be resized to match. The `resize_state()` method handles this:

```python
# After pruning: keep only Gaussians at 'keep_indices'
for name in params:
    optimizer.resize_state(name, new_n=len(keep_indices), indices=keep_indices)

# After splitting: original Gaussians + new ones
# indices maps new positions to old positions, -1 for new Gaussians
split_indices = mx.concatenate([mx.arange(N), mx.array([-1] * n_new)])
for name in params:
    optimizer.resize_state(name, new_n=N + n_new, indices=split_indices)
```

---

## 10. Performance Considerations

### 10.1 Memory

Each parameter with shape `[N, D]` requires `2 * N * D` additional floats for `exp_avg` and `exp_avg_sq`. For a typical 3DGS scene with 1M Gaussians:

| Parameter | Shape | State Memory |
|-----------|-------|-------------|
| means | [1M, 3] | 24 MB |
| quats | [1M, 4] | 32 MB |
| scales | [1M, 3] | 24 MB |
| opacities | [1M] | 8 MB |
| sh0 | [1M, 1, 3] | 24 MB |
| shN (deg 3) | [1M, 15, 3] | 360 MB |
| **Total** | | **~472 MB** |

This is the same as standard Adam. SelectiveAdam does not reduce peak memory (all state is allocated for all N), but it reduces memory bandwidth by skipping reads/writes for invisible Gaussians via `mx.where`.

### 10.2 Compute

The `mx.where` approach evaluates both branches (Adam update and identity) and selects per-element. This is less efficient than the CUDA kernel's early-return pattern, which truly skips work for invisible threads. However:

- MLX's lazy evaluation and graph compilation may fuse the `mx.where` operations
- The dominant cost in 3DGS training is rasterization, not the optimizer step
- The simplicity of the pure-MLX approach outweighs the minor overhead for MVP

A future Metal kernel optimization could replace the `mx.where` pattern with an indexed-scatter update that only touches visible elements.

### 10.3 `mx.eval` Placement

The `step()` method calls `mx.eval()` to force computation of the updated parameters and state. This is necessary because:

1. MLX uses lazy evaluation -- without `mx.eval`, the computation graph would grow unboundedly across training iterations
2. The optimizer state must be materialized before the next step reads it
3. This matches the natural synchronization point at the end of each training iteration

---

## 11. Acceptance Criteria

- [ ] All 23 tests in `tests/test_optimizer.py` pass
- [ ] `from gsplat_mlx.optimizers import SelectiveAdam` works
- [ ] `from gsplat_mlx.optimizers import adam` works
- [ ] Convergence test reaches target within tolerance (max error < 0.05)
- [ ] Selective convergence test shows visible Gaussians converge, invisible stay at initial value
- [ ] Parameters of shapes `[N]`, `[N, D]`, `[N, K, D]` all work correctly
- [ ] State resizing preserves existing state at specified indices
- [ ] Input validation rejects invalid hyperparameters
- [ ] Standalone `adam()` function produces identical output to `SelectiveAdam.step()`
- [ ] No dependency on PyTorch at runtime

---

## 12. Future Enhancements

- **Per-parameter learning rates**: Accept a `lr` dict mapping param names to learning rates in a single optimizer instance
- **Metal kernel**: Fused selective Adam kernel for Metal, skipping invisible elements entirely instead of computing both branches with `mx.where`
- **Amsgrad**: Optional amsgrad variant (max of all past v_t) for improved convergence guarantees
- **Learning rate scheduling**: Accept a callable `lr(step)` for warm-up and decay schedules
- **Gradient accumulation**: Support accumulating gradients across micro-batches before stepping
- **Cross-framework validation**: `@pytest.mark.requires_torch` tests comparing against upstream PyTorch SelectiveAdam
