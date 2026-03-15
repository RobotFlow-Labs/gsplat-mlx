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
