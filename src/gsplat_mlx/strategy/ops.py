"""Low-level densification operations for Gaussian splatting.

Provides the core parameter manipulation primitives used by densification
strategies: ``duplicate``, ``remove``, ``split``, and ``reset_opa``.

Each operation modifies parameters, optimizer state, and running strategy
state in-place (by replacing values in the mutable dicts).

Upstream reference: ``repositories/gsplat-upstream/gsplat/strategy/ops.py``

Port notes (PyTorch -> MLX):
- No ``torch.nn.Parameter`` wrapping; arrays replaced directly in dicts.
- Optimizer state keyed by param name, not Parameter identity.
- No boolean indexing for selection -- use ``mx.where`` to get indices.
- MLX arrays are immutable; all "in-place" ops create new arrays.
- ``mx.eval()`` used sparingly (only for ``.item()`` calls in strategy ops,
  which run infrequently).
"""

from typing import Callable, Dict, List, Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logit(x: mx.array) -> mx.array:
    """Inverse sigmoid: logit(x) = log(x / (1 - x)).

    Numerically stable: clamps input to avoid ``log(0)``.

    Args:
        x: Input array with values in (0, 1).

    Returns:
        Logit values with same shape as input.
    """
    eps = 1e-7
    x = mx.clip(x, eps, 1.0 - eps)
    return mx.log(x / (1.0 - x))


def _normalized_quat_to_rotmat(quat: mx.array) -> mx.array:
    """Convert normalized quaternion (wxyz) to rotation matrix.

    Args:
        quat: ``[..., 4]`` quaternion in ``(w, x, y, z)`` convention.
            Must already be normalized.

    Returns:
        Rotation matrix ``[..., 3, 3]``.
    """
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    mat = mx.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        axis=-1,
    )

    return mat.reshape(quat.shape[:-1] + (3, 3))


def _mask_to_indices(mask: mx.array) -> mx.array:
    """Convert a boolean mask to an array of indices where mask is True.

    MLX does not support single-argument ``mx.where(mask)`` (NumPy-style).
    This helper provides the equivalent: returns a 1-D int32 array of indices.

    Args:
        mask: Boolean array of shape ``[N]``.

    Returns:
        Int32 array of indices where ``mask`` is ``True``, shape ``[K]``
        where ``K = mask.sum()``.
    """
    N = mask.shape[0]
    all_indices = mx.arange(N, dtype=mx.int32)
    # Use argsort on the mask (False=0 sorts before True=1),
    # then take the last K entries where K = sum(mask)
    mx.eval(mask)
    n_true = int(mx.sum(mask.astype(mx.int32)).item())
    if n_true == 0:
        return mx.zeros((0,), dtype=mx.int32)
    if n_true == N:
        return all_indices
    # Select indices where mask is True
    # Multiply: indices * mask gives 0 for False entries (but 0 is also a valid index)
    # Better: use the complement-sort trick
    # Simplest correct approach: gather using argsort of ~mask (puts True first)
    order = mx.argsort((~mask).astype(mx.int32))
    return order[:n_true]


def _scatter_add(
    target: mx.array,
    indices: mx.array,
    values: mx.array,
) -> mx.array:
    """Scatter-add: ``target[indices[i]] += values[i]`` for all ``i``.

    Handles duplicate indices correctly (accumulates, not overwrites).

    Args:
        target: ``[N]`` target array to accumulate into.
        indices: ``[M]`` int32 indices into target.
        values: ``[M]`` values to add at those indices.

    Returns:
        New array with accumulated values.
    """
    return target.at[indices].add(values)


def _scatter_max(
    target: mx.array,
    indices: mx.array,
    values: mx.array,
) -> mx.array:
    """Scatter-max: ``target[indices[i]] = max(target[indices[i]], values[i])``.

    Handles duplicate indices correctly (takes max, not overwrites).

    Args:
        target: ``[N]`` target array.
        indices: ``[M]`` int32 indices into target.
        values: ``[M]`` values to compare.

    Returns:
        New array with element-wise maxima.
    """
    return target.at[indices].maximum(values)


# ---------------------------------------------------------------------------
# Core helper: update params + optimizer state together
# ---------------------------------------------------------------------------


def _update_param_with_optimizer(
    param_fn: Callable[[str, mx.array], mx.array],
    optimizer_fn: Callable[[str, mx.array], mx.array],
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    names: Optional[List[str]] = None,
) -> None:
    """Update parameters and their optimizer state using provided transform functions.

    This is the central mechanism for structural changes to the Gaussian set.
    When Gaussians are added, removed, or split, BOTH the parameter arrays AND
    their associated optimizer state (``exp_avg``, ``exp_avg_sq`` for Adam)
    must be resized/reorganized consistently.

    Args:
        param_fn: ``(param_name, param_array) -> new_param_array``.
            Defines how each parameter is transformed (e.g., concatenate for
            clone, index-select for removal).
        optimizer_fn: ``(state_key, state_array) -> new_state_array``.
            Defines how each optimizer state entry is transformed (typically:
            append zeros for new Gaussians, index-select for removals).
        params: Mutable dict of ``parameter_name -> mx.array``.
        optimizers: Mutable dict of ``param_name -> {state_key: mx.array}``.
            The ``"step"`` key (int/scalar) is preserved as-is.
        names: If provided, only update these parameter names. Otherwise
            update all parameters.
    """
    if names is None:
        names = list(params.keys())

    for name in names:
        p = params[name]
        new_p = param_fn(name, p)
        params[name] = new_p

        if name not in optimizers:
            # Non-trainable parameter -- skip optimizer update
            continue

        opt_state = optimizers[name]
        for key in list(opt_state.keys()):
            if key == "step":
                continue  # step count is a scalar, not resized
            v = opt_state[key]
            if isinstance(v, mx.array):
                opt_state[key] = optimizer_fn(key, v)


# ---------------------------------------------------------------------------
# duplicate
# ---------------------------------------------------------------------------


def duplicate(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
) -> None:
    """In-place duplicate Gaussians where ``mask=True``.

    After this operation:

    - ``params[k].shape[0]`` increases by ``mask.sum()``.
    - The duplicated Gaussians have the SAME parameter values as the originals.
    - The duplicated Gaussians have ZERO optimizer state (fresh start for Adam).
    - Running state (grad accumulators, counts) is also duplicated.

    Args:
        params: Parameter dict. Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state (``grad2d``, ``count``, etc.).
            Modified in-place.
        mask: Boolean array ``[N]``. ``True`` = duplicate this Gaussian.
    """
    sel = _mask_to_indices(mask)
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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def remove(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
) -> None:
    """In-place remove Gaussians where ``mask=True``.

    After this operation:

    - ``params[k].shape[0]`` decreases by ``mask.sum()``.
    - Optimizer state entries are correspondingly removed.
    - Running state entries are correspondingly removed.

    Args:
        params: Parameter dict. Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Modified in-place.
        mask: Boolean array ``[N]``. ``True`` = REMOVE this Gaussian.
    """
    sel = _mask_to_indices(~mask)

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


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def split(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    mask: mx.array,
    revised_opacity: bool = False,
) -> None:
    """In-place split Gaussians where ``mask=True`` into 2 smaller Gaussians each.

    For each split Gaussian:

    - Two new Gaussians are created.
    - Positions are offset by samples from the original's covariance ellipsoid.
    - Scales are reduced: ``new_scale = old_scale / 1.6``.
    - If ``revised_opacity=True``, opacity follows:
      ``new_opa = 1 - sqrt(1 - sigmoid(old_opa))``.
    - All other parameters (colors, SH, quats, etc.) are copied.
    - Optimizer state is zeroed for all new Gaussians.

    The original split Gaussians are REMOVED (replaced by 2 new ones).
    Net change: ``+mask.sum()`` Gaussians (remove K, add 2K).

    Args:
        params: Parameter dict. Must contain ``"means"``, ``"scales"``,
            ``"quats"``. Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Modified in-place.
        mask: Boolean array ``[N]``. ``True`` = split this Gaussian.
        revised_opacity: Use revised opacity from arXiv:2404.06109.
            Default ``False``.
    """
    sel = _mask_to_indices(mask)
    rest = _mask_to_indices(~mask)
    n_sel = sel.shape[0]

    if n_sel == 0:
        return

    # Compute displacement samples from the Gaussian covariance
    scales = mx.exp(params["scales"][sel])  # [n_sel, 3]

    # Normalize quaternions and convert to rotation matrices
    quats = params["quats"][sel]  # [n_sel, 4]
    quat_norms = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True) + 1e-12)
    quats = quats / quat_norms

    # Build rotation matrices: [n_sel, 3, 3]
    rotmats = _normalized_quat_to_rotmat(quats)

    # Sample 2 random offsets per Gaussian: [2, n_sel, 3]
    noise = mx.random.normal(shape=(2, n_sel, 3))

    # Scale the noise by the Gaussian's scale, then rotate:
    # samples = R @ diag(s) @ noise
    scaled_noise = noise * mx.expand_dims(scales, axis=0)  # [2, n_sel, 3]

    # Rotate: use einsum for batched matrix-vector multiply
    # rotmats: [n_sel, 3, 3], scaled_noise: [2, n_sel, 3]
    # -> samples: [2, n_sel, 3]
    samples = mx.einsum("nij,bnj->bni", rotmats, scaled_noise)

    n_before = mask.shape[0]

    # Build new parameter arrays
    def param_fn(name: str, p: mx.array) -> mx.array:
        if name == "means":
            # Offset positions by covariance samples
            p_sel = p[sel]  # [n_sel, 3]
            p_split = (mx.expand_dims(p_sel, axis=0) + samples).reshape(
                -1, 3
            )  # [2*n_sel, 3]
        elif name == "scales":
            # Reduce scale by factor 1.6 (in log space: subtract log(1.6))
            new_log_scale = mx.log(scales / 1.6)  # [n_sel, 3]
            p_split = mx.concatenate(
                [new_log_scale, new_log_scale], axis=0
            )  # [2*n_sel, 3]
        elif name == "opacities" and revised_opacity:
            # Revised opacity: new_opa = 1 - sqrt(1 - sigmoid(old_opa))
            old_opa = mx.sigmoid(p[sel])
            new_opa = 1.0 - mx.sqrt(1.0 - old_opa)
            new_logit = _logit(new_opa)
            p_split = mx.concatenate([new_logit, new_logit], axis=0)
        else:
            # All other params: just duplicate
            p_sel = p[sel]
            p_split = mx.concatenate([p_sel, p_sel], axis=0)

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
            v_split = mx.concatenate([v_sel, v_sel], axis=0)
            state[k] = mx.concatenate([v[rest], v_split], axis=0)


# ---------------------------------------------------------------------------
# reset_opa
# ---------------------------------------------------------------------------


def reset_opa(
    params: Dict[str, mx.array],
    optimizers: Dict[str, Dict[str, mx.array]],
    state: Dict[str, mx.array],
    value: float,
) -> None:
    """In-place reset all opacities to at most the given post-sigmoid value.

    Opacities are stored as logits (pre-sigmoid). This function clamps them
    so that ``sigmoid(logit) <= value``. The optimizer state for opacities
    is zeroed.

    Args:
        params: Parameter dict. Must contain ``"opacities"``.
            Modified in-place.
        optimizers: Optimizer state dict. Modified in-place.
        state: Running strategy state. Not modified by this op.
        value: Maximum post-sigmoid opacity value to clamp to.
    """
    logit_cap = float(_logit(mx.array(value)).item())

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
