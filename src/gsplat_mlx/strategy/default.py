"""Default densification strategy from the original 3DGS paper.

Implements the clone/split/prune/reset lifecycle from:
    *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
    (Kerbl et al., 2023, arXiv:2308.04079)

Optionally supports AbsGS (arXiv:2404.10484) for absolute gradients
and revised opacity heuristic (arXiv:2404.06109).

Upstream reference: ``repositories/gsplat-upstream/gsplat/strategy/default.py``

Port notes (PyTorch -> MLX):
- Gradients provided via ``info["{key}_grad"]`` dict, not ``retain_grad()``.
- No ``@torch.no_grad()`` needed -- MLX uses functional gradients.
- No ``torch.cuda.empty_cache()`` -- MLX uses unified memory.
- Boolean masks used via ``mx.where`` for index extraction.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mlx.core as mx

from .base import Strategy
from .ops import (
    _mask_to_indices,
    _scatter_add,
    _scatter_max,
    duplicate,
    remove,
    reset_opa,
    split,
)


@dataclass
class DefaultStrategy(Strategy):
    """Default densification strategy from the original 3DGS paper.

    Lifecycle during training:

    1. ``step_pre_backward``: Validate that ``info`` dict contains the
       gradient key. In MLX (functional gradients), this is a validation step.
    2. ``step_post_backward``: Accumulate gradient stats. Periodically
       grow/prune/reset the Gaussian set.

    The strategy operates in phases per refinement step:

    **Accumulate** (every step within refine window):
        Track per-Gaussian 2D gradient norms and visibility counts.

    **Grow** (every ``refine_every`` steps):
        Clone under-reconstructed Gaussians (high gradient, small scale).
        Split over-reconstructed Gaussians (high gradient, large scale).

    **Prune** (every ``refine_every`` steps, after grow):
        Remove low-opacity and oversized Gaussians.

    **Reset** (every ``reset_every`` steps):
        Clamp all opacities down to prevent runaway opacity.

    If ``absgrad=True``, uses absolute gradients instead of average
    gradients following AbsGS (arXiv:2404.10484). Typically requires
    ``grow_grad2d ~ 0.0008`` and ``absgrad=True`` in rasterization.

    Args:
        prune_opa: Gaussians with ``sigmoid(opacity) < prune_opa`` are pruned.
        grow_grad2d: 2D gradient threshold for clone/split decisions.
        grow_scale3d: 3D scale threshold (normalized by ``scene_scale``)
            to distinguish clone vs. split.
        grow_scale2d: 2D radius threshold for additional split criterion.
        prune_scale3d: 3D scale threshold for pruning oversized Gaussians.
        prune_scale2d: 2D radius threshold for pruning.
        refine_scale2d_stop_iter: Stop using 2D scale criteria after this
            iteration. Default 0 = disabled.
        refine_start_iter: First iteration where grow/prune can fire.
        refine_stop_iter: Last iteration for grow/prune/accumulate.
        reset_every: Reset opacities every this many steps.
        refine_every: Grow/prune fires every this many steps.
        pause_refine_after_reset: Steps to pause refinement after opacity
            reset. Default 0 = no pause.
        absgrad: Use absolute gradients (AbsGS).
        revised_opacity: Use revised opacity heuristic during splits.
        verbose: Print grow/prune/reset statistics.
        key_for_gradient: Key in ``info`` dict for the 2D gradient tensor.
    """

    # --- Pruning thresholds ---
    prune_opa: float = 0.005
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15

    # --- Growing thresholds ---
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05

    # --- Iteration schedule ---
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    pause_refine_after_reset: int = 0

    # --- Algorithm variants ---
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: str = "means2d"

    # ---- Initialization ----

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize running state. Tensor allocation deferred to first step.

        Args:
            scene_scale: Scale factor for the scene. Used to normalize 3D
                scale thresholds (``grow_scale3d``, ``prune_scale3d`` are
                multiplied by this).

        Returns:
            State dict with:

            - ``grad2d``: ``None`` (becomes ``[N]`` float32 on first step)
            - ``count``: ``None`` (becomes ``[N]`` float32 on first step)
            - ``scene_scale``: ``float``
            - ``radii``: ``None`` (only if ``refine_scale2d_stop_iter > 0``)
        """
        state: Dict[str, Any] = {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
        }
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
    ) -> None:
        """Check params contain required keys and call base sanity check.

        Required keys: ``"means"``, ``"scales"``, ``"quats"``, ``"opacities"``.

        Args:
            params: Parameter dict.
            optimizers: Optimizer state dict.

        Raises:
            AssertionError: If required keys are missing or optimizer/param
                keys are inconsistent.
        """
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
    ) -> None:
        """Pre-backward hook: validate that info contains gradient key.

        In PyTorch upstream, this calls ``retain_grad()`` on ``means2d``.
        In MLX, gradients are computed functionally, so this is validation only.

        Args:
            params: Current parameter dict.
            optimizers: Current optimizer state dict.
            state: Running strategy state.
            step: Current training iteration.
            info: Dict from rasterization forward pass.

        Raises:
            AssertionError: If ``key_for_gradient`` not found in ``info``.
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
    ) -> None:
        """Post-backward hook: accumulate stats, periodically grow/prune/reset.

        Called every training step. Refinement (grow/prune/reset) only fires
        on specific iterations based on the schedule parameters.

        Args:
            params: Current parameter dict. May be modified in-place.
            optimizers: Current optimizer state dict. Modified in-place.
            state: Running strategy state. Modified in-place.
            step: Current training iteration.
            info: Dict with rasterization outputs and gradient info.
                Must contain ``"{key_for_gradient}_grad"``, ``"radii"``,
                ``"gaussian_ids"``, ``"width"``, ``"height"``, ``"n_cameras"``.
            packed: If ``True``, info contains packed format (sparse)
                with grads shape ``[nnz, 2]``. If ``False``, dense
                ``[C, N, 2]`` arrays.
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
    ) -> None:
        """Accumulate per-Gaussian gradient statistics from the current step.

        Tracks:

        - ``grad2d[i]``: sum of 2D gradient norms for Gaussian i.
        - ``count[i]``: number of times Gaussian i was visible.
        - ``radii[i]``: max 2D radius of Gaussian i (optional).

        These are accumulated across steps within a ``refine_every`` window
        and averaged when grow/prune decisions are made.

        Args:
            params: Current parameter dict.
            state: Running strategy state. Modified in-place.
            info: Dict with rasterization outputs and gradient info.
            packed: Whether info uses packed (sparse) format.
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
        first_param = params[list(params.keys())[0]]
        n_gaussian = first_param.shape[0]

        if state["grad2d"] is None:
            state["grad2d"] = mx.zeros((n_gaussian,))
        if state["count"] is None:
            state["count"] = mx.zeros((n_gaussian,))
        if self.refine_scale2d_stop_iter > 0 and state.get("radii") is None:
            state["radii"] = mx.zeros((n_gaussian,))

        # Accumulate based on format
        if packed:
            # Packed format: grads is [nnz, 2], gaussian_ids is [nnz]
            gs_ids = info["gaussian_ids"]
            grad_norms = mx.sqrt(mx.sum(grads_scaled**2, axis=-1))
            radii = info["radii"]
            if radii.ndim > 1:
                radii = mx.max(radii, axis=-1)

            state["grad2d"] = _scatter_add(state["grad2d"], gs_ids, grad_norms)
            state["count"] = _scatter_add(
                state["count"], gs_ids, mx.ones_like(grad_norms)
            )

            if self.refine_scale2d_stop_iter > 0:
                norm_radii = radii / float(max(info["width"], info["height"]))
                state["radii"] = _scatter_max(state["radii"], gs_ids, norm_radii)
        else:
            # Dense format: grads is [C, N, 2], radii is [C, N, 2]
            # Select visible Gaussians per camera: those with all radii > 0
            C = info["radii"].shape[0]
            for c in range(C):
                visible_c = mx.all(info["radii"][c] > 0.0, axis=-1)  # [N] bool
                gs_ids = _mask_to_indices(visible_c)
                if gs_ids.shape[0] == 0:
                    continue
                grads_c = grads_scaled[c][gs_ids]  # [nnz, 2]
                grad_norms = mx.sqrt(mx.sum(grads_c ** 2, axis=-1))

                state["grad2d"] = _scatter_add(state["grad2d"], gs_ids, grad_norms)
                state["count"] = _scatter_add(
                    state["count"], gs_ids, mx.ones_like(grad_norms)
                )

                if self.refine_scale2d_stop_iter > 0:
                    radii_c = mx.max(info["radii"][c][gs_ids], axis=-1)
                    norm_radii = radii_c / float(
                        max(info["width"], info["height"])
                    )
                    state["radii"] = _scatter_max(
                        state["radii"], gs_ids, norm_radii
                    )

    def _grow_gs(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """Grow Gaussians by cloning small high-gradient ones and splitting large ones.

        Algorithm:

        1. Compute average gradient: ``grad2d / max(count, 1)``
        2. ``is_grad_high = avg_grad > grow_grad2d``
        3. ``is_small = max(exp(scales)) <= grow_scale3d * scene_scale``
        4. Clone mask = ``is_grad_high AND is_small``
        5. Split mask = ``is_grad_high AND NOT is_small``
        6. Optionally: split ``|= (radii > grow_scale2d)``
        7. Execute clone, extend split mask, then split.

        Args:
            params: Parameter dict. Modified in-place.
            optimizers: Optimizer state dict. Modified in-place.
            state: Running strategy state. Modified in-place.
            step: Current training iteration.

        Returns:
            Tuple of ``(n_duplicated, n_split)``.
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
        # mx.eval needed for .item() -- acceptable, runs infrequently
        mx.eval(is_dupli)
        n_dupli = int(mx.sum(is_dupli.astype(mx.int32)).item())

        # Split: high gradient + large scale (over-reconstruction)
        is_large = ~is_small
        is_split = is_grad_high & is_large

        # Optional: also split based on 2D screen-space size
        if step < self.refine_scale2d_stop_iter:
            is_split = is_split | (state["radii"] > self.grow_scale2d)

        mx.eval(is_split)
        n_split = int(mx.sum(is_split.astype(mx.int32)).item())

        # Execute clone first
        if n_dupli > 0:
            duplicate(
                params=params, optimizers=optimizers, state=state, mask=is_dupli
            )

        # Extend the split mask to account for newly cloned Gaussians
        # Cloned Gaussians (appended at end) should NOT be split
        is_split = mx.concatenate(
            [is_split, mx.zeros((n_dupli,), dtype=mx.bool_)]
        )

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

        1. ``is_prune = sigmoid(opacities) < prune_opa``
        2. If ``step > reset_every`` (after first opacity reset):
           ``is_prune |= max(exp(scales)) > prune_scale3d * scene_scale``
        3. Optionally: ``is_prune |= radii > prune_scale2d``
        4. Remove all pruned Gaussians.

        Args:
            params: Parameter dict. Modified in-place.
            optimizers: Optimizer state dict. Modified in-place.
            state: Running strategy state. Modified in-place.
            step: Current training iteration.

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

            # Optional screen-size pruning (disabled by default)
            if step < self.refine_scale2d_stop_iter:
                is_too_big = is_too_big | (state["radii"] > self.prune_scale2d)

            is_prune = is_prune | is_too_big

        mx.eval(is_prune)
        n_prune = int(mx.sum(is_prune.astype(mx.int32)).item())
        if n_prune > 0:
            remove(
                params=params, optimizers=optimizers, state=state, mask=is_prune
            )

        return n_prune
