"""Abstract base class for Gaussian densification strategies.

Defines the interface that all strategies must implement. Strategies are called
at specific points in the training loop:

- ``step_pre_backward``: before the backward pass (for validation, gradient setup).
- ``step_post_backward``: after the backward pass (for grow/prune/reset decisions).

Upstream reference: ``repositories/gsplat-upstream/gsplat/strategy/base.py``

Port notes (PyTorch -> MLX):
- No ``torch.nn.Parameter`` wrapping; plain ``mx.array`` in dicts.
- Optimizer state keyed by param name (string), not by Parameter identity.
- No ``requires_grad`` flag; presence in optimizer dict indicates trainability.
- No ``@torch.no_grad()`` needed; MLX uses functional ``mx.grad()``.
"""

from dataclasses import dataclass
from typing import Any, Dict

import mlx.core as mx


@dataclass
class Strategy:
    """Base class for Gaussian densification strategies.

    Subclasses implement specific densification algorithms (e.g., the original
    3DGS clone/split/prune/reset cycle, or the MCMC-based strategy).
    """

    def check_sanity(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
    ) -> None:
        """Verify that params and optimizers are consistent.

        Checks that every optimizer entry has a corresponding parameter key.
        In MLX, there is no ``requires_grad`` flag -- the presence of an
        optimizer entry is the indicator that a parameter is trainable.

        Unlike upstream (which asserts exact equality between trainable params
        and optimizer keys), we only assert that optimizer keys are a subset
        of parameter keys, since some params may be frozen (no optimizer).

        Args:
            params: Dictionary mapping parameter names to ``mx.array``.
            optimizers: Dictionary mapping parameter names to optimizer state
                dicts (e.g., ``{"exp_avg": mx.array, "exp_avg_sq": mx.array}``).

        Raises:
            AssertionError: If optimizer keys are not a subset of param keys.
        """
        param_names = set(params.keys())
        opt_names = set(optimizers.keys())
        assert opt_names.issubset(param_names), (
            "Optimizer keys must be a subset of parameter keys, "
            f"but got optimizer keys {opt_names - param_names} not in params."
        )

    def initialize_state(self, **kwargs: Any) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state dict should be passed to ``step_pre_backward()``
        and ``step_post_backward()`` on every training step.

        Returns:
            State dict. Default implementation returns an empty dict.
        """
        return {}

    def step_pre_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> None:
        """Called before the backward pass. Default: no-op.

        In PyTorch upstream, this calls ``retain_grad()`` on projected means.
        In MLX (functional gradients), this is primarily a validation step.

        Args:
            params: Current parameter dict.
            optimizers: Current optimizer state dict.
            state: Running strategy state from ``initialize_state()``.
            step: Current training iteration.
            info: Dict populated by the rasterization forward pass
                (contains projected means, radii, etc.).
        """
        pass

    def step_post_backward(
        self,
        params: Dict[str, mx.array],
        optimizers: Dict[str, Dict[str, mx.array]],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called after the backward pass. Default: no-op.

        Subclasses implement gradient accumulation and periodic
        grow/prune/reset logic here.

        Args:
            params: Current parameter dict. May be modified in-place
                (Gaussians added/removed).
            optimizers: Current optimizer state dict. Modified in-place
                to stay synchronized with params.
            state: Running strategy state. Modified in-place.
            step: Current training iteration.
            info: Dict with rasterization outputs and gradient info.
            **kwargs: Additional strategy-specific arguments.
        """
        pass
