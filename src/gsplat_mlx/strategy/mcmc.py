"""MCMC-based densification strategy (stub -- not yet implemented).

From: *3D Gaussian Splatting as Markov Chain Monte Carlo*
(Kheradmand et al., 2024, arXiv:2404.09591)

Full implementation deferred to a separate PRD.

Upstream reference: ``repositories/gsplat-upstream/gsplat/strategy/mcmc.py``
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import Strategy


@dataclass
class MCMCStrategy(Strategy):
    """MCMC-based densification strategy (stub -- not yet implemented).

    This strategy:

    - Periodically teleports low-opacity Gaussians to high-opacity regions.
    - Periodically adds new Gaussians sampled from the opacity distribution.
    - Periodically injects noise into positions for MCMC exploration.

    All methods raise ``NotImplementedError`` until the full port is complete.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    noise_injection_stop_iter: int = -1
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False

    def initialize_state(self, **kwargs: Any) -> Dict[str, Any]:
        """Not implemented. Raises ``NotImplementedError``."""
        raise NotImplementedError("MCMCStrategy is not yet ported to MLX.")

    def step_post_backward(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented. Raises ``NotImplementedError``."""
        raise NotImplementedError("MCMCStrategy is not yet ported to MLX.")
