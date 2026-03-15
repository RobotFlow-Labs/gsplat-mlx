"""Gaussian densification strategies for adaptive Gaussian management.

Provides:

- :class:`Strategy`: Abstract base class defining the densification interface.
- :class:`DefaultStrategy`: Original 3DGS clone/split/prune/reset algorithm.
- :class:`MCMCStrategy`: MCMC-based strategy (stub, not yet implemented).
"""

from .base import Strategy
from .default import DefaultStrategy
from .mcmc import MCMCStrategy

__all__ = ["Strategy", "DefaultStrategy", "MCMCStrategy"]
