"""2D Gaussian Splatting (surfel) core module.

Provides projection and rasterization primitives for 2D Gaussian surfels
(flat disk primitives embedded in 3D space).
"""

from gsplat_mlx.core_2dgs.projection_2dgs import fully_fused_projection_2dgs
from gsplat_mlx.core_2dgs.rasterization_2dgs import rasterize_to_pixels_2dgs

__all__ = [
    "fully_fused_projection_2dgs",
    "rasterize_to_pixels_2dgs",
]
