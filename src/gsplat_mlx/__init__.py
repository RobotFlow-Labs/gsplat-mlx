"""gsplat-mlx: 3D Gaussian Splatting rasterizer for Apple MLX."""

from gsplat_mlx._version import VERSION

__version__ = VERSION

from .rendering import rasterization, RenderMode, RasterizeMode
from .core.covariance import quat_scale_to_covar_preci
from .core.spherical_harmonics import spherical_harmonics
from .core.projection import fully_fused_projection, world_to_cam
from .core.intersection import isect_tiles, isect_offset_encode
from .core.rasterization import rasterize_to_pixels
from .core.accumulate import accumulate
from .core.cameras import CameraModel

__all__ = [
    "__version__",
    "rasterization",
    "RenderMode",
    "RasterizeMode",
    "quat_scale_to_covar_preci",
    "spherical_harmonics",
    "fully_fused_projection",
    "world_to_cam",
    "isect_tiles",
    "isect_offset_encode",
    "rasterize_to_pixels",
    "accumulate",
    "CameraModel",
]
