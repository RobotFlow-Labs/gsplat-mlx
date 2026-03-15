"""Core module for gsplat-mlx — constants and foundational utilities."""

from gsplat_mlx.core.intersection import (
    isect_tiles,
    isect_offset_encode,
)
from gsplat_mlx.core.constants import (
    ALPHA_THRESHOLD,
    MAX_ALPHA,
    MAX_KERNEL_DENSITY_CUTOFF,
    TRANSMITTANCE_THRESHOLD,
)
from gsplat_mlx.core.math_utils import (
    _assert_shape,
    _cross,
    _numerically_stable_norm2,
    FullPolynomialProxy,
    OddPolynomialProxy,
    EvenPolynomialProxy,
    _eval_poly_inverse_horner_newton,
    _safe_normalize,
    _rotmat_to_quat,
    _quat_normalize_rotation,
    _quat_inverse,
    _quat_rotate,
    _quat_multiply,
    _quat_slerp,
    _quat_to_rotmat,
    _quat_scale_to_matrix,
    _quat_scale_to_covar_preci,
    _quat_scale_to_preci_half,
    compute_inverse_polynomial,
)
from gsplat_mlx.core.covariance import quat_scale_to_covar_preci
from gsplat_mlx.core.spherical_harmonics import spherical_harmonics
from gsplat_mlx.core.projection import (
    world_to_cam,
    persp_proj,
    fisheye_proj,
    ortho_proj,
    fully_fused_projection,
)
from gsplat_mlx.core.rasterization import rasterize_to_pixels
from gsplat_mlx.core.rasterization_mlx import rasterize_to_pixels_mlx
from gsplat_mlx.core.accumulate import (
    render_weight_from_alpha,
    accumulate_along_rays,
    accumulate,
)
from gsplat_mlx.core.cameras import CameraModel

__all__ = [
    # Intersection
    "isect_tiles",
    "isect_offset_encode",
    # Constants
    "ALPHA_THRESHOLD",
    "MAX_ALPHA",
    "TRANSMITTANCE_THRESHOLD",
    "MAX_KERNEL_DENSITY_CUTOFF",
    # Math utilities (public API only)
    "FullPolynomialProxy",
    "OddPolynomialProxy",
    "EvenPolynomialProxy",
    "compute_inverse_polynomial",
    # Covariance (public API)
    "quat_scale_to_covar_preci",
    # Spherical harmonics
    "spherical_harmonics",
    # Projection
    "world_to_cam",
    "persp_proj",
    "fisheye_proj",
    "ortho_proj",
    "fully_fused_projection",
    # Rasterization
    "rasterize_to_pixels",
    "rasterize_to_pixels_mlx",
    # Accumulate / compositing
    "render_weight_from_alpha",
    "accumulate_along_rays",
    "accumulate",
    # Cameras
    "CameraModel",
]
