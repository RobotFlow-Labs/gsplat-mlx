"""Core module for gsplat-mlx — constants and foundational utilities."""

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

__all__ = [
    # Constants
    "ALPHA_THRESHOLD",
    "MAX_ALPHA",
    "TRANSMITTANCE_THRESHOLD",
    "MAX_KERNEL_DENSITY_CUTOFF",
    # Math utilities
    "_assert_shape",
    "_cross",
    "_numerically_stable_norm2",
    "FullPolynomialProxy",
    "OddPolynomialProxy",
    "EvenPolynomialProxy",
    "_eval_poly_inverse_horner_newton",
    "_safe_normalize",
    "_rotmat_to_quat",
    "_quat_normalize_rotation",
    "_quat_inverse",
    "_quat_rotate",
    "_quat_multiply",
    "_quat_slerp",
    "_quat_to_rotmat",
    "_quat_scale_to_matrix",
    "_quat_scale_to_covar_preci",
    "_quat_scale_to_preci_half",
    "compute_inverse_polynomial",
]
