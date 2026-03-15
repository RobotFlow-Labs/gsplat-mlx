"""Quaternion-Scale to Covariance and Precision Matrices.

Converts each 3D Gaussian's parameterization (unit quaternion + anisotropic scale)
into a full 3x3 covariance matrix and optionally its inverse (precision matrix).

Port of gsplat's ``_quat_scale_to_covar_preci`` from CUDA/PyTorch to MLX.
See PRD-03 for details.
"""

import mlx.core as mx
from typing import Optional, Tuple

from gsplat_mlx.core.math_utils import _quat_to_rotmat


# ---------------------------------------------------------------------------
# Upper-triangle index helpers (for a flattened 3x3 → 9 vector)
# ---------------------------------------------------------------------------
# For symmetric averaging:  (mat[i,j] + mat[j,i]) / 2
_TRIU_IDX = [0, 1, 2, 4, 5, 8]       # upper-tri in row-major 3x3
_TRIU_TRANSPOSE_IDX = [0, 3, 6, 4, 7, 8]  # corresponding transposed indices


def _extract_triu(mat: mx.array) -> mx.array:
    """Extract 6-element upper triangle from a [..., 3, 3] symmetric matrix.

    Averages ``(M[i,j] + M[j,i]) / 2`` for numerical symmetry.

    Args:
        mat: Symmetric matrices of shape ``[..., 3, 3]``.

    Returns:
        Upper-triangle elements of shape ``[..., 6]``.
    """
    batch_shape = mat.shape[:-2]
    flat = mat.reshape(batch_shape + (9,))
    upper = flat[..., mx.array(_TRIU_IDX)]
    upper_t = flat[..., mx.array(_TRIU_TRANSPOSE_IDX)]
    return (upper + upper_t) / 2.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def quat_scale_to_covar_preci(
    quats: mx.array,       # [..., N, 4]
    scales: mx.array,      # [..., N, 3]
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    """Convert quaternion rotation + anisotropic scale to covariance / precision.

    Uses the factored form ``Sigma = M M^T`` where ``M = R diag(s)`` which
    guarantees symmetric positive semi-definiteness by construction.

    All operations are standard differentiable MLX ops so ``mx.grad`` /
    ``mx.value_and_grad`` can back-propagate through this function directly
    (Option A from PRD-03).

    Args:
        quats:  Quaternions ``[..., 4]`` in ``(w, x, y, z)`` convention.
        scales: Per-axis scales ``[..., 3]``.  Must be > 0 for valid results.
        compute_covar: Whether to compute the covariance matrix.
        compute_preci: Whether to compute the precision (inverse covariance).
        triu: If ``True``, return only the 6-element upper triangle instead
              of the full 3x3 matrix.

    Returns:
        A tuple ``(covars, precis)`` where each is either:
        - ``[..., 3, 3]`` (``triu=False``), or
        - ``[..., 6]``    (``triu=True``), or
        - ``None``        if not requested.
    """
    # 1. Rotation matrix from quaternion (normalisation inside _quat_to_rotmat)
    R = _quat_to_rotmat(quats)  # [..., 3, 3]

    covars: Optional[mx.array] = None
    precis: Optional[mx.array] = None

    # 2. Covariance: Sigma = (R * s) (R * s)^T
    if compute_covar:
        M = R * scales[..., None, :]            # [..., 3, 3]
        covars = mx.einsum("...ij,...kj->...ik", M, M)  # [..., 3, 3]
        if triu:
            covars = _extract_triu(covars)       # [..., 6]

    # 3. Precision: Sigma^{-1} = (R / s) (R / s)^T
    if compute_preci:
        P = R * (1.0 / scales)[..., None, :]    # [..., 3, 3]
        precis = mx.einsum("...ij,...kj->...ik", P, P)  # [..., 3, 3]
        if triu:
            precis = _extract_triu(precis)       # [..., 6]

    return covars, precis
