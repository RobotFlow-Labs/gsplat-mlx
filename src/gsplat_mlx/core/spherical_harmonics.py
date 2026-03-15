"""Spherical Harmonics evaluation for 3D Gaussian Splatting.

Evaluates SH basis functions at unit directions and computes view-dependent
RGB colors from SH coefficients. Supports SH degrees 0-4 (1, 4, 9, 16, 25
basis functions).

Port of gsplat's ``_eval_sh_bases_fast`` and ``_spherical_harmonics`` from
PyTorch to MLX. Uses the fast evaluation method from:
    "Efficient Spherical Harmonic Evaluation", Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/

See PRD-04 for details.
"""

import mlx.core as mx


# ---------------------------------------------------------------------------
# Internal: SH basis evaluation
# ---------------------------------------------------------------------------


def _eval_sh_bases_fast(basis_dim: int, dirs: mx.array) -> mx.array:
    """Evaluate SH basis functions at unit directions.

    Uses the fast method from "Efficient Spherical Harmonic Evaluation"
    (Peter-Pike Sloan, JCGT 2013).

    Args:
        basis_dim: Number of basis functions (1, 4, 9, 16, or 25).
        dirs: Unit directions ``[..., 3]``.

    Returns:
        SH basis values ``[..., basis_dim]``.
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    # Degree 0: 1 basis function
    b0 = mx.full(x.shape, 0.2820947917738781, dtype=dirs.dtype)
    bases = [b0]

    if basis_dim <= 1:
        return mx.stack(bases, axis=-1)

    # Degree 1: 3 basis functions (indices 1, 2, 3)
    fTmpA = -0.48860251190292
    b1 = fTmpA * y        # Y_1^{-1}
    b2 = -fTmpA * z       # Y_1^0
    b3 = fTmpA * x        # Y_1^1
    bases.extend([b1, b2, b3])

    if basis_dim <= 4:
        return mx.stack(bases, axis=-1)

    # Degree 2: 5 basis functions (indices 4-8)
    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    b4 = fTmpA * fS1
    b5 = fTmpB * y
    b6 = 0.9461746957575601 * z2 - 0.3153915652525201
    b7 = fTmpB * x
    b8 = fTmpA * fC1
    bases.extend([b4, b5, b6, b7, b8])

    if basis_dim <= 9:
        return mx.stack(bases, axis=-1)

    # Degree 3: 7 basis functions (indices 9-15)
    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    b9 = fTmpA * fS2
    b10 = fTmpB * fS1
    b11 = fTmpC * y
    b12 = z * (1.865881662950577 * z2 - 1.119528997770346)
    b13 = fTmpC * x
    b14 = fTmpB * fC1
    b15 = fTmpA * fC2
    bases.extend([b9, b10, b11, b12, b13, b14, b15])

    if basis_dim <= 16:
        return mx.stack(bases, axis=-1)

    # Degree 4: 9 basis functions (indices 16-24)
    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    b16 = fTmpA * fS3
    b17 = fTmpB * fS2
    b18 = fTmpC * fS1
    b19 = fTmpD * y
    b20 = (
        1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346)
        + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    )
    b21 = fTmpD * x
    b22 = fTmpC * fC1
    b23 = fTmpB * fC2
    b24 = fTmpA * fC3
    bases.extend([b16, b17, b18, b19, b20, b21, b22, b23, b24])

    return mx.stack(bases, axis=-1)


# ---------------------------------------------------------------------------
# Public API: spherical_harmonics with custom VJP
# ---------------------------------------------------------------------------


def _sh_forward(degrees_to_use: int, dirs: mx.array, coeffs: mx.array) -> mx.array:
    """Forward pass for SH evaluation (shared by public API and VJP).

    Args:
        degrees_to_use: SH degree (0-4).
        dirs: View directions ``[..., 3]`` (not necessarily normalized).
        coeffs: SH coefficients ``[..., K, 3]`` where K >= (degrees_to_use+1)^2.

    Returns:
        RGB colors ``[..., 3]``.
    """
    # Normalize directions
    dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
    dirs_normalized = dirs / mx.maximum(dirs_norm, 1e-8)

    num_bases = (degrees_to_use + 1) ** 2
    bases = _eval_sh_bases_fast(num_bases, dirs_normalized)  # [..., num_bases]

    # Pad bases to match coeffs dimension if K > num_bases
    K = coeffs.shape[-2]
    if num_bases < K:
        padding = mx.zeros(bases.shape[:-1] + (K - num_bases,), dtype=bases.dtype)
        bases = mx.concatenate([bases, padding], axis=-1)

    # Dot product: sum over basis dimension
    # bases: [..., K], coeffs: [..., K, 3]
    # result: [..., 3]
    return mx.sum(mx.expand_dims(bases, axis=-1) * coeffs, axis=-2)


def _make_sh_fn(degrees_to_use: int):
    """Create a custom-function SH evaluator for a given degree.

    Uses closure to capture the integer ``degrees_to_use`` so that
    ``@mx.custom_function`` only receives array arguments.
    """
    num_bases = (degrees_to_use + 1) ** 2

    @mx.custom_function
    def _sh(dirs: mx.array, coeffs: mx.array) -> mx.array:
        return _sh_forward(degrees_to_use, dirs, coeffs)

    @_sh.vjp
    def _sh_vjp(primals, cotangent, output):
        dirs, coeffs = primals
        v_colors = cotangent  # [..., 3]

        # Recompute forward intermediates
        dirs_norm = mx.sqrt(mx.sum(dirs * dirs, axis=-1, keepdims=True))
        dirs_normalized = dirs / mx.maximum(dirs_norm, 1e-8)
        bases = _eval_sh_bases_fast(num_bases, dirs_normalized)

        K = coeffs.shape[-2]
        if num_bases < K:
            padding = mx.zeros(
                bases.shape[:-1] + (K - num_bases,), dtype=bases.dtype
            )
            bases = mx.concatenate([bases, padding], axis=-1)

        # Gradient w.r.t. coeffs (closed-form outer product)
        # v_coeffs[..., k, c] = v_colors[..., c] * bases[..., k]
        v_coeffs = mx.expand_dims(bases, axis=-1) * mx.expand_dims(v_colors, axis=-2)

        # Gradient w.r.t. dirs via auto-diff
        def fwd_for_dirs_grad(d):
            d_norm = mx.sqrt(mx.sum(d * d, axis=-1, keepdims=True))
            d_normalized = d / mx.maximum(d_norm, 1e-8)
            b = _eval_sh_bases_fast(num_bases, d_normalized)
            if num_bases < K:
                b = mx.concatenate(
                    [b, mx.zeros(b.shape[:-1] + (K - num_bases,), dtype=b.dtype)],
                    axis=-1,
                )
            return mx.sum(mx.expand_dims(b, axis=-1) * coeffs, axis=-2)

        _, v_dirs_fn = mx.vjp(fwd_for_dirs_grad, [dirs], [v_colors])
        v_dirs = v_dirs_fn[0]

        return (v_dirs, v_coeffs)

    return _sh


_SH_FN_CACHE = {}


def spherical_harmonics(
    degrees_to_use: int, dirs: mx.array, coeffs: mx.array
) -> mx.array:
    """Evaluate spherical harmonics to produce RGB colors.

    Args:
        degrees_to_use: SH degree (0-4).
        dirs: View directions ``[..., 3]`` (not necessarily normalized).
        coeffs: SH coefficients ``[..., K, 3]`` where K >= (degrees_to_use+1)^2.

    Returns:
        RGB colors ``[..., 3]``.

    Raises:
        AssertionError: If coeffs doesn't have enough basis functions for
            the requested degree.
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], (
        f"coeffs has {coeffs.shape[-2]} bases but degree {degrees_to_use} "
        f"requires {(degrees_to_use + 1) ** 2}"
    )
    if degrees_to_use not in _SH_FN_CACHE:
        _SH_FN_CACHE[degrees_to_use] = _make_sh_fn(degrees_to_use)
    return _SH_FN_CACHE[degrees_to_use](dirs, coeffs)
