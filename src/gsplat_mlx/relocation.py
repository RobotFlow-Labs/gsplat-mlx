"""Gaussian relocation for MCMC-based 3D Gaussian Splatting.

Implements the relocation equations from
"3D Gaussian Splatting as Markov Chain Monte Carlo" (arXiv:2404.09591).

The upstream gsplat uses a CUDA kernel; here we implement the same math
in pure MLX operations.
"""

import math
from typing import Tuple

import mlx.core as mx


def compute_binomial_coefficients(n_max: int) -> mx.array:
    """Precompute binomial coefficients table for relocation.

    Builds Pascal's triangle up to ``n_max`` rows.

    Args:
        n_max: Maximum number of rows/columns.

    Returns:
        ``[n_max, n_max]`` array where ``result[n, k] = C(n, k)``.
    """
    # Build in Python (small table), then convert to MLX
    table = [[0] * n_max for _ in range(n_max)]
    for n in range(n_max):
        table[n][0] = 1
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k]
    return mx.array(table, dtype=mx.float32)


def compute_relocation(
    opacities: mx.array,  # [N]
    scales: mx.array,  # [N, 3]
    ratios: mx.array,  # [N]
    binoms: mx.array,  # [n_max, n_max]
) -> Tuple[mx.array, mx.array]:
    """Compute relocated Gaussian parameters from MCMC sampling ratios.

    From "3D Gaussian Splatting as Markov Chain Monte Carlo" (arXiv:2404.09591).

    The upstream uses a CUDA kernel. We implement the math in pure MLX.

    The key equations (Equation 9 in the paper):
      - ``new_opacity = 1 - (1 - opacity)^(1/ratio)``
      - ``new_scale = scale * sqrt(opacity / denom_sum)``

    where ``denom_sum`` is computed using binomial coefficients.

    Args:
        opacities: Per-Gaussian opacities ``[N]``, values in (0, 1).
        scales: Per-Gaussian scales ``[N, 3]``, positive values.
        ratios: Sampling ratios (how many copies) ``[N]``, integer values >= 1.
        binoms: Precomputed binomial lookup ``[n_max, n_max]``.

    Returns:
        A tuple ``(new_opacities, new_scales)``:

        **new_opacities**: ``[N]``
        **new_scales**: ``[N, 3]``
    """
    N = opacities.shape[0]
    n_max = binoms.shape[0]

    # Clamp ratios to [1, n_max]
    ratios = mx.clip(ratios.astype(mx.int32), 1, n_max)

    # new_opacity = 1 - (1 - opacity)^(1/ratio)
    new_opacities = 1.0 - mx.power(
        1.0 - opacities, 1.0 / ratios.astype(mx.float32)
    )

    # Compute denom_sum for scale adjustment per the CUDA kernel:
    # For each Gaussian idx with ratio n_idx:
    #   denom_sum = sum over i in [1..n_idx] of
    #     sum over k in [0..i-1] of
    #       binoms[i-1, k] * (-1)^k / sqrt(k+1) * new_opacity^(k+1)
    #
    # Then new_scale = scale * (opacity / denom_sum)
    #
    # We vectorize this by iterating over the max possible i,k values
    # and masking based on each Gaussian's ratio.

    # We'll compute this in a loop over i and k (n_max is typically small, e.g. 10-50)
    mx.eval(ratios)
    denom_sum = mx.zeros((N,), dtype=mx.float32)

    for i in range(1, n_max + 1):
        # Only contribute for Gaussians where ratio >= i
        i_mask = (ratios >= i).astype(mx.float32)  # [N]
        for k in range(i):
            bin_coeff = binoms[i - 1, k]  # scalar
            sign = (-1.0) ** k
            term = (
                bin_coeff
                * sign
                / math.sqrt(k + 1)
                * mx.power(new_opacities, k + 1)
            )
            denom_sum = denom_sum + i_mask * term

    # Avoid division by zero
    denom_sum = mx.where(mx.abs(denom_sum) < 1e-10, mx.ones_like(denom_sum), denom_sum)

    coeff = opacities / denom_sum  # [N]
    new_scales = scales * coeff[:, None]  # [N, 3]

    return new_opacities, new_scales
