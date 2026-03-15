"""Spatial sorting of Gaussian splats for improved compression.

Sorts Gaussians by Morton code (Z-order curve) to improve spatial locality,
which leads to better compression ratios when quantising and saving as PNG.

This is a simplified alternative to the upstream PLAS-based sorting that does
not require the ``plas`` package.  It uses a Morton curve on the mean positions
which provides most of the compression benefit with zero extra dependencies.
"""

from typing import Dict

import mlx.core as mx
import numpy as np


def _interleave_bits_32(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute 3D Morton codes for coordinate arrays.

    Interleaves the bits of three 10-bit unsigned integers to produce a
    30-bit Morton code per element.

    Args:
        x: Unsigned integer x-coordinates (will be clipped to 10 bits).
        y: Unsigned integer y-coordinates.
        z: Unsigned integer z-coordinates.

    Returns:
        Array of 64-bit Morton codes.
    """
    def spread(v: np.ndarray) -> np.ndarray:
        """Spread 10 bits of v into every-third-bit positions."""
        v = v.astype(np.uint64)
        v = (v | (v << 16)) & np.uint64(0x030000FF)
        v = (v | (v << 8)) & np.uint64(0x0300F00F)
        v = (v | (v << 4)) & np.uint64(0x030C30C3)
        v = (v | (v << 2)) & np.uint64(0x09249249)
        return v

    return spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))


def sort_splats(splats: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sort Gaussians by Morton code for better spatial locality.

    Quantises the mean positions into a 10-bit grid per axis and computes
    a Morton (Z-order) code, then sorts all splat parameters by that code.
    This places spatially nearby Gaussians adjacent in memory which
    significantly improves PNG compression ratios.

    Args:
        splats: Dictionary of splat parameters.  Must contain a ``"means"``
            key with shape ``[N, 3]``.

    Returns:
        New dictionary with all arrays reordered by Morton code.
    """
    means_np = np.array(splats["means"])
    N = means_np.shape[0]

    # Quantise to 10-bit grid (0..1023)
    mins = means_np.min(axis=0)
    maxs = means_np.max(axis=0)
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-8, 1.0, ranges)
    normalised = (means_np - mins) / ranges  # [N, 3] in [0, 1]
    quantised = (normalised * 1023).astype(np.uint32).clip(0, 1023)

    morton = _interleave_bits_32(quantised[:, 0], quantised[:, 1], quantised[:, 2])
    order = np.argsort(morton).astype(np.uint32)
    order_mx = mx.array(order)

    sorted_splats = {}
    for k, v in splats.items():
        sorted_splats[k] = v[order_mx]
    return sorted_splats
