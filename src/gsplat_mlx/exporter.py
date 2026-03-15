"""Gaussian Splat exporter for PLY and antimatter15 .splat formats.

Port of ``gsplat/exporter.py`` from PyTorch to MLX / NumPy.
All heavy-lifting uses NumPy since this is I/O code, not differentiable.
"""

import math
import struct
from io import BytesIO
from typing import Dict, Literal, Optional, Union

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Log transform utilities (ported from gsplat/utils.py)
# ---------------------------------------------------------------------------


def log_transform(x: mx.array) -> mx.array:
    """Applies ``sign(x) * log1p(|x|)`` element-wise.

    Useful for compressing the dynamic range of scales or other unbounded
    parameters before saving/visualizing.

    Args:
        x: Input array.

    Returns:
        Transformed array with the same shape and dtype.
    """
    return mx.sign(x) * mx.log1p(mx.abs(x))


def inverse_log_transform(y: mx.array) -> mx.array:
    """Inverse of :func:`log_transform`: ``sign(y) * expm1(|y|)``.

    Args:
        y: Input array (output of :func:`log_transform`).

    Returns:
        Reconstructed array with the same shape and dtype.
    """
    return mx.sign(y) * mx.expm1(mx.abs(y))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert an mx.array, np.ndarray, or Python scalar to a NumPy array."""
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def sh2rgb(sh: np.ndarray) -> np.ndarray:
    """Convert degree-0 spherical harmonic coefficients to RGB.

    Args:
        sh: SH coefficients. Shape ``(N, 3)``.

    Returns:
        RGB values in [0, 1] range. Shape ``(N, 3)``.
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


# ---------------------------------------------------------------------------
# Morton code helpers (for spatial sorting)
# ---------------------------------------------------------------------------


def part1by2_vec(x: np.ndarray) -> np.ndarray:
    """Interleave bits of *x* with two zero-bits between each.

    Used internally by :func:`encode_morton3_vec`. Input values should be
    10-bit unsigned integers (0..1023).

    Args:
        x: Integer array. Shape ``(N,)``.

    Returns:
        Bit-interleaved integers. Shape ``(N,)``.
    """
    x = x.astype(np.int64)
    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8)) & 0x0300F00F
    x = (x ^ (x << 4)) & 0x030C30C3
    x = (x ^ (x << 2)) & 0x09249249
    return x


def encode_morton3_vec(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Compute 3D Morton codes (Z-order curve) for coordinate triplets.

    Args:
        x, y, z: Integer arrays each of shape ``(N,)``.

    Returns:
        Morton codes. Shape ``(N,)``.
    """
    return (part1by2_vec(z) << 2) | (part1by2_vec(y) << 1) | part1by2_vec(x)


def sort_centers(centers: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sort *indices* by 3D Morton code of the corresponding *centers*.

    This produces a spatially coherent ordering that improves compression
    and cache locality for downstream viewers.

    Args:
        centers: 3D positions. Shape ``(N, 3)``.
        indices: Index array. Shape ``(N,)``.

    Returns:
        Reordered indices. Shape ``(N,)``.
    """
    min_vals = centers.min(axis=0)
    max_vals = centers.max(axis=0)
    lengths = max_vals - min_vals
    lengths[lengths == 0] = 1.0  # prevent division by zero

    scaled = np.floor((centers - min_vals) / lengths * 1024).astype(np.int32)
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]

    morton = encode_morton3_vec(x, y, z)
    sorted_order = np.argsort(morton)
    return indices[sorted_order]


# ---------------------------------------------------------------------------
# Bit-packing helpers (for .splat format)
# ---------------------------------------------------------------------------


def pack_unorm(value: np.ndarray, bits: int) -> np.ndarray:
    """Pack floating-point values into unsigned integers of *bits* width.

    Args:
        value: Float array in [0, 1]. Shape ``(N,)``.
        bits: Target bit width.

    Returns:
        Packed integer array. Shape ``(N,)``.
    """
    t = (1 << bits) - 1
    packed = np.clip(np.floor(value * t + 0.5), 0, t)
    return packed.astype(np.int64)


def pack_111011(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Pack three floats into a 32-bit integer (11-10-11 layout).

    Args:
        x, y, z: Float arrays in [0, 1]. Each of shape ``(N,)``.

    Returns:
        Packed 32-bit integers. Shape ``(N,)``.
    """
    return (pack_unorm(x, 11) << 21) | (pack_unorm(y, 10) << 11) | pack_unorm(z, 11)


def pack_8888(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Pack four floats into a 32-bit integer (8-8-8-8 layout).

    Args:
        x, y, z, w: Float arrays in [0, 1]. Each of shape ``(N,)``.

    Returns:
        Packed 32-bit integers. Shape ``(N,)``.
    """
    return (
        (pack_unorm(x, 8) << 24)
        | (pack_unorm(y, 8) << 16)
        | (pack_unorm(z, 8) << 8)
        | pack_unorm(w, 8)
    )


def pack_rotation(q: np.ndarray) -> np.ndarray:
    """Pack normalized quaternions into 32-bit integers.

    Uses the *smallest-three* encoding: store the index of the largest
    component (2 bits) and pack the remaining three components as 10-bit
    unsigned normalized values.

    Args:
        q: Quaternion array. Shape ``(N, 4)``.

    Returns:
        Packed 32-bit integers. Shape ``(N,)``.
    """
    # Normalize
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    q = q / norms

    # Find largest component and flip so it's positive
    largest = np.argmax(np.abs(q), axis=-1)  # (N,)
    batch_idx = np.arange(q.shape[0])
    flip_mask = q[batch_idx, largest] < 0
    q[flip_mask] *= -1

    # Pre-computed index tables for the three non-largest components
    precomputed = np.array(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64
    )
    pack_idx = precomputed[largest]  # (N, 3)
    components = q[batch_idx[:, None], pack_idx]  # (N, 3)

    norm_factor = math.sqrt(2) * 0.5
    scaled = components * norm_factor + 0.5
    packed = pack_unorm(scaled, 10)  # (N, 3)

    result = (
        (largest.astype(np.int64) << 30)
        | (packed[:, 0] << 20)
        | (packed[:, 1] << 10)
        | packed[:, 2]
    )
    return result


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------


def _splat2ply_bytes(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shN: np.ndarray,
) -> bytes:
    """Serialize Gaussians to a standard binary little-endian PLY.

    This format is supported by virtually all 3DGS viewers.

    Args:
        means: Shape ``(N, 3)``.
        scales: Shape ``(N, 3)``.
        quats: Shape ``(N, 4)``.
        opacities: Shape ``(N,)``.
        sh0: Shape ``(N, 3)``.
        shN: Shape ``(N, K*3)``.

    Returns:
        Raw PLY bytes.
    """
    num_splats = means.shape[0]
    buf = BytesIO()

    # Header
    buf.write(b"ply\n")
    buf.write(b"format binary_little_endian 1.0\n")
    buf.write(f"element vertex {num_splats}\n".encode())
    buf.write(b"property float x\n")
    buf.write(b"property float y\n")
    buf.write(b"property float z\n")
    for i, (data, prefix) in enumerate([(sh0, "f_dc"), (shN, "f_rest")]):
        for j in range(data.shape[1]):
            buf.write(f"property float {prefix}_{j}\n".encode())
    buf.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buf.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buf.write(f"property float rot_{i}\n".encode())
    buf.write(b"end_header\n")

    # Data
    splat_data = np.concatenate(
        [means, sh0, shN, opacities[:, None], scales, quats], axis=1
    ).astype(np.dtype(np.float32).newbyteorder("<"))
    buf.write(splat_data.tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# .splat writer (antimatter15 format)
# ---------------------------------------------------------------------------


def _splat2splat_bytes(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
) -> bytes:
    """Serialize Gaussians to the antimatter15 ``.splat`` format.

    Each Gaussian is encoded as exactly 32 bytes:
      - 12 bytes: position (3x float32)
      - 12 bytes: scale (3x float32, exp applied)
      - 4 bytes: color (RGBA as uint8)
      - 4 bytes: rotation (quaternion as uint8)

    Args:
        means: Shape ``(N, 3)``.
        scales: Shape ``(N, 3)``.
        quats: Shape ``(N, 4)``.
        opacities: Shape ``(N,)``.
        sh0: Shape ``(N, 3)``.

    Returns:
        Raw ``.splat`` bytes.
    """
    # Preprocess
    scales_exp = np.exp(scales)
    sh0_color = sh2rgb(sh0)
    sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
    colors = np.concatenate([sh0_color, sigmoid_opacities[:, None]], axis=1)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    rots = (quats / norms) * 128 + 128
    rots = np.clip(rots, 0, 255).astype(np.uint8)

    # Sort
    num_splats = means.shape[0]
    indices = sort_centers(means, np.arange(num_splats))

    means = means[indices]
    scales_exp = scales_exp[indices]
    colors = colors[indices]
    rots = rots[indices]

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    means_np = means.astype(float_dtype)
    scales_np = scales_exp.astype(float_dtype)

    buf = BytesIO()
    for i in range(num_splats):
        buf.write(means_np[i].tobytes())
        buf.write(scales_np[i].tobytes())
        buf.write(colors[i].tobytes())
        buf.write(rots[i].tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def export_splats(
    splats: Dict[str, Union[mx.array, np.ndarray]],
    path: str,
    format: Literal["ply", "splat"] = "ply",
) -> bytes:
    """Export trained Gaussian splats to disk.

    Accepts a dictionary of parameters (as produced by the training loop)
    and writes them to *path* in the requested format.

    Args:
        splats: Dictionary with the following keys:

            - ``"means"``: Shape ``(N, 3)``
            - ``"scales"``: Shape ``(N, 3)``
            - ``"quats"``: Shape ``(N, 4)``
            - ``"opacities"``: Shape ``(N,)``
            - ``"sh0"``: Shape ``(N, 1, 3)``
            - ``"shN"``: Shape ``(N, K, 3)``

        path: Output file path (e.g. ``"output.ply"`` or ``"output.splat"``).
        format: ``"ply"`` (standard PLY) or ``"splat"`` (antimatter15).

    Returns:
        The raw bytes that were written to *path*.
    """
    # Convert everything to numpy
    np_splats = {k: _to_numpy(v) for k, v in splats.items()}

    means = np_splats["means"]
    scales = np_splats["scales"]
    quats = np_splats["quats"]
    opacities = np_splats["opacities"]
    sh0 = np_splats["sh0"]
    shN = np_splats["shN"]

    total = means.shape[0]
    assert means.shape == (total, 3), f"means must be (N, 3), got {means.shape}"
    assert scales.shape == (total, 3), f"scales must be (N, 3), got {scales.shape}"
    assert quats.shape == (total, 4), f"quats must be (N, 4), got {quats.shape}"
    assert opacities.shape == (total,), f"opacities must be (N,), got {opacities.shape}"
    assert sh0.shape == (total, 1, 3), f"sh0 must be (N, 1, 3), got {sh0.shape}"
    assert (
        shN.ndim == 3 and shN.shape[0] == total and shN.shape[2] == 3
    ), f"shN must be (N, K, 3), got {shN.shape}"

    # Reshape SH: (N, 1, 3) -> (N, 3) and (N, K, 3) -> (N, K*3)
    sh0 = sh0.squeeze(1)  # (N, 3)
    shN = shN.transpose(0, 2, 1).reshape(total, -1)  # (N, K*3)

    # Filter NaN / Inf
    invalid = (
        np.isnan(means).any(axis=1)
        | np.isinf(means).any(axis=1)
        | np.isnan(scales).any(axis=1)
        | np.isinf(scales).any(axis=1)
        | np.isnan(quats).any(axis=1)
        | np.isinf(quats).any(axis=1)
        | np.isnan(opacities)
        | np.isinf(opacities)
        | np.isnan(sh0).any(axis=1)
        | np.isinf(sh0).any(axis=1)
        | np.isnan(shN).any(axis=1)
        | np.isinf(shN).any(axis=1)
    )
    valid = ~invalid
    means = means[valid]
    scales = scales[valid]
    quats = quats[valid]
    opacities = opacities[valid]
    sh0 = sh0[valid]
    shN = shN[valid]

    if format == "ply":
        data = _splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
    elif format == "splat":
        data = _splat2splat_bytes(means, scales, quats, opacities, sh0)
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'ply' or 'splat'.")

    with open(path, "wb") as f:
        f.write(data)

    return data
