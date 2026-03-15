"""PNG-based compression for trained Gaussian splat models.

Implements 8-bit and 16-bit quantisation of splat parameters into lossless
PNG images.  Combined with Morton-code spatial sorting this achieves
significant compression (often ~10x) on standard 3DGS models.

Limitations
-----------
- SH coefficients (``shN``) are compressed via NPZ rather than k-means
  clustering.  The upstream PyTorch version uses ``torchpq`` for k-means
  compression of higher-order SH bands; that dependency is not available
  on MLX.  As a result, models with many SH bands will see less compression
  than the upstream implementation.
- The number of Gaussians is rounded down to the nearest perfect square
  (the lowest-opacity Gaussians are dropped) so that the data can be
  reshaped into a square image.

Port of ``gsplat/compression/png_compression.py`` from PyTorch to MLX.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import mlx.core as mx
import numpy as np

from gsplat_mlx.compression.sort import sort_splats


# ---------------------------------------------------------------------------
# Helpers: log-transform (same as upstream gsplat.utils)
# ---------------------------------------------------------------------------

def _log_transform(x: mx.array) -> mx.array:
    """Apply ``sign(x) * log(1 + |x|)`` element-wise."""
    return mx.sign(x) * mx.log1p(mx.abs(x))


def _inverse_log_transform(x: mx.array) -> mx.array:
    """Inverse of :func:`_log_transform`."""
    return mx.sign(x) * (mx.exp(mx.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Quaternion normalisation
# ---------------------------------------------------------------------------

def _normalize_quats(q: mx.array) -> mx.array:
    """Normalise quaternions to unit length."""
    return q / mx.maximum(mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True)), mx.array(1e-8))


# ---------------------------------------------------------------------------
# Crop to square
# ---------------------------------------------------------------------------

def _crop_to_square(splats: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Remove lowest-opacity Gaussians so that N is a perfect square."""
    n_gs = splats["means"].shape[0]
    n_side = int(n_gs ** 0.5)
    n_crop = n_gs - n_side * n_side
    if n_crop == 0:
        return splats
    # Sort by opacity descending, keep top n_side^2
    opacities_np = np.array(splats["opacities"]).ravel()
    keep = np.argsort(opacities_np)[::-1][: n_side * n_side]
    keep.sort()  # preserve relative order
    keep_mx = mx.array(keep.astype(np.uint32))
    cropped = {}
    for k, v in splats.items():
        cropped[k] = v[keep_mx]
    return cropped


# ---------------------------------------------------------------------------
# Low-level compress / decompress helpers
# ---------------------------------------------------------------------------

def _compress_png_8bit(
    compress_dir: str,
    param_name: str,
    params: mx.array,
    n_side: int,
) -> Dict[str, Any]:
    """Quantise to 8-bit and save as PNG."""
    import imageio.v2 as imageio

    arr = np.array(params).reshape(n_side, n_side, -1)
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-12, 1.0, ranges)
    normalised = (arr - mins) / ranges
    img = (normalised * 255.0).round().clip(0, 255).astype(np.uint8)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)
    return {
        "shape": list(params.shape),
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }


def _decompress_png_8bit(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
) -> mx.array:
    """Decompress 8-bit PNG back to float32."""
    import imageio.v2 as imageio

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    normalised = img.astype(np.float32) / 255.0
    mins = np.array(meta["mins"], dtype=np.float32)
    maxs = np.array(meta["maxs"], dtype=np.float32)
    arr = normalised * (maxs - mins) + mins
    arr = arr.reshape(meta["shape"])
    return mx.array(arr)


def _compress_png_16bit(
    compress_dir: str,
    param_name: str,
    params: mx.array,
    n_side: int,
) -> Dict[str, Any]:
    """Quantise to 16-bit and save as two 8-bit PNGs (low + high bytes)."""
    import imageio.v2 as imageio

    arr = np.array(params).reshape(n_side, n_side, -1)
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    ranges = maxs - mins
    ranges = np.where(ranges < 1e-12, 1.0, ranges)
    normalised = (arr - mins) / ranges
    img16 = (normalised * 65535.0).round().clip(0, 65535).astype(np.uint16)

    img_lo = (img16 & 0xFF).astype(np.uint8)
    img_hi = ((img16 >> 8) & 0xFF).astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}_l.png"), img_lo)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}_u.png"), img_hi)
    return {
        "shape": list(params.shape),
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }


def _decompress_png_16bit(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
) -> mx.array:
    """Decompress 16-bit PNG pair back to float32."""
    import imageio.v2 as imageio

    img_lo = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_hi = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img16 = img_lo.astype(np.uint16) + (img_hi.astype(np.uint16) << 8)
    normalised = img16.astype(np.float32) / 65535.0
    mins = np.array(meta["mins"], dtype=np.float32)
    maxs = np.array(meta["maxs"], dtype=np.float32)
    arr = normalised * (maxs - mins) + mins
    arr = arr.reshape(meta["shape"])
    return mx.array(arr)


def _compress_npz(
    compress_dir: str,
    param_name: str,
    params: mx.array,
    n_side: int,
) -> Dict[str, Any]:
    """Fallback: save as compressed NPZ."""
    arr = np.array(params)
    np.savez_compressed(
        os.path.join(compress_dir, f"{param_name}.npz"), arr=arr
    )
    return {"shape": list(params.shape)}


def _decompress_npz(
    compress_dir: str,
    param_name: str,
    meta: Dict[str, Any],
) -> mx.array:
    """Decompress from NPZ."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    arr = arr.reshape(meta["shape"])
    return mx.array(arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class PngCompression:
    """Compress trained Gaussians into PNG files.

    Uses quantisation and spatial sorting to achieve significant compression
    on standard 3DGS models.  Parameters are quantised to 8-bit (scales,
    quats, opacities, sh0) or 16-bit (means) and stored as lossless PNG
    images.

    .. note::
        Higher-order SH coefficients (``shN``) are stored via NPZ rather
        than k-means clustering.  This is a simplification compared to the
        upstream PyTorch implementation which uses ``torchpq`` for k-means.

    Args:
        use_sort: Whether to sort splats by Morton code before compression.
            Improves compression ratio. Default ``True``.
        verbose: Whether to print progress information. Default ``True``.
    """

    use_sort: bool = True
    verbose: bool = True

    # Mapping from parameter name to compress/decompress functions
    _compress_fns: Dict[str, Callable] = field(default_factory=dict, repr=False, init=False)
    _decompress_fns: Dict[str, Callable] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        self._compress_fns = {
            "means": _compress_png_16bit,
            "scales": _compress_png_8bit,
            "quats": _compress_png_8bit,
            "opacities": _compress_png_8bit,
            "sh0": _compress_png_8bit,
        }
        self._decompress_fns = {
            "means": _decompress_png_16bit,
            "scales": _decompress_png_8bit,
            "quats": _decompress_png_8bit,
            "opacities": _decompress_png_8bit,
            "sh0": _decompress_png_8bit,
        }

    def compress(
        self,
        splats: Dict[str, mx.array],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Compress splats to PNG files in *output_dir*.

        The splat parameters are expected to be **pre-activation** values
        (log-scales, sigmoid-pre-activation opacities, etc.).

        Args:
            splats: Dictionary of splat arrays.  Must contain at least
                ``"means"``, ``"quats"``, ``"scales"``, ``"opacities"``.
            output_dir: Directory to write compressed files into.

        Returns:
            Metadata dictionary (also saved as ``meta.json``).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Work on a copy so we don't mutate the caller's data
        splats = {k: v for k, v in splats.items()}

        # Pre-processing: log-transform means, normalise quats
        splats["means"] = _log_transform(splats["means"])
        splats["quats"] = _normalize_quats(splats["quats"])

        # Crop to perfect square
        n_gs = splats["means"].shape[0]
        n_side = int(n_gs ** 0.5)
        if n_side * n_side != n_gs:
            splats = _crop_to_square(splats)
            n_gs = splats["means"].shape[0]
            n_side = int(n_gs ** 0.5)
            if self.verbose:
                print(
                    f"PngCompression: cropped to {n_gs} Gaussians "
                    f"({n_side}x{n_side} grid)"
                )

        # Sort for spatial locality
        if self.use_sort:
            if self.verbose:
                print("PngCompression: sorting by Morton code...")
            splats = sort_splats(splats)

        # Compress each parameter
        meta: Dict[str, Any] = {}
        for param_name, param in splats.items():
            compress_fn = self._compress_fns.get(param_name, _compress_npz)
            if self.verbose:
                print(f"PngCompression: compressing '{param_name}' {list(param.shape)}")
            meta[param_name] = compress_fn(output_dir, param_name, param, n_side)

        # Save metadata
        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"PngCompression: saved to {output_dir}")

        return meta

    def decompress(self, input_dir: str) -> Dict[str, mx.array]:
        """Load compressed splats from PNG files.

        Args:
            input_dir: Directory containing compressed files and ``meta.json``.

        Returns:
            Dictionary of splat arrays (post-activation: means are
            inverse-log-transformed).
        """
        meta_path = os.path.join(input_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        splats: Dict[str, mx.array] = {}
        for param_name, param_meta in meta.items():
            decompress_fn = self._decompress_fns.get(param_name, _decompress_npz)
            splats[param_name] = decompress_fn(input_dir, param_name, param_meta)

        # Post-processing: undo log-transform on means
        splats["means"] = _inverse_log_transform(splats["means"])
        return splats
