"""Tests for PNG compression and Morton-code sorting."""

import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.compression import PngCompression, sort_splats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_splats(N=100, seed=42):
    """Create a minimal set of splat parameters for compression tests."""
    np.random.seed(seed)
    means = mx.array(np.random.uniform(-5, 5, (N, 3)).astype(np.float32))
    quats_raw = np.random.randn(N, 4).astype(np.float32)
    quats_raw /= np.linalg.norm(quats_raw, axis=-1, keepdims=True)
    quats = mx.array(quats_raw)
    scales = mx.array(np.random.uniform(-3, 1, (N, 3)).astype(np.float32))
    opacities = mx.array(np.random.uniform(-2, 2, (N,)).astype(np.float32))
    sh0 = mx.array(np.random.randn(N, 1, 3).astype(np.float32) * 0.1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh0": sh0,
    }


# ---------------------------------------------------------------------------
# Test: Morton sort produces a valid permutation
# ---------------------------------------------------------------------------

def test_sort_splats():
    """Sorting should reorder arrays without duplicating or losing elements."""
    splats = _make_splats(N=256)
    sorted_splats = sort_splats(splats)

    # Same keys
    assert set(sorted_splats.keys()) == set(splats.keys())

    # Same shapes
    for k in splats:
        assert sorted_splats[k].shape == splats[k].shape, f"Shape mismatch for {k}"

    # Means should be a permutation: same set of values, different order
    orig = np.sort(np.array(splats["means"]).ravel())
    srtd = np.sort(np.array(sorted_splats["means"]).ravel())
    np.testing.assert_allclose(orig, srtd, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: compress then decompress produces values close to original
# ---------------------------------------------------------------------------

def test_compress_decompress_roundtrip():
    """Compress then decompress should recover values within quantisation error."""
    # Use a perfect square so no cropping happens
    N = 100  # 10x10
    splats = _make_splats(N=N)

    compressor = PngCompression(use_sort=False, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor.compress(splats, tmpdir)

        # meta.json should exist
        assert os.path.exists(os.path.join(tmpdir, "meta.json"))

        decompressed = compressor.decompress(tmpdir)

    # Check all keys present
    assert set(decompressed.keys()) == set(splats.keys())

    # Means: 16-bit quantisation -> ~1e-4 relative error
    orig_means = np.array(splats["means"])
    dec_means = np.array(decompressed["means"])
    # Allow generous tolerance due to log-transform + quantisation
    np.testing.assert_allclose(orig_means, dec_means, atol=0.1, rtol=0.05)

    # Scales: 8-bit quantisation -> ~1e-2 relative error
    orig_scales = np.array(splats["scales"])
    dec_scales = np.array(decompressed["scales"])
    np.testing.assert_allclose(orig_scales, dec_scales, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Test: compressed files are smaller than raw numpy save
# ---------------------------------------------------------------------------

def test_compression_reduces_size():
    """Compressed PNG output should be smaller than raw .npy save."""
    N = 256  # 16x16 perfect square
    splats = _make_splats(N=N)

    compressor = PngCompression(use_sort=True, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor.compress(splats, tmpdir)

        # Total compressed size
        compressed_size = sum(
            os.path.getsize(os.path.join(tmpdir, f))
            for f in os.listdir(tmpdir)
        )

    # Raw size: sum of all parameter sizes in bytes (float32 = 4 bytes)
    raw_size = sum(
        np.array(v).nbytes for v in splats.values()
    )

    # Compressed should be smaller (PNG is lossless but quantised)
    assert compressed_size < raw_size, (
        f"Compressed size ({compressed_size}) should be < raw size ({raw_size})"
    )


# ---------------------------------------------------------------------------
# Test: sort + compress roundtrip
# ---------------------------------------------------------------------------

def test_sort_compress_roundtrip():
    """Sort + compress + decompress should still produce reasonable values."""
    N = 64  # 8x8
    splats = _make_splats(N=N, seed=99)

    compressor = PngCompression(use_sort=True, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor.compress(splats, tmpdir)
        decompressed = compressor.decompress(tmpdir)

    # Since sorting changes order, we compare sorted values
    orig_means_sorted = np.sort(np.array(splats["means"]).ravel())
    dec_means_sorted = np.sort(np.array(decompressed["means"]).ravel())
    np.testing.assert_allclose(orig_means_sorted, dec_means_sorted, atol=0.15, rtol=0.1)


# ---------------------------------------------------------------------------
# Test: non-square N gets cropped gracefully
# ---------------------------------------------------------------------------

def test_non_square_crop():
    """Non-square N should be cropped to nearest perfect square."""
    N = 105  # not a perfect square; nearest is 100 (10x10)
    splats = _make_splats(N=N)

    compressor = PngCompression(use_sort=False, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor.compress(splats, tmpdir)
        decompressed = compressor.decompress(tmpdir)

    # Should have 100 Gaussians
    assert decompressed["means"].shape[0] == 100
