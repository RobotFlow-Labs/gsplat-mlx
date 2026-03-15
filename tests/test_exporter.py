"""Tests for the exporter module (PLY / .splat export, Morton sort, log transforms)."""

import os
import struct
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.exporter import (
    export_splats,
    sh2rgb,
    sort_centers,
    encode_morton3_vec,
    log_transform,
    inverse_log_transform,
)


def _make_splats(n: int = 100, sh_degree: int = 3):
    """Create a minimal splats dict with *n* Gaussians."""
    K = (sh_degree + 1) ** 2 - 1  # number of higher-order SH bands
    return {
        "means": mx.random.normal((n, 3)),
        "scales": mx.random.normal((n, 3)),
        "quats": mx.random.normal((n, 4)),
        "opacities": mx.random.normal((n,)),
        "sh0": mx.random.normal((n, 1, 3)),
        "shN": mx.random.normal((n, K, 3)),
    }


# --------------------------------------------------------------------------
# PLY export
# --------------------------------------------------------------------------


class TestExportPLY:
    def test_export_ply_basic(self, tmp_path):
        """Export 100 Gaussians to PLY and verify the file exists with correct header."""
        splats = _make_splats(100)
        out = str(tmp_path / "test.ply")
        data = export_splats(splats, out, format="ply")

        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

        # Verify PLY header
        with open(out, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if line.strip() == b"end_header":
                    break

        header_str = header.decode("ascii")
        assert "ply" in header_str
        assert "format binary_little_endian 1.0" in header_str
        assert "element vertex" in header_str
        assert "property float x" in header_str
        assert "property float y" in header_str
        assert "property float z" in header_str
        assert "property float opacity" in header_str

    def test_export_ply_roundtrip(self, tmp_path):
        """Export then read back, verify means match within tolerance."""
        n = 50
        splats = _make_splats(n, sh_degree=0)
        out = str(tmp_path / "roundtrip.ply")
        export_splats(splats, out, format="ply")

        # Parse PLY: read header to find vertex count, then binary data
        with open(out, "rb") as f:
            num_props = 0
            num_verts = 0
            while True:
                line = f.readline().decode("ascii").strip()
                if line.startswith("element vertex"):
                    num_verts = int(line.split()[-1])
                if line.startswith("property float"):
                    num_props += 1
                if line == "end_header":
                    break

            raw = f.read()

        assert num_verts == n
        arr = np.frombuffer(raw, dtype=np.dtype(np.float32).newbyteorder("<"))
        arr = arr.reshape(num_verts, num_props)

        # First 3 columns are means
        means_read = arr[:, :3]
        means_orig = np.array(splats["means"])

        np.testing.assert_allclose(means_read, means_orig, atol=1e-5)


# --------------------------------------------------------------------------
# .splat export
# --------------------------------------------------------------------------


class TestExportSplat:
    def test_export_splat_basic(self, tmp_path):
        """Export in splat format. File size must be N * 32 bytes."""
        n = 80
        splats = _make_splats(n, sh_degree=0)
        out = str(tmp_path / "test.splat")
        export_splats(splats, out, format="splat")

        assert os.path.isfile(out)
        # Each Gaussian = 32 bytes (12 pos + 12 scale + 4 color + 4 rot)
        assert os.path.getsize(out) == n * 32


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------


class TestSH2RGB:
    def test_constant(self):
        """sh2rgb(0) should return 0.5, sh2rgb(1) should return C0 + 0.5."""
        C0 = 0.28209479177387814
        sh_zero = np.zeros((10, 3))
        rgb = sh2rgb(sh_zero)
        np.testing.assert_allclose(rgb, 0.5, atol=1e-7)

        sh_one = np.ones((10, 3))
        rgb = sh2rgb(sh_one)
        np.testing.assert_allclose(rgb, C0 + 0.5, atol=1e-7)


class TestMortonSort:
    def test_produces_permutation(self):
        """Morton sort should return a valid permutation of the input indices."""
        n = 200
        centers = np.random.randn(n, 3).astype(np.float32)
        indices = np.arange(n)
        sorted_idx = sort_centers(centers, indices)

        assert sorted_idx.shape == (n,)
        assert set(sorted_idx.tolist()) == set(range(n))

    def test_deterministic(self):
        """Same input should produce same output."""
        centers = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]], dtype=np.float32)
        indices = np.arange(3)
        s1 = sort_centers(centers, indices)
        s2 = sort_centers(centers, indices)
        np.testing.assert_array_equal(s1, s2)


class TestLogTransform:
    def test_roundtrip(self):
        """inverse_log_transform(log_transform(x)) should recover x."""
        x = mx.array([-5.0, -1.0, 0.0, 0.5, 3.0, 100.0])
        y = log_transform(x)
        x_rec = inverse_log_transform(y)
        np.testing.assert_allclose(np.array(x_rec), np.array(x), rtol=1e-4, atol=1e-5)

    def test_zero(self):
        """log_transform(0) == 0."""
        assert float(log_transform(mx.array(0.0))) == 0.0

    def test_sign_preservation(self):
        """log_transform preserves sign."""
        x = mx.array([-3.0, 2.0])
        y = log_transform(x)
        y_np = np.array(y)
        assert y_np[0] < 0
        assert y_np[1] > 0
