"""Tests for the color_correct module."""

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.color_correct import color_correct_quadratic, color_correct_affine


def _rand_img(h: int = 32, w: int = 32, c: int = 3) -> mx.array:
    """Create a random image in [0, 1]."""
    return mx.clip(mx.random.normal((h, w, c)) * 0.2 + 0.5, 0.0, 1.0)


# --------------------------------------------------------------------------
# Quadratic color correction
# --------------------------------------------------------------------------


class TestColorCorrectQuadratic:
    def test_identity(self):
        """When img == ref, output should equal input."""
        img = _rand_img()
        result = color_correct_quadratic(img, img)
        np.testing.assert_allclose(np.array(result), np.array(img), atol=1e-3)

    def test_shifted(self):
        """When ref = clip(img + 0.1), correction should approximate ref."""
        img = _rand_img()
        ref = mx.clip(img + 0.1, 0.0, 1.0)
        result = color_correct_quadratic(img, ref)
        # Should be reasonably close to ref
        diff = np.abs(np.array(result) - np.array(ref))
        assert diff.mean() < 0.05, f"Mean diff {diff.mean():.4f} too large"

    def test_shape(self):
        """Output shape must match input shape."""
        img = _rand_img(16, 24, 3)
        ref = _rand_img(16, 24, 3)
        result = color_correct_quadratic(img, ref)
        assert result.shape == img.shape

    def test_channel_mismatch_raises(self):
        """Mismatched channels should raise ValueError."""
        img = _rand_img(8, 8, 3)
        ref = mx.random.normal((8, 8, 4))
        with pytest.raises(ValueError, match="Channel mismatch"):
            color_correct_quadratic(img, ref)


# --------------------------------------------------------------------------
# Affine color correction
# --------------------------------------------------------------------------


class TestColorCorrectAffine:
    def test_identity(self):
        """When img == ref, output should equal input."""
        img = _rand_img()
        result = color_correct_affine(img, img)
        np.testing.assert_allclose(np.array(result), np.array(img), atol=1e-3)

    def test_shifted(self):
        """Affine correction of a shifted image should approximate ref."""
        img = _rand_img()
        ref = mx.clip(img + 0.1, 0.0, 1.0)
        result = color_correct_affine(img, ref)
        diff = np.abs(np.array(result) - np.array(ref))
        assert diff.mean() < 0.05, f"Mean diff {diff.mean():.4f} too large"

    def test_shape(self):
        """Output shape must match input shape."""
        img = _rand_img(10, 10, 3)
        ref = _rand_img(10, 10, 3)
        result = color_correct_affine(img, ref)
        assert result.shape == img.shape


# --------------------------------------------------------------------------
# Cross-method comparison
# --------------------------------------------------------------------------


class TestAffineVsQuadratic:
    def test_both_valid(self):
        """Both methods should produce valid outputs in [0, 1]."""
        img = _rand_img()
        ref = mx.clip(img * 0.8 + 0.1, 0.0, 1.0)

        q_result = np.array(color_correct_quadratic(img, ref))
        a_result = np.array(color_correct_affine(img, ref))

        assert q_result.min() >= -1e-6
        assert q_result.max() <= 1.0 + 1e-6
        assert a_result.min() >= -1e-6
        assert a_result.max() <= 1.0 + 1e-6
