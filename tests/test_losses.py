"""Unit tests for gsplat-mlx loss functions.

Tests cover:
- l1_loss: identity, positivity, gradient flow
- ssim / ssim_loss: identity, different-image behavior, gradient flow
- combined_loss: weighted combination
- _fspecial_gauss_1d: normalization and symmetry
"""

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.losses import (
    l1_loss,
    ssim,
    ssim_loss,
    combined_loss,
    _fspecial_gauss_1d,
    _gaussian_filter_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_image(H: int = 32, W: int = 32, C: int = 3, seed: int = 0) -> mx.array:
    """Generate a random image in [0, 1]."""
    np.random.seed(seed)
    return mx.array(np.random.rand(H, W, C).astype(np.float32))


# ---------------------------------------------------------------------------
# L1 Loss Tests
# ---------------------------------------------------------------------------


class TestL1Loss:
    def test_identical_images(self):
        """L1(x, x) should be exactly 0."""
        img = _random_image(seed=1)
        loss = l1_loss(img, img)
        mx.eval(loss)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_different(self):
        """L1(x, y) should be > 0 for x != y."""
        img1 = _random_image(seed=1)
        img2 = _random_image(seed=2)
        loss = l1_loss(img1, img2)
        mx.eval(loss)
        assert float(loss) > 0.0

    def test_known_value(self):
        """L1 between constant images should equal their absolute difference."""
        a = mx.ones((4, 4, 3)) * 0.3
        b = mx.ones((4, 4, 3)) * 0.7
        loss = l1_loss(a, b)
        mx.eval(loss)
        assert float(loss) == pytest.approx(0.4, abs=1e-6)

    def test_gradient_flows(self):
        """mx.grad should produce non-zero gradients through l1_loss."""
        target = _random_image(seed=10)

        def loss_fn(predicted):
            return l1_loss(predicted, target)

        predicted = _random_image(seed=11)
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(predicted)
        mx.eval(grad)

        # Gradient should be non-zero
        grad_np = np.array(grad)
        assert not np.allclose(grad_np, 0.0), "L1 gradients should be non-zero"
        assert not np.any(np.isnan(grad_np)), "L1 gradients should not contain NaN"
        assert not np.any(np.isinf(grad_np)), "L1 gradients should not contain Inf"

    def test_batch_dimension(self):
        """L1 should work with batch dimension [B, H, W, C]."""
        img1 = mx.ones((2, 8, 8, 3)) * 0.5
        img2 = mx.ones((2, 8, 8, 3)) * 0.8
        loss = l1_loss(img1, img2)
        mx.eval(loss)
        assert float(loss) == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Gaussian Kernel Tests
# ---------------------------------------------------------------------------


class TestGaussianKernel:
    def test_sums_to_one(self):
        """Gaussian kernel should sum to 1."""
        for size in [3, 5, 7, 11]:
            kernel = _fspecial_gauss_1d(size, sigma=1.5)
            mx.eval(kernel)
            assert float(mx.sum(kernel)) == pytest.approx(1.0, abs=1e-6), \
                f"Kernel of size {size} does not sum to 1"

    def test_symmetric(self):
        """Gaussian kernel should be symmetric."""
        kernel = _fspecial_gauss_1d(11, sigma=1.5)
        mx.eval(kernel)
        kernel_np = np.array(kernel)
        np.testing.assert_allclose(kernel_np, kernel_np[::-1], atol=1e-7)

    def test_peak_at_center(self):
        """Gaussian kernel peak should be at the center."""
        kernel = _fspecial_gauss_1d(11, sigma=1.5)
        mx.eval(kernel)
        kernel_np = np.array(kernel)
        center = len(kernel_np) // 2
        assert kernel_np[center] == kernel_np.max()

    def test_positive_values(self):
        """All kernel values should be positive."""
        kernel = _fspecial_gauss_1d(11, sigma=1.5)
        mx.eval(kernel)
        assert float(mx.min(kernel)) > 0.0


# ---------------------------------------------------------------------------
# Gaussian Filter Tests
# ---------------------------------------------------------------------------


class TestGaussianFilter:
    def test_preserves_constant_image(self):
        """Blurring a constant image should return the same constant."""
        img = mx.ones((1, 16, 16, 3)) * 0.5
        kernel = _fspecial_gauss_1d(5, sigma=1.0)
        filtered = _gaussian_filter_2d(img, kernel)
        mx.eval(filtered)
        np.testing.assert_allclose(
            np.array(filtered), 0.5, atol=1e-4,
            err_msg="Gaussian filter should preserve constant images",
        )

    def test_output_shape(self):
        """Output shape should match input shape."""
        img = mx.ones((2, 20, 20, 3))
        kernel = _fspecial_gauss_1d(7, sigma=1.5)
        filtered = _gaussian_filter_2d(img, kernel)
        mx.eval(filtered)
        assert filtered.shape == img.shape


# ---------------------------------------------------------------------------
# SSIM Tests
# ---------------------------------------------------------------------------


class TestSSIM:
    def test_identical_images(self):
        """SSIM of identical images should be ~1.0, so loss ~0.0."""
        img = _random_image(H=32, W=32, seed=1)
        ssim_val = ssim(img, img)
        mx.eval(ssim_val)
        assert float(ssim_val) == pytest.approx(1.0, abs=1e-4), \
            f"SSIM of identical images should be ~1.0, got {float(ssim_val)}"

    def test_identical_ssim_loss(self):
        """SSIM loss of identical images should be ~0.0."""
        img = _random_image(H=32, W=32, seed=1)
        loss = ssim_loss(img, img)
        mx.eval(loss)
        assert float(loss) == pytest.approx(0.0, abs=1e-4), \
            f"SSIM loss of identical images should be ~0.0, got {float(loss)}"

    def test_different_images_positive_loss(self):
        """SSIM loss for different images should be > 0."""
        img1 = _random_image(H=32, W=32, seed=1)
        img2 = _random_image(H=32, W=32, seed=2)
        loss = ssim_loss(img1, img2)
        mx.eval(loss)
        assert float(loss) > 0.01, \
            f"SSIM loss should be positive for different images, got {float(loss)}"

    def test_ssim_range(self):
        """SSIM value should be in [-1, 1]."""
        img1 = _random_image(H=32, W=32, seed=1)
        img2 = _random_image(H=32, W=32, seed=2)
        ssim_val = ssim(img1, img2)
        mx.eval(ssim_val)
        val = float(ssim_val)
        assert -1.0 <= val <= 1.0, f"SSIM out of range: {val}"

    def test_gradient_flows(self):
        """mx.grad should produce non-zero gradients through ssim_loss."""
        target = _random_image(H=20, W=20, seed=10)

        def loss_fn(predicted):
            return ssim_loss(predicted, target, window_size=5)

        predicted = _random_image(H=20, W=20, seed=11)
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(predicted)
        mx.eval(grad)

        grad_np = np.array(grad)
        assert not np.allclose(grad_np, 0.0), "SSIM gradients should be non-zero"
        assert not np.any(np.isnan(grad_np)), "SSIM gradients should not contain NaN"
        assert not np.any(np.isinf(grad_np)), "SSIM gradients should not contain Inf"

    def test_batch_dimension(self):
        """SSIM should work with batch dimension [B, H, W, C]."""
        img1 = _random_image(H=20, W=20, seed=1)
        img2 = _random_image(H=20, W=20, seed=2)
        # Stack into batch
        batch1 = mx.stack([img1, img1])
        batch2 = mx.stack([img2, img2])
        loss = ssim_loss(batch1, batch2)
        mx.eval(loss)
        assert float(loss) > 0.0

    def test_none_reduction(self):
        """SSIM with reduction='none' should return a spatial map."""
        img = _random_image(H=20, W=20, seed=1)
        ssim_map = ssim(img, img, reduction="none", window_size=5)
        mx.eval(ssim_map)
        assert ssim_map.ndim == 3  # [H, W, C]
        assert ssim_map.shape[0] == 20
        assert ssim_map.shape[1] == 20


# ---------------------------------------------------------------------------
# Combined Loss Tests
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    def test_identical_images(self):
        """Combined loss of identical images should be ~0."""
        img = _random_image(H=20, W=20, seed=1)
        loss = combined_loss(img, img)
        mx.eval(loss)
        assert float(loss) == pytest.approx(0.0, abs=1e-3)

    def test_weighted_combination(self):
        """Combined loss should equal the weighted sum of L1 and SSIM loss."""
        img1 = _random_image(H=20, W=20, seed=1)
        img2 = _random_image(H=20, W=20, seed=2)

        lambda_ssim = 0.3
        l1 = l1_loss(img1, img2)
        ssl = ssim_loss(img1, img2)
        expected = (1.0 - lambda_ssim) * l1 + lambda_ssim * ssl
        actual = combined_loss(img1, img2, lambda_ssim=lambda_ssim)

        mx.eval(expected, actual)
        assert float(actual) == pytest.approx(float(expected), abs=1e-5)

    def test_gradient_flows(self):
        """Gradients should flow through the combined loss."""
        target = _random_image(H=20, W=20, seed=10)

        def loss_fn(predicted):
            return combined_loss(predicted, target)

        predicted = _random_image(H=20, W=20, seed=11)
        grad_fn = mx.grad(loss_fn)
        grad = grad_fn(predicted)
        mx.eval(grad)

        grad_np = np.array(grad)
        assert not np.allclose(grad_np, 0.0), "Combined loss gradients should be non-zero"
        assert not np.any(np.isnan(grad_np)), "Gradients should not contain NaN"

    def test_lambda_zero(self):
        """With lambda_ssim=0, combined loss should equal L1 loss."""
        img1 = _random_image(H=20, W=20, seed=1)
        img2 = _random_image(H=20, W=20, seed=2)
        l1 = l1_loss(img1, img2)
        comb = combined_loss(img1, img2, lambda_ssim=0.0)
        mx.eval(l1, comb)
        assert float(comb) == pytest.approx(float(l1), abs=1e-6)

    def test_lambda_one(self):
        """With lambda_ssim=1, combined loss should equal SSIM loss."""
        img1 = _random_image(H=20, W=20, seed=1)
        img2 = _random_image(H=20, W=20, seed=2)
        ssl = ssim_loss(img1, img2)
        comb = combined_loss(img1, img2, lambda_ssim=1.0)
        mx.eval(ssl, comb)
        assert float(comb) == pytest.approx(float(ssl), abs=1e-5)
