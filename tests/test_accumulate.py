"""Tests for gsplat_mlx.core.accumulate — alpha compositing with explicit intersections.

Covers:
- render_weight_from_alpha: segment-wise exclusive cumprod
- accumulate_along_rays: weighted scatter-add
- accumulate: full compositing pipeline
"""

import mlx.core as mx
import numpy as np
import pytest


# ===================================================================
# render_weight_from_alpha tests
# ===================================================================


class TestRenderWeightFromAlpha:
    """Tests for the segment-wise exclusive cumprod weight computation."""

    def test_single_ray_basic(self):
        """One ray with 3 intersections, verify exact weight values."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.5, 0.3, 0.8])
        ray_indices = mx.array([0, 0, 0], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=1)
        mx.eval(weights, trans)

        # transmittance: [1.0, 0.5, 0.35]
        # weights: [0.5, 0.15, 0.28]
        expected_trans = np.array([1.0, 0.5, 0.35], dtype=np.float32)
        expected_weights = np.array([0.5, 0.15, 0.28], dtype=np.float32)

        np.testing.assert_allclose(np.array(trans), expected_trans, atol=1e-5)
        np.testing.assert_allclose(np.array(weights), expected_weights, atol=1e-5)
        assert np.array(weights).sum() <= 1.0 + 1e-6

    def test_opaque_first(self):
        """First alpha is MAX_ALPHA (~opaque), subsequent weights near zero."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.99, 0.5, 0.5])
        ray_indices = mx.array([0, 0, 0], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=1)
        mx.eval(weights, trans)

        w = np.array(weights)
        t = np.array(trans)

        # First element: T=1, weight=0.99
        np.testing.assert_allclose(t[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(w[0], 0.99, atol=1e-5)

        # Second: T = 1 * (1-0.99) = 0.01, weight = 0.01 * 0.5 = 0.005
        np.testing.assert_allclose(t[1], 0.01, atol=1e-5)
        np.testing.assert_allclose(w[1], 0.005, atol=1e-5)

        # Third: T = 0.01 * 0.5 = 0.005, weight = 0.005 * 0.5 = 0.0025
        np.testing.assert_allclose(t[2], 0.005, atol=1e-5)
        np.testing.assert_allclose(w[2], 0.0025, atol=1e-5)

    def test_all_transparent(self):
        """All alphas = 0.1, weights decrease exponentially."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.1, 0.1, 0.1])
        ray_indices = mx.array([0, 0, 0], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=1)
        mx.eval(weights, trans)

        # T = [1.0, 0.9, 0.81], w = [0.1, 0.09, 0.081]
        expected_weights = np.array([0.1, 0.09, 0.081], dtype=np.float32)
        np.testing.assert_allclose(np.array(weights), expected_weights, atol=1e-5)

    def test_multi_ray(self):
        """Two rays with different alphas, verify independence."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        # Ray 0: [0.5, 0.3]  Ray 1: [0.8, 0.2]
        alphas = mx.array([0.5, 0.3, 0.8, 0.2])
        ray_indices = mx.array([0, 0, 1, 1], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=2)
        mx.eval(weights, trans)

        # Ray 0: T=[1.0, 0.5], w=[0.5, 0.15]
        # Ray 1: T=[1.0, 0.2], w=[0.8, 0.04]
        expected_trans = np.array([1.0, 0.5, 1.0, 0.2], dtype=np.float32)
        expected_weights = np.array([0.5, 0.15, 0.8, 0.04], dtype=np.float32)

        np.testing.assert_allclose(np.array(trans), expected_trans, atol=1e-5)
        np.testing.assert_allclose(np.array(weights), expected_weights, atol=1e-5)

    def test_single_intersection(self):
        """One intersection per ray: weight = alpha, T = 1."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.7])
        ray_indices = mx.array([0], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=1)
        mx.eval(weights, trans)

        np.testing.assert_allclose(np.array(trans), [1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(weights), [0.7], atol=1e-6)

    def test_empty(self):
        """M = 0 returns empty arrays."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([], dtype=mx.float32)
        ray_indices = mx.array([], dtype=mx.int32)

        weights, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=0)
        mx.eval(weights, trans)

        assert weights.shape == (0,)
        assert trans.shape == (0,)

    def test_weight_sum_leq_one(self):
        """For random alphas, per-ray weight sums should be <= 1."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        np.random.seed(123)
        n_rays = 10
        counts = np.random.randint(1, 8, size=n_rays)
        alphas_list = []
        ray_indices_list = []
        for r in range(n_rays):
            alphas_list.append(np.random.uniform(0.01, 0.9, counts[r]).astype(np.float32))
            ray_indices_list.append(np.full(counts[r], r, dtype=np.int32))

        alphas = mx.array(np.concatenate(alphas_list))
        ray_indices = mx.array(np.concatenate(ray_indices_list))

        weights, _ = render_weight_from_alpha(alphas, ray_indices, n_rays=n_rays)
        mx.eval(weights)

        w_np = np.array(weights)
        ri_np = np.array(ray_indices)
        for r in range(n_rays):
            mask = ri_np == r
            assert w_np[mask].sum() <= 1.0 + 1e-5, (
                f"Ray {r}: sum(weights) = {w_np[mask].sum()}"
            )

    def test_transmittance_monotonic(self):
        """Transmittance should be non-increasing within each ray."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.3, 0.5, 0.2, 0.7, 0.1])
        ray_indices = mx.array([0, 0, 0, 1, 1], dtype=mx.int32)

        _, trans = render_weight_from_alpha(alphas, ray_indices, n_rays=2)
        mx.eval(trans)
        t = np.array(trans)

        # Ray 0: indices 0,1,2
        assert t[0] >= t[1] >= t[2]
        # Ray 1: indices 3,4
        assert t[3] >= t[4]


# ===================================================================
# accumulate_along_rays tests
# ===================================================================


class TestAccumulateAlongRays:
    """Tests for weighted scatter-add."""

    def test_scatter_basic(self):
        """Simple weighted scatter-add, 1 channel."""
        from gsplat_mlx.core.accumulate import accumulate_along_rays

        weights = mx.array([0.5, 0.3, 0.8])
        values = mx.array([[1.0], [2.0], [3.0]])  # [3, 1]
        ray_indices = mx.array([0, 0, 1], dtype=mx.int32)

        result = accumulate_along_rays(weights, values, ray_indices, n_rays=2)
        mx.eval(result)

        # ray 0: 0.5*1 + 0.3*2 = 1.1
        # ray 1: 0.8*3 = 2.4
        expected = np.array([[1.1], [2.4]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_scatter_multi_channel(self):
        """Scatter-add with C = 3 channels."""
        from gsplat_mlx.core.accumulate import accumulate_along_rays

        weights = mx.array([0.6, 0.4])
        values = mx.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])  # [2, 3]
        ray_indices = mx.array([0, 0], dtype=mx.int32)

        result = accumulate_along_rays(weights, values, ray_indices, n_rays=1)
        mx.eval(result)

        # ray 0: 0.6*[1,0,0.5] + 0.4*[0,1,0.5] = [0.6, 0.4, 0.5]
        expected = np.array([[0.6, 0.4, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_no_values(self):
        """values=None -> accumulate weights only, shape [n_rays, 1]."""
        from gsplat_mlx.core.accumulate import accumulate_along_rays

        weights = mx.array([0.3, 0.5, 0.2])
        ray_indices = mx.array([0, 1, 1], dtype=mx.int32)

        result = accumulate_along_rays(weights, None, ray_indices, n_rays=2)
        mx.eval(result)

        assert result.shape == (2, 1)
        expected = np.array([[0.3], [0.7]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)

    def test_empty(self):
        """M = 0 returns zeros."""
        from gsplat_mlx.core.accumulate import accumulate_along_rays

        weights = mx.array([], dtype=mx.float32)
        ray_indices = mx.array([], dtype=mx.int32)

        result = accumulate_along_rays(
            weights, mx.zeros((0, 3)), ray_indices, n_rays=5
        )
        mx.eval(result)

        assert result.shape == (5, 3)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-8)

    def test_no_overlap(self):
        """Each intersection on a unique ray -> output = weight * value."""
        from gsplat_mlx.core.accumulate import accumulate_along_rays

        weights = mx.array([0.5, 0.8, 0.3])
        values = mx.array([[1.0], [2.0], [3.0]])
        ray_indices = mx.array([0, 1, 2], dtype=mx.int32)

        result = accumulate_along_rays(weights, values, ray_indices, n_rays=3)
        mx.eval(result)

        expected = np.array([[0.5], [1.6], [0.9]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5)


# ===================================================================
# accumulate tests (integration)
# ===================================================================


class TestAccumulate:
    """Tests for the full compositing pipeline."""

    def test_single_gaussian_single_pixel(self):
        """One Gaussian centred on a pixel, one intersection."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        # Gaussian at pixel (2,2) centre = (2.5, 2.5)
        means2d = mx.array([[[2.5, 2.5]]])  # [1, 1, 2]
        conics = mx.array([[[1.0, 0.0, 1.0]]])  # [1, 1, 3]
        opacities = mx.array([[0.8]])  # [1, 1]
        colors = mx.array([[[1.0, 0.0, 0.0]]])  # [1, 1, 3] red

        gaussian_ids = mx.array([0], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2], dtype=mx.int32)  # pixel (2,2)
        image_ids = mx.array([0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        assert renders.shape == (1, H, W, C)
        assert alphas.shape == (1, H, W, 1)

        # delta=(0,0), sigma=0, alpha=0.8
        # render = 0.8 * [1,0,0] = [0.8, 0, 0]
        pixel_color = np.array(renders[0, 2, 2])
        np.testing.assert_allclose(pixel_color, [0.8, 0.0, 0.0], atol=1e-5)

        pixel_alpha = float(np.array(alphas[0, 2, 2, 0]))
        np.testing.assert_allclose(pixel_alpha, 0.8, atol=1e-5)

        # All other pixels should be 0
        renders_np = np.array(renders[0])
        renders_np[2, 2] = 0
        np.testing.assert_allclose(renders_np, 0.0, atol=1e-8)

    def test_two_gaussians_front_to_back(self):
        """Two Gaussians at same pixel, front-to-back compositing."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        means2d = mx.array([[[2.5, 2.5], [2.5, 2.5]]])  # [1, 2, 2]
        conics = mx.broadcast_to(
            mx.array([[[1.0, 0.0, 1.0]]]), (1, 2, 3)
        )
        opacities = mx.array([[0.5, 0.5]])  # [1, 2]
        colors = mx.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # red, green

        # Two intersections at pixel (2,2), front Gaussian first
        gaussian_ids = mx.array([0, 1], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2, 2 * W + 2], dtype=mx.int32)
        image_ids = mx.array([0, 0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        # delta=(0,0), sigma=0, alpha_0=0.5, alpha_1=0.5
        # w0=1.0*0.5=0.5, w1=0.5*0.5=0.25
        # render = 0.5*[1,0,0] + 0.25*[0,1,0] = [0.5, 0.25, 0]
        # alpha = 0.5 + 0.25 = 0.75
        pixel_color = np.array(renders[0, 2, 2])
        np.testing.assert_allclose(pixel_color, [0.5, 0.25, 0.0], atol=1e-4)
        np.testing.assert_allclose(
            float(np.array(alphas[0, 2, 2, 0])), 0.75, atol=1e-4
        )

    def test_full_occlusion(self):
        """Front Gaussian with alpha ~ MAX_ALPHA occludes the back one."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        means2d = mx.array([[[2.5, 2.5], [2.5, 2.5]]])
        conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, 2, 3))
        # Front opacity very high so alpha -> MAX_ALPHA
        opacities = mx.array([[10.0, 0.5]])  # [1, 2]
        colors = mx.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        gaussian_ids = mx.array([0, 1], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2, 2 * W + 2], dtype=mx.int32)
        image_ids = mx.array([0, 0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        # Front alpha clamped to MAX_ALPHA = 0.99
        # Back weight = (1-0.99)*0.5 = 0.005
        pixel_color = np.array(renders[0, 2, 2])
        # Red dominates, green negligible
        assert pixel_color[0] > 0.98
        assert pixel_color[1] < 0.01

    def test_multiple_pixels(self):
        """One Gaussian visible from multiple pixels with varying distance."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 1
        means2d = mx.array([[[2.5, 2.5]]])  # [1, 1, 2]
        conics = mx.array([[[1.0, 0.0, 1.0]]])  # circular
        opacities = mx.array([[1.0]])
        colors = mx.array([[[1.0]]])

        # Pixels at (2,2) and (1,1)
        gaussian_ids = mx.array([0, 0], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2, 1 * W + 1], dtype=mx.int32)
        image_ids = mx.array([0, 0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        # Pixel (2,2): delta=(0,0), sigma=0, alpha=min(1.0, MAX_ALPHA)
        # Pixel (1,1): delta=(-1,-1), sigma=0.5*(1*1+1*1)=1, alpha=min(e^-1, MAX_ALPHA)
        a22 = float(np.array(alphas[0, 2, 2, 0]))
        a11 = float(np.array(alphas[0, 1, 1, 0]))

        assert a22 > a11, "Centre pixel should have higher alpha"
        np.testing.assert_allclose(a22, 0.99, atol=1e-4)  # clamped to MAX_ALPHA
        np.testing.assert_allclose(a11, np.exp(-1.0), atol=1e-4)

    def test_multi_channel(self):
        """Channels > 3 (feature rendering)."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 2, 2, 16
        means2d = mx.array([[[0.5, 0.5]]])
        conics = mx.array([[[1.0, 0.0, 1.0]]])
        opacities = mx.array([[0.5]])
        colors = mx.ones((1, 1, C))

        gaussian_ids = mx.array([0], dtype=mx.int32)
        pixel_ids = mx.array([0], dtype=mx.int32)  # pixel (0,0)
        image_ids = mx.array([0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        assert renders.shape == (1, H, W, C)
        # All 16 channels should have same value at pixel (0,0)
        pixel_vals = np.array(renders[0, 0, 0])
        np.testing.assert_allclose(pixel_vals, pixel_vals[0], atol=1e-6)

    def test_output_shape(self):
        """Verify output shapes for single-image input."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 8, 8, 3
        N = 5
        means2d = mx.zeros((1, N, 2))
        conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, N, 3))
        opacities = mx.ones((1, N))
        colors = mx.ones((1, N, C))

        gaussian_ids = mx.array([0], dtype=mx.int32)
        pixel_ids = mx.array([0], dtype=mx.int32)
        image_ids = mx.array([0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        assert renders.shape == (1, H, W, C)
        assert alphas.shape == (1, H, W, 1)

    def test_empty_intersections(self):
        """No intersections -> all-zero output."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        means2d = mx.zeros((1, 2, 2))
        conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, 2, 3))
        opacities = mx.ones((1, 2))
        colors = mx.ones((1, 2, C))

        gaussian_ids = mx.array([], dtype=mx.int32)
        pixel_ids = mx.array([], dtype=mx.int32)
        image_ids = mx.array([], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        assert renders.shape == (1, H, W, C)
        np.testing.assert_allclose(np.array(renders), 0.0, atol=1e-8)
        np.testing.assert_allclose(np.array(alphas), 0.0, atol=1e-8)

    def test_alpha_sum_leq_one(self):
        """All output alpha values should be in [0, 1]."""
        from gsplat_mlx.core.accumulate import accumulate

        np.random.seed(99)
        H, W, C = 4, 4, 3
        N = 10
        means2d_np = np.random.uniform(0.5, 3.5, (1, N, 2)).astype(np.float32)
        conics_np = np.zeros((1, N, 3), dtype=np.float32)
        conics_np[..., 0] = 1.0
        conics_np[..., 2] = 1.0
        opacities_np = np.random.uniform(0.1, 0.9, (1, N)).astype(np.float32)
        colors_np = np.random.randn(1, N, C).astype(np.float32)

        M = 30
        g_ids = np.random.randint(0, N, M).astype(np.int32)
        p_ids = np.random.randint(0, H * W, M).astype(np.int32)
        i_ids = np.zeros(M, dtype=np.int32)

        # Sort by ray index
        ray_idx = i_ids * H * W + p_ids
        order = np.argsort(ray_idx, kind="stable")
        g_ids, p_ids, i_ids = g_ids[order], p_ids[order], i_ids[order]

        renders, alphas = accumulate(
            mx.array(means2d_np), mx.array(conics_np),
            mx.array(opacities_np), mx.array(colors_np),
            mx.array(g_ids), mx.array(p_ids), mx.array(i_ids),
            W, H,
        )
        mx.eval(alphas)

        a = np.array(alphas)
        assert np.all(a >= -1e-6), f"Negative alpha: {a.min()}"
        assert np.all(a <= 1.0 + 1e-5), f"Alpha > 1: {a.max()}"

    def test_multi_gaussian_compositing(self):
        """Three sorted Gaussians, verify front-to-back compositing at one pixel."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        # All 3 Gaussians centred on pixel (1,1)
        means2d = mx.array([[[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]]])  # [1,3,2]
        conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (1, 3, 3))
        opacities = mx.array([[0.4, 0.4, 0.4]])  # [1,3]
        colors = mx.array([[[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]])  # R, G, B

        gaussian_ids = mx.array([0, 1, 2], dtype=mx.int32)
        pixel_ids = mx.array([1 * W + 1] * 3, dtype=mx.int32)
        image_ids = mx.array([0, 0, 0], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        # sigma=0 for all, alpha=0.4 each
        # w0 = 1.0 * 0.4 = 0.4
        # w1 = 0.6 * 0.4 = 0.24
        # w2 = 0.36 * 0.4 = 0.144
        # render = 0.4*R + 0.24*G + 0.144*B = [0.4, 0.24, 0.144]
        pixel_color = np.array(renders[0, 1, 1])
        np.testing.assert_allclose(pixel_color, [0.4, 0.24, 0.144], atol=1e-4)

        # alpha = 0.4 + 0.24 + 0.144 = 0.784
        np.testing.assert_allclose(
            float(np.array(alphas[0, 1, 1, 0])), 0.784, atol=1e-4
        )

    def test_batch_images(self):
        """I=2 images, verify independent rendering."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 2, 2, 1
        N = 1
        # Image 0 has Gaussian at (0.5, 0.5), Image 1 at (1.5, 1.5)
        means2d = mx.array([
            [[0.5, 0.5]],
            [[1.5, 1.5]],
        ])  # [2, 1, 2]
        conics = mx.broadcast_to(mx.array([[[1.0, 0.0, 1.0]]]), (2, 1, 3))
        opacities = mx.array([[0.5], [0.5]])  # [2, 1]
        colors = mx.array([[[1.0]], [[2.0]]])  # [2, 1, 1]

        # Image 0: G0 at pixel (0,0); Image 1: G0 at pixel (1,1)
        gaussian_ids = mx.array([0, 0], dtype=mx.int32)
        pixel_ids = mx.array([0, 1 * W + 1], dtype=mx.int32)
        image_ids = mx.array([0, 1], dtype=mx.int32)

        renders, alphas = accumulate(
            means2d, conics, opacities, colors,
            gaussian_ids, pixel_ids, image_ids, W, H,
        )
        mx.eval(renders, alphas)

        assert renders.shape == (2, H, W, C)

        # Image 0, pixel (0,0): alpha=0.5, render = 0.5 * 1.0 = 0.5
        np.testing.assert_allclose(
            float(np.array(renders[0, 0, 0, 0])), 0.5, atol=1e-4
        )
        # Image 1, pixel (1,1): alpha=0.5, render = 0.5 * 2.0 = 1.0
        np.testing.assert_allclose(
            float(np.array(renders[1, 1, 1, 0])), 1.0, atol=1e-4
        )


# ===================================================================
# Gradient tests
# ===================================================================


class TestGradients:
    """Gradient / autodiff tests for accumulate functions."""

    def test_gradient_render_weight_alpha(self):
        """Verify mx.grad flows through render_weight_from_alpha."""
        from gsplat_mlx.core.accumulate import render_weight_from_alpha

        alphas = mx.array([0.5, 0.3, 0.8])
        ray_indices = mx.array([0, 0, 0], dtype=mx.int32)

        def loss_fn(a):
            weights, _ = render_weight_from_alpha(a, ray_indices, n_rays=1)
            return mx.sum(weights)

        grad_fn = mx.grad(loss_fn)
        g = grad_fn(alphas)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (3,)
        # Gradients should be non-zero: changing alpha affects weights
        assert np.any(np.abs(g_np) > 1e-6), (
            f"Expected non-zero gradients, got {g_np}"
        )

        # Verify via finite differences
        eps = 1e-4
        alphas_np = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        g_numerical = np.zeros(3, dtype=np.float32)
        for i in range(3):
            a_plus = alphas_np.copy()
            a_minus = alphas_np.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            w_plus, _ = render_weight_from_alpha(
                mx.array(a_plus), ray_indices, n_rays=1
            )
            w_minus, _ = render_weight_from_alpha(
                mx.array(a_minus), ray_indices, n_rays=1
            )
            mx.eval(w_plus, w_minus)
            g_numerical[i] = (float(mx.sum(w_plus)) - float(mx.sum(w_minus))) / (
                2 * eps
            )

        np.testing.assert_allclose(g_np, g_numerical, atol=1e-3, rtol=1e-2)

    def test_gradient_accumulate_colors(self):
        """Gradient of accumulated colors w.r.t. input colors."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        means2d = mx.array([[[2.5, 2.5]]])
        conics = mx.array([[[1.0, 0.0, 1.0]]])
        opacities = mx.array([[0.8]])

        gaussian_ids = mx.array([0], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2], dtype=mx.int32)
        image_ids = mx.array([0], dtype=mx.int32)

        def loss_fn(colors):
            renders, _ = accumulate(
                means2d, conics, opacities, colors,
                gaussian_ids, pixel_ids, image_ids, W, H,
            )
            return mx.sum(renders)

        colors = mx.array([[[1.0, 0.0, 0.0]]])
        grad_fn = mx.grad(loss_fn)
        g = grad_fn(colors)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (1, 1, 3)
        # Gradient w.r.t. colors should be non-zero (the weight applied to colors)
        assert np.any(np.abs(g_np) > 1e-6), (
            f"Expected non-zero gradients for colors, got {g_np}"
        )

    def test_gradient_accumulate_opacities(self):
        """Gradient w.r.t. opacities flows through accumulate."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        means2d = mx.array([[[2.5, 2.5]]])
        conics = mx.array([[[1.0, 0.0, 1.0]]])
        colors = mx.array([[[1.0, 0.5, 0.25]]])

        gaussian_ids = mx.array([0], dtype=mx.int32)
        pixel_ids = mx.array([2 * W + 2], dtype=mx.int32)
        image_ids = mx.array([0], dtype=mx.int32)

        def loss_fn(opac):
            renders, _ = accumulate(
                means2d, conics, opac, colors,
                gaussian_ids, pixel_ids, image_ids, W, H,
            )
            return mx.sum(renders)

        opacities = mx.array([[0.8]])
        grad_fn = mx.grad(loss_fn)
        g = grad_fn(opacities)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (1, 1)
        assert np.any(np.abs(g_np) > 1e-6), (
            f"Expected non-zero gradients for opacities, got {g_np}"
        )

    def test_gradient_accumulate_means2d(self):
        """Gradient w.r.t. means2d positions flows through accumulate."""
        from gsplat_mlx.core.accumulate import accumulate

        H, W, C = 4, 4, 3
        conics = mx.array([[[1.0, 0.0, 1.0]]])
        opacities = mx.array([[0.8]])
        colors = mx.array([[[1.0, 0.5, 0.25]]])

        gaussian_ids = mx.array([0], dtype=mx.int32)
        # Use an off-center pixel so sigma != 0 and gradient is non-zero
        pixel_ids = mx.array([1 * W + 1], dtype=mx.int32)
        image_ids = mx.array([0], dtype=mx.int32)

        def loss_fn(m2d):
            renders, _ = accumulate(
                m2d, conics, opacities, colors,
                gaussian_ids, pixel_ids, image_ids, W, H,
            )
            return mx.sum(renders)

        means2d = mx.array([[[2.5, 2.5]]])
        grad_fn = mx.grad(loss_fn)
        g = grad_fn(means2d)
        mx.eval(g)

        g_np = np.array(g)
        assert g_np.shape == (1, 1, 2)
        assert np.any(np.abs(g_np) > 1e-6), (
            f"Expected non-zero gradients for means2d, got {g_np}"
        )
