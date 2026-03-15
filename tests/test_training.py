"""Training-related tests for gsplat-mlx.

Tests verify:
- Gradient flow through all parameter types
- No NaN/Inf in gradients
- Loss is a scalar
- Parameter updates via optimizer

NOTE: The Tier-1 rasterizer (rasterize_to_pixels) is a NumPy reference
implementation and is NOT differentiable. These tests use a differentiable
proxy forward pass (simple 2D Gaussian rendering) to validate the training
pipeline components. Full end-to-end differentiable rendering requires
the Tier-2 pure-MLX rasterizer.

The tests here validate:
1. Loss functions accept and differentiate through image-shaped tensors
2. The SelectiveAdam optimizer correctly updates parameters
3. A minimal differentiable Gaussian-to-image pipeline produces valid gradients
"""

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.losses import l1_loss, ssim_loss, combined_loss
from gsplat_mlx.scenes import (
    create_solid_color_scene,
    create_gradient_scene,
    create_checkerboard_scene,
)
from gsplat_mlx.optimizers.selective_adam import SelectiveAdam


# ---------------------------------------------------------------------------
# Differentiable proxy renderer
# ---------------------------------------------------------------------------


def _simple_differentiable_render(
    means: mx.array,      # [N, 2] -- 2D positions
    colors: mx.array,     # [N, 3] -- RGB colors
    opacities: mx.array,  # [N] -- opacity (pre-sigmoid)
    scales: mx.array,     # [N] -- scale (log-space)
    width: int,
    height: int,
) -> mx.array:
    """Simple differentiable 2D Gaussian splatting (no projection, no SH).

    Each Gaussian is a 2D isotropic blob. This is a minimal differentiable
    forward pass for testing gradient flow, NOT a full 3DGS renderer.

    Returns:
        Image [H, W, 3] in [0, 1].
    """
    N = means.shape[0]

    # Create pixel grid [H, W, 2]
    ys = mx.arange(height, dtype=mx.float32) + 0.5
    xs = mx.arange(width, dtype=mx.float32) + 0.5
    # grid_y: [H, W], grid_x: [H, W]
    grid_y = mx.broadcast_to(ys.reshape(height, 1), (height, width))
    grid_x = mx.broadcast_to(xs.reshape(1, width), (height, width))
    pixel_coords = mx.stack([grid_x, grid_y], axis=-1)  # [H, W, 2]

    # Compute per-Gaussian contribution
    # means: [N, 2] -> [1, 1, N, 2]
    means_expanded = means.reshape(1, 1, N, 2)
    # pixel_coords: [H, W, 2] -> [H, W, 1, 2]
    coords_expanded = pixel_coords.reshape(height, width, 1, 2)

    # Squared distance: [H, W, N]
    diff = coords_expanded - means_expanded  # [H, W, N, 2]
    sq_dist = mx.sum(diff * diff, axis=-1)  # [H, W, N]

    # Gaussian weight: exp(-sq_dist / (2 * scale^2))
    actual_scales = mx.exp(scales)  # [N]
    variance = actual_scales * actual_scales  # [N]
    weights = mx.exp(-sq_dist / (2.0 * variance.reshape(1, 1, N) + 1e-8))  # [H, W, N]

    # Apply opacity via sigmoid
    alpha = mx.sigmoid(opacities)  # [N]
    weights = weights * alpha.reshape(1, 1, N)  # [H, W, N]

    # Weighted sum of colors
    # colors: [N, 3] -> [1, 1, N, 3]
    colors_expanded = colors.reshape(1, 1, N, 3)
    # weights: [H, W, N, 1]
    weighted_colors = weights.reshape(height, width, N, 1) * colors_expanded  # [H, W, N, 3]

    # Sum over Gaussians and normalize
    total_weight = mx.sum(weights, axis=-1, keepdims=True)  # [H, W, 1]
    color_sum = mx.sum(weighted_colors, axis=2)  # [H, W, 3]

    # Avoid division by zero
    rendered = color_sum / (total_weight + 1e-8)  # [H, W, 3]

    # Clamp to [0, 1]
    rendered = mx.clip(rendered, 0.0, 1.0)

    return rendered


# ---------------------------------------------------------------------------
# Test: Gradient flow through all parameter types
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Test that gradients flow to all parameter types in a differentiable path."""

    def _make_params(self, N: int = 20, width: int = 16, height: int = 16, seed: int = 42):
        """Create test parameters for the proxy renderer."""
        np.random.seed(seed)
        means = mx.array(np.random.uniform(2, width - 2, (N, 2)).astype(np.float32))
        colors = mx.array(np.random.uniform(0.1, 0.9, (N, 3)).astype(np.float32))
        opacities = mx.array(np.random.uniform(-1, 1, (N,)).astype(np.float32))
        scales = mx.array(np.random.uniform(0.5, 2.0, (N,)).astype(np.float32))
        target = mx.array(np.random.uniform(0.0, 1.0, (height, width, 3)).astype(np.float32))
        return means, colors, opacities, scales, target, width, height

    def test_gradient_flow_all_params(self):
        """One forward+backward step should give non-zero gradients for all params."""
        means, colors, opacities, scales, target, W, H = self._make_params()

        def loss_fn(means_, colors_, opacities_, scales_):
            rendered = _simple_differentiable_render(means_, colors_, opacities_, scales_, W, H)
            return l1_loss(rendered, target)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3))
        grads = grad_fn(means, colors, opacities, scales)
        mx.eval(*grads)

        param_names = ["means", "colors", "opacities", "scales"]
        for name, grad in zip(param_names, grads):
            grad_np = np.array(grad)
            assert not np.allclose(grad_np, 0.0), \
                f"Gradient for '{name}' is all zeros -- gradient does not flow"

    def test_no_nan_gradients(self):
        """No NaN or Inf should appear in any gradient after one step."""
        means, colors, opacities, scales, target, W, H = self._make_params()

        def loss_fn(means_, colors_, opacities_, scales_):
            rendered = _simple_differentiable_render(means_, colors_, opacities_, scales_, W, H)
            return combined_loss(rendered, target)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3))
        grads = grad_fn(means, colors, opacities, scales)
        mx.eval(*grads)

        param_names = ["means", "colors", "opacities", "scales"]
        for name, grad in zip(param_names, grads):
            grad_np = np.array(grad)
            assert not np.any(np.isnan(grad_np)), \
                f"NaN found in gradient for '{name}'"
            assert not np.any(np.isinf(grad_np)), \
                f"Inf found in gradient for '{name}'"

    def test_loss_is_scalar(self):
        """Loss should be a scalar (0-d) mx.array."""
        means, colors, opacities, scales, target, W, H = self._make_params()
        rendered = _simple_differentiable_render(means, colors, opacities, scales, W, H)

        loss_l1 = l1_loss(rendered, target)
        loss_ssim = ssim_loss(rendered, target, window_size=5)
        loss_comb = combined_loss(rendered, target)

        mx.eval(loss_l1, loss_ssim, loss_comb)

        assert loss_l1.ndim == 0, f"L1 loss should be scalar, got ndim={loss_l1.ndim}"
        assert loss_ssim.ndim == 0, f"SSIM loss should be scalar, got ndim={loss_ssim.ndim}"
        assert loss_comb.ndim == 0, f"Combined loss should be scalar, got ndim={loss_comb.ndim}"


# ---------------------------------------------------------------------------
# Test: Optimizer integration
# ---------------------------------------------------------------------------


class TestOptimizerIntegration:
    """Test that SelectiveAdam updates parameters after a training step."""

    def test_single_step_updates_params(self):
        """After optimizer.step, params should differ from initial values."""
        N = 20
        W, H = 16, 16
        np.random.seed(42)

        params = {
            "means": mx.array(np.random.uniform(2, W - 2, (N, 2)).astype(np.float32)),
            "colors": mx.array(np.random.uniform(0.1, 0.9, (N, 3)).astype(np.float32)),
            "opacities": mx.array(np.random.uniform(-1, 1, (N,)).astype(np.float32)),
            "scales": mx.array(np.random.uniform(0.5, 2.0, (N,)).astype(np.float32)),
        }
        target = mx.array(np.random.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32))

        # Save initial values
        initial = {k: np.array(v) for k, v in params.items()}
        mx.eval(*params.values())

        # Compute gradients
        def loss_fn(means_, colors_, opacities_, scales_):
            rendered = _simple_differentiable_render(means_, colors_, opacities_, scales_, W, H)
            return l1_loss(rendered, target)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3))
        grads_tuple = grad_fn(params["means"], params["colors"],
                              params["opacities"], params["scales"])
        mx.eval(*grads_tuple)

        grads = {
            "means": grads_tuple[0],
            "colors": grads_tuple[1],
            "opacities": grads_tuple[2],
            "scales": grads_tuple[3],
        }

        # Run optimizer step
        optimizer = SelectiveAdam(lr=1e-2)
        visibility = mx.ones(N, dtype=mx.bool_)
        updated = optimizer.step(params, grads, visibility)
        mx.eval(*updated.values())

        # Check that at least some parameters changed
        any_changed = False
        for name in params:
            if not np.allclose(np.array(updated[name]), initial[name], atol=1e-8):
                any_changed = True
        assert any_changed, "Optimizer should update at least some parameters"

    def test_multiple_steps_reduce_loss(self):
        """Multiple optimizer steps on colors only should reduce loss.

        We freeze positions/scales/opacities and only optimize colors,
        which is the most straightforward optimization target.
        """
        N = 50
        W, H = 12, 12
        np.random.seed(123)

        means = mx.array(np.random.uniform(1, W - 1, (N, 2)).astype(np.float32))
        # Start colors far from target but not at boundary
        colors = mx.array(np.full((N, 3), 0.2, dtype=np.float32))
        opacities = mx.array(np.ones((N,), dtype=np.float32) * 2.0)  # high opacity
        scales = mx.array(np.ones((N,), dtype=np.float32) * 1.0)

        # Target: bright image
        target = mx.ones((H, W, 3)) * 0.8

        optimizer = SelectiveAdam(lr=5e-2)
        visibility = mx.ones(N, dtype=mx.bool_)

        losses = []
        for step in range(20):
            def loss_fn(c):
                rendered = _simple_differentiable_render(means, c, opacities, scales, W, H)
                return l1_loss(rendered, target)

            loss_val = loss_fn(colors)
            mx.eval(loss_val)
            losses.append(float(loss_val))

            grad_fn = mx.grad(loss_fn)
            g_colors = grad_fn(colors)
            mx.eval(g_colors)

            params = {"colors": colors}
            grads = {"colors": g_colors}

            updated = optimizer.step(params, grads, visibility)
            mx.eval(*updated.values())

            colors = updated["colors"]

        # Loss should decrease overall (last < first)
        assert losses[-1] < losses[0], \
            f"Loss should decrease over training: first={losses[0]:.4f}, last={losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# Test: Scene generators
# ---------------------------------------------------------------------------


class TestScenes:
    """Test that scene generators produce valid outputs."""

    def test_solid_color_scene(self):
        scene = create_solid_color_scene(N=50, width=16, height=16)
        assert scene["means"].shape == (50, 3)
        assert scene["quats"].shape == (50, 4)
        assert scene["scales"].shape == (50, 3)
        assert scene["opacities"].shape == (50,)
        assert scene["sh_coeffs"].shape == (50, 1, 3)
        assert scene["target_image"].shape == (16, 16, 3)
        assert scene["viewmat"].shape == (1, 4, 4)
        assert scene["K"].shape == (1, 3, 3)
        assert scene["width"] == 16
        assert scene["height"] == 16

    def test_gradient_scene(self):
        scene = create_gradient_scene(N=50, width=16, height=16)
        target = scene["target_image"]
        mx.eval(target)
        target_np = np.array(target)
        # Left column should be darker than right column
        assert target_np[0, 0, 0] < target_np[0, -1, 0], \
            "Gradient scene should be darker on the left"

    def test_checkerboard_scene(self):
        scene = create_checkerboard_scene(N=100, width=16, height=16, tile_size=4)
        target = scene["target_image"]
        mx.eval(target)
        target_np = np.array(target)
        # Should have both 0 and 1 values
        assert target_np.min() == pytest.approx(0.0, abs=1e-6)
        assert target_np.max() == pytest.approx(1.0, abs=1e-6)

    def test_scenes_reproducible(self):
        """Same seed should produce identical scenes."""
        s1 = create_solid_color_scene(seed=99)
        s2 = create_solid_color_scene(seed=99)
        mx.eval(s1["means"], s2["means"])
        np.testing.assert_array_equal(np.array(s1["means"]), np.array(s2["means"]))
