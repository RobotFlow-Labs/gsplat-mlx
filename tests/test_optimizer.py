"""Tests for SelectiveAdam optimizer.

Tests cover:
1. Basic Adam update (no visibility mask / all visible)
2. Selective update (partial visibility)
3. Selective state preservation (invisible state frozen)
4. Momentum accumulation over multiple steps
5. Bias correction mode
6. Multi-shape parameters ([N], [N,3], [N,4], [N,K,3])
7. State persistence across steps
8. Zero gradients with momentum decay
9. All-invisible mask produces no change
10. Params without gradients returned unchanged
11. Convergence on quadratic (full visibility)
12. Selective convergence (partial visibility)
13. State resize without indices
14. State resize with indices
15. Input validation (6 sub-tests)
16. Standalone adam() function
"""

import mlx.core as mx
import pytest

from gsplat_mlx.optimizers import SelectiveAdam
from gsplat_mlx.optimizers.selective_adam import adam


# ---------------------------------------------------------------------------
# Test Class 1: Basic Adam Behavior (all visible)
# ---------------------------------------------------------------------------
class TestSelectiveAdamBasic:
    """Test basic Adam behavior with all Gaussians visible."""

    def test_adam_basic(self):
        """Parameters should update when all Gaussians are visible.

        Verifies the exact numerical output of one Adam step with known inputs.
        """
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # All params should have changed
        diff = mx.abs(updated["means"] - params["means"])
        assert mx.all(diff > 0).item(), "All params should update when all visible"

        # After one step with constant gradient 0.5:
        # m = (1 - 0.9) * 0.5 = 0.05
        # v = (1 - 0.999) * 0.25 = 0.00025
        # update = 0.01 * 0.05 / (sqrt(0.00025) + 1e-8)
        #        = 0.01 * 0.05 / 0.015811...
        #        = 0.01 * 3.1623...
        #        ~ 0.031623
        # new_param = 1.0 - 0.031623 ~ 0.968377
        expected_m = (1 - 0.9) * 0.5  # = 0.05
        expected_v = (1 - 0.999) * 0.25  # = 0.00025
        expected_update = 0.01 * expected_m / (mx.sqrt(mx.array(expected_v)) + 1e-8)
        expected_param = 1.0 - expected_update.item()

        assert mx.allclose(
            updated["means"],
            mx.full((N, 3), expected_param, dtype=mx.float32),
            atol=1e-6,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 2: Selective Update with Partial Visibility
# ---------------------------------------------------------------------------
class TestSelectiveAdamVisibility:
    """Test selective update with partial visibility."""

    def test_adam_selective(self):
        """Only visible Gaussians should be updated."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}

        # First 5 visible, last 5 invisible
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # Visible Gaussians should have changed
        visible_diff = mx.abs(updated["means"][:5] - params["means"][:5])
        assert mx.all(visible_diff > 0).item(), "Visible params should update"

        # Invisible Gaussians should be unchanged
        invisible_diff = mx.abs(updated["means"][5:] - params["means"][5:])
        assert mx.all(invisible_diff == 0).item(), (
            "Invisible params should not update"
        )

    def test_adam_selective_state_preserved(self):
        """Optimizer state for invisible Gaussians should be frozen."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 0.5}
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        _ = optimizer.step(params, grads, visibility)

        state = optimizer.state["means"]

        # Invisible Gaussians should have zero exp_avg and exp_avg_sq
        assert mx.all(state["exp_avg"][5:] == 0).item(), (
            "Invisible exp_avg should be zero"
        )
        assert mx.all(state["exp_avg_sq"][5:] == 0).item(), (
            "Invisible exp_avg_sq should be zero"
        )

        # Visible Gaussians should have non-zero state
        assert mx.all(state["exp_avg"][:5] != 0).item(), (
            "Visible exp_avg should be non-zero"
        )
        assert mx.all(state["exp_avg_sq"][:5] != 0).item(), (
            "Visible exp_avg_sq should be non-zero"
        )


# ---------------------------------------------------------------------------
# Test Class 3: Momentum Accumulation
# ---------------------------------------------------------------------------
class TestSelectiveAdamMomentum:
    """Test momentum accumulation over multiple steps."""

    def test_adam_momentum(self):
        """Verify exp_avg accumulates correctly over multiple steps
        with a constant gradient.
        """
        N = 4
        params = {"x": mx.ones((N,), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)
        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        g_val = 1.0

        # Step 1: m = (1-0.9)*1.0 = 0.1
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m1 = 0.1 * g_val  # = 0.1
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m1, dtype=mx.float32),
            atol=1e-6,
        ).item()

        # Step 2: m = 0.9*0.1 + 0.1*1.0 = 0.19
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m2 = 0.9 * m1 + 0.1 * g_val  # = 0.19
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m2, dtype=mx.float32),
            atol=1e-6,
        ).item()

        # Step 3: m = 0.9*0.19 + 0.1*1.0 = 0.271
        grads = {"x": mx.full((N,), g_val, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m3 = 0.9 * m2 + 0.1 * g_val  # = 0.271
        assert mx.allclose(
            optimizer.state["x"]["exp_avg"],
            mx.full((N,), m3, dtype=mx.float32),
            atol=1e-6,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 4: Bias Correction
# ---------------------------------------------------------------------------
class TestSelectiveAdamBiasCorrection:
    """Test bias correction mode."""

    def test_adam_bias_correction(self):
        """With bias_correction=True, early steps should have larger updates
        due to the 1/(1-beta^t) correction factor.
        """
        N = 4
        params_no_bc = {"x": mx.ones((N,), dtype=mx.float32)}
        params_bc = {"x": mx.ones((N,), dtype=mx.float32)}
        grads = {"x": mx.full((N,), 0.5, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        opt_no_bc = SelectiveAdam(
            lr=0.01, betas=(0.9, 0.999), eps=1e-8, bias_correction=False
        )
        opt_bc = SelectiveAdam(
            lr=0.01, betas=(0.9, 0.999), eps=1e-8, bias_correction=True
        )

        updated_no_bc = opt_no_bc.step(params_no_bc, grads, visibility)
        updated_bc = opt_bc.step(params_bc, grads, visibility)

        # Bias correction changes the update magnitude.
        # Without BC: update = lr * m / (sqrt(v) + eps) = 0.01 * 0.05 / 0.01581 ~ 0.0316
        # With BC: update = lr * (m/(1-b1)) / (sqrt(v/(1-b2)) + eps) = 0.01 * 0.5 / 0.5 = 0.01
        # The bias-corrected and non-bias-corrected updates differ.
        diff_no_bc = mx.abs(updated_no_bc["x"] - 1.0)
        diff_bc = mx.abs(updated_bc["x"] - 1.0)

        assert not mx.allclose(diff_bc, diff_no_bc, atol=1e-6).item(), (
            "Bias-corrected and non-bias-corrected updates should differ"
        )

        # Verify exact bias-corrected values at step 1:
        # m = 0.05, v = 0.00025
        # m_hat = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
        # v_hat = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
        # update = 0.01 * 0.5 / (sqrt(0.25) + 1e-8) = 0.01 * 0.5 / 0.5 = 0.01
        # new_param = 1.0 - 0.01 = 0.99
        expected_bc = 1.0 - 0.01
        assert mx.allclose(
            updated_bc["x"],
            mx.full((N,), expected_bc, dtype=mx.float32),
            atol=1e-5,
        ).item()


# ---------------------------------------------------------------------------
# Test Class 5: Multi-Shape Parameters
# ---------------------------------------------------------------------------
class TestSelectiveAdamMultiShape:
    """Test with parameters of different shapes."""

    def test_adam_multi_shape(self):
        """Optimizer should handle [N], [N,3], [N,4], [N,K,3] params."""
        N = 8
        K = 9  # SH degree 2: 9 coefficients

        params = {
            "opacities": mx.ones((N,), dtype=mx.float32),
            "means": mx.ones((N, 3), dtype=mx.float32),
            "quats": mx.ones((N, 4), dtype=mx.float32),
            "sh_coeffs": mx.ones((N, K, 3), dtype=mx.float32),
        }

        grads = {
            "opacities": mx.full((N,), 0.1, dtype=mx.float32),
            "means": mx.full((N, 3), 0.2, dtype=mx.float32),
            "quats": mx.full((N, 4), 0.3, dtype=mx.float32),
            "sh_coeffs": mx.full((N, K, 3), 0.05, dtype=mx.float32),
        }

        # Half visible
        visibility = mx.array(
            [True] * 4 + [False] * 4, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        for name in params:
            # Visible should change
            vis_diff = mx.max(mx.abs(
                updated[name][:4] - params[name][:4]
            )).item()
            assert vis_diff > 0, f"Visible {name} should update"

            # Invisible should not change
            invis_diff = mx.max(mx.abs(
                updated[name][4:] - params[name][4:]
            )).item()
            assert invis_diff == 0, f"Invisible {name} should not update"

        # Verify state shapes match param shapes
        for name in params:
            assert optimizer.state[name]["exp_avg"].shape == params[name].shape
            assert optimizer.state[name]["exp_avg_sq"].shape == params[name].shape


# ---------------------------------------------------------------------------
# Test Class 6: State Persistence
# ---------------------------------------------------------------------------
class TestSelectiveAdamStatePersistence:
    """Test that state persists correctly across steps."""

    def test_adam_state_persistence(self):
        """State should carry over between steps and accumulate correctly."""
        N = 4
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # Run 5 steps with constant gradient
        for i in range(5):
            grads = {"x": mx.full((N, 3), 1.0, dtype=mx.float32)}
            params = optimizer.step(params, grads, visibility)

        # Step counter should be 5
        assert optimizer.state["x"]["step"].item() == 5

        # exp_avg after 5 steps with constant grad=1.0:
        # m = 1.0 * (1 - 0.9^5) = 1.0 * 0.40951 = 0.40951
        # (geometric series: sum_{k=0}^{4} 0.1 * 0.9^k = 0.1 * (1-0.9^5)/(1-0.9))
        # = 0.1 * (1 - 0.59049) / 0.1 = 1.0 * 0.40951 = 0.40951
        expected_m = 1.0 * (1 - 0.9**5)
        actual_m = optimizer.state["x"]["exp_avg"][0, 0].item()
        assert abs(actual_m - expected_m) < 1e-5, (
            f"exp_avg after 5 steps: expected {expected_m}, got {actual_m}"
        )


# ---------------------------------------------------------------------------
# Test Class 7: Edge Cases
# ---------------------------------------------------------------------------
class TestSelectiveAdamEdgeCases:
    """Test edge cases."""

    def test_adam_zero_grad(self):
        """Zero gradients with all visible: momentum should decay
        (m_new = beta1 * m_old + 0 = beta1 * m_old).
        """
        N = 4
        params = {"x": mx.ones((N,), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)
        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        # Step with non-zero gradient to build up state
        grads = {"x": mx.full((N,), 1.0, dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m_after_step1 = optimizer.state["x"]["exp_avg"][0].item()

        # Step with zero gradient -- momentum should decay by beta1
        grads = {"x": mx.zeros((N,), dtype=mx.float32)}
        params = optimizer.step(params, grads, visibility)
        m_after_step2 = optimizer.state["x"]["exp_avg"][0].item()

        # m2 = 0.9 * m1 + 0.1 * 0 = 0.9 * m1
        assert abs(m_after_step2 - 0.9 * m_after_step1) < 1e-6

    def test_adam_all_invisible(self):
        """When no Gaussians are visible, nothing should change."""
        N = 10
        params = {"means": mx.ones((N, 3), dtype=mx.float32) * 42.0}
        grads = {"means": mx.ones((N, 3), dtype=mx.float32) * 100.0}
        visibility = mx.zeros((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        updated = optimizer.step(params, grads, visibility)

        # No parameters should change
        assert mx.allclose(updated["means"], params["means"]).item(), (
            "No params should update when all invisible"
        )

        # State should be initialized but all zeros (invisible -> no update)
        assert mx.all(optimizer.state["means"]["exp_avg"] == 0).item()
        assert mx.all(optimizer.state["means"]["exp_avg_sq"] == 0).item()

    def test_adam_no_grad_for_param(self):
        """Params without gradients should be returned unchanged."""
        N = 4
        params = {
            "means": mx.ones((N, 3), dtype=mx.float32),
            "colors": mx.ones((N, 3), dtype=mx.float32) * 0.5,
        }
        grads = {
            "means": mx.full((N, 3), 0.1, dtype=mx.float32),
            # No gradient for "colors"
        }
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        updated = optimizer.step(params, grads, visibility)

        # "means" should update
        assert not mx.allclose(updated["means"], params["means"]).item()
        # "colors" should be unchanged
        assert mx.allclose(updated["colors"], params["colors"]).item()


# ---------------------------------------------------------------------------
# Test Class 8: Convergence
# ---------------------------------------------------------------------------
class TestSelectiveAdamConvergence:
    """Test convergence on simple optimization problems."""

    def test_adam_convergence(self):
        """Optimize f(x) = sum((x - target)^2) with full visibility.

        Verifies that SelectiveAdam can actually minimize a quadratic.
        """
        N = 10
        target = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        target = mx.broadcast_to(target, (N, 3))

        # Start far from target
        params = {"x": mx.zeros((N, 3), dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(300):
            # Gradient of f(x) = (x - target)^2 is 2*(x - target)
            grads = {"x": 2.0 * (params["x"] - target)}
            params = optimizer.step(params, grads, visibility)

        # Should converge close to target
        error = mx.max(mx.abs(params["x"] - target)).item()
        assert error < 0.05, f"Should converge to target, max error: {error}"

    def test_adam_selective_convergence(self):
        """Only visible Gaussians should converge; invisible should stay put."""
        N = 10
        target = mx.full((N, 3), 5.0, dtype=mx.float32)

        params = {"x": mx.zeros((N, 3), dtype=mx.float32)}
        visibility = mx.array(
            [True] * 5 + [False] * 5, dtype=mx.bool_
        )

        optimizer = SelectiveAdam(lr=0.05, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(300):
            grads = {"x": 2.0 * (params["x"] - target)}
            params = optimizer.step(params, grads, visibility)

        # Visible Gaussians should converge
        visible_error = mx.max(mx.abs(params["x"][:5] - target[:5])).item()
        assert visible_error < 0.05, (
            f"Visible Gaussians should converge, max error: {visible_error}"
        )

        # Invisible Gaussians should remain at initial value (0.0)
        invisible_diff = mx.max(mx.abs(params["x"][5:])).item()
        assert invisible_diff == 0.0, (
            f"Invisible Gaussians should not move, max diff: {invisible_diff}"
        )


# ---------------------------------------------------------------------------
# Test Class 9: State Resize for Densification
# ---------------------------------------------------------------------------
class TestSelectiveAdamResize:
    """Test state resizing for densification."""

    def test_resize_state_no_indices(self):
        """Resizing without indices should zero-initialize new state."""
        N = 10
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"x": mx.full((N, 3), 0.5, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        _ = optimizer.step(params, grads, visibility)

        # State should exist
        assert "x" in optimizer.state
        old_step = optimizer.state["x"]["step"].item()

        # Resize to larger N
        optimizer.resize_state("x", new_n=20)

        # State should be resized with zeros
        assert optimizer.state["x"]["exp_avg"].shape == (20, 3)
        assert optimizer.state["x"]["exp_avg_sq"].shape == (20, 3)
        assert mx.all(optimizer.state["x"]["exp_avg"] == 0).item()
        assert mx.all(optimizer.state["x"]["exp_avg_sq"] == 0).item()
        # Step should be preserved
        assert optimizer.state["x"]["step"].item() == old_step

    def test_resize_state_with_indices(self):
        """Resizing with indices should preserve selected state."""
        N = 4
        params = {"x": mx.ones((N, 3), dtype=mx.float32)}
        grads = {"x": mx.full((N, 3), 1.0, dtype=mx.float32)}
        visibility = mx.ones((N,), dtype=mx.bool_)

        optimizer = SelectiveAdam(lr=0.01)
        _ = optimizer.step(params, grads, visibility)

        old_m = optimizer.state["x"]["exp_avg"]
        mx.eval(old_m)

        # Resize: keep Gaussians 0,1,2,3 and add 2 new ones (index -1)
        indices = mx.array([0, 1, 2, 3, -1, -1])
        optimizer.resize_state("x", new_n=6, indices=indices)

        new_m = optimizer.state["x"]["exp_avg"]
        assert new_m.shape == (6, 3)

        # First 4 should match old state
        assert mx.allclose(new_m[:4], old_m, atol=1e-7).item()
        # Last 2 should be zero
        assert mx.all(new_m[4:] == 0).item()

    def test_resize_nonexistent_param(self):
        """Resizing state for a parameter that has no state should be a no-op."""
        optimizer = SelectiveAdam(lr=0.01)
        # Should not raise
        optimizer.resize_state("nonexistent", new_n=10)
        assert "nonexistent" not in optimizer.state


# ---------------------------------------------------------------------------
# Test Class 10: Input Validation
# ---------------------------------------------------------------------------
class TestSelectiveAdamValidation:
    """Test input validation."""

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="learning rate"):
            SelectiveAdam(lr=-0.01)

    def test_invalid_beta1(self):
        with pytest.raises(ValueError, match="beta1"):
            SelectiveAdam(betas=(1.0, 0.999))

    def test_invalid_beta2(self):
        with pytest.raises(ValueError, match="beta2"):
            SelectiveAdam(betas=(0.9, 1.0))

    def test_invalid_eps(self):
        with pytest.raises(ValueError, match="epsilon"):
            SelectiveAdam(eps=-1e-8)

    def test_shape_mismatch(self):
        optimizer = SelectiveAdam()
        params = {"x": mx.ones((4, 3))}
        grads = {"x": mx.ones((4, 2))}  # wrong shape
        visibility = mx.ones((4,), dtype=mx.bool_)

        with pytest.raises(ValueError, match="Shape mismatch"):
            optimizer.step(params, grads, visibility)

    def test_unknown_grad_key(self):
        optimizer = SelectiveAdam()
        params = {"x": mx.ones((4, 3))}
        grads = {"y": mx.ones((4, 3))}  # key not in params
        visibility = mx.ones((4,), dtype=mx.bool_)

        with pytest.raises(ValueError, match="not found in params"):
            optimizer.step(params, grads, visibility)


# ---------------------------------------------------------------------------
# Test Class 11: Standalone adam() Function
# ---------------------------------------------------------------------------
class TestStandaloneAdam:
    """Test the standalone adam() function."""

    def test_standalone_adam_matches_class(self):
        """Standalone adam() should produce identical results to SelectiveAdam.step()."""
        N = 8
        param = mx.ones((N, 3), dtype=mx.float32)
        grad = mx.full((N, 3), 0.5, dtype=mx.float32)
        exp_avg = mx.zeros((N, 3), dtype=mx.float32)
        exp_avg_sq = mx.zeros((N, 3), dtype=mx.float32)
        valid = mx.array([True] * 4 + [False] * 4, dtype=mx.bool_)

        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8

        # Standalone function
        p_out, m_out, v_out = adam(param, grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps)
        mx.eval(p_out, m_out, v_out)

        # Class-based
        optimizer = SelectiveAdam(lr=lr, betas=(b1, b2), eps=eps)
        updated = optimizer.step({"x": param}, {"x": grad}, valid)

        # Should match
        assert mx.allclose(p_out, updated["x"], atol=1e-7).item(), (
            "Standalone adam() should match SelectiveAdam.step()"
        )
        assert mx.allclose(m_out, optimizer.state["x"]["exp_avg"], atol=1e-7).item()
        assert mx.allclose(v_out, optimizer.state["x"]["exp_avg_sq"], atol=1e-7).item()

    def test_standalone_adam_selective(self):
        """Standalone adam() should respect visibility mask."""
        N = 6
        param = mx.ones((N,), dtype=mx.float32) * 10.0
        grad = mx.ones((N,), dtype=mx.float32)
        exp_avg = mx.zeros((N,), dtype=mx.float32)
        exp_avg_sq = mx.zeros((N,), dtype=mx.float32)
        valid = mx.array([True, True, True, False, False, False], dtype=mx.bool_)

        p_out, m_out, v_out = adam(param, grad, exp_avg, exp_avg_sq, valid, 0.01, 0.9, 0.999, 1e-8)
        mx.eval(p_out, m_out, v_out)

        # Visible should change
        assert mx.all(mx.abs(p_out[:3] - 10.0) > 0).item()
        # Invisible should not change
        assert mx.allclose(p_out[3:], mx.full((3,), 10.0)).item()
        # Invisible state should be zero
        assert mx.all(m_out[3:] == 0).item()
        assert mx.all(v_out[3:] == 0).item()
