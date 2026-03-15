"""Tests for Gaussian densification strategy module.

Covers:
- Strategy base class sanity checking
- Low-level ops: duplicate, remove, split, reset_opa
- DefaultStrategy initialization, grow, and prune logic
- Value preservation after ops
"""

import numpy as np
import pytest

import mlx.core as mx

from gsplat_mlx.strategy.base import Strategy
from gsplat_mlx.strategy.default import DefaultStrategy
from gsplat_mlx.strategy.ops import duplicate, remove, reset_opa, split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(n: int, seed: int = 42) -> dict:
    """Create a minimal param dict for testing."""
    np.random.seed(seed)
    params = {
        "means": mx.array(np.random.randn(n, 3).astype(np.float32)),
        "quats": mx.array(
            (
                lambda q: q / np.linalg.norm(q, axis=-1, keepdims=True)
            )(np.random.randn(n, 4).astype(np.float32))
        ),
        "scales": mx.array(np.random.uniform(-2.0, 1.0, (n, 3)).astype(np.float32)),
        "opacities": mx.array(np.random.uniform(-2.0, 2.0, (n,)).astype(np.float32)),
    }
    return params


def _make_optimizers(params: dict) -> dict:
    """Create mock optimizer state matching each param."""
    optimizers = {}
    for name, p in params.items():
        optimizers[name] = {
            "exp_avg": mx.zeros_like(p),
            "exp_avg_sq": mx.zeros_like(p),
            "step": 0,
        }
    return optimizers


def _make_state(n: int) -> dict:
    """Create mock running state."""
    return {
        "grad2d": mx.zeros((n,)),
        "count": mx.zeros((n,)),
        "scene_scale": 1.0,
    }


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------


class TestStrategySanity:
    """Tests for Strategy.check_sanity."""

    def test_check_sanity_valid(self):
        """Valid params/optimizers should pass sanity check."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        strategy = Strategy()
        strategy.check_sanity(params, optimizers)  # should not raise

    def test_check_sanity_optimizer_subset(self):
        """Optimizer keys can be a subset of param keys."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        # Add a frozen param (no optimizer entry)
        params["frozen_feature"] = mx.zeros((10, 16))
        strategy = Strategy()
        strategy.check_sanity(params, optimizers)  # should not raise

    def test_check_sanity_invalid(self):
        """Optimizer key not in params should fail."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        optimizers["nonexistent"] = {"exp_avg": mx.zeros((10,))}
        strategy = Strategy()
        with pytest.raises(AssertionError, match="subset"):
            strategy.check_sanity(params, optimizers)


# ---------------------------------------------------------------------------
# duplicate
# ---------------------------------------------------------------------------


class TestDuplicate:
    """Tests for the duplicate operation."""

    def test_duplicate_basic(self):
        """N=10, mask first 3 -> N=13."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.array(
            [True, True, True, False, False, False, False, False, False, False]
        )

        duplicate(params, optimizers, state, mask)

        assert params["means"].shape[0] == 13
        assert params["scales"].shape[0] == 13
        assert params["quats"].shape[0] == 13
        assert params["opacities"].shape[0] == 13
        assert state["grad2d"].shape[0] == 13
        assert state["count"].shape[0] == 13

    def test_duplicate_preserves_values(self):
        """Cloned values must match the originals."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)

        # Save original values
        mx.eval(params["means"])
        orig_means = np.array(params["means"])

        mask = mx.array(
            [True, False, True, False, False, False, False, False, False, False]
        )
        duplicate(params, optimizers, state, mask)
        mx.eval(params["means"])

        new_means = np.array(params["means"])
        # First 10 are original
        np.testing.assert_allclose(new_means[:10], orig_means, atol=1e-7)
        # Cloned entries (indices 10, 11) should match originals (indices 0, 2)
        np.testing.assert_allclose(new_means[10], orig_means[0], atol=1e-7)
        np.testing.assert_allclose(new_means[11], orig_means[2], atol=1e-7)

    def test_duplicate_optimizer_zeros(self):
        """New Gaussians should have zero optimizer state."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        # Set some non-zero optimizer state
        optimizers["means"]["exp_avg"] = mx.ones_like(params["means"])
        state = _make_state(10)
        mask = mx.array(
            [True, True, False, False, False, False, False, False, False, False]
        )

        duplicate(params, optimizers, state, mask)
        mx.eval(optimizers["means"]["exp_avg"])

        exp_avg = np.array(optimizers["means"]["exp_avg"])
        # Original 10 should still be ones
        np.testing.assert_allclose(exp_avg[:10], 1.0, atol=1e-7)
        # New 2 should be zeros
        np.testing.assert_allclose(exp_avg[10:], 0.0, atol=1e-7)

    def test_duplicate_empty_mask(self):
        """Empty mask should be a no-op."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.zeros((10,), dtype=mx.bool_)

        duplicate(params, optimizers, state, mask)

        assert params["means"].shape[0] == 10


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    """Tests for the remove operation."""

    def test_remove_basic(self):
        """N=10, mask first 3 for removal -> N=7."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.array(
            [True, True, True, False, False, False, False, False, False, False]
        )

        remove(params, optimizers, state, mask)

        assert params["means"].shape[0] == 7
        assert params["scales"].shape[0] == 7
        assert params["quats"].shape[0] == 7
        assert params["opacities"].shape[0] == 7
        assert state["grad2d"].shape[0] == 7

    def test_remove_preserves_remaining(self):
        """Remaining values should be unchanged after removal."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)

        mx.eval(params["means"])
        orig_means = np.array(params["means"])

        # Remove indices 0, 1, 2 -- keep indices 3-9
        mask = mx.array(
            [True, True, True, False, False, False, False, False, False, False]
        )
        remove(params, optimizers, state, mask)
        mx.eval(params["means"])

        new_means = np.array(params["means"])
        np.testing.assert_allclose(new_means, orig_means[3:], atol=1e-7)

    def test_remove_empty_mask(self):
        """Empty mask should be a no-op."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.zeros((10,), dtype=mx.bool_)

        remove(params, optimizers, state, mask)

        assert params["means"].shape[0] == 10

    def test_remove_all(self):
        """Removing all Gaussians should yield shape [0, ...]."""
        params = _make_params(5)
        optimizers = _make_optimizers(params)
        state = _make_state(5)
        mask = mx.ones((5,), dtype=mx.bool_)

        remove(params, optimizers, state, mask)

        assert params["means"].shape[0] == 0
        assert params["scales"].shape[0] == 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


class TestSplit:
    """Tests for the split operation."""

    def test_split_basic(self):
        """Split 3 of 10 -> remove 3, add 6 = 13 total."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.array(
            [True, True, True, False, False, False, False, False, False, False]
        )

        split(params, optimizers, state, mask)

        # 10 - 3 (removed) + 6 (2 per split) = 13
        assert params["means"].shape[0] == 13
        assert params["scales"].shape[0] == 13
        assert params["quats"].shape[0] == 13
        assert params["opacities"].shape[0] == 13

    def test_split_reduces_scales(self):
        """Split Gaussians should have smaller scales (divided by 1.6)."""
        n = 5
        params = _make_params(n, seed=123)
        optimizers = _make_optimizers(params)
        state = _make_state(n)

        mx.eval(params["scales"])
        orig_scales = np.array(params["scales"])

        # Split all Gaussians
        mask = mx.ones((n,), dtype=mx.bool_)
        split(params, optimizers, state, mask)
        mx.eval(params["scales"])

        new_scales = np.array(params["scales"])
        # All originals removed, 2*n new ones
        assert new_scales.shape[0] == 2 * n

        # Expected log-scale: log(exp(orig_scale) / 1.6)
        expected_log_scales = np.log(np.exp(orig_scales) / 1.6)
        # First n are from first copy, next n from second copy
        np.testing.assert_allclose(
            new_scales[:n], expected_log_scales, atol=1e-5
        )
        np.testing.assert_allclose(
            new_scales[n:], expected_log_scales, atol=1e-5
        )

    def test_split_empty_mask(self):
        """Empty mask should be a no-op."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)
        mask = mx.zeros((10,), dtype=mx.bool_)

        split(params, optimizers, state, mask)

        assert params["means"].shape[0] == 10

    def test_split_optimizer_zeros(self):
        """Split Gaussians should have zero optimizer state."""
        params = _make_params(5)
        optimizers = _make_optimizers(params)
        # Set non-zero optimizer state
        optimizers["means"]["exp_avg"] = mx.ones_like(params["means"])
        state = _make_state(5)
        mask = mx.array([True, True, False, False, False])

        split(params, optimizers, state, mask)
        mx.eval(optimizers["means"]["exp_avg"])

        exp_avg = np.array(optimizers["means"]["exp_avg"])
        # First 3 are the kept originals (indices 2,3,4 -> should be ones)
        np.testing.assert_allclose(exp_avg[:3], 1.0, atol=1e-7)
        # Last 4 are from splits -> should be zeros
        np.testing.assert_allclose(exp_avg[3:], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# reset_opa
# ---------------------------------------------------------------------------


class TestResetOpa:
    """Tests for the reset_opa operation."""

    def test_reset_opa(self):
        """All opacities should be clamped to at most the target value."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = _make_state(10)

        target_value = 0.01  # post-sigmoid target
        reset_opa(params, optimizers, state, value=target_value)
        mx.eval(params["opacities"])

        # Check that sigmoid(opacities) <= target_value (with tolerance)
        opacities_sigmoid = 1.0 / (1.0 + np.exp(-np.array(params["opacities"])))
        assert np.all(
            opacities_sigmoid <= target_value + 1e-6
        ), f"Max sigmoid opacity: {opacities_sigmoid.max()}, target: {target_value}"

    def test_reset_opa_zeros_optimizer(self):
        """Optimizer state for opacities should be zeroed after reset."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        optimizers["opacities"]["exp_avg"] = mx.ones((10,))
        optimizers["opacities"]["exp_avg_sq"] = mx.ones((10,))
        state = _make_state(10)

        reset_opa(params, optimizers, state, value=0.01)
        mx.eval(optimizers["opacities"]["exp_avg"])

        np.testing.assert_allclose(
            np.array(optimizers["opacities"]["exp_avg"]), 0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            np.array(optimizers["opacities"]["exp_avg_sq"]), 0.0, atol=1e-7
        )

    def test_reset_opa_preserves_low(self):
        """Opacities already below the target should not be raised."""
        params = _make_params(10)
        # Set all opacities very low (sigmoid ~ 0.0005)
        params["opacities"] = mx.full((10,), -7.6)
        optimizers = _make_optimizers(params)
        state = _make_state(10)

        mx.eval(params["opacities"])
        orig = np.array(params["opacities"])

        reset_opa(params, optimizers, state, value=0.01)
        mx.eval(params["opacities"])

        new = np.array(params["opacities"])
        # Low values should be unchanged (clamp is a max, not a set)
        np.testing.assert_allclose(new, orig, atol=1e-7)


# ---------------------------------------------------------------------------
# DefaultStrategy
# ---------------------------------------------------------------------------


class TestDefaultStrategy:
    """Tests for DefaultStrategy."""

    def test_initialize_state(self):
        """State should have expected keys."""
        strategy = DefaultStrategy()
        state = strategy.initialize_state(scene_scale=2.0)

        assert "grad2d" in state
        assert "count" in state
        assert "scene_scale" in state
        assert state["grad2d"] is None
        assert state["count"] is None
        assert state["scene_scale"] == 2.0

    def test_initialize_state_with_scale2d(self):
        """With refine_scale2d_stop_iter > 0, state should include radii."""
        strategy = DefaultStrategy(refine_scale2d_stop_iter=5000)
        state = strategy.initialize_state()

        assert "radii" in state
        assert state["radii"] is None

    def test_check_sanity_valid(self):
        """Valid params should pass check_sanity."""
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        strategy = DefaultStrategy()
        strategy.check_sanity(params, optimizers)  # should not raise

    def test_check_sanity_missing_key(self):
        """Missing required key should fail check_sanity."""
        params = _make_params(10)
        del params["means"]
        optimizers = _make_optimizers(params)
        strategy = DefaultStrategy()
        with pytest.raises(AssertionError, match="means"):
            strategy.check_sanity(params, optimizers)

    def test_step_pre_backward(self):
        """step_pre_backward should validate info dict."""
        strategy = DefaultStrategy()
        params = _make_params(10)
        optimizers = _make_optimizers(params)
        state = strategy.initialize_state()

        # Should pass with valid info
        info = {"means2d": mx.zeros((1, 10, 2))}
        strategy.step_pre_backward(params, optimizers, state, step=0, info=info)

        # Should fail without the key
        with pytest.raises(AssertionError):
            strategy.step_pre_backward(
                params, optimizers, state, step=0, info={}
            )

    def test_grow_triggers_duplication(self):
        """High gradients + small scales should trigger duplication."""
        n = 20
        params = _make_params(n, seed=99)
        # Make scales very small so they qualify for duplication
        params["scales"] = mx.full((n, 3), -5.0)  # exp(-5) ~ 0.007
        optimizers = _make_optimizers(params)

        strategy = DefaultStrategy(
            grow_grad2d=0.0001,
            grow_scale3d=0.01,
            refine_start_iter=0,
            refine_every=1,
            reset_every=100000,
            refine_stop_iter=100000,
        )
        state = strategy.initialize_state(scene_scale=1.0)

        # Manually set high gradient accumulators
        state["grad2d"] = mx.full((n,), 0.001)  # above grow_grad2d
        state["count"] = mx.ones((n,))

        # Call _grow_gs directly
        n_dupli, n_split = strategy._grow_gs(params, optimizers, state, step=1)

        assert n_dupli > 0, "Expected some duplications"
        assert params["means"].shape[0] > n, "Gaussian count should have increased"

    def test_prune_removes_low_opacity(self):
        """Low-opacity Gaussians should be pruned."""
        n = 20
        params = _make_params(n, seed=99)
        # Set half the opacities very low (sigmoid ~ 0.0005)
        opa = np.array(params["opacities"])
        opa[:10] = -7.6  # sigmoid(-7.6) ~ 0.0005
        opa[10:] = 2.0  # sigmoid(2.0) ~ 0.88
        params["opacities"] = mx.array(opa.astype(np.float32))
        optimizers = _make_optimizers(params)

        strategy = DefaultStrategy(prune_opa=0.005)
        state = _make_state(n)

        n_prune = strategy._prune_gs(params, optimizers, state, step=0)

        assert n_prune == 10, f"Expected 10 pruned, got {n_prune}"
        assert params["means"].shape[0] == 10

    def test_full_step_post_backward(self):
        """Integration: step_post_backward should run without error."""
        n = 50
        params = _make_params(n, seed=77)
        optimizers = _make_optimizers(params)
        strategy = DefaultStrategy(
            refine_start_iter=0,
            refine_every=10,
            refine_stop_iter=1000,
            reset_every=500,
        )
        state = strategy.initialize_state(scene_scale=1.0)

        # Create mock info dict
        info = {
            "means2d": mx.zeros((1, n, 2)),
            "means2d_grad": mx.random.normal(shape=(1, n, 2)) * 0.001,
            "radii": mx.ones((1, n, 2)) * 5.0,
            "gaussian_ids": mx.arange(n),
            "width": 640,
            "height": 480,
            "n_cameras": 1,
        }

        # Run a few steps -- should not crash
        for step in range(15):
            strategy.step_pre_backward(params, optimizers, state, step, info)
            strategy.step_post_backward(
                params, optimizers, state, step, info, packed=False
            )
            # Update info to match potentially changed param count
            new_n = params["means"].shape[0]
            info["means2d"] = mx.zeros((1, new_n, 2))
            info["means2d_grad"] = mx.random.normal(shape=(1, new_n, 2)) * 0.001
            info["radii"] = mx.ones((1, new_n, 2)) * 5.0
            info["gaussian_ids"] = mx.arange(new_n)

        # Should have run without error
        assert params["means"].shape[0] > 0
