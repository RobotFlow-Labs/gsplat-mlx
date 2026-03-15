"""Tests for extended strategy ops: sample_add and inject_noise_to_position.

Covers:
- sample_add increases Gaussian count by n
- inject_noise changes means
- inject_noise preserves other parameters (quats, scales)
"""

import numpy as np
import pytest

import mlx.core as mx

from gsplat_mlx.strategy.ops import sample_add, inject_noise_to_position
from gsplat_mlx.relocation import compute_binomial_coefficients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(n: int, seed: int = 42) -> dict:
    """Create a minimal param dict for testing."""
    np.random.seed(seed)
    params = {
        "means": mx.array(np.random.randn(n, 3).astype(np.float32)),
        "quats": mx.array(
            np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
        ),
        "scales": mx.array(np.random.randn(n, 3).astype(np.float32)),
        "opacities": mx.array(np.random.randn(n).astype(np.float32)),
        "sh0": mx.array(np.random.randn(n, 1, 3).astype(np.float32)),
    }
    return params


def _make_optimizers(params: dict) -> dict:
    """Create mock optimizer state dicts."""
    opt = {}
    for name, p in params.items():
        opt[name] = {
            "exp_avg": mx.zeros_like(p),
            "exp_avg_sq": mx.zeros_like(p),
            "step": 0,
        }
    return opt


def _make_state(n: int) -> dict:
    """Create minimal strategy running state."""
    return {
        "grad2d": mx.zeros((n,), dtype=mx.float32),
        "count": mx.zeros((n,), dtype=mx.int32),
    }


# ---------------------------------------------------------------------------
# Tests: sample_add
# ---------------------------------------------------------------------------


class TestSampleAdd:
    """Tests for sample_add."""

    def test_sample_add_increases_count(self):
        """After sample_add(n=5), total Gaussians should increase by 5."""
        N = 20
        n_new = 5
        params = _make_params(N)
        optimizers = _make_optimizers(params)
        state = _make_state(N)
        binoms = compute_binomial_coefficients(10)

        sample_add(params, optimizers, state, n=n_new, binoms=binoms, seed=123)
        mx.eval(*params.values())

        expected = N + n_new
        assert params["means"].shape[0] == expected, (
            f"Expected {expected} Gaussians, got {params['means'].shape[0]}"
        )
        assert params["opacities"].shape[0] == expected
        assert params["scales"].shape[0] == expected
        assert params["quats"].shape[0] == expected

    def test_sample_add_optimizer_state_grows(self):
        """Optimizer state arrays should also grow by n_new."""
        N = 15
        n_new = 3
        params = _make_params(N)
        optimizers = _make_optimizers(params)
        state = _make_state(N)
        binoms = compute_binomial_coefficients(10)

        sample_add(params, optimizers, state, n=n_new, binoms=binoms, seed=99)
        mx.eval(*params.values())

        expected = N + n_new
        for name in params:
            assert optimizers[name]["exp_avg"].shape[0] == expected, (
                f"Optimizer exp_avg for '{name}' should have {expected} rows"
            )

    def test_sample_add_state_grows(self):
        """Running state arrays should also grow by n_new."""
        N = 10
        n_new = 4
        params = _make_params(N)
        optimizers = _make_optimizers(params)
        state = _make_state(N)
        binoms = compute_binomial_coefficients(10)

        sample_add(params, optimizers, state, n=n_new, binoms=binoms, seed=77)
        mx.eval(*params.values())

        expected = N + n_new
        assert state["grad2d"].shape[0] == expected
        assert state["count"].shape[0] == expected


# ---------------------------------------------------------------------------
# Tests: inject_noise_to_position
# ---------------------------------------------------------------------------


class TestInjectNoise:
    """Tests for inject_noise_to_position."""

    def test_inject_noise_changes_means(self):
        """Means should differ after noise injection."""
        N = 20
        params = _make_params(N)
        # Set opacities to very low values (logit << 0 => sigmoid ~ 0)
        # so that (1 - opacity) ~ 1.0, which passes the steep sigmoid gate
        params["opacities"] = mx.full((N,), -10.0, dtype=mx.float32)
        means_before = np.array(params["means"])

        inject_noise_to_position(params, scaler=1.0, seed=42)
        mx.eval(params["means"])

        means_after = np.array(params["means"])
        # With near-zero opacity, the gate opens and noise is injected
        assert not np.allclose(means_before, means_after, atol=1e-6), (
            "Means should change after noise injection for low-opacity Gaussians"
        )

    def test_inject_noise_preserves_other_params(self):
        """Quats and scales should not change after noise injection."""
        N = 15
        params = _make_params(N)
        quats_before = np.array(params["quats"]).copy()
        scales_before = np.array(params["scales"]).copy()
        opa_before = np.array(params["opacities"]).copy()

        inject_noise_to_position(params, scaler=0.5, seed=123)
        mx.eval(params["means"])

        np.testing.assert_array_equal(
            np.array(params["quats"]), quats_before,
            err_msg="Quats should not change"
        )
        np.testing.assert_array_equal(
            np.array(params["scales"]), scales_before,
            err_msg="Scales should not change"
        )
        np.testing.assert_array_equal(
            np.array(params["opacities"]), opa_before,
            err_msg="Opacities should not change"
        )

    def test_inject_noise_zero_scaler(self):
        """With scaler=0, means should not change."""
        N = 10
        params = _make_params(N)
        means_before = np.array(params["means"]).copy()

        inject_noise_to_position(params, scaler=0.0, seed=42)
        mx.eval(params["means"])

        np.testing.assert_allclose(
            np.array(params["means"]), means_before, atol=1e-10,
            err_msg="scaler=0 should not change means"
        )

    def test_inject_noise_shape_preserved(self):
        """Output means shape should match input."""
        N = 12
        params = _make_params(N)

        inject_noise_to_position(params, scaler=0.1, seed=55)
        mx.eval(params["means"])

        assert params["means"].shape == (N, 3)
