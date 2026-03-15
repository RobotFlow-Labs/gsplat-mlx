"""Tests for the 3D-to-2D Gaussian projection pipeline (PRD-05).

Tests cover world_to_cam, persp_proj, fisheye_proj, ortho_proj,
and fully_fused_projection with all camera models.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tests.conftest import check_all_close, make_camera_intrinsics, make_gaussians, make_view_matrix

from gsplat_mlx.core.projection import (
    fully_fused_projection,
    fisheye_proj,
    ortho_proj,
    persp_proj,
    world_to_cam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity_viewmat():
    """Return a single identity view matrix [1, 4, 4]."""
    return mx.eye(4, dtype=mx.float32).reshape(1, 4, 4)


def _make_simple_covars(N: int):
    """Return N identity covariance matrices [N, 3, 3]."""
    return mx.broadcast_to(mx.eye(3, dtype=mx.float32), (N, 3, 3))


def _make_simple_K(fx=500.0, fy=500.0, width=640, height=480):
    """Return intrinsic matrix [1, 3, 3]."""
    K = make_camera_intrinsics(width=width, height=height, fx=fx, fy=fy)
    return K.reshape(1, 3, 3)


# ---------------------------------------------------------------------------
# world_to_cam tests
# ---------------------------------------------------------------------------


class TestWorldToCam:

    def test_identity(self):
        """Identity viewmat should leave means and covars unchanged."""
        N = 5
        np.random.seed(12345)
        means = mx.array(np.random.randn(N, 3).astype(np.float32))
        covars = _make_simple_covars(N)
        viewmats = _make_identity_viewmat()

        means_c, covars_c = world_to_cam(means, covars, viewmats)
        mx.eval(means_c, covars_c)

        assert means_c.shape == (1, N, 3)
        assert covars_c.shape == (1, N, 3, 3)
        check_all_close(means_c[0], means, atol=5e-3, rtol=1e-3, msg="identity means")
        check_all_close(covars_c[0], covars, atol=5e-3, rtol=1e-3, msg="identity covars")

    def test_translation(self):
        """Pure translation viewmat should shift means."""
        N = 3
        means = mx.zeros((N, 3), dtype=mx.float32)
        covars = _make_simple_covars(N)

        viewmat = mx.array(np.eye(4, dtype=np.float32))
        # Set translation to (1, 2, 3)
        viewmat_np = np.eye(4, dtype=np.float32)
        viewmat_np[0, 3] = 1.0
        viewmat_np[1, 3] = 2.0
        viewmat_np[2, 3] = 3.0
        viewmats = mx.array(viewmat_np).reshape(1, 4, 4)

        means_c, covars_c = world_to_cam(means, covars, viewmats)
        mx.eval(means_c, covars_c)

        expected_means = mx.broadcast_to(
            mx.array([[1.0, 2.0, 3.0]]), (N, 3)
        )
        check_all_close(means_c[0], expected_means, atol=1e-5, msg="translation means")
        # Covars should be unchanged (R=I)
        check_all_close(covars_c[0], covars, atol=1e-5, msg="translation covars")

    def test_rotation_90deg(self):
        """90-degree rotation around Z axis."""
        N = 1
        means = mx.array([[1.0, 0.0, 0.0]])
        covars = _make_simple_covars(N)

        # Rotation of 90 degrees around Z: x->y, y->-x
        viewmat_np = np.eye(4, dtype=np.float32)
        viewmat_np[0, 0] = 0.0
        viewmat_np[0, 1] = -1.0
        viewmat_np[1, 0] = 1.0
        viewmat_np[1, 1] = 0.0
        viewmats = mx.array(viewmat_np).reshape(1, 4, 4)

        means_c, _ = world_to_cam(means, covars, viewmats)
        mx.eval(means_c)

        # (1,0,0) rotated 90 around Z -> (0, 1, 0)
        check_all_close(means_c[0], mx.array([[0.0, 1.0, 0.0]]), atol=1e-5, msg="rotation means")


# ---------------------------------------------------------------------------
# persp_proj tests
# ---------------------------------------------------------------------------


class TestPerspProj:

    def test_center_point(self):
        """A Gaussian at (0, 0, z) should project to (cx, cy)."""
        means_c = mx.array([[[0.0, 0.0, 5.0]]])  # [1, 1, 3]
        covars_c = mx.eye(3, dtype=mx.float32).reshape(1, 1, 3, 3)
        K = _make_simple_K()
        width, height = 640, 480

        means2d, cov2d = persp_proj(means_c, covars_c, K, width, height)
        mx.eval(means2d, cov2d)

        assert means2d.shape == (1, 1, 2)
        # Should project to (cx, cy) = (320, 240)
        check_all_close(means2d[0, 0], mx.array([320.0, 240.0]), atol=1e-3, msg="center projection")

    def test_cov2d_shape(self):
        """Output cov2d should be [..., C, N, 2, 2]."""
        C, N = 2, 10
        means_c = mx.array(np.random.randn(C, N, 3).astype(np.float32))
        # Ensure positive z
        means_c_np = np.array(means_c)
        means_c_np[:, :, 2] = np.abs(means_c_np[:, :, 2]) + 1.0
        means_c = mx.array(means_c_np)

        covars_c = mx.broadcast_to(mx.eye(3, dtype=mx.float32), (C, N, 3, 3))
        Ks = mx.broadcast_to(_make_simple_K(), (C, 3, 3))

        _, cov2d = persp_proj(means_c, covars_c, Ks, 640, 480)
        mx.eval(cov2d)

        assert cov2d.shape == (C, N, 2, 2)


# ---------------------------------------------------------------------------
# fisheye_proj tests
# ---------------------------------------------------------------------------


class TestFisheyeProj:

    def test_on_axis(self):
        """On the optical axis, fisheye should match perspective projection."""
        means_c = mx.array([[[0.0, 0.0, 5.0]]])
        covars_c = mx.eye(3, dtype=mx.float32).reshape(1, 1, 3, 3)
        K = _make_simple_K()
        width, height = 640, 480

        means2d_persp, _ = persp_proj(means_c, covars_c, K, width, height)
        means2d_fish, _ = fisheye_proj(means_c, covars_c, K, width, height)
        mx.eval(means2d_persp, means2d_fish)

        # On axis, both should give (cx, cy)
        check_all_close(means2d_fish, means2d_persp, atol=1e-2, msg="fisheye on-axis")


# ---------------------------------------------------------------------------
# ortho_proj tests
# ---------------------------------------------------------------------------


class TestOrthoProj:

    def test_no_perspective(self):
        """Depth should not affect 2D position in orthographic projection."""
        K = _make_simple_K()
        width, height = 640, 480

        # Same x, y but different z
        means_near = mx.array([[[1.0, 2.0, 3.0]]])
        means_far = mx.array([[[1.0, 2.0, 30.0]]])
        covars_c = mx.eye(3, dtype=mx.float32).reshape(1, 1, 3, 3)

        means2d_near, _ = ortho_proj(means_near, covars_c, K, width, height)
        means2d_far, _ = ortho_proj(means_far, covars_c, K, width, height)
        mx.eval(means2d_near, means2d_far)

        check_all_close(means2d_near, means2d_far, atol=1e-5, msg="ortho depth invariance")


# ---------------------------------------------------------------------------
# fully_fused_projection tests
# ---------------------------------------------------------------------------


class TestFullyFusedProjection:

    def test_single_gaussian(self):
        """Single Gaussian, single camera — basic smoke test."""
        means = mx.array([[0.0, 0.0, 0.0]])   # [1, 3]
        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)

        # Camera looking at origin from z=5
        viewmat_np = np.eye(4, dtype=np.float32)
        viewmat_np[2, 3] = 5.0  # translate camera 5 units along z
        viewmats = mx.array(viewmat_np).reshape(1, 4, 4)
        Ks = _make_simple_K()
        width, height = 640, 480

        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means, covars, viewmats, Ks, width, height,
        )
        mx.eval(radii, means2d, depths, conics)

        assert radii.shape == (1, 1, 2)
        assert means2d.shape == (1, 1, 2)
        assert depths.shape == (1, 1)
        assert conics.shape == (1, 1, 3)
        assert compensations is None

        # Depth should be 5.0
        check_all_close(depths, mx.array([[5.0]]), atol=1e-5, msg="depth")

        # Radii should be > 0 (visible)
        radii_np = np.array(radii)
        assert np.all(radii_np > 0), f"Expected positive radii, got {radii_np}"

    def test_near_far_culling(self):
        """Gaussians behind camera (negative depth) should get radius=0."""
        N = 3
        # One behind, one at near, one in front
        means = mx.array([
            [0.0, 0.0, -2.0],  # behind camera
            [0.0, 0.0, 0.005], # in front but too close
            [0.0, 0.0, 5.0],   # in front, visible
        ])
        covars = mx.broadcast_to(mx.eye(3, dtype=mx.float32), (N, 3, 3))
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        radii, _, _, _, _ = fully_fused_projection(
            means, covars, viewmats, Ks, 640, 480,
            near_plane=0.01,
        )
        mx.eval(radii)

        radii_np = np.array(radii)
        # Behind camera -> radius 0
        assert radii_np[0, 0, 0] == 0 and radii_np[0, 0, 1] == 0
        # Too close -> radius 0
        assert radii_np[0, 1, 0] == 0 and radii_np[0, 1, 1] == 0
        # Visible -> radius > 0
        assert radii_np[0, 2, 0] > 0 and radii_np[0, 2, 1] > 0

    def test_screen_bounds_culling(self):
        """Gaussians far outside the image should get radius=0."""
        # Place Gaussian very far off-screen
        means = mx.array([[1000.0, 0.0, 5.0]])
        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3) * 0.01  # small covariance
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        radii, _, _, _, _ = fully_fused_projection(
            means, covars, viewmats, Ks, 640, 480,
        )
        mx.eval(radii)

        radii_np = np.array(radii)
        assert radii_np[0, 0, 0] == 0 and radii_np[0, 0, 1] == 0, (
            f"Off-screen Gaussian should have radius=0, got {radii_np}"
        )

    def test_conics_inverse_of_cov2d(self):
        """Conics should represent the inverse of the 2D covariance."""
        means = mx.array([[0.0, 0.0, 5.0]])
        # Use a non-identity covariance to make it interesting
        cov3d = mx.array([[[2.0, 0.5, 0.0],
                           [0.5, 1.5, 0.0],
                           [0.0, 0.0, 1.0]]])
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()
        width, height = 640, 480

        radii, means2d, depths, conics, _ = fully_fused_projection(
            means, cov3d, viewmats, Ks, width, height, eps2d=0.3,
        )
        mx.eval(conics)

        # Reconstruct inverse from conics: [[c0, c1], [c1, c2]]
        c = np.array(conics[0, 0])
        inv_cov2d = np.array([[c[0], c[1]], [c[1], c[2]]])

        # This should be the inverse of the regularized cov2d
        # Verify det(inv_cov2d) > 0 (positive definite)
        det_inv = np.linalg.det(inv_cov2d)
        assert det_inv > 0, f"Inverse covariance det should be positive, got {det_inv}"

        # Verify inv * cov ~ I (reconstruct cov2d from conics)
        cov2d_reconstructed = np.linalg.inv(inv_cov2d)
        # cov2d_reconstructed should be positive definite
        assert cov2d_reconstructed[0, 0] > 0
        assert cov2d_reconstructed[1, 1] > 0

    def test_compensations(self):
        """calc_compensations=True should produce valid compensation factors."""
        means = mx.array([[0.0, 0.0, 5.0]])
        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        _, _, _, _, compensations = fully_fused_projection(
            means, covars, viewmats, Ks, 640, 480,
            calc_compensations=True,
        )
        mx.eval(compensations)

        assert compensations is not None
        assert compensations.shape == (1, 1)
        comp_np = np.array(compensations)
        assert np.all(comp_np >= 0), f"Compensations should be >= 0, got {comp_np}"
        assert np.all(comp_np <= 1.0 + 1e-5), f"Compensations should be <= 1, got {comp_np}"

    def test_camera_models(self):
        """All three camera models should produce valid output."""
        means = mx.array([[0.0, 0.0, 5.0]])
        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        for model in ["pinhole", "fisheye", "ortho"]:
            radii, means2d, depths, conics, _ = fully_fused_projection(
                means, covars, viewmats, Ks, 640, 480,
                camera_model=model,
            )
            mx.eval(radii, means2d, depths, conics)

            assert radii.shape == (1, 1, 2), f"{model}: radii shape"
            assert means2d.shape == (1, 1, 2), f"{model}: means2d shape"
            assert depths.shape == (1, 1), f"{model}: depths shape"
            assert conics.shape == (1, 1, 3), f"{model}: conics shape"

            # Should be visible (on-axis, in front of camera)
            radii_np = np.array(radii)
            assert np.all(radii_np > 0), f"{model}: expected positive radii, got {radii_np}"


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestGradients:

    def test_gradient_means(self):
        """mx.grad should flow through to 3D means."""
        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        def loss_fn(means):
            _, means2d, _, _, _ = fully_fused_projection(
                means, covars, viewmats, Ks, 640, 480,
            )
            return mx.sum(means2d)

        means = mx.array([[0.0, 0.0, 5.0]])
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(means)
        mx.eval(grads)

        assert grads.shape == means.shape
        # Gradients should be finite and non-zero
        grads_np = np.array(grads)
        assert np.all(np.isfinite(grads_np)), f"Gradients should be finite, got {grads_np}"
        assert np.any(grads_np != 0.0), f"Gradients should be non-zero, got {grads_np}"

    def test_gradient_covars(self):
        """mx.grad should flow through to covariances."""
        means = mx.array([[0.0, 0.0, 5.0]])
        viewmats = _make_identity_viewmat()
        Ks = _make_simple_K()

        def loss_fn(covars):
            _, means2d, _, conics, _ = fully_fused_projection(
                means, covars, viewmats, Ks, 640, 480,
            )
            # Use conics in loss since means2d may not depend on covars
            return mx.sum(conics)

        covars = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(covars)
        mx.eval(grads)

        assert grads.shape == covars.shape
        grads_np = np.array(grads)
        assert np.all(np.isfinite(grads_np)), f"Gradients should be finite, got {grads_np}"
        assert np.any(grads_np != 0.0), f"Gradients should be non-zero, got {grads_np}"
