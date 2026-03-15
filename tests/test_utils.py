"""Tests for gsplat_mlx.utils — projection matrix and depth utilities.

Covers:
  - get_projection_matrix: shape, values, near/far plane encoding
  - depth_to_points: shape, origin mapping
  - depth_to_normal: shape, flat-surface normals
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from gsplat_mlx.utils import depth_to_normal, depth_to_points, get_projection_matrix


# ---------------------------------------------------------------------------
# get_projection_matrix
# ---------------------------------------------------------------------------


class TestGetProjectionMatrix:
    """Tests for ``get_projection_matrix``."""

    def test_shape(self) -> None:
        """Output must be a [4, 4] float32 matrix."""
        P = get_projection_matrix(znear=0.01, zfar=100.0, fovX=1.0, fovY=0.8)
        assert P.shape == (4, 4), f"Expected (4, 4), got {P.shape}"
        assert P.dtype == mx.float32

    def test_values_known_fov(self) -> None:
        """Verify specific matrix entries for a 90-degree FOV.

        With fov = pi/2, tan(fov/2) = 1.0, so:
          P[0,0] = 2*znear / (2*tan_half_fovX*znear) = 1 / tan_half_fovX
          P[1,1] = 1 / tan_half_fovY
        """
        fov = math.pi / 2  # 90 degrees -> tan(45deg) = 1.0
        znear, zfar = 0.1, 100.0
        P = get_projection_matrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov)

        # With tan_half = 1.0:
        #   right = tan_half * znear = 0.1
        #   P[0,0] = 2 * znear / (2 * right) = 1.0
        assert abs(P[0, 0].item() - 1.0) < 1e-5, f"P[0,0] = {P[0,0].item()}"
        assert abs(P[1, 1].item() - 1.0) < 1e-5, f"P[1,1] = {P[1,1].item()}"

        # Off-diagonal elements in rows 0 and 1 should be zero
        assert abs(P[0, 1].item()) < 1e-7
        assert abs(P[0, 3].item()) < 1e-7
        assert abs(P[1, 0].item()) < 1e-7
        assert abs(P[1, 3].item()) < 1e-7

        # P[3, 2] = z_sign = 1.0
        assert abs(P[3, 2].item() - 1.0) < 1e-5
        # P[3, 0] = P[3, 1] = P[3, 3] = 0
        assert abs(P[3, 0].item()) < 1e-7
        assert abs(P[3, 1].item()) < 1e-7
        assert abs(P[3, 3].item()) < 1e-7

    def test_near_plane_encoding(self) -> None:
        """P[2,3] should encode -(zfar * znear) / (zfar - znear)."""
        znear, zfar = 0.5, 200.0
        fov = 1.2
        P = get_projection_matrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov)

        expected_p23 = -(zfar * znear) / (zfar - znear)
        assert abs(P[2, 3].item() - expected_p23) < 1e-4, (
            f"P[2,3] = {P[2,3].item()}, expected {expected_p23}"
        )

    def test_far_plane_encoding(self) -> None:
        """P[2,2] should encode z_sign * zfar / (zfar - znear)."""
        znear, zfar = 0.01, 50.0
        fov = 0.9
        P = get_projection_matrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov)

        z_sign = 1.0
        expected_p22 = z_sign * zfar / (zfar - znear)
        assert abs(P[2, 2].item() - expected_p22) < 1e-4, (
            f"P[2,2] = {P[2,2].item()}, expected {expected_p22}"
        )

    def test_symmetric_fov(self) -> None:
        """With symmetric FOV, P[0,2] and P[1,2] should be zero
        (right+left = 0, top+bottom = 0)."""
        P = get_projection_matrix(znear=0.1, zfar=100.0, fovX=1.0, fovY=1.0)
        # (right + left) / (right - left) = 0
        assert abs(P[0, 2].item()) < 1e-7
        assert abs(P[1, 2].item()) < 1e-7


# ---------------------------------------------------------------------------
# depth_to_points
# ---------------------------------------------------------------------------


class TestDepthToPoints:
    """Tests for ``depth_to_points``."""

    @staticmethod
    def _identity_cam_and_K(
        H: int = 4, W: int = 4, fx: float = 10.0
    ) -> tuple[mx.array, mx.array]:
        """Return identity camtoworld and a simple intrinsic matrix."""
        camtoworld = mx.eye(4, dtype=mx.float32)[None]  # [1, 4, 4]
        K = mx.array(
            [[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]],
            dtype=mx.float32,
        )[None]  # [1, 3, 3]
        return camtoworld, K

    def test_shape(self) -> None:
        """Output shape must be [..., H, W, 3]."""
        H, W = 8, 6
        depths = mx.ones((1, H, W, 1))
        camtoworld, K = self._identity_cam_and_K(H, W)
        pts = depth_to_points(depths, camtoworld, K)
        assert pts.shape == (1, H, W, 3), f"Expected (1, {H}, {W}, 3), got {pts.shape}"

    def test_batch_shape(self) -> None:
        """Batch dimensions should be preserved."""
        B, H, W = 2, 4, 4
        depths = mx.ones((B, H, W, 1))
        camtoworld = mx.broadcast_to(mx.eye(4, dtype=mx.float32), (B, 4, 4))
        K = mx.broadcast_to(
            mx.array(
                [[10, 0, 2], [0, 10, 2], [0, 0, 1]], dtype=mx.float32
            ),
            (B, 3, 3),
        )
        pts = depth_to_points(depths, camtoworld, K)
        assert pts.shape == (B, H, W, 3)

    def test_zero_depth_at_origin(self) -> None:
        """With depth=0 and identity camera, all points should be at origin."""
        H, W = 4, 4
        depths = mx.zeros((1, H, W, 1))
        camtoworld, K = self._identity_cam_and_K(H, W)
        pts = depth_to_points(depths, camtoworld, K)
        mx.eval(pts)
        assert mx.allclose(
            pts, mx.zeros_like(pts), atol=1e-6
        ), "Zero depth should yield origin points"

    def test_unit_depth_z_component(self) -> None:
        """With identity camera and z_depth=True, z component should equal depth."""
        H, W = 4, 4
        depth_val = 5.0
        depths = mx.full((1, H, W, 1), depth_val)
        camtoworld, K = self._identity_cam_and_K(H, W, fx=100.0)
        pts = depth_to_points(depths, camtoworld, K, z_depth=True)
        mx.eval(pts)
        # z-component of every point should be depth_val
        z_vals = pts[..., 2]
        assert mx.allclose(
            z_vals, mx.full(z_vals.shape, depth_val), atol=1e-4
        ), f"z-component should be {depth_val}"


# ---------------------------------------------------------------------------
# depth_to_normal
# ---------------------------------------------------------------------------


class TestDepthToNormal:
    """Tests for ``depth_to_normal``."""

    @staticmethod
    def _identity_cam_and_K(
        H: int = 8, W: int = 8, fx: float = 100.0
    ) -> tuple[mx.array, mx.array]:
        camtoworld = mx.eye(4, dtype=mx.float32)[None]
        K = mx.array(
            [[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]],
            dtype=mx.float32,
        )[None]
        return camtoworld, K

    def test_shape(self) -> None:
        """Output shape must match input spatial dims: [..., H, W, 3]."""
        H, W = 8, 8
        depths = mx.ones((1, H, W, 1))
        camtoworld, K = self._identity_cam_and_K(H, W)
        normals = depth_to_normal(depths, camtoworld, K)
        assert normals.shape == (1, H, W, 3), (
            f"Expected (1, {H}, {W}, 3), got {normals.shape}"
        )

    def test_border_zeros(self) -> None:
        """Border pixels should have zero normals (padding from finite diffs)."""
        H, W = 8, 8
        depths = mx.ones((1, H, W, 1)) * 5.0
        camtoworld, K = self._identity_cam_and_K(H, W)
        normals = depth_to_normal(depths, camtoworld, K)
        mx.eval(normals)

        # Top row, bottom row, left col, right col should be zero
        top = normals[0, 0, :, :]
        bottom = normals[0, -1, :, :]
        left = normals[0, :, 0, :]
        right = normals[0, :, -1, :]
        for name, edge in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
            assert mx.allclose(
                edge, mx.zeros_like(edge), atol=1e-6
            ), f"Border normals ({name}) should be zero"

    def test_flat_surface_normals(self) -> None:
        """Constant depth with identity camera should give normals ~ (0, 0, +-1).

        The exact sign depends on cross-product convention (dx cross dy).
        With the identity camera looking along +z, the surface normal of a
        flat plane at constant z-depth should point along the z-axis.
        """
        H, W = 16, 16
        depths = mx.ones((1, H, W, 1)) * 3.0
        camtoworld, K = self._identity_cam_and_K(H, W, fx=200.0)
        normals = depth_to_normal(depths, camtoworld, K)
        mx.eval(normals)

        # Check interior pixels only (exclude 1-pixel border which is zero)
        inner = normals[0, 1:-1, 1:-1, :]  # [H-2, W-2, 3]

        # x and y components should be near zero
        assert mx.mean(mx.abs(inner[..., 0])).item() < 0.05, (
            "Normal x-component should be near zero for flat surface"
        )
        assert mx.mean(mx.abs(inner[..., 1])).item() < 0.05, (
            "Normal y-component should be near zero for flat surface"
        )

        # z component should have magnitude close to 1.0
        z_mag = mx.mean(mx.abs(inner[..., 2])).item()
        assert z_mag > 0.9, (
            f"Normal z-component magnitude should be ~1.0, got {z_mag}"
        )

    def test_batch_dims(self) -> None:
        """Multiple batch dimensions should be supported."""
        B, H, W = 2, 8, 8
        depths = mx.ones((B, H, W, 1)) * 2.0
        camtoworld = mx.broadcast_to(mx.eye(4, dtype=mx.float32), (B, 4, 4))
        K = mx.broadcast_to(
            mx.array(
                [[100, 0, 4], [0, 100, 4], [0, 0, 1]], dtype=mx.float32
            ),
            (B, 3, 3),
        )
        normals = depth_to_normal(depths, camtoworld, K)
        assert normals.shape == (B, H, W, 3)
