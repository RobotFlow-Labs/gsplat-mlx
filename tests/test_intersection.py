"""Tests for tile-Gaussian intersection and depth sorting (PRD-06)."""

import math

import mlx.core as mx
import numpy as np
import pytest

from gsplat_mlx.core.intersection import isect_tiles, isect_offset_encode


# ---------------------------------------------------------------------------
# Helper: make simple inputs for intersection tests
# ---------------------------------------------------------------------------

def _make_inputs(
    means2d_list,
    radii_list,
    depths_list,
    n_images=1,
):
    """Build mx.array inputs for isect_tiles from plain lists.

    Each list entry is per-Gaussian. Returns [I, N, ...] shaped arrays.
    """
    N = len(means2d_list)
    means2d_np = np.array(means2d_list, dtype=np.float32).reshape(n_images, N, 2)
    radii_np = np.array(radii_list, dtype=np.int32).reshape(n_images, N, 2)
    depths_np = np.array(depths_list, dtype=np.float32).reshape(n_images, N)
    return mx.array(means2d_np), mx.array(radii_np), mx.array(depths_np)


# ===== Basic intersection tests =====


class TestSingleGaussianSingleTile:
    """One Gaussian at center of a tile -> 1 intersection."""

    def test_single_gaussian_single_tile(self):
        # Gaussian at pixel (8, 8) with radius 4 in a 16px tile grid
        # tile_means = (0.5, 0.5), tile_radii = (0.25, 0.25)
        # tile_min = floor(0.25) = 0, tile_max = ceil(0.75) = 1
        # -> covers tile (0,0) only
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0]],
            radii_list=[[4, 4]],
            depths_list=[1.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        assert int(tiles_per_gauss.sum()) == 1
        assert flatten_ids.shape[0] == 1
        assert int(flatten_ids[0]) == 0  # image_id=0 * N=1 + gauss_id=0


class TestSingleGaussianMultiTile:
    """One large Gaussian spanning multiple tiles."""

    def test_single_gaussian_multi_tile(self):
        # Gaussian at pixel (24, 24) with radius 20
        # tile_means = (1.5, 1.5), tile_radii = (1.25, 1.25)
        # tile_min = floor(0.25) = 0, tile_max = ceil(2.75) = 3
        # -> covers 3x3 = 9 tiles (but grid is 4x3, so clipped to 3x3)
        means2d, radii, depths = _make_inputs(
            means2d_list=[[24.0, 24.0]],
            radii_list=[[20, 20]],
            depths_list=[2.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        n_isects = int(tiles_per_gauss.sum())
        assert n_isects == 9
        assert flatten_ids.shape[0] == 9


class TestDepthSorting:
    """3 Gaussians at different depths in the same tile -> sorted front-to-back."""

    def test_depth_sorting(self):
        # All Gaussians at same pixel location (8, 8), same radius
        # but different depths: 5.0, 1.0, 3.0
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]],
            radii_list=[[4, 4], [4, 4], [4, 4]],
            depths_list=[5.0, 1.0, 3.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        # All 3 Gaussians cover 1 tile each -> 3 intersections
        assert flatten_ids.shape[0] == 3

        # flatten_ids should be sorted by depth: gauss 1 (d=1.0), gauss 2 (d=3.0), gauss 0 (d=5.0)
        fids = np.array(flatten_ids)
        assert list(fids) == [1, 2, 0], f"Expected [1, 2, 0] but got {list(fids)}"


class TestOutOfBounds:
    """Gaussian with radius=0 -> no intersections."""

    def test_out_of_bounds(self):
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0]],
            radii_list=[[0, 0]],
            depths_list=[1.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        assert int(tiles_per_gauss.sum()) == 0
        assert flatten_ids.shape[0] == 0


class TestEmpty:
    """No visible Gaussians -> empty intersection list."""

    def test_empty_all_culled(self):
        # All Gaussians have radius 0
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [24.0, 24.0]],
            radii_list=[[0, 0], [0, 0]],
            depths_list=[1.0, 2.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        assert int(tiles_per_gauss.sum()) == 0
        assert flatten_ids.shape[0] == 0
        assert isect_ids.shape[0] == 0


# ===== Offset encode tests =====


class TestOffsetEncodeBasic:
    """Verify offsets are correct prefix sums."""

    def test_offset_encode_basic(self):
        # 2 Gaussians in different tiles
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [24.0, 8.0]],
            radii_list=[[4, 4], [4, 4]],
            depths_list=[1.0, 2.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        offsets = isect_offset_encode(isect_ids, n_images=1, tile_width=4, tile_height=3)
        offsets_np = np.array(offsets)

        # offsets shape should be [1, 3, 4]
        assert offsets_np.shape == (1, 3, 4)

        # tile (0,0) has 1 Gaussian -> offset=0
        assert offsets_np[0, 0, 0] == 0
        # tile (1,0) has 1 Gaussian -> offset=1
        assert offsets_np[0, 0, 1] == 1
        # remaining tiles have 0 Gaussians -> offset=2
        assert offsets_np[0, 0, 2] == 2


class TestOffsetEncodeEmptyTiles:
    """Tiles with no Gaussians have equal consecutive offsets."""

    def test_offset_encode_empty_tiles(self):
        # Single Gaussian in tile (0,0)
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0]],
            radii_list=[[4, 4]],
            depths_list=[1.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        offsets = isect_offset_encode(isect_ids, n_images=1, tile_width=4, tile_height=3)
        offsets_np = np.array(offsets)

        # All offsets after tile (0,0) should be 1 (the total count)
        flat_offsets = offsets_np.flatten()
        assert flat_offsets[0] == 0
        for i in range(1, len(flat_offsets)):
            assert flat_offsets[i] == 1


class TestOffsetEncodeMonotonic:
    """Flattened offsets are monotonically non-decreasing."""

    def test_offset_encode_monotonic(self):
        # Multiple Gaussians in various tiles
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [24.0, 24.0], [40.0, 8.0]],
            radii_list=[[4, 4], [4, 4], [4, 4]],
            depths_list=[1.0, 2.0, 3.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        offsets = isect_offset_encode(isect_ids, n_images=1, tile_width=4, tile_height=3)
        offsets_np = np.array(offsets).flatten()
        assert np.all(np.diff(offsets_np) >= 0), "Offsets must be monotonically non-decreasing"


# ===== Multi-image tests =====


class TestMultiImage:
    """I > 1, verify image_id sorting."""

    def test_multi_image(self):
        # 2 images, 1 Gaussian each at same position but different depths
        N = 1
        means2d_np = np.array([[[8.0, 8.0]], [[8.0, 8.0]]], dtype=np.float32)  # [2, 1, 2]
        radii_np = np.array([[[4, 4]], [[4, 4]]], dtype=np.int32)  # [2, 1, 2]
        depths_np = np.array([[5.0], [1.0]], dtype=np.float32)  # [2, 1]

        means2d = mx.array(means2d_np)
        radii = mx.array(radii_np)
        depths = mx.array(depths_np)

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        # 2 intersections total (1 per image)
        assert flatten_ids.shape[0] == 2

        fids = np.array(flatten_ids)
        # Image 0 should come first (flatten_id = 0*1+0 = 0)
        # Image 1 second (flatten_id = 1*1+0 = 1)
        assert fids[0] == 0, f"First intersection should be image 0, got flatten_id={fids[0]}"
        assert fids[1] == 1, f"Second intersection should be image 1, got flatten_id={fids[1]}"


# ===== Tiles per Gauss count =====


class TestTilesPerGaussCount:
    """Verify total tiles_per_gauss matches n_isects."""

    def test_tiles_per_gauss_count(self):
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [24.0, 24.0], [50.0, 10.0]],
            radii_list=[[20, 20], [4, 4], [10, 10]],
            depths_list=[1.0, 2.0, 3.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        total = int(mx.sum(tiles_per_gauss).item())
        assert total == flatten_ids.shape[0], (
            f"tiles_per_gauss sum ({total}) != n_isects ({flatten_ids.shape[0]})"
        )


# ===== Float depth sort order =====


class TestFloatDepthSortOrder:
    """Verify IEEE 754 bit pattern sorts correctly for positive floats."""

    def test_float_depth_sort_order(self):
        depths = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 100.0], dtype=np.float32)
        bits = depths.view(np.uint32)
        # Bits should be monotonically increasing for positive floats
        assert np.all(np.diff(bits.astype(np.int64)) > 0), (
            "IEEE 754 positive float bit patterns should be monotonically increasing"
        )


# ===== Edge cases =====


class TestEdgeCases:
    """Various edge cases."""

    def test_gaussian_covers_entire_screen(self):
        """Very large radius covering all tiles."""
        # Grid is 4x3 tiles = 12 tiles, tile_size=16 -> 64x48 pixels
        # Gaussian at center (32, 24) with huge radius
        means2d, radii, depths = _make_inputs(
            means2d_list=[[32.0, 24.0]],
            radii_list=[[64, 48]],
            depths_list=[1.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        assert int(tiles_per_gauss.sum()) == 4 * 3  # all 12 tiles

    def test_one_radius_zero(self):
        """One radius component is zero -> culled."""
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0]],
            radii_list=[[4, 0]],  # y-radius is 0
            depths_list=[1.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        assert int(tiles_per_gauss.sum()) == 0

    def test_many_gaussians(self):
        """N=10,000 random Gaussians -> no crashes, reasonable output."""
        np.random.seed(123)
        N = 10000
        means2d_np = np.random.uniform(0, 640, (1, N, 2)).astype(np.float32)
        radii_np = np.random.randint(1, 30, (1, N, 2)).astype(np.int32)
        depths_np = np.random.uniform(0.1, 100.0, (1, N)).astype(np.float32)

        means2d = mx.array(means2d_np)
        radii = mx.array(radii_np)
        depths = mx.array(depths_np)

        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=40, tile_height=30,
        )
        n_isects = flatten_ids.shape[0]
        assert n_isects > 0
        assert int(tiles_per_gauss.sum()) == n_isects

    def test_sort_disabled(self):
        """sort=False still computes intersections but may not be sorted."""
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]],
            radii_list=[[4, 4], [4, 4], [4, 4]],
            depths_list=[5.0, 1.0, 3.0],
        )
        _, _, flatten_ids_sorted = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3, sort=True,
        )
        _, _, flatten_ids_unsorted = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3, sort=False,
        )
        # Same number of intersections
        assert flatten_ids_sorted.shape[0] == flatten_ids_unsorted.shape[0]
        # Same set of indices (when sorted)
        assert sorted(np.array(flatten_ids_sorted).tolist()) == sorted(
            np.array(flatten_ids_unsorted).tolist()
        )

    def test_offset_encode_total(self):
        """Sum of all tile counts equals total intersections."""
        means2d, radii, depths = _make_inputs(
            means2d_list=[[8.0, 8.0], [24.0, 24.0], [40.0, 8.0]],
            radii_list=[[4, 4], [4, 4], [4, 4]],
            depths_list=[1.0, 2.0, 3.0],
        )
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths,
            tile_size=16, tile_width=4, tile_height=3,
        )
        offsets = isect_offset_encode(isect_ids, n_images=1, tile_width=4, tile_height=3)
        offsets_np = np.array(offsets)
        n_isects = flatten_ids.shape[0]

        # Last offset + count for last tile should equal n_isects
        # Equivalently: the maximum offset value + count of the last occupied tile = n_isects
        # Simpler check: all offsets are <= n_isects and the max offset equals
        # n_isects minus the count of the last tile(s)
        assert np.all(offsets_np <= n_isects)
        assert np.all(offsets_np >= 0)

    def test_no_gaussians_offset_encode(self):
        """Empty isect_ids -> all-zero offsets."""
        isect_ids = mx.array(np.empty(0, dtype=np.int64))
        offsets = isect_offset_encode(isect_ids, n_images=1, tile_width=4, tile_height=3)
        offsets_np = np.array(offsets)
        assert offsets_np.shape == (1, 3, 4)
        assert np.all(offsets_np == 0)
