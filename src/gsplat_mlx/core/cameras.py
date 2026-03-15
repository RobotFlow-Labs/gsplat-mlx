"""Camera model type definitions for gsplat-mlx.

Mirrors the CameraModel type alias from upstream gsplat.
"""

from typing import Literal

CameraModel = Literal["pinhole", "ortho", "fisheye"]
