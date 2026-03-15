"""Constants mirrored from upstream gsplat CUDA kernels.

These thresholds control rasterization behavior and must match
the upstream values exactly for numerical compatibility.
"""

# Minimum alpha value for a Gaussian contribution to be considered visible.
# Contributions below this are skipped during rasterization.
ALPHA_THRESHOLD: float = 1.0 / 255.0

# Maximum alpha value clamped during rasterization to prevent
# fully opaque Gaussians from blocking all transmittance.
MAX_ALPHA: float = 0.99

# Minimum transmittance before a pixel is considered fully occluded.
# Once transmittance drops below this, no further Gaussians are composited.
TRANSMITTANCE_THRESHOLD: float = 1e-4

# Kernel density cutoff for the 2D Gaussian evaluation.
# Points beyond this Mahalanobis distance are not rendered.
# Corresponds to exp(-0.5 * cutoff) ≈ 0.0113.
MAX_KERNEL_DENSITY_CUTOFF: float = 0.0113
