"""legomosaic — turn an image into an artistic, physically-buildable LEGO mosaic.

Outputs a rendered preview image and a BrickLink Studio-compatible .ldr file.
"""
from .pipeline import BuildConfig, BuildOutputs, full_build, quick_preview
from .preprocess import ImageAdjust, STYLES
from .colors import ALL_COLORS, PALETTES

__all__ = [
    "BuildConfig", "BuildOutputs", "full_build", "quick_preview",
    "ImageAdjust", "STYLES", "ALL_COLORS", "PALETTES",
]
__version__ = "1.0.0"
