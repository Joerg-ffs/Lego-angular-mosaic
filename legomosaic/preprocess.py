"""Image preprocessing: crop, tone controls, art styles, resize to the stud grid."""
from dataclasses import dataclass

import cv2
import numpy as np

STYLES = ["Photo", "Poster", "Comic", "Soft", "Pop",
          "Roman Mosaic", "Sketch", "Ink", "Stained Glass", "Neon", "Vintage"]

# Palette preset that pairs naturally with each style (the app auto-selects
# it when the style changes; the user can still override).
STYLE_PALETTES = {
    "Photo": "Full", "Poster": "Full", "Comic": "Classic Bright",
    "Soft": "Full", "Pop": "Pop / Vivid",
    "Roman Mosaic": "Muted / Earth", "Sketch": "Grayscale",
    "Ink": "Grayscale", "Stained Glass": "Pop / Vivid",
    "Neon": "Cool / Ocean", "Vintage": "Portrait / Warm",
}


@dataclass
class ImageAdjust:
    crop_left: float = 0.0     # fractions of the image removed from each side
    crop_right: float = 0.0
    crop_top: float = 0.0
    crop_bottom: float = 0.0
    saturation: float = 1.0    # 0..2
    hue_shift: float = 0.0     # degrees, -180..180
    brightness: float = 0.0    # -100..100
    contrast: float = 1.0      # 0.5..2
    style: str = "Photo"


def apply_adjustments(rgb: np.ndarray, adj: ImageAdjust) -> np.ndarray:
    """Apply crop + tone + style to an RGB uint8 image."""
    h, w = rgb.shape[:2]
    x0 = int(w * np.clip(adj.crop_left, 0, 0.45))
    x1 = w - int(w * np.clip(adj.crop_right, 0, 0.45))
    y0 = int(h * np.clip(adj.crop_top, 0, 0.45))
    y1 = h - int(h * np.clip(adj.crop_bottom, 0, 0.45))
    img = rgb[y0:y1, x0:x1].copy()

    # Hue / saturation in HSV space.
    if adj.saturation != 1.0 or adj.hue_shift != 0.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + adj.hue_shift / 2.0) % 180.0  # OpenCV hue is 0..180
        hsv[..., 1] = np.clip(hsv[..., 1] * adj.saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Brightness / contrast around mid-grey.
    if adj.brightness != 0.0 or adj.contrast != 1.0:
        f = img.astype(np.float32)
        f = (f - 127.5) * adj.contrast + 127.5 + adj.brightness
        img = np.clip(f, 0, 255).astype(np.uint8)

    img = _apply_style(img, adj.style)
    return img


def _apply_style(img: np.ndarray, style: str) -> np.ndarray:
    if style == "Poster":
        # Strong bilateral smoothing + luminance quantization: flat color fields.
        sm = cv2.bilateralFilter(img, 11, 60, 60)
        sm = cv2.bilateralFilter(sm, 11, 60, 60)
        lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
        lab[..., 0] = (lab[..., 0] // 32) * 32 + 16
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if style == "Comic":
        sm = cv2.bilateralFilter(img, 9, 75, 75)
        gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 4)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        boosted = cv2.convertScaleAbs(sm, alpha=1.15, beta=0)
        return (boosted.astype(np.float32) * (0.35 + 0.65 * edges)).astype(np.uint8)
    if style == "Soft":
        sm = cv2.bilateralFilter(img, 15, 40, 40)
        return cv2.GaussianBlur(sm, (0, 0), 1.2)
    if style == "Pop":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        sm = cv2.bilateralFilter(boosted, 11, 80, 80)
        lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
        lab[..., 0] = (lab[..., 0] // 42) * 42 + 21
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def to_grid(img: np.ndarray, width_studs: int, ss: int,
            max_height_studs: int = 160) -> np.ndarray:
    """Resize to (H*ss, W*ss) where W = width_studs and H keeps aspect ratio."""
    h, w = img.shape[:2]
    height_studs = max(4, min(max_height_studs, round(width_studs * h / w)))
    out = cv2.resize(img, (width_studs * ss, height_studs * ss),
                     interpolation=cv2.INTER_AREA)
    return out
