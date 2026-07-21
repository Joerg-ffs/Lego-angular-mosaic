"""Focal-point detection: give the subject more detail than the background.

Builds a per-stud focus map in [0, 1] from three classical-CV signals
(no deep-learning dependencies):
- spectral-residual saliency (Hou & Zhang 2007) — generic "what stands out"
- Haar-cascade face detection (ships with OpenCV) — strong boost on faces
- a mild center prior — subjects are usually near the middle

The engine uses the map to scale the detail-part placement penalty: focal
regions get shaped parts eagerly, backgrounds stay calm with large flat fills.
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def compute_focus_map(img_rgb: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Focus map at stud resolution (grid_h, grid_w), values in [0, 1]."""
    h, w = img_rgb.shape[:2]
    sw = 160
    sh = max(16, round(sw * h / w))
    small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Spectral residual saliency.
    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(fft)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=4)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

    # Face boost (frontal + profile cascades).
    faces = np.zeros_like(sal)
    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sx, sy = sw / w, sh / h
    for cascade_name in ("haarcascade_frontalface_default.xml",
                         "haarcascade_profileface.xml"):
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            dets = cascade.detectMultiScale(gray_full, scaleFactor=1.15,
                                            minNeighbors=6,
                                            minSize=(max(24, w // 12),) * 2)
        except Exception:
            dets = ()
        for (x, y, fw, fh) in dets:
            cv2.ellipse(faces,
                        (int((x + fw / 2) * sx), int((y + fh / 2) * sy)),
                        (int(fw * 0.75 * sx), int(fh * 0.9 * sy)),
                        0, 0, 360, 1.0, -1)
    if faces.any():
        faces = cv2.GaussianBlur(faces, (0, 0), sigmaX=5)
        faces /= faces.max() + 1e-9

    # Mild center prior.
    yy, xx = np.mgrid[0:sh, 0:sw]
    center = np.exp(-(((xx / sw - 0.5) ** 2 + (yy / sh - 0.5) ** 2)
                      / (2 * 0.34 ** 2)))

    f = 0.65 * sal + 1.0 * faces + 0.22 * center.astype(np.float32)
    # Robust normalization: the top ~5% of the map maps to 1.0.
    f = np.clip(f / (np.percentile(f, 95) + 1e-9), 0, 1)
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=3)
    return cv2.resize(f, (grid_w, grid_h), interpolation=cv2.INTER_AREA)


def focus_heatmap_image(focus: np.ndarray, px: int = 10) -> Image.Image:
    """Blue (background) -> red (focal) visualization of the focus map."""
    f = np.clip(focus, 0, 1)
    rgb = np.stack([f * 255, f * 80 + 20, (1 - f) * 220], axis=2).astype(np.uint8)
    h, w = f.shape
    return Image.fromarray(rgb).resize((w * px, h * px), Image.Resampling.NEAREST)
