"""Render a BuildResult as a realistic mosaic image.

Draws every part's true outline (diagonals, arcs), seam lines between parts,
and studs on plate parts — so the preview is an honest picture of the build.
"""
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .engine import BuildResult, DetailPlacement, SolidPlacement


def _shade(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, v * f))) for v in rgb)


def _draw_stud(d: ImageDraw.ImageDraw, cx: float, cy: float, px: int,
               rgb: Tuple[int, int, int]):
    r = px * 0.31
    d.ellipse([cx - r, cy - r, cx + r, cy + r],
              fill=_shade(rgb, 1.08), outline=_shade(rgb, 0.82), width=max(1, px // 24))
    r2 = r * 0.72
    d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=_shade(rgb, 1.0))


def _draw_piece_polys(d: ImageDraw.ImageDraw, polys, ox: float, oy: float,
                      px: int, rgb, outline_f: float = 0.72):
    lw = max(1, px // 16)
    for poly in polys:
        pts = [((ox + x) * px, (oy + z) * px) for x, z in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=rgb, outline=_shade(rgb, outline_f), width=lw)


def render(result: BuildResult, px: int = 28,
           backing_rgb: Tuple[int, int, int] = (40, 40, 40),
           show_studs: bool = True) -> Image.Image:
    W, H = result.width, result.height
    img = Image.new("RGB", (W * px, H * px), backing_rgb)
    d = ImageDraw.Draw(img)

    # Base layer (visible wherever the art layer leaves gaps).
    if result.bg_grid is not None:
        for y in range(H):
            for x in range(W):
                idx = result.bg_grid[y, x]
                if idx >= 0:
                    rgb = result.palette[idx].rgb
                    d.rectangle([x * px, y * px, (x + 1) * px, (y + 1) * px],
                                fill=rgb)
        # Faint base-plate stud texture in exposed areas is drawn after the
        # art layer below (only gaps remain visible).

    # Art-layer solid fill parts.
    lw = max(1, px // 16)
    for s in result.solids:
        rgb = s.color.rgb
        d.rectangle([s.x * px, s.y * px, (s.x + s.w) * px - 1,
                     (s.y + s.h) * px - 1],
                    fill=rgb, outline=_shade(rgb, 0.72), width=lw)
        if s.studded and show_studs:
            for dy in range(s.h):
                for dx in range(s.w):
                    _draw_stud(d, (s.x + dx + 0.5) * px, (s.y + dy + 0.5) * px,
                               px, rgb)

    # Detail elements: draw bg slot first (as flat color under the shapes),
    # then each piece's true geometry.
    for det in result.details:
        el = det.oel.element
        if el.has_bg:
            bg_rgb = det.slot_colors[el.bg_slot].rgb
            d.rectangle([det.x * px, det.y * px,
                         (det.x + det.oel.w) * px - 1,
                         (det.y + det.oel.h) * px - 1], fill=bg_rgb)
        # Named background regions first (flat, no outline: it's exposed base).
        for piece in det.oel.pieces:
            if piece.part_id is None:
                rgb = det.slot_colors[piece.slot].rgb
                for poly in piece.polys:
                    pts = [((det.x + x) * px, (det.y + z) * px) for x, z in poly]
                    if len(pts) >= 3:
                        d.polygon(pts, fill=rgb)
        for piece in det.oel.pieces:
            if piece.part_id is None:
                continue
            rgb = det.slot_colors[piece.slot].rgb
            _draw_piece_polys(d, piece.polys, det.x, det.y, px, rgb)
            if piece.studded and show_studs:
                # Studs at cell centers fully inside the piece.
                for dy in range(det.oel.h):
                    for dx in range(det.oel.w):
                        cx, cz = dx + 0.5, dy + 0.5
                        if _point_in_polys(piece.polys, cx, cz, margin=0.42):
                            _draw_stud(d, (det.x + cx) * px, (det.y + cz) * px,
                                       px, rgb)

    return img


def _point_in_polys(polys, x: float, z: float, margin: float = 0.0) -> bool:
    """True if the square around (x, z) of half-size `margin` is inside a poly."""
    probes = [(x, z)]
    if margin > 0:
        probes += [(x - margin, z - margin), (x + margin, z - margin),
                   (x - margin, z + margin), (x + margin, z + margin)]
    for poly in polys:
        if all(_point_in_poly(poly, ppx, ppz) for ppx, ppz in probes):
            return True
    return False


def _point_in_poly(poly, x: float, z: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        if (z1 > z) != (z2 > z):
            xin = x1 + (z - z1) / (z2 - z1) * (x2 - x1)
            if x < xin:
                inside = not inside
    return inside


def render_quantized_preview(image_ss: np.ndarray, ss: int,
                             palette, px: int = 12) -> Image.Image:
    """Fast preview: per-stud nearest palette color, no placement engine."""
    from .colors import palette_lab, srgb_to_lab
    Hs, Ws = image_ss.shape[:2]
    H, W = Hs // ss, Ws // ss
    cell = image_ss.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    lab = srgb_to_lab(cell)
    pal = palette_lab(palette)
    d2 = ((lab[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(axis=3)
    idx = d2.argmin(axis=2)
    rgbs = np.array([c.rgb for c in palette], dtype=np.uint8)
    out = rgbs[idx]
    img = Image.fromarray(out).resize((W * px, H * px), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(img)
    grid = (0, 0, 0, 40)
    for x in range(0, W + 1):
        d.line([(x * px, 0), (x * px, H * px)], fill=(0, 0, 0), width=1)
    for y in range(0, H + 1):
        d.line([(0, y * px), (W * px, y * px)], fill=(0, 0, 0), width=1)
    return img
