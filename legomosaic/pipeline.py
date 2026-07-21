"""End-to-end pipeline: image + settings -> preview / full build outputs."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .colors import (BY_NAME, PALETTES, LegoColor, gobricks_filter,
                     palette_colors)
from .engine import BuildResult, generate
from .ldraw import bill_of_materials, export_bricklink_xml, export_ldr
from .preprocess import ImageAdjust, apply_adjustments, to_grid
from .render import render, render_quantized_preview


@dataclass
class BuildConfig:
    width_studs: int = 48
    adjust: ImageAdjust = field(default_factory=ImageAdjust)
    palette_names: List[str] = field(default_factory=lambda: PALETTES["Full"])
    layers: int = 2                       # 1 = single, 2 = base + art, 3 = deep SNOT
    backing_name: str = "Black"           # bg color for 1-layer builds
    use_diagonals: bool = True
    use_curves: bool = True
    use_dots: bool = True
    use_wings: bool = True
    use_snot: bool = True                 # SNOT stripe modules (deep builds)
    detail: float = 0.6                   # 0 = only flat, 1 = maximum detail parts
    focus_strength: float = 0.5           # 0 = uniform detail; 1 = strongly
    #                                       concentrate detail on focal points
    plate_finish: bool = False            # fill with studded plates instead of tiles
    dither: float = 0.0                   # 0..1 Floyd-Steinberg strength
    gobricks_only: bool = False           # restrict palette to colors GoBricks
    #                                       makes (orderable at brickwith.com)
    supersample: int = 6                  # sub-stud scoring resolution
    render_px: int = 28


@dataclass
class BuildOutputs:
    result: BuildResult
    image: Image.Image
    ldr_text: str
    bom: List[Dict[str, object]]
    stats: Dict[str, object]
    focus_image: Optional[Image.Image] = None
    bl_xml: str = ""                      # BrickLink wanted-list XML (order list)


def _prepare(image_rgb: np.ndarray, cfg: BuildConfig) -> np.ndarray:
    img = apply_adjustments(image_rgb, cfg.adjust)
    return to_grid(img, cfg.width_studs, cfg.supersample)


def _resolve_palette(cfg: BuildConfig) -> List[LegoColor]:
    names = list(cfg.palette_names) or PALETTES["Full"]
    if cfg.gobricks_only:
        names = gobricks_filter(names) or PALETTES["GoBricks (brickwith.com)"]
    pal = palette_colors(names)
    return pal or palette_colors(PALETTES["Full"])


def quick_preview(image_rgb: np.ndarray, cfg: BuildConfig,
                  px: int = 10) -> Image.Image:
    """Fast palette-quantized preview (no placement engine)."""
    grid = _prepare(image_rgb, cfg)
    pal = _resolve_palette(cfg)
    return render_quantized_preview(grid, cfg.supersample, pal, px=px)


def full_build(image_rgb: np.ndarray, cfg: BuildConfig,
               title: str = "LEGO Mosaic") -> BuildOutputs:
    grid = _prepare(image_rgb, cfg)
    pal = _resolve_palette(cfg)
    backing = BY_NAME.get(cfg.backing_name)
    focus_map = None
    focus_img = None
    if cfg.focus_strength > 0:
        from .focus import compute_focus_map, focus_heatmap_image
        Hs, Ws = grid.shape[:2]
        gh, gw = Hs // cfg.supersample, Ws // cfg.supersample
        adjusted = apply_adjustments(image_rgb, cfg.adjust)
        focus_map = compute_focus_map(adjusted, gh, gw)
        focus_img = focus_heatmap_image(focus_map)
    result = generate(
        grid, cfg.supersample, pal,
        layers=cfg.layers, backing=backing,
        use_diagonals=cfg.use_diagonals, use_curves=cfg.use_curves,
        use_dots=cfg.use_dots, use_wings=cfg.use_wings,
        use_snot=cfg.use_snot,
        detail=cfg.detail, plate_finish=cfg.plate_finish,
        dither=cfg.dither,
        focus_map=focus_map, focus_strength=cfg.focus_strength,
    )
    backing_rgb = backing.rgb if (cfg.layers == 1 and backing) else (40, 40, 40)
    img = render(result, px=cfg.render_px, backing_rgb=backing_rgb)
    ldr = export_ldr(result, title=title)
    bl_xml = export_bricklink_xml(result)
    bom = bill_of_materials(result)
    total = sum(r["Qty"] for r in bom)
    stats = dict(result.stats)
    stats.update({
        "total_parts": total,
        "unique_lots": len(bom),
        "size_studs": f'{result.width} x {result.height}',
        "size_cm": f'{result.width * 0.8:.1f} x {result.height * 0.8:.1f} cm',
    })
    return BuildOutputs(result, img, ldr, bom, stats, focus_img, bl_xml)
