"""Placement engine.

The core idea that makes the output artistic rather than "an image made of
squares": every detail element (diagonal, curve, wing, dot) is scored by how
much it *reduces real perceptual error* versus just filling its footprint with
per-stud flat colors. The error is computed in CIE Lab at sub-stud resolution
against the element's true part geometry, and the optimal LEGO color for every
color slot at every possible position is found exactly (closed form over the
palette) using convolution sums. Detail parts therefore appear precisely where
the artwork benefits — along contours, curves and color boundaries.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .colors import LegoColor, palette_lab, srgb_to_lab
from .parts import (OrientedElement, PLATE_IDS, TILE_IDS, build_library)


@dataclass
class DetailPlacement:
    oel: OrientedElement
    x: int                       # cell column of top-left corner
    y: int                       # cell row
    slot_colors: List[LegoColor]  # one per slot (bg slot last if present)


@dataclass
class SolidPlacement:
    part_id: str
    x: int
    y: int
    w: int
    h: int
    color: LegoColor
    studded: bool


@dataclass
class BuildResult:
    width: int
    height: int
    details: List[DetailPlacement]
    solids: List[SolidPlacement]          # art-layer fill
    base: List[SolidPlacement]            # visible bg layer (under the art)
    bg_grid: Optional[np.ndarray]         # (H, W) palette idx of base color, or None
    palette: List[LegoColor]
    layers: int                           # 1, 2, or 3 (deep / SNOT)
    fillers_l1: List[SolidPlacement] = field(default_factory=list)  # deep only
    fillers_l2: List[SolidPlacement] = field(default_factory=list)  # deep only
    pocket_grid: Optional[np.ndarray] = None   # (H, W) bool, SNOT pocket cells
    stats: Dict[str, float] = field(default_factory=dict)


def _conv_sums(lab: np.ndarray, lab2: np.ndarray, mask: np.ndarray,
               ss: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-position sums of Lab and |Lab|^2 under `mask`, at cell stride.

    Returns (st[3, Hc, Wc], st2[Hc, Wc], n_pixels) where position (i, j) means
    the mask's top-left corner at cell (row i, col j).
    """
    k = mask.astype(np.float32)
    n = float(k.sum())
    h, w = k.shape
    st = np.stack([
        cv2.filter2D(lab[..., c], -1, k, anchor=(0, 0),
                     borderType=cv2.BORDER_CONSTANT)
        for c in range(3)
    ])
    st2 = cv2.filter2D(lab2, -1, k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    Hs, Ws = lab.shape[:2]
    vy, vx = Hs - h + 1, Ws - w + 1
    return (st[:, :vy:ss, :vx:ss], st2[:vy:ss, :vx:ss], n)


def _best_color(st: np.ndarray, st2: np.ndarray, n: float,
                pal: np.ndarray, allowed: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact optimal palette color per position for one mask.

    cost(color c) = st2 - 2 c . st + n |c|^2, minimized over the palette.
    Returns (min_cost, argmin_idx) arrays shaped like st2.
    """
    idxs = np.arange(len(pal)) if allowed is None else np.asarray(allowed)
    best = np.full(st2.shape, np.inf, dtype=np.float32)
    arg = np.zeros(st2.shape, dtype=np.int32)
    for i in idxs:
        c = pal[i]
        cost = st2 - 2.0 * (c[0] * st[0] + c[1] * st[1] + c[2] * st[2]) \
               + n * float(c @ c)
        better = cost < best
        best[better] = cost[better]
        arg[better] = i
    return best, arg


def _fill_rectangles(color_idx: np.ndarray, free: np.ndarray,
                     part_ids: Dict[Tuple[int, int], str],
                     palette: List[LegoColor], studded: bool
                     ) -> List[SolidPlacement]:
    """Greedy merge of same-color free cells into the largest available parts."""
    H, W = color_idx.shape
    free = free.copy()
    sizes = sorted(part_ids.keys(), key=lambda s: -(s[0] * s[1]))
    out: List[SolidPlacement] = []
    for y in range(H):
        for x in range(W):
            if not free[y, x]:
                continue
            ci = color_idx[y, x]
            for (w, h) in sizes:
                for (pw, ph) in ((w, h), (h, w)) if w != h else ((w, h),):
                    if x + pw <= W and y + ph <= H:
                        block = (free[y:y + ph, x:x + pw]
                                 & (color_idx[y:y + ph, x:x + pw] == ci))
                        if block.all():
                            out.append(SolidPlacement(
                                part_ids[(w, h)], x, y, pw, ph,
                                palette[ci], studded))
                            free[y:y + ph, x:x + pw] = False
                            break
                else:
                    continue
                break
    return out


def _dither_cells(cell_rgb: np.ndarray, occupied: np.ndarray,
                  pal_lab: np.ndarray, strength: float) -> np.ndarray:
    """Floyd-Steinberg dithering at stud resolution over free cells only."""
    H, W = cell_rgb.shape[:2]
    work = cell_rgb.astype(np.float64).copy()
    out = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            px = np.clip(work[y, x], 0, 255)
            lab = srgb_to_lab(px)
            idx = int(np.argmin(((pal_lab - lab) ** 2).sum(axis=1)))
            out[y, x] = idx
            if occupied[y, x]:
                continue  # color decided by a detail part; don't diffuse
            # Diffuse the quantization error (in RGB) to unvisited free cells.
            err = (px - np.array(_PAL_RGB[idx])) * strength
            for dx, dy, f in ((1, 0, 7 / 16), (-1, 1, 3 / 16),
                              (0, 1, 5 / 16), (1, 1, 1 / 16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and not occupied[ny, nx]:
                    work[ny, nx] += err * f
    return out


_PAL_RGB: List[Tuple[int, int, int]] = []


def generate(image_ss: np.ndarray, ss: int, palette: List[LegoColor],
             layers: int = 2,
             backing: Optional[LegoColor] = None,
             use_diagonals: bool = True, use_curves: bool = True,
             use_dots: bool = True, use_wings: bool = True,
             use_snot: bool = True,
             detail: float = 0.6,
             plate_finish: bool = False,
             dither: float = 0.0,
             focus_map: Optional[np.ndarray] = None,
             focus_strength: float = 0.0,
             max_detail_parts: int = 4000) -> BuildResult:
    """Run the full placement pipeline on a preprocessed (H*ss, W*ss, 3) image."""
    global _PAL_RGB
    _PAL_RGB = [c.rgb for c in palette]

    Hs, Ws = image_ss.shape[:2]
    H, W = Hs // ss, Ws // ss
    lab = srgb_to_lab(image_ss.astype(np.float64)).astype(np.float32)
    lab2 = (lab ** 2).sum(axis=2)
    pal = palette_lab(palette).astype(np.float32)

    # ---- baseline: best flat color per cell --------------------------------
    cell = lab.reshape(H, ss, W, ss, 3)
    cell_st = cell.sum(axis=(1, 3)).transpose(2, 0, 1)          # (3, H, W)
    cell_st2 = lab2.reshape(H, ss, W, ss).sum(axis=(1, 3))       # (H, W)
    n_cell = float(ss * ss)
    cell_cost, cell_idx = _best_color(cell_st, cell_st2, n_cell, pal)

    # Integral image of baseline cost for O(1) rectangle sums.
    integ = np.zeros((H + 1, W + 1), dtype=np.float64)
    integ[1:, 1:] = np.cumsum(np.cumsum(cell_cost, axis=0), axis=1)

    # Structure tensor per cell (for the orientation-consistency check):
    # a straight-boundary detail part is only allowed where the local edge
    # direction actually matches its boundary angle.
    gx = cv2.Scharr(lab[..., 0], cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lab[..., 0], cv2.CV_32F, 0, 1)
    def _cellsum(a):
        s = a.reshape(H, ss, W, ss).sum(axis=(1, 3)).astype(np.float64)
        out = np.zeros((H + 1, W + 1))
        out[1:, 1:] = np.cumsum(np.cumsum(s, axis=0), axis=1)
        return out
    i_jxx, i_jxy, i_jyy = _cellsum(gx * gx), _cellsum(gx * gy), _cellsum(gy * gy)

    def _rect(ii, h, w, vy, vx):
        return (ii[h:vy + h, w:vx + w] - ii[:vy, w:vx + w]
                - ii[h:vy + h, :vx] + ii[:vy, :vx])

    # ---- candidate generation ----------------------------------------------
    if layers == 1 and backing is None:
        backing = palette[0]
    backing_idx = None
    if backing is not None:
        backing_idx = next((i for i, c in enumerate(palette)
                            if c.code == backing.code), None)
        if backing_idx is None:
            palette = palette + [backing]
            pal = palette_lab(palette).astype(np.float32)
            backing_idx = len(palette) - 1

    library = build_library(ss, use_diagonals, use_curves, use_dots, use_wings,
                            use_snot=use_snot, deep=(layers == 3))
    if layers == 1:
        # A 1-layer build sits on the user's own (studded) baseplate: elements
        # whose piece boundaries cross stud positions physically collide with
        # those studs and can't be made safe, so they are excluded here.
        library = [oel for oel in library if not oel.smooth_cells.any()]
    # Convert the 0..1 detail slider to a per-piece placement penalty in
    # summed-Lab^2 units (ss^2 pixels per stud; deltaE ~6 per pixel is subtle).
    piece_penalty = (12.0 + (1.0 - detail) * 220.0) * ss * ss

    # Focal-point weighting: the penalty shrinks in focal regions (subject
    # gets shaped parts eagerly) and grows in the background (calm fills).
    focus_integ = None
    focus_raw_integ = None
    if focus_map is not None and focus_strength > 0:
        fmap = np.clip(focus_map, 0, 1)
        fscale = np.clip(1.0 + focus_strength * (1.0 - 2.0 * fmap), 0.12, 2.0)
        focus_integ = np.zeros((H + 1, W + 1))
        focus_integ[1:, 1:] = np.cumsum(np.cumsum(fscale, axis=0), axis=1)
        focus_raw_integ = np.zeros((H + 1, W + 1))
        focus_raw_integ[1:, 1:] = np.cumsum(np.cumsum(fmap, axis=0), axis=1)

    cand_gain: List[np.ndarray] = []
    cand_meta: List[Tuple[OrientedElement, np.ndarray, np.ndarray, np.ndarray]] = []
    for oel in library:
        h, w = oel.h, oel.w
        if h > H or w > W:
            continue
        n_pieces = len(oel.pieces)
        if oel.element.category == "snot":
            # A SNOT module is one pre-designed assembly decision, not seven
            # independent placements — charge it like a two-piece element.
            n_pieces = 2
        total_cost = None
        slot_args = []
        for s in range(oel.masks.shape[0]):
            st, st2, n = _conv_sums(lab, lab2, oel.masks[s], ss)
            if n < 1:  # degenerate mask
                slot_args.append(np.zeros(st2.shape, dtype=np.int32))
                continue
            allowed = None
            if layers == 1 and s in oel.element.bg_slots:
                allowed = np.array([backing_idx])
            cost, arg = _best_color(st, st2, n, pal, allowed)
            total_cost = cost if total_cost is None else total_cost + cost
            slot_args.append(arg)
        vy, vx = total_cost.shape
        base = (integ[h:vy + h, w:vx + w] - integ[:vy, w:vx + w]
                - integ[h:vy + h, :vx] + integ[:vy, :vx])
        if focus_integ is not None:
            fmean = _rect(focus_integ, h, w, vy, vx) / (h * w)
            gain = base - total_cost - piece_penalty * n_pieces * fmean
        else:
            gain = base - total_cost - piece_penalty * n_pieces
        # A detail part must FIT the image well, not merely be "less bad" than
        # flat fill. Require a real relative improvement. This threshold is
        # the placement gatekeeper, so the focal-point weighting acts here
        # too: focal regions accept parts more readily, backgrounds less.
        rel = 0.80 - 0.20 * (1.0 - detail)
        if focus_raw_integ is not None:
            fraw = _rect(focus_raw_integ, h, w, vy, vx) / (h * w)
            rel_arr = np.clip(rel + focus_strength * 0.5 * (fraw - 0.45),
                              0.30, 0.95)
            gain[total_cost > rel_arr * base] = -1.0
        else:
            gain[total_cost > rel * base] = -1.0
        # Orientation consistency: where the image has a clearly directional
        # edge, a straight-boundary part must align with it (within tolerance).
        # This is what stops "sawtooth" — 45-degree tiles invented along a
        # horizontal/vertical boundary that happens to fall mid-cell.
        bdeg = oel.element.boundary_deg
        if bdeg is not None:
            jxx = _rect(i_jxx, h, w, vy, vx)
            jxy = _rect(i_jxy, h, w, vy, vx)
            jyy = _rect(i_jyy, h, w, vy, vx)
            tr = jxx + jyy + 1e-9
            coher = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr
            grad_dir = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
            edge_dir = (grad_dir + 90.0) % 180.0
            elem_dir = (bdeg + 90.0 * oel.k) % 180.0
            diff = np.abs(edge_dir - elem_dir) % 180.0
            diff = np.minimum(diff, 180.0 - diff)
            gain[(coher > 0.5) & (diff > 15.0)] = -1.0
        cand_gain.append(gain)
        cand_meta.append((oel, gain, np.stack(slot_args), None))

    # Flatten candidates with positive gain, sort by gain density.
    entries = []
    for oel, gain, args, _ in cand_meta:
        ys, xs = np.where(gain > 0)
        # Rank by gain / sqrt(area): a middle ground between total gain and
        # per-cell density, so large set pieces (bullseyes, 4x4 curves) can
        # claim their region before a swarm of 1x1 parts fragments it.
        area = float(oel.w * oel.h) ** 0.5
        for y, x in zip(ys, xs):
            entries.append((gain[y, x] / area, gain[y, x], oel, args, y, x))
    entries.sort(key=lambda e: -e[0])

    # ---- greedy placement ---------------------------------------------------
    occupied = np.zeros((H, W), dtype=bool)
    details: List[DetailPlacement] = []
    bg_grid = np.full((H, W), -1, dtype=np.int32) if layers >= 2 else None
    smooth_grid = np.zeros((H, W), dtype=bool)
    pocket_grid = np.zeros((H, W), dtype=bool)
    for _, g, oel, args, y, x in entries:
        if len(details) >= max_detail_parts:
            break
        if occupied[y:y + oel.h, x:x + oel.w].any():
            continue
        slot_colors = [palette[int(args[s, y, x])]
                       for s in range(args.shape[0])]
        details.append(DetailPlacement(oel, x, y, slot_colors))
        occupied[y:y + oel.h, x:x + oel.w] = True
        smooth_grid[y:y + oel.h, x:x + oel.w] |= oel.smooth_cells
        if oel.element.category == "snot":
            # SNOT pockets replace the L2/L3 structure under themselves.
            pocket_grid[y:y + oel.h, x:x + oel.w] = True
        if layers >= 2:
            # Each background slot colors the base plates under its region.
            # If two bg regions share a cell, the one covering more of it wins
            # (a base plate is one color per cell).
            best_cov = np.zeros((oel.h, oel.w))
            for s in oel.element.bg_slots:
                cov = oel.masks[s].reshape(oel.h, ss, oel.w, ss).sum(axis=(1, 3))
                bg_idx = int(args[s, y, x])
                take = cov > best_cov
                for dy, dx in zip(*np.where(take)):
                    bg_grid[y + dy, x + dx] = bg_idx
                best_cov = np.maximum(best_cov, cov)

    # ---- fill the rest with flat parts -------------------------------------
    if dither > 0:
        cell_rgb = image_ss.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
        fill_idx = _dither_cells(cell_rgb, occupied, pal.astype(np.float64), dither)
    else:
        fill_idx = cell_idx
    part_ids = PLATE_IDS if plate_finish else TILE_IDS
    solids = _fill_rectangles(fill_idx, ~occupied, part_ids, palette,
                              studded=plate_finish)

    # ---- base layer (the visible bg plate level under the art) -------------
    base_parts: List[SolidPlacement] = []
    fillers_l1: List[SolidPlacement] = []
    fillers_l2: List[SolidPlacement] = []
    if layers >= 2:
        exposed = bg_grid >= 0
        if exposed.any():
            fill_color = int(np.bincount(bg_grid[exposed]).argmax())
        else:
            fill_color = int(np.bincount(fill_idx.ravel()).argmax())
        base_grid = np.where(exposed, bg_grid, fill_color)
        # Cells where an art piece's boundary crosses the stud get a smooth
        # TILE below instead of a plate — the stud would collide with the
        # piece's cut/curved wall (this is the fix for the Studio-reported
        # intersections on 5091/5092/wings/macaroni and the 35787 pair seam).
        # SNOT pockets bring their own internal structure: no base parts there.
        base_parts = _fill_rectangles(base_grid, ~smooth_grid & ~pocket_grid,
                                      PLATE_IDS, palette, studded=True)
        base_parts += _fill_rectangles(base_grid, smooth_grid & ~pocket_grid,
                                       TILE_IDS, palette, studded=False)
        bg_grid = base_grid
        if layers == 3:
            # Deep build: two structural plate levels below the visible base.
            # L1 covers everything (SNOT brackets clip onto its studs); L2
            # skips pockets (the stripes occupy that space).
            const = np.full((H, W), fill_color, dtype=np.int32)
            fillers_l1 = _fill_rectangles(const, np.ones((H, W), dtype=bool),
                                          PLATE_IDS, palette, studded=True)
            fillers_l2 = _fill_rectangles(const, ~pocket_grid,
                                          PLATE_IDS, palette, studded=True)

    stats = {
        "width": W, "height": H,
        "detail_parts": sum(1 for d in details for p in d.oel.pieces
                            if p.part_id is not None),
        "snot_modules": sum(1 for d in details
                            if d.oel.element.category == "snot"),
        "fill_parts": len(solids),
        "base_parts": len(base_parts) + len(fillers_l1) + len(fillers_l2),
    }
    return BuildResult(W, H, details, solids, base_parts, bg_grid,
                       palette, layers, fillers_l1, fillers_l2,
                       pocket_grid, stats)
