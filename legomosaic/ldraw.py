"""LDraw (.ldr) export — opens directly in BrickLink Studio, LDView, LeoCAD.

Conventions used (single source of truth shared with parts.py):
- Image column -> LDraw X, image row -> LDraw Z, one stud = 20 LDU.
- +Y is down in LDraw. Plates/tiles are 8 LDU tall with their origin at the top
  face, so a part resting on the table sits at y = -8, and a part on top of a
  base plate layer sits at y = -16.
- A quarter-turn clockwise (in image space) is Ry(-90):  [0 0 -1 / 0 1 0 / 1 0 0].
"""
from collections import Counter
from typing import Dict, List, Tuple

from .colors import BRICKLINK_COLOR_ID, GOBRICKS, LegoColor
from .engine import BuildResult, SolidPlacement
from .parts import PART_NAMES, PLATE_IDS, TILE_IDS

# LDraw still uses the historical b-suffixed ids for a few molds; LEGO,
# BrickLink and GoBricks/brickwith all use the modern suffix-free ids.
BUY_IDS: Dict[str, str] = {"3070b": "3070", "3069b": "3069", "3068b": "3068"}


def buy_id(part_id: str) -> str:
    """The id to use when ordering (BrickLink / brickwith / GoBricks)."""
    return BUY_IDS.get(part_id, part_id)

ROT_MATRICES = {
    0: (1, 0, 0, 0, 1, 0, 0, 0, 1),
    1: (0, 0, -1, 0, 1, 0, 1, 0, 0),
    2: (-1, 0, 0, 0, 1, 0, 0, 0, -1),
    3: (0, 0, 1, 0, 1, 0, -1, 0, 0),
}

# BrickLink Studio ships pre-official versions of a few parts whose geometry
# was later re-oriented in the official LDraw release. Verified by direct mesh
# comparison (Studio's ldraw/UnOfficial/5091.dat vs official 2025-03): the
# 45-degree cut tiles are modeled 90 degrees off in Studio. Value = extra
# quarter turns CW to add at emission so the file looks right in Studio.
# (In LDView/LeoCAD, which use the official library, these two parts will
# appear rotated 90 degrees until Studio and LDraw converge.)
STUDIO_ROT_FIX: Dict[str, int] = {"5091": 1, "5092": 1}

# part id -> canonical (w, h) footprint as defined in the .dat file
_CANONICAL: Dict[str, Tuple[int, int]] = {}
for _ids in (TILE_IDS, PLATE_IDS):
    for (w, h), pid in _ids.items():
        _CANONICAL[pid] = (w, h)


def _line(color: int, x: float, y: float, z: float, k: int, part_id: str) -> str:
    m = ROT_MATRICES[k]
    coords = " ".join(f"{v:g}" for v in (x, y, z))
    mat = " ".join(str(v) for v in m)
    return f"1 {color} {coords} {mat} {part_id}.dat"


def export_ldr(result: BuildResult, title: str = "LEGO Mosaic") -> str:
    W, H = result.width, result.height
    cx, cz = W * 10.0, H * 10.0  # model center offset (LDU)
    art_y = {1: -8.0, 2: -16.0, 3: -32.0}[result.layers]
    base_y = {2: -8.0, 3: -24.0}.get(result.layers)
    desc = {1: "single layer (mount on a baseplate)",
            2: "two layers (plate base + tile art)",
            3: "deep SNOT build (4 levels: 2 structural + base + tile art)"}

    out: List[str] = [
        f"0 {title}",
        "0 Name: mosaic.ldr",
        "0 Author: legomosaic generator",
        f"0 // {W} x {H} studs, {desc[result.layers]}",
        "0 BFC CERTIFY CCW",
        "",
    ]

    if result.layers == 3:
        out.append("0 // ---- Structural filler levels 1-2 (plates) ----")
        for s in result.fillers_l1:
            out.append(_emit_solid(s, cx, cz, y=-8.0))
        for s in result.fillers_l2:
            out.append(_emit_solid(s, cx, cz, y=-16.0))
        out.append("")
    if base_y is not None:
        out.append("0 // ---- Base layer (plates) ----")
        for s in result.base:
            out.append(_emit_solid(s, cx, cz, y=base_y))
        out.append("")

    out.append("0 // ---- Art layer: fill parts ----")
    for s in result.solids:
        out.append(_emit_solid(s, cx, cz, y=art_y))
    out.append("")

    out.append("0 // ---- Art layer: detail parts ----")
    for det in result.details:
        for piece in det.oel.pieces:
            if piece.part_id is None:
                continue  # background region: base layer shows, no part
            ox, oz = piece.origin
            x = (det.x + ox) * 20.0 - cx
            z = (det.y + oz) * 20.0 - cz
            color = det.slot_colors[piece.slot].code
            k = (piece.rot + STUDIO_ROT_FIX.get(piece.part_id, 0)) % 4
            y = art_y if piece.y_ldu is None else float(piece.y_ldu)
            if piece.matrix is not None:
                # Sideways part: element rotation composed with its base pose.
                import numpy as np
                m = (np.array(ROT_MATRICES[k]).reshape(3, 3)
                     @ np.array(piece.matrix).reshape(3, 3))
                mat = " ".join(str(int(v)) for v in m.ravel())
                coords = " ".join(f"{v:g}" for v in (x, y, z))
                out.append(f"1 {color} {coords} {mat} {piece.part_id}.dat")
            else:
                out.append(_line(color, x, y, z, k, piece.part_id))
    out.append("")
    return "\n".join(out)


def _emit_solid(s: SolidPlacement, cx: float, cz: float, y: float) -> str:
    w0, h0 = _CANONICAL[s.part_id]
    k = 0 if (s.w, s.h) == (w0, h0) else 1
    x = (s.x + s.w / 2.0) * 20.0 - cx
    z = (s.y + s.h / 2.0) * 20.0 - cz
    return _line(s.color.code, x, y, z, k, s.part_id)


def bill_of_materials(result: BuildResult) -> List[Dict[str, object]]:
    """Aggregate part counts: one row per (part, color)."""
    counter: Counter = Counter()
    for s in (result.base + result.solids
              + result.fillers_l1 + result.fillers_l2):
        counter[(s.part_id, s.color.code, s.color.name)] += 1
    for det in result.details:
        for piece in det.oel.pieces:
            if piece.part_id is None:
                continue
            c = det.slot_colors[piece.slot]
            counter[(piece.part_id, c.code, c.name)] += 1
    rows = []
    for (pid, code, cname), n in sorted(counter.items(),
                                        key=lambda kv: (-kv[1], kv[0])):
        gob = GOBRICKS.get(code)
        rows.append({
            "Part": pid,
            "Buy ID": buy_id(pid),
            "Part name": PART_NAMES.get(pid, pid),
            "Color": cname,
            "LDraw code": code,
            "GoBricks color": gob[0] if gob else "—",
            "Qty": n,
        })
    return rows


def _bom_counter(result: BuildResult) -> Counter:
    counter: Counter = Counter()
    for s in (result.base + result.solids
              + result.fillers_l1 + result.fillers_l2):
        counter[(s.part_id, s.color.code)] += 1
    for det in result.details:
        for piece in det.oel.pieces:
            if piece.part_id is None:
                continue
            counter[(piece.part_id, det.slot_colors[piece.slot].code)] += 1
    return counter


def export_bricklink_xml(result: BuildResult) -> str:
    """BrickLink wanted-list XML — importable at brickwith.com and BrickLink.

    Uses modern purchase ids (3070, not 3070b) and BrickLink color ids, which
    both sites resolve. Colors with no BrickLink mapping are skipped (none
    exist when the GoBricks palette is used).
    """
    lines = ["<INVENTORY>"]
    for (pid, code), n in sorted(_bom_counter(result).items()):
        bl_color = BRICKLINK_COLOR_ID.get(code)
        if bl_color is None:
            continue
        lines.append(
            "  <ITEM>"
            f"<ITEMTYPE>P</ITEMTYPE><ITEMID>{buy_id(pid)}</ITEMID>"
            f"<COLOR>{bl_color}</COLOR><MINQTY>{n}</MINQTY>"
            "</ITEM>")
    lines.append("</INVENTORY>")
    return "\n".join(lines)
