"""Part library: real LEGO elements with their true 2D footprint geometry.

Every shape here was derived from the official LDraw part files (see ldraw_ref/),
so the preview render, the placement scoring and the .ldr export all share one
verified single source of truth.

Coordinate convention (element frame): x = studs right (image column),
z = studs down (image row); the element footprint spans [0,w] x [0,h].
A piece's `origin` is where the LDraw part origin sits in that frame, and
`rot` is the part's own rotation (quarter turns clockwise in image space)
relative to its .dat file's identity orientation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
from PIL import Image, ImageDraw

Poly = List[Tuple[float, float]]


def arc(cx: float, cz: float, r: float, a0: float, a1: float, n: int = 28) -> Poly:
    """Arc points from angle a0 to a1 (radians, x-right / z-down frame)."""
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cz + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


@dataclass(frozen=True)
class Piece:
    part_id: Optional[str]       # LDraw filename stem ("35787"); None = a
    #                              background REGION (base shows through) that
    #                              gets its own color slot but emits no part
    slot: int                    # which color slot paints this piece
    origin: Tuple[float, float]  # LDraw origin position, element frame (studs)
    rot: int                     # quarter turns CW vs. the .dat identity pose
    polys: Tuple[Poly, ...]      # covered region (empty tuple = invisible)
    studded: bool = False        # visible studs on top (plates / wings)
    y_ldu: Optional[float] = None  # absolute LDraw Y of the part origin
    #                                (None = the art-tile level); lets SNOT
    #                                modules place parts at intermediate depths
    matrix: Optional[Tuple[int, ...]] = None  # base 3x3 rotation (row-major)
    #                                applied BEFORE the element's Y-rotation;
    #                                used to stand plates on their side (SNOT)


# Rotation standing a plate on its side: local +Y (thickness, top face at
# local y=0) maps to element +z, local +Z (width) maps vertical. The part's
# studs (local y<0) then point toward -z, clipping into the next stripe.
M_SIDEWAYS = (1, 0, 0, 0, 0, -1, 0, 1, 0)


@dataclass(frozen=True)
class Element:
    key: str
    label: str
    w: int
    h: int
    pieces: Tuple[Piece, ...]
    has_bg: bool                 # uncovered area shows the layer below (extra slot)
    category: str                # diagonal | curve | dot | wing | solid
    rotations: Tuple[int, ...] = (0, 1, 2, 3)
    boundary_deg: Optional[float] = None  # straight color-boundary direction
    # (degrees mod 180 in image space at rotation 0; None = curved/no boundary)
    deep_only: bool = False      # requires the deep (SNOT) build thickness

    @property
    def n_slots(self) -> int:
        n = max(p.slot for p in self.pieces) + 1
        return n + 1 if self.has_bg else n

    @property
    def bg_slot(self) -> Optional[int]:
        return self.n_slots - 1 if self.has_bg else None

    @property
    def bg_slots(self) -> Tuple[int, ...]:
        """Every slot that shows the base layer: named bg regions + implicit."""
        real = {p.slot for p in self.pieces if p.part_id is not None}
        named = tuple(sorted({p.slot for p in self.pieces
                              if p.part_id is None} - real))
        return named + ((self.bg_slot,) if self.has_bg else ())


# ---------------------------------------------------------------------------
# Verified base geometry (identity rotation), from the official .dat files.
# ---------------------------------------------------------------------------

# Tile 2x2 Triangular: covers the top-left half, hypotenuse (0,2)->(2,0), 3 LDU chamfer.
TRI_2x2: Poly = [(0, 0), (0, 1.85), (0.15, 1.85), (1.85, 0.15), (1.85, 0)]
# Tile 1x2 Cut Left 45: full left stud + 45deg cut removing the (2,1) corner.
CUT_L_1x2: Poly = [(0, 0), (0, 1), (1, 1), (1.85, 0.15), (1.85, 0)]
# Tile 1x2 Cut Right 45: cut removes the (2,0) corner.
CUT_R_1x2: Poly = [(0, 0), (1, 0), (1.85, 0.85), (1.85, 1), (0, 1)]
# Wing 2x2 Left/Right (plates): shallow diagonal, sliver of bg along one side.
WING22_L: Poly = [(0, 0), (1, 0), (1.925, 1.85), (1.925, 2), (0, 2)]
WING22_R: Poly = [(1, 0), (2, 0), (2, 2), (0.075, 2), (0.075, 1.85)]
# Wing 2x3 Left/Right (no chamfer variant): 2 wide, 3 deep, ~18deg diagonal.
WING23_L: Poly = [(0, 0), (1, 0), (2, 3), (0, 3)]
WING23_R: Poly = [(1, 0), (2, 0), (2, 3), (0, 3)]
# Tile 1x1 Corner Round: quarter disc r=1 centered on the cell's bottom-left corner.
QUARTER_11: Poly = [(0, 0)] + arc(0, 1, 1.0, -math.pi / 2, 0.0)
# Tile 1x1 with Rounded End: rounded end faces up (-z).
HALF_ROUND_11: Poly = ([(0, 1), (0, 0.5)] + arc(0.5, 0.5, 0.5, math.pi, 2 * math.pi)
                       + [(1, 0.5), (1, 1)])
# Tile 1x1 Round.
ROUND_11: Poly = arc(0.5, 0.5, 0.5, 0, 2 * math.pi, 40)
# Tile 2x2 with two curved cutouts (3396): 2x2 minus quarter discs at (0,0) and (2,2).
S_BODY_2x2: Poly = ([(0, 1)] + arc(0, 0, 1.0, math.pi / 2, 0.0)
                    + [(1, 0), (2, 0)] + arc(2, 2, 1.0, -math.pi / 2, -math.pi)
                    + [(1, 2), (0, 2)])
S_CORNER_TL: Poly = [(0, 0)] + arc(0, 0, 1.0, 0.0, math.pi / 2)          # disc at (0,0)
S_CORNER_BR: Poly = [(2, 2)] + arc(2, 2, 1.0, -math.pi, -math.pi / 2)    # disc at (2,2)
# Tile 2x2 Corner Round (macaroni): quarter ring r 1..2 centered bottom-left corner.
MACARONI_2x2: Poly = (arc(0, 2, 2.0, -math.pi / 2, 0.0)
                      + arc(0, 2, 1.0, 0.0, -math.pi / 2))
MAC_INNER_Q: Poly = [(0, 2)] + arc(0, 2, 1.0, -math.pi / 2, 0.0)         # 25269 fits here
# Tile 3x3 / 4x4 Corner Round: quarter rings r 2..3 / 3..4, same convention.
MACARONI_3x3: Poly = (arc(0, 3, 3.0, -math.pi / 2, 0.0)
                      + arc(0, 3, 2.0, 0.0, -math.pi / 2))
MACARONI_4x4: Poly = (arc(0, 4, 4.0, -math.pi / 2, 0.0)
                      + arc(0, 4, 3.0, 0.0, -math.pi / 2))
# Tile 2x2 Round: full disc r=1 centered on the 2x2.
ROUND_22: Poly = arc(1, 1, 1.0, 0, 2 * math.pi, 48)
# Plate 3x3 with 2x2 rounded corner: square minus corner beyond arc r=2 @ (1,1).
RC_33: Poly = ([(0, 0), (3, 0), (3, 1)] + arc(1, 1, 2.0, 0.0, math.pi / 2)
               + [(1, 3), (0, 3)])
# Plate 4x4 with corner round: quarter disc r=4 centered bottom-left corner.
QDISC_44: Poly = [(0, 4)] + arc(0, 4, 4.0, -math.pi / 2, 0.0)
# Wing 2x4 Left/Right (no chamfer): 2 wide, 4 deep, ~76 degree diagonal.
WING24_L: Poly = [(0, 0), (1, 0), (2, 4), (0, 4)]
WING24_R: Poly = [(1, 0), (2, 0), (2, 4), (0, 4)]

# Solid rectangle part ids (full coverage): (w, h) -> part id.
TILE_IDS: Dict[Tuple[int, int], str] = {
    (1, 1): "3070b", (2, 1): "3069b", (3, 1): "63864", (4, 1): "2431",
    (2, 2): "3068b", (3, 2): "26603", (4, 2): "87079",
}
PLATE_IDS: Dict[Tuple[int, int], str] = {
    (1, 1): "3024", (2, 1): "3023", (3, 1): "3623", (4, 1): "3710",
    (2, 2): "3022", (3, 2): "3021", (4, 2): "3020",
}
PART_NAMES: Dict[str, str] = {
    "3070b": "Tile 1 x 1", "3069b": "Tile 1 x 2", "63864": "Tile 1 x 3",
    "2431": "Tile 1 x 4", "3068b": "Tile 2 x 2", "26603": "Tile 2 x 3",
    "87079": "Tile 2 x 4",
    "3024": "Plate 1 x 1", "3023": "Plate 1 x 2", "3623": "Plate 1 x 3",
    "3710": "Plate 1 x 4", "3022": "Plate 2 x 2", "3021": "Plate 2 x 3",
    "3020": "Plate 2 x 4",
    "35787": "Tile 2 x 2 Triangular", "5091": "Tile 1 x 2 Cut Left 45",
    "5092": "Tile 1 x 2 Cut Right 45", "25269": "Tile 1 x 1 Corner Round",
    "24246": "Tile 1 x 1 Rounded End", "98138": "Tile 1 x 1 Round",
    "3396": "Tile 2 x 2 Two Curved Cutouts", "27925": "Tile 2 x 2 Macaroni",
    "24299": "Wing Plate 2 x 2 Left", "24307": "Wing Plate 2 x 2 Right",
    "43723": "Wing Plate 2 x 3 Left", "43722": "Wing Plate 2 x 3 Right",
    "41770": "Wing Plate 2 x 4 Left", "41769": "Wing Plate 2 x 4 Right",
    "79393": "Tile 3 x 3 Macaroni", "27507": "Tile 4 x 4 Macaroni",
    "14769": "Tile 2 x 2 Round", "30357": "Plate 3 x 3 Round Corner",
    "30565": "Plate 4 x 4 Round Corner",
}


def _P(part_id, slot, origin, rot, *polys, studded=False, y_ldu=None, matrix=None):
    return Piece(part_id, slot, origin, rot, tuple(polys), studded, y_ldu, matrix)


DETAIL_ELEMENTS: List[Element] = [
    # --- 45-degree diagonals -------------------------------------------------
    Element("diag22", "2x2 diagonal (two triangular tiles)", 2, 2, (
        _P("35787", 0, (1, 1), 0, TRI_2x2),
        _P("35787", 1, (1, 1), 2, [(2 - x, 2 - z) for x, z in TRI_2x2]),
    ), has_bg=False, category="diagonal", boundary_deg=135.0, rotations=(0, 1)),
    Element("diag22_bg", "2x2 triangular tile over base", 2, 2, (
        _P("35787", 0, (1, 1), 0, TRI_2x2),
    ), has_bg=True, category="diagonal", boundary_deg=135.0),
    Element("cut12L", "1x2 45-cut tile (left)", 2, 1, (
        _P("5091", 0, (1, 0.5), 0, CUT_L_1x2),
    ), has_bg=True, category="diagonal", boundary_deg=135.0),
    Element("cut12R", "1x2 45-cut tile (right)", 2, 1, (
        _P("5092", 0, (1, 0.5), 0, CUT_R_1x2),
    ), has_bg=True, category="diagonal", boundary_deg=45.0),
    # --- shallow-angle wings (plates) ---------------------------------------
    Element("wing22L", "2x2 wing plate (left)", 2, 2, (
        _P("24299", 0, (1, 1), 0, WING22_L, studded=True),
    ), has_bg=True, category="wing", boundary_deg=63.4),
    Element("wing22R", "2x2 wing plate (right)", 2, 2, (
        _P("24307", 0, (1, 1), 0, WING22_R, studded=True),
    ), has_bg=True, category="wing", boundary_deg=116.6),
    Element("wing23L", "2x3 wing plate (left)", 2, 3, (
        _P("43723", 0, (1, 1.5), 0, WING23_L, studded=True),
    ), has_bg=True, category="wing", boundary_deg=71.6),
    Element("wing23R", "2x3 wing plate (right)", 2, 3, (
        _P("43722", 0, (1, 1.5), 0, WING23_R, studded=True),
    ), has_bg=True, category="wing", boundary_deg=108.4),
    # --- curves --------------------------------------------------------------
    Element("quarter11", "1x1 quarter-round tile", 1, 1, (
        _P("25269", 0, (0.5, 0.5), 0, QUARTER_11),
    ), has_bg=True, category="curve"),
    Element("half11", "1x1 rounded-end tile", 1, 1, (
        _P("24246", 0, (0.5, 0.5), 0, HALF_ROUND_11),
    ), has_bg=True, category="curve"),
    Element("scurve22", "2x2 S-curve (3396 + two quarter rounds)", 2, 2, (
        _P("3396", 0, (1, 1), 0, S_BODY_2x2),
        _P("25269", 1, (0.5, 0.5), 1, S_CORNER_TL),
        _P("25269", 2, (1.5, 1.5), 3, S_CORNER_BR),
    ), has_bg=False, category="curve", rotations=(0, 1)),
    # Macaroni arcs get an explicit inner background REGION (part_id None):
    # inner and outer base colors are independent, so an arc stroke can sit on
    # a real color boundary (circle rims) instead of needing one bg color.
    Element("mac22", "2x2 macaroni arc over base", 2, 2, (
        _P("27925", 0, (0.5, 1.5), 0, MACARONI_2x2),
        _P(None, 1, (0, 0), 0, MAC_INNER_Q),
    ), has_bg=True, category="curve"),
    Element("mac22q", "2x2 macaroni + quarter-round center", 2, 2, (
        _P("27925", 0, (0.5, 1.5), 0, MACARONI_2x2),
        _P("25269", 1, (0.5, 1.5), 0, MAC_INNER_Q),
    ), has_bg=True, category="curve"),
    Element("mac33", "3x3 macaroni arc over base", 3, 3, (
        _P("79393", 0, (0, 3), 0, MACARONI_3x3),
        _P(None, 1, (0, 0), 0, [(0, 3)] + arc(0, 3, 2.0, -math.pi / 2, 0.0)),
    ), has_bg=True, category="curve"),
    Element("mac44", "4x4 macaroni arc over base", 4, 4, (
        _P("27507", 0, (0, 4), 0, MACARONI_4x4),
        _P(None, 1, (0, 0), 0, [(0, 4)] + arc(0, 4, 3.0, -math.pi / 2, 0.0)),
    ), has_bg=True, category="curve"),
    Element("bullseye33", "3x3 concentric arcs (3 colors)", 3, 3, (
        _P("79393", 0, (0, 3), 0, MACARONI_3x3),
        _P("27925", 1, (0.5, 2.5), 0, [(x, z + 1) for x, z in MACARONI_2x2]),
        _P("25269", 2, (0.5, 2.5), 0, [(x, z + 1) for x, z in MAC_INNER_Q]),
    ), has_bg=True, category="curve"),
    Element("bullseye44", "4x4 concentric arcs (4 colors)", 4, 4, (
        _P("27507", 0, (0, 4), 0, MACARONI_4x4),
        _P("79393", 1, (0, 4), 0, [(x, z + 1) for x, z in MACARONI_3x3]),
        _P("27925", 2, (0.5, 3.5), 0, [(x, z + 2) for x, z in MACARONI_2x2]),
        _P("25269", 3, (0.5, 3.5), 0, [(x, z + 2) for x, z in MAC_INNER_Q]),
    ), has_bg=True, category="curve"),
    Element("round22", "2x2 round tile over base", 2, 2, (
        _P("14769", 0, (1, 1), 0, ROUND_22),
    ), has_bg=True, category="dot", rotations=(0,)),
    Element("rc33", "3x3 rounded-corner plate", 3, 3, (
        _P("30357", 0, (0.5, 0.5), 0, RC_33, studded=True),
    ), has_bg=True, category="wing"),
    Element("qdisc44", "4x4 quarter-disc plate", 4, 4, (
        _P("30565", 0, (2, 2), 0, QDISC_44, studded=True),
    ), has_bg=True, category="wing"),
    Element("wing24L", "2x4 wing plate (left)", 2, 4, (
        _P("41770", 0, (1, 2), 0, WING24_L, studded=True),
    ), has_bg=True, category="wing", boundary_deg=76.0),
    Element("wing24R", "2x4 wing plate (right)", 2, 4, (
        _P("41769", 0, (1, 2), 0, WING24_R, studded=True),
    ), has_bg=True, category="wing", boundary_deg=104.0),
    # --- dots ----------------------------------------------------------------
    Element("dot11", "1x1 round tile over base", 1, 1, (
        _P("98138", 0, (0.5, 0.5), 0, ROUND_11),
    ), has_bg=True, category="dot", rotations=(0,)),
    # --- SNOT stripe pocket (deep builds only) -------------------------------
    # 2x3 pocket, verified dimension-by-dimension from the official meshes:
    # a Bracket 1x2-1x2 Up (99780) clips onto the base layer in the back row
    # (its base at the L2 plate level, y=-16); its sideways studs (center
    # y=-22, exactly 10 LDU below the deep-mode art surface at -32) carry a
    # chain of three sideways 1x2 plates and one sideways 1x2 tile, whose top
    # edges sit flush with the art surface and whose bottoms rest exactly on
    # the base plates' stud tops (-12). Result: five 0.4-stud color stripes
    # (bracket flange edge + 4 stripes), 2.5x the normal resolution, plus a
    # normally-tiled back cell hiding the anchor. All parts fully clutched.
    Element("snot5", "2x3 SNOT pocket (5 sideways stripes)", 2, 3, (
        _P("3069b", 0, (1, 0.2), 0, [(0, 0.2), (2, 0.2), (2, 0.6), (0, 0.6)],
           y_ldu=-22, matrix=M_SIDEWAYS),
        _P("3023", 1, (1, 0.6), 0, [(0, 0.6), (2, 0.6), (2, 1.0), (0, 1.0)],
           y_ldu=-22, matrix=M_SIDEWAYS),
        _P("3023", 2, (1, 1.0), 0, [(0, 1.0), (2, 1.0), (2, 1.4), (0, 1.4)],
           y_ldu=-22, matrix=M_SIDEWAYS),
        _P("3023", 3, (1, 1.4), 0, [(0, 1.4), (2, 1.4), (2, 1.8), (0, 1.8)],
           y_ldu=-22, matrix=M_SIDEWAYS),
        _P("99780", 3, (1, 2.5), 0, [(0, 1.8), (2, 1.8), (2, 2.0), (0, 2.0)],
           y_ldu=-16),
        _P("3023", 4, (1, 2.5), 0, y_ldu=-24),      # hidden cap plate
        _P("3069b", 4, (1, 2.5), 0, [(0, 2), (2, 2), (2, 3), (0, 3)]),
    ), has_bg=False, category="snot", deep_only=True),
]


def rotate_point(x: float, z: float, w: int, h: int, k: int) -> Tuple[float, float]:
    """Rotate a point k quarter-turns clockwise inside a w x h footprint.

    Returns coordinates in the rotated footprint (which is h x w for odd k).
    """
    k %= 4
    if k == 0:
        return x, z
    if k == 1:
        return h - z, x
    if k == 2:
        return w - x, h - z
    return z, w - x


def rotate_poly(poly: Poly, w: int, h: int, k: int) -> Poly:
    return [rotate_point(x, z, w, h, k) for x, z in poly]


@dataclass(frozen=True)
class OrientedElement:
    """An element in a specific rotation, with rasterized slot masks."""
    element: Element
    k: int                      # quarter turns CW
    w: int
    h: int
    pieces: Tuple[Piece, ...]   # geometry already rotated into this orientation
    masks: np.ndarray           # (n_slots, h*ss, w*ss) bool — scoring/render masks
    ss: int
    smooth_cells: np.ndarray = None  # (h, w) bool: base cell below must be a
    # smooth TILE (no stud), because a piece boundary crosses that stud

    @property
    def key(self) -> str:
        return f"{self.element.key}@{self.k}"


def point_in_poly(poly: Poly, x: float, z: float) -> bool:
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


def _stud_conflict_cells(pieces: Sequence[Piece], w: int, h: int) -> np.ndarray:
    """Which cells must sit on a smooth (studless) base.

    A stud (radius ~0.3 stud at each cell center) is fine when it lies fully
    under ONE piece's plastic (normal hollow-underside connection) or fully in
    open background (it just shows). But when a piece's cut/curved boundary
    crosses the stud, the stud collides with that wall — the physical bug that
    Studio flags. Sample radius 0.28 leaves ~0.4 LDU of mesh tolerance, which
    is what real parts like the 1x1 quarter tile rely on when nesting.
    """
    R = 0.28
    samples = [(0.0, 0.0)] + [(R * math.cos(a), R * math.sin(a))
                              for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    real = [p for p in pieces if p.part_id is not None]
    out = np.zeros((h, w), dtype=bool)
    for cy in range(h):
        for cx in range(w):
            px, pz = cx + 0.5, cy + 0.5
            per_piece = []
            for p in real:
                hits = sum(1 for dx, dz in samples
                           if any(point_in_poly(poly, px + dx, pz + dz)
                                  for poly in p.polys))
                per_piece.append(hits)
            total = len(samples)
            fully_inside_one = any(hits == total for hits in per_piece)
            fully_outside_all = all(hits == 0 for hits in per_piece)
            out[cy, cx] = not (fully_inside_one or fully_outside_all)
    return out


def _rasterize(polys: Sequence[Poly], w: int, h: int, ss: int) -> np.ndarray:
    img = Image.new("1", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(img)
    for poly in polys:
        pts = [(x * ss, z * ss) for x, z in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def orient_element(el: Element, k: int, ss: int) -> OrientedElement:
    w, h = (el.w, el.h) if k % 2 == 0 else (el.h, el.w)
    pieces = []
    piece_masks: List[np.ndarray] = []
    for p in el.pieces:
        polys = tuple(rotate_poly(poly, el.w, el.h, k) for poly in p.polys)
        origin = rotate_point(*p.origin, el.w, el.h, k)
        pieces.append(Piece(p.part_id, p.slot, origin, (p.rot + k) % 4, polys,
                            p.studded, p.y_ldu, p.matrix))
        piece_masks.append(_rasterize(polys, w, h, ss))
    n_slots = el.n_slots
    masks = np.zeros((n_slots, h * ss, w * ss), dtype=bool)
    for p, m in zip(pieces, piece_masks):
        masks[p.slot] |= m
    # Make slots strictly disjoint (arc rasterization can overlap by a pixel):
    # earlier slots win.
    claimed = np.zeros((h * ss, w * ss), dtype=bool)
    for s in range(n_slots - 1 if el.has_bg else n_slots):
        masks[s] &= ~claimed
        claimed |= masks[s]
    if el.has_bg:
        masks[el.bg_slot] = ~claimed
    else:
        # Full-coverage element: fold rasterization slack (chamfer seams,
        # arc gaps) into slot 0 so the masks exactly partition the footprint.
        masks[0] |= ~claimed
    if el.category == "snot":
        # SNOT pockets manage their own vertical structure: the stripe bottoms
        # rest exactly on the base plates' stud tops (verified geometry), so
        # the per-cell stud analysis (which assumes flat tiles) doesn't apply.
        smooth = np.zeros((h, w), dtype=bool)
    else:
        smooth = _stud_conflict_cells(pieces, w, h)
    return OrientedElement(el, k, w, h, tuple(pieces), masks, ss, smooth)


def build_library(ss: int,
                  use_diagonals: bool = True,
                  use_curves: bool = True,
                  use_dots: bool = True,
                  use_wings: bool = True,
                  use_snot: bool = False,
                  deep: bool = False) -> List[OrientedElement]:
    """All enabled detail elements in all their rotations, rasterized at ss."""
    out: List[OrientedElement] = []
    enabled = {
        "diagonal": use_diagonals, "curve": use_curves,
        "dot": use_dots, "wing": use_wings,
        "snot": use_snot and deep,
    }
    for el in DETAIL_ELEMENTS:
        if not enabled.get(el.category, False):
            continue
        if el.deep_only and not deep:
            continue
        for k in el.rotations:
            out.append(orient_element(el, k, ss))
    return out
