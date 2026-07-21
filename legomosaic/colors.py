"""LEGO color definitions, verified against the official LDraw LDConfig.ldr (2026-05-29 update).

Every color here is a real, solid (non-transparent, non-pearl) LDraw color whose
code BrickLink Studio understands. RGB values are the official LDraw values.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LegoColor:
    code: int          # LDraw color code (what goes into the .ldr file)
    name: str
    rgb: Tuple[int, int, int]


# Verified from LDConfig.ldr — solid colors only.
ALL_COLORS: List[LegoColor] = [
    LegoColor(0, "Black", (27, 42, 52)),
    LegoColor(1, "Blue", (30, 90, 168)),
    LegoColor(2, "Green", (0, 133, 43)),
    LegoColor(3, "Dark Turquoise", (6, 157, 159)),
    LegoColor(4, "Red", (180, 0, 0)),
    LegoColor(5, "Dark Pink", (211, 53, 157)),
    LegoColor(10, "Bright Green", (88, 171, 65)),
    LegoColor(11, "Light Turquoise", (0, 170, 164)),
    LegoColor(13, "Pink", (246, 169, 187)),
    LegoColor(14, "Yellow", (250, 200, 10)),
    LegoColor(15, "White", (244, 244, 244)),
    LegoColor(18, "Light Yellow", (255, 214, 127)),
    LegoColor(19, "Tan", (215, 186, 140)),
    LegoColor(22, "Purple", (103, 31, 129)),
    LegoColor(25, "Orange", (214, 121, 35)),
    LegoColor(26, "Magenta", (144, 31, 118)),
    LegoColor(27, "Lime", (165, 202, 24)),
    LegoColor(28, "Dark Tan", (137, 125, 98)),
    LegoColor(29, "Bright Pink", (255, 158, 205)),
    LegoColor(30, "Medium Lavender", (160, 110, 185)),
    LegoColor(31, "Lavender", (205, 164, 222)),
    LegoColor(68, "Very Light Orange", (253, 195, 131)),
    LegoColor(70, "Reddish Brown", (95, 49, 9)),
    LegoColor(71, "Light Bluish Grey", (150, 150, 150)),
    LegoColor(72, "Dark Bluish Grey", (100, 100, 100)),
    LegoColor(73, "Medium Blue", (115, 150, 200)),
    LegoColor(74, "Medium Green", (127, 196, 117)),
    LegoColor(77, "Light Pink", (254, 204, 207)),
    LegoColor(78, "Light Nougat", (255, 201, 149)),
    LegoColor(84, "Medium Nougat", (170, 125, 85)),
    LegoColor(92, "Nougat", (187, 128, 90)),
    LegoColor(85, "Medium Lilac", (68, 26, 145)),
    LegoColor(191, "Bright Light Orange", (252, 172, 0)),
    LegoColor(212, "Bright Light Blue", (157, 195, 247)),
    LegoColor(216, "Rust", (135, 43, 23)),
    LegoColor(226, "Bright Light Yellow", (255, 236, 108)),
    LegoColor(232, "Sky Blue", (119, 201, 216)),
    LegoColor(272, "Dark Blue", (25, 50, 90)),
    LegoColor(288, "Dark Green", (0, 69, 26)),
    LegoColor(308, "Dark Brown", (53, 33, 0)),
    LegoColor(320, "Dark Red", (114, 0, 18)),
    LegoColor(321, "Dark Azure", (70, 155, 195)),
    LegoColor(322, "Medium Azure", (104, 195, 226)),
    LegoColor(323, "Light Aqua", (211, 242, 234)),
    LegoColor(326, "Yellowish Green", (226, 249, 154)),
    LegoColor(330, "Olive Green", (119, 119, 78)),
    LegoColor(335, "Sand Red", (136, 96, 94)),
    LegoColor(353, "Coral", (255, 109, 119)),
    LegoColor(378, "Sand Green", (112, 142, 124)),
    LegoColor(379, "Sand Blue", (112, 129, 154)),
    LegoColor(462, "Medium Orange", (245, 134, 36)),
    LegoColor(484, "Dark Orange", (145, 80, 28)),
]

BY_NAME: Dict[str, LegoColor] = {c.name: c for c in ALL_COLORS}
BY_CODE: Dict[int, LegoColor] = {c.code: c for c in ALL_COLORS}

# ---------------------------------------------------------------------------
# GoBricks (brickwith.com) availability.
# LDraw code -> (GoBricks color no., BrickLink color id), from the official
# GoBricks color chart (solid colors only; pearl/metallic/trans excluded).
# Colors NOT in this table (e.g. 18 Light Yellow, 68 Very Light Orange,
# 335 Sand Red) are real LEGO colors that GoBricks does not produce.
# ---------------------------------------------------------------------------
GOBRICKS: Dict[int, Tuple[str, int]] = {
    0: ("080", 11),    # Black
    1: ("050", 7),     # Blue
    2: ("040", 6),     # Green
    3: ("243", 39),    # Dark Turquoise
    4: ("010", 5),     # Red
    5: ("012", 47),    # Dark Pink
    10: ("043", 36),   # Bright Green
    14: ("030", 3),    # Yellow
    15: ("090", 1),    # White
    19: ("031", 2),    # Tan
    25: ("021", 4),    # Orange
    26: ("013", 71),   # Magenta
    27: ("042", 34),   # Lime
    28: ("034", 69),   # Dark Tan
    29: ("011", 104),  # Bright Pink
    30: ("062", 157),  # Medium Lavender
    31: ("063", 154),  # Lavender
    70: ("081", 88),   # Reddish Brown
    71: ("071", 86),   # Light Bluish Grey
    72: ("072", 85),   # Dark Bluish Grey
    73: ("052", 42),   # Medium Blue
    78: ("032", 90),   # Light Nougat
    84: ("084", 150),  # Medium Nougat
    85: ("060", 89),   # Medium Lilac (BL: Dark Purple)
    92: ("038", 28),   # Nougat
    191: ("036", 110),  # Bright Light Orange
    212: ("053", 105),  # Bright Light Blue
    226: ("033", 103),  # Bright Light Yellow
    272: ("055", 63),   # Dark Blue
    288: ("047", 80),   # Dark Green
    308: ("082", 120),  # Dark Brown
    320: ("014", 59),   # Dark Red
    321: ("051", 153),  # Dark Azure
    322: ("046", 156),  # Medium Azure
    323: ("045", 152),  # Light Aqua
    326: ("044", 158),  # Yellowish Green
    330: ("049", 155),  # Olive Green
    353: ("018", 220),  # Coral
    378: ("048", 48),   # Sand Green
    379: ("054", 55),   # Sand Blue
    484: ("083", 68),   # Dark Orange
}

# BrickLink color ids for the remaining (non-GoBricks) colors, so the
# order-list export still works with an unrestricted palette.
_BRICKLINK_EXTRA: Dict[int, int] = {
    11: 40,    # Light Turquoise
    13: 23,    # Pink
    18: 33,    # Light Yellow
    22: 24,    # Purple
    68: 96,    # Very Light Orange
    74: 37,    # Medium Green
    77: 56,    # Light Pink
    216: 27,   # Rust
    232: 87,   # Sky Blue
    335: 58,   # Sand Red
    462: 31,   # Medium Orange
}
BRICKLINK_COLOR_ID: Dict[int, int] = {
    **{code: bl for code, (_, bl) in GOBRICKS.items()}, **_BRICKLINK_EXTRA}


def gobricks_filter(names: Sequence[str]) -> List[str]:
    """Restrict a list of color names to colors GoBricks manufactures."""
    return [n for n in names if n in BY_NAME and BY_NAME[n].code in GOBRICKS]

# Named palette presets (subsets by color name).
PALETTES: Dict[str, List[str]] = {
    "Full": [c.name for c in ALL_COLORS],
    "GoBricks (brickwith.com)": [c.name for c in ALL_COLORS
                                 if c.code in GOBRICKS],
    "Classic Bright": ["Black", "White", "Red", "Blue", "Yellow", "Green",
                       "Orange", "Light Bluish Grey", "Dark Bluish Grey", "Tan",
                       "Bright Green", "Dark Blue", "Dark Red", "Reddish Brown"],
    "Portrait / Warm": ["Black", "White", "Light Nougat", "Medium Nougat", "Tan",
                        "Dark Tan", "Very Light Orange", "Reddish Brown", "Dark Brown",
                        "Dark Orange", "Rust", "Sand Red", "Light Bluish Grey",
                        "Dark Bluish Grey", "Light Yellow", "Dark Red"],
    "Cool / Ocean": ["Black", "White", "Blue", "Dark Blue", "Medium Blue",
                     "Bright Light Blue", "Sky Blue", "Medium Azure", "Dark Azure",
                     "Dark Turquoise", "Light Turquoise", "Sand Blue",
                     "Light Bluish Grey", "Dark Bluish Grey"],
    "Grayscale": ["Black", "Dark Bluish Grey", "Light Bluish Grey", "White"],
    "Muted / Earth": ["Black", "White", "Tan", "Dark Tan", "Sand Green", "Sand Blue",
                      "Sand Red", "Reddish Brown", "Dark Brown", "Medium Nougat",
                      "Dark Green", "Dark Blue", "Dark Red", "Light Bluish Grey",
                      "Dark Bluish Grey", "Rust"],
    "Pop / Vivid": ["Black", "White", "Red", "Orange", "Yellow", "Lime",
                    "Bright Green", "Dark Turquoise", "Medium Azure", "Blue",
                    "Dark Pink", "Magenta", "Bright Pink", "Purple",
                    "Bright Light Orange"],
}


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert float sRGB array (..., 3) in [0,255] to CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def palette_colors(names: Sequence[str]) -> List[LegoColor]:
    return [BY_NAME[n] for n in names if n in BY_NAME]


def palette_lab(colors: Sequence[LegoColor]) -> np.ndarray:
    return srgb_to_lab(np.array([c.rgb for c in colors], dtype=np.float64))
