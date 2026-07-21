---
title: LEGO Mosaic Studio
emoji: 🧱
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Turn any image into a buildable LEGO mosaic (.ldr for Studio)
---

# 🧱 LEGO Mosaic Studio

Turn any image into an **artistic, physically-buildable LEGO mosaic** — not a
boring pixel grid. The generator uses real angled and curved LEGO tiles
(triangular tiles, 45°-cut tiles, quarter-rounds, macaroni arcs, S-curves,
wing plates) exactly where they carve a better line than square studs can,
and exports a `.ldr` file that opens directly in **BrickLink Studio**.

This replaces the old Colab notebook prototype (`Claud_Lego_v3.ipynb`).

## Quick start

```
pip install -r requirements.txt
python app.py
```

A browser window opens at http://127.0.0.1:7860:

1. **Upload** an image.
2. **Tune** — crop, saturation/hue/brightness/contrast, art style (Photo /
   Poster / Comic / Soft / Pop), color palette, part types, build thickness,
   detail level, dithering. The live preview updates as you drag.
3. **Build** — you get the full render (true part shapes), a **bill of
   materials**, and downloads for the **`.ldr`** and **`.png`** (also saved to
   `output/`).

Or from Python:

```python
import numpy as np
from PIL import Image
from legomosaic import BuildConfig, full_build

img = np.array(Image.open("photo.jpg").convert("RGB"))
out = full_build(img, BuildConfig(width_studs=64, detail=0.8))
out.image.save("mosaic.png")
open("mosaic.ldr", "w").write(out.ldr_text)
```

## Why the output is physically buildable

- **Two-layer architecture** (default): a full base layer of plates, with the
  art layer of smooth tiles on top. Curved tiles (quarter-rounds, macaroni)
  only cover part of their footprint — the base layer's color shows through
  the gaps, which is exactly how official LEGO Art sets do curves. In 1-layer
  mode, mount the build on a baseplate; gaps show a single backing color.
- **Zero overlaps, full coverage** — enforced by an occupancy grid and
  verified by `test_build.py` on every build configuration.
- **Every part and color is real**: all part geometry was derived from the
  official LDraw part library (kept in `ldraw_ref/` for reference), and all
  colors come from the official `LDConfig.ldr` (solid colors only, with their
  real LDraw codes). One caveat: not every shaped tile has been produced in
  every color — check lot availability on BrickLink before ordering, and use
  the palette picker to restrict colors if needed.
- Heights, rotations and positions in the `.ldr` are consistent (base plates
  at one level, art tiles one plate higher), so Studio shows a clean build
  with no floating or colliding parts.

## How the "artistic" part works

For every possible position of every shaped part (in every rotation), the
engine computes — in CIE Lab color space, at sub-stud resolution, against the
part's *true* geometry — how much visual error the part removes compared to
plain per-stud fill, choosing the optimal palette color for each region of the
part in closed form. A part is only placed when:

1. it wins a real relative improvement (not just "less bad" than squares), and
2. its boundary direction agrees with the local image edge direction
   (structure-tensor check) — this prevents 45° "sawtooth" noise along
   horizontal/vertical edges while lining true diagonals and curves with
   matching parts.

Everything left over is filled with the largest tiles that fit, merged per
color. It's fast: a 96-stud-wide build takes about a second.

## Part vocabulary

| Part | Role |
|---|---|
| 35787 Tile 2x2 Triangular | 45° edges (paired, or single over base) |
| 5091 / 5092 Tile 1x2 Cut 45° | fine 45° edges |
| 24299 / 24307 Wing 2x2, 43723 / 43722 Wing 2x3, 41770 / 41769 Wing 2x4 | shallow angles (~63° / ~72° / ~76°) |
| 25269 Tile 1x1 Corner Round | convex curves |
| 24246 Tile 1x1 Rounded End | bumps / line ends |
| 27925 / 79393 / 27507 Macaroni 2x2 / 3x3 / 4x4 | arc strokes at radius 1–4 studs (inner and outer base colors independent) |
| bullseyes (27507+79393+27925+25269 nested) | concentric multi-color discs |
| 3396 + 2x 25269 | S-curves in a 2x2 |
| 30357 / 30565 Round-Corner Plate 3x3 / 4x4 | big convex corner curves |
| 98138 Tile 1x1 Round, 14769 Tile 2x2 Round | dots / texture |
| SNOT module: 99780 bracket + sideways 1x2 plates/tile | five 0.4-stud color stripes in a 2x3 pocket (2.5x resolution) — deep builds |
| Tiles 1x1…2x4 / Plates 1x1…2x4 | fill and base layer |

## Deep (SNOT) builds and focal-point detail

- **Build thickness "3 deep / SNOT"** adds two structural plate levels below
  the base, making the build one brick deep. That unlocks the SNOT stripe
  module: a Bracket 1x2-1x2 Up anchored in the base carries a chain of
  sideways 1x2 plates whose 8-LDU edges show as five independent color
  stripes, flush with the tile surface — 2.5x the normal resolution where
  gradients and fine banding need it. Every dimension was derived from the
  official meshes and the whole module is stud-clutched (nothing loose).
- **Focal-point detail**: saliency (spectral residual) + face detection
  (OpenCV Haar) + a mild center prior produce a focus map (shown in the app
  after a build). The "Focal detail" slider concentrates shaped parts on the
  subject and calms the background — in testing it moved ~46% more detail
  elements into the focal region at strength 0.8 with no background increase.

## Ordering from GoBricks (brickwith.com)

Enable **GoBricks mode** (checkbox next to the palette picker, or
`BuildConfig(gobricks_only=True)`) to restrict any palette to the colors
GoBricks actually manufactures, verified against the official GoBricks color
chart. LEGO colors GoBricks doesn't make (Light Yellow 18, Very Light
Orange 68, Sand Red 335, Pink, Purple, Rust, Sky Blue, Medium Orange, ...)
are excluded so nothing in the build comes back "Unknown color". The
"GoBricks (brickwith.com)" palette preset shows the full 41-color range,
including six GoBricks-made colors added for it (Coral, Olive Green, Light
Aqua, Lavender, Medium Lavender, Nougat). Every shaped part in the vocabulary
is a mold GoBricks produces (e.g. 25269 = GDS-1307, 5091/5092 =
GDS-90554/90553).

Each build now also exports an **order list (.xml)** in BrickLink wanted-list
format, which brickwith.com's part-list import accepts directly. It uses the
modern purchase ids (`3070`, `3069`, `3068` — not LDraw's legacy `3070b`
etc.) and BrickLink color ids, so parts resolve instead of showing "Unknown
Part". The BOM table shows the same **Buy ID** and **GoBricks color no.**
per lot. Individual lots can still be temporarily out of stock — brickwith
shows live stock at import time, so swap any sold-out lot's color there if
needed.

## Geometry verification

`verify_ldr.py` resolves every part's real mesh from the official LDraw
library and checks a generated `.ldr` for (a) same-layer part overlaps and
(b) base-layer studs sliced by an art part's cut/curved wall — the collision
class BrickLink Studio flags. The engine prevents (b) by construction: any
base cell whose stud would cross a piece boundary (a 45° cut, a wing diagonal,
the seam between paired triangular tiles, a macaroni wall) gets a smooth
**tile** instead of a plate in the base layer, while every art part keeps at
least one full stud to clutch. In 1-layer mode (your own studded baseplate),
elements that would need a studless cell are excluded automatically.
`test_build.py` runs both mesh checks as part of the suite.

## Files

- `app.py` — the web app (Gradio)
- `legomosaic/` — the engine: `colors` (official palette), `parts` (verified
  geometry), `preprocess` (crop/tone/styles), `engine` (placement),
  `render` (true-shape preview), `ldraw` (.ldr export + BOM), `pipeline`
- `test_build.py` — end-to-end test with hard validity checks (coverage,
  overlap, LDR syntax, BOM consistency) — run `python test_build.py`
- `ldraw_ref/` — official LDraw part files used to verify geometry
- `Claud_Lego_v3.ipynb` — the old Colab prototype (kept for history)
