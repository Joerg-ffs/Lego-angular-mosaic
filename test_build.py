"""Headless end-to-end test + physical-validity checks."""
import sys
import numpy as np
from PIL import Image, ImageDraw

from legomosaic import BuildConfig, full_build
from legomosaic.parts import orient_element, DETAIL_ELEMENTS


def make_test_image(w=480, h=360):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    # vertical gradient sky
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(30 + 120 * t), int(60 + 80 * t), 200))
    # sun (circle -> curves)
    d.ellipse([w * 0.6, h * 0.1, w * 0.85, h * 0.45], fill=(250, 200, 30))
    # mountain (diagonals)
    d.polygon([(0, h), (w * 0.45, h * 0.25), (w * 0.8, h)], fill=(70, 60, 80))
    d.polygon([(w * 0.4, h), (w * 0.75, h * 0.45), (w, h)], fill=(110, 90, 100))
    # foreground band
    d.rectangle([0, h * 0.85, w, h], fill=(30, 90, 40))
    return np.array(img)


def validate(result):
    """Hard checks: full coverage, no overlaps, every cell owned exactly once."""
    H, W = result.height, result.width
    art = np.zeros((H, W), dtype=int)
    for s in result.solids:
        art[s.y:s.y + s.h, s.x:s.x + s.w] += 1
    for det in result.details:
        art[det.y:det.y + det.oel.h, det.x:det.x + det.oel.w] += 1
    assert (art <= 1).all(), f"art layer overlap! max={art.max()}"
    assert (art == 1).all(), f"art layer gaps! {int((art == 0).sum())} cells uncovered"
    pocket = (result.pocket_grid if result.pocket_grid is not None
              else np.zeros((H, W), dtype=bool))

    def cover(parts):
        g = np.zeros((H, W), dtype=int)
        for s in parts:
            g[s.y:s.y + s.h, s.x:s.x + s.w] += 1
        return g

    if result.layers >= 2:
        base = cover(result.base)
        assert (base[~pocket] == 1).all(), "base layer gaps/overlaps"
        assert (base[pocket] == 0).all(), "base parts inside SNOT pocket"
    if result.layers == 3:
        l1 = cover(result.fillers_l1)
        assert (l1 == 1).all(), "L1 filler must cover everything"
        l2 = cover(result.fillers_l2)
        assert (l2[~pocket] == 1).all() and (l2[pocket] == 0).all(), "L2 filler bad"
    print("  validity: full coverage, zero overlap, all levels OK")


def validate_ldr(text):
    n = 0
    for line in text.splitlines():
        if line.startswith("1 "):
            t = line.split()
            assert len(t) == 15, f"bad LDR line: {line}"
            [float(v) for v in t[2:14]]
            assert t[14].endswith(".dat")
            n += 1
    print(f"  ldr: {n} part lines, all syntactically valid")
    return n


def validate_masks():
    """Every oriented element's masks must exactly partition its footprint."""
    for el in DETAIL_ELEMENTS:
        for k in el.rotations:
            oel = orient_element(el, k, 6)
            total = oel.masks.sum(axis=0)
            assert (total == 1).all(), f"{el.key}@{k}: masks don't partition"
    print("  masks: all elements partition their footprints exactly")


def make_gradient_image(w=480, h=480):
    """Smooth gradients + a focal object: exercises SNOT stripes and focus."""
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):  # smooth sunset gradient (SNOT stripe territory)
        t = y / h
        d.line([(0, y), (w, y)],
               fill=(int(40 + 200 * t), int(30 + 120 * t), int(90 + 40 * t)))
    d.ellipse([w * 0.3, h * 0.25, w * 0.7, h * 0.65], fill=(250, 220, 80))
    return np.array(img)


if __name__ == "__main__":
    validate_masks()
    img = make_test_image()
    for layers in (2, 1):
        print(f"\n=== layers={layers} ===")
        cfg = BuildConfig(width_studs=48, layers=layers, detail=0.7)
        out = full_build(img, cfg)
        print("  stats:", out.stats)
        validate(out.result)
        nparts = validate_ldr(out.ldr_text)
        assert nparts == out.stats["total_parts"], "BOM total != LDR part count"
        out.image.save(f"output/test_render_layers{layers}.png")
        open(f"output/test_mosaic_layers{layers}.ldr", "w").write(out.ldr_text)

    print(f"\n=== layers=3 (deep / SNOT) + focus ===")
    cfg = BuildConfig(width_studs=48, layers=3, detail=0.8, focus_strength=0.6)
    out = full_build(make_gradient_image(), cfg)
    print("  stats:", out.stats)
    validate(out.result)
    nparts = validate_ldr(out.ldr_text)
    assert nparts == out.stats["total_parts"], "BOM total != LDR part count"
    out.image.save("output/test_render_layers3.png")
    open("output/test_mosaic_layers3.ldr", "w").write(out.ldr_text)

    print("\n=== GoBricks mode (brickwith.com orderability) ===")
    import xml.etree.ElementTree as ET
    from legomosaic.colors import GOBRICKS
    cfg = BuildConfig(width_studs=48, gobricks_only=True, detail=0.7)
    out = full_build(make_gradient_image(), cfg)
    codes = {r["LDraw code"] for r in out.bom}
    assert codes <= set(GOBRICKS), f"non-GoBricks colors used: {codes - set(GOBRICKS)}"
    assert not any(str(r["Buy ID"]).endswith("b") for r in out.bom), \
        "legacy b-suffix id leaked into Buy ID"
    root = ET.fromstring(out.bl_xml)
    items = root.findall("ITEM")
    assert len(items) == len(out.bom), "order-list lots != BOM lots"
    assert sum(int(i.find("MINQTY").text) for i in items) \
        == sum(r["Qty"] for r in out.bom), "order-list qty != BOM qty"
    validate(out.result)
    print(f"  {out.stats['total_parts']} parts, {len(out.bom)} lots — "
          "all colors GoBricks-orderable, XML order list consistent")

    print("\n=== mesh-level geometry verification (real LDraw meshes) ===")
    from verify_ldr import check, check_studs
    for layers in (2, 1, 3):
        path = f"output/test_mosaic_layers{layers}.ldr"
        print(f"--- {path} ---")
        assert not check(path), "mesh overlap!"
        assert not check_studs(path), "stud/wall conflict!"
    print("\nALL CHECKS PASSED")
