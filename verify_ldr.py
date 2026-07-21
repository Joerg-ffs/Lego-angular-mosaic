"""Ground-truth collision check for generated .ldr files.

Resolves every part's REAL mesh from the official LDraw library (recursively,
including primitives, downloading missing files into ldraw_ref/), transforms it
into world space, and rasterizes each part's XZ footprint per layer to find
genuine part-vs-part overlaps. Also dumps a top-down debug render of the true
geometry so it can be compared against the app's own preview.
"""
import math
import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REF = Path(__file__).parent / "ldraw_ref"
BASE_URLS = [
    "https://library.ldraw.org/library/official/parts/",
    "https://library.ldraw.org/library/official/p/",
    "https://library.ldraw.org/library/unofficial/parts/",
    "https://library.ldraw.org/library/unofficial/p/",
]


def _fetch(name: str) -> Path:
    """Locate (or download) an LDraw file by its reference name."""
    rel = name.lower().replace("\\", "/")
    local = REF / rel
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    for base in BASE_URLS:
        url = base + rel
        r = subprocess.run(["curl", "-s", "-f", "--retry", "3", "-o",
                            str(local), url], capture_output=True)
        if r.returncode == 0 and local.exists() and local.stat().st_size > 0:
            return local
    raise FileNotFoundError(f"could not resolve LDraw file {name}")


@lru_cache(maxsize=None)
def load_tris(name: str):
    """All triangles of a part (recursively resolved), in the part's own frame.

    Returns an (N, 3, 3) float array of triangle vertices.
    """
    path = _fetch(name)
    tris = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        t = raw.split()
        if not t:
            continue
        if t[0] == "3":
            v = np.array([float(x) for x in t[2:11]]).reshape(3, 3)
            tris.append(v)
        elif t[0] == "4":
            v = np.array([float(x) for x in t[2:14]]).reshape(4, 3)
            tris.append(v[[0, 1, 2]])
            tris.append(v[[0, 2, 3]])
        elif t[0] == "1":
            pos = np.array([float(x) for x in t[2:5]])
            M = np.array([float(x) for x in t[5:14]]).reshape(3, 3)
            sub = load_tris(t[14])
            if len(sub):
                tris.append((sub @ M.T + pos).reshape(-1, 3, 3))
    if not tris:
        return np.zeros((0, 3, 3))
    flat = [x.reshape(-1, 3, 3) if x.ndim > 2 else x[None] for x in tris]
    return np.concatenate(flat, axis=0)


def body_tris(name: str):
    """A part's mesh with its studs removed, in the part's LOCAL frame.

    Studs interpenetrate the mating part's hollows by design (that's the
    connection), so collision checks must run on stud-free bodies. Standard
    parts: everything at local y <= 0 (studs + top face; walls and bottom
    still carry the full footprint). The 99780 bracket needs its own rule:
    its flange legitimately lives above y=0 — only its two stud groups
    (flange studs at z < -14, base studs y in [-4,0] around |z| < 8) go.
    """
    t = load_tris(name)
    stem = name.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if stem == "99780.dat":
        drop = (np.all(t[:, :, 2] < -13.9, axis=1)
                | (np.all(t[:, :, 1] > -4.01, axis=1)
                   & np.all(t[:, :, 1] < 0.01, axis=1)
                   & np.all(np.abs(t[:, :, 2]) < 8.0, axis=1)))
        return t[~drop]
    return t[~np.all(t[:, :, 1] <= 0.01, axis=1)]


def stud_tris_local(name: str):
    """Only a part's VERTICAL top studs, in its local frame.

    The 99780 bracket's flange studs point sideways — they are the SNOT
    connection, verified by the module geometry and the cross-level check,
    not by the vertical stud-coverage rule.
    """
    t = load_tris(name)
    stem = name.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if stem == "99780.dat":
        keep = (np.all(t[:, :, 1] > -4.01, axis=1)
                & np.all(t[:, :, 1] < 0.01, axis=1)
                & np.all(np.abs(t[:, :, 2]) < 8.0, axis=1)
                & (t[:, :, 1].min(axis=1) < -0.5))
        return t[keep]
    keep = (np.all(t[:, :, 1] <= 0.01, axis=1)
            & (t[:, :, 1].min(axis=1) >= -4.01)
            & (t[:, :, 1].min(axis=1) < -0.5))
    return t[keep]


def parse_ldr(path):
    """Yield (color, pos, M, part_name, line_no) for each type-1 line.

    Emitted files target BrickLink Studio, whose 5091/5092 geometry is rotated
    90 degrees vs. the official library this verifier uses — undo that here so
    the checks reflect what Studio actually displays (see ldraw.STUDIO_ROT_FIX).
    """
    from legomosaic.ldraw import ROT_MATRICES, STUDIO_ROT_FIX
    for i, raw in enumerate(open(path, encoding="utf-8", errors="replace")):
        t = raw.split()
        if t and t[0] == "1":
            color = int(t[1])
            pos = np.array([float(x) for x in t[2:5]])
            M = np.array([float(x) for x in t[5:14]]).reshape(3, 3)
            stem = t[14].rsplit(".", 1)[0]
            fix = STUDIO_ROT_FIX.get(stem, 0)
            if fix:
                inv = np.array(ROT_MATRICES[(4 - fix) % 4],
                               dtype=float).reshape(3, 3)
                M = M @ inv
            yield color, pos, M, t[14], i + 1


def check(path, px_per_ldu=2.0, shrink=0.75, min_overlap_ldu2=2.0,
          debug_png=None, ignore_studs=True):
    """Detect real overlaps between parts that live in the same layer,
    plus cross-level intrusions from sideways (SNOT) parts."""
    parts = []
    for color, pos, M, name, ln in parse_ldr(path):
        tris = body_tris(name).copy()
        world = tris @ M.T + pos
        parts.append(dict(name=name, pos=pos, ln=ln, world=world, color=color,
                          layer=round(pos[1])))
    if not parts:
        print("no parts found")
        return []

    all_pts = np.concatenate([p["world"].reshape(-1, 3) for p in parts])
    x0, z0 = all_pts[:, 0].min() - 4, all_pts[:, 2].min() - 4
    x1, z1 = all_pts[:, 0].max() + 4, all_pts[:, 2].max() + 4
    Wp = int((x1 - x0) * px_per_ldu) + 1
    Hp = int((z1 - z0) * px_per_ldu) + 1

    layers = {}
    for idx, p in enumerate(parts):
        layers.setdefault(p["layer"], []).append(idx)

    overlaps = []
    dbg = Image.new("RGB", (Wp, Hp), (255, 255, 255)) if debug_png else None
    ddbg = ImageDraw.Draw(dbg) if dbg else None

    for layer, idxs in sorted(layers.items()):
        owner = np.full((Hp, Wp), -1, dtype=np.int32)
        conflict = {}
        for idx in idxs:
            p = parts[idx]
            w = p["world"]  # already stud-free via body_tris()
            img = Image.new("1", (Wp, Hp), 0)
            d = ImageDraw.Draw(img)
            for tri in w:
                pts = [((v[0] - x0) * px_per_ldu, (v[2] - z0) * px_per_ldu)
                       for v in tri]
                d.polygon(pts, fill=1)
            m = np.array(img, dtype=np.uint8)
            # Erode = true inward offset, so parts merely sharing an edge
            # never register; only genuine interpenetration survives.
            import cv2
            it = max(1, int(round(shrink * px_per_ldu)))
            m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=it).astype(bool)
            hit = m & (owner >= 0)
            if hit.any():
                for other in np.unique(owner[hit]):
                    area = float((hit & (owner == other)).sum()) / px_per_ldu ** 2
                    if area >= min_overlap_ldu2:
                        conflict[(int(other), idx)] = area
            owner[m & (owner < 0)] = idx
            if ddbg is not None:
                col = tuple(int(v) for v in np.random.default_rng(idx).integers(60, 220, 3))
                for tri in w:
                    pts = [((v[0] - x0) * px_per_ldu, (v[2] - z0) * px_per_ldu)
                           for v in tri]
                    ddbg.polygon(pts, outline=col)
        for (a, b), area in conflict.items():
            pa, pb = parts[a], parts[b]
            overlaps.append((pa, pb, area))
            print(f"OVERLAP layer y={layer}: {pa['name']} (line {pa['ln']}) x "
                  f"{pb['name']} (line {pb['ln']})  area={area:.1f} LDU^2  "
                  f"at {pa['pos'][::2]} / {pb['pos'][::2]}")

    # ---- cross-level pass: sideways / multi-level parts (SNOT) -------------
    # Special parts (origin off the 8-LDU grid, or the bracket) are compared
    # against every part whose body genuinely shares a vertical interval with
    # them (> 1 LDU): both parts' geometry is restricted STRICTLY to that
    # shared interval (so parts merely stacked on each other never register),
    # bbox-filled, eroded, and intersected.
    special = [i for i, p in enumerate(parts)
               if (p["layer"] % 8 != 0) or p["name"].lower().startswith("99780")]
    if special:
        import cv2
        ranges = []
        boxes = []
        for p in parts:
            ys = p["world"][:, :, 1]
            ranges.append((ys.min(), ys.max()))
            boxes.append((p["world"][:, :, 0].min(), p["world"][:, :, 0].max(),
                          p["world"][:, :, 2].min(), p["world"][:, :, 2].max()))

        def interval_bbox_mask(i, lo, hi):
            w = parts[i]["world"]
            ys = w[:, :, 1]
            sel = w[(ys.min(axis=1) < hi - 0.05) & (ys.max(axis=1) > lo + 0.05)]
            if not len(sel):
                return None
            img = Image.new("1", (Wp, Hp), 0)
            d = ImageDraw.Draw(img)
            xs, zs = sel[:, :, 0].ravel(), sel[:, :, 2].ravel()
            d.rectangle([(xs.min() - x0) * px_per_ldu,
                         (zs.min() - z0) * px_per_ldu,
                         (xs.max() - x0) * px_per_ldu,
                         (zs.max() - z0) * px_per_ldu], fill=1)
            m = cv2.erode(np.array(img, dtype=np.uint8),
                          np.ones((3, 3), np.uint8),
                          iterations=max(1, int(round(shrink * px_per_ldu))))
            return m.astype(bool)

        seen = set()
        for i in special:
            for j in range(len(parts)):
                if j == i or (min(i, j), max(i, j)) in seen:
                    continue
                lo = max(ranges[i][0], ranges[j][0])
                hi = min(ranges[i][1], ranges[j][1])
                if hi - lo <= 1.0:
                    continue  # stacked/touching, not sharing vertical space
                bi, bj = boxes[i], boxes[j]
                if (bi[1] < bj[0] - 2 or bj[1] < bi[0] - 2
                        or bi[3] < bj[2] - 2 or bj[3] < bi[2] - 2):
                    continue  # far apart horizontally
                mi = interval_bbox_mask(i, lo, hi)
                mj = interval_bbox_mask(j, lo, hi)
                if mi is None or mj is None:
                    continue
                area = float((mi & mj).sum()) / px_per_ldu ** 2
                if area >= 3.0:
                    seen.add((min(i, j), max(i, j)))
                    pa, pb = parts[i], parts[j]
                    overlaps.append((pa, pb, area))
                    print(f"CROSS-LEVEL OVERLAP y=[{lo:.0f},{hi:.0f}]: "
                          f"{pa['name']} (line {pa['ln']}) x {pb['name']} "
                          f"(line {pb['ln']})  area={area:.1f} LDU^2")

    if dbg:
        dbg.save(debug_png)
        print(f"debug render -> {debug_png}")
    if not overlaps:
        print(f"NO OVERLAPS ({len(parts)} parts checked, "
              f"{len(layers)} layers)")
    return overlaps


def check_studs(path, ppl=4.0, sliver_tol_ldu2=8.0):
    """Cross-layer check: no base-layer stud may be sliced by an art part's wall.

    Per stud: legal iff (a) essentially fully under ONE art part (normal
    hollow-underside connection; other parts may only touch it by a sub-LDU
    mesh sliver, e.g. the 1x1 quarter tile's arc passes 0.14 LDU inside the
    stud cylinder by design), or (b) essentially fully in open background.
    Anything else = a part wall genuinely slicing the stud (what Studio flags).
    """
    parts = []
    for color, pos, M, name, ln in parse_ldr(path):
        body = body_tris(name) @ M.T + pos           # stud-free
        st = stud_tris_local(name)
        vertical = abs(M[1, 1] - 1.0) < 0.01         # part mounted upright?
        studw = (st @ M.T + pos) if (vertical and len(st)) else np.zeros((0, 3, 3))
        parts.append(dict(name=name, pos=pos, ln=ln, body=body, studs=studw,
                          yr=(body[:, :, 1].min(), body[:, :, 1].max()),
                          layer=round(pos[1])))
    layer_ys = sorted({p["layer"] for p in parts})
    if len(layer_ys) < 2:
        print("single layer file: no stud check needed")
        return []

    all_pts = np.concatenate([p["body"].reshape(-1, 3) for p in parts])
    x0, z0 = all_pts[:, 0].min() - 4, all_pts[:, 2].min() - 4
    Wp = int((all_pts[:, 0].max() + 4 - x0) * ppl) + 1
    Hp = int((all_pts[:, 2].max() + 4 - z0) * ppl) + 1

    def footprint(p):
        img = Image.new("1", (Wp, Hp), 0)
        d = ImageDraw.Draw(img)
        for tri in p["body"]:
            d.polygon([((v[0] - x0) * ppl, (v[2] - z0) * ppl) for v in tri],
                      fill=1)
        return np.array(img, dtype=bool)

    conflicts = []
    n_studs = 0
    r_px = 6.0 * ppl
    yy, xx = np.ogrid[:Hp, :Wp]
    # Check every stud-bearing level: a part at origin level L has its studs
    # in the band [L-4, L]; anything whose body crosses that band must obey
    # the coverage rule (fully over one part, or fully open).
    for L in layer_ys:
        stud_parts = [p for p in parts if p["layer"] == L and len(p["studs"])]
        if not stud_parts:
            continue
        above = [p for p in parts
                 if p["layer"] != L
                 and min(p["yr"][1], L - 0.01) - max(p["yr"][0], L - 3.99) > 0.5]
        owner = np.full((Hp, Wp), -1, dtype=np.int32)
        for idx, p in enumerate(above):
            owner[footprint(p)] = idx
        for b in stud_parts:
            cents = b["studs"].mean(axis=1)
            keys = {(round((c[0] - 10) / 20) * 20 + 10,
                     round((c[2] - 10) / 20) * 20 + 10) for c in cents}
            for sx, sz in keys:
                n_studs += 1
                cx, cz = (sx - x0) * ppl, (sz - z0) * ppl
                disc = (xx - cx) ** 2 + (yy - cz) ** 2 <= r_px ** 2
                owners, counts = np.unique(owner[disc], return_counts=True)
                areas = {int(o): float(c) / ppl ** 2
                         for o, c in zip(owners, counts)}
                open_area = areas.pop(-1, 0.0)
                covered = sorted(areas.items(), key=lambda kv: -kv[1])
                total = open_area + sum(areas.values())
                main = covered[0][1] if covered else 0.0
                rest = sum(a for _, a in covered[1:])
                ok = (main >= total - sliver_tol_ldu2
                      and rest <= sliver_tol_ldu2) \
                     or (open_area >= total - sliver_tol_ldu2)
                if not ok:
                    slicers = [above[i]["name"] + f"(line {above[i]['ln']})"
                               for i, _ in covered]
                    conflicts.append((b, (sx, sz), covered, open_area))
                    print(f"STUD CONFLICT level {L}: stud of {b['name']} "
                          f"(line {b['ln']}) at ({sx:g},{sz:g}) sliced by "
                          f"{slicers}  open={open_area:.0f} LDU^2")
    if not conflicts:
        print(f"NO STUD CONFLICTS ({n_studs} studs checked, "
              f"{len(layer_ys)} levels)")
    return conflicts


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "output/test_mosaic_layers2.ldr"
    check(target, debug_png="ldr_geometry_debug.png")
    check_studs(target)
