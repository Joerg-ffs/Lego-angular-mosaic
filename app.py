"""LEGO Mosaic Studio — local web app.

Run:  python app.py   (opens at http://127.0.0.1:7860)

Upload an image, tune the look with live preview, then build: you get a
realistic render, a parts list, and a .ldr file that opens in BrickLink Studio.
"""
import datetime
import os
from pathlib import Path

import gradio as gr
import numpy as np

from legomosaic import BuildConfig, ImageAdjust, PALETTES, STYLES, full_build, quick_preview
from legomosaic.colors import ALL_COLORS

# Locally, keep builds in ./output. On a hosted platform (Render, HF Spaces,
# etc.) the filesystem is ephemeral and ./output may not be writable, so use a
# temp dir there (Gradio serves the download files from disk either way).
_hosted = os.environ.get("PORT") is not None or os.environ.get("SPACE_ID") is not None
if _hosted:
    import tempfile
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "lego_mosaic_output"
else:
    OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COLOR_NAMES = [c.name for c in ALL_COLORS]
BACKING_CHOICES = ["Black", "White", "Light Bluish Grey", "Dark Bluish Grey",
                   "Tan", "Dark Blue", "Red", "Yellow"]


def _make_config(width, cl, cr, ct, cb, sat, hue, bright, contrast, style,
                 preset, custom_colors, gobricks, layers, backing, diag,
                 curves, dots, wings, snot, detail, focus, plate_finish,
                 dither) -> BuildConfig:
    names = list(custom_colors) if custom_colors else PALETTES[preset]
    n_layers = 1 if layers.startswith("1") else (3 if layers.startswith("3") else 2)
    return BuildConfig(
        width_studs=int(width),
        adjust=ImageAdjust(
            crop_left=cl / 100, crop_right=cr / 100,
            crop_top=ct / 100, crop_bottom=cb / 100,
            saturation=sat, hue_shift=hue, brightness=bright,
            contrast=contrast, style=style,
        ),
        palette_names=names,
        layers=n_layers,
        backing_name=backing,
        use_diagonals=diag, use_curves=curves, use_dots=dots, use_wings=wings,
        use_snot=snot,
        detail=detail, focus_strength=focus,
        plate_finish=plate_finish, dither=dither,
        gobricks_only=bool(gobricks),
    )


def preview(image, *args):
    if image is None:
        return None
    cfg = _make_config(*args)
    return quick_preview(image, cfg, px=10)


def build(image, *args):
    if image is None:
        raise gr.Error("Upload an image first.")
    cfg = _make_config(*args)
    out = full_build(image, cfg, title="LEGO Mosaic")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ldr_path = OUTPUT_DIR / f"mosaic_{stamp}.ldr"
    ldr_path.write_text(out.ldr_text, encoding="ascii")
    png_path = OUTPUT_DIR / f"mosaic_{stamp}.png"
    out.image.save(png_path)
    xml_path = OUTPUT_DIR / f"mosaic_{stamp}_order.xml"
    xml_path.write_text(out.bl_xml, encoding="ascii")
    s = out.stats
    stats_md = (
        f"**{s['size_studs']} studs**  ({s['size_cm']}) &nbsp;•&nbsp; "
        f"**{s['total_parts']} parts** in {s['unique_lots']} lots &nbsp;•&nbsp; "
        f"{s['detail_parts']} shaped detail parts, {s['fill_parts']} fill, "
        f"{s['base_parts']} base plates"
    )
    bom_rows = [[r["Part"], r["Buy ID"], r["Part name"], r["Color"],
                 r["LDraw code"], r["GoBricks color"], r["Qty"]]
                for r in out.bom]
    return (out.image, str(ldr_path), str(png_path), str(xml_path),
            stats_md, bom_rows, out.focus_image)


with gr.Blocks(title="LEGO Mosaic Studio") as demo:
    gr.Markdown("# 🧱 LEGO Mosaic Studio\n"
                "Upload an image, tune the live preview, then **Build** for the "
                "full artistic mosaic + BrickLink Studio `.ldr` file.")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="numpy", label="Source image")
            with gr.Accordion("Crop", open=False):
                cl = gr.Slider(0, 45, 0, step=1, label="Crop left %")
                cr = gr.Slider(0, 45, 0, step=1, label="Crop right %")
                ct = gr.Slider(0, 45, 0, step=1, label="Crop top %")
                cb = gr.Slider(0, 45, 0, step=1, label="Crop bottom %")
            with gr.Accordion("Tone & style", open=True):
                sat = gr.Slider(0, 2, 1.0, step=0.05, label="Saturation")
                hue = gr.Slider(-180, 180, 0, step=5, label="Hue shift (deg)")
                bright = gr.Slider(-80, 80, 0, step=5, label="Brightness")
                contrast = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Contrast")
                style = gr.Radio(STYLES, value="Photo", label="Art style")
        with gr.Column(scale=1):
            width = gr.Slider(24, 128, 48, step=2, label="Mosaic width (studs)")
            preset = gr.Dropdown(list(PALETTES.keys()), value="Full",
                                 label="Color palette")
            custom_colors = gr.Dropdown(COLOR_NAMES, multiselect=True, value=None,
                                        label="Custom palette (overrides preset "
                                              "when colors are selected)")
            gobricks = gr.Checkbox(False,
                                   label="GoBricks mode — only colors GoBricks "
                                         "makes, orderable at brickwith.com")
            layers = gr.Radio(["2 layers (plate base + tile art)",
                               "1 layer (mount on a baseplate)",
                               "3 deep / SNOT (2 extra plate levels)"],
                              value="2 layers (plate base + tile art)",
                              label="Build thickness")
            backing = gr.Dropdown(BACKING_CHOICES, value="Black",
                                  label="Backing color (1-layer builds)")
            with gr.Accordion("Part types", open=True):
                diag = gr.Checkbox(True, label="45° diagonals (triangular / cut tiles)")
                curves = gr.Checkbox(True, label="Curves (quarter-round, macaroni, S-curve)")
                dots = gr.Checkbox(True, label="Round 1x1 dots")
                wings = gr.Checkbox(True, label="Shallow-angle wing plates")
                snot = gr.Checkbox(True, label="SNOT stripe modules (deep builds: "
                                               "2.5x resolution sideways plates)")
            detail = gr.Slider(0, 1, 0.7, step=0.05,
                               label="Detail (how eagerly shaped parts are used)")
            focus = gr.Slider(0, 1, 0.5, step=0.05,
                              label="Focal detail (auto-detected subject gets "
                                    "more shaped parts than the background)")
            plate_finish = gr.Checkbox(False, label="Studded finish (plates instead of smooth tiles)")
            dither = gr.Slider(0, 1, 0.0, step=0.1,
                               label="Dithering (smoother gradients, noisier fields)")
        with gr.Column(scale=2):
            preview_img = gr.Image(label="Live preview (palette + grid only)",
                                   interactive=False)
            build_btn = gr.Button("🔨 Build mosaic", variant="primary")
            result_img = gr.Image(label="Built mosaic (true part shapes)",
                                  interactive=False)
            stats_md = gr.Markdown()
            with gr.Row():
                ldr_file = gr.File(label="LDraw file (.ldr) — open in BrickLink Studio")
                png_file = gr.File(label="Render (.png)")
                xml_file = gr.File(label="Order list (.xml) — import at "
                                         "brickwith.com or BrickLink")
            focus_img = gr.Image(label="Detected focal points (red = more detail)",
                                 interactive=False)
            bom_table = gr.Dataframe(
                headers=["Part", "Buy ID", "Part name", "Color", "LDraw code",
                         "GoBricks color", "Qty"],
                label="Bill of materials", interactive=False)

    controls = [width, cl, cr, ct, cb, sat, hue, bright, contrast, style,
                preset, custom_colors, gobricks, layers, backing, diag,
                curves, dots, wings, snot, detail, focus, plate_finish,
                dither]
    for c in [image, width, cl, cr, ct, cb, sat, hue, bright, contrast, style,
              preset, custom_colors, gobricks]:
        c.change(preview, inputs=[image] + controls, outputs=preview_img,
                 show_progress="hidden")
    build_btn.click(build, inputs=[image] + controls,
                    outputs=[result_img, ldr_file, png_file, xml_file,
                             stats_md, bom_table, focus_img])

if __name__ == "__main__":
    # Hosted (Render, HF Spaces, etc.) sets a PORT env var and needs the server
    # bound to 0.0.0.0. Locally, open a browser at 127.0.0.1:7860.
    port = os.environ.get("PORT")
    hosted = port is not None or os.environ.get("SPACE_ID") is not None
    if hosted:
        demo.launch(server_name="0.0.0.0",
                    server_port=int(port) if port else 7860)
    else:
        demo.launch(inbrowser=True)
