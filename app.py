"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.

Two tools on one page:
  • File Converter — fast, lossless-where-possible format conversion (no models).
  • Upscale & Enhance — Real-ESRGAN upscaling, optional NAFNet deblur + sharpen.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile

import gradio as gr
from PIL import Image

from upscaler.convert import FORMATS, convert, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler, resolve_device
from upscaler.models.registry import DEBLUR_MODELS, MODELS
from upscaler.sharpen import unsharp_mask

# Cache loaded models so switching images doesn't reload weights every run.
# Keyed by (model, device, onnx) so torch and ONNX engines are cached separately.
_UP_CACHE: dict[tuple, object] = {}
_DB_CACHE: dict[tuple, object] = {}


def _onnx_engines():
    try:
        from upscaler.onnx_engine import OnnxDeblurrer, OnnxUpscaler
    except ImportError as e:
        raise gr.Error(
            'ONNX backend needs optional deps — install with: pip install -e ".[onnx]"'
        ) from e
    return OnnxUpscaler, OnnxDeblurrer


def _get_upscaler(model: str, device: str, tile: int, onnx: bool):
    key = (model, device, onnx)
    up = _UP_CACHE.get(key)
    if up is None or getattr(up, "tile", None) != tile:
        if onnx:
            OnnxUpscaler, _ = _onnx_engines()
            up = OnnxUpscaler(model=model, device=device, tile=tile)
        else:
            up = Upscaler(model=model, device=device, tile=tile)
        _UP_CACHE[key] = up
    return up


def _get_deblurrer(model: str, device: str, onnx: bool):
    key = (model, device, onnx)
    db = _DB_CACHE.get(key)
    if db is None:
        if onnx:
            _, OnnxDeblurrer = _onnx_engines()
            db = OnnxDeblurrer(model=model, device=device)
        else:
            db = Deblurrer(model=model, device=device)
        _DB_CACHE[key] = db
    return db


# -- File converter ----------------------------------------------------------

def convert_image(image, fmt, quality, lossless):
    if image is None:
        raise gr.Error("Upload an image to convert first.")
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    data = convert(img, fmt, quality=int(quality), lossless=bool(lossless))

    fd, path = tempfile.mkstemp(suffix=f".{extension_for(fmt)}")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    kb = len(data) / 1024
    note = "lossless" if (lossless and fmt == "WebP") or not FORMATS[fmt][2] else f"q{int(quality)}"
    return path, f"✅ Converted to **{fmt}** ({note}) · {kb:,.1f} KB · {img.width}×{img.height}px"


# -- Image <-> PDF -----------------------------------------------------------

def build_pdf(files):
    if not files:
        raise gr.Error("Add at least one image.")
    images = [Image.open(f) for f in files]
    data = images_to_pdf(images)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path, f"✅ {len(images)} image(s) → PDF · {len(data) / 1024:,.1f} KB"


def extract_pdf(pdf_file, dpi):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")
    try:
        pages = pdf_to_images(pdf_file, dpi=int(dpi))
    except ImportError as e:
        raise gr.Error(str(e)) from e

    fd, zpath = tempfile.mkstemp(suffix=".zip")
    with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w") as z:
        for i, im in enumerate(pages, 1):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            z.writestr(f"page_{i:03d}.png", buf.getvalue())
    return zpath, pages, f"✅ {len(pages)} page(s) → PNG · {int(dpi)} dpi (zip)"


# -- Upscale & enhance -------------------------------------------------------

def enhance(image, model, device, deblur, deblur_model, sharpen, tile, onnx):
    if image is None:
        raise gr.Error("Upload an image to enhance first.")
    src = image if isinstance(image, Image.Image) else Image.fromarray(image)
    stages = []
    if deblur:
        src = _get_deblurrer(deblur_model, device, onnx).deblur(src)
        stages.append(f"deblur `{deblur_model}`")
    up = _get_upscaler(model, device, int(tile), onnx)
    result = up.upscale(src)
    stages.append(f"upscale ×{up.scale}")
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")
    backend = "onnx" if onnx else getattr(up, "device", None) and up.device.type
    info = (
        "✅ " + " → ".join(stages)
        + f" · backend `{backend}` · {result.width}×{result.height}px"
    )
    return result, info


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
_DEBLUR_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in DEBLUR_MODELS.values()]
_DEVICES = ["auto", "cpu", "cuda", "mps"]

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Manrope"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#FAFAF9",
    body_text_color="#1C1917",
    body_text_color_subdued="#78716C",
    block_background_fill="#FFFFFF",
    block_border_color="#EAE8E4",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(28,25,23,0.04)",
    block_label_text_weight="600",
    block_label_text_color="#57534E",
    input_background_fill="#FFFFFF",
    input_border_color="#E7E5E4",
    input_border_color_focus="#A8A29E",
    button_primary_background_fill="#1C1917",
    button_primary_background_fill_hover="#3F3B37",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#1C1917",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_border_color="#E7E5E4",
    button_large_radius="12px",
    button_small_radius="10px",
    # A genuine dark palette (warm stone), so the dark toggle looks intentional.
    body_background_fill_dark="#0C0A09",
    body_text_color_dark="#F5F5F4",
    body_text_color_subdued_dark="#A8A29E",
    block_background_fill_dark="#1C1917",
    block_border_color_dark="#292524",
    block_label_text_color_dark="#D6D3D1",
    input_background_fill_dark="#1C1917",
    input_border_color_dark="#292524",
    input_border_color_focus_dark="#57534E",
    button_primary_background_fill_dark="#FAFAF9",
    button_primary_background_fill_hover_dark="#E7E5E4",
    button_primary_text_color_dark="#1C1917",
    button_secondary_background_fill_dark="#1C1917",
    button_secondary_border_color_dark="#292524",
)

# Apply the saved light/dark preference on load (default light), and a toggle
# that flips it. Both go through Gradio's own ?__theme mechanism (one reload).
_APPLY_THEME_JS = """
() => {
  const u = new URL(window.location.href);
  const saved = localStorage.getItem('upscaler-theme') || 'light';
  if (u.searchParams.get('__theme') !== saved) {
    u.searchParams.set('__theme', saved);
    window.location.replace(u.toString());
  }
}
"""

_TOGGLE_THEME_JS = """
() => {
  const u = new URL(window.location.href);
  const cur = u.searchParams.get('__theme')
              || localStorage.getItem('upscaler-theme') || 'light';
  const next = cur === 'dark' ? 'light' : 'dark';
  localStorage.setItem('upscaler-theme', next);
  u.searchParams.set('__theme', next);
  window.location.replace(u.toString());
}
"""

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* Custom CSS uses Gradio theme vars (--body-text-color etc.) so it adapts to
   both light and dark automatically. */
.gradio-container { max-width: 100% !important; padding: 4px 40px 56px !important;
    position: relative; }

#hero { padding: 34px 2px 16px; }
#hero .brand { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em;
    margin: 0; color: var(--body-text-color); }
#hero .sub { color: var(--body-text-color-subdued); margin: 8px 0 14px;
    font-size: 1.02rem; max-width: 64ch; }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
    border: 1px solid var(--border-color-primary); border-radius: 999px;
    font-size: 0.8rem; color: var(--body-text-color-subdued);
    background: var(--block-background-fill); font-weight: 500; }
.pill .dot { width: 7px; height: 7px; border-radius: 999px; background: #22C55E; }

/* theme toggle, floated top-right */
#theme-toggle { position: absolute; top: 20px; right: 40px; z-index: 50;
    width: auto !important; min-width: 0 !important; flex: none !important; }

.section-card { border-radius: 18px !important; padding: 24px !important; }
.sec-head .eyebrow { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--body-text-color-subdued); opacity: .8; }
.sec-head h2 { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em;
    margin: 3px 0 5px; color: var(--body-text-color); }
.sec-head p { color: var(--body-text-color-subdued); margin: 0; font-size: 0.92rem;
    max-width: 72ch; }
.col-label { font-weight: 600; color: var(--body-text-color); font-size: 0.92rem; }
.spacer { height: 22px; }

/* clean, friendly drop zones for image/file inputs */
.drop { border: 1.5px dashed var(--border-color-primary) !important;
    background: var(--block-background-fill) !important;
    border-radius: 14px !important; box-shadow: none !important;
    transition: border-color .15s ease; }
.drop:hover { border-color: var(--body-text-color-subdued) !important; }

footer { display: none !important; }
"""


def _section_head(eyebrow: str, title: str, desc: str) -> str:
    return (
        f'<div class="sec-head"><div class="eyebrow">{eyebrow}</div>'
        f"<h2>{title}</h2><p>{desc}</p></div>"
    )


def build_demo() -> gr.Blocks:
    device_name = resolve_device("auto").type
    with gr.Blocks(title="Upscaler") as demo:
        gr.HTML(
            '<div id="hero">'
            '<div class="brand">Upscaler</div>'
            '<div class="sub">A quiet little toolbox for images — convert formats, '
            "make and split PDFs, and upscale with AI. Everything runs locally; "
            "nothing is uploaded.</div>"
            f'<span class="pill"><span class="dot"></span>running locally · {device_name}</span>'
            "</div>"
        )
        theme_btn = gr.Button(
            "◐ Light / Dark", elem_id="theme-toggle", size="sm", variant="secondary"
        )
        theme_btn.click(None, js=_TOGGLE_THEME_JS)

        # ---- Section 1: File Converter ----
        with gr.Group(elem_classes="section-card"):
            gr.HTML(_section_head(
                "Convert", "File Converter",
                "Change image format — fast, no AI models.",
            ))
            with gr.Row():
                with gr.Column(scale=1):
                    conv_in = gr.Image(
                        label="Image", type="pil", sources=["upload", "clipboard"],
                        height=260, elem_classes="drop",
                    )
                    with gr.Row():
                        conv_fmt = gr.Dropdown(
                            list(FORMATS), value="PNG", label="Convert to", scale=2
                        )
                        conv_quality = gr.Slider(
                            1, 100, value=90, step=1, label="Quality (lossy)", scale=3
                        )
                    conv_lossless = gr.Checkbox(value=False, label="Lossless WebP")
                    conv_btn = gr.Button("Convert", variant="primary", size="lg")
                with gr.Column(scale=1):
                    conv_file = gr.File(label="Download converted file")
                    conv_info = gr.Markdown()

        gr.HTML('<div class="spacer"></div>')

        # ---- Section 2: Image <-> PDF ----
        with gr.Group(elem_classes="section-card"):
            gr.HTML(_section_head(
                "Documents", "Image ⇄ PDF",
                "Combine images into a PDF, or extract a PDF's pages back to PNGs.",
            ))
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="col-label">Images → PDF '
                            "<span style='color:#A8A29E;font-weight:400'>"
                            "(multiple = multi-page)</span></div>")
                    pdf_imgs_in = gr.File(
                        label="Images", file_count="multiple", file_types=["image"],
                        elem_classes="drop",
                    )
                    pdf_build_btn = gr.Button("Build PDF", variant="primary", size="lg")
                    pdf_build_out = gr.File(label="Download PDF")
                    pdf_build_info = gr.Markdown()
                with gr.Column(scale=1):
                    gr.HTML('<div class="col-label">PDF → Images</div>')
                    pdf_in = gr.File(label="PDF", file_count="single", file_types=[".pdf"],
                                     elem_classes="drop")
                    pdf_dpi = gr.Slider(72, 300, value=150, step=1, label="Render DPI")
                    pdf_extract_btn = gr.Button("Extract pages", variant="primary", size="lg")
                    pdf_extract_out = gr.File(label="Download pages (ZIP)")
                    pdf_gallery = gr.Gallery(label="Pages", columns=4, height=220)
                    pdf_extract_info = gr.Markdown()

        gr.HTML('<div class="spacer"></div>')

        # ---- Section 3: Upscale & Enhance ----
        with gr.Group(elem_classes="section-card"):
            gr.HTML(_section_head(
                "AI", "Upscale & Enhance",
                "Real-ESRGAN super-resolution, with optional deblur and sharpening. "
                "Tip: for an already-decent photo, prefer the ×2 model — ×4 can "
                "over-process clean images.",
            ))
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(
                        label="Input", type="pil", sources=["upload", "clipboard"],
                        height=260, elem_classes="drop",
                    )
                    model = gr.Dropdown(
                        _MODEL_CHOICES, value="realesrgan-x4plus", label="Upscale model"
                    )
                    sharpen = gr.Slider(
                        0.0, 3.0, value=0.0, step=0.1,
                        label="Sharpen (unsharp mask) — 0 = off",
                    )
                    with gr.Accordion("Deblur (motion blur)", open=False):
                        deblur = gr.Checkbox(value=False, label="Deblur first (NAFNet)")
                        deblur_model = gr.Dropdown(
                            _DEBLUR_CHOICES, value="nafnet-gopro-width64",
                            label="Deblur model",
                        )
                    with gr.Accordion("Advanced", open=False):
                        device = gr.Dropdown(_DEVICES, value="auto", label="Device")
                        onnx = gr.Checkbox(
                            value=False,
                            label="ONNX Runtime backend (exports once; torch-free, "
                            "often faster on CPU)",
                        )
                        tile = gr.Slider(
                            0, 1024, value=512, step=64,
                            label="Tile size (0 = off; lower if you run out of memory)",
                        )
                    run = gr.Button("Enhance", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out = gr.Image(label="Result", type="pil", format="png", height=260)
                    info = gr.Markdown()

        conv_btn.click(
            convert_image,
            [conv_in, conv_fmt, conv_quality, conv_lossless],
            [conv_file, conv_info],
        )
        pdf_build_btn.click(build_pdf, [pdf_imgs_in], [pdf_build_out, pdf_build_info])
        pdf_extract_btn.click(
            extract_pdf, [pdf_in, pdf_dpi], [pdf_extract_out, pdf_gallery, pdf_extract_info]
        )
        run.click(
            enhance,
            [inp, model, device, deblur, deblur_model, sharpen, tile, onnx],
            [out, info],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("UPSCALER_PORT", "7860")),
        theme=THEME,
        css=_CSS,
        js=_APPLY_THEME_JS,
    )
