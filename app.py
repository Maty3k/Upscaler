"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.

Two tools on one page:
  • File Converter — fast, lossless-where-possible format conversion (no models).
  • Upscale & Enhance — Real-ESRGAN upscaling, optional NAFNet deblur + sharpen.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
from PIL import Image

from upscaler.convert import FORMATS, convert, extension_for
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

_CSS = """
.gradio-container { max-width: 1080px !important; margin: auto; }
#hero h1 { margin-bottom: 0; }
.section-card { border: 1px solid var(--border-color-primary);
    border-radius: 12px; padding: 16px; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Upscaler", css=_CSS) as demo:
        with gr.Column(elem_id="hero"):
            gr.Markdown(
                "# 🖼️ Upscaler\n"
                "Local, open-source image tools — nothing leaves your machine. "
                f"Device: `{resolve_device('auto').type}`."
            )

        # ---- Section 1: File Converter ----
        with gr.Group(elem_classes="section-card"):
            gr.Markdown("## 📁 File Converter\nChange image format — fast, no AI models.")
            with gr.Row():
                with gr.Column(scale=1):
                    conv_in = gr.Image(
                        label="Image", type="pil", sources=["upload", "clipboard"],
                        height=240,
                    )
                    with gr.Row():
                        conv_fmt = gr.Dropdown(
                            list(FORMATS), value="PNG", label="Convert to", scale=2
                        )
                        conv_quality = gr.Slider(
                            1, 100, value=90, step=1, label="Quality (lossy)", scale=3
                        )
                    conv_lossless = gr.Checkbox(value=False, label="Lossless WebP")
                    conv_btn = gr.Button("Convert", variant="primary")
                with gr.Column(scale=1):
                    conv_file = gr.File(label="Download converted file")
                    conv_info = gr.Markdown()

        gr.Markdown("---")

        # ---- Section 2: Upscale & Enhance ----
        with gr.Group(elem_classes="section-card"):
            gr.Markdown(
                "## 🔍 Upscale & Enhance\n"
                "Real-ESRGAN super-resolution, with optional deblur and sharpening. "
                "*Tip: for an already-decent photo, prefer the **×2** model — ×4 can "
                "over-process clean images.*"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(
                        label="Input", type="pil", sources=["upload", "clipboard"],
                        height=240,
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
                    run = gr.Button("Enhance", variant="primary")
                with gr.Column(scale=1):
                    out = gr.Image(label="Result", type="pil", format="png", height=240)
                    info = gr.Markdown()

        conv_btn.click(
            convert_image,
            [conv_in, conv_fmt, conv_quality, conv_lossless],
            [conv_file, conv_info],
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
        theme=gr.themes.Soft(),
    )
