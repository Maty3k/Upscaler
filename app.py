"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.
"""

from __future__ import annotations

import gradio as gr
from PIL import Image

from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler, resolve_device
from upscaler.models.registry import DEBLUR_MODELS, MODELS
from upscaler.sharpen import unsharp_mask

# Cache loaded models so switching images doesn't reload weights every run.
_UP_CACHE: dict[tuple[str, str], Upscaler] = {}
_DB_CACHE: dict[tuple[str, str], Deblurrer] = {}


def _get_upscaler(model: str, device: str, tile: int) -> Upscaler:
    key = (model, device)
    up = _UP_CACHE.get(key)
    if up is None or up.tile != tile:
        up = Upscaler(model=model, device=device, tile=tile)
        _UP_CACHE[key] = up
    return up


def _get_deblurrer(model: str, device: str) -> Deblurrer:
    key = (model, device)
    db = _DB_CACHE.get(key)
    if db is None:
        db = Deblurrer(model=model, device=device)
        _DB_CACHE[key] = db
    return db


def enhance(image, model, device, deblur, deblur_model, sharpen, tile):
    if image is None:
        raise gr.Error("Please upload an image first.")
    src = image if isinstance(image, Image.Image) else Image.fromarray(image)
    stages = []
    if deblur:
        src = _get_deblurrer(deblur_model, device).deblur(src)
        stages.append(f"deblur `{deblur_model}`")
    up = _get_upscaler(model, device, int(tile))
    result = up.upscale(src)
    stages.append(f"upscale ×{up.scale}")
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
        stages.append(f"sharpen {sharpen:g}")
    info = (
        " → ".join(stages)
        + f" · device `{up.device.type}` · {result.width}×{result.height}px"
    )
    return result, info


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
_DEBLUR_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in DEBLUR_MODELS.values()]
_DEVICES = ["auto", "cpu", "cuda", "mps"]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Upscaler") as demo:
        gr.Markdown(
            "# 🔍 Upscaler\n"
            "Local, open-source image **upscaling + sharpening** "
            f"(Real-ESRGAN). Auto-detected device: `{resolve_device('auto').type}`."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Input", type="pil", sources=["upload", "clipboard"])
                model = gr.Dropdown(
                    _MODEL_CHOICES, value="realesrgan-x4plus", label="Upscale model"
                )
                deblur = gr.Checkbox(
                    value=False, label="Deblur first (NAFNet) — for motion blur"
                )
                deblur_model = gr.Dropdown(
                    _DEBLUR_CHOICES, value="nafnet-gopro-width64", label="Deblur model"
                )
                sharpen = gr.Slider(
                    0.0, 3.0, value=0.0, step=0.1,
                    label="Sharpen (unsharp mask) — 0 = off",
                )
                with gr.Accordion("Advanced", open=False):
                    device = gr.Dropdown(_DEVICES, value="auto", label="Device")
                    tile = gr.Slider(
                        0, 1024, value=512, step=64,
                        label="Tile size (0 = no tiling; lower if you run out of memory)",
                    )
                run = gr.Button("Enhance", variant="primary")
            with gr.Column():
                out = gr.Image(label="Result", type="pil", format="png")
                info = gr.Markdown()

        run.click(
            enhance,
            [inp, model, device, deblur, deblur_model, sharpen, tile],
            [out, info],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
