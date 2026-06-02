"""Local drag-and-drop GUI for Upscaler. Run: `python app.py` then open the URL.

Everything runs on your machine — Gradio just serves a local web UI. No data
leaves your computer.
"""

from __future__ import annotations

import gradio as gr
from PIL import Image

from upscaler.engine import Upscaler, resolve_device
from upscaler.models.registry import MODELS
from upscaler.sharpen import unsharp_mask

# Cache loaded models so switching images doesn't reload weights every run.
_CACHE: dict[tuple[str, str], Upscaler] = {}


def _get_upscaler(model: str, device: str, tile: int) -> Upscaler:
    key = (model, device)
    up = _CACHE.get(key)
    if up is None or up.tile != tile:
        up = Upscaler(model=model, device=device, tile=tile)
        _CACHE[key] = up
    return up


def enhance(image, model, device, sharpen, tile):
    if image is None:
        raise gr.Error("Please upload an image first.")
    up = _get_upscaler(model, device, int(tile))
    result = up.upscale(image if isinstance(image, Image.Image) else Image.fromarray(image))
    if sharpen > 0:
        result = unsharp_mask(result, strength=float(sharpen))
    info = (
        f"**{up.spec.name}** · upscaled ×{up.scale} · device `{up.device.type}` · "
        f"{result.width}×{result.height}px"
        + (f" · sharpen {sharpen:g}" if sharpen > 0 else "")
    )
    return result, info


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
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
                    _MODEL_CHOICES, value="realesrgan-x4plus", label="Model"
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
                run = gr.Button("Upscale", variant="primary")
            with gr.Column():
                out = gr.Image(label="Result", type="pil", format="png")
                info = gr.Markdown()

        run.click(enhance, [inp, model, device, sharpen, tile], [out, info])
    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
