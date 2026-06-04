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


# -- Video (frame-by-frame) --------------------------------------------------

def upscale_video_ui(video_path, model, sharpen, device, tile, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    from upscaler.video import upscale_video

    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    def cb(i, n):
        progress(i / n, desc=f"Upscaling frame {i}/{n}")

    try:
        upscale_video(
            video_path, out, model=model, device=device, tile=int(tile),
            sharpen=float(sharpen), progress_cb=cb,
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e
    return out, "✅ Done — preview and download below."


_CONVERT_METHODS = ["Change image format", "Images → PDF", "PDF → Images"]


def _switch_method(choice):
    """Show only the group for the selected conversion method."""
    return (
        gr.update(visible=choice == _CONVERT_METHODS[0]),
        gr.update(visible=choice == _CONVERT_METHODS[1]),
        gr.update(visible=choice == _CONVERT_METHODS[2]),
    )


_MODEL_CHOICES = [(f"{s.name}  (×{s.scale}) — {s.notes}", s.name) for s in MODELS.values()]
_DEBLUR_CHOICES = [(f"{s.name} — {s.notes}", s.name) for s in DEBLUR_MODELS.values()]
_DEVICES = ["auto", "cpu", "cuda", "mps"]

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Manrope"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_lg,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#E7E3DB",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F1EEE8",
    body_text_color="#1A1714",
    body_text_color_subdued="#6E675E",
    block_background_fill="#FFFFFF",
    block_border_color="#D5CEC2",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(28,25,23,0.05), 0 8px 24px rgba(28,25,23,0.06)",
    block_label_text_weight="600",
    block_label_text_color="#57534E",
    block_label_background_fill="#FFFFFF",
    block_label_border_color="#E2DED7",
    block_title_text_color="#3F3B37",
    block_info_text_color="#78716C",
    panel_background_fill="#FFFFFF",
    input_background_fill="#FFFFFF",
    input_border_color="#DED9D1",
    input_border_color_focus="#0D9488",
    button_primary_background_fill="#0D9488",
    button_primary_background_fill_hover="#0F766E",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#0D9488",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_border_color="#E7E5E4",
    button_large_radius="12px",
    button_small_radius="10px",
    # Warm, softer dark palette with layered surfaces (not near-black), so the
    # dark toggle has depth instead of feeling like a void.
    body_background_fill_dark="#161311",
    background_fill_primary_dark="#322C27",
    background_fill_secondary_dark="#211D1A",
    body_text_color_dark="#FAF7F3",
    body_text_color_subdued_dark="#BCB2A7",
    block_background_fill_dark="#322C27",
    block_border_color_dark="#473F37",
    block_label_text_color_dark="#EFE9E2",
    block_label_background_fill_dark="#3A332D",
    block_label_border_color_dark="#473F37",
    block_title_text_color_dark="#F1ECE6",
    block_info_text_color_dark="#BCB2A7",
    panel_background_fill_dark="#322C27",
    input_background_fill_dark="#262119",
    input_border_color_dark="#473F37",
    input_border_color_focus_dark="#2DD4BF",
    button_primary_background_fill_dark="#14B8A6",
    button_primary_background_fill_hover_dark="#2DD4BF",
    button_primary_text_color_dark="#06231F",
    button_secondary_background_fill_dark="#2E2A26",
    button_secondary_border_color_dark="#3B342D",
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
   both light and dark automatically. --ac is the accent (teal), brighter in dark. */
.gradio-container { --ac: #0F766E; --ac-weak: rgba(13,148,136,.10);
    --tex: rgba(28,25,23,.07); }
.dark .gradio-container, .dark { --ac: #2DD4BF; --ac-weak: rgba(45,212,191,.13);
    --tex: rgba(255,255,255,.06); }

/* gradio-app carries the .dark scope, so fill the viewport with IT (html/body
   sit outside the scope and would otherwise show a strip behind the app). The
   html/body fallback covers light mode; gradio-app (100vh) covers dark. */
html, body { background: var(--body-background-fill) !important; }
gradio-app { display: block; min-height: 100vh;
    background: var(--body-background-fill) !important; }
/* The container has the accent + texture vars in scope, so the warm glow and a
   faint dot texture go here (not on gradio-app, where those vars are undefined). */
.gradio-container { max-width: 100% !important; padding: 6px 44px 64px !important;
    position: relative;
    background:
      radial-gradient(1100px 460px at 18% -8%, var(--ac-weak), transparent 62%),
      radial-gradient(var(--tex) 1px, transparent 1.4px) 0 0 / 22px 22px,
      var(--body-background-fill) !important; }

/* Cheap transitions on interactive controls ONLY — color/border, no box-shadow
   or transform on every .block (that caused heavy repaints / ~20fps jank). */
button, .tab-nav button, .drop, .item, .dropdown-arrow {
    transition: background-color .18s ease, border-color .18s ease, color .18s ease; }
/* transform/box-shadow only on the single button being hovered (cheap). */
.gradio-container button.primary {
    transition: background-color .18s ease, transform .12s ease, box-shadow .2s ease; }
.gradio-container button.primary:hover { transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(13,148,136,.28); }
.gradio-container button.primary:active { transform: translateY(0); box-shadow: none; }
.tab-nav button:hover { color: var(--body-text-color) !important; }

/* Entrance fades — only opacity + a tiny transform, GPU-composited (will-change)
   so they stay smooth. Hero/section heads fill `both` (load-time); tab/accordion
   bodies use no fill (default opacity 1) so they can't get stuck invisible. */
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: none; } }
#hero, .sec-head, .tabitem, [data-testid="accordion-content"], ul.options, .options {
    will-change: opacity, transform; }
#hero { animation: fadeUp .55s cubic-bezier(.22,.61,.36,1) both; }
.sec-head { animation: fadeUp .55s cubic-bezier(.22,.61,.36,1) .05s both; }
.tabitem { animation: fadeUp .45s cubic-bezier(.22,.61,.36,1); }
[data-testid="accordion-content"] { animation: fadeUp .4s cubic-bezier(.22,.61,.36,1); }
.label-wrap .icon { transition: transform .25s cubic-bezier(.22,.61,.36,1) !important; }

/* dropdown list: smooth open (was an abrupt instant pop) */
@keyframes ddOpen { from { opacity: 0; transform: translateY(-5px) scale(.99); }
    to { opacity: 1; transform: none; } }
ul.options, .options { animation: ddOpen .2s cubic-bezier(.22,.61,.36,1);
    transform-origin: top center; }
ul.options .item, .options .item { transition: background-color .12s ease; }
.dropdown-arrow { transition: transform .25s cubic-bezier(.22,.61,.36,1); }

@media (prefers-reduced-motion: reduce) {
    #hero, .sec-head, .tabitem, [data-testid="accordion-content"],
    ul.options, .options { animation: none; } }
@keyframes livepulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.45); }
    70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }

#hero { padding: 32px 2px 18px; margin-bottom: 8px;
    border-bottom: 1px solid var(--border-color-primary); }
#hero .brandrow { display: flex; align-items: center; gap: 11px; }
#hero .logo { color: var(--ac); display: inline-flex; }
#hero .brand { font-size: 2.05rem; font-weight: 800; letter-spacing: -0.03em;
    margin: 0; color: var(--body-text-color); }
#hero .sub { color: var(--body-text-color-subdued); margin: 9px 0 14px;
    font-size: 1.02rem; max-width: 64ch; }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
    border: 1px solid var(--border-color-primary); border-radius: 999px;
    font-size: 0.8rem; color: var(--body-text-color-subdued);
    background: var(--block-background-fill); font-weight: 500; }
.pill .dot { width: 7px; height: 7px; border-radius: 999px; background: #22C55E;
    animation: livepulse 2.4s ease-out infinite; }

/* theme toggle, floated top-right */
#theme-toggle { position: absolute; top: 20px; right: 40px; z-index: 50;
    width: auto !important; min-width: 0 !important; flex: none !important; }

/* tab bar: accent the selected tab */
.tabitem { padding-top: 28px !important; }
.tab-nav { gap: 2px; }
.tab-nav button { font-weight: 600 !important; font-size: 0.98rem !important;
    color: var(--body-text-color-subdued) !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
.tab-nav button.selected { color: var(--ac) !important;
    border-bottom: 2px solid var(--ac) !important; }

/* section heads: accent eyebrow w/ icon + underlined title */
.sec-head { margin-bottom: 8px; }
.sec-head .eyebrow { display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--ac); }
.sec-head .eyebrow .ic { display: inline-flex; align-items: center; }
.sec-head h2 { position: relative; display: inline-block; font-size: 1.32rem;
    font-weight: 700; letter-spacing: -0.02em; margin: 5px 0 9px;
    padding-bottom: 8px; color: var(--body-text-color); }
.sec-head h2::after { content: ""; position: absolute; left: 0; bottom: 0;
    width: 40px; height: 2px; border-radius: 2px; background: var(--ac); }
.sec-head p { color: var(--body-text-color-subdued); margin: 0; font-size: 0.92rem;
    max-width: 72ch; line-height: 1.55; }
.col-label { font-weight: 600; color: var(--body-text-color); font-size: 0.92rem;
    border-left: 3px solid var(--ac); padding-left: 9px; }
.spacer { height: 22px; }

/* clean, friendly drop zones for image/file inputs */
.drop { border: 1.5px dashed var(--border-color-primary) !important;
    background: var(--block-background-fill) !important;
    border-radius: 14px !important; box-shadow: none !important;
    transition: border-color .15s ease, background .15s ease; }
.drop:hover { border-color: var(--ac) !important; background: var(--ac-weak) !important; }

footer { display: none !important; }
"""


def _svg(paths: str) -> str:
    return (
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )


# Section / brand icons (stroke = currentColor, so they pick up the accent).
ICON_AI = _svg('<path d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5'
               'l4.6-1.9z"/><path d="M19 14l.6 1.6 1.6.6-1.6.6L19 19l-.6-1.6'
               '-1.6-.6 1.6-.6z"/>')
ICON_CONVERT = _svg('<path d="M7 4 3 8l4 4"/><path d="M3 8h14"/>'
                    '<path d="m17 20 4-4-4-4"/><path d="M21 16H7"/>')
ICON_PDF = _svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 '
                '2-2V8z"/><path d="M14 3v5h5"/>')
ICON_LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.1" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5z"/>'
    '<path d="m3 13 9 5 9-5"/></svg>'
)


def _section_head(eyebrow: str, title: str, desc: str, icon: str = "") -> str:
    return (
        '<div class="sec-head">'
        f'<div class="eyebrow"><span class="ic">{icon}</span>{eyebrow}</div>'
        f"<h2>{title}</h2><p>{desc}</p></div>"
    )


def build_demo() -> gr.Blocks:
    device_name = resolve_device("auto").type
    with gr.Blocks(title="Upscaler") as demo:
        gr.HTML(
            '<div id="hero">'
            f'<div class="brandrow"><span class="logo">{ICON_LOGO}</span>'
            '<span class="brand">Upscaler</span></div>'
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

        with gr.Tabs():
            # ---- Tab 1: Upscale & Enhance ----
            with gr.Tab("Upscale"):
                gr.HTML(_section_head(
                    "AI", "Upscale & Enhance",
                    "Real-ESRGAN super-resolution, with optional deblur and "
                    "sharpening. Tip: for an already-decent photo, prefer the ×2 "
                    "model — ×4 can over-process clean images.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        inp = gr.Image(
                            label="Input", type="pil",
                            sources=["upload", "clipboard"], height=300,
                            elem_classes="drop",
                        )
                        model = gr.Dropdown(
                            _MODEL_CHOICES, value="realesrgan-x4plus",
                            label="Upscale model",
                            info="×2 is gentler for already-good photos · ×4 adds "
                            "the most detail but can over-process · anime model is "
                            "for illustrations & line art.",
                        )
                        sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen (unsharp mask) — 0 = off",
                            info="Crispens edges after upscaling. Keep it low — too "
                            "high adds halos around edges.",
                        )
                        with gr.Accordion("Deblur (motion blur)", open=False):
                            deblur = gr.Checkbox(
                                value=False, label="Deblur first (NAFNet)",
                                info="Only for genuinely motion-blurred shots — it "
                                "softens images that are already sharp.",
                            )
                            deblur_model = gr.Dropdown(
                                _DEBLUR_CHOICES, value="nafnet-gopro-width64",
                                label="Deblur model",
                                info="width64 = best quality · width32 = faster.",
                            )
                        with gr.Accordion("Advanced", open=False):
                            device = gr.Dropdown(
                                _DEVICES, value="auto", label="Device",
                                info="auto picks a GPU if available (CUDA / Apple "
                                "MPS), otherwise CPU.",
                            )
                            onnx = gr.Checkbox(
                                value=False, label="ONNX Runtime backend",
                                info="Exports the model to ONNX once, then runs "
                                "without PyTorch — often faster on CPU.",
                            )
                            tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Processes big images in tiles to save memory. "
                                "Lower this if you hit out-of-memory errors.",
                            )
                        run = gr.Button("Enhance", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        out = gr.Image(
                            label="Result", type="pil", format="png", height=300
                        )
                        info = gr.Markdown()

            # ---- Tab 2: Video (frame-by-frame) ----
            with gr.Tab("Video"):
                gr.HTML(_section_head(
                    "AI · Video", "Video Upscaler",
                    "Upscale a clip frame-by-frame (offline, audio kept). ×2 is "
                    "faster and flickers less than ×4. Long clips take a while — "
                    "needs ffmpeg installed.",
                    icon=ICON_AI,
                ))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Input video", sources=["upload"])
                        vid_model = gr.Dropdown(
                            _MODEL_CHOICES, value="realesrgan-x2plus",
                            label="Upscale model",
                            info="×2 recommended for video — faster, less shimmer "
                            "between frames.",
                        )
                        vid_sharpen = gr.Slider(
                            0.0, 3.0, value=0.0, step=0.1,
                            label="Sharpen per frame — 0 = off",
                            info="Be gentle on video; sharpening can amplify "
                            "frame-to-frame flicker.",
                        )
                        with gr.Accordion("Advanced", open=False):
                            vid_device = gr.Dropdown(
                                _DEVICES, value="auto", label="Device",
                                info="auto picks a GPU if available, else CPU.",
                            )
                            vid_tile = gr.Slider(
                                0, 1024, value=512, step=64,
                                label="Tile size (0 = off)",
                                info="Lower if you hit out-of-memory on big frames.",
                            )
                        vid_btn = gr.Button("Upscale video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        vid_out = gr.Video(label="Result")
                        vid_info = gr.Markdown()

            # ---- Tab 3: Convert (all conversions, picked via dropdown) ----
            with gr.Tab("Convert"):
                gr.HTML(_section_head(
                    "Convert", "Convert & Documents",
                    "Pick what you want to do — change image format, build a PDF "
                    "from images, or split a PDF back into images.",
                    icon=ICON_CONVERT,
                ))
                method = gr.Dropdown(
                    _CONVERT_METHODS, value=_CONVERT_METHODS[0],
                    label="What do you want to do?",
                    info="Switch between converting an image's format, building a "
                    "PDF from images, or splitting a PDF back into images.",
                )

                # -- Method A: change image format --
                with gr.Column(visible=True) as grp_format:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            conv_in = gr.Image(
                                label="Image", type="pil",
                                sources=["upload", "clipboard"], height=300,
                                elem_classes="drop",
                            )
                            conv_fmt = gr.Dropdown(
                                list(FORMATS), value="PNG", label="Convert to",
                                info="PNG / TIFF keep full quality · JPEG, WebP, "
                                "AVIF, HEIC are smaller but lossy.",
                            )
                            conv_quality = gr.Slider(
                                1, 100, value=90, step=1, label="Quality (lossy)",
                                info="Only affects lossy formats. Higher = better "
                                "looking but larger file.",
                            )
                            conv_lossless = gr.Checkbox(
                                value=False, label="Lossless WebP",
                                info="Encode WebP with no quality loss (bigger file).",
                            )
                            conv_btn = gr.Button("Convert", variant="primary", size="lg")
                        with gr.Column(scale=1):
                            conv_file = gr.File(label="Download converted file")
                            conv_info = gr.Markdown()

                # -- Method B: images -> PDF --
                with gr.Column(visible=False) as grp_topdf:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            pdf_imgs_in = gr.File(
                                label="Images (multiple = multi-page, in order)",
                                file_count="multiple", file_types=["image"],
                                elem_classes="drop",
                            )
                            pdf_build_btn = gr.Button(
                                "Build PDF", variant="primary", size="lg"
                            )
                        with gr.Column(scale=1):
                            pdf_build_out = gr.File(label="Download PDF")
                            pdf_build_info = gr.Markdown()

                # -- Method C: PDF -> images --
                with gr.Column(visible=False) as grp_frompdf:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            pdf_in = gr.File(
                                label="PDF", file_count="single",
                                file_types=[".pdf"], elem_classes="drop",
                            )
                            pdf_dpi = gr.Slider(
                                72, 300, value=150, step=1, label="Render DPI",
                                info="Higher = sharper, larger PNGs. 150 is a good "
                                "default; 300 for print quality.",
                            )
                            pdf_extract_btn = gr.Button(
                                "Extract pages", variant="primary", size="lg"
                            )
                        with gr.Column(scale=1):
                            pdf_extract_out = gr.File(label="Download pages (ZIP)")
                            pdf_gallery = gr.Gallery(label="Pages", columns=4, height=220)
                            pdf_extract_info = gr.Markdown()

                method.change(
                    _switch_method, method, [grp_format, grp_topdf, grp_frompdf]
                )

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
        vid_btn.click(
            upscale_video_ui,
            [vid_in, vid_model, vid_sharpen, vid_device, vid_tile],
            [vid_out, vid_info],
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
