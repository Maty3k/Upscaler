"""Save / load shareable Lian Li panel layouts as self-contained .json.

A layout captures the composition + overlays (a :class:`~upscaler.panel.PanelParams`),
NOT the source media — re-upload your image/video after loading one. Sticker
images are base64-PNG embedded so a saved file renders identically on another
machine. Mirrors ``config.py``'s ``~/.upscaler`` convention and its best-effort,
unknown-key-tolerant loading (so layouts saved by older/newer versions still open).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image

from upscaler.panel import PanelParams

SCHEMA = 1
PRESETS_DIR = Path(os.environ.get(
    "UPSCALER_PANEL_PRESETS", Path.home() / ".upscaler" / "panel_presets"))

_PARAM_KEYS = ("orientation", "fit", "zoom", "off_x", "off_y", "bg_type",
               "bg_color", "bg_color2", "bg_angle")


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_img(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA")


def to_dict(p: PanelParams) -> dict:
    """Serialize to a JSON-safe dict (sticker PIL images → base64 PNG)."""
    overlays = []
    for ov in p.overlays:
        # Drop private (underscore) keys — per-export caches like _scroll_span
        # must not leak into shared layout files.
        o = {k: v for k, v in ov.items() if not str(k).startswith("_")}
        img = o.pop("image", None)
        if img is not None:
            o["image_b64"] = _img_to_b64(img)
        overlays.append(o)
    return {
        "schema": SCHEMA,
        "params": {k: getattr(p, k) for k in _PARAM_KEYS},
        "overlays": overlays,
    }


# Numeric overlay fields and their expected types — hand-edited layouts often
# quote numbers ("size": "180"), which would load fine and then crash every
# preview render. Coerce here; unparseable values fall back to the default.
_OVERLAY_NUM_FIELDS = {
    "size": int, "x": float, "y": float, "rotation": float, "stroke_w": int,
    "speed": float, "cps": float, "scale": float, "opacity": float,
}


def from_dict(d: dict) -> PanelParams:
    """Rebuild a PanelParams, tolerant of missing/unknown keys (best-effort)."""
    d = d or {}
    params = d.get("params") or {}
    defaults = PanelParams()
    kw = {}
    for k in _PARAM_KEYS:
        dv = getattr(defaults, k)
        v = params.get(k, dv)
        try:
            kw[k] = type(dv)(v)
        except (TypeError, ValueError):
            kw[k] = dv
    overlays = []
    for raw in d.get("overlays") or []:
        o = dict(raw)
        b64 = o.pop("image_b64", None)
        if b64:
            try:
                o["image"] = _b64_to_img(b64)
            except Exception:
                continue  # skip a corrupt sticker rather than fail the whole load
        for key, cast in _OVERLAY_NUM_FIELDS.items():
            if key in o:
                try:
                    o[key] = cast(float(o[key]))
                except (TypeError, ValueError):
                    o.pop(key)  # downstream .get(key, default) fills it in
        overlays.append(o)
    return PanelParams(overlays=overlays, **kw)


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "", (name or "").strip()) or "layout"
    return name[:60]


def save_layout(p: PanelParams, name: str) -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRESETS_DIR / f"{_safe_name(name)}.json"
    dest.write_text(json.dumps(to_dict(p), indent=2))
    return dest


def load_layout(path) -> PanelParams:
    return from_dict(json.loads(Path(path).read_text()))
