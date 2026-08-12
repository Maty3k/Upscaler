"""Torch-free inference via ONNX Runtime.

These engines import only ``onnxruntime``, ``numpy``, and ``Pillow`` — no torch.
(The one-time export in ``onnx_export`` does need torch; once the ``.onnx`` file
is cached, inference is torch-free and often faster than torch on CPU.)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from upscaler.engine import (CancelledError, OutputTooLargeError,
                             _OUTPUT_BYTES_PER_PIXEL)
from upscaler.memory import (available_ram_bytes, budget_from,
                             describe_shortfall, total_ram_bytes)
from upscaler.models.registry import (
    DeblurSpec,
    ModelSpec,
    resolve_deblur_model,
    resolve_model,
)
from upscaler.onnx_export import export_deblur, export_upscale

CancelCb = Optional[Callable[[], bool]]
ProgressCb = Optional[Callable[[int, int], None]]


def _providers(device: str) -> list[str]:
    avail = ort.get_available_providers()
    if device in ("cuda", "auto") and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # DirectML (the onnxruntime-directml build) reaches any DX12 GPU on native
    # Windows — AMD/Intel/NVIDIA — which torch cannot do for AMD cards.
    if device in ("cuda", "auto", "gpu", "dml") and "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _session(model_path, device: str) -> "ort.InferenceSession":
    providers = _providers(device)
    so = ort.SessionOptions()
    if providers[0] == "DmlExecutionProvider":
        # Per ORT docs DirectML needs memory patterns off (our spatial dims are
        # dynamic, so the pattern cache would be wrong anyway).
        so.enable_mem_pattern = False
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


def _to_chw(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])


def _to_image(arr: np.ndarray) -> Image.Image:
    out = np.clip(arr, 0.0, 1.0)[0].transpose(1, 2, 0)
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")


class OnnxUpscaler:
    """RRDBNet upscaler running on ONNX Runtime (torch-free inference)."""

    def __init__(
        self,
        model: Optional[str] = None,
        scale: Optional[int] = None,
        device: str = "cpu",
        tile: int = 512,
        tile_pad: int = 32,
    ):
        self.spec: ModelSpec = resolve_model(model=model, scale=scale)
        self.scale = self.spec.scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.sess = _session(export_upscale(self.spec), device)
        self.provider = self.sess.get_providers()[0]
        self._inp = self.sess.get_inputs()[0].name

    def _run(self, x: np.ndarray) -> np.ndarray:
        # ×2 / ×1 models pixel-unshuffle by 2 / 4, so the input H/W must be a
        # multiple of that; pad odd-sized images/tiles up and crop the result
        # back (mirrors OnnxDeblurrer). ×4 needs no padding.
        m = 2 if self.scale == 2 else 4 if self.scale == 1 else 1
        _, _, h, w = x.shape
        if m > 1:
            ph, pw = (-h) % m, (-w) % m
            if ph or pw:
                x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
        out = self.sess.run(None, {self._inp: x.astype(np.float32)})[0]
        return out[:, :, : h * self.scale, : w * self.scale]

    def upscale(
        self,
        image: Image.Image,
        progress_cb: ProgressCb = None,
        should_cancel: CancelCb = None,
    ) -> Image.Image:
        """Mirror of the torch engine's signature, so callers can drive per-tile
        progress and cooperative cancel regardless of backend."""
        self.check_output_fits(image.width, image.height)
        x = _to_chw(image)
        if self.tile > 0:
            out = self._run_tiled(x, progress_cb=progress_cb, should_cancel=should_cancel)
        else:
            if should_cancel and should_cancel():
                raise CancelledError("Cancelled.")
            out = self._run(x)
            if progress_cb:
                progress_cb(1, 1)
        return _to_image(out)

    def output_bytes(self, width_px: int, height_px: int) -> int:
        """Peak host memory the result will occupy — same shape as the torch
        engine's, since the output buffers and conversions are the same."""
        s = self.scale
        return _OUTPUT_BYTES_PER_PIXEL * width_px * s * height_px * s

    def check_output_fits(self, width_px: int, height_px: int) -> None:
        """Raise OutputTooLargeError if the result can't be held in memory.

        The ONNX path allocates the same whole-output buffer as the torch one,
        so it needs the same guard — a DirectML GPU doesn't make host memory
        any bigger.
        """
        budget = budget_from(total_ram_bytes(), available_ram_bytes())
        if budget is None:
            return
        need = self.output_bytes(width_px, height_px)
        if need <= budget:
            return
        gb = 1 << 30
        room = describe_shortfall(need, total_ram_bytes(), available_ram_bytes())
        max_px = budget / (_OUTPUT_BYTES_PER_PIXEL * self.scale * self.scale)
        raise OutputTooLargeError(
            f"A x{self.scale} upscale of {width_px}x{height_px} needs about "
            f"{need / gb:.1f} GB, and {room}. Use a smaller scale, start from "
            f"an image of about {max_px / 1e6:.1f} megapixels, or set a "
            f"smaller Output size."
        )

    def upscale_file(self, src, dst) -> Image.Image:
        result = self.upscale(Image.open(src))
        result.save(dst)
        return result

    def _run_tiled(
        self,
        x: np.ndarray,
        progress_cb: ProgressCb = None,
        should_cancel: CancelCb = None,
    ) -> np.ndarray:
        """numpy mirror of engine.tiled inference (RRDBNet is fully local)."""
        b, c, h, w = x.shape
        s = self.scale
        out = np.zeros((b, c, h * s, w * s), dtype=np.float32)
        n_x = (w + self.tile - 1) // self.tile
        n_y = (h + self.tile - 1) // self.tile
        total = n_x * n_y
        done = 0
        for ty in range(n_y):
            for tx in range(n_x):
                if should_cancel and should_cancel():
                    raise CancelledError("Cancelled.")
                x0, y0 = tx * self.tile, ty * self.tile
                x1, y1 = min(x0 + self.tile, w), min(y0 + self.tile, h)
                px0, py0 = max(x0 - self.tile_pad, 0), max(y0 - self.tile_pad, 0)
                px1, py1 = min(x1 + self.tile_pad, w), min(y1 + self.tile_pad, h)
                tile_out = self._run(x[:, :, py0:py1, px0:px1])
                ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
                ox1, oy1 = ox0 + (x1 - x0) * s, oy0 + (y1 - y0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = tile_out[:, :, oy0:oy1, ox0:ox1]
                done += 1
                if progress_cb:
                    progress_cb(done, total)
        return out


class OnnxDeblurrer:
    """NAFNet deblur running on ONNX Runtime (torch-free inference).

    The exported graph is the shape-preserving body, which assumes spatial dims
    divisible by the padder size — so we pad here (matching the torch path's
    zero padding) and crop the result back.
    """

    def __init__(self, model: Optional[str] = None, device: str = "cpu"):
        self.spec: DeblurSpec = resolve_deblur_model(model)
        self.padder = 2 ** len(self.spec.enc_blk_nums)
        self.sess = _session(export_deblur(self.spec), device)
        self.provider = self.sess.get_providers()[0]
        self._inp = self.sess.get_inputs()[0].name

    def deblur(self, image: Image.Image) -> Image.Image:
        x = _to_chw(image)
        _, _, h, w = x.shape
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)))  # constant 0, matches torch
        out = self.sess.run(None, {self._inp: x.astype(np.float32)})[0]
        return _to_image(out[:, :, :h, :w])
