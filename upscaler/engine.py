"""The Upscaler engine: load a pretrained model and run tiled inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from upscaler.models.registry import ModelSpec, resolve_model
from upscaler.models.rrdbnet import RRDBNet
from upscaler.models.weights import ensure_weights


def resolve_device(device: str = "auto") -> torch.device:
    """Map 'auto'/'cpu'/'cuda'/'mps' to a concrete torch.device.

    'auto' prefers CUDA, then Apple-Silicon MPS, then CPU.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _convert_esrgan_oldarch(sd: dict) -> dict:
    """Remap an *old-arch* ESRGAN state dict (xinntao/ESRGAN ``model.*`` keys, as
    used by many community models — 4x-UltraSharp, Remacri, NMKD, …) to the
    Real-ESRGAN RRDBNet naming the vendored generator expects. Only key names are
    translated; the weights are untouched. A new-arch dict is returned unchanged.
    """
    if not any(k.startswith("model.") for k in sd):
        return sd
    rdb = re.compile(r"^model\.1\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)$")
    trunk = re.compile(r"^model\.1\.sub\.\d+\.(weight|bias)$")  # conv after the RRDBs
    head = {"model.0.": "conv_first.", "model.3.": "conv_up1.",
            "model.6.": "conv_up2.", "model.8.": "conv_hr.", "model.10.": "conv_last."}
    out = {}
    for k, v in sd.items():
        m = rdb.match(k)
        if m:
            i, j, c, wb = m.groups()
            out[f"body.{i}.rdb{j}.conv{c}.{wb}"] = v
            continue
        if trunk.match(k):
            out[f"conv_body.{k.rsplit('.', 1)[1]}"] = v
            continue
        for old, new in head.items():
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            out[k] = v
    return out


def _load_state_dict(path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # Official Real-ESRGAN checkpoints wrap weights under params_ema / params.
    for key in ("params_ema", "params"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    # Community ESRGAN .pth often use the old `model.*` layer naming — translate.
    return _convert_esrgan_oldarch(ckpt)


@dataclass(frozen=True)
class LoadedModel:
    """Everything a caller needs after loading a checkpoint through spandrel.

    Bundling these means callers never re-derive ``scale``/``use_fp16``/padding
    from the raw descriptor — they read them off here. ``net`` is the spandrel
    ``ImageModelDescriptor``, already moved to the device and put in eval mode,
    and is callable as ``net(tensor) -> tensor``.
    """

    net: object
    scale: int
    use_fp16: bool
    pad_to: int
    input_channels: int
    output_channels: int


def load_spandrel(
    path,
    device: torch.device,
    *,
    fp16: bool = False,
    require_channels: Optional[int] = 3,
) -> LoadedModel:
    """Load any spandrel-supported checkpoint and wrap it in a ``LoadedModel``.

    This is the single place in the codebase that imports spandrel, registers the
    extra architectures, and runs ``ModelLoader().load_from_file(...).to().eval()``
    — the engine, face restoration, colorize and inpaint all funnel through here
    so there is exactly one spandrel-loading code path.

    ``path`` is an already-resolved weights file (callers pass
    ``ensure_weights(spec)``), and ``device`` is a concrete ``torch.device``.

    ``require_channels`` guards the descriptor's input/output channel count: the
    tiled RGB upscale path passes ``3``; pass ``None`` to accept any shape (a
    1→3 colorizer, or a masked inpainter that is called with two tensors).
    """
    try:
        import spandrel
        import spandrel_extra_arches as _sea
    except ImportError as e:
        raise RuntimeError(
            "This model needs extra packages. Install them with: "
            'pip install -e ".[face]"'
        ) from e
    _sea.install()  # register GFPGAN/CodeFormer/HAT/DRCT/… into spandrel's registry
    desc = spandrel.ModelLoader().load_from_file(str(path))
    desc = desc.to(device).eval()

    ic = getattr(desc, "input_channels", None)
    oc = getattr(desc, "output_channels", None)
    if require_channels is not None and (ic != require_channels or oc != require_channels):
        raise RuntimeError(
            f"This model expects {ic}→{oc} channels, but this feature needs "
            f"{require_channels}-channel RGB. Pick a different model."
        )

    sr = getattr(desc, "size_requirements", None)
    pad_to = max(int(getattr(sr, "multiple_of", 1) or 1), 1) if sr is not None else 1
    use_fp16 = bool(
        fp16
        and getattr(device, "type", None) == "cuda"
        and getattr(desc, "supports_half", False)
    )
    return LoadedModel(
        net=desc,
        scale=int(getattr(desc, "scale", 1) or 1),
        use_fp16=use_fp16,
        pad_to=pad_to,
        input_channels=ic if ic is not None else (require_channels or 3),
        output_channels=oc if oc is not None else (require_channels or 3),
    )


class Upscaler:
    """Wraps a pretrained Real-ESRGAN generator for inference.

    Example
    -------
    >>> up = Upscaler(scale=4)
    >>> up.upscale_file("in.jpg", "out.png")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        scale: Optional[int] = None,
        device: str = "auto",
        tile: int = 512,
        tile_pad: int = 32,
        fp16: bool = False,
    ):
        self.spec: ModelSpec = resolve_model(model=model, scale=scale)
        self.device = resolve_device(device)
        self.tile = tile
        self.tile_pad = tile_pad
        # fp16 is a CUDA-only win; CPU/MPS stay fp32 for correctness.
        self.use_fp16 = fp16 and self.device.type == "cuda"
        # Native RRDBNet computes its pad multiple from `scale`; the spandrel path
        # overrides this with the loaded model's size requirement.
        self._pad: Optional[int] = None

        if getattr(self.spec, "loader", "rrdbnet") == "spandrel":
            # Arch-agnostic path: spandrel auto-detects HAT/DRCT/SwinIR/… and the
            # existing tiler drives it. Scale comes from the model, not the spec.
            lm = load_spandrel(ensure_weights(self.spec), self.device, fp16=fp16)
            self.net = lm.net
            self._scale = lm.scale
            self.use_fp16 = lm.use_fp16
            self._pad = lm.pad_to
            if self.use_fp16:
                try:
                    self.net = self.net.half()
                except Exception:
                    self.use_fp16 = False
        else:
            self._scale = self.spec.scale
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=self.spec.scale,
                num_feat=self.spec.num_feat,
                num_block=self.spec.num_block,
                num_grow_ch=self.spec.num_grow_ch,
            )
            net.load_state_dict(_load_state_dict(ensure_weights(self.spec)), strict=True)
            net.eval()
            net.to(self.device)
            if self.use_fp16:
                net.half()
            self.net = net

    @property
    def scale(self) -> int:
        return self._scale

    # -- public API -------------------------------------------------------

    @torch.inference_mode()
    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale a PIL RGB image by the model's native scale factor."""
        rgb = image.convert("RGB")
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()

        out = self._run_tiled(x) if self.tile > 0 else self._net(x)

        out = out.clamp_(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")

    def _net(self, t: torch.Tensor) -> torch.Tensor:
        """Run the model, first padding spatial dims up to the multiple that
        RRDBNet's pixel_unshuffle requires (2 for ×2 models, 4 for ×1), then
        cropping the result back — so odd-sized images/tiles don't trip the
        ``spatial dims must be divisible by scale`` assert. (×4 needs no padding.)
        """
        if self._pad is not None:  # spandrel path: pad to the model's requirement
            m = self._pad
        else:  # native RRDBNet: pixel_unshuffle needs 2 (×2) / 4 (×1); ×4 none
            m = 2 if self.scale == 2 else 4 if self.scale == 1 else 1
        if m == 1:
            return self.net(t)
        _, _, h, w = t.shape
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        return self.net(t)[:, :, : h * self.scale, : w * self.scale]

    def upscale_file(self, src, dst) -> Image.Image:
        result = self.upscale(Image.open(src))
        result.save(dst)
        return result

    # -- tiling -----------------------------------------------------------

    def _run_tiled(self, x: torch.Tensor) -> torch.Tensor:
        """Process a large image tile-by-tile to bound memory.

        Each tile is padded with surrounding context (``tile_pad``) to avoid seams,
        then the padded border is cropped off the output before stitching.
        """
        b, c, h, w = x.shape
        s = self.scale
        out = x.new_zeros((b, c, h * s, w * s))
        n_x = (w + self.tile - 1) // self.tile
        n_y = (h + self.tile - 1) // self.tile

        for ty in range(n_y):
            for tx in range(n_x):
                x0, y0 = tx * self.tile, ty * self.tile
                x1, y1 = min(x0 + self.tile, w), min(y0 + self.tile, h)
                # input tile + padding, clamped to image bounds
                px0, py0 = max(x0 - self.tile_pad, 0), max(y0 - self.tile_pad, 0)
                px1, py1 = min(x1 + self.tile_pad, w), min(y1 + self.tile_pad, h)

                tile_out = self._net(x[:, :, py0:py1, px0:px1])

                # map the (unpadded) tile region into the padded output tile
                ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
                ox1, oy1 = ox0 + (x1 - x0) * s, oy0 + (y1 - y0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = tile_out[:, :, oy0:oy1, ox0:ox1]
        return out
