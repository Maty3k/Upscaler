"""The Upscaler engine: load a pretrained model and run tiled inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from upscaler.memory import (AVAILABLE_BUDGET, RAM_BUDGET,
                             available_ram_bytes, budget_from,
                             describe_shortfall, device_budget_bytes,
                             total_ram_bytes)
from upscaler.models.registry import ModelSpec, resolve_model
from upscaler.models.rrdbnet import RRDBNet
from upscaler.models.weights import ensure_weights

# Bytes of host memory per *output* pixel at the peak of `upscale`. Traceable
# to specific allocations rather than fitted: the uint8 accumulation buffer
# `_run_tiled` quantizes each tile into is 3 channels x 1 byte, and the PIL
# copy `Image.fromarray` makes adds another 3. (This was 30 when the tiler
# stitched a float32 buffer and `np.round` made a second one — per-tile
# quantization removed both, which is why the guard's whole cost model was
# re-derived below rather than just re-priced.)
_OUTPUT_BYTES_PER_PIXEL = 6

# The input as `upscale` builds it: float32 x 3 channels per *input* pixel,
# alive on the device for the entire run — and on unified memory the device
# draws from the same RAM the buffers above do.
_INPUT_BYTES_PER_PIXEL = 12

# The model's own working set, which tiling bounds but does not remove — and
# which the old guard didn't model at all. That omission is how the incident
# escaped it: on unified memory an oversized run doesn't raise, it pages (the
# job that proved it sat at 7.9 GB RSS with ~40k pageins/10s on an 8 GB M2,
# with no exception ever thrown). Measured on that machine at tile=256,
# tile_pad=32, in MB:
#   accel x2: 1130 — MPS driver_allocated plateaus after the first tile and
#             stays flat to the last (1130.5 / 1130.4 / 1130.4 / 1130.4);
#             64 MB of that is weights, the rest cached tile blocks. On
#             unified memory this reservation is *wired* host RAM.
#   accel x4: 2146 — same flat shape; the extra ~1 GB is the x4 upsampling
#             tail (conv_up2/conv_hr activations at output resolution).
#   cpu   x2: 1371 — peak RSS 1708 MB minus the 337 MB post-load baseline.
#             Larger than the MPS figure, not smaller: CPU inference
#             materializes intermediates the MPS caching allocator recycles.
#   cpu   x4: not measured; the accel pair's x4 tail (2146 - 1130) is added
#             to the measured cpu x2 figure.
_FIXED_WORKING_SET_MB = {
    "cpu": {2: 1371, 4: 2387},
    "accel": {2: 1130, 4: 2146},
}

# The table was measured at this tile size and padding. What the net actually
# saw per block was therefore (256 + 2*32)^2 = 320^2 *processed* pixels — the
# tile plus the context _run_tiled pads it with — and the working set scales
# with that processed area, not the bare tile square. (Scaling by (tile/256)^2
# instead overcharges tile 512 by ~1.23x and undercharges tile 64 by ~2.5x.)
# Whole image when tiling is off. Rough (it ignores the fixed weights), but
# the guard only needs the order of magnitude to be right.
_FIXED_MEASURED_TILE = 256
_FIXED_MEASURED_PAD = 32


def _fixed_working_set_bytes(device_type: str, scale: int, tile: int,
                             width_px: int, height_px: int,
                             tile_pad: int = _FIXED_MEASURED_PAD) -> int:
    """Size-independent working set of the model for this run, in bytes.

    ``tile <= 0`` means the whole image is one tile, so the "fixed" cost grows
    with the image itself — exactly the unguarded whole-image forward that
    used to swap instead of raise. An image smaller than the tile is likewise
    its own (single) tile — and tiles are clipped *per dimension*, so a
    skinny panorama is charged its real ``min(tile + pads, dim)`` extent in
    each direction, never the full square.
    """
    table = _FIXED_WORKING_SET_MB["cpu" if device_type == "cpu" else "accel"]
    mb = table.get(scale)
    if mb is None:
        # Unmeasured scale: the measured x2/x4 pair fits mb = base + tail*s^2
        # (the tail is the upsampling stages running at output resolution).
        tail = (table[4] - table[2]) / (16 - 4)
        mb = table[2] + tail * (scale * scale - 4)
    if tile > 0:
        # The largest block the net sees: tile plus its pad on both sides,
        # clamped per dimension to the image (the tiler never pads past the
        # border).
        area = (min(tile + 2 * tile_pad, width_px)
                * min(tile + 2 * tile_pad, height_px))
    else:
        area = width_px * height_px
    measured_area = (_FIXED_MEASURED_TILE + 2 * _FIXED_MEASURED_PAD) ** 2
    return int(mb * (1 << 20) * (area / float(measured_area)))


class OutputTooLargeError(RuntimeError):
    """The upscaled result won't fit in memory. Message is user-facing."""


class CancelledError(RuntimeError):
    """Raised when a user cancels a running job. Subclasses RuntimeError so the
    app's existing ``except RuntimeError`` handlers swallow it without a scary
    traceback."""


def tile_count(w: int, h: int, tile: int) -> int:
    """How many tiles :meth:`Upscaler._run_tiled` will process for a ``w``×``h``
    image. Returns 1 when tiling is off (``tile <= 0``)."""
    if tile <= 0:
        return 1
    return ((w + tile - 1) // tile) * ((h + tile - 1) // tile)


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


_SPANDREL_INSTALLED = False  # spandrel_extra_arches.install() must run only once


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
    # Register GFPGAN/CodeFormer/HAT/DRCT/… into spandrel's registry — but only
    # once per process: install() raises DuplicateArchitectureError if re-run, so
    # loading a second spandrel model (e.g. GFPGAN after CodeFormer) would crash.
    global _SPANDREL_INSTALLED
    if not _SPANDREL_INSTALLED:
        _sea.install()
        _SPANDREL_INSTALLED = True
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
    def upscale(
        self,
        image: Image.Image,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Image.Image:
        """Upscale a PIL RGB image by the model's native scale factor.

        ``progress_cb(done, total)`` is called after each tile (or once, (1, 1),
        when tiling is off). ``should_cancel()`` is polled at each tile boundary;
        returning True raises :class:`CancelledError`.
        """
        rgb = image.convert("RGB")
        self.check_output_fits(rgb.width, rgb.height)
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()

        if self.tile > 0:
            # The tiler quantizes straight into a uint8 buffer, so the result
            # needs no further conversion here.
            return Image.fromarray(
                self._run_tiled(x, progress_cb=progress_cb,
                                should_cancel=should_cancel),
                mode="RGB",
            )

        if should_cancel and should_cancel():
            raise CancelledError("Cancelled.")
        out = self._net(x)
        if progress_cb:
            progress_cb(1, 1)
        arr = out.clamp_(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        # Drop the input before np.round makes its float copy — it's dead
        # weight now. On MPS/CUDA `del out` also frees the device-side result
        # (`.cpu()` above copied it to the host), halving that path's peak; on
        # the CPU device it frees nothing yet, because the .float().cpu()
        # chain is a no-op on a host fp32 tensor and `arr` still shares
        # `out`'s storage until np.round materializes its copy.
        del out, x
        return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")

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

    def output_bytes(self, width_px: int, height_px: int) -> int:
        """Host memory the result of this upscale will occupy at its peak."""
        s = self.scale
        return _OUTPUT_BYTES_PER_PIXEL * width_px * s * height_px * s

    def check_output_fits(self, width_px: int, height_px: int) -> None:
        """Raise OutputTooLargeError if this run can't be held in memory.

        Three allocations coexist at the peak: the model's working set (fixed
        once tiling is on — see _FIXED_WORKING_SET_MB), the float32 input
        tensor, and the output buffers. All three are checked here, up front,
        because torch's response to not fitting is not an error: on unified
        memory the MPS allocator raises only past ~1.7x its recommended
        working set and never consults free RAM, so an oversized run
        "succeeds" into swap — minutes to hours of thrashing for a job that
        can't finish, with nothing on stderr.
        """
        fixed = _fixed_working_set_bytes(self.device.type, self.scale,
                                         self.tile, width_px, height_px,
                                         self.tile_pad)
        gb = 1 << 30
        if self.device.type != "cpu":
            # The accelerator holds the whole working set at once, and its
            # budget can be far tighter than the host's — a busy 8 GB M2
            # offers MPS well under 2 GB while the tile=512 working set is
            # over 4. Refusing here names the one lever that changes it.
            dev_budget = device_budget_bytes(self.device)
            if dev_budget is not None and fixed > dev_budget:
                per = (f"at Tile size {self.tile}" if self.tile > 0 else
                       f"untiled at {width_px}x{height_px}")
                raise OutputTooLargeError(
                    f"The model needs about {fixed / gb:.1f} GB of working "
                    f"memory {per}, and {self.device.type} can spare about "
                    f"{dev_budget / gb:.1f} GB. Lower the Tile size (working "
                    f"memory grows with its square), or close some apps and "
                    f"try again."
                )
        total, available = total_ram_bytes(), available_ram_bytes()
        budget = budget_from(total, available)
        if budget is None:  # unknown platform — let it try rather than block
            return
        if self.tile > 0 and total is not None and available is not None:
            # RAM_BUDGET's half-of-total share was calibrated when `need` was
            # a few transient buffers. A tiled run's fixed term is different:
            # a bounded, *measured* resident set that may legitimately hold
            # more than half of a small machine's RAM — provided that much is
            # actually free. (At the shipped Tile 512 it is ~3.6-7.6 GB; the
            # half-of-total cap alone would refuse that on every 8/16 GB
            # machine however idle, with "close some apps" advice that could
            # never succeed.) So with tiling on, the fixed term answers only
            # to free RAM, while the calibrated share keeps guarding the
            # transient buffers: need <= min(free share, total share + fixed)
            # is exactly "everything fits in what's free" AND "the buffers
            # fit the calibrated share". Untiled, `fixed` *is* the rough
            # whole-image extrapolation that the cap exists for, and keeps it.
            budget = min(int(available * AVAILABLE_BUDGET),
                         int(total * RAM_BUDGET) + fixed)
        need = (fixed + _INPUT_BYTES_PER_PIXEL * width_px * height_px
                + self.output_bytes(width_px, height_px))
        if need <= budget:
            return
        room = describe_shortfall(need, total, available)
        # The levers a user actually has, in the order they should try them.
        smaller_scale = "" if self.scale <= 2 else (
            " Use a ×2 model instead of ×4 (a quarter of the pixels), or")
        headroom = budget - fixed
        # Bytes each input pixel adds: its float32 copy plus scale^2 output px.
        per_px = (_INPUT_BYTES_PER_PIXEL
                  + _OUTPUT_BYTES_PER_PIXEL * self.scale * self.scale)
        if headroom > 0:
            advice = (f"{smaller_scale} start from an image of about "
                      f"{headroom / per_px / 1e6:.1f} megapixels or set a "
                      f"smaller Output size.")
        elif smaller_scale:
            advice = (f"{smaller_scale} lower the Tile size (working memory "
                      f"grows with its square).")
        else:
            advice = (" Lower the Tile size (working memory grows with its "
                      "square), or close some apps and try again.")
        raise OutputTooLargeError(
            f"A ×{self.scale} upscale of {width_px}x{height_px} produces "
            f"{width_px * self.scale}x{height_px * self.scale} and needs about "
            f"{need / gb:.1f} GB, and {room}.{advice}"
        )

    def upscale_file(self, src, dst) -> Image.Image:
        result = self.upscale(Image.open(src))
        result.save(dst)
        return result

    # -- tiling -----------------------------------------------------------

    def _run_tiled(
        self,
        x: torch.Tensor,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Process a large image tile-by-tile, returning the HWC uint8 result.

        Each tile is padded with surrounding context (``tile_pad``) to avoid
        seams, then the padded border is cropped off the output before
        stitching. The tile grid partitions the input, so every output pixel
        is written exactly once — and 8-bit quantization is elementwise, so
        quantizing each tile as it lands is bit-identical to quantizing a
        stitched float image. That is why the accumulator is uint8: the old
        float32 buffer plus the whole-image `np.round` twin it forced cost
        24 bytes per output pixel (a 4000×3000 ×4 output alone ~2.3 GB of
        float); this holds 3, and the float copies stay tile-sized.
        """
        _b, _c, h, w = x.shape
        s = self.scale
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
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
                # input tile + padding, clamped to image bounds
                px0, py0 = max(x0 - self.tile_pad, 0), max(y0 - self.tile_pad, 0)
                px1, py1 = min(x1 + self.tile_pad, w), min(y1 + self.tile_pad, h)

                tile_out = self._net(x[:, :, py0:py1, px0:px1])

                # map the (unpadded) tile region into the padded output tile
                ox0, oy0 = (x0 - px0) * s, (y0 - py0) * s
                ox1, oy1 = ox0 + (x1 - x0) * s, oy0 + (y1 - y0) * s
                # clamp on the device, quantize on the host — same ops in the
                # same order the whole-image conversion used (fp16 tiles are
                # upcast by .float() exactly as the old final .float() did).
                arr = (tile_out[:, :, oy0:oy1, ox0:ox1].clamp_(0, 1)
                       .squeeze(0).permute(1, 2, 0).float().cpu().numpy())
                out[y0 * s:y1 * s, x0 * s:x1 * s] = (
                    np.round(arr * 255.0).astype(np.uint8)
                )
                done += 1
                if progress_cb:
                    progress_cb(done, total)
        return out
