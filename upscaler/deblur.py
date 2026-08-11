"""Model-based deblur stage using NAFNet (GoPro motion-deblur weights).

NAFNet's channel attention pools globally, so the network is run on the whole
image (it pads internally to a valid size) rather than tiled — tiling would
introduce seams. Deblur is meant to run at the image's native resolution,
*before* upscaling, so memory is usually fine — but a large photo (say 12MP)
holds full-resolution activations for every encoder stage at once and does
overflow a GPU. Since tiling isn't an option, `deblur` retries on CPU: slower,
but the same pixels, and it beats failing the whole job.

That retry is only worth taking if the result fits in RAM. On a machine with
less memory than the image needs, torch doesn't fail — it pages, and a job that
would have errored in seconds instead crawls for hours while the whole system
swaps. So the fallback is gated on an estimate first, and refuses with an
actionable message when the image can't fit.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from upscaler.engine import _load_state_dict, resolve_device
from upscaler.models.nafnet import NAFNet
from upscaler.models.registry import DeblurSpec, resolve_deblur_model
from upscaler.models.weights import ensure_weights


def _empty_cache(device: torch.device) -> None:
    """Release cached blocks so the CPU retry isn't fighting the failed run."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


# Peak working set per (model width x pixel), measured on CPU by running the
# width-64 model at 256/384/512/768px and fitting the high-water RSS. Predicts
# 9.4 GB for a 2560x1440 image, against the 9.5 GiB the MPS allocator actually
# demanded at that size — close enough to size a guard with.
_PEAK_BYTES_PER_UNIT = 40

# Fraction of total RAM the deblur may plan to occupy. The rest is the OS, the
# app itself (~700 MB with weights loaded), and the margin that keeps an
# estimate this rough from tipping the machine into swap.
_RAM_BUDGET = 0.5


class DeblurTooLargeError(RuntimeError):
    """The image can't be cleaned up in the memory this machine has.

    Carries a message meant to be shown to the user as-is.
    """


def _total_ram_bytes() -> Optional[int]:
    """Physical RAM, or None if the platform won't say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows has no sysconf
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys) or None
    except Exception:
        return None


class Deblurrer:
    """Wraps a pretrained NAFNet deblur model for whole-image inference.

    Example
    -------
    >>> db = Deblurrer()
    >>> sharp = db.deblur(Image.open("blurry.jpg"))
    """

    def __init__(self, model: Optional[str] = None, device: str = "auto"):
        self.spec: DeblurSpec = resolve_deblur_model(model)
        self.device = resolve_device(device)

        net = NAFNet(
            img_channel=3,
            width=self.spec.width,
            middle_blk_num=self.spec.middle_blk_num,
            enc_blk_nums=self.spec.enc_blk_nums,
            dec_blk_nums=self.spec.dec_blk_nums,
        )
        net.load_state_dict(_load_state_dict(ensure_weights(self.spec)), strict=True)
        net.eval()
        net.to(self.device)
        self.net = net

    def _run(self, x: torch.Tensor, device: torch.device) -> np.ndarray:
        """Run the net on `device`, returning HWC float output on the host."""
        self.net.to(device)
        y = self.net(x.to(device))
        return y.clamp_(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def estimate_bytes(self, width_px: int, height_px: int) -> int:
        """Rough peak working set for deblurring an image of this size."""
        return _PEAK_BYTES_PER_UNIT * self.spec.width * width_px * height_px

    def _check_cpu_headroom(self, width_px: int, height_px: int) -> None:
        """Raise DeblurTooLargeError if a CPU run would page instead of fit."""
        total = _total_ram_bytes()
        if total is None:  # unknown platform — let it try rather than block
            return
        need = self.estimate_bytes(width_px, height_px)
        budget = int(total * _RAM_BUDGET)
        if need <= budget:
            return
        gb = 1 << 30
        max_px = budget // (_PEAK_BYTES_PER_UNIT * self.spec.width)
        smaller = "" if self.spec.width <= 32 else (
            ", switch to a width-32 clean-up model (about half the memory)")
        raise DeblurTooLargeError(
            f"Clean-up needs roughly {need / gb:.1f} GB for a "
            f"{width_px}x{height_px} image, and this machine has "
            f"{total / gb:.0f} GB of RAM. Turn off Clean up{smaller}, or scale "
            f"the image down to about {max_px / 1e6:.1f} megapixels first."
        )

    @torch.inference_mode()
    def deblur(self, image: Image.Image) -> Image.Image:
        """Deblur a PIL image, returning a same-resolution result.

        Falls back to CPU if the accelerator runs out of memory, provided the
        image fits in RAM — see the module docstring for why this can't be
        solved by tiling, and why the fallback has to be gated.

        Raises DeblurTooLargeError (message fit to show the user) when neither
        the accelerator nor RAM can hold the image.
        """
        rgb = image.convert("RGB")
        # A CPU-only machine gets the same answer up front, without waiting for
        # a doomed run to fail first.
        if self.device.type == "cpu":
            self._check_cpu_headroom(rgb.width, rgb.height)
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        try:
            out = self._run(x, self.device)
        except RuntimeError as e:
            if self.device.type == "cpu" or "out of memory" not in str(e).lower():
                raise
            _empty_cache(self.device)
            try:
                self._check_cpu_headroom(rgb.width, rgb.height)
            except DeblurTooLargeError as too_big:
                raise too_big from e
            print(
                f"clean-up: {self.device.type} ran out of memory on a "
                f"{rgb.width}x{rgb.height} image — retrying on CPU (slower)",
                file=sys.stderr,
            )
            try:
                out = self._run(x, torch.device("cpu"))
            finally:
                self.net.to(self.device)  # leave the model where callers expect it
        return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")
