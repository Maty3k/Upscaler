"""Model-based deblur stage using NAFNet (GoPro motion-deblur weights).

NAFNet's channel attention pools globally, so the network is run on the whole
image (it pads internally to a valid size) rather than tiled — tiling would
introduce seams. Deblur is meant to run at the image's native resolution,
*before* upscaling, so memory is usually fine — but a large photo (say 12MP)
holds full-resolution activations for every encoder stage at once and does
overflow a GPU. Since tiling isn't an option, `deblur` recovers by shrinking
the activations instead, cheapest first:

1. half precision on the same device — halves every activation, and a
   2560x1440 image that overflows float32 on an 8 GB M2 finishes in ~27s;
2. CPU in float32 — same pixels, but minutes rather than seconds;
3. refuse, when even CPU would have to page.

Step 3 matters as much as the others. On a machine with less memory than the
image needs, torch doesn't fail — it pages, and a job that would have errored
in seconds instead crawls for hours while the whole system swaps. So the CPU
step is gated on an estimate and refuses with an actionable message.
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
    """Release cached blocks so the CPU retry isn't fighting the failed run.

    Best-effort: `torch.mps` exists on machines with no Metal backend at all
    and raises when called, and we'd rather run the retry than propagate a
    failure to free memory we were about to stop using anyway.
    """
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (RuntimeError, AttributeError):
        pass


# Peak working set per (model width x pixel), measured on CPU by running the
# width-64 model at 256/384/512/768px and fitting the high-water RSS. Predicts
# 9.4 GB for a 2560x1440 image, against the 9.5 GiB the MPS allocator actually
# demanded at that size — close enough to size a guard with.
_PEAK_BYTES_PER_UNIT = 40

# Fraction of total RAM the deblur may plan to occupy. The rest is the OS, the
# app itself (~700 MB with weights loaded), and the margin that keeps an
# estimate this rough from tipping the machine into swap.
_RAM_BUDGET = 0.5

# Fraction of *currently free* RAM it may take. Total-RAM sizing alone isn't
# enough: the same 2560x1440 image that ran in 31s in a fresh process thrashed
# for minutes in a long-lived server on the same 8 GB machine, because the
# memory was there in principle and gone in practice.
_AVAILABLE_BUDGET = 0.8

# Backends where reduced-precision inference is a real speedup rather than an
# emulated slowdown — on CPU it runs on software kernels and would be slower
# than the float32 run we're recovering from.
_HALF_PRECISION_DEVICES = frozenset({"cuda", "mps"})

# bfloat16, NOT float16. LayerNorm2d's eps of 1e-6 is below float16's smallest
# normal (6.1e-5), so in fp16 it underflows to zero and the normalize divides
# by an unprotected sqrt(var) — output stays finite but saturates (measured:
# 150/255 mean error, only 34% of pixels intact). bfloat16 keeps float32's
# exponent range, so eps survives: 1.4/255 mean error, 99.4% of pixels
# identical after 8-bit rounding.
_HALF_DTYPE = torch.bfloat16


if sys.platform == "win32":  # module scope so both RAM probes can use it
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


def _available_ram_bytes() -> Optional[int]:
    """RAM that could be handed out right now, or None if the platform won't say.

    Deliberately counts only what the OS considers reclaimable without paging:
    MemAvailable on Linux, ullAvailPhys on Windows, and free + inactive +
    purgeable pages on macOS (its "free" alone is near zero on a warm machine
    and would refuse everything).
    """
    try:  # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    if sys.platform == "darwin":
        try:
            import re
            import subprocess

            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            free = 0
            for label in ("Pages free", "Pages inactive", "Pages purgeable"):
                m = re.search(rf"{label}:\s+(\d+)", out)
                if m:
                    free += int(m.group(1))
            return free * page or None
        except Exception:
            return None

    try:  # Windows
        import ctypes

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys) or None
    except Exception:
        return None


def _device_budget_bytes(device: torch.device) -> Optional[int]:
    """What the accelerator says it can spare, or None if it won't say.

    MPS reports a recommended working-set size (5.33 GiB on an 8 GB M2, well
    under the allocator's nominal ceiling) — but that figure is static, and on
    unified memory the GPU is drawing from the same pool as everything else, so
    it's capped by what's actually free. CUDA already reports live free memory.
    """
    try:
        if device.type == "mps" and torch.backends.mps.is_available():
            budget = int(torch.mps.recommended_max_memory())
            available = _available_ram_bytes()
            if available is not None:
                budget = min(budget, int(available * _AVAILABLE_BUDGET))
            return budget or None
        if device.type == "cuda" and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(device)
            return int(free) or None
    except (RuntimeError, AttributeError, AssertionError):
        return None
    return None


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

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys) or None
    except Exception:
        return None


def _memory_budget(width_px: int, height_px: int) -> Optional[int]:
    """Bytes the deblur may plan to use, or None if nothing can be measured.

    The tighter of "a share of the machine" and "a share of what's free right
    now" — the first keeps a quiet machine from being over-committed, the
    second keeps a busy one from paging.
    """
    limits = []
    total = _total_ram_bytes()
    if total is not None:
        limits.append(int(total * _RAM_BUDGET))
    available = _available_ram_bytes()
    if available is not None:
        limits.append(int(available * _AVAILABLE_BUDGET))
    return min(limits) if limits else None


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

    def _run(self, x: torch.Tensor, device: torch.device,
             dtype: torch.dtype = torch.float32) -> np.ndarray:
        """Run the net on `device` in `dtype`, returning HWC float32 on the host."""
        try:
            self.net.to(device=device, dtype=dtype)
            y = self.net(x.to(device=device, dtype=dtype))
            return y.float().clamp_(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        finally:
            # Leave the model where (and how) callers expect to find it.
            self.net.to(device=self.device, dtype=torch.float32)

    def estimate_bytes(self, width_px: int, height_px: int,
                       dtype: torch.dtype = torch.float32) -> int:
        """Rough peak working set for deblurring an image of this size.

        The constant was fitted at float32; half precision halves every
        activation, so scale by the element size.
        """
        scale = torch.empty((), dtype=dtype).element_size() / 4
        return int(_PEAK_BYTES_PER_UNIT * self.spec.width * width_px
                   * height_px * scale)

    def _check_cpu_headroom(self, width_px: int, height_px: int) -> None:
        """Raise DeblurTooLargeError if a CPU run would page instead of fit."""
        budget = _memory_budget(width_px, height_px)
        if budget is None:  # unknown platform — let it try rather than block
            return
        need = self.estimate_bytes(width_px, height_px)
        if need <= budget:
            return
        gb = 1 << 30
        total = _total_ram_bytes()
        available = _available_ram_bytes()
        # Say which wall was hit: "buy more RAM" and "close some tabs" are very
        # different pieces of advice.
        if (total is not None and available is not None
                and available * _AVAILABLE_BUDGET < total * _RAM_BUDGET):
            room = (f"only {available / gb:.1f} GB is free right now (of "
                    f"{total / gb:.0f} GB). Close some apps and try again, or "
                    f"turn off Clean up")
        elif total is not None:
            room = f"this machine has {total / gb:.0f} GB of RAM. Turn off Clean up"
        else:
            room = "there isn't enough free memory. Turn off Clean up"
        max_px = budget // (_PEAK_BYTES_PER_UNIT * self.spec.width)
        smaller = "" if self.spec.width <= 32 else (
            ", switch to a width-32 clean-up model (about half the memory)")
        raise DeblurTooLargeError(
            f"Clean-up needs roughly {need / gb:.1f} GB for a "
            f"{width_px}x{height_px} image, and {room}{smaller}, or scale "
            f"the image down to about {max_px / 1e6:.1f} megapixels first."
        )

    @torch.inference_mode()
    def deblur(self, image: Image.Image) -> Image.Image:
        """Deblur a PIL image, returning a same-resolution result.

        Starts in the widest precision the device can hold, then on an OOM
        retries reduced precision and finally CPU — see `_retry_after_oom` for
        why that order, and the module docstring for why tiling isn't the
        answer instead.

        Raises DeblurTooLargeError (message fit to show the user) when nothing
        in that ladder can hold the image.
        """
        rgb = image.convert("RGB")
        # A CPU-only machine gets the same answer up front, without waiting for
        # a doomed run to fail first.
        if self.device.type == "cpu":
            self._check_cpu_headroom(rgb.width, rgb.height)
        x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)

        dtype = self._starting_dtype(rgb.width, rgb.height)
        out = None
        try:
            out = self._run(x, self.device, dtype)
        except RuntimeError as e:
            if self.device.type == "cpu" or "out of memory" not in str(e).lower():
                raise
        # Reduced precision can finish and still be garbage, so a successful
        # run is only accepted once it's checked.
        if out is not None and dtype is not torch.float32 and not np.isfinite(out).all():
            print(
                "clean-up: reduced precision produced non-finite values — "
                "falling back to CPU",
                file=sys.stderr,
            )
            out = None

        # Deliberately outside the except block: the exception's traceback pins
        # every frame of the failed forward pass, and with them the activations
        # we just ran out of room for. Retrying inside the handler retries
        # against a device that is still full — bfloat16 needs half the memory
        # and still OOMs there. Leaving the block drops `e` and frees them.
        if out is None:
            out = self._retry_after_oom(x, rgb.width, rgb.height,
                                        skip_half=dtype is not torch.float32)
        return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")

    def _starting_dtype(self, width_px: int, height_px: int) -> torch.dtype:
        """Pick the precision to open with, skipping a run doomed to OOM.

        Letting float32 fail first costs the whole forward pass — ~45s of the
        ~72s a 2560x1440 image took on an 8 GB M2, all of it thrown away. When
        the device reports a budget and float32 clearly exceeds it, start at
        reduced precision and spend that time on the run that can finish.
        """
        if self.device.type not in _HALF_PRECISION_DEVICES:
            return torch.float32
        budget = _device_budget_bytes(self.device)
        if budget is None:
            return torch.float32
        if self.estimate_bytes(width_px, height_px) <= budget:
            return torch.float32
        if self.estimate_bytes(width_px, height_px, _HALF_DTYPE) > budget:
            return torch.float32  # neither fits; let the ladder reach CPU
        gb = 1 << 30
        print(
            f"clean-up: float32 needs about "
            f"{self.estimate_bytes(width_px, height_px) / gb:.1f} GB for a "
            f"{width_px}x{height_px} image and {self.device.type} offers "
            f"{budget / gb:.1f} GB — starting in "
            f"{str(_HALF_DTYPE).replace('torch.', '')}",
            file=sys.stderr,
        )
        return _HALF_DTYPE

    def _retry_after_oom(self, x: torch.Tensor, width_px: int, height_px: int,
                         skip_half: bool = False) -> np.ndarray:
        """Recover from an accelerator OOM, cheapest option first.

        Half precision halves every activation and stays on the GPU: measured
        on an 8 GB M2, a 2560x1440 image that overflows float32 finishes in
        ~27s in bfloat16. The same image on CPU is minutes at best, and pages
        into hours if it doesn't fit in RAM. So bf16 is tried first, and CPU
        is the last resort rather than the first.

        Any failure in the bf16 attempt (out of memory, or a backend that
        can't do bfloat16 at all) just falls through to CPU — we're already
        recovering, and the original error is the one worth reporting.

        Must be called after leaving the handler for the original OOM, so the
        memory that failed has actually been released.
        """
        _empty_cache(self.device)
        if not skip_half and self.device.type in _HALF_PRECISION_DEVICES:
            try:
                out = self._run(x, self.device, _HALF_DTYPE)
            except (RuntimeError, TypeError):
                _empty_cache(self.device)
            else:
                # Cheap insurance against a run that "succeeds" into garbage.
                if np.isfinite(out).all():
                    print(
                        f"clean-up: {self.device.type} ran out of memory on a "
                        f"{width_px}x{height_px} image — retried in "
                        f"{str(_HALF_DTYPE).replace('torch.', '')}",
                        file=sys.stderr,
                    )
                    return out
                print(
                    "clean-up: reduced precision produced non-finite values — "
                    "falling back to CPU",
                    file=sys.stderr,
                )
                _empty_cache(self.device)

        self._check_cpu_headroom(width_px, height_px)  # raises DeblurTooLargeError
        print(
            f"clean-up: {self.device.type} ran out of memory on a "
            f"{width_px}x{height_px} image — retrying on CPU (slower)",
            file=sys.stderr,
        )
        return self._run(x, torch.device("cpu"))
