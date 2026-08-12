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

The same is true of the *accelerator* on unified memory, which is why every
rung is checked against a budget before it runs rather than after it fails:
the MPS allocator raises "out of memory" only past ~1.7x its recommended
working set (~9 GB on an 8 GB M2) and never consults free RAM, so a job sized
between the two — a 7 GB float32 forward against 1 GB free — "succeeds" into
swap instead of raising (observed: 7.9 GB RSS, ~40k pageins/10s, and the OOM
the recovery ladder was waiting for never came). An image that fits no
precision on the device therefore goes straight to CPU, or refuses, without
touching the accelerator at all.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from upscaler.engine import _load_state_dict, resolve_device
from upscaler.memory import (AVAILABLE_BUDGET as _AVAILABLE_BUDGET,
                             RAM_BUDGET as _RAM_BUDGET,
                             available_ram_bytes as _available_ram_bytes,
                             budget_from as _budget_from,
                             device_budget_bytes as _device_budget_bytes,
                             describe_shortfall as _describe_shortfall,
                             empty_device_cache as _empty_cache,
                             total_ram_bytes as _total_ram_bytes)
from upscaler.models.nafnet import NAFNet
from upscaler.models.registry import DeblurSpec, resolve_deblur_model
from upscaler.models.weights import ensure_weights


# Peak working set per (model width x pixel), measured on CPU by running the
# width-64 model at 256/384/512/768px and fitting the high-water RSS. Predicts
# 9.4 GB for a 2560x1440 image, against the 9.5 GiB the MPS allocator actually
# demanded at that size — close enough to size a guard with.
_PEAK_BYTES_PER_UNIT = 40

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


class DeblurTooLargeError(RuntimeError):
    """The image can't be cleaned up in the memory this machine has.

    Carries a message meant to be shown to the user as-is.
    """


def _memory_budget(width_px: int, height_px: int) -> Optional[int]:
    """Bytes the deblur may plan to use, or None if nothing can be measured.

    Reads the module-level probes rather than calling into upscaler.memory
    directly, so tests can pin this module's view of the machine.
    """
    return _budget_from(_total_ram_bytes(), _available_ram_bytes())


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
        room = _describe_shortfall(need, _total_ram_bytes(), _available_ram_bytes())
        max_px = budget // (_PEAK_BYTES_PER_UNIT * self.spec.width)
        smaller = "" if self.spec.width <= 32 else (
            ", switch to a width-32 clean-up model (about half the memory)")
        raise DeblurTooLargeError(
            f"Clean-up needs roughly {need / gb:.1f} GB for a "
            f"{width_px}x{height_px} image, and {room}. Turn off Clean "
            f"up{smaller}, or scale the image down to about "
            f"{max_px / 1e6:.1f} megapixels first."
        )

    @torch.inference_mode()
    def deblur(self, image: Image.Image) -> Image.Image:
        """Deblur a PIL image, returning a same-resolution result.

        Starts in the widest precision the device's budget can hold — or goes
        straight to CPU, never touching the accelerator, when no precision
        fits (see `_starting_dtype` for why "run it and catch the OOM" cannot
        work there). An OOM that arrives anyway still walks the recovery
        ladder in `_retry_after_oom`; the module docstring covers why tiling
        isn't the answer instead.

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
        if dtype is None:
            # Nothing fits the device. Don't touch it: an oversized MPS
            # allocation on unified memory doesn't fail fast, it "succeeds"
            # into swap (the incident: 7.9 GB RSS, ~40k pageins/10s, no OOM
            # ever raised), so the ladder below would never get to run.
            out = self._cpu_instead_of_device(x, rgb.width, rgb.height)
            return Image.fromarray(np.round(out * 255.0).astype(np.uint8),
                                   mode="RGB")
        out = None
        half_unsupported: Optional[str] = None  # the swallowed error's text
        try:
            out = self._run(x, self.device, dtype)
        except (RuntimeError, TypeError) as e:
            if self.device.type == "cpu":
                raise
            msg = str(e).lower()
            if isinstance(e, RuntimeError) and "out of memory" in msg:
                pass  # the recovery ladder below exists for exactly this
            elif (dtype is not torch.float32
                  and ("bfloat16" in msg or "scalartype" in msg)):
                # A bf16 start on a backend that can't do bf16 — not a memory
                # problem, so the OOM ladder is the wrong answer. Fall through
                # to the CPU path instead of crashing. Matched narrowly on the
                # dtype-support wording (torch phrases these as "BFloat16 is
                # not supported ..." / "Got unsupported ScalarType BFloat16")
                # so a genuine backend or model bug from the same run still
                # propagates instead of being silently converted into a
                # minutes-long CPU re-run behind a wrong "out of memory" note.
                half_unsupported = str(e)
            else:
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
            if half_unsupported:
                out = self._cpu_instead_of_device(
                    x, rgb.width, rgb.height,
                    why=(f"{self.device.type} can't run "
                         f"{str(_HALF_DTYPE).replace('torch.', '')} "
                         f"({half_unsupported})"))
            else:
                out = self._retry_after_oom(x, rgb.width, rgb.height,
                                            skip_half=dtype is not torch.float32)
        return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGB")

    def _starting_dtype(self, width_px: int,
                        height_px: int) -> Optional[torch.dtype]:
        """Pick the precision to open with, or None when nothing fits the device.

        Letting float32 fail first costs the whole forward pass — ~45s of the
        ~72s a 2560x1440 image took on an 8 GB M2, all of it thrown away. When
        the device reports a budget and float32 clearly exceeds it, start at
        reduced precision and spend that time on the run that can finish.

        Returns None when no precision fits, and the caller must not touch the
        accelerator at all. This used to return float32 "to let the ladder
        reach CPU" — but the ladder is triggered by an OOM exception that
        unified memory never raises for a job sized between free RAM and the
        allocator's ~1.7x watermark. The one that proved it: a 2560x1072
        clean-up whose 7 GB float32 forward drove an 8 GB M2 to 7.9 GB RSS and
        ~40k pageins/10s with nothing on stderr.
        """
        if self.device.type == "cpu":  # sized by _check_cpu_headroom up front
            return torch.float32
        budget = _device_budget_bytes(self.device)
        if budget is None:  # nothing measurable — keep today's behaviour
            return torch.float32
        if self.estimate_bytes(width_px, height_px) <= budget:
            return torch.float32
        if (self.device.type in _HALF_PRECISION_DEVICES
                and self.estimate_bytes(width_px, height_px, _HALF_DTYPE) <= budget):
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
        return None  # neither fits — stay off the accelerator entirely

    def _cpu_instead_of_device(
        self, x: torch.Tensor, width_px: int, height_px: int,
        why: str = "not enough free memory for the GPU",
    ) -> np.ndarray:
        """Run on the CPU because the accelerator can't take this image at all.

        Reached without an OOM: either no precision fits the device budget, or
        the backend can't run the reduced precision that would have fit —
        ``why`` says which, so the stderr note states the real reason rather
        than blaming memory for a backend gap. Same gate as every CPU run —
        refuse rather than page.
        """
        _empty_cache(self.device)
        self._check_cpu_headroom(width_px, height_px)  # raises DeblurTooLargeError
        print(f"clean-up: {why} — running on CPU (slower)", file=sys.stderr)
        return self._run(x, torch.device("cpu"))

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
