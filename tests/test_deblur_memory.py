"""Tests for the deblur out-of-memory fallback and its RAM guard.

NAFNet can't be tiled, so an image too big for the GPU is retried on CPU — but
only when it fits in RAM. Without the guard the retry silently pages, turning a
fast error into an hours-long swap storm.

Deblurrer is built with object.__new__ throughout: the guard is arithmetic on
the spec, so these never need the 259 MB checkpoint on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from upscaler.deblur import _HALF_DTYPE, Deblurrer, DeblurTooLargeError
from upscaler.models.registry import resolve_deblur_model

GB = 1 << 30
OOM = ("MPS backend out of memory (MPS allocated: 8.61 GiB, max allowed: "
       "9.07 GiB). Tried to allocate 900.00 MiB")


@pytest.fixture(autouse=True)
def _pin_machine(monkeypatch):
    """Detach every test from the machine it runs on.

    The device budget reads real hardware and free RAM moves under our feet, so
    both are pinned: no device budget (float32 start everywhere) and no
    availability reading (sizing falls back to total RAM). Tests about either
    override theirs explicitly."""
    monkeypatch.setattr("upscaler.deblur._device_budget_bytes", lambda device: None)
    monkeypatch.setattr("upscaler.deblur._available_ram_bytes", lambda: None)


def _available(monkeypatch, gib):
    monkeypatch.setattr("upscaler.deblur._available_ram_bytes",
                        lambda: int(gib * GB))


def _budget(monkeypatch, gib):
    monkeypatch.setattr("upscaler.deblur._device_budget_bytes",
                        lambda device: int(gib * GB))


def _deblurrer(device="mps", model="nafnet-gopro-width64"):
    db = object.__new__(Deblurrer)
    db.spec = resolve_deblur_model(model)
    db.device = torch.device(device)
    db.net = None  # only the stubbed _run is exercised
    return db


def _stub_run(db, fails_on=(("mps", torch.float32),), nonfinite=()):
    """Replace _run, recording each (device, dtype) attempt.

    `fails_on` attempts raise an OOM; `nonfinite` ones return NaN-laden output
    (a reduced-precision run that "succeeds" into garbage).
    """
    tried = []

    def run(x, device, dtype=torch.float32):
        attempt = (device.type, dtype)
        tried.append(attempt)
        if attempt in fails_on:
            raise RuntimeError(OOM)
        _, _, h, w = x.shape
        out = np.zeros((h, w, 3), dtype=np.float32)
        if attempt in nonfinite:
            out[0, 0, 0] = np.nan
        return out

    db._run = run
    db.net = _FakeNet()
    return tried


def _devices(tried):
    return [d for d, _ in tried]


class _FakeNet:
    def to(self, *a, **kw):  # _run restores the model's device and dtype
        return self


def test_estimate_scales_with_pixels_and_width():
    big = _deblurrer(model="nafnet-gopro-width64")
    small = _deblurrer(model="nafnet-gopro-width32")
    # linear in pixel count
    assert big.estimate_bytes(1000, 1000) == 4 * big.estimate_bytes(500, 500)
    # the width-32 model needs half of what width-64 needs
    assert small.estimate_bytes(800, 600) * 2 == big.estimate_bytes(800, 600)


def test_guard_rejects_image_larger_than_ram(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    db = _deblurrer()
    with pytest.raises(DeblurTooLargeError) as exc:
        db._check_cpu_headroom(2560, 1440)  # ~9.4 GB estimated
    msg = str(exc.value)
    assert "2560x1440" in msg and "8 GB" in msg
    assert "megapixels" in msg  # tells the user what size would work


def test_guard_allows_image_that_fits(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 32 * GB)
    _deblurrer()._check_cpu_headroom(2560, 1440)  # 9.4 GB of a 16 GB budget


def test_guard_skipped_when_ram_unknown(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: None)
    monkeypatch.setattr("upscaler.deblur._available_ram_bytes", lambda: None)
    _deblurrer()._check_cpu_headroom(20000, 20000)  # unknown platform: don't block


def test_oom_retries_half_precision_before_cpu(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db)
    out = db.deblur(Image.new("RGB", (64, 48)))
    assert tried == [("mps", torch.float32), ("mps", _HALF_DTYPE)]
    assert "cpu" not in _devices(tried), "bf16 succeeded — CPU is minutes slower"
    assert out.size == (64, 48) and out.mode == "RGB"


def test_falls_through_to_cpu_when_half_precision_also_ooms(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db, fails_on=(("mps", torch.float32), ("mps", _HALF_DTYPE)))
    out = db.deblur(Image.new("RGB", (64, 48)))
    assert _devices(tried) == ["mps", "mps", "cpu"]
    assert out.size == (64, 48)


def test_non_finite_half_precision_output_is_rejected(monkeypatch):
    # Reduced precision can overflow to NaN/Inf — a "successful" run that
    # returns garbage must not be handed back as a result.
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db, nonfinite=(("mps", _HALF_DTYPE),))
    out = db.deblur(Image.new("RGB", (64, 48)))
    assert _devices(tried) == ["mps", "mps", "cpu"]
    assert np.isfinite(np.asarray(out)).all()


def test_cpu_is_skipped_for_half_precision(monkeypatch):
    # CPU bf16 runs on emulated kernels — slower than the float32 run we're
    # recovering from, so the ladder must not try it.
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db, fails_on=(("mps", torch.float32), ("mps", _HALF_DTYPE)))
    db.deblur(Image.new("RGB", (64, 48)))
    assert ("cpu", _HALF_DTYPE) not in tried


def test_oom_refuses_instead_of_swapping_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    db = _deblurrer()
    tried = _stub_run(db, fails_on=(("mps", torch.float32), ("mps", _HALF_DTYPE)))
    with pytest.raises(DeblurTooLargeError):
        db.deblur(Image.new("RGB", (2560, 1440)))
    assert "cpu" not in _devices(tried), "must not start the run it can't finish"


def test_cpu_only_machine_refuses_up_front(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    db = _deblurrer(device="cpu")
    tried = _stub_run(db, fails_on=())
    with pytest.raises(DeblurTooLargeError):
        db.deblur(Image.new("RGB", (2560, 1440)))
    assert tried == [], "no point running at all when it cannot fit"


def test_half_precision_estimate_is_half_of_float32():
    db = _deblurrer()
    full = db.estimate_bytes(2560, 1440)
    assert db.estimate_bytes(2560, 1440, _HALF_DTYPE) == full // 2


def test_non_oom_error_still_propagates(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    db.net = _FakeNet()

    def boom(x, device, dtype=torch.float32):
        raise RuntimeError("some unrelated failure")

    db._run = boom
    with pytest.raises(RuntimeError, match="some unrelated failure"):
        db.deblur(Image.new("RGB", (16, 16)))


def test_too_large_is_a_runtimeerror_for_existing_handlers():
    # app.py and cli.py both catch RuntimeError around the restore stage.
    assert issubclass(DeblurTooLargeError, RuntimeError)


def test_empty_cache_is_a_noop_without_the_backend(monkeypatch):
    # torch.mps exists on Linux/Windows runners and raises when called; freeing
    # a cache we're about to abandon must never sink the retry.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.mps, "empty_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("no MPS backend")),
    )
    from upscaler.deblur import _empty_cache

    _empty_cache(torch.device("mps"))  # must not raise


def test_half_dtype_is_bfloat16_not_float16():
    """LayerNorm2d's eps is 1e-6, below float16's smallest normal (6.1e-5).

    In fp16 it underflows to zero, the normalize divides by an unprotected
    sqrt(var), and the output saturates while staying finite — measured at
    150/255 mean error against float32, versus 1.4/255 for bfloat16.
    """
    assert _HALF_DTYPE is torch.bfloat16
    eps = 1e-6
    assert torch.tensor(eps, dtype=torch.float16).item() < torch.finfo(torch.float16).tiny
    assert torch.tensor(eps, dtype=torch.bfloat16).item() > 0


def test_starts_in_float32_when_the_image_fits_the_device(monkeypatch):
    _budget(monkeypatch, 16)
    assert _deblurrer()._starting_dtype(1280, 720) is torch.float32


def test_skips_the_doomed_float32_run_when_the_device_is_too_small(monkeypatch):
    # 8.8 GB in float32, 4.4 GB in bfloat16, against an M2's reported 5.33 GiB:
    # the float32 attempt can only fail, and failing costs a full forward pass.
    _budget(monkeypatch, 5.33)
    assert _deblurrer()._starting_dtype(2560, 1440) is _HALF_DTYPE


def test_starts_in_float32_when_even_half_precision_will_not_fit(monkeypatch):
    # Nothing fits — open in float32 so the ladder runs its normal course down
    # to CPU rather than silently degrading precision for no benefit.
    _budget(monkeypatch, 1)
    assert _deblurrer()._starting_dtype(2560, 1440) is torch.float32


def test_predictive_start_does_not_retry_the_same_precision(monkeypatch):
    _budget(monkeypatch, 5.33)
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db, fails_on=(("mps", _HALF_DTYPE),))
    db.deblur(Image.new("RGB", (2560, 1440)))
    assert tried == [("mps", _HALF_DTYPE), ("cpu", torch.float32)], (
        "opened in bf16 and it OOMed — retrying bf16 would just fail again")


def test_cpu_device_never_consults_the_device_budget(monkeypatch):
    _budget(monkeypatch, 0.001)
    assert _deblurrer(device="cpu")._starting_dtype(64, 64) is torch.float32


def test_busy_machine_is_refused_even_with_enough_total_ram(monkeypatch):
    """The bug this check exists for: 32 GB installed, almost none of it free.

    Sizing against total RAM alone said yes, and the run then paged — measured
    as 31s in a fresh process versus minutes of thrashing in a loaded one on
    the same image and settings.
    """
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 32 * GB)
    _available(monkeypatch, 2)
    with pytest.raises(DeblurTooLargeError) as exc:
        _deblurrer()._check_cpu_headroom(2560, 1440)
    msg = str(exc.value)
    assert "free right now" in msg, msg
    assert "Close some apps" in msg, "a busy machine needs different advice than a small one"


def test_small_machine_message_talks_about_installed_ram(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    _available(monkeypatch, 7)  # plenty free; the machine is just small
    with pytest.raises(DeblurTooLargeError) as exc:
        _deblurrer()._check_cpu_headroom(2560, 1440)
    assert "8 GB of RAM" in str(exc.value)
    assert "free right now" not in str(exc.value)


def test_budget_is_the_tighter_of_total_and_available(monkeypatch):
    from upscaler.deblur import _memory_budget
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 16 * GB)
    _available(monkeypatch, 1)
    assert _memory_budget(100, 100) == int(1 * GB * 0.8)  # availability binds
    _available(monkeypatch, 100)
    assert _memory_budget(100, 100) == int(16 * GB * 0.5)  # total binds


def test_free_ram_is_read_when_available(monkeypatch):
    # Not mocked: the probe must actually return something on this platform,
    # or the whole check silently degrades to total-RAM sizing.
    monkeypatch.undo()
    from upscaler.deblur import _available_ram_bytes
    got = _available_ram_bytes()
    assert got is None or got > 0
