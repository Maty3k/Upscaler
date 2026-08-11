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

from upscaler.deblur import Deblurrer, DeblurTooLargeError
from upscaler.models.registry import resolve_deblur_model

GB = 1 << 30
OOM = ("MPS backend out of memory (MPS allocated: 8.61 GiB, max allowed: "
       "9.07 GiB). Tried to allocate 900.00 MiB")


def _deblurrer(device="mps", model="nafnet-gopro-width64"):
    db = object.__new__(Deblurrer)
    db.spec = resolve_deblur_model(model)
    db.device = torch.device(device)
    db.net = None  # only the stubbed _run is exercised
    return db


def _stub_run(db, fails_on=("mps",)):
    """Replace _run: raise an OOM on `fails_on` devices, else return a real array."""
    tried = []

    def run(x, device):
        tried.append(device.type)
        if device.type in fails_on:
            raise RuntimeError(OOM)
        _, _, h, w = x.shape
        return np.zeros((h, w, 3), dtype=np.float32)

    db._run = run
    db.net = _FakeNet()
    return tried


class _FakeNet:
    def to(self, device):  # the fallback restores the model's device
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
    _deblurrer()._check_cpu_headroom(20000, 20000)  # unknown platform: don't block


def test_oom_falls_back_to_cpu_when_it_fits(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    tried = _stub_run(db)
    out = db.deblur(Image.new("RGB", (64, 48)))
    assert tried == ["mps", "cpu"]
    assert out.size == (64, 48) and out.mode == "RGB"


def test_oom_refuses_instead_of_swapping_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    db = _deblurrer()
    tried = _stub_run(db)
    with pytest.raises(DeblurTooLargeError):
        db.deblur(Image.new("RGB", (2560, 1440)))
    assert tried == ["mps"], "must not start the CPU run it can't finish"


def test_cpu_only_machine_refuses_up_front(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 8 * GB)
    db = _deblurrer(device="cpu")
    tried = _stub_run(db, fails_on=())
    with pytest.raises(DeblurTooLargeError):
        db.deblur(Image.new("RGB", (2560, 1440)))
    assert tried == [], "no point running at all when it cannot fit"


def test_non_oom_error_still_propagates(monkeypatch):
    monkeypatch.setattr("upscaler.deblur._total_ram_bytes", lambda: 64 * GB)
    db = _deblurrer()
    db.net = _FakeNet()

    def boom(x, device):
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
