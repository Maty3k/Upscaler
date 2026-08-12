"""The upscale side of the memory guard.

Tiling bounds the model's working set but not the output buffer, which grows
with the square of the scale factor and is allocated whole. A x4 upscale of a
2560x1440 image on a loaded machine doesn't fail — it pages, and the job never
finishes. These pin the refusal.

Upscaler is built with object.__new__: the check is arithmetic on the scale, so
none of this needs weights on disk.
"""

from __future__ import annotations

import pytest
from PIL import Image

from upscaler import memory
from upscaler.engine import _OUTPUT_BYTES_PER_PIXEL, OutputTooLargeError, Upscaler

GB = 1 << 30


@pytest.fixture(autouse=True)
def _pin_machine(monkeypatch):
    """Detach from the machine running the suite; tests set their own numbers."""
    monkeypatch.setattr(memory, "total_ram_bytes", lambda: 16 * GB)
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 16 * GB)


def _machine(monkeypatch, total_gb, available_gb):
    monkeypatch.setattr("upscaler.engine.total_ram_bytes", lambda: int(total_gb * GB))
    monkeypatch.setattr("upscaler.engine.available_ram_bytes",
                        lambda: int(available_gb * GB))


def _upscaler(scale=4):
    up = object.__new__(Upscaler)
    up._scale = scale  # `scale` is a read-only property over this
    return up


def test_output_bytes_scales_with_the_square_of_the_scale():
    assert _upscaler(4).output_bytes(1000, 1000) == 4 * _upscaler(2).output_bytes(1000, 1000)


def test_output_bytes_matches_the_documented_per_pixel_cost():
    up = _upscaler(2)
    assert up.output_bytes(100, 50) == _OUTPUT_BYTES_PER_PIXEL * 200 * 100


def test_the_case_that_wedged_the_app_is_refused(monkeypatch):
    # x4 of 2560x1440 -> 10240x5760, about 1.65 GB of output buffers. Free RAM
    # is the 1.18 GB actually measured on the machine while this job thrashed.
    _machine(monkeypatch, total_gb=8, available_gb=1.18)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(4).check_output_fits(2560, 1440)
    msg = str(exc.value)
    assert "10240x5760" in msg, "say how big the result would be, not just the input"
    assert "×2 model" in msg, "the first lever to try"


def test_same_image_at_x2_is_allowed(monkeypatch):
    _machine(monkeypatch, total_gb=8, available_gb=2.1)
    _upscaler(2).check_output_fits(2560, 1440)  # a quarter of the pixels — fits


def test_roomy_machine_allows_the_big_one(monkeypatch):
    _machine(monkeypatch, total_gb=64, available_gb=48)
    _upscaler(4).check_output_fits(2560, 1440)


def test_busy_machine_says_so(monkeypatch):
    _machine(monkeypatch, total_gb=64, available_gb=1)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(4).check_output_fits(4000, 3000)
    assert "free right now" in str(exc.value)


def test_x2_message_does_not_suggest_x2(monkeypatch):
    _machine(monkeypatch, total_gb=8, available_gb=0.5)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(2).check_output_fits(4000, 3000)
    assert "×2 model" not in str(exc.value), "already at x2 — suggest something else"
    assert "megapixels" in str(exc.value)


def test_unknown_platform_does_not_block(monkeypatch):
    monkeypatch.setattr("upscaler.engine.total_ram_bytes", lambda: None)
    monkeypatch.setattr("upscaler.engine.available_ram_bytes", lambda: None)
    _upscaler(4).check_output_fits(20000, 20000)


def test_guard_runs_before_any_work(monkeypatch):
    # It must refuse from upscale() itself, not from deep inside tiling after
    # the buffer has already been allocated.
    _machine(monkeypatch, total_gb=8, available_gb=1)
    up = _upscaler(4)
    up.use_fp16 = False
    up.tile = 256

    def boom(*a, **kw):
        raise AssertionError("model ran despite the guard")

    up._run_tiled = boom
    up._net = boom
    with pytest.raises(OutputTooLargeError):
        Upscaler.upscale(up, Image.new("RGB", (4000, 3000)))


def test_error_is_a_runtimeerror_for_existing_handlers():
    # app.py and cli.py wrap the upscale in `except RuntimeError`.
    assert issubclass(OutputTooLargeError, RuntimeError)
