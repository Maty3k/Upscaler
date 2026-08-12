"""The upscale side of the memory guard.

Tiling bounds the model's working set per tile, but three allocations still
coexist at the peak: that working set (measured, size-independent once tiling
is on), the float32 input tensor, and the output buffers. On unified memory
none of them failing to fit produces an error — the job pages instead (the
incident that forced this model: 7.9 GB RSS and ~40k pageins/10s on an 8 GB
M2, with no exception ever raised). These pin the refusal.

Upscaler is built with object.__new__: the check is arithmetic on the scale,
tile and device type, so none of this needs weights on disk.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from upscaler import memory
from upscaler.engine import (_INPUT_BYTES_PER_PIXEL, _OUTPUT_BYTES_PER_PIXEL,
                             _fixed_working_set_bytes, OutputTooLargeError,
                             Upscaler)

GB = 1 << 30
MB = 1 << 20


@pytest.fixture(autouse=True)
def _pin_machine(monkeypatch):
    """Detach from the machine running the suite; tests set their own numbers."""
    monkeypatch.setattr(memory, "total_ram_bytes", lambda: 16 * GB)
    monkeypatch.setattr(memory, "available_ram_bytes", lambda: 16 * GB)
    monkeypatch.setattr("upscaler.engine.total_ram_bytes", lambda: 16 * GB)
    monkeypatch.setattr("upscaler.engine.available_ram_bytes", lambda: 16 * GB)
    # No accelerator budget unless a test says otherwise — the probe would
    # otherwise ask the real hardware.
    monkeypatch.setattr("upscaler.engine.device_budget_bytes",
                        lambda device: None)


def _machine(monkeypatch, total_gb, available_gb):
    monkeypatch.setattr("upscaler.engine.total_ram_bytes", lambda: int(total_gb * GB))
    monkeypatch.setattr("upscaler.engine.available_ram_bytes",
                        lambda: int(available_gb * GB))


def _upscaler(scale=4, tile=256, device="cpu", tile_pad=32):
    up = object.__new__(Upscaler)
    up._scale = scale  # `scale` is a read-only property over this
    up.tile = tile  # the fixed working set is sized per processed tile
    up.tile_pad = tile_pad  # ...which includes the context padding
    up.device = torch.device(device)
    return up


def test_output_bytes_scales_with_the_square_of_the_scale():
    assert _upscaler(4).output_bytes(1000, 1000) == 4 * _upscaler(2).output_bytes(1000, 1000)


def test_output_bytes_matches_the_documented_per_pixel_cost():
    up = _upscaler(2)
    assert up.output_bytes(100, 50) == _OUTPUT_BYTES_PER_PIXEL * 200 * 100


def test_fixed_working_set_scales_with_the_processed_area():
    # Measured at tile=256 with tile_pad=32, i.e. a processed block of 320^2
    # px. Other Tile settings scale by the *processed* area, pad included —
    # scaling by the bare tile square would overcharge tile 512 by ~1.23x
    # (enough to categorically refuse default-tile jobs on small machines)
    # and undercharge tile 64 by ~2.5x.
    big_img = (10000, 10000)
    got = _fixed_working_set_bytes("cpu", 2, 512, *big_img)
    ref = _fixed_working_set_bytes("cpu", 2, 256, *big_img)
    assert got == pytest.approx(ref * (512 + 64) ** 2 / (256 + 64) ** 2)


def test_fixed_working_set_of_a_skinny_image_is_clipped_per_dimension():
    # A 20000x200 strip at tile 512: every real tile is at most 576x200
    # processed px. Charging the full 576^2 square would refuse panoramas
    # that fit with room to spare.
    strip = _fixed_working_set_bytes("cpu", 2, 512, 20000, 200)
    square = _fixed_working_set_bytes("cpu", 2, 512, 20000, 20000)
    assert strip == pytest.approx(square * 200 / (512 + 2 * 32))


def test_fixed_working_set_of_a_small_image_is_the_image():
    # A 100x100 image at Tile 512 runs as one 100x100 tile — sizing it by the
    # tile would refuse tiny jobs on the default settings.
    assert (_fixed_working_set_bytes("cpu", 2, 512, 100, 100)
            == _fixed_working_set_bytes("cpu", 2, 0, 100, 100))


def test_untiled_fixed_working_set_grows_with_the_image():
    # tile=0 means the whole image is one tile: the "fixed" cost is the
    # unguarded whole-image forward that used to swap instead of raise.
    small = _fixed_working_set_bytes("accel", 2, 0, 512, 512)
    big = _fixed_working_set_bytes("accel", 2, 0, 1024, 1024)
    assert big == pytest.approx(4 * small)


def test_the_case_that_wedged_the_app_is_refused(monkeypatch):
    # x4 of 2560x1440 -> 10240x5760. Free RAM is the 1.18 GB actually measured
    # on the machine while this job thrashed — against it even the model's
    # ~2.3 GB tile working set (measured: 2146 MB on MPS, larger on CPU) does
    # not fit, let alone the ~0.4 GB of input/output buffers on top.
    _machine(monkeypatch, total_gb=8, available_gb=1.18)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(4).check_output_fits(2560, 1440)
    msg = str(exc.value)
    assert "10240x5760" in msg, "say how big the result would be, not just the input"
    assert "×2 model" in msg, "the first lever to try"
    assert "Tile" in msg, "with no headroom past the working set, Tile is the other lever"


def test_same_image_at_x2_is_allowed(monkeypatch):
    # x2 fixed working set 1371 MB (measured, cpu) + 44 MB input + 89 MB
    # output = ~1.5 GB against a 1.68 GB budget — fits, and did in practice.
    _machine(monkeypatch, total_gb=8, available_gb=2.1)
    _upscaler(2).check_output_fits(2560, 1440)


def test_roomy_machine_allows_the_big_one(monkeypatch):
    _machine(monkeypatch, total_gb=64, available_gb=48)
    _upscaler(4).check_output_fits(2560, 1440)


def test_busy_machine_says_so(monkeypatch):
    _machine(monkeypatch, total_gb=64, available_gb=1)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(4).check_output_fits(4000, 3000)
    assert "free right now" in str(exc.value)


def test_x2_message_does_not_suggest_x2(monkeypatch):
    # Tile 64 keeps the fixed working set small (~86 MB) so the refusal here
    # is about the image itself and the message can size a workable one.
    _machine(monkeypatch, total_gb=8, available_gb=0.5)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(2, tile=64).check_output_fits(4000, 3000)
    assert "×2 model" not in str(exc.value), "already at x2 — suggest something else"
    assert "megapixels" in str(exc.value)


def test_untiled_whole_image_run_is_refused(monkeypatch):
    # The old guard modelled only the output buffers, so tile=0 on a big
    # image sailed through and swapped inside the whole-image forward. The
    # working set of x2 at 2560x1072 untiled is ~57 GB — no machine here fits.
    _machine(monkeypatch, total_gb=64, available_gb=48)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(2, tile=0).check_output_fits(2560, 1072)
    assert "Tile" in str(exc.value), "turning tiling on is the fix"


def test_accelerator_working_set_is_checked_against_the_device(monkeypatch):
    # The incident machine: MPS offering ~1.76 GB (0.8 x 2.2 GB free) while
    # the tile=512 x2 working set is ~3.6 GB. The host numbers are roomy on
    # purpose — the device budget alone must refuse, and name the Tile lever.
    _machine(monkeypatch, total_gb=64, available_gb=48)
    monkeypatch.setattr("upscaler.engine.device_budget_bytes",
                        lambda device: int(1.76 * GB))
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(2, tile=512, device="mps").check_output_fits(2560, 1440)
    assert "Tile" in str(exc.value)


def test_accelerator_with_room_falls_through_to_the_host_check(monkeypatch):
    _machine(monkeypatch, total_gb=64, available_gb=48)
    monkeypatch.setattr("upscaler.engine.device_budget_bytes",
                        lambda device: 8 * GB)
    _upscaler(2, tile=256, device="mps").check_output_fits(2560, 1440)


def test_idle_16gb_machine_allows_x4_at_the_default_tile(monkeypatch):
    # The shipped defaults — tile 512 (UI slider, CLI, and the video path) —
    # on an idle 16 GB machine. The fixed working set (~7.6 GB cpu) exceeds
    # RAM_BUDGET's half-of-total share, but that share was calibrated for
    # transient buffers: a measured resident working set may use more than
    # half of RAM when that much is actually free, so this must pass.
    _machine(monkeypatch, total_gb=16, available_gb=12)
    _upscaler(4, tile=512).check_output_fits(1500, 1000)


def test_idle_8gb_machine_allows_x2_at_the_default_tile(monkeypatch):
    _machine(monkeypatch, total_gb=8, available_gb=6)
    _upscaler(2, tile=512).check_output_fits(1920, 1080)


def test_busy_16gb_machine_still_refuses_x4_at_the_default_tile(monkeypatch):
    # The exemption answers to free RAM, not to nothing: the same job on the
    # same machine with 2 GB free must still refuse rather than page.
    _machine(monkeypatch, total_gb=16, available_gb=2)
    with pytest.raises(OutputTooLargeError) as exc:
        _upscaler(4, tile=512).check_output_fits(1500, 1000)
    assert "free right now" in str(exc.value)


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


def test_need_counts_all_three_allocations(monkeypatch):
    # The budget sits above fixed + input but below the full need — only a
    # model that also counts the output buffers refuses here.
    up = _upscaler(4, tile=256)
    fixed = _fixed_working_set_bytes("cpu", 4, 256, 2000, 1500)
    partial = fixed + _INPUT_BYTES_PER_PIXEL * 2000 * 1500
    full = partial + up.output_bytes(2000, 1500)
    budget_gb = (partial + (full - partial) // 2) / (0.8 * GB)
    _machine(monkeypatch, total_gb=1000, available_gb=budget_gb)
    with pytest.raises(OutputTooLargeError):
        up.check_output_fits(2000, 1500)


def test_error_is_a_runtimeerror_for_existing_handlers():
    # app.py and cli.py wrap the upscale in `except RuntimeError`.
    assert issubclass(OutputTooLargeError, RuntimeError)
