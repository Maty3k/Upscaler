"""Exact-resolution output wired through app.enhance.

The model never runs here — a stub upscaler stands in — so these are about
ordering and plumbing: that the crop happens *before* the expensive stages, and
that the result lands on the requested pixel count.
"""

from __future__ import annotations

import pytest
from PIL import Image

import app
from upscaler import fit


class _StubUpscaler:
    """Records what it was handed, and enlarges by a fixed factor."""

    def __init__(self, scale=2):
        self.scale = scale
        self.device = None
        self.seen = None

    def upscale(self, img, progress_cb=None, should_cancel=None):
        self.seen = img.size
        return img.resize((img.width * self.scale, img.height * self.scale))


@pytest.fixture
def stub(monkeypatch):
    up = _StubUpscaler()
    monkeypatch.setattr(app, "_get_upscaler", lambda *a, **kw: up)
    monkeypatch.setattr(app.library, "save_image", lambda *a, **kw: None)
    return up


def _run(src, out_size, custom="", position=50.0):
    (before, after), info = app.enhance(
        src, "x2", "cpu", False, "nafnet-gopro-width64", 1.0, 0.0, 256, False,
        out_size, custom_size=custom, crop_position=position,
    )
    return before, after, info


def test_preset_lands_on_the_exact_size(stub):
    _, after, info = _run(Image.new("RGB", (4000, 3000)), "Ultrawide QHD · 3440×1440")
    assert after.size == (3440, 1440)
    assert "3440×1440" in info


def test_crop_happens_before_the_upscaler_runs(stub):
    # The whole point of cropping first: the model must never see the pixels
    # that are about to be thrown away.
    _run(Image.new("RGB", (4000, 3000)), "Ultrawide QHD · 3440×1440")
    assert stub.seen is not None
    assert stub.seen[1] < 3000, "height should already be cropped when it arrives"
    expected = fit.crop_box_for_aspect(4000, 3000, 3440, 1440)
    assert stub.seen == (expected[2] - expected[0], expected[3] - expected[1])


def test_custom_size_is_parsed_and_applied(stub):
    _, after, _ = _run(Image.new("RGB", (2000, 2000)), app._EXACT_CUSTOM, "2560x1080")
    assert after.size == (2560, 1080)


def test_unparseable_custom_size_falls_back_to_no_resize(stub):
    # A typo must not crash the job or silently invent a size.
    _, after, _ = _run(Image.new("RGB", (1000, 800)), app._EXACT_CUSTOM, "very wide")
    assert after.size == (2000, 1600)  # just the x2 model output


def _gradient_source():
    """Vertical gradient so the kept band is identifiable by its brightness."""
    src = Image.new("RGB", (400, 300))
    for y in range(300):
        for x in range(400):
            src.putpixel((x, y), (y, y, y))
    return src


def test_position_selects_which_band_is_kept(stub):
    src = _gradient_source()
    _, top, _ = _run(src, "Ultrawide QHD · 3440×1440", position=0)
    _, bottom, _ = _run(src, "Ultrawide QHD · 3440×1440", position=100)
    assert top.getpixel((0, 0))[0] < bottom.getpixel((0, 0))[0], (
        "position 0 should keep the darker (upper) rows")


def test_intermediate_position_keeps_a_band_between_the_rails(stub):
    # 25% must land strictly between the top edge and center — the slider is a
    # continuum, not a fancier spelling of the three old anchors.
    src = _gradient_source()
    _, at_zero, _ = _run(src, "Ultrawide QHD · 3440×1440", position=0)
    _, quarter, _ = _run(src, "Ultrawide QHD · 3440×1440", position=25)
    _, center, _ = _run(src, "Ultrawide QHD · 3440×1440", position=50)
    assert at_zero.getpixel((0, 0))[0] < quarter.getpixel((0, 0))[0] \
        < center.getpixel((0, 0))[0], (
        "25% should keep a band strictly above center's, below the top edge's")


def test_before_image_matches_the_result_shape(stub):
    # The comparison slider swipes between the two; a crop changes the shape, so
    # the untouched original would slide against the result misaligned.
    before, after, _ = _run(Image.new("RGB", (4000, 3000)),
                            "Ultrawide QHD · 3440×1440")
    assert before.size == after.size


def test_longest_edge_presets_still_keep_aspect_ratio(stub):
    _, after, _ = _run(Image.new("RGB", (2000, 1000)), "Full HD 1080p · 1920px")
    assert after.size == (1920, 960), "unchanged behaviour for the old presets"


def test_no_target_leaves_the_model_output_alone(stub):
    _, after, _ = _run(Image.new("RGB", (300, 200)), "Model default (×2/×4)")
    assert after.size == (600, 400)
