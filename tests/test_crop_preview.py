"""The crop preview under the Upscale tab's position slider.

These test app._crop_preview directly — it's the single renderer every UI event
routes through, so proving it here proves the preview wherever it's triggered.
The assertions are about geometry (which pixels stay bright, where the frame
sits), not just visibility: a preview that highlights the wrong band is worse
than none at all.
"""

from __future__ import annotations

import pytest
from PIL import Image

import app
from upscaler import fit

_ULTRAWIDE = "Ultrawide QHD · 3440×1440"
_ACCENT = (45, 212, 191)  # app._PREVIEW_ACCENT ("#2DD4BF") as RGB


def _white_source():
    # Larger than the 640px thumbnail cap, so the resize path actually runs.
    return Image.new("RGB", (800, 600), (255, 255, 255))


def test_hidden_without_an_image():
    upd = app._crop_preview(None, _ULTRAWIDE, "", 50)
    assert upd["visible"] is False
    assert "value" not in upd, "nothing should be rendered when there's no input"


@pytest.mark.parametrize("out_size,custom", [
    ("Model default (×2/×4)", ""),        # no resize at all
    ("Full HD 1080p · 1920px", ""),       # longest-edge preset — never crops
    (app._EXACT_CUSTOM, "very wide"),     # junk custom size — no target to show
    (app._EXACT_CUSTOM, ""),              # custom chosen but not yet typed
])
def test_hidden_without_an_exact_target(out_size, custom):
    upd = app._crop_preview(_white_source(), out_size, custom, 50)
    assert upd["visible"] is False


@pytest.mark.parametrize("position", [0, 50, 100])
def test_bright_band_and_frame_sit_exactly_on_the_crop_box(position):
    upd = app._crop_preview(_white_source(), _ULTRAWIDE, "", position)
    assert upd["visible"] is True
    preview = upd["value"]
    assert preview.size == (640, 480), "thumbnail must cap at 640px"

    # The preview must use the same geometry the job will, just at thumb scale.
    box = fit.crop_box_for_aspect(640, 480, 3440, 1440, position=position / 100)
    x = 320
    mid_y = (box[1] + box[3]) // 2

    # Kept region: untouched brightness (away from the hairline frame).
    assert preview.getpixel((x, mid_y)) == (255, 255, 255)
    # Doomed region: dimmed hard enough to read as "gone", on both sides that
    # exist for this position.
    if box[1] > 0:
        assert preview.getpixel((x, box[1] - 5))[0] < 120
    if box[3] < 480:
        assert preview.getpixel((x, box[3] + 4))[0] < 120
    # The 1px accent frame is drawn inward from the box edge — the box's first
    # and last rows ARE the frame, which pins its exact location.
    assert preview.getpixel((x, box[1])) == _ACCENT
    assert preview.getpixel((x, box[3] - 1)) == _ACCENT
    # ...and one row further in is already kept content, proving the frame is a
    # hairline rather than the chunky band users found confusing.
    assert preview.getpixel((x, box[1] + 1)) == (255, 255, 255)


def test_custom_size_renders_once_it_parses():
    upd = app._crop_preview(_white_source(), app._EXACT_CUSTOM, "2560x1080", 50)
    assert upd["visible"] is True
    assert upd["value"].size == (640, 480)


def test_source_image_is_never_mutated():
    # thumbnail() works in place — the renderer must operate on a copy, or the
    # image the user is about to enhance shrinks to 640px behind their back.
    src = _white_source()
    before = src.tobytes()
    app._crop_preview(src, _ULTRAWIDE, "", 30)
    assert src.size == (800, 600)
    assert src.tobytes() == before
