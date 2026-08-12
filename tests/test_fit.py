"""Tests for exact-resolution fitting (ultrawide / phone / tablet targets)."""

from __future__ import annotations

import pytest
from PIL import Image

from upscaler import fit


def test_photo_to_ultrawide_trims_top_and_bottom():
    # A 4:3 photo is not wide *enough* for 21:9, so the width is kept whole and
    # the height is cut — the letterbox crop, not a side crop.
    left, top, right, bottom = fit.crop_box_for_aspect(4000, 3000, 3440, 1440)
    assert (right - left) == 4000, "the wide axis is already short — keep it all"
    assert (bottom - top) < 3000
    assert abs((right - left) / (bottom - top) - 3440 / 1440) < 0.01


def test_photo_to_phone_portrait_trims_the_sides():
    left, top, right, bottom = fit.crop_box_for_aspect(4000, 3000, 1179, 2556)
    assert (bottom - top) == 3000, "a portrait target needs all the height there is"
    assert (right - left) < 4000
    assert abs((right - left) / (bottom - top) - 1179 / 2556) < 0.01


def test_matching_aspect_is_not_cropped():
    assert fit.crop_box_for_aspect(2560, 1440, 3840, 2160) == (0, 0, 2560, 1440)


def test_crop_is_centred_by_default():
    left, _, right, _ = fit.crop_box_for_aspect(4000, 3000, 3440, 1440)
    assert left == (4000 - (right - left)) // 2


@pytest.mark.parametrize("anchor,expect_left", [("left", 0), ("center", None), ("right", None)])
def test_horizontal_anchors(anchor, expect_left):
    left, _, right, _ = fit.crop_box_for_aspect(4000, 3000, 1000, 2000, anchor)
    width = right - left
    if anchor == "left":
        assert left == 0
    elif anchor == "right":
        assert right == 4000
    else:
        assert left == (4000 - width) // 2


def test_top_anchor_keeps_the_top_when_height_is_cut():
    # Faces sit high in a frame; a top anchor is the reason this option exists.
    # It only bites when the crop takes height — i.e. a wider-than-source target.
    _, top, _, bottom = fit.crop_box_for_aspect(4000, 3000, 3440, 1440, "top")
    assert top == 0
    assert bottom < 3000
    _, centred_top, _, _ = fit.crop_box_for_aspect(4000, 3000, 3440, 1440)
    assert centred_top > 0, "centre would have cut the top off"


def test_bad_anchor_is_rejected():
    with pytest.raises(ValueError, match="anchor"):
        fit.crop_box_for_aspect(100, 100, 50, 50, "diagonal")


def test_zero_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        fit.crop_box_for_aspect(0, 100, 50, 50)


def test_plan_picks_the_smallest_scale_that_covers_the_target():
    # 1720x720 crop -> x2 gives 3440x1440 exactly; x4 would be wasted work.
    p = fit.plan(1720, 720, 3440, 1440, scales=(1, 2, 4))
    assert p.scale == 2
    assert p.upscaled == (3440, 1440)
    assert p.downsamples


def test_plan_skips_the_model_when_the_source_already_covers_it():
    p = fit.plan(6000, 4000, 1920, 1080, scales=(1, 2, 4))
    assert p.scale == 1, "already big enough — enlarging then shrinking wastes time"
    assert p.downsamples


def test_plan_steps_up_when_x2_falls_short():
    p = fit.plan(800, 340, 3440, 1440, scales=(1, 2, 4))
    assert p.scale == 4


def test_plan_uses_the_largest_scale_when_nothing_covers_the_target():
    p = fit.plan(200, 85, 5120, 1440, scales=(1, 2, 4))
    assert p.scale == 4
    assert not p.downsamples, "has to enlarge on resample — flagged, not hidden"


def test_plan_crops_before_measuring_the_scale():
    # A 4:3 source for an ultrawide: the crop is what the scale must cover, not
    # the original. Measuring the uncropped height would over-pick the scale.
    p = fit.plan(4000, 3000, 3440, 1440, scales=(1, 2, 4))
    assert p.crop_size == (4000, round(4000 * 1440 / 3440))
    assert p.scale == 1


def test_end_to_end_lands_on_the_exact_target():
    src = Image.new("RGB", (4000, 3000), (10, 120, 200))
    p = fit.plan(src.width, src.height, 3440, 1440)
    cropped = fit.crop(src, 3440, 1440)
    assert cropped.size == p.crop_size
    out = fit.resize_exact(cropped, 3440, 1440)
    assert out.size == (3440, 1440)


def test_crop_returns_the_original_when_nothing_to_do():
    src = Image.new("RGB", (2560, 1440))
    assert fit.crop(src, 3840, 2160) is src


@pytest.mark.parametrize("text,expected", [
    ("3440x1440", (3440, 1440)),
    ("3440 × 1440", (3440, 1440)),
    ("1920*1080", (1920, 1080)),
    ("  2560X1080  ", (2560, 1080)),
])
def test_parse_custom_sizes(text, expected):
    assert fit.parse_target(text) == expected


@pytest.mark.parametrize("text", ["", "wide", "1920", "1920x", "0x100", "-5x9",
                                  "1920x1080x720", "99999x99999"])
def test_parse_rejects_junk(text):
    assert fit.parse_target(text) is None


def test_presets_are_sane():
    for name, (w, h) in fit.TARGET_PRESETS.items():
        assert w > 0 and h > 0, name
        assert "×" in name, f"{name} should show its size to the user"
