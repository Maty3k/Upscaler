"""Blur toolbox: every kind and mask shape, strength scaling, alpha, the graded
ramp, previews, and the `upscaler blur` CLI. Pure PIL/numpy — no ffmpeg, no
models."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import blur
from upscaler.cli import main


def _photo(w=320, h=200):
    """Flat grey with a sharp vertical black pole and a bright dot."""
    img = Image.new("RGB", (w, h), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.line([(w // 2, 0), (w // 2, h)], fill=(0, 0, 0), width=4)
    d.ellipse([40, 40, 48, 48], fill=(255, 255, 255))
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


# ── strength + kinds ──────────────────────────────────────────────────────────

def test_radius_scales_with_the_short_side():
    assert blur.radius_px(100, (1000, 4000)) == pytest.approx(100)
    assert blur.radius_px(50, (400, 300)) == pytest.approx(15)
    assert blur.radius_px(0, (400, 300)) == 0 and blur.radius_px(500, (400, 300)) == 30


@pytest.mark.parametrize("kind", blur.KINDS)
def test_every_kind_keeps_size_and_is_identity_at_zero(kind):
    img = _photo()
    out = blur.blur_image(img, blur.BlurParams(kind=kind, strength=25, highlights=50))
    assert out.size == img.size and out.mode == "RGB"
    same = blur.blur_image(img, blur.BlurParams(kind=kind, strength=0))
    assert same.tobytes() == img.tobytes()


def test_unknown_kind_or_shape_rejected():
    with pytest.raises(ValueError):
        blur.blur_image(_photo(), blur.BlurParams(kind="swirl"))
    with pytest.raises(ValueError):
        blur.build_mask((10, 10), blur.MaskParams(shape="star"))


def test_gaussian_softens_the_pole():
    img = _photo()
    out = _arr(blur.blur_image(img, blur.BlurParams(strength=40)))
    assert out[100, 160, 0] > 40            # pole no longer pure black
    assert out[100, 20, 0] == pytest.approx(128, abs=1)   # flat area untouched


def test_motion_streaks_follow_the_angle():
    img = _photo()
    horiz = _arr(blur.blur_image(img, blur.BlurParams(kind="motion", strength=50, angle=0)))
    vert = _arr(blur.blur_image(img, blur.BlurParams(kind="motion", strength=50, angle=90)))
    # 20px streaks (strength 50 → 10px radius) smear the vertical pole ±10px
    # sideways; vertical streaks leave its neighbours alone
    assert horiz[100, 152, 0] < 120 and horiz[100, 168, 0] < 120
    assert horiz[100, 130, 0] == pytest.approx(128, abs=2)
    assert vert[100, 152, 0] == pytest.approx(128, abs=2)
    assert vert[100, 160, 0] < 10           # still black along its own direction


def test_spin_and_zoom_keep_the_centre_and_move_the_edges():
    img = _photo()
    for kind in ("spin", "zoom"):
        out = _arr(blur.blur_image(img, blur.BlurParams(kind=kind, strength=60)))
        assert np.abs(out[100, 160] - _arr(img)[100, 160]).max() < 60   # centre stays dark-ish
        assert np.abs(out - _arr(img)).max() > 30                        # something moved


def test_pixelate_makes_flat_blocks():
    img = _photo()
    out = _arr(blur.blur_image(img, blur.BlurParams(kind="pixelate", strength=100)))  # 20px blocks
    block = out[0:20, 0:20]
    assert np.abs(block - block[0, 0]).max() == 0


def test_surface_smooths_noise_but_keeps_the_edge():
    rng = np.random.default_rng(1)
    base = np.full((120, 200, 3), 128, np.float32)
    base[:, 100:] = 30                                    # a strong vertical edge
    noisy = np.clip(base + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
    out = _arr(blur.blur_image(Image.fromarray(noisy), blur.BlurParams(kind="surface", strength=30, threshold=20)))
    assert out[10:110, 20:80].std() < np.asarray(noisy, np.float32)[10:110, 20:80].std()   # flatter
    assert out[60, 99, 0] > 100 and out[60, 101, 0] < 60                                   # edge intact


def test_lens_blooms_highlights():
    img = _photo()
    plain = _arr(blur.blur_image(img, blur.BlurParams(kind="lens", strength=50, highlights=0)))
    bloom = _arr(blur.blur_image(img, blur.BlurParams(kind="lens", strength=50, highlights=80)))
    # 10px radius: inside the dot's disc (6px from its centre) bloom is brighter
    assert bloom[44, 50].mean() > plain[44, 50].mean() + 5


# ── masks ─────────────────────────────────────────────────────────────────────

def test_mask_shapes_inside_outside_and_feather():
    size = (200, 100)
    whole = np.asarray(blur.build_mask(size, blur.MaskParams()))
    assert whole.min() == 255
    rect = np.asarray(blur.build_mask(size, blur.MaskParams(shape="rectangle", w=50, h=50, feather=0)))
    assert rect[50, 100] == 255 and rect[5, 5] == 0
    inv = np.asarray(blur.build_mask(size, blur.MaskParams(shape="rectangle", w=50, h=50, feather=0, outside=True)))
    assert inv[50, 100] == 0 and inv[5, 5] == 255
    ell = np.asarray(blur.build_mask(size, blur.MaskParams(shape="ellipse", w=50, h=50, feather=0)))
    assert ell[50, 100] == 255 and ell[26, 51] == 0     # corner of the bounding box is outside
    band = np.asarray(blur.build_mask(size, blur.MaskParams(shape="band", h=20, feather=0)))
    assert band[50, 10] == 255 and band[5, 10] == 0 and band[95, 10] == 0
    tilted = np.asarray(blur.build_mask(size, blur.MaskParams(shape="band", h=20, angle=90, feather=0)))
    assert tilted[50, 100] == 255 and tilted[50, 5] == 0   # now a vertical strip
    soft = np.asarray(blur.build_mask(size, blur.MaskParams(shape="ellipse", w=50, h=50, feather=20)))
    assert 0 < soft[50, 145] < 255                         # feathered edge has in-between values


def test_painted_mask_resizes_and_empty_means_nothing():
    small = Image.new("L", (20, 10), 0)
    ImageDraw.Draw(small).rectangle([0, 0, 9, 9], fill=255)
    mask = np.asarray(blur.build_mask((200, 100), blur.MaskParams(shape="painted", painted=small, feather=0)))
    assert mask[50, 20] == 255 and mask[50, 180] == 0
    none = np.asarray(blur.build_mask((20, 10), blur.MaskParams(shape="painted")))
    assert none.max() == 0


# ── apply ─────────────────────────────────────────────────────────────────────

def test_apply_leaves_the_unmasked_side_untouched():
    img = _photo()
    m = blur.MaskParams(shape="rectangle", x=50, y=50, w=30, h=100, feather=0)
    out = _arr(blur.apply(img, blur.BlurParams(strength=50), m))
    assert np.array_equal(out[:, :40], _arr(img)[:, :40])   # far left: original
    assert out[100, 160, 0] > 40                            # inside: pole blurred


def test_apply_keeps_alpha_and_graded_ramp_lands_between():
    img = _photo().convert("RGBA")
    img.putalpha(Image.new("L", img.size, 77))
    m = blur.MaskParams(shape="band", h=30, feather=25, outside=True)
    graded = blur.apply(img, blur.BlurParams(strength=60), m)
    flat = blur.apply(img, blur.BlurParams(strength=60), blur.MaskParams(shape="band", h=30, feather=25,
                                                                        outside=True, progressive=False))
    assert graded.mode == "RGBA" and graded.getchannel("A").getpixel((3, 3)) == 77
    g, f, o = _arr(graded), _arr(flat), _arr(img)
    # Band 60px thick around y=100, feather 50px: the weight is exactly 0.5 at
    # y=45. There the graded ramp IS the half-strength blur, while the
    # cross-fade is a 50/50 mix that still carries a sharp copy (ghosting).
    half = _arr(blur.blur_image(img.convert("RGB"), blur.BlurParams(strength=30)))
    assert np.abs(g[45, 140:180] - half[45, 140:180]).max() <= 3
    assert np.abs(f[45, 140:180] - half[45, 140:180]).max() > 10
    assert np.array_equal(g[100, :40], o[100, :40])       # centre of the band: sharp


def test_preview_pair_downscales_only_when_needed():
    big = Image.new("RGB", (2400, 1200), (10, 20, 30))
    before, after = blur.preview_pair(big, blur.BlurParams(), blur.MaskParams(), max_edge=600)
    assert before.size == (600, 300) == after.size
    small = _photo()
    b2, a2 = blur.preview_pair(small, blur.BlurParams(), blur.MaskParams(), max_edge=600)
    assert b2.size == small.size


def test_describe_mentions_kind_and_region():
    txt = blur.describe(blur.BlurParams(kind="motion", strength=50, angle=15),
                        blur.MaskParams(shape="band", outside=True, feather=20), (1000, 500))
    assert "motion" in txt and "outside the band" in txt and "15°" in txt
    assert "whole image" in blur.describe(blur.BlurParams(), blur.MaskParams(), (100, 100))


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_blur_defaults_and_shapes(tmp_path):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["blur", str(src)]) == 0
    out = tmp_path / "p_blur.png"
    assert out.exists() and Image.open(out).size == (320, 200)
    dst = tmp_path / "face.jpg"
    assert main(["blur", str(src), "-o", str(dst), "--kind", "pixelate", "--shape", "ellipse",
                 "--x", "50", "--y", "50", "--w", "30", "--h", "60", "--strength", "60"]) == 0
    assert Image.open(dst).format == "JPEG"
    tilt = tmp_path / "tilt.png"
    assert main(["blur", str(src), "-o", str(tilt), "--shape", "band", "--h", "30", "--outside",
                 "--feather", "20", "--kind", "lens", "--highlights", "40"]) == 0
    assert np.array_equal(_arr(Image.open(tilt))[100, :40], _arr(_photo())[100, :40])


def test_cli_blur_painted_mask_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.png")
    mask = tmp_path / "m.png"
    Image.new("L", (32, 20), 255).save(mask)
    out = tmp_path / "out"
    assert main(["blur", str(src_dir), "-o", str(out), "--shape", "painted", "--mask", str(mask),
                 "--kind", "box", "--strength", "40"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_blur.png", "b_blur.png"]


def test_cli_blur_errors(tmp_path, capsys):
    assert main(["blur", str(tmp_path / "nope.png")]) == 2
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["blur", str(src), "--center", "bad"]) == 2
    assert "X,Y" in capsys.readouterr().err
    assert main(["blur", str(src), "--shape", "painted", "--mask", str(tmp_path / "missing.png")]) == 2
