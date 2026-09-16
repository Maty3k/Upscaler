"""The sharpening toolbox: each method, the halo cap, threshold, edge-aware
gating, luminance-only sharpening, the 1:1 preview and the CLI.

The old `unsharp_mask` helper (used by the upscale pipeline, video and the
`--sharpen` flag) is covered in test_architecture.py and is untouched here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from upscaler import blur, sharpen
from upscaler.cli import main


def _edgy(w=240, h=160, cell=8):
    """A softened fine checkerboard: real texture at the scale sharpening
    works on. A single step edge is no good as a target — sharpening makes it
    steeper but the total gradient across a step is conserved, so the usual
    crispness measure wouldn't move at all."""
    a = np.indices((h, w)).sum(axis=0) // cell % 2
    img = Image.fromarray((a * 150 + 50).astype(np.uint8)[..., None].repeat(3, 2), "RGB")
    return img.filter(ImageFilter.GaussianBlur(2.0))


def _step(w=240, h=160):
    """One softened vertical edge — for measuring overshoot and spread."""
    img = Image.new("RGB", (w, h), (60, 60, 60))
    ImageDraw.Draw(img).rectangle([w // 2, 0, w, h], fill=(200, 200, 200))
    return img.filter(ImageFilter.GaussianBlur(2.0))


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _acutance(img):
    """Mean gradient magnitude — how crisp the picture reads."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(g)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _noisy(level=4.0, size=200):
    rng = np.random.default_rng(0)
    a = np.full((size, size, 3), 128, np.float32) + rng.normal(0, level, (size, size, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


# ── the methods ───────────────────────────────────────────────────────────────

def test_zero_amount_is_a_no_op():
    img = _edgy()
    assert sharpen.SharpenParams(amount=0).is_identity()
    assert sharpen.apply(img, sharpen.SharpenParams(amount=0)).tobytes() == img.tobytes()
    assert not sharpen.SharpenParams(amount=1).is_identity()


@pytest.mark.parametrize("kind", sharpen.KINDS)
def test_every_method_sharpens_and_keeps_the_image(kind):
    img = _edgy()
    out = sharpen.sharpen_image(img, sharpen.SharpenParams(kind=kind, amount=140, radius=1.5))
    assert out.size == img.size and out.mode == "RGB"
    assert _acutance(out) > _acutance(img) * 1.1


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        sharpen.sharpen_image(_edgy(), sharpen.SharpenParams(kind="deconvolve"))


def test_amount_scales_the_effect():
    img = _edgy()
    weak = _acutance(sharpen.sharpen_image(img, sharpen.SharpenParams(amount=40, halo=100)))
    strong = _acutance(sharpen.sharpen_image(img, sharpen.SharpenParams(amount=200, halo=100)))
    assert _acutance(img) < weak < strong


def test_radius_is_pixels_and_clamped():
    assert sharpen.radius_px(1.0) == 1.0
    assert sharpen.radius_px(0.0) == sharpen.MIN_RADIUS
    assert sharpen.radius_px(999) == sharpen.MAX_RADIUS
    # a wider radius spreads the effect further from the edge
    img = _step()
    base = _arr(img)
    near = lambda r: float(np.abs(   # noqa: E731
        _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(amount=150, radius=r, halo=100)))
        - base)[:, 105:112].mean())
    assert near(4.0) > near(0.5)


# ── the controls ──────────────────────────────────────────────────────────────

def test_halo_limit_caps_the_overshoot_in_output_units():
    img = _step()
    base = _arr(img)
    for halo, cap in ((20, 0.20 * 0.5 * 255), (10, 0.10 * 0.5 * 255), (5, 0.05 * 0.5 * 255)):
        out = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
            amount=250, radius=2.0, halo=halo, threshold=0)))
        assert np.abs(out - base).max() <= cap + 1.5      # the number means what it says
    loose = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=250, radius=2.0, halo=100, threshold=0)))
    tight = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=250, radius=2.0, halo=5, threshold=0)))
    # left uncapped, the edge rings well past what a low cap allows
    assert np.abs(loose - base).max() > np.abs(tight - base).max() * 3


def test_threshold_protects_flat_noise():
    noisy = _noisy()
    before = _arr(noisy).std()
    amplified = _arr(sharpen.sharpen_image(noisy, sharpen.SharpenParams(
        amount=200, threshold=0))).std()
    protected = _arr(sharpen.sharpen_image(noisy, sharpen.SharpenParams(
        amount=200, threshold=40))).std()
    assert amplified > before * 1.5
    assert protected == pytest.approx(before, abs=0.3)


def test_smart_spares_flat_areas_but_keeps_the_edge():
    rng = np.random.default_rng(1)
    base = np.full((120, 240, 3), 128, np.float32)
    base[:, 120:] = 40
    img = Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))
    plain = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=200, threshold=0, kind="unsharp")))
    smart = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=200, threshold=0, kind="smart")))
    flat = (slice(None), slice(10, 100))
    assert smart[flat].std() < plain[flat].std() * 0.8        # noise left alone
    contrast = lambda a: abs(float(a[60, 115, 0]) - float(a[60, 125, 0]))   # noqa: E731
    assert contrast(smart) > contrast(plain) * 0.9            # the edge still snaps
    # Higher sensitivity is stricter, so it spares more of the flat area —
    # which is what the "skin-safe" preset is counting on.
    lenient = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=200, threshold=0, kind="smart", edge_sensitivity=0)))
    strict = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=200, threshold=0, kind="smart", edge_sensitivity=100)))
    assert strict[flat].std() < lenient[flat].std()


def test_luminance_only_avoids_color_fringes():
    img = Image.new("RGB", (120, 60), (200, 40, 40))
    ImageDraw.Draw(img).rectangle([60, 0, 120, 60], fill=(40, 40, 200))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    base = _arr(img)
    lum = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=250, radius=1.5, luminance_only=True, threshold=0, halo=100)))
    per_channel = _arr(sharpen.sharpen_image(img, sharpen.SharpenParams(
        amount=250, radius=1.5, luminance_only=False, threshold=0, halo=100)))
    edge = (30, slice(55, 66))
    assert np.abs(lum[edge] - base[edge]).max() < np.abs(per_channel[edge] - base[edge]).max()


def test_protect_shadows_and_highlights():
    dark = Image.new("RGB", (80, 40), (12, 12, 12))
    ImageDraw.Draw(dark).rectangle([40, 0, 80, 40], fill=(50, 50, 50))
    dark = dark.filter(ImageFilter.GaussianBlur(1.5))
    plain = _arr(sharpen.sharpen_image(dark, sharpen.SharpenParams(amount=200, threshold=0)))
    held = _arr(sharpen.sharpen_image(dark, sharpen.SharpenParams(
        amount=200, threshold=0, protect_shadows=100)))
    base = _arr(dark)
    assert np.abs(held - base).max() < np.abs(plain - base).max()
    bright = Image.new("RGB", (80, 40), (250, 250, 250))
    ImageDraw.Draw(bright).rectangle([40, 0, 80, 40], fill=(200, 200, 200))
    bright = bright.filter(ImageFilter.GaussianBlur(1.5))
    p2 = _arr(sharpen.sharpen_image(bright, sharpen.SharpenParams(amount=200, threshold=0)))
    h2 = _arr(sharpen.sharpen_image(bright, sharpen.SharpenParams(
        amount=200, threshold=0, protect_highlights=100)))
    assert np.abs(h2 - _arr(bright)).max() < np.abs(p2 - _arr(bright)).max()


# ── regions, preview, presets ─────────────────────────────────────────────────

def test_region_limits_sharpening_and_alpha_survives():
    img = _edgy().convert("RGBA")  # noqa: E501
    img.putalpha(Image.new("L", img.size, 55))
    m = blur.MaskParams(shape="rectangle", x=75, y=50, w=40, h=100, feather=0)
    out = sharpen.apply(img, sharpen.SharpenParams(amount=200), m)
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((2, 2)) == 55
    d = _arr(out) - _arr(img)
    assert np.array_equal(d[:, :80], np.zeros_like(d[:, :80]))
    assert np.abs(d[:, 150:200]).max() > 0


def test_preview_is_a_real_1_to_1_crop():
    big = Image.new("RGB", (2400, 1600), (90, 90, 90))
    ImageDraw.Draw(big).rectangle([1200, 0, 2400, 1600], fill=(180, 180, 180))
    big = big.filter(ImageFilter.GaussianBlur(2))
    before, after = sharpen.preview_pair(big, sharpen.SharpenParams(amount=150))
    assert before.size == (sharpen.PREVIEW_EDGE, sharpen.PREVIEW_EDGE) == after.size
    # it is a genuine crop of the source, not a resize
    left, top = sharpen._crop_origin(big.size, before.size, (0.5, 0.5))
    assert np.array_equal(_arr(before), _arr(big.crop((left, top, left + before.width,
                                                      top + before.height))))
    # the centre moves the window
    off, _ = sharpen.preview_pair(big, sharpen.SharpenParams(), center=(0.0, 0.0))
    assert not np.array_equal(_arr(off), _arr(before))
    # a small photo is shown whole
    small = _edgy()
    assert sharpen.preview_pair(small, sharpen.SharpenParams())[0].size == small.size


def test_preview_keeps_a_region_lined_up():
    big = Image.new("RGB", (2000, 2000), (70, 70, 70))
    ImageDraw.Draw(big).rectangle([1000, 0, 2000, 2000], fill=(190, 190, 190))
    big = big.filter(ImageFilter.GaussianBlur(2))
    m = blur.MaskParams(shape="rectangle", x=50, y=50, w=20, h=100, feather=0)
    _before, after = sharpen.preview_pair(big, sharpen.SharpenParams(amount=200), m)
    full = sharpen.apply(big, sharpen.SharpenParams(amount=200), m)
    left, top = sharpen._crop_origin(big.size, after.size, (0.5, 0.5))
    expected = full.crop((left, top, left + after.width, top + after.height))
    assert np.abs(_arr(after) - _arr(expected)).max() <= 1   # same window, same result


def test_presets_and_describe():
    img = _edgy()
    assert sharpen.PRESET_NAMES[0] == sharpen.PRESET_NONE and len(sharpen.PRESETS) >= 6
    for name in sharpen.PRESETS:
        out = sharpen.sharpen_image(img, sharpen.preset(name))
        assert _acutance(out) > _acutance(img), name
    assert sharpen.preset(sharpen.PRESET_NONE).is_identity()
    assert sharpen.preset("nonsense").is_identity()
    assert sharpen.preset("Punchy") is not sharpen.PRESETS["Punchy"]
    txt = sharpen.describe(sharpen.SharpenParams(kind="smart", amount=90),
                           blur.MaskParams(shape="faces", faces=[(0, 0, 1, 1)]), (800, 600))
    assert "smart" in txt and "amount 90" in txt and "radius 1px" in txt
    assert "luminance only" in txt and "inside the 1 face" in txt
    assert sharpen.describe(sharpen.SharpenParams(amount=0)) == "no sharpening"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_sharpen_preset_and_overrides(tmp_path):
    src = tmp_path / "p.png"
    _edgy().save(src)
    assert main(["sharpen", str(src)]) == 0
    assert (tmp_path / "p_sharp.png").exists()
    assert _acutance(Image.open(tmp_path / "p_sharp.png")) > _acutance(_edgy())
    dst = tmp_path / "punch.jpg"
    assert main(["sharpen", str(src), "-o", str(dst), "--preset", "Punchy"]) == 0
    assert Image.open(dst).format == "JPEG"
    # an explicit flag wins over the preset
    off = tmp_path / "off.png"
    assert main(["sharpen", str(src), "-o", str(off), "--preset", "Punchy", "--amount", "0"]) == 0
    assert np.array_equal(_arr(Image.open(off)), _arr(_edgy()))


def test_cli_sharpen_flags_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _edgy().save(src_dir / f"{n}.png")
    out = tmp_path / "out"
    assert main(["sharpen", str(src_dir), "-o", str(out), "--kind", "texture",
                 "--amount", "120", "--detail-balance", "30"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_sharp.png", "b_sharp.png"]
    # --all-channels turns off luminance-only
    both = tmp_path / "both.png"
    assert main(["sharpen", str(src_dir / "a.png"), "-o", str(both), "--all-channels",
                 "--amount", "150"]) == 0
    lum_only = tmp_path / "lum.png"
    assert main(["sharpen", str(src_dir / "a.png"), "-o", str(lum_only), "--amount", "150"]) == 0
    assert not np.array_equal(_arr(Image.open(both)), _arr(Image.open(lum_only)))


def test_cli_sharpen_errors(tmp_path, capsys):
    assert main(["sharpen", str(tmp_path / "nope.png")]) == 2
    assert "not found" in capsys.readouterr().err
    src = tmp_path / "p.png"
    _edgy().save(src)
    assert main(["sharpen", str(src), "--shape", "painted", "--mask", str(tmp_path / "no.png")]) == 2
