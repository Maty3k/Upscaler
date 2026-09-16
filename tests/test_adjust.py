"""Color & light: every slider's direction and magnitude, the fixed pipeline
order, masks, auto, presets and the `upscaler adjust` CLI. Pure PIL/numpy."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import adjust, blur
from upscaler.cli import main

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _photo(w=200, h=120):
    """A blue sky over green ground, with a near-neutral grey block and a
    bright white dot — enough structure to test tone and color separately."""
    img = Image.new("RGB", (w, h), (60, 110, 190))
    d = ImageDraw.Draw(img)
    d.rectangle([0, h * 2 // 3, w, h], fill=(50, 90, 40))
    d.rectangle([20, 20, 70, 60], fill=(130, 128, 126))
    d.ellipse([150, 15, 170, 35], fill=(252, 252, 250))
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _lum(img):
    return (_arr(img) * LUMA).sum(axis=2)


def _grey(x=119):
    return Image.new("RGB", (8, 8), (x, x, x))


# ── light ─────────────────────────────────────────────────────────────────────

def test_neutral_params_change_nothing():
    img = _photo()
    assert adjust.AdjustParams().is_identity()
    assert adjust.apply(img, adjust.AdjustParams()).tobytes() == img.tobytes()
    assert not adjust.AdjustParams(contrast=1).is_identity()


@pytest.mark.parametrize("amount,stops", [(-100, -2.0), (-50, -1.0), (50, 1.0), (100, 2.0)])
def test_exposure_is_real_stops_in_linear_light(amount, stops):
    out = adjust.adjust_image(_grey(), adjust.AdjustParams(exposure=amount))
    lin_in = (119 / 255) ** 2.2
    lin_out = (out.getpixel((0, 0))[0] / 255) ** 2.2
    assert np.log2(lin_out / lin_in) == pytest.approx(stops, abs=0.2)


def test_contrast_pivots_around_mid_grey():
    mid, dark, bright = _grey(128), _grey(60), _grey(200)
    p = adjust.AdjustParams(contrast=50)
    assert adjust.adjust_image(mid, p).getpixel((0, 0))[0] == pytest.approx(128, abs=1)
    assert adjust.adjust_image(dark, p).getpixel((0, 0))[0] < 60
    assert adjust.adjust_image(bright, p).getpixel((0, 0))[0] > 200
    flat = adjust.adjust_image(dark, adjust.AdjustParams(contrast=-100))
    assert flat.getpixel((0, 0))[0] == pytest.approx(128, abs=1)


def test_highlights_and_shadows_only_touch_their_own_end():
    dark, bright = _grey(30), _grey(225)
    lift = adjust.AdjustParams(shadows=60)
    assert adjust.adjust_image(dark, lift).getpixel((0, 0))[0] > 50
    assert adjust.adjust_image(bright, lift).getpixel((0, 0))[0] == pytest.approx(225, abs=2)
    recover = adjust.AdjustParams(highlights=-60)
    assert adjust.adjust_image(bright, recover).getpixel((0, 0))[0] < 200
    assert adjust.adjust_image(dark, recover).getpixel((0, 0))[0] == pytest.approx(30, abs=2)
    # neither can push a pixel out of range
    assert adjust.adjust_image(_grey(255), adjust.AdjustParams(shadows=100)).getpixel((0, 0))[0] == 255
    assert adjust.adjust_image(_grey(0), adjust.AdjustParams(highlights=-100)).getpixel((0, 0))[0] == 0


def test_levels_stretch_and_gamma_lifts_midtones():
    stretched = adjust.adjust_image(_grey(128), adjust.AdjustParams(black_point=25, white_point=75))
    assert stretched.getpixel((0, 0))[0] == pytest.approx(128, abs=2)   # midpoint stays
    assert adjust.adjust_image(_grey(70), adjust.AdjustParams(black_point=25)).getpixel((0, 0))[0] < 70
    assert adjust.adjust_image(_grey(210), adjust.AdjustParams(white_point=80)).getpixel((0, 0))[0] == 255
    up = adjust.adjust_image(_grey(100), adjust.AdjustParams(gamma=1.8)).getpixel((0, 0))[0]
    down = adjust.adjust_image(_grey(100), adjust.AdjustParams(gamma=0.6)).getpixel((0, 0))[0]
    assert down < 100 < up


def test_clarity_adds_local_contrast_without_a_global_shift():
    img = _photo()
    out = adjust.adjust_image(img, adjust.AdjustParams(clarity=60))
    assert _lum(out).mean() == pytest.approx(_lum(img).mean(), abs=6)   # brightness roughly held
    assert _lum(out).std() > _lum(img).std()                            # more local variation


# ── color ─────────────────────────────────────────────────────────────────────

def test_temperature_and_tint_shift_the_right_channels_and_hold_brightness():
    grey = _grey(128)
    warm = adjust.adjust_image(grey, adjust.AdjustParams(temperature=60)).getpixel((0, 0))
    cool = adjust.adjust_image(grey, adjust.AdjustParams(temperature=-60)).getpixel((0, 0))
    assert warm[0] > 128 > warm[2] and cool[2] > 128 > cool[0]
    magenta = adjust.adjust_image(grey, adjust.AdjustParams(tint=60)).getpixel((0, 0))
    green = adjust.adjust_image(grey, adjust.AdjustParams(tint=-60)).getpixel((0, 0))
    assert magenta[1] < magenta[0] and green[1] > green[0]
    for shifted in (warm, cool, magenta, green):
        assert float(np.dot(shifted, LUMA)) == pytest.approx(128, abs=8)


def test_saturation_and_vibrance():
    img = _photo()
    grey = adjust.adjust_image(img, adjust.AdjustParams(saturation=-100))
    a = _arr(grey)
    assert np.abs(a[..., 0] - a[..., 2]).max() < 2          # fully desaturated
    boosted = _arr(adjust.adjust_image(img, adjust.AdjustParams(saturation=60)))
    base = _arr(img)
    spread = lambda x: (x.max(axis=2) - x.min(axis=2)).mean()   # noqa: E731
    assert spread(boosted) > spread(base)
    # Vibrance boosts muted color proportionally more than vivid color, and
    # leaves the vivid sky far more alone than plain saturation does.
    vib = _arr(adjust.adjust_image(img, adjust.AdjustParams(vibrance=100)))
    sat100 = _arr(adjust.adjust_image(img, adjust.AdjustParams(saturation=100)))
    neutral, sky = (slice(20, 60), slice(20, 70)), (slice(0, 15), slice(100, 140))
    assert spread(vib[neutral]) / spread(base[neutral]) > spread(vib[sky]) / spread(base[sky])
    assert spread(vib[sky]) < spread(sat100[sky])


def test_hue_rotation_moves_colors_and_spares_grey():
    assert adjust.adjust_image(_grey(128), adjust.AdjustParams(hue=120)).getpixel((0, 0)) == \
        pytest.approx((128, 128, 128), abs=3)
    red = Image.new("RGB", (4, 4), (220, 30, 30))
    turned = adjust.adjust_image(red, adjust.AdjustParams(hue=120)).getpixel((0, 0))
    assert turned[1] > turned[0]                             # red → toward green


def test_mono_mix_and_tone():
    img = _photo()
    plain = adjust.adjust_image(img, adjust.AdjustParams(mono=True))
    a = _arr(plain)
    assert np.abs(a[..., 0] - a[..., 2]).max() < 2
    # dropping the blue mix darkens the (blue) sky
    dark_sky = adjust.adjust_image(img, adjust.AdjustParams(mono=True, mono_red=70, mono_green=30, mono_blue=0))
    assert _arr(dark_sky)[5, 100, 0] < a[5, 100, 0]
    warm = adjust.adjust_image(img, adjust.AdjustParams(mono=True, tone_strength=100, tone_color="#d8b070"))
    px = warm.getpixel((100, 5))
    assert px[0] > px[2]                                     # sepia: warm
    assert adjust.adjust_image(img, adjust.AdjustParams(mono=True, mono_red=0, mono_green=0,
                                                        mono_blue=0)).size == img.size   # no divide by zero


# ── regions ───────────────────────────────────────────────────────────────────

def test_mask_limits_the_adjustment_and_alpha_survives():
    img = _photo().convert("RGBA")
    img.putalpha(Image.new("L", img.size, 88))
    m = blur.MaskParams(shape="band", y=20, h=20, feather=0)      # a strip near the top
    out = adjust.apply(img, adjust.AdjustParams(exposure=-80), m)
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((3, 3)) == 88
    d = _arr(out) - _arr(img)
    assert d[24].mean() < -20          # inside the band: darkened
    assert np.array_equal(d[110], np.zeros_like(d[110]))   # outside: untouched


def test_graduated_filter_via_band_outside():
    img = _photo()
    m = blur.MaskParams(shape="band", y=100, h=10, feather=40, outside=True)
    d = _arr(adjust.apply(img, adjust.AdjustParams(exposure=-60), m)) - _arr(img)
    assert d[5].mean() < -10 and abs(d[-5].mean()) < 1      # top darkened, bottom kept


# ── auto + presets ────────────────────────────────────────────────────────────

def test_auto_expands_range_and_neutralises_a_cast():
    img = _photo()
    dull = Image.fromarray((_arr(img) * 0.45 + 40).astype(np.uint8))
    p = adjust.auto_params(dull)
    fixed = adjust.adjust_image(dull, p)
    assert np.ptp(_arr(fixed)) > np.ptp(_arr(dull)) + 50
    assert 0 <= p.black_point <= 25 and 60 <= p.white_point <= 100 and 0.7 <= p.gamma <= 1.5
    # a warm grey card is pulled back toward neutral
    card = Image.new("RGB", (40, 30), (150, 140, 120))
    before = card.getpixel((0, 0))
    after = adjust.adjust_image(card, adjust.auto_params(card)).getpixel((0, 0))
    assert (after[0] - after[2]) < (before[0] - before[2])
    # auto keeps settings it doesn't own
    assert adjust.auto_params(img, adjust.AdjustParams(clarity=40, mono=True)).clarity == 40
    # a flat image is left alone rather than stretched into noise
    flat = adjust.auto_params(Image.new("RGB", (20, 20), (128, 128, 128)))
    assert flat.black_point == 0 and flat.white_point == 100


def test_presets_all_render_and_none_resets():
    img = _photo()
    assert adjust.PRESET_NAMES[0] == adjust.PRESET_NONE and len(adjust.PRESETS) >= 10
    for name in adjust.PRESETS:
        out = adjust.adjust_image(img, adjust.preset(name))
        assert out.size == img.size and out.mode == "RGB"
    assert adjust.preset(adjust.PRESET_NONE).is_identity()
    assert adjust.preset("nonsense").is_identity()
    assert adjust.preset("Punchy") is not adjust.PRESETS["Punchy"]   # a copy, never the shared one


def test_preview_pair_and_describe():
    big = Image.new("RGB", (2400, 1200), (80, 90, 100))
    before, after = adjust.preview_pair(big, adjust.AdjustParams(exposure=20), max_edge=600)
    assert before.size == (600, 300) == after.size
    txt = adjust.describe(adjust.AdjustParams(exposure=10, temperature=-15.476, gamma=1.1239,
                                              mono=True, tone_strength=60),
                          blur.MaskParams(shape="ellipse"))
    assert "exposure +10" in txt and "temperature -15.5" in txt and "gamma 1.12" in txt
    assert "black & white · 60% tone" in txt and "inside the ellipse" in txt
    assert adjust.describe(adjust.AdjustParams()) == "no changes"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_adjust_auto_preset_and_overrides(tmp_path):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["adjust", str(src), "--auto"]) == 0
    assert (tmp_path / "p_adjusted.png").exists()
    dst = tmp_path / "warm.jpg"
    assert main(["adjust", str(src), "-o", str(dst), "--preset", "Warm golden", "--contrast", "25"]) == 0
    assert Image.open(dst).format == "JPEG"
    # an explicit flag wins over the preset it came with
    plain = tmp_path / "a.png"
    over = tmp_path / "b.png"
    assert main(["adjust", str(src), "-o", str(plain), "--preset", "Punchy"]) == 0
    assert main(["adjust", str(src), "-o", str(over), "--preset", "Punchy", "--contrast", "0"]) == 0
    assert _arr(Image.open(plain)).std() > _arr(Image.open(over)).std()


def test_cli_adjust_mono_region_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.png")
    out = tmp_path / "out"
    assert main(["adjust", str(src_dir), "-o", str(out), "--mono", "--mono-mix", "40,50,10",
                 "--tone-strength", "60"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_adjusted.png", "b_adjusted.png"]
    grad = tmp_path / "grad.png"
    assert main(["adjust", str(src_dir / "a.png"), "-o", str(grad), "--exposure", "-60",
                 "--shape", "band", "--y", "100", "--h", "10", "--feather", "40", "--outside"]) == 0
    d = _arr(Image.open(grad)) - _arr(_photo())
    assert d[5].mean() < -10 and abs(d[-5].mean()) < 1


def test_cli_adjust_errors(tmp_path, capsys):
    assert main(["adjust", str(tmp_path / "nope.png")]) == 2
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["adjust", str(src), "--mono-mix", "bad"]) == 2
    assert "R,G,B" in capsys.readouterr().err
    assert main(["adjust", str(src), "--shape", "painted", "--mask", str(tmp_path / "no.png")]) == 2
