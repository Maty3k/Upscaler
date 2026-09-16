"""Effects and film looks: each effect's direction and shape, the fixed stack
order, looks, regions and the `upscaler effects` CLI. Pure PIL/numpy."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import blur, effects
from upscaler.cli import main

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _photo(w=240, h=160):
    """Mid grey with a bright blob and a dark block — something for the
    highlight- and shadow-driven effects to bite on."""
    img = Image.new("RGB", (w, h), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.ellipse([w // 2 - 20, h // 2 - 20, w // 2 + 20, h // 2 + 20], fill=(252, 250, 245))
    d.rectangle([10, 10, 50, 40], fill=(20, 20, 24))
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _lum(img):
    return (_arr(img) * LUMA).sum(axis=2)


def _grey(x=128, w=64, h=64):
    return Image.new("RGB", (w, h), (x, x, x))


# ── the stack ─────────────────────────────────────────────────────────────────

def test_neutral_params_change_nothing():
    img = _photo()
    assert effects.EffectParams().is_identity()
    assert effects.apply(img, effects.EffectParams()).tobytes() == img.tobytes()
    assert not effects.EffectParams(grain=1).is_identity()
    assert not effects.EffectParams(posterize=4).is_identity()


@pytest.mark.parametrize("kw", [
    dict(grain=40), dict(grain=40, grain_size=3), dict(halation=60), dict(leak=60),
    dict(vignette=60), dict(vignette=-60), dict(aberration=60), dict(duotone=100),
    dict(posterize=4), dict(dither=100), dict(halftone=100), dict(scanlines=50),
    dict(glitch=50),
])
def test_every_effect_keeps_size_and_mode(kw):
    img = _photo()
    out = effects.effect_image(img, effects.EffectParams(**kw))
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(_arr(out), _arr(img))       # it actually did something


# ── film ──────────────────────────────────────────────────────────────────────

def test_grain_adds_noise_without_shifting_brightness():
    flat = _grey(128, 200, 200)
    out = effects.effect_image(flat, effects.EffectParams(grain=50))
    assert _lum(out).std() > 5                              # noise appeared
    assert _lum(out).mean() == pytest.approx(128, abs=3)    # but the level held
    # grain is weakest at the extremes, strongest in the midtones
    mid = _lum(effects.effect_image(_grey(128, 120, 120), effects.EffectParams(grain=60))).std()
    dark = _lum(effects.effect_image(_grey(4, 120, 120), effects.EffectParams(grain=60))).std()
    assert mid > dark * 3


def test_coarse_grain_clumps():
    fine = effects.effect_image(_grey(128, 160, 160), effects.EffectParams(grain=60, grain_size=1))
    coarse = effects.effect_image(_grey(128, 160, 160), effects.EffectParams(grain=60, grain_size=5))
    # neighbouring pixels differ less when the grain is clumped into blobs
    diff = lambda im: np.abs(np.diff(_lum(im), axis=1)).mean()   # noqa: E731
    assert diff(coarse) < diff(fine)


def test_halation_glows_around_highlights_only():
    img = _photo()
    out = _arr(effects.effect_image(img, effects.EffectParams(halation=80, halation_radius=4)))
    base = _arr(img)
    near = (slice(60, 66), slice(140, 150))                 # just outside the bright blob
    far = (slice(140, 150), slice(200, 220))                # a plain mid-grey corner
    assert out[near].mean() > base[near].mean() + 3
    assert out[far].mean() == pytest.approx(base[far].mean(), abs=1.5)


def test_halation_threshold_controls_how_much_glows():
    img = _photo()
    high = _arr(effects.effect_image(img, effects.EffectParams(halation=80, halation_threshold=90)))
    low = _arr(effects.effect_image(img, effects.EffectParams(halation=80, halation_threshold=20)))
    assert low.mean() > high.mean()


def test_light_leak_is_directional_and_only_lightens():
    img = _grey(100, 200, 200)
    d = _arr(effects.effect_image(img, effects.EffectParams(leak=80, leak_angle=0))) - _arr(img)
    assert d.min() >= -0.5                                  # screen blend never darkens
    assert d[:, -20:].mean() > d[:, :20].mean() + 5         # angle 0 → strongest on the right
    turned = _arr(effects.effect_image(img, effects.EffectParams(leak=80, leak_angle=180))) - _arr(img)
    assert turned[:, :20].mean() > turned[:, -20:].mean() + 5


# ── lens ──────────────────────────────────────────────────────────────────────

def test_vignette_darkens_the_corners_and_spares_the_centre():
    img = _grey(180, 200, 200)
    out = _arr(effects.effect_image(img, effects.EffectParams(vignette=70)))
    assert out[5, 5].mean() < 150 and out[100, 100].mean() == pytest.approx(180, abs=1)
    bright = _arr(effects.effect_image(img, effects.EffectParams(vignette=-70)))
    assert bright[5, 5].mean() > 190
    # a bigger radius protects more of the frame
    wide = _arr(effects.effect_image(img, effects.EffectParams(vignette=70, vignette_radius=95)))
    assert wide[5, 5].mean() > out[5, 5].mean()


def test_aberration_fringes_grow_toward_the_corners():
    # Two identical squares: one at the centre, one out near a corner. The
    # channels are scaled about the centre, so only the outer one fringes.
    img = Image.new("RGB", (240, 240), (10, 10, 10))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 140, 140], fill=(240, 240, 240))   # centre
    d.rectangle([12, 12, 52, 52], fill=(240, 240, 240))       # corner
    out = _arr(effects.effect_image(img, effects.EffectParams(aberration=90)))
    fringe = np.abs(out[..., 0] - out[..., 2])
    # The shift is proportional to distance from the centre, so the far
    # square's edges split several times harder than the middle one's.
    assert fringe[5:60, 5:60].mean() > 3 * fringe[95:145, 95:145].mean() > 0
    assert fringe[170:230, 170:230].max() == 0                # flat area: nothing to fringe
    assert _arr(effects.effect_image(img, effects.EffectParams(aberration=30)))[..., 0].std() > 0


# ── print ─────────────────────────────────────────────────────────────────────

def test_duotone_maps_brightness_between_two_colors():
    ramp = Image.fromarray(np.repeat(np.linspace(0, 255, 256, dtype=np.uint8)[None, :], 20, 0)
                           [..., None].repeat(3, 2), "RGB")
    out = effects.effect_image(ramp, effects.EffectParams(
        duotone=100, duotone_dark="#000080", duotone_light="#ffff00"))
    dark, light = out.getpixel((2, 10)), out.getpixel((253, 10))
    assert dark[2] > dark[0] and light[0] > light[2]         # navy shadows, yellow highlights
    half = out.getpixel((128, 10))
    assert dark[0] < half[0] < light[0]                      # monotonic in between


def test_posterize_and_dither_quantise():
    img = _photo()
    out = _arr(effects.effect_image(img, effects.EffectParams(posterize=4)))
    assert len(np.unique(out[..., 0])) <= 4
    one_bit = _arr(effects.effect_image(img, effects.EffectParams(dither=100, dither_levels=2)))
    assert set(np.unique(one_bit).tolist()) <= {0.0, 255.0}   # 2 levels per channel
    assert len(np.unique(one_bit[..., 0])) == 2


def test_halftone_ink_coverage_tracks_the_tone():
    ramp = Image.fromarray(np.repeat(np.linspace(0, 255, 400, dtype=np.uint8)[None, :], 160, 0)
                           [..., None].repeat(3, 2), "RGB")
    out = np.asarray(effects.effect_image(ramp, effects.EffectParams(
        halftone=100, halftone_cell=3)), dtype=np.float32) / 255.0
    for tone in (0.25, 0.5, 0.75):
        col = int(tone * 399)
        ink = 1.0 - out[:, max(0, col - 30):col + 30].mean()
        assert ink == pytest.approx(1.0 - tone, abs=0.06)     # 50% grey prints ~50% ink


def test_halftone_cell_size_changes_the_dot_pitch():
    img = _grey(128, 200, 200)
    fine = _arr(effects.effect_image(img, effects.EffectParams(halftone=100, halftone_cell=0.5)))
    coarse = _arr(effects.effect_image(img, effects.EffectParams(halftone=100, halftone_cell=4)))
    flips = lambda a: (np.diff(a[..., 0] > 128, axis=1) != 0).sum()   # noqa: E731
    assert flips(fine) > flips(coarse)


# ── screen ────────────────────────────────────────────────────────────────────

def test_scanlines_alternate_rows():
    img = _grey(200, 100, 60)
    out = _lum(effects.effect_image(img, effects.EffectParams(scanlines=60, scanline_spacing=4)))
    rows = out.mean(axis=1)
    assert rows.max() > rows.min() + 20
    assert rows.max() == pytest.approx(200, abs=2)           # the lit rows are untouched


def test_glitch_is_deterministic_per_seed():
    img = _photo()
    a = effects.effect_image(img, effects.EffectParams(glitch=60, glitch_seed=3))
    b = effects.effect_image(img, effects.EffectParams(glitch=60, glitch_seed=3))
    c = effects.effect_image(img, effects.EffectParams(glitch=60, glitch_seed=4))
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()


# ── looks, regions, plumbing ──────────────────────────────────────────────────

def test_looks_all_render_and_none_resets():
    img = _photo()
    assert effects.PRESET_NAMES[0] == effects.PRESET_NONE and len(effects.PRESETS) >= 10
    for name in effects.PRESETS:
        out = effects.effect_image(img, effects.preset(name))
        assert out.size == img.size
        assert not np.array_equal(_arr(out), _arr(img)), name
    assert effects.preset(effects.PRESET_NONE).is_identity()
    assert effects.preset("nonsense").is_identity()
    assert effects.preset("Lomo") is not effects.PRESETS["Lomo"]    # a copy


def test_region_limits_the_effects_and_alpha_survives():
    img = _photo().convert("RGBA")
    img.putalpha(Image.new("L", img.size, 66))
    m = blur.MaskParams(shape="rectangle", x=50, y=50, w=40, h=100, feather=0)
    out = effects.apply(img, effects.EffectParams(halftone=100, posterize=4), m)
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((2, 2)) == 66
    d = _arr(out) - _arr(img)
    assert np.array_equal(d[:, :40], np.zeros_like(d[:, :40]))     # outside: untouched
    assert np.abs(d[:, 100:140]).max() > 0                         # inside: changed


def test_preview_pair_and_describe():
    big = Image.new("RGB", (2400, 1200), (90, 90, 90))
    before, after = effects.preview_pair(big, effects.EffectParams(grain=20), max_edge=600)
    assert before.size == (600, 300) == after.size
    txt = effects.describe(effects.EffectParams(grain=20, halation=40, posterize=6),
                           blur.MaskParams(shape="faces", faces=[(0.1, 0.1, 0.2, 0.2)]))
    assert "grain 20" in txt and "halation 40" in txt and "posterize 6 levels" in txt
    assert "inside the 1 face" in txt
    assert effects.describe(effects.EffectParams()) == "no effects"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_effects_look_and_overrides(tmp_path):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["effects", str(src)]) == 0
    assert (tmp_path / "p_fx.png").exists()
    dst = tmp_path / "vintage.jpg"
    assert main(["effects", str(src), "-o", str(dst), "--look", "Faded vintage"]) == 0
    assert Image.open(dst).format == "JPEG"
    # an explicit flag wins over the look it came with
    off = tmp_path / "off.png"
    assert main(["effects", str(src), "-o", str(off), "--look", "Lomo", "--vignette", "0",
                 "--grain", "0", "--aberration", "0"]) == 0
    assert np.array_equal(_arr(Image.open(off)), _arr(_photo()))


def test_cli_effects_flags_and_folder(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.png")
    out = tmp_path / "out"
    assert main(["effects", str(src_dir), "-o", str(out), "--halftone", "100",
                 "--halftone-cell", "2", "--posterize", "4"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_fx.png", "b_fx.png"]
    leak = tmp_path / "leak.png"
    assert main(["effects", str(src_dir / "a.png"), "-o", str(leak), "--leak", "70",
                 "--leak-color", "#00ff00", "--leak-angle", "0"]) == 0
    d = _arr(Image.open(leak)) - _arr(_photo())
    assert d[..., 1].mean() > d[..., 2].mean()          # a green leak lifts green most


def test_cli_effects_errors(tmp_path, capsys):
    assert main(["effects", str(tmp_path / "nope.png")]) == 2
    assert "not found" in capsys.readouterr().err
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["effects", str(src), "--shape", "painted", "--mask", str(tmp_path / "no.png")]) == 2
