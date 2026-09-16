"""Depth-of-field blur: the depth-driven mask, the multi-level composite, tap
to focus, and the CLI.

Almost all of it runs on a synthetic depth map, so no model is downloaded. The
one test that runs the real network is skipped unless onnxruntime is installed
*and* the weights are already cached, so the suite never reaches the network.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from upscaler import blur, depth
from upscaler.cli import main
from upscaler.models.registry import DEFAULT_DEPTH_MODEL, DEPTH_MODELS
from upscaler.models.weights import WEIGHTS_DIR

try:
    import onnxruntime  # noqa: F401
    _ORT = True
except ImportError:
    _ORT = False
_CACHED = (WEIGHTS_DIR / DEPTH_MODELS[DEFAULT_DEPTH_MODEL].filename).is_file()


def _ramp_depth(w=200, h=100) -> Image.Image:
    """A depth map running far (left) to near (right)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (h, 1)), "L")


def _photo(w=200, h=100):
    """Fine vertical stripes, so blurring anywhere is obvious."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, ::4] = 255
    return Image.fromarray(a, "RGB")


def _sharpness(arr) -> float:
    return float(np.abs(np.diff(arr.astype(np.float32), axis=1)).mean())


# ── the registry ──────────────────────────────────────────────────────────────

def test_depth_models_are_registered_and_pinned():
    assert DEFAULT_DEPTH_MODEL in DEPTH_MODELS
    for spec in DEPTH_MODELS.values():
        assert spec.sha256 and len(spec.sha256) == 64
        assert spec.filename.endswith(".onnx") and spec.size == 518
    # the quantised one is the default: same depth, a quarter of the download
    assert "q" in DEFAULT_DEPTH_MODEL


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown depth model"):
        depth._session("depth-anything-xl")


# ── tap to focus ──────────────────────────────────────────────────────────────

def test_focus_at_reads_the_depth_under_the_point():
    dmap = _ramp_depth()
    assert depth.focus_at(dmap, 0, 50) == pytest.approx(0, abs=3)      # far edge
    assert depth.focus_at(dmap, 100, 50) == pytest.approx(100, abs=3)  # near edge
    assert depth.focus_at(dmap, 50, 50) == pytest.approx(50, abs=3)
    # out-of-range taps are clamped rather than raising: it is a click handler
    assert 0 <= depth.focus_at(dmap, -20, 200) <= 100


def test_focus_at_ignores_a_single_odd_pixel():
    """A median over a neighbourhood, so one stray value can't throw the focus."""
    arr = np.full((100, 100), 200, np.uint8)
    arr[50, 50] = 0
    assert depth.focus_at(Image.fromarray(arr, "L"), 50, 50) == pytest.approx(78, abs=3)


# ── the mask ──────────────────────────────────────────────────────────────────

def test_depth_is_a_blur_only_region():
    assert blur.DEPTH not in blur.SHAPES          # the other tabs don't offer it
    assert blur.DEPTH in blur.BLUR_SHAPES
    with pytest.raises(ValueError):
        blur.build_mask((10, 10), blur.MaskParams(shape="nonsense"))


def test_the_focused_distance_stays_sharp_and_the_rest_ramps():
    dmap = _ramp_depth()
    m = blur.MaskParams(shape=blur.DEPTH, depth=dmap, focus=50, dof=20, feather=0)
    mask = np.asarray(blur.build_mask((200, 100), m), np.float32)
    row = mask[50]
    assert row[100] == 0                          # at the focus distance: sharp
    assert row[0] > 200 and row[-1] > 200         # both extremes: fully blurred
    # Sampled inside the ramp, not past it: with dof 20 the blur is already at
    # full strength by x=60, so comparing there would compare two maxima.
    assert 0 < row[75] < row[65] <= row[60]


def test_depth_of_field_width_controls_how_much_stays_sharp():
    dmap = _ramp_depth()
    sharp_share = lambda dof: (np.asarray(blur.build_mask(   # noqa: E731
        (200, 100), blur.MaskParams(shape=blur.DEPTH, depth=dmap, focus=50,
                                    dof=dof, feather=0)), np.float32) < 10).mean()
    assert sharp_share(5) < sharp_share(30) < sharp_share(80)


def test_no_depth_map_means_no_blur():
    img = _photo()
    m = blur.MaskParams(shape=blur.DEPTH, depth=None)
    assert np.array_equal(np.asarray(blur.apply(img, blur.BlurParams(strength=60), m)),
                          np.asarray(img))


# ── the composite ─────────────────────────────────────────────────────────────

def test_blur_grows_with_distance_from_the_focus():
    img = _photo(400, 100)
    dmap = _ramp_depth(400, 100)
    # A wide depth of field so the ramp spans the frame; a narrow one would
    # saturate and make the middle and far ends equally blurred by definition.
    m = blur.MaskParams(shape=blur.DEPTH, depth=dmap, focus=100, dof=60, feather=0)
    out = np.asarray(blur.apply(img, blur.BlurParams(strength=30), m), np.float32)
    near = _sharpness(out[:, 340:390])       # at the focus distance
    mid = _sharpness(out[:, 180:230])
    far = _sharpness(out[:, 10:60])          # farthest from it
    assert near > mid > far, (near, mid, far)


def test_the_composite_does_not_ghost_a_sharp_copy_into_the_blur():
    """A plain cross-fade of sharp against blurred leaves a double exposure at
    mid weights; picking between real blur levels must not."""
    img = _photo(400, 100)
    dmap = _ramp_depth(400, 100)
    m = blur.MaskParams(shape=blur.DEPTH, depth=dmap, focus=100, dof=0, feather=0)
    graded = np.asarray(blur.apply(img, blur.BlurParams(strength=60), m), np.float32)
    # A half-blurred band should look like a half-strength blur, not like a mix
    # of the original and a full blur.
    band = slice(180, 230)
    half = np.asarray(blur.blur_image(img, blur.BlurParams(strength=30)), np.float32)
    crossfade = (np.asarray(img, np.float32) +
                 np.asarray(blur.blur_image(img, blur.BlurParams(strength=60)),
                            np.float32)) / 2
    to_half = abs(_sharpness(graded[:, band]) - _sharpness(half[:, band]))
    to_mix = abs(_sharpness(graded[:, band]) - _sharpness(crossfade[:, band]))
    assert to_half < to_mix


def test_alpha_survives_a_depth_blur():
    img = _photo(120, 80).convert("RGBA")
    img.putalpha(Image.new("L", img.size, 77))
    m = blur.MaskParams(shape=blur.DEPTH, depth=_ramp_depth(120, 80), focus=50, dof=10)
    out = blur.apply(img, blur.BlurParams(strength=50), m)
    assert out.mode == "RGBA" and out.getchannel("A").getpixel((3, 3)) == 77


def test_describe_names_the_focus_and_the_depth_of_field():
    m = blur.MaskParams(shape=blur.DEPTH, depth=_ramp_depth(), focus=68.2, dof=25)
    text = blur.describe(blur.BlurParams(kind="lens", strength=50), m, (200, 100))
    assert "depth of field" in text and "68.2" in text and "25 deep" in text


# ── CLI ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_depth(monkeypatch):
    """Stand in for the network so the CLI path runs with no model."""
    monkeypatch.setattr(depth, "estimate", lambda img, **kw: _ramp_depth(*img.size))
    return depth


def test_cli_depth_blur_with_an_explicit_focus(tmp_path, fake_depth):
    src = tmp_path / "p.png"
    _photo(400, 100).save(src)
    out = tmp_path / "dof.png"
    assert main(["blur", str(src), "-o", str(out), "--shape", "depth",
                 "--focus", "100", "--dof", "10", "--strength", "60"]) == 0
    arr = np.asarray(Image.open(out).convert("RGB"), np.float32)
    assert _sharpness(arr[:, 340:390]) > _sharpness(arr[:, 10:60])


def test_cli_focus_at_picks_the_subject(tmp_path, fake_depth, capsys):
    src = tmp_path / "p.png"
    _photo(400, 100).save(src)
    out = tmp_path / "dof.png"
    assert main(["blur", str(src), "-o", str(out), "--shape", "depth",
                 "--focus-at", "10,50", "--dof", "10", "--strength", "60"]) == 0
    err = capsys.readouterr().err
    assert "focusing at 10,50" in err
    arr = np.asarray(Image.open(out).convert("RGB"), np.float32)
    assert _sharpness(arr[:, 10:60]) > _sharpness(arr[:, 340:390])   # the far end is sharp


def test_cli_rejects_a_bad_focus_point(tmp_path, fake_depth, capsys):
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["blur", str(src), "--shape", "depth", "--focus-at", "nonsense"]) == 2
    assert "X,Y percentages" in capsys.readouterr().err


def test_depth_is_not_offered_by_the_other_commands(tmp_path, capsys):
    src = tmp_path / "p.png"
    _photo().save(src)
    with pytest.raises(SystemExit):
        main(["adjust", str(src), "--shape", "depth"])


# ── the real network ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_ORT and _CACHED),
                    reason="needs onnxruntime and the cached depth weights")
def test_the_real_model_puts_the_near_object_in_front():
    """A big bright block on a dark ground should read as nearer than the
    edges of the frame."""
    arr = np.full((240, 320, 3), 30, np.uint8)
    arr[60:200, 80:240] = 230
    dmap = depth.estimate(Image.fromarray(arr, "RGB"))
    assert dmap.size == (320, 240) and dmap.mode == "L"
    a = np.asarray(dmap, np.float32)
    # Normalised across the frame; the resize back to the photo's size
    # interpolates, so the endpoints land near rather than exactly on 0 and 255.
    assert a.min() <= 5 and a.max() >= 245
    assert a[130, 160] != a[5, 5]                     # it distinguishes them at all
