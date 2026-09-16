"""Fitting a file-size budget: the search, the levers, and the CLI."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from upscaler import optimize as op
from upscaler.cli import main


def _photo(w=1200, h=1200, seed=0):
    """Noise, not a flat colour: a flat image compresses to almost nothing and
    would let every budget pass without exercising the search."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None].repeat(h, 0).repeat(3, 2)
    return Image.fromarray(np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255)
                           .astype(np.uint8), "RGB")


# ── reading a budget ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("500KB", 512000), ("500 kb", 512000), ("500k", 512000),
    ("2MB", 2 * 1024 ** 2), ("1.5mb", int(1.5 * 1024 ** 2)),
    ("750000", 750000), (4096, 4096),
])
def test_parse_size_forms(text, expected):
    assert op.parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "garbage", "0", "-5", "KB", None, 0])
def test_parse_size_rejects_nonsense(bad):
    assert op.parse_size(bad) is None


def test_human_size_keeps_a_decimal_where_it_matters():
    assert op.human_size(8300).endswith("KB") and "8.1" in op.human_size(8300)
    assert op.human_size(512000) == "500 KB"
    assert op.human_size(3 * 1024 ** 2) == "3.00 MB"
    assert op.human_size(400) == "400 B"


# ── the search ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("target", ["400KB", "150KB", "60KB"])
def test_a_reachable_budget_is_met(target):
    res = op.optimize(_photo(), op.OptimizeParams(target=target))
    assert res.fits and res.nbytes <= res.target_bytes
    assert res.nbytes > res.target_bytes * 0.3      # and it doesn't waste the budget


def test_quality_is_the_highest_that_fits():
    """The point of searching rather than stepping: one notch higher must not
    fit, or a better version was left on the table."""
    img = _photo()
    p = op.OptimizeParams(target="120KB", fmt="JPEG")
    res = op.optimize(img, p)
    assert res.fits and res.quality is not None
    if res.quality < p.max_quality and res.scale >= 0.999:
        bigger = op.convert(img, "JPEG", quality=res.quality + 1)
        assert len(bigger) > res.target_bytes


def test_a_generous_budget_keeps_full_size_and_top_quality():
    img = _photo()
    res = op.optimize(img, op.OptimizeParams(target="10MB"))
    assert res.fits and res.size == img.size and res.scale == 1.0
    assert res.quality == 95                        # the ceiling, not beyond it


def test_resizing_only_happens_when_quality_alone_cannot_get_there():
    img = _photo()
    # Derive the roomy budget from the picture rather than guessing one: a
    # target the image already meets at quality 60 must be reachable without
    # touching its dimensions.
    roomy_target = len(op.convert(img, "WebP", quality=60))
    roomy = op.optimize(img, op.OptimizeParams(target=roomy_target))
    assert roomy.fits and roomy.size == img.size and roomy.scale == 1.0
    tight = op.optimize(img, op.OptimizeParams(target="15KB"))
    assert tight.fits and tight.size[0] < img.width and tight.scale < 1.0


def test_no_resize_keeps_the_size_and_admits_it_missed():
    img = _photo()
    res = op.optimize(img, op.OptimizeParams(target="5KB", allow_resize=False))
    assert res.size == img.size and not res.fits
    assert "still over" in op.describe(res)


def test_an_impossible_budget_stops_rather_than_shrinking_forever():
    res = op.optimize(_photo(), op.OptimizeParams(target="200"))
    assert not res.fits and res.scale >= op.MIN_SCALE
    assert res.attempts < 60                        # it converges, it doesn't spin


def test_max_edge_caps_the_picture_first():
    res = op.optimize(_photo(2000, 1000), op.OptimizeParams(target="5MB", max_edge=500))
    assert max(res.size) == 500 and res.size == (500, 250)


def test_quality_floor_is_respected():
    img = _photo()
    res = op.optimize(img, op.OptimizeParams(target="8KB", min_quality=70))
    assert res.quality is None or res.quality >= 70


# ── formats ───────────────────────────────────────────────────────────────────

def test_auto_prefers_webp_and_keeps_alpha():
    img = _photo(400, 400)
    assert op.pick_format(img) == "WebP"
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 128))
    res = op.optimize(rgba, op.OptimizeParams(target="300KB"))
    out = Image.open(io.BytesIO(res.data))
    assert res.fmt == "WebP" and "A" in out.getbands()


def test_webp_beats_jpeg_at_the_same_budget():
    """Why auto reaches for WebP: at one budget it holds a much higher
    quality setting than JPEG can."""
    img = _photo()
    jpeg = op.optimize(img, op.OptimizeParams(target="100KB", fmt="JPEG"))
    webp = op.optimize(img, op.OptimizeParams(target="100KB", fmt="WebP"))
    assert jpeg.fits and webp.fits
    assert webp.quality > jpeg.quality or webp.size[0] > jpeg.size[0]


def test_png_trades_colours_then_size():
    img = _photo(600, 600)
    res = op.optimize(img, op.OptimizeParams(target="120KB", fmt="PNG"))
    assert res.fmt == "PNG" and res.quality is None
    assert res.colors in op.PNG_COLOR_STEPS or res.scale < 1.0
    assert min(op.PNG_COLOR_STEPS) >= 64            # never posterises below this


def test_explicit_format_and_extension():
    res = op.optimize(_photo(300, 300), op.OptimizeParams(target="2MB", fmt="JPEG"))
    assert res.fmt == "JPEG" and res.extension == "jpg"
    assert Image.open(io.BytesIO(res.data)).format == "JPEG"


def test_bad_target_and_format_are_rejected():
    with pytest.raises(ValueError, match="can't read the size"):
        op.optimize(_photo(200, 200), op.OptimizeParams(target="nonsense"))
    with pytest.raises(ValueError, match="unknown format"):
        op.optimize(_photo(200, 200), op.OptimizeParams(fmt="XYZ"))


def test_describe_reports_what_it_took():
    res = op.optimize(_photo(), op.OptimizeParams(target="15KB"))
    text = op.describe(res)
    assert "WebP" in text and "quality" in text and "encodes" in text
    assert "scaled to" in text


# ── CLI ───────────────────────────────────────────────────────────────────────

def _save(path, size=(1200, 1200), quality=96):
    _photo(*size).save(path, quality=quality)
    return path


def test_cli_optimize_hits_the_budget(tmp_path):
    src = _save(tmp_path / "p.jpg")
    assert main(["optimize", str(src), "-t", "150KB"]) == 0
    out = tmp_path / "p_opt.webp"
    assert out.exists() and out.stat().st_size <= 150 * 1024


def test_cli_optimize_skips_files_already_under_budget(tmp_path, capsys):
    """Re-encoding a file that already fits only throws away quality."""
    src = _save(tmp_path / "small.jpg", size=(200, 200), quality=70)
    assert main(["optimize", str(src), "-t", "5MB"]) == 0
    assert not (tmp_path / "small_opt.webp").exists()
    assert "under budget (skipped)" in capsys.readouterr().err
    assert main(["optimize", str(src), "-t", "5MB", "--force"]) == 0
    assert (tmp_path / "small_opt.webp").exists()


def test_cli_optimize_reports_failure_to_reach_the_target(tmp_path, capsys):
    src = _save(tmp_path / "p.jpg")
    assert main(["optimize", str(src), "-t", "5KB", "--no-resize"]) == 1
    assert "still over" in capsys.readouterr().err


def test_cli_optimize_folder_and_format(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _save(src_dir / f"{n}.jpg")
    out = tmp_path / "out"
    assert main(["optimize", str(src_dir), "-o", str(out), "-t", "120KB", "-f", "JPEG"]) == 0
    files = sorted(p.name for p in out.glob("*.jpg"))
    assert files == ["a_opt.jpg", "b_opt.jpg"]
    assert all((out / f).stat().st_size <= 120 * 1024 for f in files)


def test_cli_optimize_errors(tmp_path, capsys):
    assert main(["optimize", str(tmp_path / "nope.jpg")]) == 2
    src = _save(tmp_path / "p.jpg")
    assert main(["optimize", str(src), "-t", "bogus"]) == 2
    assert "can't read the size" in capsys.readouterr().err
