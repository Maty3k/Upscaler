"""Steam Workshop Showcase tile builder.

Pure-PIL paths (geometry, compositing, slicing, stills, the budget ladder's
order and walk) run everywhere; the real APNG encode is gated on ffmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from upscaler import panel, steam

ffmpeg = shutil.which("ffmpeg")


def _ramp(w=640, h=360):
    """A horizontal colour ramp: every column has a distinct red value, so
    slice continuity across the gaps is checkable pixel by pixel."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for x in range(w):
        for y in range(h):
            px[x, y] = (int(255 * x / (w - 1)), 60, 200)
    return img


# ── geometry ──────────────────────────────────────────────────────────────────

def test_layout_matches_steam_css_at_1x_and_2x():
    one = steam.layout(steam.ShowcaseParams(scale=1))
    assert (one.tile_w, one.gap, one.tile_h) == (122, 4, 122)
    assert one.width == 5 * 122 + 4 * 4 == 626
    two = steam.layout(steam.ShowcaseParams(scale=2))
    assert (two.tile_w, two.gap) == (245, 8)
    assert two.width == 5 * 245 + 4 * 8


def test_boxes_tile_the_row_with_gaps_between():
    lay = steam.layout(steam.ShowcaseParams(tile_h=200))
    boxes = lay.boxes
    assert len(boxes) == steam.N_SLOTS
    assert boxes[0][0] == 0 and boxes[-1][2] == lay.width
    for (_, _, r, _), (l2, _, _, _) in zip(boxes, boxes[1:]):
        assert l2 - r == lay.gap
    assert all(b[3] == 200 and b[1] == 0 for b in boxes)


def test_tile_height_is_clamped():
    assert steam.layout(steam.ShowcaseParams(tile_h=1)).tile_h == steam.MIN_TILE_H
    assert steam.layout(steam.ShowcaseParams(tile_h=10_000)).tile_h == steam.MAX_TILE_H


# ── compositing + slicing ─────────────────────────────────────────────────────

def test_compose_row_exact_size_for_every_fit():
    src = _ramp(400, 300)
    for fit in steam.FITS:
        for scale in (1, 2):
            p = steam.ShowcaseParams(fit=fit, scale=scale, tile_h=150)
            lay = steam.layout(p)
            row = steam.compose_row(src, p)
            assert row.size == (lay.width, lay.height), (fit, scale)


def test_slices_continue_across_the_gaps():
    # Stretch a ramp over the row: tile i must start exactly where the row
    # does at its box's left edge, and the 4px gap columns must be dropped.
    p = steam.ShowcaseParams(fit="stretch")
    row = steam.compose_row(_ramp(), p)
    tiles = steam.slice_row(row, p)
    lay = steam.layout(p)
    assert [t.size for t in tiles] == [(lay.tile_w, lay.tile_h)] * 5
    for tile, (l, _, r, _) in zip(tiles, lay.boxes):
        assert tile.getpixel((0, 10)) == row.getpixel((l, 10))
        assert tile.getpixel((lay.tile_w - 1, 10)) == row.getpixel((r - 1, 10))
    # neighbouring tiles differ by the gap's worth of ramp, never equal
    assert tiles[1].getpixel((0, 10))[0] > tiles[0].getpixel((lay.tile_w - 1, 10))[0]


def test_contain_shows_background_colour():
    p = steam.ShowcaseParams(fit="contain", bg_color="#ff0000")
    row = steam.compose_row(Image.new("RGB", (100, 400), (0, 0, 255)), p)
    assert row.getpixel((2, 60))[0] > 200        # side gap → red background
    assert row.getpixel((313, 60))[2] > 200      # centre → blue source


def test_preview_renders_without_media_and_with_frame():
    empty = steam.preview(None, steam.ShowcaseParams())
    assert empty.mode == "RGB" and empty.width == 1120
    with_frame = steam.preview(None, steam.ShowcaseParams(tile_h=300), frame=_ramp())
    assert with_frame.height > empty.height  # taller row → taller preview


# ── stills ────────────────────────────────────────────────────────────────────

def test_export_stills_writes_five_numbered_pngs(tmp_path):
    src = tmp_path / "in.png"
    _ramp().save(src)
    res = steam.export_stills(str(src), steam.ShowcaseParams(), out_dir=str(tmp_path / "out"), stem="pic")
    assert [Path(x).name for x in res.paths] == [f"pic_{i}.png" for i in range(1, 6)]
    assert all(Image.open(x).size == (122, 122) for x in res.paths)
    assert not res.animated and res.fits and len(res.sizes) == 5
    assert "still PNG" in steam.describe(res)


def test_export_stills_rejects_undecodable(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        steam.export_stills(str(bad), steam.ShowcaseParams())


# ── budget ladder ─────────────────────────────────────────────────────────────

def test_budget_ladder_order_and_bounds():
    ladder = steam.budget_ladder(30)
    assert ladder[0] == (30, None, 1.0)      # exactly what was asked first
    assert ladder[1] == (30, 256, 1.0)       # palette before touching fps
    fps = [f for f, _, _ in ladder]
    assert all(f <= 30 for f in fps) and fps == sorted(fps, reverse=True)
    fracs = [fr for _, _, fr in ladder]
    assert fracs == sorted(fracs, reverse=True) and fracs[-1] == 0.25
    assert ladder[-1][:2] == (10, 128)       # most compressed rung last
    # a low request never climbs above itself
    assert all(f == 10 for f, _, _ in steam.budget_ladder(10))
    assert steam.budget_ladder(12)[2][0] == 10


def test_budget_ladder_gif_is_palette_only():
    ladder = steam.budget_ladder(30, "gif")
    assert ladder[0] == (30, 256, 1.0)          # no truecolor rung for GIF
    assert all(c is not None for _, c, _ in ladder)
    assert len(ladder) == len(steam.budget_ladder(30)) - 1   # the duplicate collapsed
    assert ladder[-1] == steam.budget_ladder(30)[-1]


def test_extract_size_shrinks_big_sources_only():
    lay = steam.layout(steam.ShowcaseParams())
    # 4K cover into a 626×122 row: width-limited → 626 wide, aspect kept
    assert steam.extract_size(3840, 2160, lay, steam.ShowcaseParams()) == (626, 352)
    # stretch → exactly the row
    assert steam.extract_size(3840, 2160, lay, steam.ShowcaseParams(fit="stretch")) == (626, 122)
    # manual zoom scales the target with it
    assert steam.extract_size(3840, 2160, lay, steam.ShowcaseParams(fit="manual", zoom=2))[0] == 1252
    # a small source would be enlarged — keep its native pixels
    assert steam.extract_size(300, 200, lay, steam.ShowcaseParams()) is None


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace ffmpeg extraction + encoding with fakes: 24 frames come from
    PIL, and each 'encode' writes files whose size tracks frames × colours."""
    def fake_extract(src_path, fps, start, dur, into, size=None, progress=None,
                     cancel=None, total_hint=0):
        base = Image.new("RGB", (64, 36), (200, 30, 30))
        for i in range(1, 25):
            base.save(os.path.join(into, f"frame_{i:05d}.png"))
        return 24

    class Calls(list):
        fmt = None   # the format the last encode was asked for

    calls = Calls()

    def fake_encode(pattern, base_fps, out_fps, colors, lay, paths, fmt="apng"):
        seq_dir = os.path.dirname(pattern)
        n = len([f for f in os.listdir(seq_dir) if f.startswith("s_")])
        frames = n * out_fps / base_fps
        size = int(frames * (300 if colors is None else colors))
        calls.append((out_fps, colors, n))
        calls.fmt = fmt
        for path in paths:
            Path(path).write_bytes(b"x" * size)

    monkeypatch.setattr(steam, "_extract_frames", fake_extract)
    monkeypatch.setattr(panel, "_first_image", lambda p: Image.new("RGB", (1920, 1080)))
    monkeypatch.setattr(panel, "media_duration", lambda p: 2.0)
    monkeypatch.setattr(steam, "_encode_slices", fake_encode)
    return calls


def test_export_animated_honours_cancel(tmp_path, fake_pipeline):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    seen = []

    def cancel():
        seen.append(1)
        return len(seen) > 3   # let a few frames composite, then pull the plug

    before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("steam_work_")}
    with pytest.raises(steam.CancelledError):
        steam.export_animated(str(src), steam.ShowcaseParams(), fps=12, max_mb=0,
                              out_dir=str(tmp_path / "out"), cancel=cancel)
    assert fake_pipeline == []   # never reached the encoder
    after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("steam_work_")}
    assert after <= before       # the work dir was cleaned up


def test_export_animated_stops_at_first_fitting_rung(tmp_path, fake_pipeline):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    cap_mb = 5000 / (1024 * 1024)  # sizes: 7200 → 6144 → 5120 → 2560 fits
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=12, max_mb=cap_mb,
                                out_dir=str(tmp_path / "out"))
    assert res.fits and res.animated
    assert (res.fps, res.colors, res.attempts) == (10, 128, 4)
    assert res.duration == pytest.approx(2.0)   # length untouched
    assert [c[:2] for c in fake_pipeline] == [(12, None), (12, 256), (10, 256), (10, 128)]
    assert len(res.paths) == 5 and all(os.path.getsize(p) == 2560 for p in res.paths)


def test_export_animated_reports_over_budget_after_last_rung(tmp_path, fake_pipeline):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=12,
                                max_mb=100 / (1024 * 1024), out_dir=str(tmp_path / "out"))
    assert not res.fits
    assert res.attempts == len(steam.budget_ladder(12))
    assert res.duration == pytest.approx(0.5)   # 25% of the 2 s clip
    assert fake_pipeline[-1][2] == 6            # 24 frames × 0.25
    assert "over the" in steam.describe(res, 100 / (1024 * 1024))


def test_export_animated_no_budget_encodes_once(tmp_path, fake_pipeline):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=12, max_mb=0,
                                loop_mode="boomerang", out_dir=str(tmp_path / "out"))
    assert res.fits and res.attempts == 1 and res.colors is None
    assert fake_pipeline == [(12, None, 46)]    # boomerang: 1..24 then 23..2


def test_export_animated_gif_uses_gif_paths_and_palette(tmp_path, fake_pipeline):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=12, max_mb=0,
                                out_dir=str(tmp_path / "out"), fmt="gif")
    assert res.fmt == "gif" and res.colors == 256 and fake_pipeline.fmt == "gif"
    assert [Path(p).name for p in res.paths] == [f"steam_{i}.gif" for i in range(1, 6)]
    assert "GIF · 12 fps · 256 colours" in steam.describe(res)
    with pytest.raises(ValueError):
        steam.export_animated(str(src), steam.ShowcaseParams(), fmt="webp")


# ── real ffmpeg ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_extract_frames_scales_and_reports(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=8:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=True,
    )
    into = tmp_path / "raw"
    into.mkdir()
    n = steam._extract_frames(str(src), 8, 0.0, 1.0, str(into), size=(100, 56),
                              progress=lambda *a, **k: None, total_hint=8)
    assert n >= 7
    assert Image.open(into / "frame_00001.png").size == (100, 56)
    assert not (into / "_ffmpeg.err").exists()
    # a cancel flag that's already set stops it before any frame lands
    into2 = tmp_path / "raw2"
    into2.mkdir()
    with pytest.raises(steam.CancelledError):
        steam._extract_frames(str(src), 8, 0.0, 1.0, str(into2), cancel=lambda: True)


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_export_animated_writes_looping_apngs(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=8:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=True,
    )
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=8, max_mb=0.05,
                                out_dir=str(tmp_path / "out"))
    assert res.fits and len(res.paths) == 5
    for path in res.paths:
        with Image.open(path) as im:
            assert im.format == "PNG" and im.is_animated and im.n_frames > 1
            assert im.size == (122, 122)
        assert os.path.getsize(path) <= 0.05 * 1024 * 1024


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_export_animated_writes_looping_gifs(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=8:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=True,
    )
    res = steam.export_animated(str(src), steam.ShowcaseParams(), fps=8, max_mb=0.05,
                                out_dir=str(tmp_path / "out"), fmt="gif")
    assert res.fits and res.fmt == "gif"
    for path in res.paths:
        assert path.endswith(".gif")
        with Image.open(path) as im:
            assert im.format == "GIF" and im.is_animated and im.n_frames > 1
            assert im.size == (122, 122)
