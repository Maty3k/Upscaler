"""`upscaler steam` CLI. Stills need no ffmpeg; the clip path is gated."""

import shutil
import subprocess

import pytest
from PIL import Image

from upscaler.cli import main

ffmpeg = shutil.which("ffmpeg")


def test_steam_still_default_output_dir(tmp_path):
    src = tmp_path / "photo.png"
    Image.new("RGB", (800, 450), (10, 200, 40)).save(src)
    assert main(["steam", str(src), "--height", "150", "--pan-y", "25"]) == 0
    out = tmp_path / "photo_steam"
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == [f"photo_steam_{i}.png" for i in range(1, 6)]
    assert Image.open(out / "photo_steam_1.png").size == (122, 150)
    assert (out / "photo_steam_1.png").read_bytes()[-1] == 0x21     # hexified by default


def test_steam_no_hexify(tmp_path):
    src = tmp_path / "p.png"
    Image.new("RGB", (300, 300), (9, 9, 9)).save(src)
    out = tmp_path / "raw"
    assert main(["steam", str(src), "-o", str(out), "--no-hexify"]) == 0
    assert (out / "p_steam_1.png").read_bytes()[-1] == 0x82         # intact IEND CRC


def test_steam_hidpi_and_explicit_dir(tmp_path):
    src = tmp_path / "a.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(src)
    out = tmp_path / "tiles"
    assert main(["steam", str(src), "-o", str(out), "--hidpi", "--fit", "contain"]) == 0
    assert Image.open(out / "a_steam_3.png").size == (245, 244)


def test_steam_errors(tmp_path, capsys):
    assert main(["steam", str(tmp_path / "missing.png")]) == 2
    assert "not found" in capsys.readouterr().err
    src = tmp_path / "b.png"
    Image.new("RGB", (50, 50)).save(src)
    assert main(["steam", str(src), "-o", str(tmp_path / "file.png")]) == 2
    assert "directory" in capsys.readouterr().err


def test_steam_how_to_upload(capsys):
    assert main(["steam", "--how-to-upload"]) == 0
    assert "consumer_app_id" in capsys.readouterr().out


@pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")
def test_steam_clip_exports_apngs(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=8:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=True,
    )
    out = tmp_path / "out"
    rc = main(["steam", str(src), "-o", str(out), "--fps", "8", "--loop", "boomerang"])
    assert rc == 0
    files = sorted(out.glob("clip_steam_*.png"))
    assert len(files) == 5
    with Image.open(files[0]) as im:
        assert im.is_animated and im.n_frames > 1
    assert files[0].read_bytes()[-1] == 0x21                  # hexified by default
    # --gif switches the animated tiles to GIF (untouched here so PIL can walk it)
    assert main(["steam", str(src), "-o", str(tmp_path / "gifs"), "--fps", "8", "--gif",
                 "--no-hexify"]) == 0
    with Image.open(tmp_path / "gifs" / "clip_steam_1.gif") as im:
        assert im.format == "GIF" and im.is_animated and im.n_frames > 1
    # --still forces PNG stills even for a clip
    assert main(["steam", str(src), "-o", str(tmp_path / "stills"), "--still"]) == 0
    with Image.open(tmp_path / "stills" / "clip_steam_1.png") as im:
        assert not getattr(im, "is_animated", False)
