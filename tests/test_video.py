"""Video upscaling tests. Skipped entirely if ffmpeg isn't available."""

import shutil
import subprocess

import pytest

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")


def _probe_wh(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


@pytest.fixture
def tiny_video(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=size=40x32:rate=8:duration=0.5",
         "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, check=True,
    )
    return src


def test_upscale_video_doubles_dimensions(tiny_video, tmp_path):
    from upscaler.video import upscale_video

    dst = tmp_path / "out.mp4"
    upscale_video(tiny_video, dst, model="realesrgan-x2plus", device="cpu", tile=0)
    assert dst.exists()
    assert _probe_wh(dst) == (80, 64)  # 40x32 -> x2


def test_progress_callback_fires(tiny_video, tmp_path):
    from upscaler.video import upscale_video

    seen = []
    upscale_video(
        tiny_video, tmp_path / "o.mp4", model="realesrgan-x2plus",
        device="cpu", tile=0, progress_cb=lambda i, n: seen.append((i, n)),
    )
    assert seen and seen[-1][0] == seen[-1][1] and seen[-1][1] >= 1


def _probe_fps(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    num, den = out.split("/")
    return round(int(num) / int(den))


def test_interpolation_boosts_fps(tiny_video, tmp_path):
    from upscaler.video import upscale_video

    dst = tmp_path / "smooth.mp4"
    upscale_video(tiny_video, dst, model="realesrgan-x2plus", device="cpu",
                  tile=0, interpolate_fps=30)
    assert _probe_fps(dst) == 30           # was 8 fps
    assert _probe_wh(dst) == (80, 64)      # still ×2 upscaled


def test_missing_input_raises(tmp_path):
    from upscaler.video import upscale_video

    with pytest.raises(FileNotFoundError):
        upscale_video(tmp_path / "nope.mp4", tmp_path / "o.mp4", device="cpu")
