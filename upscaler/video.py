"""Frame-by-frame video upscaling: ffmpeg → per-frame Real-ESRGAN → ffmpeg.

Offline only (render-and-wait). Each frame is upscaled independently, so very
fine detail can shimmer slightly between frames — fine for most footage; a
temporal model would be needed to fully remove it.

Needs ffmpeg: a system install (preferred) or the bundled binary from the
optional ``imageio-ffmpeg`` package. Audio from the source is preserved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from upscaler.engine import Upscaler
from upscaler.sharpen import unsharp_mask

ProgressCb = Optional[Callable[[int, int], None]]


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg`) or "
            '`pip install imageio-ffmpeg`.'
        ) from e


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"ffmpeg failed:\n{tail}")


def _probe(src: Path) -> tuple[str, bool]:
    """Return (fps as an 'num/den' string for ffmpeg, has_audio)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "30", True  # reasonable fallback when only ffmpeg is present
    info = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "json", str(src)],
        capture_output=True, text=True,
    )
    fps = "30"
    try:
        fps = json.loads(info.stdout)["streams"][0]["r_frame_rate"]
    except Exception:
        pass
    aud = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True,
    ).stdout.strip()
    return fps, bool(aud)


def upscale_video(
    src,
    dst,
    *,
    upscaler: Optional[Upscaler] = None,
    model: Optional[str] = None,
    scale: Optional[int] = 2,
    device: str = "auto",
    tile: int = 512,
    sharpen: float = 0.0,
    crf: int = 18,
    interpolate_fps: Optional[int] = None,
    target_long_edge: Optional[int] = None,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    progress_cb: ProgressCb = None,
) -> Path:
    """Upscale every frame of ``src`` and write the result to ``dst`` (keeps audio).

    Pass a prebuilt ``upscaler`` to reuse a loaded model. ``progress_cb(i, total)``
    is called after each frame. ``interpolate_fps`` (e.g. 60) adds motion-
    interpolated frames (ffmpeg minterpolate; slow). ``target_long_edge`` (e.g.
    3840 for 4K) fits the longest side to that many pixels after AI upscaling.
    ``trim_start``/``trim_end`` (seconds) process only that slice of the clip —
    great for testing settings on a few seconds before the full render.
    Returns the output path.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    ff = _ffmpeg()
    up = upscaler or Upscaler(model=model, scale=scale, device=device, tile=tile)
    fps, has_audio = _probe(src)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        in_dir, out_dir = tdp / "in", tdp / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # optional trim window (seconds): -ss seeks the start, -t limits duration
        start = float(trim_start) if trim_start and trim_start > 0 else 0.0
        seek = ["-ss", str(start)] if start > 0 else []
        dur = []
        if trim_end and float(trim_end) > start:
            dur = ["-t", str(float(trim_end) - start)]

        # 1. extract frames (within the trim window if given)
        _run([ff, "-y", *seek, "-i", str(src), *dur, str(in_dir / "f_%06d.png")])
        frames = sorted(in_dir.glob("f_*.png"))
        if not frames:
            raise RuntimeError(
                "No frames could be read from the video — it may be corrupt or in "
                "an unsupported format."
            )

        # 2. upscale each frame
        for i, fr in enumerate(frames, 1):
            result = up.upscale(Image.open(fr))
            if sharpen > 0:
                result = unsharp_mask(result, strength=sharpen)
            result.save(out_dir / fr.name)
            if progress_cb:
                progress_cb(i, len(frames))

        # 3. re-encode at the original fps, muxing the source audio back in
        cmd = [ff, "-y", "-framerate", fps, "-i", str(out_dir / "f_%06d.png")]
        if has_audio:
            # seek the audio to the same start; -shortest trims it to the frames
            cmd += [*seek, "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-shortest"]
        vf = []
        if target_long_edge:
            # fit the longest edge to T px (keep aspect; -2 keeps the other side
            # even for yuv420p). Scale before interpolation to keep it cheaper.
            t = int(target_long_edge)
            vf.append(
                f"scale=w='if(gte(iw,ih),{t},-2)':h='if(gte(iw,ih),-2,{t})':"
                "flags=lanczos"
            )
        if interpolate_fps:
            # motion-compensated interpolation to a higher fps (duration unchanged,
            # so audio stays in sync). Heavy but built into ffmpeg.
            vf.append(f"minterpolate=fps={int(interpolate_fps)}:mi_mode=mci:"
                      "mc_mode=aobmc:me_mode=bidir:vsbmc=1")
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                "-movflags", "+faststart", str(dst)]
        _run(cmd)
    return dst
