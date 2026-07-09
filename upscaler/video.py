"""Frame-by-frame video upscaling: ffmpeg → per-frame Real-ESRGAN → ffmpeg.

Offline only (render-and-wait). Each frame is upscaled independently, so very
fine detail can shimmer slightly between frames — fine for most footage; a
temporal model would be needed to fully remove it.

Renders checkpoint per frame by default (``resume=True``): a cancelled or
crashed job re-run with the same video + settings skips every frame it already
finished instead of starting over. Checkpoints live in ``upscaler/resume/``
(override with ``UPSCALER_RESUME_DIR``) and are deleted on success; abandoned
ones are swept after 30 days.

Needs ffmpeg: a system install (preferred) or the bundled binary from the
optional ``imageio-ffmpeg`` package. Audio from the source is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from upscaler.engine import CancelledError, Upscaler
from upscaler.sharpen import unsharp_mask

ProgressCb = Optional[Callable[[int, int], None]]
CancelCb = Optional[Callable[[], bool]]

# Interrupted renders older than this are swept on the next video job.
_RESUME_MAX_AGE_S = 30 * 24 * 3600


def _resume_root() -> Path:
    """Where interrupted-render checkpoints live (env override, else alongside
    the package like the weights cache). Read per call so tests can redirect it."""
    return Path(
        os.environ.get("UPSCALER_RESUME_DIR", Path(__file__).resolve().parent / "resume")
    )


def _job_dir(src: Path, engine, sharpen: float,
             trim_start: Optional[float], trim_end: Optional[float]) -> Path:
    """Checkpoint dir for this (video, frame-affecting settings) combination.

    Keyed on the file's content head + size rather than its path: the GUI
    re-uploads to a fresh temp path each session, and the same footage should
    resume regardless of where it sits. Settings that only affect the final
    encode (crf, fps interpolation, target size) are deliberately excluded, so
    changing them still reuses every upscaled frame.
    """
    h = hashlib.sha256()
    with open(src, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(str(src.stat().st_size).encode())
    h.update(f"|{engine.spec.name}|{engine.scale}|{engine.__class__.__name__}"
             f"|{float(sharpen)}|{trim_start or 0}|{trim_end or 0}".encode())
    return _resume_root() / h.hexdigest()[:24]


def _sweep_stale_jobs(root: Path) -> None:
    """Drop abandoned checkpoints so half-finished renders can't hoard disk."""
    try:
        cutoff = time.time() - _RESUME_MAX_AGE_S
        for d in root.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


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
        # Only ffmpeg present (e.g. the bundled imageio-ffmpeg binary, which
        # ships no ffprobe): parse the `-i` banner instead of assuming 30 fps —
        # re-encoding a 24/60 fps clip at 30 changes its speed and desyncs audio.
        info = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-i", str(src)],
            capture_output=True, text=True,
        )
        err = info.stderr or ""
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", err)
        return (m.group(1) if m else "30"), ("Audio:" in err)
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
    should_cancel: CancelCb = None,
    onnx: bool = False,
    resume: bool = True,
) -> Path:
    """Upscale every frame of ``src`` and write the result to ``dst`` (keeps audio).

    Pass a prebuilt ``upscaler`` to reuse a loaded model. ``progress_cb(i, total)``
    is called after each frame. ``interpolate_fps`` (e.g. 60) adds motion-
    interpolated frames (ffmpeg minterpolate; slow). ``target_long_edge`` (e.g.
    3840 for 4K) fits the longest side to that many pixels after AI upscaling.
    ``trim_start``/``trim_end`` (seconds) process only that slice of the clip —
    great for testing settings on a few seconds before the full render.

    ``resume`` (default on) checkpoints each finished frame to disk, so a
    cancelled/crashed render picks up where it left off when re-run with the
    same video and settings. The checkpoint is deleted once ``dst`` is written.
    Returns the output path.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    ff = _ffmpeg()
    if upscaler is not None:
        up = upscaler
    elif onnx:
        # ONNX Runtime backend — with the DirectML build this is the way to an
        # AMD/Intel GPU on native Windows, where torch is CPU-only.
        from upscaler.onnx_engine import OnnxUpscaler

        up = OnnxUpscaler(model=model, scale=scale, device=device, tile=tile)
    else:
        up = Upscaler(model=model, scale=scale, device=device, tile=tile)
    fps, has_audio = _probe(src)

    if resume:
        work = _job_dir(src, up, sharpen, trim_start, trim_end)
        _sweep_stale_jobs(work.parent)
        work.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
        tdp = work
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        tdp = Path(tmp_ctx.name)
    try:
        in_dir, out_dir = tdp / "in", tdp / "out"
        in_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        extract_marker = tdp / "extract.done"

        # optional trim window (seconds): -ss seeks the start, -t limits duration
        start = float(trim_start) if trim_start and trim_start > 0 else 0.0
        seek = ["-ss", str(start)] if start > 0 else []
        dur = []
        if trim_end and float(trim_end) > start:
            dur = ["-t", str(float(trim_end) - start)]

        # 1. extract frames (within the trim window if given). The marker means
        # a previous run finished extracting — trust its frames; anything short
        # of that could be a partial extraction, so start over.
        if not extract_marker.exists():
            for old in in_dir.glob("f_*.png"):
                old.unlink()
            _run([ff, "-y", *seek, "-i", str(src), *dur, str(in_dir / "f_%06d.png")])
            extract_marker.touch()
        frames = sorted(in_dir.glob("f_*.png"))
        if not frames:
            raise RuntimeError(
                "No frames could be read from the video — it may be corrupt or in "
                "an unsupported format."
            )

        # 2. upscale each frame (both engines accept should_cancel), skipping
        # frames a previous interrupted run already finished
        for i, fr in enumerate(frames, 1):
            done = out_dir / fr.name
            if resume and done.exists() and done.stat().st_size > 0:
                if progress_cb:
                    progress_cb(i, len(frames))
                continue
            if should_cancel and should_cancel():
                raise CancelledError("Cancelled.")
            result = up.upscale(Image.open(fr), should_cancel=should_cancel)
            if sharpen > 0:
                result = unsharp_mask(result, strength=sharpen)
            # write-then-rename: a crash mid-save must not leave a truncated
            # PNG that the next resume would trust (and ffmpeg would choke on)
            part = done.with_name(done.name + ".part")
            result.save(part, format="PNG")
            os.replace(part, done)
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

        # success — the checkpoint has served its purpose
        if resume:
            shutil.rmtree(tdp, ignore_errors=True)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    return dst
