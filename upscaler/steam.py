"""Steam profile Workshop Showcase — slice one picture (or clip) into the five
tiles Steam shows side by side.

Geometry comes from Steam's own profile stylesheet (``profilev2.css``): the
profile's left column is 652px wide with 10px of padding, so a showcase row is
632px; the Workshop Showcase puts five ``max-width: 20%`` containers in that
row, each with a 2px margin, so every tile renders 122.4px wide, neighbours are
separated by a 4px gap, and the row has a 2px margin at either end. Tile height
is free — the image is shown at ``width: 100%`` with its own aspect — so a
square 122px tile is the stock look and taller rows are allowed.

Everything is composed with the Lian Li panel compositor (same fit / pan / zoom
/ background rules) onto a canvas that spans exactly the five tiles *and the
four gaps between them*, then cut into slices — so a picture continues across
the gaps instead of jumping. Animated sources export one APNG per slice via
ffmpeg (or one GIF per slice), shrunk step by step (palette → fps → length)
until every file fits the upload cap; stills export five plain PNGs with PIL
only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from PIL import Image, ImageDraw

from upscaler import panel
from upscaler.video import _ffmpeg

# ── Steam geometry (display / CSS pixels) ─────────────────────────────────────
N_SLOTS = 5
ROW_CSS_W = 632.0        # profile column (652px) minus 10px padding each side
TILE_CSS_W = 122.4       # 20% of the row, minus a 2px margin on each side
GAP_CSS = 4.0            # 2px right margin + the neighbour's 2px left margin
DEFAULT_TILE_H = 122     # square tiles — how Steam's empty-slot placeholders look
MIN_TILE_H, MAX_TILE_H = 40, 600
SCALES: dict[str, int] = {
    "1× · native (122px tiles)": 1,
    "2× · HiDPI (245px tiles)": 2,
}
DEFAULT_SCALE_LABEL = next(iter(SCALES))

# ── Upload limits ─────────────────────────────────────────────────────────────
# Steam documents 8 MB per artwork item; community reports put the reliable
# ceiling lower, so the budget defaults to 5 MB and stays user-adjustable.
DEFAULT_MAX_MB = 5.0
STEAM_MAX_MB = 8.0
MAX_DURATION_SEC = 60    # showcase loops are short — bound the frame extraction
MAX_FPS = panel.MAX_FPS
LOOP_STYLES = panel.LOOP_STYLES
FITS = panel.FITS
ANIM_FORMATS = ("apng", "gif")   # APNG keeps full colour; GIF is always ≤ 256 colours

UPLOAD_GUIDE = """\
Steam only shows animated tiles if you upload them as **artwork that Steam files
under the Workshop**, which needs a one-line browser trick on the upload page:

1. Open `https://steamcommunity.com/sharedfiles/edititem/767/3/` in your browser
   (logged in).
2. Press **F12**, pick the **Console** tab, paste this and press Enter:
   `$J('[name=consumer_app_id]').val(480);$J('[name=file_type]').val(0);$J('[name=visibility]').val(0);`
3. Choose **tile 1**, give it a title, tick the agreement and upload. Repeat
   steps 1–3 for tiles 2 → 5 (same title is fine).
4. On your profile, **Edit Profile → Showcases → Workshop Showcase**, then
   click each empty slot and pick the tiles in order 1 → 5.

Each file must stay under Steam's per-item cap (8 MB documented; ≤ 5 MB is the
safe bet), which is what the size budget below is for. The trick is community
knowledge, not a Steam feature, so if an upload fails, check a current
"Steam workshop showcase animated" guide for changes.
"""


class CancelledError(Exception):
    """The caller's cancel flag turned true mid-export."""


@dataclass
class ShowcaseParams:
    """Everything that shapes the row. ``tile_h`` is in display (1×) pixels;
    ``scale`` renders the same layout at 1× or 2× for HiDPI screens."""
    fit: str = "cover"
    zoom: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0
    bg_color: str = "#000000"
    tile_h: int = DEFAULT_TILE_H
    scale: int = 1


@dataclass(frozen=True)
class Layout:
    """Integer pixel geometry of the composed row at a given render scale."""
    scale: int
    tile_w: int
    tile_h: int
    gap: int

    @property
    def width(self) -> int:
        return N_SLOTS * self.tile_w + (N_SLOTS - 1) * self.gap

    @property
    def height(self) -> int:
        return self.tile_h

    @property
    def boxes(self) -> list[tuple[int, int, int, int]]:
        """PIL crop boxes (left, top, right, bottom) of the five tiles."""
        step = self.tile_w + self.gap
        return [(i * step, 0, i * step + self.tile_w, self.tile_h) for i in range(N_SLOTS)]


def layout(p: ShowcaseParams) -> Layout:
    s = max(1, int(p.scale))
    tile_h = max(MIN_TILE_H, min(MAX_TILE_H, int(p.tile_h)))
    return Layout(s, round(TILE_CSS_W * s), tile_h * s, round(GAP_CSS * s))


# ── Compositing ───────────────────────────────────────────────────────────────
def _panel_params(p: ShowcaseParams) -> panel.PanelParams:
    return panel.PanelParams(
        fit=p.fit, zoom=float(p.zoom), off_x=float(p.off_x), off_y=float(p.off_y),
        bg_type="solid", bg_color=p.bg_color,
    )


def compose_row(src: Image.Image | None, p: ShowcaseParams, fast: bool = False) -> Image.Image:
    """The whole row — five tiles plus the gaps between them — as one image."""
    lay = layout(p)
    return panel.compose_frame(src, _panel_params(p), fast=fast, canvas=(lay.width, lay.height))


def slice_row(row: Image.Image, p: ShowcaseParams) -> list[Image.Image]:
    """Cut a composed row into the five tile images (the gaps are dropped)."""
    return [row.crop(box) for box in layout(p).boxes]


# ── Preview ───────────────────────────────────────────────────────────────────
_PROFILE_BG = (23, 29, 37)        # Steam profile page ground
_SHOWCASE_BG = (16, 18, 20)       # .showcase_content_bg
_HEADER_FG = (255, 255, 255)
_DIM_BG = (22, 22, 26)
_GAP_INK = (10, 11, 13)


def preview(src_path: str | None, p: ShowcaseParams,
            frame: Image.Image | None = None, width: int = 1000) -> Image.Image:
    """Editor preview: on top, the framing view (source dimmed, the kept row
    bright, the four gap columns blacked out); below it, a Steam-style mockup
    of the Workshop Showcase box with the five tiles and their real spacing."""
    lay = layout(p)
    pp = _panel_params(p)
    src = frame if frame is not None else panel._first_image(src_path)

    ds = width / lay.width
    disp_h = max(1, round(lay.height * ds))
    comp = compose_row(src, p, fast=True).resize((width, disp_h), Image.BILINEAR)
    gap_cols = [(round(x1 * ds), round((x1 + lay.gap) * ds)) for (_, _, x1, _) in lay.boxes[:-1]]

    # ── framing panel ──
    pad_x = round(width * 0.06)
    pad_y = max(28, round(disp_h * 0.6))
    fw, fh = width + 2 * pad_x, disp_h + 2 * pad_y
    framing = Image.new("RGB", (fw, fh), _DIM_BG)
    if src is not None:
        dw, dh = panel._drawn_size(src.width, src.height, lay.width, lay.height, p.fit, p.zoom)
        px, py = panel._paste_pos(dw, dh, lay.width, lay.height, pp)
        rs = src.convert("RGBA").resize((max(1, round(dw * ds)), max(1, round(dh * ds))), Image.BILINEAR)
        framing.paste(rs, (round(pad_x + px * ds), round(pad_y + py * ds)), rs)
        framing = Image.blend(framing, Image.new("RGB", (fw, fh), _DIM_BG), 0.5)
    framing.paste(comp, (pad_x, pad_y))
    d = ImageDraw.Draw(framing)
    for gx0, gx1 in gap_cols:
        d.rectangle([pad_x + gx0, pad_y, pad_x + max(gx0 + 1, gx1) - 1, pad_y + disp_h - 1], fill=_GAP_INK)
    d.rectangle([pad_x, pad_y, pad_x + width - 1, pad_y + disp_h - 1], outline=(255, 255, 255), width=2)

    # ── Steam mockup panel ──
    tiles = [comp.crop((round(x0 * ds), 0, round(x1 * ds), disp_h)) for (x0, _, x1, _) in lay.boxes]
    box_pad = round(8 * ds)
    head_h = round(34 * ds)
    mock_h = head_h + box_pad * 2 + disp_h + round(18 * ds)
    mock = Image.new("RGB", (fw, mock_h), _PROFILE_BG)
    d = ImageDraw.Draw(mock)
    font = panel._load_font("Arial", max(11, round(15 * ds)))
    d.text((pad_x, round(8 * ds)), "Workshop Showcase", fill=_HEADER_FG, font=font)
    small = panel._load_font("Arial", max(9, round(11 * ds)))
    d.text((pad_x + width, round(12 * ds)), "5 items", fill=(140, 150, 160), font=small, anchor="ra")
    by0 = head_h
    d.rounded_rectangle([pad_x - box_pad, by0, pad_x + width + box_pad - 1, by0 + disp_h + 2 * box_pad - 1],
                        radius=round(5 * ds), fill=_SHOWCASE_BG)
    for tile, (x0, _, _, _) in zip(tiles, lay.boxes):
        mock.paste(tile, (pad_x + round(x0 * ds), by0 + box_pad))

    out = Image.new("RGB", (fw, fh + mock_h), _PROFILE_BG)
    out.paste(framing, (0, 0))
    out.paste(mock, (0, fh))
    return out


# ── Export ────────────────────────────────────────────────────────────────────
@dataclass
class ExportResult:
    paths: list[str]
    layout: Layout
    fmt: str = "png"   # png (stills) | apng | gif
    fps: int = 0
    colors: int | None = None
    duration: float = 0.0
    sizes: list[int] | None = None
    fits: bool = True
    attempts: int = 1
    animated: bool = False

    @property
    def max_size(self) -> int:
        return max(self.sizes) if self.sizes else 0


def _tile_paths(out_dir: str, stem: str, ext: str = ".png") -> list[str]:
    return [os.path.join(out_dir, f"{stem}_{i}{ext}") for i in range(1, N_SLOTS + 1)]


def export_stills(src_path: str | None, p: ShowcaseParams,
                  out_dir: str | None = None, stem: str = "steam") -> ExportResult:
    """Five still PNG tiles from the first frame of the source (PIL only)."""
    src = panel._first_image(src_path)
    if src is None:
        raise ValueError("Couldn't decode the uploaded media — it may be corrupt or an "
                         "unsupported format.")
    out_dir = out_dir or tempfile.mkdtemp(prefix="steam_")
    os.makedirs(out_dir, exist_ok=True)
    paths = _tile_paths(out_dir, stem)
    for tile, path in zip(slice_row(compose_row(src, p), p), paths):
        tile.save(path, "PNG", optimize=True)
    return ExportResult(paths, layout(p), sizes=[os.path.getsize(x) for x in paths])


def budget_ladder(fps: int, fmt: str = "apng") -> list[tuple[int, int | None, float]]:
    """Encode attempts as ``(fps, palette colours or None, length fraction)``,
    best quality first: truecolor as asked → a 256-colour palette (near
    invisible on tiles this small) → lower fps → 128 colours → a shorter clip.
    The last rung is the most compressed we'll go. GIF is always a palette, so
    its ladder starts at the 256-colour rung."""
    fps = max(1, min(MAX_FPS, int(fps)))
    fps_steps: list[int] = []
    for f in (fps, 24, 20, 15, 12, 10):
        if f <= fps and f not in fps_steps:
            fps_steps.append(f)
    ladder: list[tuple[int, int | None, float]] = [(fps, None, 1.0), (fps, 256, 1.0)]
    ladder += [(f, 256, 1.0) for f in fps_steps[1:]]
    low = fps_steps[-1]
    ladder.append((low, 128, 1.0))
    ladder += [(low, 128, frac) for frac in (0.75, 0.5, 0.35, 0.25)]
    if fmt == "gif":
        ladder = [(f, c or 256, fr) for f, c, fr in ladder]
        ladder = [step for i, step in enumerate(ladder) if i == 0 or step != ladder[i - 1]]
    return ladder


def _trim_window(src_path: str, trim_start: float, trim_end: float) -> tuple[float, float, bool, bool]:
    """(start, duration, explicit_end, duration_known) for the export window."""
    dur_total = panel.media_duration(src_path)
    start = max(0.0, float(trim_start or 0))
    has_end = bool(trim_end and float(trim_end) > start)
    if has_end:
        dur = min(float(trim_end) - start, MAX_DURATION_SEC)
    elif dur_total > start:
        dur = min(dur_total - start, MAX_DURATION_SEC)
    else:
        dur = MAX_DURATION_SEC  # unknown length: let ffmpeg read to EOF, capped
    return start, dur, has_end, has_end or dur_total > start


def extract_size(src_w: int, src_h: int, lay: Layout, p: ShowcaseParams) -> tuple[int, int] | None:
    """The size frames can be shrunk to *during extraction* without changing
    the composition: the fit's drawn size depends only on the source aspect,
    so a pre-scaled frame lands on the row identically. Returns None when the
    fit would enlarge the source (then the original pixels are the best we
    have) — a 4K phone clip otherwise hits the disk as thousands of 4K PNGs."""
    dw, dh = panel._drawn_size(src_w, src_h, lay.width, lay.height, p.fit, p.zoom)
    if dw >= src_w or dh >= src_h:
        return None
    return dw, dh


def _extract_frames(src_path: str, fps: float, start: float, dur: float, into: str,
                    size: tuple[int, int] | None = None, progress=None, cancel=None,
                    total_hint: int = 0) -> int:
    """Extract ``frame_%05d.png`` files, scaled to ``size`` on the way out.
    Streams ffmpeg's own ``-progress`` feed so the UI keeps moving through a
    long decode (and anything proxying it sees traffic), and kills ffmpeg as
    soon as ``cancel()`` turns true. Returns the frame count."""
    vf = f"fps={fps}"
    if size:
        vf += f",scale={size[0]}:{size[1]}:flags=lanczos"
    cmd = [_ffmpeg(), "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src_path)]
    if dur > 0:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", vf, "-vsync", "0", os.path.join(into, "frame_%05d.png")]

    err_path = os.path.join(into, "_ffmpeg.err")
    last = 0.0
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True,
                                encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:  # one key=value per line, a block every ~0.5 s
                if cancel and cancel():
                    raise CancelledError("Export cancelled.")
                if not line.startswith("frame=") or not progress:
                    continue
                try:
                    frame = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                now = time.perf_counter()
                if now - last >= 0.5:
                    last = now
                    if total_hint:
                        progress(0.05 + 0.15 * min(1.0, frame / total_hint),
                                 desc=f"Extracting frame {frame}/{total_hint}…")
                    else:
                        progress(0.1, desc=f"Extracting frame {frame}…")
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if rc != 0:
        with open(err_path, errors="replace") as fh:
            tail = "\n".join(fh.read().strip().splitlines()[-6:])
        raise RuntimeError(f"ffmpeg failed:\n{tail}")
    os.remove(err_path)
    return len([f for f in os.listdir(into) if f.startswith("frame_")])


def _link(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _sequence(comp_dir: str, n_use: int, mode: str, fps: int) -> str:
    """Materialise the frame order for one encode attempt (first ``n_use``
    composited frames, with the loop style applied) and return the ffmpeg
    input pattern. Rebuilt from scratch per attempt so a shorter retry never
    picks up stale frames from a longer one."""
    seq = os.path.join(comp_dir, "seq")
    shutil.rmtree(seq, ignore_errors=True)
    if mode in ("boomerang", "crossfade") and n_use >= 4:
        return panel._apply_loop(comp_dir, n_use, mode, fps)
    os.makedirs(seq)
    for i in range(1, n_use + 1):
        _link(os.path.join(comp_dir, f"c_{i:05d}.png"), os.path.join(seq, f"s_{i:05d}.png"))
    return os.path.join(seq, "s_%05d.png")


def _encode_slices(pattern: str, base_fps: int, out_fps: int, colors: int | None,
                   lay: Layout, paths: list[str], fmt: str = "apng") -> None:
    """One ffmpeg run: crop the composited row into the five tiles and write
    each as a looping APNG (palette-quantised when ``colors`` is set) or GIF
    (always a palette; ordered dither because it LZW-compresses far better)."""
    gif = fmt == "gif"
    if gif:
        colors = colors or 256
    graph = [f"[0:v]fps={out_fps},split={N_SLOTS}" + "".join(f"[s{i}]" for i in range(N_SLOTS))]
    for i, (x0, y0, x1, y1) in enumerate(lay.boxes):
        crop = f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        if colors:
            dither = "bayer" if gif else "sierra2_4a"
            graph.append(
                f"[s{i}]{crop},split[c{i}a][c{i}b];"
                f"[c{i}a]palettegen=max_colors={colors}[p{i}];"
                f"[c{i}b][p{i}]paletteuse=dither={dither}[o{i}]"
            )
        else:
            graph.append(f"[s{i}]{crop},format=rgb24[o{i}]")
    cmd = [_ffmpeg(), "-y", "-framerate", str(base_fps), "-i", pattern,
           "-filter_complex", ";".join(graph)]
    for i, path in enumerate(paths):
        cmd += ["-map", f"[o{i}]"]
        cmd += ["-f", "gif", "-loop", "0"] if gif else ["-f", "apng", "-plays", "0", "-pred", "mixed"]
        cmd.append(path)
    panel._ffmpeg_run(cmd)
    for path in paths:
        if not os.path.exists(path) or not os.path.getsize(path):
            raise RuntimeError(f"ffmpeg produced an empty {fmt.upper()}.")


def export_animated(
    src_path: str | None,
    p: ShowcaseParams,
    *,
    fps: int = 30,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    loop_mode: str = "normal",
    max_mb: float = DEFAULT_MAX_MB,
    out_dir: str | None = None,
    stem: str = "steam",
    fmt: str = "apng",
    progress=None,
    cancel=None,
) -> ExportResult:
    """Five looping APNG (or GIF) tiles. Frames are extracted once at ``fps``
    (already shrunk to the size the row needs) and composited once; then the
    budget ladder re-encodes (cheap at tile size) until every tile is ≤
    ``max_mb`` (0 disables the budget). Returns the first attempt that fits,
    or the last (smallest) one with ``fits=False``. ``cancel`` is polled
    throughout and raises ``CancelledError`` when it turns true."""
    if not src_path:
        raise ValueError("Upload media first.")
    if fmt not in ANIM_FORMATS:
        raise ValueError(f"fmt must be one of {ANIM_FORMATS}, not {fmt!r}")
    fps = max(1, min(MAX_FPS, int(fps)))
    lay = layout(p)
    base = panel._first_image(src_path)
    if base is None:
        raise ValueError("Couldn't decode the uploaded media — it may be corrupt or an "
                         "unsupported format.")
    start, dur, has_end, known = _trim_window(src_path, trim_start, trim_end)

    def check_cancel() -> None:
        if cancel and cancel():
            raise CancelledError("Export cancelled.")

    work = tempfile.mkdtemp(prefix="steam_work_")
    raw = os.path.join(work, "raw")
    comp = os.path.join(work, "comp")
    os.makedirs(raw)
    os.makedirs(comp)
    try:
        if progress:
            progress(0.05, desc="Extracting frames…")
        n = _extract_frames(
            src_path, fps, start, dur, raw, size=extract_size(base.width, base.height, lay, p),
            progress=progress, cancel=cancel,
            total_hint=int(round(dur * fps)) if known else 0,
        )
        if n <= 1:
            hold = dur if has_end else min(dur, 3.0)
            n = max(2, int(round(hold * fps)))
            for i in range(1, n + 1):
                base.save(os.path.join(raw, f"frame_{i:05d}.png"))

        files = sorted(f for f in os.listdir(raw) if f.startswith("frame_"))
        if not files:
            raise ValueError("No frames could be extracted from the source.")
        n = len(files)
        t0 = time.perf_counter()
        for i, fn in enumerate(files, 1):
            check_cancel()
            with Image.open(os.path.join(raw, fn)) as fr:
                compose_row(fr.convert("RGB"), p).save(os.path.join(comp, f"c_{i:05d}.png"))
            if progress and (i % 10 == 0 or i == n):
                el = time.perf_counter() - t0
                progress(0.1 + 0.5 * i / n, desc=f"Compositing frame {i}/{n} · {i / n:.0%} · "
                                                 f"{int(el // 60)}:{int(el % 60):02d} elapsed")

        out_dir = out_dir or tempfile.mkdtemp(prefix="steam_")
        os.makedirs(out_dir, exist_ok=True)
        paths = _tile_paths(out_dir, stem, ".gif" if fmt == "gif" else ".png")
        cap = int(max_mb * 1024 * 1024) if max_mb and max_mb > 0 else 0
        attempts = budget_ladder(fps, fmt) if cap else budget_ladder(fps, fmt)[:1]

        result = ExportResult(paths, lay, fmt=fmt, animated=True)
        for k, (f, colors, frac) in enumerate(attempts, 1):
            check_cancel()
            n_use = max(2, int(round(n * frac)))
            if progress:
                what = f"{f} fps · {colors or 'full'} colours"
                if frac < 1:
                    what += f" · {n_use / fps:.1f}s"
                progress(0.6 + 0.35 * k / len(attempts),
                         desc=f"Encoding {fmt.upper()} ({what})…")
            _encode_slices(_sequence(comp, n_use, loop_mode, fps), fps, f, colors, lay, paths, fmt)
            sizes = [os.path.getsize(x) for x in paths]
            result = ExportResult(paths, lay, fmt=fmt, fps=f, colors=colors,
                                  duration=n_use / fps, sizes=sizes,
                                  fits=(not cap or max(sizes) <= cap),
                                  attempts=k, animated=True)
            if result.fits:
                break
        if progress:
            progress(1.0)
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fmt_size(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MB" if n >= 1024 * 1024 else f"{max(1, round(n / 1024))} KB"


def describe(res: ExportResult, max_mb: float = 0.0) -> str:
    """A short human summary of an export (shared by the GUI and CLI)."""
    lay = res.layout
    mb = [s / (1024 * 1024) for s in (res.sizes or [])]
    sizes = " · ".join(_fmt_size(s) for s in (res.sizes or []))
    head = (f"{N_SLOTS} tiles · {lay.tile_w}×{lay.tile_h}px each · "
            f"{lay.gap}px gaps ({lay.scale}×)")
    if not res.animated:
        return f"{head} · still PNG\n{sizes}"
    detail = (f"{res.fmt.upper()} · {res.fps} fps · {res.colors or 'full'} colours · "
              f"{res.duration:.1f}s · looping")
    if res.attempts > 1:
        detail += f" · shrunk in {res.attempts} steps to fit"
    line = f"{head}\n{detail}\n{sizes}"
    if not res.fits and max_mb:
        line += (f"\n⚠ Largest tile is {max(mb):.2f} MB, over the {max_mb:g} MB budget "
                 "even at the smallest setting — trim the clip or lower the tile height.")
    return line
