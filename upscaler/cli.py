"""Command-line interface: ``upscaler <input> [-o out] [--scale N] [--sharpen ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataclasses import replace

from PIL import Image
from tqdm import tqdm

from upscaler import fit
from upscaler.convert import FORMATS, convert_file, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.video import upscale_video
from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler
from upscaler.models.registry import DEBLUR_MODELS, DEFAULT_DEBLUR_MODEL, MODELS
from upscaler.sharpen import unsharp_mask

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
               ".gif", ".avif", ".heic", ".ico", ".tga", ".ppm"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


def _gather_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return [path]


def _output_path(src: Path, out: Path | None, scale: int) -> Path:
    if out and out.suffix:  # explicit file target
        out.parent.mkdir(parents=True, exist_ok=True)  # fail early, not post-compute
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{src.stem}_x{scale}.png"


def _no_clobber(src: Path, dst: Path, tag: str) -> Path:
    """Never let a derived output path silently overwrite the source in place
    (e.g. `upscaler convert photo.png -f PNG` with no -o)."""
    try:
        same = dst.resolve() == src.resolve()
    except OSError:
        same = str(dst) == str(src)
    return dst.with_name(f"{dst.stem}_{tag}{dst.suffix}") if same else dst


def _convert_output_path(src: Path, out: Path | None, fmt: str) -> Path:
    ext = extension_for(fmt)
    if out and out.suffix:  # explicit file target
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return _no_clobber(src, out_dir / f"{src.stem}.{ext}", "converted")


def build_convert_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler convert",
        description="Convert image file formats (fast, no AI models).",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file or directory.")
    p.add_argument("-o", "--output", type=Path, help="Output file or directory.")
    p.add_argument(
        "-f", "--format", choices=list(FORMATS),
        help="Target format. If omitted, inferred from -o's extension.",
    )
    p.add_argument(
        "-q", "--quality", type=int, default=90,
        help="Quality 1-100 for lossy formats (JPEG/WebP). Default 90.",
    )
    p.add_argument("--lossless", action="store_true", help="Lossless WebP.")
    return p


# Secondary spellings of extensions whose FORMATS entry lists the primary one.
_EXT_ALIASES = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff", "heif": "heic"}


def _format_from_output(out: Path | None) -> str | None:
    if out and out.suffix:
        ext = out.suffix.lstrip(".").lower()
        ext = _EXT_ALIASES.get(ext, ext)
        return next((k for k, v in FORMATS.items() if v[1] == ext), None)
    return None


def run_convert(argv: list[str]) -> int:
    args = build_convert_parser().parse_args(argv)

    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    fmt = args.format or _format_from_output(args.output)
    if fmt is None:
        print(
            "error: specify --format, or -o with a known extension "
            f"({', '.join(v[1] for v in FORMATS.values())}).",
            file=sys.stderr,
        )
        return 2

    inputs = _gather_inputs(args.input)
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder",
              file=sys.stderr)
        return 2

    failed = 0
    for src in tqdm(inputs, disable=len(inputs) == 1, desc="convert"):
        try:
            dst = _convert_output_path(src, args.output, fmt)
            convert_file(src, dst, fmt=fmt, quality=args.quality, lossless=args.lossless)
        except (Image.UnidentifiedImageError, OSError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0 if failed < len(inputs) else 2


def build_pdf_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler pdf", description="Image ⇄ PDF conversion."
    )
    sub = p.add_subparsers(dest="action", required=True)

    b = sub.add_parser("build", help="Combine images into a (multi-page) PDF.")
    b.add_argument(
        "inputs", type=Path, nargs="+",
        help="Image files (kept in the given order) and/or directories.",
    )
    b.add_argument("-o", "--output", type=Path, required=True, help="Output .pdf path.")

    e = sub.add_parser("extract", help="Render a PDF's pages to PNGs.")
    e.add_argument("input", type=Path, help="Input .pdf file.")
    e.add_argument(
        "-o", "--output", type=Path,
        help="Output directory (default: <pdf-stem>_pages next to the PDF).",
    )
    e.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150).")
    return p


def run_pdf(argv: list[str]) -> int:
    args = build_pdf_parser().parse_args(argv)

    if args.action == "build":
        paths: list[Path] = []
        for inp in args.inputs:
            if not inp.exists():
                print(f"error: input not found: {inp}", file=sys.stderr)
                return 2
            paths.extend(_gather_inputs(inp))
        if not paths:
            print("error: no images to combine", file=sys.stderr)
            return 2
        images = []
        for p in paths:
            try:
                images.append(Image.open(p))
            except (Image.UnidentifiedImageError, OSError) as e:
                print(f"error on {p.name}: {e} (skipped)", file=sys.stderr)
        if not images:
            print("error: none of the inputs are readable images", file=sys.stderr)
            return 2
        try:
            # Image.open is lazy: a truncated file passes open() and fails here.
            data = images_to_pdf(images)
        except OSError as e:
            print(f"error: couldn't build the PDF: {e}", file=sys.stderr)
            return 2
        if args.output.parent != Path():
            args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print(f"→ {args.output} ({len(images)} page(s))", file=sys.stderr)
        return 0

    # extract
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        pages = pdf_to_images(str(args.input), dpi=args.dpi)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = args.output or args.input.parent / f"{args.input.stem}_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(pages, 1):
        im.save(out_dir / f"{args.input.stem}_p{i:03d}.png")
    print(f"→ {out_dir} ({len(pages)} page(s))", file=sys.stderr)
    return 0


def build_video_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler video",
        description="Upscale a video frame-by-frame (offline; keeps audio). "
        "Needs ffmpeg.",
    )
    p.add_argument(
        "input", type=Path, help="Input video file, or a directory of videos."
    )
    p.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output .mp4 file, or a directory (required for a folder of videos).",
    )
    p.add_argument(
        "-s", "--scale", type=int, default=2, choices=(2, 4),
        help="Upscale factor (default: 2 — gentler/faster, less flicker).",
    )
    p.add_argument("-m", "--model", choices=sorted(MODELS), help="Explicit model (overrides --scale).")
    p.add_argument(
        "--sharpen", nargs="?", type=float, const=1.0, default=0.0,
        help="Unsharp strength per frame (default off).",
    )
    p.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device (default: auto).",
    )
    p.add_argument("--tile", type=int, default=512, help="Tile size, 0 disables tiling.")
    p.add_argument("--crf", type=int, default=18, help="x264 quality (lower is better; default 18).")
    p.add_argument(
        "--fps", type=int, default=None,
        help="Motion-interpolate to this fps for smoother motion (e.g. 60). Slow.",
    )
    p.add_argument(
        "--size", type=int, default=None, metavar="PX",
        help="Fit the longest edge to PX after upscaling (e.g. 3840 for 4K).",
    )
    p.add_argument(
        "--start", type=float, default=None, metavar="SEC",
        help="Trim: start time in seconds (process only from here).",
    )
    p.add_argument(
        "--end", type=float, default=None, metavar="SEC",
        help="Trim: end time in seconds (process only up to here).",
    )
    p.add_argument(
        "--onnx", action="store_true",
        help="Use the ONNX Runtime backend per frame (with onnxruntime-directml "
        "this runs on AMD/Intel GPUs on Windows, where torch is CPU-only).",
    )
    return p


def _video_output_path(src: Path, out: Path, scale: int) -> Path:
    if out.suffix:  # explicit file target
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{src.stem}_x{scale}.mp4"


def run_video(argv: list[str]) -> int:
    args = build_video_parser().parse_args(argv)
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    if args.input.is_dir():
        inputs = sorted(p for p in args.input.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
        if not inputs:
            print(f"error: no videos found in {args.input}", file=sys.stderr)
            return 2
    else:
        inputs = [args.input]

    if len(inputs) > 1 and args.output.suffix:
        print("error: --output must be a directory when processing a folder",
              file=sys.stderr)
        return 2

    # Build the model once and reuse it across every clip.
    if args.onnx:
        from upscaler.onnx_engine import OnnxUpscaler

        up = OnnxUpscaler(
            model=args.model, scale=args.scale, device=args.device, tile=args.tile
        )
        backend = "GPU (DirectML)" if up.provider.startswith("Dml") else up.provider
    else:
        up = Upscaler(
            model=args.model, scale=args.scale, device=args.device, tile=args.tile
        )
        backend = up.device.type
    print(f"upscaling {len(inputs)} clip(s) ×{up.scale} on {backend} "
          "(this can take a while)…", file=sys.stderr)

    failures = 0
    for idx, src in enumerate(inputs, 1):
        dst = _video_output_path(src, args.output, up.scale)
        last = [0]

        def cb(i: int, total: int, _idx=idx, _src=src) -> None:
            if i != last[0]:
                last[0] = i
                tag = f"[{_idx}/{len(inputs)}] " if len(inputs) > 1 else ""
                print(f"\r  {tag}{_src.name}: frame {i}/{total}",
                      end="", file=sys.stderr, flush=True)

        try:
            upscale_video(
                src, dst, upscaler=up, sharpen=args.sharpen, crf=args.crf,
                interpolate_fps=args.fps, target_long_edge=args.size,
                trim_start=args.start, trim_end=args.end, progress_cb=cb,
            )
            print(f"\n→ {dst}", file=sys.stderr)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"\nerror on {src.name}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


_ONNX_HINT = 'background removal needs onnxruntime. Install it with: pip install -e ".[onnx]"'


def _png_output_path(src: Path, out: Path | None) -> Path:
    if out and out.suffix:  # explicit file target
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return _no_clobber(src, out_dir / f"{src.stem}.png", "cutout")


def build_removebg_parser() -> argparse.ArgumentParser:
    from upscaler.background import BG_MODELS, DEFAULT_BG_MODEL

    p = argparse.ArgumentParser(
        prog="upscaler removebg",
        description="Remove the background — outputs a transparent PNG.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file or directory.")
    p.add_argument("-o", "--output", type=Path, help="Output .png file or directory.")
    p.add_argument(
        "-m", "--model", choices=sorted(BG_MODELS),
        help=f"Background model (default: {DEFAULT_BG_MODEL}).",
    )
    p.add_argument(
        "--feather", type=int, default=0,
        help="Soften the cut-out edge by N px (default 0).",
    )
    return p


def run_removebg(argv: list[str]) -> int:
    args = build_removebg_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    inputs = _gather_inputs(args.input)
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder",
              file=sys.stderr)
        return 2

    from upscaler.background import remove_background

    kw = {"model": args.model} if args.model else {}
    failed = 0
    for src in tqdm(inputs, disable=len(inputs) == 1, desc="removebg"):
        try:
            out = remove_background(Image.open(src), feather=args.feather, **kw)
            dst = _png_output_path(src, args.output)
            out.save(dst)
        except ImportError:
            print(f"error: {_ONNX_HINT}", file=sys.stderr)
            return 2
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0 if failed < len(inputs) else 2


def build_batch_parser() -> argparse.ArgumentParser:
    from upscaler.background import BG_MODELS

    p = argparse.ArgumentParser(
        prog="upscaler batch",
        description="Run one operation over many images, writing results to a folder. "
        "Unreadable files are skipped without aborting the batch.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Directory or a single image.")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--op", choices=("upscale", "convert", "removebg"), default="upscale",
        help="Operation to run on each image (default: upscale).",
    )
    # upscale options
    p.add_argument("-s", "--scale", type=int, default=4, choices=(2, 4))
    p.add_argument("-m", "--model", choices=sorted(MODELS))
    p.add_argument("--sharpen", nargs="?", type=float, const=1.0, default=0.0)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--tile", type=int, default=512)
    # convert options
    p.add_argument("-f", "--format", choices=list(FORMATS))
    p.add_argument("-q", "--quality", type=int, default=90)
    # removebg options
    p.add_argument("--bg-model", choices=sorted(BG_MODELS))
    p.add_argument("--feather", type=int, default=0)
    return p


def run_batch(argv: list[str]) -> int:
    args = build_batch_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.suffix:
        print("error: --output must be a directory", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input)
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.op == "upscale":
        up = Upscaler(model=args.model, scale=args.scale, device=args.device, tile=args.tile)

        def do(src: Path) -> None:
            res = up.upscale(Image.open(src))
            if args.sharpen > 0:
                res = unsharp_mask(res, strength=args.sharpen)
            res.save(out_dir / f"{src.stem}_x{up.scale}.png")
    elif args.op == "convert":
        fmt = args.format or "PNG"  # FORMATS keys are display names, not exts

        def do(src: Path) -> None:
            convert_file(src, out_dir / f"{src.stem}.{extension_for(fmt)}",
                         fmt=fmt, quality=args.quality)
    else:  # removebg
        from upscaler.background import remove_background
        kw = {"model": args.bg_model} if args.bg_model else {}

        def do(src: Path) -> None:
            out = remove_background(Image.open(src), feather=args.feather, **kw)
            out.save(out_dir / f"{src.stem}.png")

    failed = 0
    for src in tqdm(inputs, desc=f"batch {args.op}"):
        try:
            do(src)
        except ImportError:
            print(f"error: {_ONNX_HINT}", file=sys.stderr)
            return 2
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e} (skipped)", file=sys.stderr)
            failed += 1
    n_ok = len(inputs) - failed
    print(f"→ {out_dir}  ({n_ok}/{len(inputs)} ok)", file=sys.stderr)
    return 0 if n_ok > 0 else 2


def build_steam_parser() -> argparse.ArgumentParser:
    from upscaler import steam

    p = argparse.ArgumentParser(
        prog="upscaler steam",
        description="Cut an image, GIF or video into the five tiles of a Steam profile "
        "Workshop Showcase, at Steam's exact tile widths and gaps so the picture "
        "lines up across all five. Stills export five PNGs; clips export five "
        "looping APNGs (or GIFs with --gif) shrunk step by step to fit the upload "
        "cap (needs ffmpeg).",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image, GIF or video file.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output directory (default: <name>_steam/ next to the input).",
    )
    p.add_argument(
        "--still", action="store_true",
        help="Export still PNG tiles from the first frame, even for a clip.",
    )
    p.add_argument(
        "--gif", action="store_true",
        help="Animated tiles as GIF (256 colours, smaller) instead of full-colour APNG.",
    )
    p.add_argument(
        "--preset", choices=sorted(steam.PRESET_KEYS),
        help="Shape the row from the source's aspect ratio: auto (portrait → repeat, "
        "else whole), banner (square tiles, crop), whole (whole picture across the "
        "row, no crop), center (one tile in the middle, transparent around), repeat "
        "(the whole picture in every tile). Overrides --fit/--zoom/--pan/--height.",
    )
    p.add_argument(
        "--repeat", action="store_true",
        help="Fit the whole picture into one tile and repeat it in all five.",
    )
    p.add_argument(
        "--fit", choices=steam.FITS, default="cover",
        help="cover crops to fill, contain letterboxes, stretch distorts, manual "
        "uses --zoom (default: cover).",
    )
    p.add_argument("--zoom", type=float, default=1.0, help="Zoom for --fit manual (default 1.0).")
    p.add_argument(
        "--pan-x", type=float, default=0.0, metavar="PCT",
        help="Pan left/right, -100..100 (cover and manual fit).",
    )
    p.add_argument(
        "--pan-y", type=float, default=0.0, metavar="PCT",
        help="Pan up/down, -100..100 (cover and manual fit).",
    )
    p.add_argument(
        "--width", type=int, default=steam.DEFAULT_TILE_W, metavar="PX",
        help=f"Tile width in pixels, {steam.MIN_TILE_W}-{steam.MAX_TILE_W} (default "
        f"{steam.DEFAULT_TILE_W}, pixel-for-pixel at Steam's display size; 150 is the "
        "size most guides use). Gaps scale with it.",
    )
    p.add_argument(
        "--height", type=int, default=steam.DEFAULT_TILE_H, metavar="PX",
        help=f"Tile height in pixels, {steam.MIN_TILE_H}-{steam.MAX_TILE_H} "
        f"(default {steam.DEFAULT_TILE_H}: square tiles).",
    )
    p.add_argument(
        "--hidpi", action="store_true",
        help="Double the tile size (245px wide) so it stays crisp on Retina / HiDPI screens.",
    )
    p.add_argument(
        "--bg", default="#000000", metavar="HEX|transparent",
        help="Letterbox colour (default #000000), or 'transparent' to leave uncovered "
        "areas see-through so Steam's backdrop shows.",
    )
    p.add_argument(
        "--fps", type=int, default=24,
        help="Frame rate for animated tiles (default 24; the size budget may lower it).",
    )
    p.add_argument("--start", type=float, default=0.0, metavar="SEC", help="Trim: start time.")
    p.add_argument(
        "--end", type=float, default=0.0, metavar="SEC",
        help=f"Trim: end time (default: end of clip, capped at {steam.MAX_DURATION_SEC}s).",
    )
    p.add_argument(
        "--loop", choices=steam.LOOP_STYLES, default="normal",
        help="normal restarts, boomerang plays forward then back, crossfade blends the join.",
    )
    p.add_argument(
        "--max-mb", type=float, default=steam.DEFAULT_MAX_MB,
        help=f"Per-tile size budget in MB, 0 disables (default {steam.DEFAULT_MAX_MB:g}).",
    )
    p.add_argument(
        "--no-hexify", action="store_true",
        help="Leave files untouched. By default each tile's last byte is set to 0x21 "
        "(the 'hexify' step) so Steam keeps animations instead of flattening them.",
    )
    p.add_argument(
        "--how-to-upload", action="store_true",
        help="Print the Steam upload steps (browser-console trick) and exit.",
    )
    return p


def run_steam(argv: list[str]) -> int:
    from upscaler import panel, steam

    args = build_steam_parser().parse_args(argv)
    if args.how_to_upload:
        print(steam.UPLOAD_GUIDE)
        return 0
    if args.input is None or not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    out_dir = args.output or args.input.with_name(f"{args.input.stem}_steam")
    if out_dir.suffix and not out_dir.is_dir():
        print("error: --output must be a directory", file=sys.stderr)
        return 2

    mult = 2 if args.hidpi else 1
    p = steam.ShowcaseParams(
        fit=args.fit, zoom=args.zoom, off_x=args.pan_x, off_y=args.pan_y,
        bg_color=args.bg, tile_w=steam.HIDPI_TILE_W if args.hidpi else args.width,
        tile_h=args.height * mult, repeat=args.repeat,
    )
    if args.preset:
        first = panel._first_image(str(args.input))
        if first is None:
            print(f"error: couldn't decode {args.input}", file=sys.stderr)
            return 2
        p = steam.apply_preset(steam.PRESET_KEYS[args.preset], first.width, first.height, p)
    animated = not args.still and panel.media_kind(str(args.input)) == "animated"
    stem = f"{args.input.stem}_steam"
    try:
        if animated:
            last = [""]

            def progress(frac: float, desc: str = "") -> None:
                if desc and desc != last[0]:
                    last[0] = desc
                    print(f"  {desc}", file=sys.stderr)

            res = steam.export_animated(
                str(args.input), p, fps=args.fps, trim_start=args.start,
                trim_end=args.end, loop_mode=args.loop, max_mb=args.max_mb,
                out_dir=str(out_dir), stem=stem, fmt="gif" if args.gif else "apng",
                hexify_for_steam=not args.no_hexify, progress=progress,
            )
        else:
            res = steam.export_stills(str(args.input), p, out_dir=str(out_dir), stem=stem,
                                      hexify_for_steam=not args.no_hexify)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(steam.describe(res, args.max_mb), file=sys.stderr)
    for path in res.paths:
        print(path)
    print("Upload the tiles in order 1 → 5 — `upscaler steam --how-to-upload` has the steps.",
          file=sys.stderr)
    return 0 if res.fits else 1


def build_blur_parser() -> argparse.ArgumentParser:
    from upscaler import blur

    p = argparse.ArgumentParser(
        prog="upscaler blur",
        description="Blur a photo (no AI): gaussian, box, motion, spin, zoom, lens "
        "bokeh, pixelate or surface blur, over the whole image or through a "
        "rectangle / ellipse / band / painted mask with feathering.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder "
        "of images. Default: <name>_blur.png next to the input.",
    )
    p.add_argument("--kind", choices=blur.KINDS, default="gaussian", help="Blur type (default gaussian).")
    p.add_argument(
        "--strength", type=float, default=30.0,
        help="0-100, relative to the image's short side (100 = a radius of 10%% of it). Default 30.",
    )
    p.add_argument("--angle", type=float, default=0.0, help="motion: streak direction in degrees (0 = horizontal).")
    p.add_argument("--center", default="50,50", metavar="X,Y", help="spin/zoom: centre as %% of width,height (default 50,50).")
    p.add_argument("--highlights", type=float, default=0.0, help="lens: bokeh highlight bloom 0-100.")
    p.add_argument("--threshold", type=float, default=25.0, help="surface: edge protection 0-100 (default 25).")
    _add_region_args(p, verb="blur")
    p.add_argument("--no-progressive", action="store_true", help="Cross-fade one blur instead of ramping half → full through the feather.")
    p.add_argument("-q", "--quality", type=int, default=92, help="Quality for JPEG/WebP outputs (default 92).")
    return p


def _add_region_args(p: argparse.ArgumentParser, verb: str, feather: float = 10.0) -> None:
    """The region flags shared by `blur` and `adjust` — which part of the photo
    the effect lands on."""
    from upscaler import blur

    p.add_argument("--shape", choices=blur.SHAPES, default="whole",
                   help=f"Where to {verb} (default whole).")
    p.add_argument("--x", type=float, default=50.0, help="Shape centre X, %% of width.")
    p.add_argument("--y", type=float, default=50.0, help="Shape centre Y, %% of height.")
    p.add_argument("--w", type=float, default=50.0, help="Rectangle/ellipse width, %% of width.")
    p.add_argument("--h", type=float, default=50.0,
                   help="Rectangle/ellipse height or band thickness, %% of height.")
    p.add_argument("--mask-angle", type=float, default=0.0, help="band: tilt in degrees.")
    p.add_argument("--roundness", type=float, default=0.0, help="rectangle: corner rounding 0-100.")
    p.add_argument("--feather", type=float, default=feather,
                   help=f"Edge softness, %% of the short side (default {feather:g}).")
    p.add_argument("--outside", action="store_true",
                   help=f"{verb.capitalize()} outside the shape instead of inside it.")
    p.add_argument("--mask", type=Path,
                   help=f"Painted mask image (white = {verb}) for --shape painted.")
    p.add_argument("--face-pad", type=float, default=25.0, metavar="PCT",
                   help="--shape faces: grow each detected face's oval by this %% "
                   "(default 25, enough for hair and chin).")
    p.add_argument("--face-confidence", type=float, default=0.6, metavar="C",
                   help="--shape faces: detector confidence 0.05-0.99 (default 0.6; "
                   "lower finds more faces and more false positives).")


def _region_from_args(args) -> "tuple[object, int]":
    """(MaskParams, exit code) — the code is non-zero when --mask is missing."""
    from upscaler import blur

    painted = None
    if args.mask:
        if not args.mask.is_file():
            print(f"error: mask not found: {args.mask}", file=sys.stderr)
            return None, 2
        painted = Image.open(args.mask).convert("L")
    return blur.MaskParams(
        shape=args.shape, x=args.x, y=args.y, w=args.w, h=args.h, angle=args.mask_angle,
        roundness=args.roundness, feather=args.feather, outside=args.outside,
        progressive=not getattr(args, "no_progressive", True), painted=painted,
        face_pad=args.face_pad,
    ), 0


def _with_faces(mp, img, confidence: float, name: str):
    """Fill in a "faces" mask by detecting them in this image. Returns the mask
    unchanged for every other shape, and None when nothing was found (the
    caller skips the file rather than writing an identical copy)."""
    from upscaler import face

    if mp.shape != "faces":
        return mp
    found = face.detect_faces(img, confidence=confidence)
    if not found:
        print(f"{name}: no faces found (skipped)", file=sys.stderr)
        return None
    print(f"{name}: {len(found)} face(s) found", file=sys.stderr)
    return replace(mp, faces=[f.box() for f in found])


def _suffixed_output_path(src: Path, out: Path | None, suffix: str) -> Path:
    if out and out.suffix:
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{src.stem}_{suffix}.png"


def _blur_output_path(src: Path, out: Path | None) -> Path:
    return _suffixed_output_path(src, out, "blur")


def _adjust_output_path(src: Path, out: Path | None) -> Path:
    return _suffixed_output_path(src, out, "adjusted")


def run_blur(argv: list[str]) -> int:
    from upscaler import blur

    args = build_blur_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2
    try:
        cx, cy = (float(v) for v in args.center.split(","))
    except ValueError:
        print("error: --center must be X,Y (percent), e.g. 50,50", file=sys.stderr)
        return 2
    mp, code = _region_from_args(args)
    if code:
        return code
    bp = blur.BlurParams(kind=args.kind, strength=args.strength, angle=args.angle,
                         center_x=cx, center_y=cy, highlights=args.highlights,
                         threshold=args.threshold)
    failed = 0
    for src in inputs:
        dst = _blur_output_path(src, args.output)
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            region = _with_faces(mp, img, args.face_confidence, src.name)
            if region is None:
                continue
            out = blur.apply(img, bp, region)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {blur.describe(bp, region, img.size)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


_ADJUST_OPTS = [
    ("exposure", "Brightness, -100..100 (±2 stops)."),
    ("contrast", "Contrast, -100..100."),
    ("highlights", "Highlights, -100 (recover) .. 100 (lift)."),
    ("shadows", "Shadows, -100 (deepen) .. 100 (open up)."),
    ("black-point", "Where black starts, 0..50 (%% of range)."),
    ("white-point", "Where white starts, 50..100 (%% of range)."),
    ("gamma", "Midtone brightness, 0.2..3.0 (1 = unchanged)."),
    ("clarity", "Local contrast / punch, -100..100."),
    ("temperature", "White balance, -100 (cool) .. 100 (warm)."),
    ("tint", "White balance, -100 (green) .. 100 (magenta)."),
    ("hue", "Hue rotation in degrees, -180..180."),
    ("saturation", "Saturation, -100 (grey) .. 100."),
    ("vibrance", "Boosts muted colors only, -100..100."),
    ("tone-strength", "How strongly --tone is mixed into a black & white, 0..100."),
]


def build_adjust_parser() -> argparse.ArgumentParser:
    from upscaler import adjust

    p = argparse.ArgumentParser(
        prog="upscaler adjust",
        description="Color and light: exposure, contrast, highlights and shadows, white "
        "balance, vibrance, black & white — over the whole photo or through a shape, a "
        "graduated band or a painted mask. No AI.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder of "
        "images. Default: <name>_adjusted.png next to the input.",
    )
    p.add_argument("--preset", choices=list(adjust.PRESETS),
                   help="Start from a named look, then apply any flags on top.")
    p.add_argument("--auto", action="store_true",
                   help="Set levels, midtones and white balance from the photo itself.")
    for name, help_text in _ADJUST_OPTS:
        p.add_argument(f"--{name}", type=float, default=None, help=help_text)
    p.add_argument("--mono", action="store_true", help="Convert to black & white.")
    p.add_argument("--mono-mix", default=None, metavar="R,G,B",
                   help="Black & white channel mix, e.g. 30,59,11 (defaults to 30,59,11).")
    p.add_argument("--tone", default=None, metavar="HEX",
                   help="Tone color for a sepia / cyanotype look, used with --tone-strength.")
    _add_region_args(p, verb="adjust", feather=15.0)
    p.add_argument("-q", "--quality", type=int, default=92,
                   help="Quality for JPEG/WebP outputs (default 92).")
    return p


def run_adjust(argv: list[str]) -> int:
    from upscaler import adjust

    args = build_adjust_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2
    mp, code = _region_from_args(args)
    if code:
        return code

    base = adjust.preset(args.preset) if args.preset else adjust.AdjustParams()
    if args.mono:
        base.mono = True
    if args.mono_mix:
        try:
            base.mono_red, base.mono_green, base.mono_blue = (float(v) for v in args.mono_mix.split(","))
        except ValueError:
            print("error: --mono-mix must be R,G,B, e.g. 30,59,11", file=sys.stderr)
            return 2
    if args.tone:
        base.tone_color = args.tone

    failed = 0
    for src in inputs:
        dst = _adjust_output_path(src, args.output)
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            # Auto reads each photo, so a folder gets per-image levels; explicit
            # flags are applied last and always win.
            p = adjust.auto_params(img, base) if args.auto else replace(base)
            for name, _help in _ADJUST_OPTS:
                value = getattr(args, name.replace("-", "_"))
                if value is not None:
                    setattr(p, name.replace("-", "_"), value)
            region = _with_faces(mp, img, args.face_confidence, src.name)
            if region is None:
                continue
            out = adjust.apply(img, p, region)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {adjust.describe(p, region)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


# (flag, dataclass field, help) — floats unless the field is an int.
_EFFECT_OPTS = [
    ("grain", "Film grain, 0..100."),
    ("grain-size", "Grain coarseness, 1 (fine) .. 6 (clumpy)."),
    ("halation", "Glow bleeding out of highlights, 0..100."),
    ("halation-threshold", "How bright a pixel must be to glow, 0..99 (default 65)."),
    ("halation-radius", "How far the glow spreads, %% of the short side (default 2)."),
    ("leak", "A colored wash across the frame, 0..100."),
    ("leak-angle", "Which side the leak comes from, degrees (default 45)."),
    ("leak-softness", "1 (one edge) .. 100 (the whole frame), default 60."),
    ("vignette", "-100 (bright corners) .. 100 (dark corners)."),
    ("vignette-radius", "How much of the middle stays untouched, 0..99 (default 60)."),
    ("vignette-feather", "How gradually the vignette falls off, 1..100 (default 50)."),
    ("aberration", "Red/blue fringing toward the corners, 0..100."),
    ("duotone", "Recolor between two colors by brightness, 0..100."),
    ("dither", "Ordered dithering, 0..100."),
    ("halftone", "Dot screen, 0..100."),
    ("halftone-cell", "Dot size, %% of the short side (default 1)."),
    ("halftone-angle", "Screen angle in degrees (default 45)."),
    ("scanlines", "CRT scanlines, 0..100."),
    ("scanline-spacing", "Gap between scanlines (default 3)."),
    ("glitch", "Displaced bands and torn channels, 0..100."),
]
_EFFECT_INT_OPTS = [
    ("posterize", "Flatten to this many levels per channel, 2..32 (0 = off)."),
    ("dither-levels", "Colors per channel for --dither, 2..16 (default 4)."),
    ("glitch-seed", "Which random tear --glitch produces (default 7)."),
]
_EFFECT_COLOR_OPTS = [
    ("halation-color", "Glow color (default #ff5522)."),
    ("leak-color", "Light-leak color (default #ff8a3d)."),
    ("duotone-dark", "Duotone shadow color (default #1b2a4a)."),
    ("duotone-light", "Duotone highlight color (default #ffd9a0)."),
]


def build_effects_parser() -> argparse.ArgumentParser:
    from upscaler import effects

    p = argparse.ArgumentParser(
        prog="upscaler effects",
        description="Effects and film looks: grain, halation, light leaks, vignette, "
        "chromatic aberration, duotone, posterize, dither, halftone, scanlines and "
        "glitch. They stack, and 0 turns one off. No AI.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder of "
        "images. Default: <name>_fx.png next to the input.",
    )
    p.add_argument("--look", choices=list(effects.PRESETS),
                   help="Start from a ready-made look, then apply any flags on top.")
    for name, help_text in _EFFECT_OPTS:
        p.add_argument(f"--{name}", type=float, default=None, help=help_text)
    for name, help_text in _EFFECT_INT_OPTS:
        p.add_argument(f"--{name}", type=int, default=None, help=help_text)
    for name, help_text in _EFFECT_COLOR_OPTS:
        p.add_argument(f"--{name}", default=None, metavar="HEX", help=help_text)
    _add_region_args(p, verb="affect", feather=15.0)
    p.add_argument("-q", "--quality", type=int, default=92,
                   help="Quality for JPEG/WebP outputs (default 92).")
    return p


def run_effects(argv: list[str]) -> int:
    from upscaler import effects

    args = build_effects_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2
    mp, code = _region_from_args(args)
    if code:
        return code

    p = effects.preset(args.look) if args.look else effects.EffectParams()
    for name, _help in _EFFECT_OPTS + _EFFECT_INT_OPTS + _EFFECT_COLOR_OPTS:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            setattr(p, name.replace("-", "_"), value)

    failed = 0
    for src in inputs:
        dst = _suffixed_output_path(src, args.output, "fx")
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            region = _with_faces(mp, img, args.face_confidence, src.name)
            if region is None:
                continue
            out = effects.apply(img, p, region)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {effects.describe(p, region)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def build_sharpen_parser() -> argparse.ArgumentParser:
    from upscaler import sharpen as st

    p = argparse.ArgumentParser(
        prog="upscaler sharpen",
        description="Sharpen a photo: unsharp mask, high-pass overlay, edge-aware "
        "(leaves skin and sky alone) or two-scale texture, with halo control. The "
        "radius is in pixels, because that is the scale real detail lives at. No AI.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder of "
        "images. Default: <name>_sharp.png next to the input.",
    )
    p.add_argument("--preset", choices=list(st.PRESETS),
                   help="Start from a preset, then apply any flags on top.")
    p.add_argument("--kind", choices=st.KINDS, default=None,
                   help="Sharpening method (default unsharp).")
    p.add_argument("--amount", type=float, default=None,
                   help="How hard to push, 0..300 (100 = a classic full-strength unsharp).")
    p.add_argument("--radius", type=float, default=None,
                   help=f"Edge width in pixels, {st.MIN_RADIUS}..{st.MAX_RADIUS} (default 1).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Leave flat areas alone, 0..100 (default 4).")
    p.add_argument("--halo", type=float, default=None,
                   help="Cap the bright/dark rim an edge may gain, 0..100 (default 35).")
    p.add_argument("--all-channels", action="store_true",
                   help="Sharpen every color channel instead of brightness only "
                   "(brightness-only avoids colored fringes).")
    p.add_argument("--protect-shadows", type=float, default=None, help="Hold back in the darks, 0..100.")
    p.add_argument("--protect-highlights", type=float, default=None, help="Hold back in the brights, 0..100.")
    p.add_argument("--edge-sensitivity", type=float, default=None,
                   help="--kind smart: how strictly it follows edges, 0..100 (default 50).")
    p.add_argument("--detail-balance", type=float, default=None,
                   help="--kind texture: fine (0) … structure (100), default 50.")
    _add_region_args(p, verb="sharpen", feather=15.0)
    p.add_argument("-q", "--quality", type=int, default=92,
                   help="Quality for JPEG/WebP outputs (default 92).")
    return p


def run_sharpen(argv: list[str]) -> int:
    from upscaler import sharpen as st

    args = build_sharpen_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2
    mp, code = _region_from_args(args)
    if code:
        return code

    p = st.preset(args.preset) if args.preset else st.SharpenParams()
    for name in ("kind", "amount", "radius", "threshold", "halo", "protect_shadows",
                 "protect_highlights", "edge_sensitivity", "detail_balance"):
        value = getattr(args, name)
        if value is not None:
            setattr(p, name, value)
    if args.all_channels:
        p.luminance_only = False

    failed = 0
    for src in inputs:
        dst = _suffixed_output_path(src, args.output, "sharp")
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            region = _with_faces(mp, img, args.face_confidence, src.name)
            if region is None:
                continue
            out = st.apply(img, p, region)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {st.describe(p, region, img.size)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError, RuntimeError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def build_frame_parser() -> argparse.ArgumentParser:
    from upscaler import frame

    p = argparse.ArgumentParser(
        prog="upscaler crop",
        description="Crop, straighten and frame: any aspect ratio, tilt correction, "
        "leaning-vertical correction, an exact output size, borders, rounded corners "
        "and a drop shadow. No AI.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder of "
        "images. Default: <name>_framed.png next to the input.",
    )
    p.add_argument("--preset", choices=list(frame.PRESETS),
                   help="Start from a preset, then apply any flags on top.")
    p.add_argument("--aspect", default=None,
                   help='Shape: a named one (e.g. "Square · 1:1") or a ratio like 16:9, '
                   "4/5, 1200x800.")
    p.add_argument("--mode", choices=frame.CROP_MODES, default=None,
                   help="fill crops to the shape (default); fit keeps the whole photo "
                   "and fills the margin.")
    p.add_argument("--position", default=None, metavar="X,Y",
                   help="Which part survives the crop, as percentages (default 50,50).")
    p.add_argument("--zoom", type=float, default=None, help=f"Crop in tighter, 1..{frame.MAX_ZOOM}.")
    p.add_argument("--straighten", type=float, default=None, metavar="DEG",
                   help=f"Level a tilted horizon, ±{frame.MAX_STRAIGHTEN}°.")
    p.add_argument("--rotate", type=int, choices=frame.ROTATIONS, default=None,
                   help="Quarter turns, clockwise.")
    p.add_argument("--flip-h", action="store_true", help="Mirror left to right.")
    p.add_argument("--flip-v", action="store_true", help="Flip top to bottom.")
    p.add_argument("--lean-h", type=float, default=None, metavar="N",
                   help="Correct converging horizontals, -100..100.")
    p.add_argument("--lean-v", type=float, default=None, metavar="N",
                   help="Correct converging verticals, -100..100.")
    p.add_argument("--size", default=None, metavar="WxH",
                   help="Land on an exact pixel size, e.g. 1920x1080.")
    p.add_argument("--border", type=float, default=None, metavar="PCT",
                   help="Margin around the photo, %% of its short side.")
    p.add_argument("--border-style", choices=frame.BORDER_STYLES, default=None,
                   help="solid color, or a zoomed blurred copy of the photo.")
    p.add_argument("--border-color", default=None, metavar="HEX", help="Border color (default #ffffff).")
    p.add_argument("--border-blur", type=float, default=None, help="How soft the blurred fill is, 0..100.")
    p.add_argument("--radius", type=float, default=None, metavar="PCT",
                   help="Rounded corners, %% of the short side.")
    p.add_argument("--shadow", type=float, default=None, help="Drop shadow 0..100 (needs a border).")
    p.add_argument("-q", "--quality", type=int, default=92,
                   help="Quality for JPEG/WebP outputs (default 92).")
    return p


def run_frame(argv: list[str]) -> int:
    from upscaler import frame

    args = build_frame_parser().parse_args(argv)
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2

    p = frame.preset(args.preset) if args.preset else frame.FrameParams()
    if args.aspect is not None:
        # A named shape, otherwise treat it as a custom ratio.
        if args.aspect in frame.ASPECTS:
            p.aspect = args.aspect
        elif frame.parse_aspect(args.aspect):
            p.aspect, p.custom_aspect = frame.CUSTOM_ASPECT, args.aspect
        else:
            print(f"error: can't read the ratio {args.aspect!r} — try 16:9 or 1200x800",
                  file=sys.stderr)
            return 2
    if args.position is not None:
        try:
            p.position_x, p.position_y = (float(v) for v in args.position.split(","))
        except ValueError:
            print("error: --position must be X,Y percentages, e.g. 50,30", file=sys.stderr)
            return 2
    for flag, field in (("mode", "crop_mode"), ("zoom", "zoom"), ("straighten", "straighten"),
                        ("rotate", "rotate"), ("lean_h", "keystone_h"), ("lean_v", "keystone_v"),
                        ("size", "out_size"), ("border", "border"),
                        ("border_style", "border_style"), ("border_color", "border_color"),
                        ("border_blur", "border_blur"), ("radius", "corner_radius"),
                        ("shadow", "shadow")):
        value = getattr(args, flag)
        if value is not None:
            setattr(p, field, value)
    if args.flip_h:
        p.flip_h = True
    if args.flip_v:
        p.flip_v = True
    if p.out_size and not fit.parse_target(p.out_size):
        print(f"error: can't read the size {p.out_size!r} — try 1920x1080", file=sys.stderr)
        return 2

    failed = 0
    for src in inputs:
        dst = _suffixed_output_path(src, args.output, "framed")
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            out = frame.apply(img, p)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {frame.describe(p, img.size)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def build_watermark_parser() -> argparse.ArgumentParser:
    from upscaler import watermark as wm

    p = argparse.ArgumentParser(
        prog="upscaler watermark",
        description="Stamp a signature, caption or logo onto a photo — in a corner or "
        "tiled across the whole frame. Sizes are a share of the photo, so one setting "
        "suits a whole folder of mixed pictures. No AI.",
    )
    p.add_argument("input", type=Path, nargs="?", help="Image file, or a directory of images.")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output file (format from the extension) or a directory for a folder of "
        "images. Default: <name>_wm.png next to the input.",
    )
    p.add_argument("--preset", choices=list(wm.PRESETS),
                   help="Start from a preset, then apply any flags on top.")
    p.add_argument("--text", default=None, help='The text to stamp (default "© Your Name").')
    p.add_argument("--logo", type=Path, default=None,
                   help="Stamp this image instead of text (a transparent PNG works best).")
    p.add_argument("--font", default=None, help="Font name; see --list-fonts.")
    p.add_argument("--list-fonts", action="store_true", help="List the available fonts and exit.")
    p.add_argument("--size", type=float, default=None,
                   help="Text size as %% of the photo's short side (default 4).")
    p.add_argument("--logo-size", type=float, default=None,
                   help="Logo width as %% of the photo's width (default 18).")
    p.add_argument("--color", default=None, metavar="HEX", help="Text color (default #ffffff).")
    p.add_argument("--outline", default=None, metavar="HEX", help="Outline color (default #000000).")
    p.add_argument("--outline-width", type=float, default=None,
                   help="Outline thickness as %% of the text size (default 8; 0 = none).")
    p.add_argument("--shadow", type=float, default=None, help="Drop shadow 0..100 (default 45).")
    p.add_argument("--position", choices=wm.POSITIONS, default=None,
                   help=f"Where it sits (default {wm.DEFAULT_POSITION}), or 'tiled'.")
    p.add_argument("--margin", type=float, default=None,
                   help="Distance from the edge, %% of the short side (default 3).")
    p.add_argument("--opacity", type=float, default=None, help="0..100 (default 70).")
    p.add_argument("--rotation", type=float, default=None, metavar="DEG", help="Tilt the mark.")
    p.add_argument("--tile-gap", type=float, default=None,
                   help="--position tiled: gap between repeats, %% of the short side.")
    p.add_argument("--tile-angle", type=float, default=None,
                   help="--position tiled: angle of the pattern in degrees.")
    p.add_argument("-q", "--quality", type=int, default=92,
                   help="Quality for JPEG/WebP outputs (default 92).")
    return p


def run_watermark(argv: list[str]) -> int:
    from upscaler import watermark as wm

    args = build_watermark_parser().parse_args(argv)
    if args.list_fonts:
        for name in wm.FONT_NAMES:
            print(name)
        return 0
    if args.input is None or not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    inputs = _gather_inputs(args.input) if args.input.is_dir() else [args.input]
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output and args.output.suffix:
        print("error: --output must be a directory when processing a folder", file=sys.stderr)
        return 2

    logo = None
    if args.logo:
        if not args.logo.is_file():
            print(f"error: logo not found: {args.logo}", file=sys.stderr)
            return 2
        logo = Image.open(args.logo).convert("RGBA")

    p = wm.preset(args.preset) if args.preset else wm.WatermarkParams()
    if args.logo and args.preset is None:
        p.kind = "logo"          # --logo alone is enough to mean "stamp this"
    for flag, field in (("text", "text"), ("font", "font"), ("size", "size"),
                        ("logo_size", "logo_scale"), ("color", "color"),
                        ("outline", "outline"), ("outline_width", "outline_width"),
                        ("shadow", "shadow"), ("position", "position"),
                        ("margin", "margin"), ("opacity", "opacity"),
                        ("rotation", "rotation"), ("tile_gap", "tile_gap"),
                        ("tile_angle", "tile_angle")):
        value = getattr(args, flag)
        if value is not None:
            setattr(p, field, value)
    if args.text is not None:
        p.kind = "text"
    if p.kind == "logo" and logo is None:
        print("error: --logo is required for a logo watermark", file=sys.stderr)
        return 2
    if args.font and args.font not in wm.FONTS:
        print(f"error: unknown font {args.font!r} — see --list-fonts", file=sys.stderr)
        return 2

    failed = 0
    for src in inputs:
        dst = _suffixed_output_path(src, args.output, "wm")
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            out = wm.apply(img, p, logo)
            if dst.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
                out = out.convert("RGB")
            save_kw = {"quality": args.quality} if dst.suffix.lower() in (".jpg", ".jpeg", ".webp") else {}
            out.save(dst, **save_kw)
            print(f"{src.name}: {wm.describe(p)} → {dst}", file=sys.stderr)
        except (Image.UnidentifiedImageError, OSError, ValueError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler",
        description="Local image upscaling + sharpening (pretrained Real-ESRGAN).",
        epilog="Subcommands: `upscaler convert <input> -o out.webp` (format), "
        "`upscaler pdf build *.png -o out.pdf` / `upscaler pdf extract in.pdf`, "
        "`upscaler video in.mp4 -o out.mp4`, "
        "`upscaler removebg <input> -o out.png`, "
        "`upscaler batch <dir> -o <dir> --op upscale|convert|removebg`, "
        "`upscaler steam clip.mp4 -o <dir>` (Steam Workshop Showcase tiles), "
        "`upscaler blur photo.jpg --kind lens --shape ellipse --outside` (blur toolbox), "
        "`upscaler adjust photo.jpg --auto` (color and light), "
        "`upscaler effects photo.jpg --look \"Film grain\"` (effects and film looks), "
        "`upscaler sharpen photo.jpg --preset Standard` (sharpening toolbox), "
        "`upscaler crop photo.jpg --aspect 1:1 --border 6` (crop and frame), "
        "`upscaler watermark ./folder --text \"© Me\"` (signature or logo). "
        "Add --face to restore faces after upscaling.",
    )
    p.add_argument(
        "input", type=Path, nargs="?", help="Image file or a directory of images."
    )
    p.add_argument("-o", "--output", type=Path, help="Output file or directory.")
    p.add_argument(
        "-s", "--scale", type=int, default=4, choices=(2, 4),
        help="Upscale factor (default: 4). Ignored if --model is given.",
    )
    p.add_argument(
        "-m", "--model", choices=sorted(MODELS), help="Explicit model (overrides --scale)."
    )
    p.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device (default: auto).",
    )
    p.add_argument(
        "--deblur", action="store_true",
        help="Deblur with NAFNet before upscaling (good for motion blur).",
    )
    p.add_argument(
        "--deblur-model", choices=sorted(DEBLUR_MODELS),
        help=f"NAFNet deblur model (default: {DEFAULT_DEBLUR_MODEL}).",
    )
    p.add_argument(
        "--sharpen", nargs="?", type=float, const=1.0, default=0.0,
        help="Apply unsharp mask after upscaling. Optional strength (default 1.0).",
    )
    p.add_argument("--tile", type=int, default=512, help="Tile size, 0 disables tiling.")
    p.add_argument("--fp16", action="store_true", help="Half precision (CUDA only).")
    p.add_argument(
        "--onnx", action="store_true",
        help="Use the ONNX Runtime backend (exports once, then torch-free).",
    )
    p.add_argument(
        "--face", action="store_true",
        help="Restore faces after upscaling (GFPGAN; needs the [face] extra).",
    )
    p.add_argument(
        "--face-strength", type=float, default=0.8,
        help="Face restoration strength 0..1 (default 0.8).",
    )
    p.add_argument("--list-models", action="store_true", help="List models and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    # Piped/redirected output on Windows defaults to cp1252, which can't encode
    # the arrows/emoji in our own status lines (or torch's) — degrade to '?'
    # instead of dying with UnicodeEncodeError.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    # Optional subcommands; bare `upscaler <input>` stays the upscaler. Dispatch
    # is by argv[0] string (like a real filename can't be 'convert'/'batch'/…).
    if argv and argv[0] == "convert":
        return run_convert(argv[1:])
    if argv and argv[0] == "pdf":
        return run_pdf(argv[1:])
    if argv and argv[0] == "video":
        return run_video(argv[1:])
    if argv and argv[0] == "removebg":
        return run_removebg(argv[1:])
    if argv and argv[0] == "batch":
        return run_batch(argv[1:])
    if argv and argv[0] == "steam":
        return run_steam(argv[1:])
    if argv and argv[0] == "blur":
        return run_blur(argv[1:])
    if argv and argv[0] == "adjust":
        return run_adjust(argv[1:])
    if argv and argv[0] == "effects":
        return run_effects(argv[1:])
    if argv and argv[0] == "sharpen":
        return run_sharpen(argv[1:])
    if argv and argv[0] == "crop":
        return run_frame(argv[1:])
    if argv and argv[0] == "watermark":
        return run_watermark(argv[1:])

    args = build_parser().parse_args(argv)

    if args.list_models:
        print("Upscale models:")
        for name, spec in sorted(MODELS.items()):
            print(f"  {name:24s} ×{spec.scale}  {spec.notes}")
        print("Deblur models (--deblur-model):")
        for name, spec in sorted(DEBLUR_MODELS.items()):
            print(f"  {name:24s}      {spec.notes}")
        return 0

    if args.input is None:
        print("error: input is required (or use --list-models)", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    inputs = _gather_inputs(args.input)
    if not inputs:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2
    if len(inputs) > 1 and args.output is not None and args.output.suffix:
        print("error: --output must be a directory when processing a folder",
              file=sys.stderr)
        return 2

    if args.onnx:
        from upscaler.onnx_engine import OnnxDeblurrer, OnnxUpscaler

        deblurrer = (
            OnnxDeblurrer(model=args.deblur_model, device=args.device)
            if args.deblur else None
        )
        up = OnnxUpscaler(
            model=args.model, scale=args.scale, device=args.device, tile=args.tile
        )
        backend = "onnx"
    else:
        deblurrer = (
            Deblurrer(model=args.deblur_model, device=args.device) if args.deblur else None
        )
        up = Upscaler(
            model=args.model, scale=args.scale, device=args.device,
            tile=args.tile, fp16=args.fp16,
        )
        backend = up.device.type

    # Face restoration is lazy: only imported (and only pulls the [face] extra)
    # when --face is given, so a plain torch-only install is unaffected.
    face_restorer = None
    if args.face:
        try:
            from upscaler.face import FaceRestorer
            face_restorer = FaceRestorer(device=args.device)
        except (ImportError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    stages = (f"deblur={deblurrer.spec.name} " if deblurrer else "") + (
        f"upscale={up.spec.name} ×{up.scale}"
    ) + (" face=gfpgan" if face_restorer else "")
    print(f"{stages} backend={backend}", file=sys.stderr)

    failed = 0
    for src in tqdm(inputs, disable=len(inputs) == 1, desc="images"):
        try:
            img = Image.open(src)
            if deblurrer:
                img = deblurrer.deblur(img)
            result = up.upscale(img)
            if face_restorer is not None:
                result = face_restorer.restore(result, args.face_strength)
            if args.sharpen > 0:
                result = unsharp_mask(result, strength=args.sharpen)
            dst = _output_path(src, args.output, up.scale)
            result.save(dst)
        # RuntimeError/ValueError cover weight-download and inference failures —
        # a folder batch should report and continue, not dump a traceback.
        except (Image.UnidentifiedImageError, OSError, RuntimeError, ValueError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0 if failed < len(inputs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
