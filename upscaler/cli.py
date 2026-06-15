"""Command-line interface: ``upscaler <input> [-o out] [--scale N] [--sharpen ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from upscaler.convert import FORMATS, convert_file, extension_for
from upscaler.document import images_to_pdf, pdf_to_images
from upscaler.video import upscale_video
from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler
from upscaler.models.registry import DEBLUR_MODELS, DEFAULT_DEBLUR_MODEL, MODELS
from upscaler.sharpen import unsharp_mask

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


def _gather_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return [path]


def _output_path(src: Path, out: Path | None, scale: int) -> Path:
    if out and out.suffix:  # explicit file target
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{src.stem}_x{scale}.png"


def _convert_output_path(src: Path, out: Path | None, fmt: str) -> Path:
    ext = extension_for(fmt)
    if out and out.suffix:  # explicit file target
        return out
    out_dir = out if out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{src.stem}.{ext}"


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


def _format_from_output(out: Path | None) -> str | None:
    if out and out.suffix:
        ext = out.suffix.lstrip(".").lower()
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
        data = images_to_pdf(images)
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
    up = Upscaler(
        model=args.model, scale=args.scale, device=args.device, tile=args.tile
    )
    print(f"upscaling {len(inputs)} clip(s) ×{up.scale} on {up.device.type} "
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler",
        description="Local image upscaling + sharpening (pretrained Real-ESRGAN).",
        epilog="Subcommands: `upscaler convert <input> -o out.webp` (format), "
        "`upscaler pdf build *.png -o out.pdf` / `upscaler pdf extract in.pdf`, "
        "`upscaler video in.mp4 -o out.mp4`.",
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
    p.add_argument("--list-models", action="store_true", help="List models and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Optional subcommands; bare `upscaler <input>` stays the upscaler.
    if argv and argv[0] == "convert":
        return run_convert(argv[1:])
    if argv and argv[0] == "pdf":
        return run_pdf(argv[1:])
    if argv and argv[0] == "video":
        return run_video(argv[1:])

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

    stages = (f"deblur={deblurrer.spec.name} " if deblurrer else "") + (
        f"upscale={up.spec.name} ×{up.scale}"
    )
    print(f"{stages} backend={backend}", file=sys.stderr)

    failed = 0
    for src in tqdm(inputs, disable=len(inputs) == 1, desc="images"):
        try:
            img = Image.open(src)
            if deblurrer:
                img = deblurrer.deblur(img)
            result = up.upscale(img)
            if args.sharpen > 0:
                result = unsharp_mask(result, strength=args.sharpen)
            dst = _output_path(src, args.output, up.scale)
            result.save(dst)
        except (Image.UnidentifiedImageError, OSError) as e:
            print(f"error on {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0 if failed < len(inputs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
