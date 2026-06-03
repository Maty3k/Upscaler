"""Command-line interface: ``upscaler <input> [-o out] [--scale N] [--sharpen ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from upscaler.convert import FORMATS, convert_file, extension_for
from upscaler.deblur import Deblurrer
from upscaler.engine import Upscaler
from upscaler.models.registry import DEBLUR_MODELS, DEFAULT_DEBLUR_MODEL, MODELS
from upscaler.sharpen import unsharp_mask

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


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

    for src in tqdm(inputs, disable=len(inputs) == 1, desc="convert"):
        dst = _convert_output_path(src, args.output, fmt)
        convert_file(src, dst, fmt=fmt, quality=args.quality, lossless=args.lossless)
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler",
        description="Local image upscaling + sharpening (pretrained Real-ESRGAN).",
        epilog="Subcommand: `upscaler convert <input> -o out.webp` — format conversion.",
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
    # Optional `convert` subcommand; bare `upscaler <input>` stays the upscaler.
    if argv and argv[0] == "convert":
        return run_convert(argv[1:])

    args = build_parser().parse_args(argv)

    if args.list_models:
        print("Upscale models:")
        for name, spec in sorted(MODELS.items()):
            print(f"  {name:24s} x{spec.scale}  {spec.notes}")
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
        f"upscale={up.spec.name} x{up.scale}"
    )
    print(f"{stages} backend={backend}", file=sys.stderr)

    for src in tqdm(inputs, disable=len(inputs) == 1, desc="images"):
        img = Image.open(src)
        if deblurrer:
            img = deblurrer.deblur(img)
        result = up.upscale(img)
        if args.sharpen > 0:
            result = unsharp_mask(result, strength=args.sharpen)
        dst = _output_path(src, args.output, up.scale)
        result.save(dst)
        if len(inputs) == 1:
            print(f"→ {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
