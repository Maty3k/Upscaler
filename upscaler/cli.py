"""Command-line interface: ``upscaler <input> [-o out] [--scale N] [--sharpen ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscaler",
        description="Local image upscaling + sharpening (pretrained Real-ESRGAN).",
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
    p.add_argument("--list-models", action="store_true", help="List models and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
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

    deblurrer = (
        Deblurrer(model=args.deblur_model, device=args.device) if args.deblur else None
    )
    up = Upscaler(
        model=args.model, scale=args.scale, device=args.device,
        tile=args.tile, fp16=args.fp16,
    )
    stages = (f"deblur={deblurrer.spec.name} " if deblurrer else "") + (
        f"upscale={up.spec.name} x{up.scale}"
    )
    print(f"{stages} device={up.device.type}", file=sys.stderr)

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
