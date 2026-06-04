# Upscaler

Local, open-source image **upscaling + sharpening**. Runs entirely on your
machine (CPU, NVIDIA CUDA, or Apple-Silicon MPS) on top of pretrained
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) weights — no cloud, no API keys.

> Upscaling, deblur, a Gradio GUI, and an ONNX backend all work end-to-end.
> See [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md) for full planning, design
> decisions, the "why it can make photos worse" lesson, and the
> train-your-own-model playbook (incl. AMD/Windows/ROCm).

## Install

Python **3.9–3.12** recommended (PyTorch wheels).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # extras: ".[gui]" (GUI), ".[onnx]" (ONNX backend), ".[dev]" (tests)
```

## Usage

### CLI

```bash
# 4x upscale (default model)
upscaler photo.jpg -o photo_4x.png

# 2x, and apply a sharpening pass afterwards
upscaler photo.jpg --scale 2 --sharpen

# deblur motion blur (NAFNet) before upscaling
upscaler blurry.jpg --deblur --scale 4

# stronger sharpen, explicit device
upscaler photo.jpg --sharpen 1.5 --device mps

# batch a whole folder
upscaler ./input_dir -o ./output_dir --scale 4

# anime / illustration model
upscaler art.png --model realesrgan-x4plus-anime

# upscale a video frame-by-frame (offline, keeps audio; needs ffmpeg)
upscaler video clip.mp4 -o clip_2x.mp4 --scale 2

# ONNX Runtime backend (exports once from the .pth, then torch-free + often
# faster on CPU). Works with --deblur and batching too.
upscaler photo.jpg --scale 4 --onnx

upscaler --list-models
```

#### Convert formats (no AI)

```bash
upscaler convert photo.png -o photo.webp        # format from extension
upscaler convert photo.png -f JPEG -q 80        # explicit format + quality
upscaler convert photo.png -o out.webp --lossless
upscaler convert ./folder -o ./out -f WebP      # batch a directory
```

Supports PNG / JPEG / WebP / AVIF / TIFF / GIF / BMP / ICO / TGA / PPM. Alpha is
flattened onto a white background for formats that can't store it (JPEG/BMP/PPM).

### Video (frame-by-frame)

```bash
upscaler video clip.mp4 -o clip_2x.mp4 --scale 2          # keeps audio
upscaler video clip.mp4 -o clip_2x_60.mp4 --scale 2 --fps 60   # + smooth to 60fps
upscaler video clip.mp4 -o clip_4k.mp4 --scale 4 --size 3840    # fit longest edge to 4K
upscaler video ./clips -o ./out --scale 2                 # batch a whole folder
```

Offline frame-by-frame upscaling (split → upscale each frame → re-encode + mux
audio). `--fps` adds motion-interpolated frames (ffmpeg `minterpolate`) for
smoother motion — duration unchanged, audio stays in sync, but it's slow. Needs
**ffmpeg** (system install, or `pip install -e ".[video]"` for a bundled binary). It's a render-and-wait feature — minutes per minute of footage —
and since frames are upscaled independently, very fine detail can shimmer slightly
between frames (a temporal model would be needed to fully remove that).

#### Image ⇄ PDF

```bash
upscaler pdf build a.png b.png c.png -o out.pdf   # images → multi-page PDF
upscaler pdf build ./folder -o out.pdf            # all images in a directory
upscaler pdf extract in.pdf -o ./pages --dpi 200  # PDF pages → PNGs
upscaler pdf extract in.pdf                        # → ./in_pages/ next to the PDF
```

Weights download automatically on first use and are cached under
`upscaler/weights/` (override with `UPSCALER_WEIGHTS_DIR`).

### GUI (drag-and-drop)

```bash
pip install -e ".[gui]"
python app.py            # opens a local web UI at http://127.0.0.1:7860
```

A full-width local web app with three tools: **File Converter** (PNG/JPEG/WebP/
BMP/TIFF), **Image ⇄ PDF** (combine images into a PDF, or extract a PDF's pages
to PNGs), and **Upscale & Enhance**. Runs entirely on your machine — nothing is
uploaded anywhere. (PDF support uses `pypdfium2`, included in the `.[gui]` extra
or installable on its own via `.[pdf]`.)

### Library

```python
from PIL import Image
from upscaler import Upscaler, enhance

# reuse one loaded model across many images
up = Upscaler(scale=4, device="auto")
up.upscale_file("in.jpg", "out.png")

# one-shot upscale + sharpen
result = enhance(Image.open("in.jpg"), scale=2, sharpen=1.0)
result.save("out.png")
```

## How it works

- `upscaler/models/rrdbnet.py` — the RRDBNet generator, vendored so we don't
  depend on the fragile `basicsr`/`realesrgan` stack. Layer names match the
  official checkpoints, which load with `strict=True`.
- `upscaler/models/registry.py` + `weights.py` — model registry and lazy,
  integrity-checked weight download.
- `upscaler/engine.py` — device selection and **tiled inference** (large images
  are processed in padded tiles to bound memory and avoid seams).
- `upscaler/models/nafnet.py` + `deblur.py` — vendored **NAFNet** and the deblur
  stage. NAFNet's channel attention pools globally, so it runs on the whole image
  (not tiled) and is applied at native resolution before upscaling.
- `upscaler/pipeline.py` + `sharpen.py` — `enhance()`: optional deblur → upscale
  → optional unsharp mask.
- `upscaler/onnx_export.py` + `onnx_engine.py` — export each model to ONNX with
  dynamic shapes (one-time, needs torch) and run it via ONNX Runtime. The engines
  import only `onnxruntime`/`numpy`/`Pillow`, so cached `.onnx` files run
  torch-free. Verified to match the torch output (≤1/255 per pixel).

## Performance notes

- **CPU works** but is slow on large images; keep `--tile` at 512 or lower.
- **Apple Silicon:** `--device mps` is much faster than CPU.
- **CUDA:** add `--fp16` for a speed/memory win.

## Testing

```bash
pip install -e ".[dev]"
pytest        # architecture + tiling tests; run on CPU, no weights download
```

## Roadmap

- [x] Phase 0 — scaffold, packaging, license
- [x] Phase 1 — Real-ESRGAN upscaling (lib + CLI), tiling, lazy weights, unsharp sharpen
- [x] Phase 2 — model-based deblur stage (NAFNet) for genuinely blurry input
- [x] Phase 3 — Gradio drag-and-drop GUI (`app.py`)
- [x] Phase 4 — ONNX Runtime path for faster, PyTorch-free CPU inference (`--onnx`)

## Licensing

This project is **Apache-2.0** (see `LICENSE`). The pretrained weights are from
the official Real-ESRGAN releases and carry their own (BSD-3-Clause) terms; they
are downloaded at runtime and never redistributed in this repo. Credit to
Xintao Wang et al. for Real-ESRGAN and to BasicSR for the RRDBNet architecture,
and to Chen et al. / megvii-research for NAFNet (MIT). NAFNet deblur weights are
mirrored on Hugging Face (`nyanko7/nafnet-models`); the upstream originals are on
the official NAFNet Google Drive.
