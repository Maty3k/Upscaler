# Upscaler

Local, open-source image **upscaling + sharpening**. Runs entirely on your
machine (CPU, NVIDIA CUDA, or Apple-Silicon MPS) on top of pretrained
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) weights — no cloud, no API keys.

> Status: **Phase 1** — core upscaling works end-to-end via library + CLI.
> A model-based deblur stage and a drag-and-drop GUI are planned (see [Roadmap](#roadmap)).

## Install

Python **3.9–3.12** recommended (PyTorch wheels).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for tests, ".[gui]" for the planned GUI
```

## Usage

### CLI

```bash
# 4x upscale (default model)
upscaler photo.jpg -o photo_4x.png

# 2x, and apply a sharpening pass afterwards
upscaler photo.jpg --scale 2 --sharpen

# stronger sharpen, explicit device
upscaler photo.jpg --sharpen 1.5 --device mps

# batch a whole folder
upscaler ./input_dir -o ./output_dir --scale 4

# anime / illustration model
upscaler art.png --model realesrgan-x4plus-anime

upscaler --list-models
```

Weights download automatically on first use and are cached under
`upscaler/weights/` (override with `UPSCALER_WEIGHTS_DIR`).

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
- `upscaler/pipeline.py` + `sharpen.py` — upscale, then optional unsharp mask.

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
- [ ] Phase 2 — model-based deblur stage (NAFNet) for genuinely blurry input
- [ ] Phase 3 — Gradio drag-and-drop GUI (`app.py`)
- [ ] Phase 4 — ONNX Runtime path for faster, PyTorch-free CPU inference

## Licensing

This project is **Apache-2.0** (see `LICENSE`). The pretrained weights are from
the official Real-ESRGAN releases and carry their own (BSD-3-Clause) terms; they
are downloaded at runtime and never redistributed in this repo. Credit to
Xintao Wang et al. for Real-ESRGAN and to BasicSR for the RRDBNet architecture.
