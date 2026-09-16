# Upscaler

**Free, private AI photo enhancement on your own computer.** Upscale, sharpen,
deblur, colorize, remove objects and backgrounds, upscale videos — the same
kind of results the paid cloud upscalers charge a subscription for, running
100% locally on top of pretrained
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) weights.

- **No subscription, no credits, no watermark** — open source, Apache-2.0
- **No upload** — your photos never leave your machine; works offline after
  the first model download
- **No account, no API keys** — install once, use forever
- Runs on plain CPUs, NVIDIA CUDA, Apple Silicon, and AMD/Intel GPUs (ONNX)

## Quick start

```bash
pip install "local-upscaler[gui]"
upscaler-gui                          # opens the app in your browser
```

Drag a photo in, click **Enhance**, done. Model weights download automatically
(with checksum verification) the first time you use them.

### Updating

Already installed? New versions ship on PyPI, so upgrading is one line — then
restart `upscaler-gui`:

```bash
pip install -U "local-upscaler[gui,video]"    # [video] bundles ffmpeg (Steam tiles, video)
```

On Windows, if `pip` isn't recognised, use `py -m pip install -U "local-upscaler[gui,video]"`.
`pip show local-upscaler` prints the version you have; the newest is on
[PyPI](https://pypi.org/project/local-upscaler/) and under
[Releases](https://github.com/Maty3k/Upscaler/releases).

> **Never used a terminal before?** Follow the step-by-step
> **[Getting Started guide](docs/GETTING-STARTED.md)** — it starts at
> "install Python" and ends at your first enhanced photo, in baby steps,
> for Windows, Mac, and Linux.

Prefer the command line? The same install gives you the `upscaler` command —
full reference below. Extras: `[gui]` (GUI), `[onnx]` (ONNX backend + Remove
BG), `[face]` (face restore, Colorize), `[video]` (bundled ffmpeg).

> **On a Windows PC with an AMD GPU?** See
> [`docs/SETUP-WINDOWS-AMD.md`](docs/SETUP-WINDOWS-AMD.md) for a full
> GPU-accelerated setup (WSL2 + ROCm) — dramatically faster than CPU/MPS for video.

> Curious about the internals? [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md)
> has the full planning, design decisions, the "why it can make photos worse"
> lesson, and the train-your-own-model playbook.

<details>
<summary>Install from source instead (development)</summary>

```bash
git clone https://github.com/Maty3k/Upscaler.git && cd Upscaler
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[gui]"     # or ".[dev]" for tests
```

</details>

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

# restore faces after upscaling (GFPGAN; needs the [face] extra)
upscaler portrait.jpg --scale 4 --face --face-strength 0.8

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

Supports PNG / JPEG / WebP / AVIF / HEIC / JPEG 2000 / TIFF / GIF / BMP / ICO /
ICNS / TGA / PCX / DIB / SGI / PPM (AVIF needs Pillow ≥ 11.2 or pillow-heif;
HEIC needs pillow-heif). Alpha is flattened onto a white background for formats
that can't store it (JPEG/BMP/PPM/PCX).

#### Remove background & batch

```bash
upscaler removebg photo.jpg -o cutout.png          # transparent PNG (needs [onnx])
upscaler removebg ./folder -o ./out --feather 2     # batch a directory
upscaler batch ./folder -o ./out --op upscale --scale 2     # upscale every image
upscaler batch ./folder -o ./out --op convert -f WebP       # convert every image
upscaler batch ./folder -o ./out --op removebg              # cut out every image
```

`batch` runs one operation over many images and skips unreadable files without
aborting. The GUI also has **Colorize** (DDColor) and **Inpaint / object removal**
(LaMa) tabs — both fully local; Colorize needs the `[face]` extra, Inpaint needs
only torch.

### Video (frame-by-frame)

```bash
upscaler video clip.mp4 -o clip_2x.mp4 --scale 2          # keeps audio
upscaler video clip.mp4 -o clip_2x_60.mp4 --scale 2 --fps 60   # + smooth to 60fps
upscaler video clip.mp4 -o clip_4k.mp4 --scale 4 --size 3840    # fit longest edge to 4K
upscaler video clip.mp4 -o test.mp4 --scale 2 --start 0 --end 5  # trim: first 5s only
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

#### Color & light (no AI)

```bash
upscaler adjust photo.jpg --auto                                   # levels, midtones, white balance
upscaler adjust photo.jpg --preset "Warm golden" --contrast 20
upscaler adjust photo.jpg --exposure 25 --shadows 40 --highlights -30   # rescue a backlit shot
upscaler adjust sky.jpg --exposure -50 --shape band --h 25 --feather 30 # graduated filter
upscaler adjust photo.jpg --mono --mono-mix 40,50,10 --tone-strength 60 # toned black & white
upscaler adjust ./folder -o ./out --auto                           # per-image auto over a folder
```

Exposure, contrast, highlights and shadows, black and white points, midtones,
clarity, temperature, tint, hue, vibrance, saturation, and a black-and-white
conversion with a channel mixer and split tone. `--auto` reads the photo and
sets its levels, midtones and white balance; `--preset` picks from eleven
looks. Exposure and white balance are computed in linear light, so a stop is a
stop. The same region flags as `blur` apply any of it to just a shape, a
graduated band, a painted mask or every detected face. The GUI's **Color & Light** tab has all of it
with a live before/after.

#### Blur (no AI)

```bash
upscaler blur photo.jpg --strength 40                          # whole image, gaussian
upscaler blur photo.jpg --kind pixelate --shape faces --strength 50                   # hide every face
upscaler blur photo.jpg --kind lens --shape faces --outside --face-pad 45             # portrait mode
upscaler blur photo.jpg --kind lens --highlights 60 --shape ellipse --outside          # bokeh around a subject
upscaler blur street.jpg --kind gaussian --shape band --h 30 --feather 20 --outside    # tilt-shift
upscaler blur car.jpg --kind motion --angle 15 --strength 50                           # speed streaks
upscaler blur ./folder -o ./out --kind surface --strength 20                           # smooth skin/noise, keep edges
```

Eight blur kinds (`gaussian`, `box`, `motion`, `spin`, `zoom`, `lens`,
`pixelate`, `surface`) over the whole image or through a `rectangle`,
`ellipse`, `band`, `painted` (`--mask white-is-blur.png`) or `faces` mask,
with feathering, `--outside` to flip the region, and a graded ramp through the
feather. **`--shape faces` finds every face for you** — pixelate them for
privacy, or add `--outside` with lens blur for a portrait-mode look. It needs
OpenCV from the `[face]` extra, and works on the `adjust` command too. Strength is relative to the image's short side, so a setting looks
the same at any resolution. The GUI's **Blur** tab has the same controls with
a live before/after preview and a brush for painted masks.

#### Steam Workshop Showcase tiles

```bash
upscaler steam clip.mp4 -o ./tiles                  # five looping APNG tiles (needs ffmpeg)
upscaler steam photo.jpg --height 200 --pan-y 20    # five still PNG tiles, a taller row
upscaler steam cutout.png --fit contain --bg transparent --width 150 --height 150
upscaler steam tiktok.mp4 -o ./tiles --preset auto    # portrait → 5 full-height copies
upscaler steam clip.mp4 --fps 15 --end 4 --loop boomerang --max-mb 5
upscaler steam clip.mp4 -o ./tiles --gif            # GIF tiles instead (256 colours, smaller)
upscaler steam --how-to-upload                      # the browser-console upload steps
```

Cuts one picture or clip into the five tiles Steam shows side by side in a
profile's **Workshop Showcase**, at Steam's exact geometry (122 px tiles, 4 px
gaps; `--width` / `--height` set the exported pixel size, `--hidpi` doubles it,
and the gaps scale along) so the image lines up across all five. `--bg
transparent` leaves letterbox gaps or a cut-out's see-through area empty so
Steam's own backdrop shows. `--preset` shapes the row from the source's aspect
ratio: `auto` (a portrait TikTok / Reels clip repeats in every tile at full
height, anything else spans the row uncropped), `banner`, `whole`, `center`
(one tile in the middle) or `repeat`. The GUI's Preset dropdown does the same,
with Auto as the default. Clips become one looping animated PNG (or GIF) per tile,
shrunk in steps (256 colours → lower fps → shorter clip) until each file fits the
`--max-mb` budget (default 5 MB; Steam documents 8 MB). Tiles come out
"hexified" (last byte set to 21, the hex-editor step the guides describe, so
Steam keeps the animation; `--no-hexify` to skip), and uploading them needs a
one-line browser-console trick that `--how-to-upload` prints.

Weights download automatically on first use and are cached under
`upscaler/weights/` (override with `UPSCALER_WEIGHTS_DIR`).

### GUI (drag-and-drop)

```bash
upscaler-gui             # opens the app in your browser (http://127.0.0.1:7860)
# from a source checkout: python app.py
```

A full local web app with a tab per tool: **Upscale & Enhance** (with deblur /
denoise, JPEG de-blocking, face restore), **Colorize** (DDColor), **Remove
Objects** (LaMa inpainting), **Remove BG**, **Video** upscaling, **Convert &
Documents** (formats + image ⇄ PDF), **Batch**, **Color & Light** (exposure,
contrast, white balance, vibrance, black & white, with one-click Auto), a
**Blur** toolbox (gaussian,
motion, spin, zoom, lens bokeh, pixelate, surface — whole image, a shaped or
tilt-shift band, a painted mask, or every detected face), a **Lian Li Screen** composer for the 8.8″ case
panel, a **Steam Showcase** tile cutter for your profile's Workshop Showcase,
and a **Library** of everything you export. Runs
entirely on your machine — nothing is uploaded anywhere. (PDF support uses
`pypdfium2`, included in the `.[gui]` extra or installable on its own via
`.[pdf]`.)

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
- **AMD / Intel GPU on native Windows:** torch can't reach these, but the ONNX
  engine can via DirectML — `pip uninstall onnxruntime` then
  `pip install -e ".[directml]"`, and add `--onnx` (CLI) or tick the ONNX
  checkbox (GUI Upscale/Video → Advanced). For maximum AMD speed use WSL2 +
  ROCm instead: see `docs/SETUP-WINDOWS-AMD.md`.

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

This project is **Apache-2.0** (see `LICENSE`). Pretrained weights are
downloaded at runtime and never redistributed in this repo; each carries its own
upstream terms. The core Real-ESRGAN weights are BSD-3-Clause, but several
optional models are **not**: the community upscalers (4x-UltraSharp, Remacri,
NMKD) and the CodeFormer face restorer are non-commercial — their dropdown
entries say so; check upstream terms before commercial use. Credit to
Xintao Wang et al. for Real-ESRGAN and to BasicSR for the RRDBNet architecture,
and to Chen et al. / megvii-research for NAFNet (MIT). NAFNet deblur weights are
mirrored on Hugging Face (`nyanko7/nafnet-models`); the upstream originals are on
the official NAFNet Google Drive.
