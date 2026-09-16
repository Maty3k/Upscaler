# Upscaler

**A free, private photo editor that runs on your own computer.** It started as
an AI upscaler and grew into the whole darkroom — the kind of results the paid
cloud services charge a subscription for, running 100% locally on top of
pretrained [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) weights.

**With AI:** upscale and enlarge · deblur and denoise · restore faces ·
colorize black-and-white · remove objects · cut out backgrounds ·
depth-of-field blur · upscale video frame by frame

**Without AI, and instantly:** color and light with one-click Auto · film
effects and looks · sharpen · blur · crop, straighten and frame · watermark ·
convert formats · fit a file-size budget · see and strip the GPS location a
photo records · images ⇄ PDF · Steam showcase tiles · a composer for the
Lian Li 8.8″ case screen

**Over a whole folder:** batch any one of them, or save a chain of edits as a
recipe and run the lot in one command.

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

#### Recipes — a saved chain of edits, over a whole folder

```bash
upscaler recipe --list                              # the built-in ones
upscaler recipe "Web-ready" ./holiday -o ./out      # run one over a folder
upscaler recipe "Film look" --save mine.json        # save it, edit it, keep it
upscaler recipe mine.json photo.jpg
upscaler recipe "Blur every face" ./photos -o ./safe
```

Every other tool does one thing to one photo. A recipe is the ordered list:
*level it, warm it, sharpen it, sign it, and squeeze it under 500 KB* becomes
one command over two hundred holiday photos. Steps name the tools you already
know and the settings those tools already take, so anything you can do in a tab
you can put in a recipe, including restricting a step to a shape, to every
detected face, or to depth.

Saved as plain JSON you can edit. Settings a step doesn't recognise are
ignored, so a recipe written against another version still runs, but a step
naming a tool that doesn't exist is refused rather than skipped — a silently
skipped step gives you a file that looks right and isn't. What comes out at the
end (the format, a size budget, whether the metadata goes) belongs to the
recipe rather than to any step. Seven ready-made ones ship, and the GUI has
them under **Batch → Recipe**, where the JSON is editable in place.

#### See and remove metadata (no AI)

```bash
upscaler metadata photo.jpg                      # what does this file reveal?
upscaler metadata ./folder                       # check a whole folder, writes nothing
upscaler metadata photo.jpg --remove             # strip it, losslessly
upscaler metadata photo.jpg --remove --mode "remove location only"
upscaler metadata ./folder --remove --in-place
```

Your camera writes a block of data next to the pixels that travels with the
file: **where the photo was taken** to within a few metres, when, the camera
and lens, the body's **serial number**, and often a small embedded copy of the
picture that a crop may not have regenerated. Large platforms strip it on
upload; forums, email attachments, file transfers and your own site do not.

**Cleaning a JPEG or PNG here is lossless.** Re-saving through an image library
would drop the metadata but re-compress the picture and cost quality every
time. Instead the private segments are cut out and the compressed image data is
copied through untouched, so the pixels come out identical byte for byte, which
the tests assert. Other formats fall back to a re-encode and say so.

Inspection is the default and never writes anything. The orientation tag is
kept by default, since phones store some photos sideways plus a tag saying to
rotate them, and dropping it would lay the picture on its side. In the GUI it
is **Convert → Remove metadata (privacy)**.

#### Fit a file-size budget (no AI)

```bash
upscaler optimize photo.jpg -t 500KB               # under half a meg, best quality that fits
upscaler optimize photo.jpg -t 2MB -f JPEG         # when the site won't take WebP
upscaler optimize ./folder -o ./out -t 1MB         # a whole folder for an email
upscaler optimize photo.jpg -t 200KB --no-resize   # keep the dimensions, whatever it costs
upscaler optimize photo.jpg -t 300KB --max-edge 1920
```

Encoder quality is **searched, not guessed**: the picture is encoded into
memory and measured over and over, because how many bytes a photo takes
depends entirely on what is in it. The result is the highest quality that
still fits. The picture is only shrunk if quality alone can't reach the
target, and the scale is estimated from how far over budget it was rather
than stepped down blindly. `auto` picks WebP, which carries the same picture
in roughly half a JPEG's bytes. Metadata is always stripped, which drops the
GPS coordinates along with the bytes, and a file that already fits is left
alone rather than needlessly re-encoded. The GUI has it under
**Convert → Fit a file-size budget**.

#### Watermark (no AI)

```bash
upscaler watermark photo.jpg --text "© Your Name"
upscaler watermark ./folder -o ./out --text "© Studio" --position "bottom left"
upscaler watermark photo.jpg --preset "Proof (tiled)"        # can't be cropped off
upscaler watermark photo.jpg --logo logo.png --logo-size 20 --opacity 85
upscaler watermark photo.jpg --text DRAFT --position tiled --tile-angle 45 --opacity 20
upscaler watermark --list-fonts
```

Text or an image, placed in any of nine positions or **tiled** across the whole
frame, with opacity, rotation and margin. Text gets an outline and a soft
shadow by default, because a white signature is invisible on a bright sky
without them. Every size is a share of the photo, so one setting suits a whole
folder of mixed pictures — which is the point of running it over a directory.

#### Crop & frame (no AI)

```bash
upscaler crop photo.jpg --aspect 1:1                        # square, centred
upscaler crop photo.jpg --aspect 9:16 --mode fit --border-style "blurred photo"
upscaler crop photo.jpg --straighten 4 --lean-v 30          # level it, fix leaning walls
upscaler crop photo.jpg --preset Polaroid                   # white mat + drop shadow
upscaler crop photo.jpg --size 2560x1440 --aspect 16:9      # exact wallpaper
upscaler crop ./folder -o ./out --aspect 4:5 --position 50,30
```

Crop to a named shape or any ratio, choose what survives with `--position`,
zoom in further, straighten a tilted horizon, and correct converging verticals
or horizontals. A straighten or a lean is **trimmed back to the largest
rectangle of real pixels**, so it never leaves empty corners. `--mode fit`
keeps the whole photo and fills the margin instead of cropping, with a solid
colour or a zoomed blurred copy of the photo. Then a border, rounded corners,
a drop shadow, and an exact output size. Ten presets cover the common posts,
mats and wallpapers.

#### Sharpen (no AI)

```bash
upscaler sharpen photo.jpg --preset Standard
upscaler sharpen photo.jpg --amount 120 --radius 1.2 --halo 25
upscaler sharpen portrait.jpg --preset "Portrait (skin-safe)"      # edge-aware, spares skin
upscaler sharpen soft.jpg --kind high-pass --amount 180 --radius 2
upscaler sharpen photo.jpg --kind texture --shape faces            # just the faces
upscaler sharpen ./folder -o ./out --preset "After upscaling"
```

Four methods: the classic **unsharp** mask, a **high-pass** overlay that lifts
edges without shifting overall tone, an edge-aware **smart** pass that leaves
skin, sky and noise alone, and a two-scale **texture** pass. A **halo limit**
caps how far an edge may overshoot, which is what separates sharpening from an
outlined look, and sharpening runs on brightness only by default so edges don't
pick up colored fringes. The radius is in **pixels**, because that is the scale
real detail lives at — so the GUI's preview is a genuine 1:1 crop rather than a
shrunken copy, which would hide the very artefacts you are checking for.

#### Effects & film looks (no AI)

```bash
upscaler effects photo.jpg --look "Film grain"
upscaler effects photo.jpg --grain 30 --halation 50 --vignette 40      # stack them yourself
upscaler effects photo.jpg --look "Newspaper print"                    # halftone dot screen
upscaler effects photo.jpg --duotone 100 --duotone-dark "#10203f" --duotone-light "#f2c76b"
upscaler effects photo.jpg --look "VHS glitch" --glitch-seed 42        # a different tear
upscaler effects ./folder -o ./out --look Lomo
```

Eleven effects that **stack**: grain with adjustable coarseness, halation
(the glow that bleeds out of highlights), light leaks, vignette, chromatic
aberration, duotone, posterize, ordered dither, halftone dot screens,
scanlines and glitch. Twelve looks combine them. Everything is sized relative
to the photo, so a setting looks the same at any resolution, and the same
region flags as `blur` restrict effects to a shape, a band, a painted mask or
every detected face. The GUI's **Effects** tab has all of it with a live
before/after.

#### Blur (no AI)

```bash
upscaler blur photo.jpg --strength 40                          # whole image, gaussian
upscaler blur photo.jpg --kind pixelate --shape faces --strength 50                   # hide every face
upscaler blur photo.jpg --kind lens --shape faces --outside --face-pad 45             # portrait mode
upscaler blur photo.jpg --kind lens --shape depth --focus-at 50,60 --dof 20           # real depth of field
upscaler blur photo.jpg --kind lens --highlights 60 --shape ellipse --outside          # bokeh around a subject
upscaler blur street.jpg --kind gaussian --shape band --h 30 --feather 20 --outside    # tilt-shift
upscaler blur car.jpg --kind motion --angle 15 --strength 50                           # speed streaks
upscaler blur ./folder -o ./out --kind surface --strength 20                           # smooth skin/noise, keep edges
```

Eight blur kinds (`gaussian`, `box`, `motion`, `spin`, `zoom`, `lens`,
`pixelate`, `surface`) over the whole image or through a `rectangle`,
`ellipse`, `band`, `painted` (`--mask white-is-blur.png`) or `faces` mask,
with feathering, `--outside` to flip the region, and a graded ramp through the
feather.

**`--shape depth` is a real depth of field.** A depth model works out how far
away every pixel is, and the blur grows with distance from whatever you focus
on, so a distant wall softens more than a nearby one. Point at your subject
with `--focus-at X,Y` as percentages, the way you tap a phone screen, and set
how deep the sharp zone runs with `--dof`. In the GUI you click the depth map
itself. Several blur levels are composited rather than cross-fading one blurred
copy against the sharp one, which would leave a double exposure at the
half-blurred distances. It needs the `[onnx]` extra, and the model is Depth
Anything V2 Small, Apache-2.0, a 26 MB download.

**`--shape faces` finds every face for you** — pixelate them for
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
contrast, white balance, vibrance, black & white, with one-click Auto),
**Effects** (grain, halation, light leaks, vignette, duotone, halftone,
dither, scanlines, glitch — twelve ready-made film looks), **Sharpen** (unsharp,
high-pass, edge-aware and texture, with halo control), **Crop & Frame** (any
aspect, straighten, lean correction, exact sizes, borders and shadows),
**Watermark** (text or logo, in a corner or tiled), a file-size
budget fitter, a **metadata cleaner** that shows what a photo reveals and
strips it losslessly, **recipes** that run a whole saved chain of edits over a
folder, a
**Blur** toolbox (gaussian,
motion, spin, zoom, lens bokeh, pixelate, surface — whole image, a shaped or
tilt-shift band, a painted mask, every detected face, or a real depth of field), a **Lian Li Screen** composer for the 8.8″ case
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
