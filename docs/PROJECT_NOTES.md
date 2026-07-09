# Upscaler — Project Notes & Planning

A single reference for what this project is, how it was built, the decisions
behind it, and how to (eventually) train your own model. Written so you can pick
it back up cold months from now.

- **Repo:** https://github.com/Maty3k/Upscaler (private)
- **Goal:** local, open-source image **upscaling + sharpening + deblur** — no
  cloud, no API keys, runs on CPU / NVIDIA CUDA / Apple MPS / AMD ROCm.
- **Approach taken:** *wrap pretrained models* (Real-ESRGAN, NAFNet) rather than
  train from scratch. This got a working, polished tool in ~days instead of
  months. See [Training your own](#training-your-own-model) for the from-scratch path.

---

## 1. What exists today (status)

All phases complete and on `main`. Test suite: **21 passing** (CPU, no downloads).

| Phase | What | Key files |
|---|---|---|
| 0/1 | Real-ESRGAN **upscaling** (2x/4x), tiling, lazy weights, unsharp sharpen | `engine.py`, `models/rrdbnet.py`, `models/registry.py`, `models/weights.py`, `cli.py` |
| 2 | NAFNet **deblur** stage (motion blur), runs before upscaling | `deblur.py`, `models/nafnet.py` |
| 3 | **Gradio GUI** (drag-and-drop) | `app.py` |
| 4 | **ONNX Runtime** backend — torch-free, often faster on CPU (`--onnx`) | `onnx_export.py`, `onnx_engine.py` |
| — | SHA-256 checksums pinned on all weights | `models/registry.py`, `scripts/print_checksums.py` |
| — | Served locally at `upscaler.test` via Herd nginx proxy → Gradio | (see [Serving](#5-serving-locally-via-herd)) |

### Models in the registry
- `realesrgan-x4plus` — general 4x (default)
- `realesrgan-x2plus` — general 2x (gentler; better for already-decent photos)
- `realesrgan-x4plus-anime` — illustration / line art
- community upscalers via spandrel — `4x-ultrasharp`, `4x-remacri`, NMKD (non-commercial; see notes)
- `nafnet-gopro-width64` / `width32` — motion deblur; `nafnet-sidd-width64` — denoise
- FBCNN (JPEG de-block) · GFPGAN / CodeFormer (faces) · DDColor (colorize) ·
  Big-LaMa (inpaint) · U²-Net (background removal)

Run `upscaler --list-models` (or open Settings → Models & downloads in the GUI)
for the live list.

Weights are **downloaded lazily on first use**, cached in `upscaler/weights/`
(gitignored), and verified against pinned SHA-256.

---

## 2. How to use it

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # extras: ".[gui]" ".[onnx]" ".[dev]"

# CLI
upscaler photo.jpg -o out.png --scale 4
upscaler photo.jpg --scale 2 --sharpen        # gentler + sharpen pass
upscaler blurry.jpg --deblur --scale 4        # NAFNet deblur first
upscaler ./folder -o ./out --scale 2          # batch a directory
upscaler photo.jpg --scale 4 --onnx           # torch-free ONNX backend
upscaler --list-models

# GUI
pip install -e ".[gui]"
UPSCALER_PORT=7860 python app.py              # http://127.0.0.1:7860

# library
python -c "from upscaler import enhance; from PIL import Image; \
  enhance(Image.open('in.jpg'), scale=2, sharpen=1.0).save('out.png')"
```

---

## 3. Key design decisions (the "why")

- **Vendored the model architectures** (`rrdbnet.py`, `nafnet.py`) instead of
  depending on `basicsr`/`realesrgan`. Those packages break on modern
  torch/torchvision (the infamous `functional_tensor` import). Our vendored
  layer names match the official checkpoints exactly, so weights load with
  `strict=True` — which doubles as a correctness check.
- **Tiled inference for upscaling** (`engine.py`): big images are processed in
  padded tiles to bound memory and avoid seams. RRDBNet is fully local, so
  tiled output == untiled output (there's a test asserting this).
- **NAFNet runs whole-image, NOT tiled.** Its channel attention pools globally,
  so tiling would create per-tile seams. Padding/cropping handled in the wrapper.
- **Weights never committed.** Large + carry upstream licenses. Downloaded at
  runtime, checksum-verified.
- **ONNX export moves dynamic padding out of the graph** (`NAFNet.body()`), so a
  single `.onnx` handles any image size. Verified parity vs torch: upscale
  ≤1/255 per pixel, deblur exact.

### Licensing
- This project: **Apache-2.0**.
- Real-ESRGAN / RRDBNet: BSD-3-Clause. NAFNet: MIT.
- NAFNet weights come from a **community HF mirror** (`nyanko7/nafnet-models`).
  Before any public release: re-host weights under our own account and re-pin
  checksums (`scripts/print_checksums.py`).

---

## 4. Important lesson: "it made my images worse"

This is **expected behavior**, not a bug, when you run a 4x **restoration GAN**
on an **already-decent photo**.

- Real-ESRGAN `x4plus` is trained on *heavily degraded* inputs (JPEG, downscaled,
  noisy). It assumes the input is bad and repaints detail. On a clean photo it
  strips real texture (skin/hair/foliage) and adds a waxy "GAN texture."
- A 4x result viewed *fit-to-screen* gets downscaled by your viewer, so the
  altered texture reads as "lower quality." **Compare at 100% zoom, on a crop.**

**Fixes / guidance:**
- Use **`--scale 2`** (x2plus) for already-good photos — much gentler.
- Keep **deblur off** unless the image is genuinely motion-blurred (it softens
  sharp images).
- Faces are a known Real-ESRGAN weakness (waxiness) — that's what the
  face-restoration stage (GFPGAN/CodeFormer, `--face` / the "Restore faces"
  accordion) is for.
- Rule of thumb: **restoration models help bad inputs, hurt good inputs.**

---

## 5. Serving locally via Herd

`~/Herd` is parked, so the folder is reachable at `upscaler.test`. But Herd
serves PHP and this is a Python app, so we **proxy** the domain to Gradio:

```bash
# one-time: create the nginx proxy (persists across reboots)
herd proxy upscaler http://127.0.0.1:7860 --secure

# each session: start the app on the matching port
cd ~/Herd/Upscaler && UPSCALER_PORT=7860 .venv/bin/python app.py
```

Then `https://upscaler.test` serves the GUI.

### Auto-start across reboots (macOS LaunchAgent)

A LaunchAgent runs the Gradio app at login and restarts it on crash, so the link
survives reboots without manually relaunching. The agent lives at
`~/Library/LaunchAgents/build.artisan.upscaler.plist`; a committed copy +
install/manage instructions are in `scripts/build.artisan.upscaler.plist`.

```bash
launchctl kickstart -k gui/$(id -u)/build.artisan.upscaler   # restart after code changes
launchctl print        gui/$(id -u)/build.artisan.upscaler   # status
launchctl bootout      gui/$(id -u)/build.artisan.upscaler   # stop + unload
```

Logs: `~/Library/Logs/upscaler-gui.log`. After editing `app.py`, `kickstart -k`
to pick up changes.

---

## Training your own model

The big question we explored: *can I train my own upscaler?* Short answer for
the current setup (AMD RX 9070 XT 16GB, Windows): **yes, fine-tuning is very
realistic; from-scratch is possible but rarely worth it.**

### 6.1 Hardware / software reality (AMD, 2026)

- **RX 9070 XT (RDNA4, gfx1201) is officially supported by ROCm 7.2** (released
  Jan 2026). PyTorch-ROCm trains on it out of the box. 16GB VRAM is comfortably
  enough for super-resolution training (patch size / batch size are the VRAM
  drivers, and there's headroom).
- **Windows path = WSL2.** You don't need to leave Windows: WSL2 runs Ubuntu
  *inside* Windows, and AMD's Adrenalin 26.2.2 driver enables ROCm there for the
  9070 XT. Caveats: RDNA4 is newer → less community troubleshooting, occasional
  "GPU not detected" hiccups, slightly slower than bare-metal Ubuntu.
- **Fallback:** `torch-directml` runs on plain Windows (any DX12 GPU) with no
  WSL, but it's slower, lags PyTorch versions, and some ops fall to CPU. Toy
  experiments only.
- In PyTorch-ROCm, the device still reports as `"cuda"` — `torch.cuda.is_available()`
  returning `True` is the sanity check that ROCm sees the card.

### 6.2 From scratch vs fine-tune

| | From scratch | Fine-tune |
|---|---|---|
| Time | **months** (architecture + GAN stability + tuning) | **weeks** |
| Compute | large | modest |
| Expected result | likely **worse** than free Real-ESRGAN | can **beat** stock *on your niche* |
| Worth it for | learning / research | a specific image domain (your photos, scans, a game's art, etc.) |

**Key insight:** good hardware shortens each *run*; it does not shortcut the
*research*. The thing that actually determines quality is the **degradation
model** (how you synthesize "bad" inputs from "good" images), not the network.
Real-ESRGAN's whole edge is its degradation pipeline.

### 6.3 Recommended sequence (next time)

1. **Set up WSL2 + ROCm 7.2 + PyTorch-ROCm**, confirm `torch.cuda.is_available()`.
2. **Pick a narrow domain** with a clear "good image" dataset (this is where a
   custom model can win).
3. **Build the degradation pipeline** — blur, downscale, noise, JPEG — to make
   training pairs. Spend effort here; it matters most.
4. **Toy run first:** small RRDBNet, small patches (e.g. 64–128px), L1 loss,
   confirm the loop trains and loss drops on your card.
5. **Scale up:** more blocks, add perceptual (LPIPS/VGG) + GAN loss, longer
   training. Evaluate with PSNR/SSIM/LPIPS against a Lanczos baseline AND by eye.
6. **Or skip 5 and just fine-tune** the existing Real-ESRGAN weights on your
   pairs — far faster, usually the better ROI.

### 6.4 If you want serious from-scratch training

Rent **cloud NVIDIA GPUs** (RunPod / Vast.ai / Lambda, ~$1–4/hr). A fine-tune is
tens of dollars; from-scratch is hundreds. The 9070 XT is fine for development
and fine-tuning, but for long from-scratch runs the mature CUDA ecosystem saves
more time than it costs.

### Running it on the Windows/AMD box
Full GPU-accelerated setup guide (WSL2 + ROCm for the RX 9070 XT, plus DirectML
and CPU fallbacks): [`docs/SETUP-WINDOWS-AMD.md`](SETUP-WINDOWS-AMD.md). Expect
~tens-of-times faster video upscaling there than on the Mac's MPS.

### Sources (training research)
- ROCm compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- ROCm WSL2 support (Radeon): https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html
- AMD enables PyTorch on Radeon RX 7000/9000 (Win + Linux): https://www.techpowerup.com/341329/amd-enables-pytorch-on-radeon-rx-7000-9000-gpus-with-windows-and-linux-preview
- ROCm in WSL2 — install & limits: https://tillcode.com/amd-rocm-in-wsl2-pytorch-installation-limitations/
- Real-ESRGAN on RX 9070 XT (GitHub issue): https://github.com/xinntao/Real-ESRGAN/issues/986

---

## 7. Open items / future work

- [x] **x2 default for photos** — available via presets/model dropdown (config default stays x4).
- [x] **Face restoration** (GFPGAN/CodeFormer) — shipped (`--face`, "Restore faces" accordion).
- [x] **GUI ONNX toggle** — backend selectable in the GUI's Advanced section.
- [ ] **Re-host NAFNet weights** under our own account; re-pin checksums.
- [x] **CI** — GitHub Actions running `pytest` on the CPU path (Linux/macOS/Windows).
- [ ] **Real speed benchmark** — ONNX vs torch on actual hardware (correctness verified, speed not).
- [ ] **`train/` module** — degradation → dataset → RRDBNet → loss → loop, ROCm-ready (for the training plan above).
- [x] **Auto-start** the Gradio app so `upscaler.test` survives reboots (LaunchAgent).
