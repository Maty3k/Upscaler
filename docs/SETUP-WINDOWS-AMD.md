# Setup on Windows + AMD (RX 9070 XT) — full guide

A step-by-step for running Upscaler on a **Windows PC with a Radeon RX 9070 XT
(RDNA4) and a Ryzen 7 7800X3D**, with real GPU acceleration. Goal: turn the
hours-long Mac renders into minutes.

> **TL;DR**: the GPU does the upscaling, so we want PyTorch using your 9070 XT.
> The supported way to do that on Windows is **WSL2 + ROCm** (Path A). If that
> fights you, there's a **DirectML** fallback on native Windows (Path B), and a
> **CPU-only** last resort (Path C). Verify the GPU is actually used, then run.

---

## 0. What matters (and what doesn't)

- **GPU = the 9070 XT** does ~all the AI upscaling. This is the whole speedup.
- **CPU = 7800X3D** only feeds frames + runs ffmpeg encode/decode. It's plenty;
  it won't bottleneck, but it also isn't where the speed comes from.
- **ROCm** is AMD's CUDA equivalent. Under PyTorch-ROCm, the device still reports
  as `"cuda"` (HIP masquerades as CUDA) and the app's `--device auto` picks it up.
- RDNA4 / 9070 XT support landed in **ROCm 7.2 (Jan 2026)** and WSL2 support via
  the **AMD Adrenalin 26.2.x** driver — it's new, so expect the occasional rough
  edge. Always cross-check the current versions on AMD's pages (links at the end).

Pick a path:

| Path | Where | Speed | Difficulty |
|---|---|---|---|
| **A — WSL2 + ROCm** (recommended) | Ubuntu inside Windows | Fast (GPU) | Medium |
| **B — DirectML** | Native Windows | Medium (GPU, less optimized) | Easy |
| **C — CPU only** | Native Windows | Slow | Easiest |

---

## Path A — WSL2 + ROCm (recommended)

WSL2 runs a real Ubuntu *inside* Windows — you don't dual-boot or leave Windows.

### A1. Update the AMD driver (Windows side)
Install the latest **AMD Software: Adrenalin Edition** (26.2.x or newer) from
amd.com. This is what exposes the GPU to ROCm inside WSL2. Reboot.

### A2. Install WSL2 + Ubuntu 24.04
In an **Administrator PowerShell**:
```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```
Reboot if prompted, set your Ubuntu username/password. From now on, work **inside
the Ubuntu terminal** (open "Ubuntu" from the Start menu) unless noted.

> ROCm officially supports Ubuntu 24.04 / 22.04. Use 24.04.

### A3. Install ROCm in WSL2
Follow AMD's **"Use ROCm on Radeon GPUs → WSL"** guide (link at the end) — it's
the authoritative, version-correct source. The shape of it:
```bash
sudo apt update && sudo apt upgrade -y
# AMD provides an installer script / .deb for the WSL ROCm runtime; grab the
# current one from the AMD WSL page and run it, e.g.:
#   wget <amdgpu-install .deb from AMD WSL page>
#   sudo apt install ./amdgpu-install_*.deb
#   sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
sudo usermod -aG render,video $USER   # then close & reopen the Ubuntu terminal
```
> Don't install the regular Linux GPU driver in WSL — use AMD's **WSL** usecase
> (`--usecase=wsl,rocm`). The Windows Adrenalin driver provides the kernel side.

Verify ROCm sees the card:
```bash
rocminfo | grep -i "gfx\|Marketing"     # should show gfx1201 / Radeon RX 9070 XT
```

### A4. Python + venv (in Ubuntu)
```bash
sudo apt install -y python3 python3-venv python3-pip git ffmpeg
mkdir -p ~/code && cd ~/code
```

### A5. Get the project
It's a **private** GitHub repo, so authenticate first:
```bash
# option 1: GitHub CLI
sudo apt install -y gh && gh auth login        # choose GitHub.com, HTTPS
gh repo clone Maty3k/Upscaler
# option 2: plain git (will prompt for a GitHub token as the password)
# git clone https://github.com/Maty3k/Upscaler.git
cd Upscaler
```

### A6. Virtual env + PyTorch-ROCm
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
Install the **ROCm build of PyTorch** — get the exact command from
<https://pytorch.org/get-started/locally/> (pick Linux → Pip → ROCm). It looks
like:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# (use whatever ROCm version pytorch.org lists as current for your ROCm install)
```

### A7. Install Upscaler + extras
```bash
pip install -e ".[gui,onnx,video]"
```
(That pulls in Gradio, the ONNX backend deps, pypdfium2, pillow-heif, and the
imageio-ffmpeg fallback. You already installed system `ffmpeg` in A4, which is
preferred.)

### A8. Confirm the GPU is actually being used
```bash
python -c "import torch; print('torch', torch.__version__); print('GPU visible:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
You want `GPU visible: True` and your 9070 XT named. (ROCm reports as CUDA — that's
expected and correct.) If it says `False`, see Troubleshooting.

### A9. Run it
**CLI (image):**
```bash
upscaler photo.jpg -o out.png --scale 2 --device cuda
```
**CLI (video, the fast win):**
```bash
upscaler video clip.mp4 -o out.mp4 --scale 2 --device cuda
# trim-test first to confirm the GPU is hot:
upscaler video clip.mp4 -o test.mp4 --scale 2 --device cuda --start 0 --end 3
```
**GUI:**
```bash
pip install -e ".[gui]"   # already done if you used A7
python app.py             # open http://127.0.0.1:7860 in your Windows browser
```
WSL2 forwards localhost, so the URL just works in your normal browser.

While a render runs, open **Windows Task Manager → Performance → GPU** and watch
utilization spike — that's your confirmation the 9070 XT is doing the work.

---

## Path B — Native Windows + DirectML (fallback)

No WSL. Slower than ROCm and lags PyTorch versions, but easy and uses the GPU.

```powershell
# install Python 3.11/3.12 from python.org, then in PowerShell:
winget install Gyan.FFmpeg            # or: pip install imageio-ffmpeg
git clone https://github.com/Maty3k/Upscaler.git
cd Upscaler
python -m venv .venv
.venv\Scripts\activate
pip install torch-directml            # DirectML backend
pip install -e ".[gui,onnx,video]"
```
DirectML doesn't expose itself as `cuda`, so the app's `--device cuda/auto` won't
pick it up out of the box — DirectML support would need a small code tweak to
route tensors to the `dml` device. **Practical recommendation: prefer Path A.**
If you want, ping me and I'll add a `--device dml` option; otherwise use ROCm.

---

## Path C — CPU only (last resort)

Works anywhere, no GPU setup, but slow (similar order to the Mac):
```powershell
git clone https://github.com/Maty3k/Upscaler.git
cd Upscaler && python -m venv .venv && .venv\Scripts\activate
pip install torch                     # CPU build
pip install -e ".[gui,onnx,video]"
upscaler photo.jpg -o out.png --scale 2 --device cpu
```
Fine for occasional single images; not for video.

---

## Performance tips (once it runs)

- **Use ×2, not ×4, for 1080p → 4K.** 1080p × 2 = exact 4K; ×4 renders 8K and
  throws most away (~4× slower). `--scale 2` or the `realesrgan-x2plus` model.
- **Trim-test first.** `--start 0 --end 3` (or the Trim box in the GUI) — confirm
  the look and that the GPU is engaged before committing to a full clip.
- **Tile size**: with 16 GB VRAM you can likely raise `--tile` (e.g. 0 = whole
  frame, or 1024) for fewer tiles/overhead. If you hit out-of-memory, lower it.
- **Skip `--fps` interpolation** unless you want it — it's a heavy extra pass.
- **ONNX backend** (`--onnx`) is mainly a CPU win; with ROCm, plain torch on the
  GPU is the fast path.

---

## Troubleshooting

- **`torch.cuda.is_available()` is False (Path A):**
  - Reopen the Ubuntu terminal after `usermod -aG render,video`.
  - Confirm `rocminfo` lists gfx1201. If not, the WSL ROCm runtime or Adrenalin
    driver version is off — recheck A1/A3 against AMD's current WSL page.
  - Make sure you installed the **ROCm** torch wheel (A6), not the default CUDA
    one. `python -c "import torch; print(torch.version.hip)"` should be non-None.
  - As a last resort some RDNA cards need `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
    — the 9070 XT is officially supported so you shouldn't need it; try only if
    stuck.
- **`ffmpeg not found`** (video tab): `sudo apt install ffmpeg` (WSL) or
  `winget install Gyan.FFmpeg` (Windows), or rely on the bundled
  `imageio-ffmpeg` from the `.[video]` extra.
- **GUI URL won't open**: run `python app.py`, then open the printed
  `http://127.0.0.1:7860` in your Windows browser (WSL forwards localhost).
- **Private-repo clone fails**: you must be authenticated as the repo owner —
  `gh auth login`, or use a GitHub Personal Access Token as the git password.
- **Out-of-memory during a frame**: lower `--tile` (e.g. 512 → 256).

---

## Reference links (check for current versions)

- ROCm compatibility matrix: <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- Use ROCm on Radeon — WSL: <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html>
- PyTorch install selector (get the ROCm wheel command): <https://pytorch.org/get-started/locally/>
- AMD enables PyTorch on Radeon (Win + Linux): <https://www.techpowerup.com/341329/amd-enables-pytorch-on-radeon-rx-7000-9000-gpus-with-windows-and-linux-preview>

---

## Expected payoff

On the Mac (MPS): ~1 min per 1080p frame → hours per clip. On the 9070 XT via
ROCm: expect **roughly ~0.5–2 s per frame** (not a measured promise — RDNA4 ROCm
is new), i.e. **tens of times faster**, turning a 5-hour clip into minutes. Do a
3-second trim test on day one to see your real per-frame time, then scale up.
