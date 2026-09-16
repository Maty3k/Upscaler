# Getting Started — from zero to your first upscaled photo

This guide assumes **nothing**: not that you have Python, not that you've ever
opened a terminal. Follow it top to bottom and in about 10 minutes you'll have
a free, private photo enhancer running on your own computer.

Everything runs **locally**. Your photos are never uploaded anywhere, there is
no account, no subscription, and no watermark.

---

## Step 1 — Install Python (one time)

Upscaler is a Python program, so your computer needs Python first. Version
**3.10, 3.11, or 3.12** is ideal.

### Windows

1. Go to <https://www.python.org/downloads/> and click the yellow
   **Download Python 3.12.x** button.
2. Open the file it downloaded.
3. **Important:** on the very first screen, tick the checkbox at the bottom
   that says **"Add python.exe to PATH"**. Don't skip this — it's what makes
   the commands below work.
4. Click **Install Now** and let it finish.

### Mac

1. Go to <https://www.python.org/downloads/> and click
   **Download Python 3.12.x**.
2. Open the downloaded `.pkg` file and click through the installer
   (Continue → Continue → Agree → Install).

### Linux

You almost certainly already have Python. If not:
`sudo apt install python3 python3-pip python3-venv` (Ubuntu/Debian) or your
distro's equivalent.

---

## Step 2 — Open a terminal

This is the black window where you type commands. You'll only need two.

- **Windows:** press the **Windows key**, type `powershell`, press **Enter**.
- **Mac:** press **Cmd + Space**, type `terminal`, press **Enter**.
- **Linux:** you know where it is. 🙂

---

## Step 3 — Install Upscaler

Copy this line, paste it into the terminal (on Windows: right-click pastes),
and press **Enter**:

```
pip install "local-upscaler[gui]"
```

> **Mac/Linux:** if `pip` isn't found, use `pip3` instead:
> `pip3 install "local-upscaler[gui]"`

It will download for a few minutes (the AI engine, PyTorch, is large — roughly
2 GB). When the text stops scrolling and you can type again, it's done.

---

## Step 4 — Start it

Type this and press **Enter**:

```
upscaler-gui
```

After a few seconds your **web browser opens by itself** with the Upscaler
app. That page is running on *your* computer — the address `127.0.0.1` means
"this machine", not the internet.

Leave the terminal window open while you use the app (it *is* the app —
closing it stops the program). Next time you want to use Upscaler, you only
repeat Steps 2 and 4.

---

## Step 5 — Enhance your first photo

1. In the browser, you're on the **Upscale & Enhance** tab.
2. **Drag a photo** into the big image box (or click it to browse).
3. Pick a scale — **2×** doubles the width and height, **4×** quadruples them.
4. Click the **Enhance** button.
5. The first time only, it downloads the AI model (~65 MB) — you'll see the
   progress. After that it's cached and instant.
6. Compare before/after with the slider, then click **download** on the
   result to save it.

That's it. You've upscaled a photo, for free, without it ever leaving your PC.

---

## Updating to the latest version

New features and fixes come out as new versions. To get them, open a terminal
(Step 2) and run:

```
pip install -U "local-upscaler[gui,video]"
```

> **Windows, if `pip` is not recognised:** use
> `py -m pip install -U "local-upscaler[gui,video]"` instead.

Wait for it to finish, close the app if it's running (close its terminal
window), and start it again with `upscaler-gui`. That's the whole update.

---

## What else is in there?

Each tab is a separate tool, all local:

| Tab | What it does |
|---|---|
| **Upscale & Enhance** | Make photos larger and sharper; fix blur, denoise, restore faces |
| **Colorize** | Add color to black-and-white photos |
| **Objects** | Paint over something (a person, a sign, a wire) and it disappears |
| **Remove BG** | Cut out the subject, transparent background |
| **Color & Light** | Brightness, contrast, color and black & white — with a one-click Auto |
| **Watermark** | Sign your photos with text or a logo — in a corner, tiled, or behind the subject so a headline passes behind the person |
| **Design** | Start from a finished layout — thumbnail, quote card, poster, title slide — and fill in the words and the photo |
| **Screenshot** | Make a screen capture presentable: padding, rounded corners and a shadow on a colour field, in a window or browser frame, leaned back in 3D |
| **Crop** | Crop to any shape, straighten a tilted photo, add a border or a drop shadow |
| **Sharpen** | Bring out detail four different ways, with halo control so edges stay natural |
| **Effects** | Film looks — grain, glow, light leaks, vignette, halftone, retro dithering and glitch |
| **Blur** | Eight kinds of blur — soft, motion, bokeh, pixelate, tilt-shift — on the whole photo, a shape, where you paint, every face it finds, or a real depth of field |
| **Video** | Upscale whole videos frame-by-frame (slow but works) |
| **Convert & Documents** | Convert between PNG/JPEG/WebP/HEIC/…, squeeze a photo under a size limit, remove the GPS location a photo records, images ⇄ PDF |
| **Batch** | Apply one operation — or a whole saved recipe — to a folder of images |
| **Lian Li** | Compose media for the Lian Li 8.8″ case screen at its exact size |
| **Steam** | Cut a picture or clip into the five tiles of a Steam profile Workshop Showcase |
| **Library** | Everything you've exported, in one place |

---

## When something doesn't work

**"`pip` is not recognized" / "`upscaler-gui` is not recognized" (Windows)**
Python was installed without the **Add to PATH** checkbox. Easiest fix: run
the Python installer again, choose **Modify**, and enable it — or uninstall
and reinstall with the box ticked. Then open a **new** PowerShell window.

**"`pip: command not found`" (Mac/Linux)**
Use `pip3` and `python3` instead of `pip` and `python`.

**The browser doesn't open**
Look in the terminal for a line like `Running on local URL: http://127.0.0.1:7860`
and open that address in your browser yourself.

**It's slow**
Normal on a plain CPU — a big photo at 4× can take a minute or two.
- **Mac (M1/M2/M3/M4):** it automatically uses the Apple GPU, much faster.
- **NVIDIA GPU:** it automatically uses CUDA.
- **AMD/Intel GPU on Windows:** see the DirectML note in the main README, or
  the full guide in [`SETUP-WINDOWS-AMD.md`](SETUP-WINDOWS-AMD.md).

**"Couldn't download …" on first enhance**
The model download needs internet *once*. Check your connection and click
Enhance again — interrupted downloads are thrown away and restarted safely.

**Updating later**

```
pip install --upgrade "local-upscaler[gui]"
```

---

## Prefer typing commands? (optional)

Everything the GUI does is also a command — handy for folders full of images:

```
upscaler photo.jpg                       # 4x upscale → photo_upscaled.png
upscaler photo.jpg --scale 2 --sharpen   # 2x + sharpen
upscaler ./my_folder -o ./out --scale 4  # whole folder
upscaler --help                          # everything else
```

See the [main README](../README.md) for the full command reference.
