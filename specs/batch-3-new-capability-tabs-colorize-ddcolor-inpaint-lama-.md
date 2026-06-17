# Batch 3 — New capability tabs — Colorize (DDColor) + Inpaint (LaMa) + CLI parity

**Scope:** Batch 3

## Goal
Add two new spandrel-backed capability tabs to the Gradio app — Colorize (DDColor, predicts chroma and merges with source luminance) and Object-removal/Inpaint (LaMa, brush-mask) — each with a before/after slider and automatic Library save, mirroring the existing Remove-BG tab. Then bring the CLI to parity with the GUI by adding `removebg` and `batch` subcommands and a `--face` flag on the default upscale path, all lazy-importing the optional `.[face]` extras behind the same friendly "install .[face]" message the GUI already uses.

## Dependencies
- 3A and 3B both add new optional-dependency-gated modules under upscaler/ and new registry dataclasses to upscaler/models/registry.py; they should land in coordination to avoid registry.py merge conflicts but are otherwise independent and can be built in parallel.
- 3A, 3B and 3C all assume the [face] extra already installs spandrel + spandrel_extra_arches + opencv-python-headless (verified in pyproject.toml). If Batch 1's shared spandrel loader helper lands first, 3A/3B should reuse it instead of each calling spandrel.ModelLoader directly (soft dependency on Batch 1, not blocking).
- 3C's --face path depends on the existing upscaler/face.py FaceRestorer (already present) — no dependency on 3A/3B. 3C removebg/batch depend only on existing upscaler/background.py + convert + engine.
- ensure_weights + sha256 pinning (upscaler/models/weights.py) is reused unchanged by 3A and 3B; pinning real checksums for DDColor/LaMa requires a one-time download + scripts/print_checksums.py run at implementation time (cannot be done offline in this spec).

---

## Item 3A — Colorize tab using DDColor (spandrel torch path)  _(effort: L)_

Add upscaler/colorize.py wrapping spandrel's DDColor architecture to colorize a grayscale/faded photo. DECISION: use the torch/spandrel path, NOT an onnxruntime path like background.py. Rationale verified by reading code: spandrel's DDColor descriptor (spandrel_extra_arches/architectures/DDColor/__init__.py) ships a custom call_fn (`_call`) that already implements the full DDColor colorization pipeline in-tensor — it takes a (1,1,H,W) grayscale-L tensor, predicts 2-channel AB chroma at the model's internal input_size (256x256), resizes AB back to full res, concatenates with the ORIGINAL source L channel, and converts LAB->RGB, returning a full (1,3,H,W) RGB tensor. So 'predicts chroma, merge with source luminance' is handled inside spandrel; our code only feeds the L channel and receives RGB. This matches the established [face] extra (spandrel already a dep) and reuses ensure_weights + sha256 pinning, unlike background.py's separate onnxruntime path which would require sourcing/exporting an ONNX DDColor (no pinned ONNX export exists). Module mirrors upscaler/face.py structure: a `_deps()` lazy import raising the same friendly RuntimeError ('Colorization needs extra packages. Install them with: pip install -e ".[face]"'), a Colorizer class that lazily loads weights via spandrel.ModelLoader().load_from_file(str(ensure_weights(spec))), and a `colorize(img, strength=1.0)->Image` method. The L tensor is built from img.convert('L') normalized to [0,1] as shape (1,1,H,W); spandrel handles the rest. Add a `strength` blend in [0,1] that lerps the colorized RGB back toward a grayscale RGB of the source so users can dial down saturation (consistent with face_strength/restore_strength semantics elsewhere). Add ColorizeSpec dataclass + COLORIZE_MODELS registry dict + DEFAULT_COLORIZE_MODEL to upscaler/models/registry.py (mirroring FaceSpec/FACE_MODELS), with a sha256-pinned DDColor checkpoint (e.g. ddcolor_modelscope / ddcolor_artistic — pin the chosen .pth's real sha256 via scripts/print_checksums.py at implementation time; MUST be a 2-output-channel AB checkpoint or spandrel._call raises ValueError). Wire a `colorize_ui(image, model, strength, progress=gr.Progress())` handler in app.py mirroring remove_bg_ui: guard None input with gr.Error, convert to RGB/L, run, save result with library.save_image(result,'colorize'), return (original,result) for the ImageSlider plus an info Markdown. Add a 'Colorize' gr.Tab after 'Remove BG' with gr.Image input + model Dropdown + strength Slider + Tips accordion + Colorize/Clear buttons on the left, and a gr.ImageSlider (elem_classes=['loupe']) before/after + info on the right, registering run/clear .click wiring in build_demo alongside the other tabs.

**Files touched**
- upscaler/colorize.py
- upscaler/models/registry.py
- app.py
- tests/test_colorize.py
- pyproject.toml
- README.md

**New files**
- upscaler/colorize.py
- tests/test_colorize.py

**Registry / data additions**
- upscaler/models/registry.py: add ColorizeSpec frozen dataclass (name,url,filename,sha256,notes) + COLORIZE_MODELS dict with at least one sha256-pinned 2-output-channel DDColor checkpoint + DEFAULT_COLORIZE_MODEL constant

**UI changes**
- app.py: new 'Colorize' gr.Tab (placed after the 'Remove BG' tab) with input gr.Image, model gr.Dropdown(_COLORIZE_CHOICES), strength gr.Slider(0..1, default 1.0), Tips accordion, Colorize primary + Clear secondary buttons, and right column gr.ImageSlider(type='pil', elem_classes=['loupe']) + gr.Markdown info; ICON reuse (ICON_AI) for section head
- app.py: _COLORIZE_CHOICES list comprehension built from colorize.COLORIZE_MODELS mirroring _BG_CHOICES; colorize_ui handler + colorize_btn.click / colorize_clear.click wiring inside build_demo

**Tests**
- tests/test_colorize.py::test_colorize_models_pinned — every spec in COLORIZE_MODELS has a 64-char sha256 and non-empty url/filename (network-free, mirrors test_face_models_pinned)
- tests/test_colorize.py::test_colorize_unknown_model_raises — Colorizer(model='nope') (or colorize() with a bad model) raises ValueError listing available models, WITHOUT downloading (construct so the model-name check happens before ensure_weights, like FaceRestorer)
- tests/test_colorize.py::test_colorize_deps_message — monkeypatch the lazy import to raise ImportError and assert the raised RuntimeError text contains 'pip install -e ".[face]"' (mirrors face _deps message)
- tests/test_colorize.py::test_colorize_grayscale_input_returns_rgb_same_size — pytest.importorskip('spandrel','spandrel_extra_arches'); skip with a clear message if weights unavailable offline; otherwise assert output is mode RGB and same WxH as input on a tiny synthetic image (guarded like test_no_face_image_passes_through)
- tests/test_colorize.py::test_colorize_strength_zero_is_grayscale — at strength=0 the output equals (within tolerance) a grayscale RGB of the source (verifies the luminance-merge/blend math without asserting exact model colors); importorskip-guarded

**Acceptance criteria**
- Reviewer can run `pytest tests/test_colorize.py` with NO weights present and NO network and see all non-model tests pass (pinned-registry, unknown-model, deps-message); model-inference tests skip cleanly when spandrel/weights are unavailable, never error
- upscaler/colorize.py imports spandrel ONLY inside a _deps()/lazy path; importing the module with the [face] extra absent does not raise, and triggering colorize without the extra raises gr.Error/RuntimeError whose message is exactly the friendly 'pip install -e ".[face]"' string
- With the [face] extra + a pinned DDColor checkpoint present, the Colorize tab turns a grayscale image into a plausibly colorized RGB image of identical dimensions, the result is auto-saved to the Library (a colorize_<timestamp>.png appears via library.list_items()), and the before/after ImageSlider shows original vs colorized
- Colorize strength slider visibly changes saturation: strength=1 full color, strength=0 returns the source as grayscale-RGB (source luminance preserved at all strengths)
- Every COLORIZE_MODELS entry has a real 64-hex sha256 pinned in registry.py; pyproject.toml/README note that Colorize uses the existing [face] extra (no new optional-dependency group added)

**Risks**
- Picking a DDColor checkpoint whose num_output_channels != 2 will make spandrel's _call raise ValueError('only supports 2 output channels'); the spec must pin an AB-output checkpoint (verified: _call asserts model.num_output_channels==2). Implementer must confirm the chosen .pth loads as a 2-channel model before pinning.
- DDColor weights are large (convnext-L backbone) and licensing/host stability of the .pth mirror must be checked before pinning a sha256; could not verify a specific download URL/sha256 here (no network used)
- DDColor downsamples to 256x256 internally for the chroma prediction, so very large images get chroma from a low-res pass — fine (it's how DDColor works) but worth a Tips note
- Could NOT run an actual DDColor inference (no weights downloaded), so the end-to-end colorization quality and exact memory/runtime on CPU are unverified; only the spandrel call contract was verified by reading source

---

## Item 3B — Object-removal / Inpaint tab using LaMa with a brush mask  _(effort: L)_

Add upscaler/inpaint.py wrapping spandrel's LaMa architecture for object removal. VERIFIED from spandrel source (spandrel/architectures/LaMa/__init__.py): LaMa.load() returns a MaskedImageModelDescriptor (NOT the regular ImageModelDescriptor), whose __call__(image, mask) takes image as a (1,3,H,W) float[0,1] tensor and mask as a (1,1,H,W) tensor with values 0=keep / 1=inpaint, returns (1,3,H,W) RGB; size_requirements minimum=16, multiple_of=8 (the descriptor auto-pads/crops). So the module must call `desc(img_tensor, mask_tensor)` — distinct from the single-arg upscale/colorize calls. Module mirrors face.py/colorize.py: lazy `_deps()` with the same 'pip install -e ".[face]"' message, an Inpainter class loading weights via spandrel.ModelLoader().load_from_file(str(ensure_weights(spec))), an `inpaint(image, mask)->Image` method that builds the two tensors (image RGB->(1,3,H,W); mask: a single-channel array thresholded to 0/1->(1,1,H,W)), runs desc(image,mask) under torch.inference_mode, and returns RGB. Add InpaintSpec + INPAINT_MODELS + DEFAULT_INPAINT_MODEL to registry.py (sha256-pinned Big-LaMa .pt/.ckpt). UI: use gr.ImageEditor (Gradio 6.15.2 has it, verified) configured with a white Brush and no other layers so the user paints over the object to remove; the editor returns a dict with 'background' (the source) and 'layers'/'composite' — the handler derives the binary mask from the painted layer alpha (where the user drew) so the user never uploads a separate mask file. Handler inpaint_ui(editor_value, model, progress) guards empty input, extracts background+mask, raises a friendly gr.Error if nothing was painted, runs Inpainter.inpaint, saves with library.save_image(result,'inpaint'), and returns (original, result) for an ImageSlider + info. Add an 'Inpaint' gr.Tab after 'Colorize' with the gr.ImageEditor on the left + model dropdown + Tips + Remove/Clear buttons, and the before/after ImageSlider + info on the right; wire .click in build_demo.

**Files touched**
- upscaler/inpaint.py
- upscaler/models/registry.py
- app.py
- tests/test_inpaint.py
- pyproject.toml
- README.md

**New files**
- upscaler/inpaint.py
- tests/test_inpaint.py

**Registry / data additions**
- upscaler/models/registry.py: add InpaintSpec frozen dataclass (name,url,filename,sha256,notes) + INPAINT_MODELS dict with a sha256-pinned LaMa (Big-LaMa) checkpoint + DEFAULT_INPAINT_MODEL constant

**UI changes**
- app.py: new 'Inpaint' gr.Tab (after 'Colorize') with a gr.ImageEditor(type='pil', brush=gr.Brush(colors=['#ffffff'], color_mode='fixed'), layers=False or sources=['upload','clipboard']) for paint-the-mask UX, a model gr.Dropdown(_INPAINT_CHOICES), Tips accordion, 'Remove object' primary + 'Clear' secondary buttons; right column gr.ImageSlider(type='pil', elem_classes=['loupe']) + gr.Markdown info
- app.py: _INPAINT_CHOICES built from inpaint.INPAINT_MODELS; inpaint_ui handler + .click wiring in build_demo; mask-derivation helper that converts the ImageEditor painted layer to a 0/1 L-mask the same size as the background

**Tests**
- tests/test_inpaint.py::test_inpaint_models_pinned — every INPAINT_MODELS spec has a 64-char sha256 + non-empty url/filename (network-free)
- tests/test_inpaint.py::test_inpaint_unknown_model_raises — Inpainter(model='nope') raises ValueError before any download
- tests/test_inpaint.py::test_inpaint_deps_message — monkeypatched ImportError yields the friendly 'pip install -e ".[face]"' RuntimeError text
- tests/test_inpaint.py::test_mask_from_editor_layer — unit-test the app-level mask-derivation helper: given a fake ImageEditor dict (background + a layer with a painted white blob on transparent), it returns a single-channel L image, same size as background, with 1/255 where painted and 0 elsewhere (no model, no network)
- tests/test_inpaint.py::test_inpaint_empty_mask_raises — inpaint_ui with an editor value that has no painted pixels raises gr.Error('paint over the object…') and never loads the model
- tests/test_inpaint.py::test_inpaint_runs_and_preserves_size — pytest.importorskip('spandrel'); skip if weights unavailable offline; else assert desc(image,mask) path returns RGB same WxH as input on a tiny image with a small painted mask

**Acceptance criteria**
- `pytest tests/test_inpaint.py` passes with NO network/weights for all non-model tests (pinned-registry, unknown-model, deps-message, mask-derivation, empty-mask); model tests skip cleanly otherwise
- upscaler/inpaint.py uses the MaskedImageModelDescriptor two-argument call `desc(image_tensor, mask_tensor)` (image (1,3,H,W), mask (1,1,H,W) with 0=keep/1=inpaint) — reviewer can confirm by reading the call site matches the verified spandrel contract
- In the GUI, a user can paint over an object with the brush (no separate mask upload), click Remove object, and get a result where the painted region is plausibly filled; the result is auto-saved to the Library (inpaint_<timestamp>.png) and shown in the before/after ImageSlider
- Painting nothing and clicking Remove gives a clear friendly error (no crash, no model download); a too-small image is auto-padded by the descriptor (size_requirements multiple_of=8) and still returns a correctly-cropped same-size result
- INPAINT_MODELS entries are sha256-pinned; README/pyproject note Inpaint reuses the [face] extra

**Risks**
- LaMa Big-LaMa weights distribution: the canonical release is a zip of a Hydra checkpoint, not always a single spandrel-loadable .pt; implementer must source a spandrel-compatible single-file LaMa state_dict (these exist on community mirrors) and verify spandrel.ModelLoader loads it before pinning a sha256 — UNVERIFIED here (no download).
- gr.ImageEditor's return shape/keys vary across Gradio majors; on 6.15.2 it returns an EditorValue dict (background/layers/composite). The mask-derivation helper must be written against the actual 6.15.2 shape and unit-tested with a representative dict; exact layer alpha semantics were not runtime-verified.
- Deriving a clean binary mask from a soft/anti-aliased brush stroke needs a threshold; too low includes feathered edges. Spec uses alpha>0 -> 1, which is simple and testable, but quality tuning is left to implementation.
- Could NOT run real LaMa inference (no weights); end-to-end fill quality, CPU runtime and memory are unverified — only the spandrel masked-descriptor call contract was verified by reading source.

---

## Item 3C — CLI parity — `removebg`, `batch`, and `--face` on upscale  _(effort: M)_

Bring upscaler/cli.py to parity with the GUI. CONFIRMED current CLI surface (read cli.py:331-411 main + run_convert/run_pdf/run_video): subcommands convert/pdf/video are dispatched in main() by argv[0]; bare `upscaler <input>` is the upscaler with --scale/--model/--device/--deblur/--deblur-model/--sharpen/--tile/--fp16/--onnx/--list-models and folder-input via _gather_inputs; NO removebg, NO batch, NO --face flag exist today (grep confirmed). The GUI already has the logic: app.py enhance() takes face=/face_strength= and uses _get_face_restorer; remove_bg_ui calls background.remove_background; batch_process runs one op over many files. Item adds three CLI capabilities mirroring run_convert/run_video structure (build_*_parser + run_* + dispatch in main):\n(1) `--face` / `--face-strength` flags on build_parser(): after the existing upscale (and optional sharpen) in main()'s per-file loop, when args.face, lazily `from upscaler.face import FaceRestorer` and run restorer.restore(result, args.face_strength); the import sits inside the loop/function so torch-only users are unaffected, and FaceRestorer._deps() already raises the friendly 'pip install -e ".[face]"' RuntimeError — catch ImportError/RuntimeError and print that message to stderr + return 2 (matching the GUI's friendly behavior). Note: --face only applies to the default torch path (the GUI face stage runs torch GFPGAN); document that.\n(2) `removebg` subcommand: build_removebg_parser (input file-or-dir, -o file-or-dir, -m/--model from background.BG_MODELS, --feather int) + run_removebg mirroring run_convert: _gather_inputs, the same 'output must be a directory for a folder' guard, per-file background.remove_background then save .png, tqdm over multiple, stderr arrow on single; dispatched in main() via argv[0]=='removebg'. background.py is a light onnxruntime path (already a [gui]/[onnx] dep) so no [face] needed; wrap its RuntimeError/ValueError with a friendly stderr message.\n(3) `batch` subcommand: build_batch_parser (inputs dir/files, -o output dir, --op {upscale,convert,removebg}, plus op-specific flags: --model/--scale/--sharpen for upscale, --format/--quality for convert, --bg-model/--feather for removebg) + run_batch that loops resiliently (skip+count failures like batch_process) writing each result into the output dir; dispatched via argv[0]=='batch'. Reuse Upscaler/convert/background already imported. Update build_parser epilog + README CLI section to document the three additions.

**Files touched**
- upscaler/cli.py
- tests/test_cli_parity.py
- README.md

**New files**
- tests/test_cli_parity.py

**Tests**
- tests/test_cli_parity.py::test_removebg_single_file — monkeypatch background.remove_background to return a small RGBA image (no model/download); main(['removebg', src, '-o', dst]) == 0 and dst PNG exists with an alpha channel
- tests/test_cli_parity.py::test_removebg_dir_to_single_file_rejected — a directory of 2 images with -o pointing at a single .png returns 2 and prints 'must be a directory' (mirrors test_cli_convert_dir_to_single_file_is_rejected)
- tests/test_cli_parity.py::test_removebg_missing_input_errors — main(['removebg','/no/such.png']) == 2
- tests/test_cli_parity.py::test_batch_upscale_dir — monkeypatch upscaler.cli.Upscaler with a fake (like test_bare_input_still_routes_to_upscaler) ; main(['batch', dir, '-o', outdir, '--op','upscale','--scale','2']) processes all files and writes outputs, rc==0
- tests/test_cli_parity.py::test_batch_removebg_and_convert_ops — fake background.remove_background + real convert; assert each op routes correctly and writes the expected extensions; one unreadable file is skipped (rc still 0) proving resilience
- tests/test_cli_parity.py::test_face_flag_friendly_message_when_extra_missing — monkeypatch upscaler.face.FaceRestorer (or its import) to raise the [face] RuntimeError; main([src,'-o',dst,'--face']) returns 2 and stderr contains 'pip install -e \".[face]\"'
- tests/test_cli_parity.py::test_face_flag_invokes_restorer_when_available — monkeypatch a fake FaceRestorer whose .restore records it was called; with a fake Upscaler, main([src,'-o',dst,'--face','--face-strength','0.5']) calls restore(result,0.5) exactly once and rc==0
- tests/test_cli_parity.py::test_bare_upscale_without_face_does_not_import_face — with --face absent, run main on a fake Upscaler and assert upscaler.face is never imported (e.g. monkeypatch builtins import or assert sys.modules unaffected) — proves torch-only users aren't forced into [face]

**Acceptance criteria**
- `upscaler removebg <img> -o out.png` produces a transparent PNG; `upscaler removebg ./folder -o ./out` batches a directory; `upscaler removebg ./folder -o out.png` (single-file target for a folder) errors with rc 2 and 'must be a directory' — matching run_convert's guard
- `upscaler batch ./folder -o ./out --op upscale --scale 2` (and --op convert / --op removebg) runs the chosen operation over every image, skipping unreadable files without aborting the batch, exit 0 when at least one succeeds
- `upscaler <img> -o out.png --face` runs the upscale then GFPGAN face restoration when the [face] extra is installed (calls FaceRestorer.restore with --face-strength, default mirrors GUI 0.8 or 1.0); when the extra is missing it prints exactly the friendly 'pip install -e ".[face]"' message to stderr and returns 2 — no traceback
- All new CLI optional-extra imports (face) are LAZY (inside the run function / per-file loop), so a plain `pip install -e .` user running a normal upscale never imports spandrel/cv2 — verified by a test asserting upscaler.face stays unimported without --face
- `pytest tests/test_cli_parity.py` passes fully offline (all model work is monkeypatched); README CLI section + build_parser epilog document removebg, batch, and --face

**Risks**
- The GUI face stage runs only on the torch path; if a user combines --face with --onnx, the friendliest behavior is to still run torch GFPGAN on the ONNX-upscaled result (FaceRestorer is torch-based regardless). Spec keeps --face working on both backends by running the restorer on the final result, but this couples a torch import into the --onnx path only when --face is set; acceptable and documented.
- removebg uses onnxruntime (the [gui]/[onnx] extra). A truly minimal `pip install -e .` (no extras) install lacks onnxruntime, so `upscaler removebg` will fail its lazy onnxruntime import; the friendly message should point at the right extra ('.[onnx]' or '.[gui]') rather than '.[face]'. Implementer must choose the correct extra string for removebg (NOT [face]).
- argparse subcommand dispatch in main() is by argv[0] string match; adding 'removebg'/'batch' must not shadow a real filename also called removebg/batch — consistent with the existing convert/pdf/video design, but worth a one-line comment.
- batch's --op flag set must stay in sync with GUI _BATCH_OPS (Upscale/Convert/Remove background); divergence would confuse users. Spec mirrors the three GUI ops exactly.

---

## Verified facts (from reading the code)
- CLI current surface (upscaler/cli.py:331-411 main()): subcommands are dispatched by argv[0] for 'convert' (cli.py:334-335), 'pdf' (336-337), 'video' (338-339); bare input is the upscaler. Flags on build_parser (cli.py:286-328): input, -o/--output, -s/--scale{2,4}, -m/--model, --device, --deblur, --deblur-model, --sharpen, --tile, --fp16, --onnx, --list-models. NO removebg/batch/colorize/inpaint subcommand and NO --face flag exist (grep returned nothing for those in cli.py).
- upscaler/cli.py:24-27 _gather_inputs handles directory-vs-single-file; the 'output must be a directory when processing a folder' guard appears at cli.py:93-96 (convert), 249-252 (video), 363-366 (upscale) — the exact pattern run_removebg/run_batch must mirror.
- The GUI already implements the per-feature logic that the CLI lacks: app.py enhance() signature includes face=False, face_strength=1.0 (app.py:201-202) and runs _get_face_restorer(device).restore(result, face_strength) (app.py:225-227); remove_bg_ui calls background.remove_background (app.py:140-162); batch_process runs one of Upscale/Convert/Remove background over many files, skipping failures (app.py:421-491).
- _get_face_restorer (app.py:76-83) lazily does `from upscaler.face import FaceRestorer`; upscaler/face.py:31-44 _deps() raises RuntimeError('Face restoration needs extra packages. Install them with: pip install -e ".[face]"') — this is the exact friendly message string 3C must reuse for --face and 3A/3B for their deps.
- background.py is an onnxruntime path: upscaler/background.py:61-69 _session() does `import onnxruntime as ort` lazily and creates ort.InferenceSession(..., providers=['CPUExecutionProvider']); resize-in to spec.size then mask resized back to img.size (background.py:77,88); remove_background returns RGBA with putalpha (background.py:108-110). This is the model for a removebg CLI command and confirms removebg needs onnxruntime (an [onnx]/[gui] extra), NOT [face].
- ensure_weights (upscaler/models/weights.py:68-85) downloads lazily, verifies spec.sha256 (64-hex) and deletes+raises on mismatch; registry specs are frozen dataclasses with a sha256 field (registry.py:17-29 ModelSpec, 126-136 DeblurSpec, 193-220 FaceSpec/FACE_DETECTOR). New ColorizeSpec/InpaintSpec mirror FaceSpec.
- spandrel registry contains 52 archs including DDColor and LaMa: ran `spandrel_extra_arches.install(); spandrel.MAIN_REGISTRY` — output listed spandrel_extra_arches.architectures.DDColor.DDColorArch and spandrel.architectures.LaMa.LaMaArch (plus HAT, DRCT, SwinIR, FBCNN, GFPGAN, CodeFormer, etc.). spandrel + spandrel_extra_arches + cv2 + onnxruntime + torch + gradio are all INSTALLED in /Users/anamariaradulescu/Herd/Upscaler/.venv.
- spandrel DDColor descriptor (read spandrel_extra_arches/architectures/DDColor/__init__.py via inspect): DDColorArch.load returns ImageModelDescriptor(..., purpose='Restoration', supports_half=False, scale=1, input_channels=1, output_channels=3, tiling=INTERNAL, call_fn=_call). The custom _call REQUIRES model.num_output_channels==2 (raises ValueError otherwise), takes input shape (1,1,H,W), builds L from input, predicts 2-channel AB at model.input_size, F.interpolate AB back to (1,2,H,W), concatenates the ORIGINAL orig_l with AB (output_lab = hstack([orig_l, output_ab_resize])), and lab_to_rgb -> returns (1,3,H,W). So 'predict chroma + merge with source luminance' is done inside spandrel; our code feeds the L channel and gets RGB. => torch/spandrel path, not onnxruntime.
- spandrel LaMa descriptor (read spandrel/architectures/LaMa/__init__.py via inspect): LaMaArch.load returns MaskedImageModelDescriptor(..., purpose='Inpainting', supports_half=False, input_channels=in_nc-1 (=3), output_channels=out_nc (=3), size_requirements=SizeRequirements(minimum=16, multiple_of=8)). MaskedImageModelDescriptor.__call__(image, mask): image is (1,3,H,W) float[0,1]; mask is (1,1,H,W) with values 0=keep/1=inpaint; returns (1,3,H,W) clamped [0,1]; auto-pads to size_requirements then crops result. => two-argument call, distinct from upscale/colorize single-arg.
- spandrel public API surface confirmed: `spandrel` exposes ModelLoader, ImageModelDescriptor, and MaskedImageModelDescriptor (dir(spandrel) check). ModelLoader.load_from_file(path) and load_from_state_dict(sd) both return a ModelDescriptor. Descriptors expose .to/.eval/.device/.purpose (matches the load_from_file(str(ensure_weights(spec))).to(device).eval() pattern already used in upscaler/face.py:63-64).
- Gradio version is 6.15.2 and exposes gr.ImageEditor, gr.ImageMask, gr.ImageSlider, gr.Brush (all True) — so the brush-mask Inpaint UI and the before/after ImageSlider used elsewhere (app.py:1370 gr.ImageSlider for Upscale) are available.
- Tab structure (app.py build_demo, ~1225-1971): gr.Tabs() contains Tab('Upscale'), Tab('Video'), Tab('Convert'), Tab('Remove BG') (app.py:1578), Tab('Batch') (1630), Tab('Lian Li Screen'), Tab('Library'); Settings is a separate hidden Column toggled by the gear. New Colorize/Inpaint tabs slot in after 'Remove BG'. Remove-BG wiring (app.py:2033-2038) and _BG_CHOICES (app.py:137) are the direct template for the new tabs' handler + choices + .click wiring.
- Existing test conventions: optional-dep tests use pytest.importorskip('cv2'/'spandrel'/'spandrel_extra_arches'/'onnxruntime') and skip offline (tests/test_face.py:18,35-42; tests/test_highsev_fixes.py:32-33); registry-pinning tests assert len(sha256)==64 (tests/test_weights.py:9-12, tests/test_face.py:10-13); CLI tests call cli.main([...]) and monkeypatch Upscaler with a fake (tests/test_cli_convert.py:49-68); the dir-to-single-file rejection assertion 'must be a directory' (tests/test_highsev_fixes.py:63-74). New tests mirror these exactly.
- pyproject.toml [project.optional-dependencies] face = ['spandrel>=0.4','spandrel_extra_arches>=0.2','opencv-python-headless>=4.9']; onnx = ['onnxruntime>=1.16',...]; gui includes onnxruntime>=1.16. So Colorize/Inpaint correctly reuse [face]; removebg correctly needs [onnx]/[gui] (onnxruntime), confirming 3C's removebg friendly-message must NOT say [face].
- library auto-save pattern: library.save_image(img, kind) and library.save_path(path, kind) name files <kind>_<timestamp><ext> (upscaler/library.py:53-74); GUI handlers call library.save_image(result,'upscale') etc. New handlers use kinds 'colorize' and 'inpaint'. list_items() (library.py:77-96) classifies by extension for the Library tab.

## Open questions
- Which exact DDColor checkpoint to pin? spandrel's _call requires a 2-output-channel (AB) DDColor; the common candidates are ddcolor_modelscope / ddcolor_paper / ddcolor_artistic. Need a stable mirror URL + real sha256 (run scripts/print_checksums.py after a one-time download — not possible offline here). Also confirm the chosen .pth loads as num_output_channels==2.
- Which LaMa checkpoint to pin? The canonical Big-LaMa release is a zipped Hydra checkpoint, not always a single spandrel-loadable state_dict. Need to confirm a single-file LaMa .pt/.ckpt that spandrel.ModelLoader.load_from_file accepts, plus a stable host and sha256. UNVERIFIED offline.
- Exact gr.ImageEditor (6.15.2) EditorValue return shape and the painted-layer alpha semantics for deriving the binary mask — needs a quick runtime check in the running app to lock the mask-derivation helper (spec assumes background + layers with alpha>0 = painted).
- Default --face-strength for the CLI: GUI default is 0.8 (app.py:1322). Confirm whether CLI should default to 0.8 (match GUI) or 1.0 (full). Spec leaves it as a small decision; recommend 0.8 for parity.
- Should Colorize/Inpaint expose a device dropdown (auto/cpu/cuda/mps) like other tabs, or default to 'auto' silently? DDColor/LaMa are heavy on CPU; a device control is consistent with the app but adds UI. Not blocking — recommend a single 'auto' default plus an Advanced device dropdown to match the Upscale/Remove-BG ergonomics.
- Could NOT run real DDColor or LaMa inference (no weights downloaded, kept offline), so output quality, CPU runtime, and peak memory are unverified — only the spandrel call contracts (input/output tensor shapes and the L-merge/mask semantics) were verified by reading spandrel source.
