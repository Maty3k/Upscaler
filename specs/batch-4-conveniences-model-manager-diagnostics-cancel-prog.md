# Batch 4 — Conveniences — Model Manager + Diagnostics + Cancel/progress

**Scope:** Batch 4

## Goal
Add three local-only conveniences to the Gradio toolbox: a Model Manager, a Diagnostics report, and Cancel/progress on long jobs.

## Dependencies
- 4B depends on 4A: system_report() reuses manage.total_bytes()/human_size() and the weights.WEIGHTS_DIR plumbing introduced in 4A, so 4A's manage.py must land first (or both ship in the same manage.py).
- 4A and 4B both add cases to tests/test_manage.py — land them together to avoid file churn.
- 4C is independent of 4A/4B (touches engine.py + the Upscale/Batch click wiring), but all three modify app.py's build_demo, so coordinate the single app.py edit to avoid conflicts.
- All three are gated by the established workflow: feature -> pytest -> branch -> commit -> PR.

---

## Item 4A — In-app Model Manager in Settings (list / size / total / pre-download / remove)  _(effort: M)_

Add a torch-free, gradio-free helper module upscaler/manage.py that enumerates every weight spec across the four registries (MODELS, DEBLUR_MODELS, FACE_MODELS + FACE_DETECTOR, background.BG_MODELS) and reports, per spec: registry group, name, filename, notes, whether the file is present under weights.WEIGHTS_DIR, and its on-disk size. It also computes total disk usage of the weights dir and exposes download_one(filename)->str and remove_one(filename)->str operations. Then wire a 'Models & downloads' accordion into the existing Settings page (app.py settings_view, the gr.Column at app.py:1973) using a gr.Dataframe for the listing plus a refresh button, a per-action dropdown of filenames, a 'Download selected' button (calls manage.download_one which calls ensure_weights), a 'Remove selected' button (calls manage.remove_one which unlinks in WEIGHTS_DIR), and a Markdown total/status line.

WHY a new module: app.py imports torch via `from upscaler.engine import Upscaler, resolve_device` (app.py:28) and `from upscaler.deblur import Deblurrer` (app.py:27), and upscaler/__init__.py imports torch transitively (upscaler/__init__.py:3-5). torch is NOT installed in this env, so any logic placed in app.py cannot be unit-tested. upscaler/models/weights.py, upscaler/models/registry.py and upscaler/background.py are all torch-free (verified: none import torch), so manage.py can import them and be tested with monkeypatch.setattr(weights,'WEIGHTS_DIR',tmp_path) exactly like tests/test_weights.py:16.

DATA SHAPE: define a small dataclass ManagedSpec(group:str, name:str, filename:str, present:bool, size_bytes:int, url:str, notes:str, spec:object). manage.list_specs() returns list[ManagedSpec] built by iterating: MODELS.values() (group 'Upscale'), DEBLUR_MODELS.values() (group 'Clean-up'), FACE_MODELS.values()+[FACE_DETECTOR] (group 'Faces'), background.BG_MODELS.values() (group 'Background'). Each item resolves dest = weights.WEIGHTS_DIR / spec.filename, present = dest.is_file(), size_bytes = dest.stat().st_size if present else 0 (guard OSError -> 0). manage.total_bytes() sums size of every *.pth/*.onnx file actually in WEIGHTS_DIR (not just registered ones, so orphaned/partial files count). manage.human_size(n) formats B/KB/MB/GB. download_one(filename) maps filename->spec via a {s.filename:s for s in all specs} dict and calls weights.ensure_weights(spec); returns a status string; surfaces RuntimeError message from ensure_weights as the status (ensure_weights already raises friendly RuntimeError on network/checksum failure, weights.py:62-84). remove_one(filename) resolves WEIGHTS_DIR/filename, unlink(missing_ok=True), returns status; refuses paths that escape WEIGHTS_DIR (resolve() and check is_relative_to) and refuses filenames not in the registry to avoid arbitrary deletes.

UI: in the Settings left column add `with gr.Accordion("Models & downloads", open=False):` containing a gr.Dataframe(headers=["Group","Model","File","Status","Size"], interactive=False) seeded on build and refreshed, a gr.Markdown for total usage, a gr.Dropdown(choices=filenames) to pick a model, a 'Download selected' primary button, a 'Remove selected' secondary button, and a gr.Markdown status line. Add app.py handlers _mm_rows() (returns dataframe rows + total markdown + dropdown choices update), _mm_download(filename) and _mm_remove(filename) (call manage.* then return _mm_rows() outputs + an action status string). Wire .click on the two buttons and a refresh button; also refresh the table when the Settings page is opened (extend the existing settings_btn.click lambda at app.py:2121 to also output the table, or add a dedicated refresh button). Note FACE_MODELS/BG_MODELS are NOT currently imported in app.py — the table is built in manage.py so app.py only needs `from upscaler import manage`.

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/app.py

**New files**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/manage.py
- /Users/anamariaradulescu/Herd/Upscaler/tests/test_manage.py

**Registry / data additions**
- No registry entries added; manage.py READS MODELS/DEBLUR_MODELS/FACE_MODELS/FACE_DETECTOR/BG_MODELS. BG_MODELS specs have no sha256 (background.py:36-49) so download still works (ensure_weights only verifies when spec.sha256 is set, weights.py:77).

**UI changes**
- New 'Models & downloads' accordion in the Settings page left column (inside the gr.Column at app.py:1973-2022), placed after the 'Where your files live' accordion (app.py:2007): a non-interactive gr.Dataframe listing every model with Group/Model/File/Status/Size, a total-disk-usage Markdown, a filename Dropdown, Download/Remove buttons, and a status Markdown.

**Tests**
- tests/test_manage.py::test_list_specs_covers_all_four_registries — monkeypatch weights.WEIGHTS_DIR to tmp_path; assert {s.group for s in manage.list_specs()} == {'Upscale','Clean-up','Faces','Background'} and len(list_specs) == len(MODELS)+len(DEBLUR_MODELS)+len(FACE_MODELS)+1+len(BG_MODELS) (the +1 is FACE_DETECTOR).
- tests/test_manage.py::test_present_and_size_reported — monkeypatch WEIGHTS_DIR to tmp_path, write tmp_path/'RealESRGAN_x4plus.pth' with 1234 bytes; find that spec in list_specs(); assert present is True and size_bytes == 1234; assert an absent spec has present False and size_bytes 0.
- tests/test_manage.py::test_total_bytes_sums_weight_files — write two files (one .pth 1000B, one .onnx 500B, plus a stray .part that must be ignored or counted per spec) into tmp_path; assert manage.total_bytes() == 1500 for the two weight files.
- tests/test_manage.py::test_human_size_formats — assert human_size(0)=='0 B', human_size(2048) endswith 'KB', human_size(1572864) endswith 'MB'.
- tests/test_manage.py::test_remove_one_unlinks_and_is_safe — write tmp_path/'u2net.onnx'; manage.remove_one('u2net.onnx') -> file gone, status mentions 'Removed'; remove_one('../evil') and remove_one('not_a_registered_file.pth') raise/return refusal WITHOUT touching anything outside WEIGHTS_DIR.
- tests/test_manage.py::test_download_one_calls_ensure_weights — monkeypatch manage's ensure_weights (or weights.ensure_weights) with a fake that creates the dest file and records the spec; call manage.download_one('u2net.onnx'); assert the fake was called with the BGSpec for u2net and status mentions success.
- tests/test_manage.py::test_download_one_surfaces_failure — monkeypatch ensure_weights to raise RuntimeError('Couldn't download…'); assert download_one returns a status string containing the message and does NOT raise.

**Acceptance criteria**
- upscaler/manage.py imports cleanly WITHOUT torch or gradio installed (verified constraint: torch and gradio are absent in this env; weights/registry/background are torch-free).
- manage.list_specs() returns one entry per spec across all four registries plus the YuNet FACE_DETECTOR (total == len(MODELS)+len(DEBLUR_MODELS)+len(FACE_MODELS)+1+len(BG_MODELS)), each tagged with the right group string.
- For a spec whose file exists under WEIGHTS_DIR, present is True and size_bytes equals the real file size; for an absent file present is False and size_bytes is 0; OSError while stat-ing is swallowed to 0.
- manage.total_bytes() returns the summed size of weight files in WEIGHTS_DIR and human_size renders B/KB/MB/GB.
- download_one(filename) routes the filename to the correct spec and calls weights.ensure_weights; a download/checksum failure is caught and returned as a friendly status string (never an unhandled traceback).
- remove_one(filename) deletes only files inside WEIGHTS_DIR that correspond to a registered filename; path-traversal or unknown filenames are refused without deleting anything.
- In the running app the Settings page shows the Models table, the total disk usage, and Download/Remove actions that update the table after each action (manual check via the verify/run skill if torch+gradio are installed).

**Risks**
- Cannot run the full app in this env (no torch, no gradio), so the Gradio wiring (gr.Dataframe shape, dropdown refresh) is unverified at spec time — only manage.py is unit-tested. Reviewer should run `python app.py` with the [face]/[gui] extras to confirm the accordion renders.
- BG_MODELS specs lack sha256 (background.py:36-49), so a Download of u2net/u2netp is integrity-UNverified — acceptable (matches current behavior) but worth a one-line note in the UI status.
- Removing a model that is currently cached in app's in-memory _UP_CACHE/_DB_CACHE/_FACE_CACHE (app.py:34-36) only deletes the file, not the loaded net; a re-run will re-download. Acceptable; mention in acceptance that remove affects disk only.
- Gradio gr.Dataframe value must be a list-of-lists (or pandas); ensure rows are plain str/str so it renders without pandas.

---

## Item 4B — Telemetry-free Diagnostics 'Copy system report' panel in Settings  _(effort: S)_

Add manage.system_report()->str (torch-free-safe: torch and any optional dep are probed defensively) that builds a plain-text, copy-pasteable report and render it in Settings as a gr.Code box with Gradio's built-in copy button. The report includes ONLY local, non-identifying system facts — no telemetry, no network calls.

CONTENT (each line best-effort, never raises):
- App: upscaler.__version__ if importable else 'unknown' (guard ImportError since __init__ imports torch).
- Platform: platform.platform(), platform.python_version(), platform.machine().
- Optional deps via importlib.util.find_spec for: torch, torchvision, gradio, onnxruntime, onnx, spandrel, spandrel_extra_arches, cv2, pypdfium2, pillow_heif, imageio_ffmpeg, numpy, PIL — print 'present'/'missing' for each (find_spec must be wrapped in try/except since some packages raise on probe).
- ffmpeg: shutil.which('ffmpeg') and shutil.which('ffprobe') -> path or 'not found'.
- Torch/devices: only if torch is importable — torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available() (guarded getattr exactly like engine.resolve_device, engine.py:27), and the resolved device from engine.resolve_device('auto').type. If torch is missing, print 'torch: missing — running report without device probe'. Import engine lazily inside the function so manage.py stays torch-free at module import.
- Paths & space: weights.WEIGHTS_DIR, whether it exists, total weight bytes (manage.total_bytes via human_size), library.LIBRARY_DIR, config.CONFIG_PATH, and shutil.disk_usage(WEIGHTS_DIR.parent or home).free formatted with human_size.
- Env overrides actually in effect: UPSCALER_WEIGHTS_DIR, UPSCALER_LIBRARY, UPSCALER_CONFIG, UPSCALER_PORT (print the value or '(default)').

UI: in the Settings RIGHT column (next to the Windows guide, app.py:2023-2025) or as a new accordion in the left column, add `with gr.Accordion("Diagnostics", open=False):` containing gr.Code(value=manage.system_report(), language=None, label='System report', interactive=False) — gr.Code shows a copy icon by default — plus a 'Refresh report' button whose .click returns gr.update(value=manage.system_report()). Add a one-line note: 'Nothing here is uploaded — copy and paste it into a bug report yourself.'

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/app.py

**UI changes**
- New 'Diagnostics' accordion in the Settings page (app.py settings_view) with a read-only gr.Code box (built-in copy button) and a 'Refresh report' button. Placed in the Settings layout (left column after Models, or right column under the Windows guide).

**Tests**
- tests/test_manage.py::test_system_report_is_string_with_sections — call manage.system_report(); assert it's a non-empty str containing 'Platform', 'Python', 'ffmpeg', 'Weights dir', and at least one of the optional-dep names (e.g. 'gradio').
- tests/test_manage.py::test_system_report_no_torch_is_safe — it must not raise even though torch is not installed in this env; assert 'torch' appears and the report contains either 'missing' or a version, and does NOT raise ImportError.
- tests/test_manage.py::test_system_report_reports_missing_dep — pick a dep guaranteed absent (e.g. monkeypatch importlib.util.find_spec to return None for a chosen name, or assert a known-absent package like 'spandrel' shows 'missing') and assert that name is annotated missing.
- tests/test_manage.py::test_system_report_honors_env_dir — monkeypatch weights.WEIGHTS_DIR to tmp_path; assert str(tmp_path) appears in the report so the report reflects the live weights dir.
- tests/test_manage.py::test_system_report_no_network — (defensive) monkeypatch urllib.request.urlopen to raise; assert system_report() still succeeds, proving it makes no network calls.

**Acceptance criteria**
- manage.system_report() returns a single multi-line string and NEVER raises, even with torch/gradio/optional deps missing (each probe is individually guarded).
- The report contains: app version (or 'unknown'), platform + python + machine, find_spec present/missing for the listed optional deps, ffmpeg/ffprobe which() result, torch/CUDA/MPS/resolved-device when torch is present (gracefully skipped when absent), weights dir + total weight size + free disk space, library and config paths, and the in-effect UPSCALER_* env overrides.
- No telemetry and no network: the function performs zero outbound requests (proven by the urlopen-raises test).
- In the running app the Diagnostics accordion shows the report in a gr.Code box with a working copy button and a Refresh button that regenerates it.
- manage.py still imports without torch (the engine/resolve_device call is lazy inside system_report).

**Risks**
- engine.resolve_device must be imported lazily inside system_report to keep manage.py torch-free at import time; getting that wrong would break 4A's tests too.
- importlib.util.find_spec can itself raise (e.g. for namespace/partial packages) — every probe must be individually try/excepted.
- shutil.disk_usage needs an existing path; if WEIGHTS_DIR doesn't exist yet, fall back to WEIGHTS_DIR.parent or Path.home().
- gr.Code copy-button behavior is a Gradio version detail; unverified here because gradio isn't installed — reviewer confirms in-app.

---

## Item 4C — Cancel button + real per-tile/per-image progress on Enhance and Batch jobs  _(effort: M)_

Thread an optional progress callback through the upscaler so the long image-Enhance and Batch jobs report real progress and can be cancelled.

ENGINE (upscaler/engine.py): add an optional `progress_cb: Optional[Callable[[int,int],None]] = None` param to Upscaler.upscale (engine.py:120) and to _run_tiled (engine.py:156). In _run_tiled, the total tile count is n_x*n_y (computed at engine.py:165-166); after finishing each tile call progress_cb(done, n_x*n_y). When tiling is off (self.tile<=0, the `else self._net(x)` branch at engine.py:129) call progress_cb(1,1) at the end. upscale() forwards progress_cb to _run_tiled. Keep the signature backward-compatible (default None) so existing callers — video.py:131 `up.upscale(Image.open(fr))`, panel_enhance_source (app.py:629), batch_process (app.py:442), cli — are unaffected. Cooperative cancel: also accept `should_cancel: Optional[Callable[[],bool]] = None`; check it at the top of each tile loop iteration and raise a dedicated CancelledError (define `class CancelledError(RuntimeError)` in engine.py) so callers can distinguish a user cancel from a real failure.

APP — Enhance (app.py:201 enhance, wired at app.py:2042): add `progress=gr.Progress()` param (matching remove_bg_ui at app.py:140). Build a tile callback `def cb(done,total): progress(done/max(total,1), desc=f'Upscaling tile {done}/{total}')` and pass progress_cb=cb into up.upscale (the call at app.py:217). Wrap the up.upscale call so engine.CancelledError is caught and turned into a gr.Error('Cancelled.') or a soft return.

APP — Batch (app.py:421 batch_process, wired at app.py:1700): already calls progress(i/n, ...) per file (app.py:437); leave that, but make each upscale forward progress_cb for sub-progress is optional. The cancel story for batch is the cooperative flag.

CANCEL UI: add a Cancel button next to Enhance (app.py:1364-1368 row with `run`/`clear`) and next to Batch's 'Process all' (app.py:1687). In build_demo capture the click event objects: `run_evt = run.click(enhance, ...)` (app.py:2042) and `batch_evt = batch_run.click(...)` (app.py:1700), then wire `cancel_btn.click(None, None, None, cancels=[run_evt])` and `batch_cancel.click(None,None,None,cancels=[batch_evt])`. Gradio's cancels= aborts the running generator/event and is the supported mechanism. Because cancels= terminates the worker, the cooperative should_cancel flag in the engine is a secondary nicety; primary cancel is the gradio cancels= wiring. NOTE: cancels= requires the cancelled event to have been assigned to a variable (the .click(...) return value) — restructure the two .click calls to capture their return value.

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/engine.py
- /Users/anamariaradulescu/Herd/Upscaler/app.py

**UI changes**
- Add a '✕ Cancel' secondary button in the Enhance button row (app.py:1364-1368, alongside Enhance/Clear).
- Add a '✕ Cancel' secondary button next to Batch 'Process all' (app.py:1687).
- Enhance now shows real per-tile progress text ('Upscaling tile k/N') via gr.Progress instead of a generic spinner; Batch keeps per-file progress.

**Tests**
- tests/test_engine_progress.py::test_progress_cb_called_per_tile — build a fake Upscaler instance OR (since torch is absent) test the progress arithmetic in isolation: construct a tiny stand-in that calls _run_tiled logic with a recording cb. PRACTICAL approach without torch: factor the tile-count math is already n_x*n_y; add a pure helper engine.tile_count(w,h,tile)->int and unit-test it: tile_count(100,100,512)==1, tile_count(1000,1000,512)==4, tile_count(0..) edge. Then a separate test (skipped when torch missing via pytest.importorskip('torch')) builds a real Upscaler on a tiny image and asserts the cb's final call equals (n,n) and is monotonic.
- tests/test_engine_progress.py::test_tile_count_helper — assert engine.tile_count matches (w+tile-1)//tile * (h+tile-1)//tile for several sizes, and returns 1 when tile<=0.
- tests/test_engine_progress.py::test_cancelled_error_is_runtimeerror — assert issubclass(engine.CancelledError, RuntimeError) so existing `except RuntimeError` handlers in app.py (e.g. enhance app.py:218) still catch it gracefully.
- tests/test_engine_progress.py::test_upscale_forwards_progress_cb (torch-gated) — pytest.importorskip('torch'); patch Upscaler to skip weight load (or use a monkeypatched net that returns input) and assert progress_cb is called with increasing done up to total, and that should_cancel raising CancelledError stops the loop early.
- tests/test_manage.py / existing tests still pass — run the full suite to confirm the new engine kwargs are backward compatible (video.py, batch_process, cli unaffected since defaults are None).

**Acceptance criteria**
- engine.Upscaler.upscale and engine._run_tiled accept progress_cb (and should_cancel) with default None; ALL existing callers (upscaler/video.py:131, app.py:629/442, cli) continue to work unchanged.
- When progress_cb is supplied and tiling is on, it is called once per tile with monotonically increasing done from 1..n_x*n_y where n=n_x*n_y; when tiling is off it is called once with (1,1).
- A new engine.CancelledError subclasses RuntimeError; raising it from the tile loop is caught by app.py's existing `except RuntimeError` paths without showing a scary traceback.
- The Enhance tab shows live 'Upscaling tile k/N' progress (via gr.Progress) and has a Cancel button that aborts the running job via Gradio cancels= (the run.click event is captured into a variable and referenced in cancel_btn.click(cancels=[...])).
- The Batch tab has a Cancel button wired to cancels=[batch_evt] that stops the run; existing per-file progress (app.py:437) still displays.
- engine.tile_count(w,h,tile) is a pure, torch-free helper returning the exact tile count used by _run_tiled, and the suite passes with torch absent (torch-gated tests skip cleanly).

**Risks**
- Gradio cancels= only works when the .click(...) return value (the event) is captured and the server runs with a worker that supports cancellation; behavior is version-dependent and UNVERIFIED here (gradio not installed). Reviewer must confirm Cancel actually interrupts a real upscale in-app.
- cancels= aborts the whole event; the in-memory model cache (_UP_CACHE) and any half-written Library file from a cancelled batch could leave partial state — batch_process already writes per-file and skips failures, so a cancel mid-batch loses only the in-flight ZIP. Acceptable but note it.
- Threading should_cancel through requires the engine to check it inside @torch.inference_mode(); raising there is fine but must occur before/after a full tile, not mid-net, to avoid leaving GPU state. Tile-boundary checks are sufficient.
- gr.Progress's desc updates per tile could be chatty for many tiles on large images; that's cosmetic.
- Cannot run torch-dependent engine tests here, so the per-tile callback is verified only via the pure tile_count helper plus a torch-gated test that will run in a full CI env.

---

## Verified facts (from reading the code)
- weights.WEIGHTS_DIR is `Path(os.environ.get('UPSCALER_WEIGHTS_DIR', <pkg>/weights))` and ensure_weights(spec) downloads to WEIGHTS_DIR/spec.filename, verifies sha256 only when set, and re-raises network errors as a friendly RuntimeError — upscaler/models/weights.py:21-23, 68-85, 62-65.
- ensure_weights uses dest.unlink(missing_ok=True) on checksum mismatch (weights.py:80); the remove pattern (unlink) is consistent with 4A's remove_one.
- registry.py defines four spec containers: MODELS (8 entries, all with sha256) at registry.py:37-98; DEBLUR_MODELS (3, all sha256) at registry.py:141-177; FACE_MODELS (1: gfpgan-v1.4, sha256) at registry.py:202-210; FACE_DETECTOR (YuNet, sha256) at registry.py:214-220. ModelSpec/DeblurSpec/FaceSpec all expose .name/.url/.filename/.sha256/.notes (registry.py:18-29,127-136,194-199).
- background.BG_MODELS holds 2 specs (u2net, u2netp) and they have NO sha256 (BGSpec defaults sha256=None and the entries omit it) — upscaler/background.py:35-50, 23-30; so a Model-Manager download of these is integrity-unverified but still works (ensure_weights verifies only when spec.sha256 truthy).
- upscaler/models/weights.py, upscaler/models/registry.py, and upscaler/background.py do NOT import torch (grep -l 'import torch' returned none), so a new manage.py importing them is torch-free.
- upscaler/__init__.py imports torch transitively via `from upscaler.deblur import Deblurrer` and `from upscaler.engine import Upscaler` (__init__.py:3-5); deblur.py imports torch (deblur.py:14); engine.py imports torch (engine.py:9). app.py imports both (app.py:27-28). Therefore app.py cannot be imported without torch.
- Neither torch nor gradio is installed in this environment (python -c 'import torch' and 'import gradio' both fail), so unit tests for the new feature must avoid importing app.py/engine/deblur and instead test manage.py + a pure engine helper; existing tests already gate heavy imports inside functions (tests/test_batch_and_mockup.py:30 `import app` inside the test; tests/test_face.py uses pytest.importorskip).
- config.py: CONFIG_PATH = Path(os.environ.get('UPSCALER_CONFIG', ~/.upscaler/config.json)), DEFAULTS = {device:'auto', model:'realesrgan-x4plus', output_dir:''}, load() merges saved over DEFAULTS — config.py:15-36.
- library.LIBRARY_DIR = Path(os.environ.get('UPSCALER_LIBRARY', ~/.upscaler/library)) — library.py:20-22.
- engine.resolve_device('auto') prefers cuda, then guarded mps via getattr(torch.backends,'mps',None) and mps.is_available(), else cpu — engine.py:18-29; this is the exact device-probe pattern Diagnostics (4B) must reuse, lazily.
- engine._run_tiled computes n_x=(w+tile-1)//tile and n_y=(h+tile-1)//tile (engine.py:165-166) and loops ty in range(n_y), tx in range(n_x) (engine.py:168-169); total tiles = n_x*n_y — this is the denominator for 4C progress. upscale() chooses `self._run_tiled(x) if self.tile>0 else self._net(x)` (engine.py:129).
- Upscaler.upscale signature is `def upscale(self, image)` with @torch.inference_mode() (engine.py:120-121); adding progress_cb/should_cancel with default None is backward-compatible.
- video.upscale_video already defines ProgressCb = Optional[Callable[[int,int],None]] and calls progress_cb(i, len(frames)) per frame (video.py:25, 88, 135-136); app.upscale_video_ui builds `def cb(i,n): progress(i/n, desc=...)` and passes progress_cb=cb (app.py:357-365) — the established progress pattern 4C should mirror for tiles.
- remove_bg_ui shows the gr.Progress pattern: `def remove_bg_ui(image, model, feather, progress=gr.Progress())` then progress(0.2, desc=...) (app.py:140,144,153) — the template for adding progress to enhance().
- enhance() is at app.py:201, wired by run.click(enhance, [...], [out, info], show_progress_on=[out]) at app.py:2042-2048; the up.upscale call is app.py:217; its except catches (RuntimeError, AssertionError, OSError, ValueError) at app.py:218 — so a CancelledError(RuntimeError) would be caught here.
- batch_process is at app.py:421, already takes progress=gr.Progress() and calls progress(i/n, desc=f'{op} · {i+1}/{n}') (app.py:422,437); wired by batch_run.click(batch_process,[...],[batch_gallery,batch_zip,batch_info], show_progress_on=[batch_gallery]) at app.py:1700-1707. The Enhance button row is app.py:1364-1368 (run + clear); Batch 'Process all' button is app.py:1687.
- The Settings page is a hidden gr.Column(visible=False) named settings_view at app.py:1973, opened by settings_btn.click toggling main_tabs/settings_view (app.py:2121-2128), with save_settings wired at app.py:2129 and the 'Where your files live' accordion (showing library.LIBRARY_DIR and config.CONFIG_PATH) at app.py:2007-2013 — the natural place to insert the Model Manager (4A) and Diagnostics (4B) accordions.
- open_library_folder() (app.py:756-771) already does per-OS reveal via subprocess open/xdg-open/os.startfile — the precedent for any OS-specific file action; remove_one in 4A should NOT shell out, it just unlinks.
- Existing test convention for tmp weights dir: tests/test_weights.py:16 uses monkeypatch.setattr(weights,'WEIGHTS_DIR', tmp_path); tests/test_config.py:7 monkeypatches config.CONFIG_PATH; tests/test_batch_and_mockup.py:29 monkeypatches library.LIBRARY_DIR — 4A/4B tests follow these.
- pyproject [face] extra = spandrel>=0.4, spandrel_extra_arches>=0.2, opencv-python-headless>=4.9; [gui] includes gradio>=4.0; pytest testpaths=['tests'] — pyproject.toml face/gui extras and [tool.pytest.ini_options].

## Open questions
- Does the installed Gradio version support cancels= and gr.Code's built-in copy button as assumed? Could not verify — gradio is not installed in this dev env. Reviewer must confirm in-app with the [gui] extra.
- Should the Model Manager 'Remove' also evict the in-memory model caches (_UP_CACHE/_DB_CACHE/_FACE_CACHE in app.py:34-36)? Spec treats remove as disk-only; flag if product wants cache eviction too.
- Should Diagnostics include GPU model/VRAM (e.g. torch.cuda.get_device_name)? Left out to stay minimal and avoid driver-dependent failures; easy to add behind a guard if wanted.
- Is per-tile sub-progress desired for Batch's individual images, or is per-file progress (already present) enough? Spec keeps per-file only for Batch to limit churn.
