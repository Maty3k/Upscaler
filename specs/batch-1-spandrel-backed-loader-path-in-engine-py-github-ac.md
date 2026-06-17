# Batch 1 — Spandrel-backed loader path in engine.py + GitHub Actions CI safety net

**Scope:** Batch 1: Foundation — arch-agnostic model loader + CI safety net

## Goal
Let the EXISTING tiler (`_run_tiled`/`_net`) drive any spandrel-registered architecture (HAT, DRCT, SwinIR, etc.) through `Upscaler`, while keeping the native RRDBNet path byte-for-byte unchanged, then lock the project behind a macOS+Ubuntu CI that runs pytest, imports the app, lists models, and adds a registry-integrity guard plus a build_demo smoke test.

## Dependencies
- 1A and 1B are independent and can be built/committed in parallel; both should land before any later batch that registers a real spandrel arch (HAT/DRCT/SwinIR) since that batch depends on 1A's `loader='spandrel'` path and 1B's registry guard.
- 1B's CI (run pytest) will execute 1A's new tests once both are merged — no hard code dependency, but merging 1A first lets CI exercise it immediately.

---

## Item 1A — Add a spandrel loader branch to Upscaler so the existing tiler drives any spandrel arch  _(effort: M)_

Add an optional opt-in field to ModelSpec (e.g. `loader: str = "rrdbnet"`, allowed values "rrdbnet"|"spandrel") in upscaler/models/registry.py. In Upscaler.__init__ (upscaler/engine.py L99-112) branch on spec.loader: the existing RRDBNet construction stays the DEFAULT path untouched; a new `_load_spandrel()` helper installs spandrel_extra_arches, calls `spandrel.ModelLoader().load_from_file(str(ensure_weights(self.spec)))`, moves the descriptor to self.device, sets .eval(), and stores it as self.net. Discover scale from the loaded model rather than the spec: introduce `self._scale` set to `descriptor.scale` for the spandrel path (and `self.spec.scale` for native), and make the `scale` property (L114-116) return `self._scale` so `_run_tiled`/`_net` keep working. The spandrel descriptor is itself callable (`descriptor(tensor) -> tensor`, verified: spandrel 0.4.2 ImageModelDescriptor.__call__(self, image)), so `_run_tiled` (which calls `self._net`) needs only that `self.net(t)` invokes the descriptor — already true. Handle fp16 by ANDing the existing CUDA-only gate with `descriptor.supports_half` (do NOT call .half() if unsupported). Handle `_net` padding: the native RRDBNet padding logic (L140 `m = 2 if scale==2 else 4 if scale==1 else 1`) is RRDBNet-specific; for the spandrel path use `descriptor.size_requirements` to compute the pad multiple instead (or a conservative `max(size_req.multiple_of, 1)` / fall back to no padding when size_requirements is None) — keep native behavior identical by gating on spec.loader. Validate input/output channels are 3 (raise a clear RuntimeError otherwise) since the tiler assumes 3-channel RGB. Do NOT register any new model in this batch — add ONE example/proof spec is out of scope; just make the path selectable and covered by a construction test using a fake/tiny descriptor (monkeypatched) so no weights download.

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/engine.py
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/models/registry.py

**Registry / data additions**
- ModelSpec gains `loader: str = "rrdbnet"` field (default keeps every existing spec on the native RRDBNet path with zero behavior change)

**Tests**
- tests/test_spandrel_loader.py::test_native_path_unchanged — construct Upscaler for an existing RRDBNet spec via the test's existing __new__ bypass pattern (see tests/test_architecture.py L43-56) is not enough; instead assert `resolve_model('realesrgan-x4plus').loader == 'rrdbnet'` and that the default field exists on ModelSpec.
- tests/test_spandrel_loader.py::test_spandrel_branch_builds_and_uses_descriptor_scale — monkeypatch a fake `spandrel.ModelLoader` whose `load_from_file` returns a fake descriptor exposing `.scale=2`, `.supports_half=False`, `.input_channels=3`, `.output_channels=3`, `.size_requirements=None`, `.to()->self`, `.eval()->self`, and a `__call__` that does nearest-neighbor 2x upsample; build an Upscaler against a spec with loader='spandrel' on device='cpu' (monkeypatch ensure_weights to return a dummy path) and assert `up.scale == 2` and that `up.upscale(small_img)` returns an image exactly 2x the input size.
- tests/test_spandrel_loader.py::test_spandrel_fp16_gated_by_supports_half — with device forced 'cpu' (or descriptor.supports_half=False) assert `up.use_fp16 is False` even when fp16=True is requested.
- tests/test_spandrel_loader.py::test_spandrel_rejects_non_rgb — fake descriptor with input_channels=1 must raise RuntimeError with a clear message.

**Acceptance criteria**
- `ModelSpec` has a `loader` field defaulting to "rrdbnet"; every existing entry in MODELS resolves with loader=='rrdbnet' (no per-entry edits required).
- Constructing `Upscaler` for any existing model produces the SAME RRDBNet object path as before (no spandrel import on the native path — verify spandrel is imported lazily only inside the spandrel branch).
- For a spec with loader=='spandrel', `Upscaler.scale` equals the loaded descriptor's `.scale` (not spec.scale), and `_run_tiled` runs without error against that scale.
- fp16 is enabled only when device is cuda AND the descriptor reports supports_half; CPU/MPS and unsupported-half archs stay fp32.
- A non-3-channel descriptor raises a clear RuntimeError rather than crashing inside the tiler.
- All four new pytest cases pass on CPU with no network access (ensure_weights and spandrel.ModelLoader are monkeypatched).
- `upscaler --list-models` and `import app` still succeed unchanged.

**Risks**
- spandrel descriptor's `__call__`/forward expects a 4D NCHW float tensor in [0,1] like RRDBNet; if some archs expect a different normalization the tiler output would be wrong — not verifiable without downloading a real non-RRDBNet checkpoint, so this batch only proves plumbing with a fake descriptor.
- `size_requirements` semantics (multiple_of / minimum / square) vary per arch; the conservative padding approach may under-pad for archs needing square or minimum-size inputs. Mitigated by gating native behavior and documenting that real arch specs land in a later batch.
- spandrel_extra_arches.install() mutates a global registry; calling it inside the loader is idempotent but adds import cost only on the spandrel path.

---

## Item 1B — GitHub Actions CI (macOS + ubuntu) + registry-guard / build_demo / checksum pytest  _(effort: M)_

Create .github/workflows/ci.yml (none exists today — verified: /Users/anamariaradulescu/Herd/Upscaler/.github/workflows is absent). Matrix over os=[ubuntu-latest, macos-latest] and a single Python (3.11). Steps: checkout; setup-python with `cache: pip`; install CPU torch FIRST from the CPU wheel index (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) since that is the slow step and must be pip-cached; then `pip install -e .[dev,gui]` (gui pulls gradio so build_demo and `import app` work); then run `pytest`, `python -c "import app"`, and `upscaler --list-models`. Add three pytest cases: (1) a build_demo construction/smoke test, (2) the ensure_weights checksum-mismatch delete-and-raise path with a tiny fake file (note: tests/test_weights.py L15-28 ALREADY covers this for DeblurSpec — the new test should cover a ModelSpec to avoid duplication, or be skipped if judged redundant; reviewer's call), and (3) a registry guard asserting EVERY spec in MODELS, DEBLUR_MODELS, FACE_MODELS, and FACE_DETECTOR has a 64-hex-char sha256 AND an https:// url. The existing test_weights.py L9-13 only checks MODELS+DEBLUR_MODELS sha length (not hex, not url, not FACE_*); test_face.py L10-13 checks FACE sha+url presence. The new guard consolidates and strengthens all of them.

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/pyproject.toml

**New files**
- /Users/anamariaradulescu/Herd/Upscaler/.github/workflows/ci.yml
- /Users/anamariaradulescu/Herd/Upscaler/tests/test_build_demo.py
- /Users/anamariaradulescu/Herd/Upscaler/tests/test_registry_guard.py

**Tests**
- tests/test_build_demo.py::test_build_demo_constructs — `pytest.importorskip('gradio')`; call `app.build_demo()` and assert it returns a `gradio.Blocks` instance; this exercises config.load() seeding and the whole tab tree without launching a server.
- tests/test_registry_guard.py::test_every_spec_is_pinned_and_https — iterate MODELS, DEBLUR_MODELS, FACE_MODELS values + FACE_DETECTOR; assert `re.fullmatch(r'[0-9a-f]{64}', spec.sha256)` and `spec.url.startswith('https://')` and `spec.filename`. (Verified all 12 current specs pass this today.)
- tests/test_registry_guard.py::test_checksum_mismatch_removes_modelspec_file — monkeypatch weights.WEIGHTS_DIR to tmp_path, pre-write a tiny fake file matching a ModelSpec.filename with sha256='0'*64, assert ensure_weights raises RuntimeError matching 'Checksum mismatch' and the file is unlinked (mirrors test_weights.py pattern but for ModelSpec).

**Acceptance criteria**
- .github/workflows/ci.yml exists and triggers on push + pull_request.
- CI matrix includes BOTH ubuntu-latest and macos-latest.
- CI installs CPU torch via the pytorch CPU wheel index BEFORE `pip install -e .[dev,gui]`, and enables pip caching (cache: pip on actions/setup-python, with a cache key that includes pyproject.toml) so torch isn't re-downloaded every run.
- CI runs all three: `pytest`, `python -c "import app"`, and `upscaler --list-models`, and the job fails if any returns non-zero.
- `pytest` passes locally and in CI with NO network (all new tests are offline; build_demo test importorskips gradio).
- test_registry_guard asserts 64-hex sha256 + https url for MODELS, DEBLUR_MODELS, FACE_MODELS, and FACE_DETECTOR (a superset of the checks currently split across test_weights.py and test_face.py).
- build_demo test returns a gr.Blocks without launching a server or downloading weights.

**Risks**
- macos-latest is arm64; the pytorch CPU wheel index must have an arm64 wheel for the chosen torch version (it does for torch>=2.x, but pinning is safer). If the bare `torch` CPU wheel resolution fails on macOS, fall back to plain `pip install torch` (PyPI macOS wheels are already CPU-only) — could not verify the exact macOS wheel availability from inside this sandbox.
- `.[gui]` pulls gradio 6.x (installed locally is 6.15.2) which is heavy; if CI time is a concern the build_demo test could be moved behind importorskip and gui installed only on ubuntu. Reviewer should confirm acceptable CI minutes.
- The existing test_weights.py::test_all_registered_weights_are_pinned and test_face.py::test_face_models_pinned overlap with the new guard; leaving all three is harmless but redundant — decide whether to delete the weaker two.

---

## Verified facts (from reading the code)
- upscaler/engine.py L99-107: Upscaler.__init__ hardcodes `net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=self.spec.scale, num_feat=..., num_block=..., num_grow_ch=...)` — there is no arch-agnostic branch today.
- upscaler/engine.py L97: `self.use_fp16 = fp16 and self.device.type == 'cuda'` — fp16 is CUDA-only.
- upscaler/engine.py L107: weights loaded via `net.load_state_dict(_load_state_dict(ensure_weights(self.spec)), strict=True)`.
- upscaler/engine.py L114-116: `scale` is a property returning `self.spec.scale` — for a spandrel path scale must instead come from the loaded descriptor.
- upscaler/engine.py L134-147 (`_net`): padding multiple is computed as `m = 2 if self.scale==2 else 4 if self.scale==1 else 1`, which is RRDBNet/pixel_unshuffle-specific.
- upscaler/engine.py L156-182 (`_run_tiled`): tiler uses `s = self.scale` and calls `self._net(...)` per tile — it is arch-agnostic as long as `self.net(t)` returns an s× tensor.
- upscaler/models/registry.py L17-29: `ModelSpec` is a frozen dataclass with name/url/scale/filename/num_block/num_feat/num_grow_ch/sha256/notes — NO field to opt into a non-RRDBNet loader.
- upscaler/face.py L52-65: face restorer already loads ANY spandrel arch via `self._spandrel.ModelLoader().load_from_file(str(ensure_weights(self._spec)))` after `spandrel_extra_arches install()` (called as `_sea.install()` at L53), then `.to(device).eval()`; the descriptor is invoked as `net(t)` at L93.
- spandrel 0.4.2 is installed (.venv); ImageModelDescriptor.__init__ takes scale, input_channels, output_channels, supports_half, supports_bfloat16, size_requirements, tiling, purpose, and is callable via __call__(self, image: Tensor) -> Tensor — so the descriptor can be dropped straight into the existing tiler.
- No .github/ directory exists at repo root (verified: `ls .github/workflows` -> No such file or directory).
- app.py L1193: `def build_demo() -> gr.Blocks:`; it calls `config.load()` and builds the full tab tree; `import app` succeeds today and `upscaler --list-models` prints the model list (both verified by running them).
- tests/test_weights.py L9-13 asserts only MODELS+DEBLUR_MODELS have a 64-char sha256 (length only, not hex, not url); L15-28 already tests the checksum delete-and-raise path for a DeblurSpec with a pre-written fake file and tmp_path WEIGHTS_DIR monkeypatch.
- tests/test_face.py L10-13 asserts FACE_MODELS + FACE_DETECTOR have a 64-char sha256 and non-empty url+filename.
- No tests/conftest.py exists.
- All 12 current specs (7 MODELS, 3 DEBLUR_MODELS, 1 FACE_MODELS, 1 FACE_DETECTOR) already satisfy the proposed guard: sha256 matches ^[0-9a-f]{64}$ and url starts with https:// (verified by script).
- Local toolchain: torch 2.12.0, gradio 6.15.2, spandrel 0.4.2 in .venv.
- pyproject.toml L22 `dev` extra = pytest+onnxruntime+onnx+onnxscript+pypdfium2+pillow-heif+imageio-ffmpeg; L23 `gui` extra includes gradio>=4.0; L32 `face` extra = spandrel>=0.4 + spandrel_extra_arches>=0.2 + opencv-python-headless>=4.9.
- upscaler/deblur.py L17 reuses `from upscaler.engine import _load_state_dict, resolve_device`, confirming engine.py is the shared home for load/device helpers — the new `_load_spandrel` helper belongs there too.

## Open questions
- Should a real spandrel-arch model (e.g. a small SwinIR or HAT) be registered in this batch to prove the path end-to-end, or is a monkeypatched fake descriptor sufficient? The prompt scopes registering real archs to later batches, so 1A only proves plumbing — confirm that is acceptable.
- macos-latest is arm64; need to confirm the exact torch version pin that has an arm64 CPU wheel on the pytorch CPU index, or fall back to PyPI torch on macOS. Could not verify wheel availability from inside the sandbox.
- Whether to delete the now-redundant test_weights.py::test_all_registered_weights_are_pinned and test_face.py::test_face_models_pinned once test_registry_guard supersedes them, or keep all three.
- Whether CI should install `.[gui]` on both OSes (heavier, but needed for the build_demo/import-app smoke tests) or only on ubuntu to save macOS minutes.
