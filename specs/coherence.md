# Coherence — loader API & build order
## Canonical loader API (Batches 2 & 3 build on this)
CANONICAL LOADER API (engine.py) — the stable interface Batches 2 & 3 build on.

OPT-IN: ModelSpec (registry.py:17-29) gains one field: `loader: str = "rrdbnet"` (allowed: "rrdbnet" | "spandrel"). Every existing spec keeps loader=="rrdbnet" with zero edits, so the native RRDBNet path at engine.py:99-112 is byte-for-byte unchanged. Note ModelSpec is `@dataclass(frozen=True)`, so the new field must be added with a default AFTER sha256/notes are still defaulted (it can sit anywhere after `scale`/`filename` since all remaining fields are defaulted).

FUNCTION SIGNATURE (module-level helper in engine.py, the single shared entry point — colocate with _load_state_dict/resolve_device per the deblur.py:17 precedent):

    def load_spandrel(path, device, *, fp16: bool = False, require_channels: int | None = 3) -> LoadedModel:
        ...

RETURN CONTRACT — return a small frozen dataclass (NOT a bare descriptor) so callers never re-derive scale/half/padding:

    @dataclass(frozen=True)
    class LoadedModel:
        net: object        # the spandrel ImageModelDescriptor, callable as net(t)->t, already .to(device).eval()
        scale: int         # = descriptor.scale  (DISCOVERED from the model, never the spec)
        use_fp16: bool     # = fp16 and device.type=="cuda" and descriptor.supports_half
        pad_to: int        # = max(descriptor.size_requirements.multiple_of, 1) if size_requirements else 1
        input_channels: int
        output_channels: int

BEHAVIOR (verified against installed spandrel 0.4.2 — descriptor exposes .scale/.input_channels/.output_channels/.supports_half/.size_requirements{minimum,multiple_of,square}; SizeRequirements/MaskedImageModelDescriptor confirmed present):
  1. Lazy import inside the function: `import spandrel; import spandrel_extra_arches as _sea; _sea.install()` — mirrors face.py:52-53. Wrap ImportError into the SAME friendly RuntimeError string as face.py:37-39 ('... pip install -e \".[face]\"').
  2. `desc = spandrel.ModelLoader().load_from_file(str(path))` (path is the result of ensure_weights, passed by the caller — load_spandrel does NOT call ensure_weights, keeping it spec-agnostic and reusable by Batch 3's DDColor/LaMa which use ColorizeSpec/InpaintSpec).
  3. `desc = desc.to(device).eval()` (mirrors face.py:64).
  4. scale = desc.scale; use_fp16 gating ANDs CUDA + supports_half (never call .half() when unsupported); pad_to from size_requirements.multiple_of.
  5. If require_channels is not None, raise a clear RuntimeError when desc.input_channels != require_channels or desc.output_channels != require_channels. Upscaler passes require_channels=3 (the tiler assumes RGB). Batch 3 DDColor (1->3) and LaMa (masked) pass require_channels=None and use the descriptor directly — they do NOT route through Upscaler's tiler.

HOW Upscaler CONSUMES IT (engine.py:99-116): branch on spec.loader. Native path unchanged. Spandrel path: `lm = load_spandrel(ensure_weights(self.spec), self.device, fp16=fp16); self.net=lm.net; self._scale=lm.scale; self.use_fp16=lm.use_fp16; self._pad=lm.pad_to`. Add backing field `self._scale` (set to self.spec.scale on the native path) and change the `scale` property (engine.py:114-116) to return `self._scale`. In `_net` (engine.py:134-147), gate the padding: native keeps the existing `m = 2 if scale==2 else 4 if scale==1 else 1`; spandrel uses `m = self._pad`. _run_tiled (engine.py:156-182) is already arch-agnostic and needs no change.

SCALE DISCOVERY (the load-bearing rule): scale ALWAYS comes from descriptor.scale on the spandrel path, never from spec.scale. Spec.scale is meaningless/ignored for loader=="spandrel". This is what lets Batch 2/3 register HAT/DRCT/SwinIR/FBCNN without knowing their scale up front.

## Build order
- 1A — lands the canonical load_spandrel API + ModelSpec.loader field; every other spandrel feature (2A,2B,3A,3B) depends on this exact interface, so it must be first.
- 1B — CI matrix + registry-integrity/build_demo guards; independent of 1A, parallelizable, but should land early so CI exercises all later batches and the sha256-pin guard becomes a forcing function for 2A/2B/3A/3B unpinned weights.
- 2B — CodeFormer: pure FaceSpec dict-add reusing the EXISTING face.py path (no new loader code, lowest risk spandrel win); also fixes the real _get_face_restorer cache-key bug (app.py:77 keyed by device only). Does not strictly need 1A but should follow it to keep one loader story.
- 2A — FBCNN restorer: new RestoreSpec + restore.py calling load_spandrel (require_channels=3, whole-image like deblur.py:48-55). Depends on 1A's loader; independent of 2B.
- 3A — Colorize/DDColor tab: new module + ColorizeSpec, calls load_spandrel with require_channels=None (1->3). Depends on 1A; parallel with 3B but coordinate registry.py edits.
- 3B — Inpaint/LaMa tab: MaskedImageModelDescriptor two-arg call desc(image,mask); does NOT use load_spandrel's channel guard (require_channels=None) and bypasses Upscaler's tiler. Depends on 1A's lazy-import/install convention; coordinate registry.py with 3A.
- 3C — CLI parity (removebg/batch/--face): depends only on existing face.py/background.py, NOT on 1A/2/3A/3B. Can land any time after 1B; placed here to follow the GUI features it mirrors.
- 4A — Model Manager (torch-free manage.py): independent of all spandrel work; can land any time but benefits from full registry set, so after 2/3 add their specs so the table covers them.
- 4B — Diagnostics report: depends on 4A (reuses manage.total_bytes/human_size); land right after 4A in the same manage.py.
- 4C — Cancel + per-tile progress: touches engine.py + app.py build_demo; independent of loader work but coordinate the single app.py edit with 2A/2B/3A/3B.
- 5A — Live clock overlay: refactors compose_frame signature + extracts _render_text_layer; must precede 5B.
- 5B — Animated text motion: builds on 5A's compose_frame/frame-context and shared _render_text_layer; after 5A.
- 5C — Save/load .json layouts: schema must include 5A clock + 5B motion fields, so it lands last in Batch 5.

## Cross-batch notes
- Single source of truth for the spandrel load: 1A's load_spandrel must be the ONLY place that does `import spandrel; spandrel_extra_arches.install(); ModelLoader().load_from_file(...).to().eval()`. Refactor face.py onto it. 2A/3A/3B then call load_spandrel(ensure_weights(spec), device, require_channels=...) and must NOT re-implement the import/install dance.
- require_channels is the seam that makes one helper serve all three call shapes: Upscaler/FBCNN pass 3 (tiler-safe RGB); DDColor passes None (1->3); LaMa passes None and ignores LoadedModel.net's single-arg assumption entirely (it uses the MaskedImageModelDescriptor two-arg call desc(image,mask) — 3B should call ModelLoader directly OR load_spandrel must tolerate masked descriptors; recommend 3B uses load_spandrel only for the import/to/eval plumbing and then does its own two-arg call).
- Centralize pin-checking: introduce one ALL_SPECS list (or a reflective iterator over every *_MODELS dict) in registry.py so 1B's guard, 4A's manage.list_specs, and every batch's new registry are covered automatically and stay in sync. This single change neutralizes the fragmented pin-test inconsistency above.
- All weight URLs/sha256 for FBCNN, CodeFormer, DDColor, LaMa are UNVERIFIED/uncomputable offline. Every spec that adds a new weight (2A,2B,3A,3B) MUST run scripts/print_checksums.py after a one-time download before merge; 1B's strengthened guard (64-hex + https) is the forcing function that will fail CI until pinned — that is the intended gate.
- app.py build_demo is touched by 2A, 2B, 3A, 3B, 4A, 4B, 4C, and the whole of Batch 5. Serialize these app.py edits or expect heavy conflicts; the flat-component-list invariant in Batch 5 (N_TEXT/_TEXT_FIELDS/N_OVERLAY_VALS at app.py:497-501) is especially fragile and should land in one coordinated pass.
- Licensing: CodeFormer (S-Lab non-commercial) and likely DDColor/LaMa mirrors are NOT BSD/Apache like the core Real-ESRGAN weights. Follow the existing '(non-commercial)' notes precedent and surface it in the UI dropdown label, as 2B/3A specs already require.
- Verified offline: installed spandrel 0.4.2 descriptor exposes scale/input_channels/output_channels/supports_half/size_requirements{minimum,multiple_of,square} and MaskedImageModelDescriptor exists — the LoadedModel contract is buildable today against the real library.

## Inconsistencies flagged
- DUPLICATED loader-wiring: face.py:52-53,61-65 ALREADY does exactly what load_spandrel will do (lazy import + _sea.install() + ModelLoader().load_from_file().to().eval()). 1A must extract this into the shared load_spandrel helper AND refactor face.py:_gfpgan to call it, otherwise there are two divergent spandrel-load code paths. None of the specs say to refactor face.py onto the new helper — 1A should explicitly do so (face.py uses require_channels=None / keeps its own forward).
- WeightSpec type-alias drift: weights.py:18 is `Union[ModelSpec, DeblurSpec]` but ensure_weights duck-types on filename/url/sha256, so FaceSpec/BGSpec already pass at runtime (the alias is already a lie). 2A says widen it for RestoreSpec, 3A/3B add ColorizeSpec/InpaintSpec but do NOT mention widening WeightSpec. Pick ONE batch (1A, since it owns engine/loader) to widen WeightSpec to all spec types once, or accept the alias stays cosmetic — but the three specs are inconsistent about who fixes it.
- Pin-test coverage is fragmented and double-claimed: 1B's test_registry_guard claims to be the single superset guard over MODELS/DEBLUR_MODELS/FACE_MODELS/FACE_DETECTOR, yet 2A also says to EXTEND test_weights.py for ARTIFACT_MODELS, and 3A/3B add their own per-batch pinned tests, and 4A's test asserts a count == len(MODELS)+...+len(BG_MODELS). If 1B's guard is the single source of truth it must iterate the NEW registries (ARTIFACT/COLORIZE/INPAINT/BG) too — but those don't exist when 1B lands. Decision: make 1B's guard import a single canonical ALL_SPECS aggregator (or iterate every *_MODELS dict reflectively) so new batches auto-register; otherwise each batch silently escapes the guard. BG_MODELS notably has sha256=None (background.py:36-49), so the guard must EXEMPT or special-case BG_MODELS or it will fail.
- Scale property mutation risk: 1A changes engine.py:114-116 `scale` from `return self.spec.scale` to `return self._scale`. deblur.py and any external caller reading `.spec.scale` directly are unaffected, but 4C's tile_count helper and _run_tiled both read `self.scale` — confirm 1A lands the `_scale` backing field before 4C builds on tile counts. Cross-batch, not contradictory, but order-sensitive.
- fp16 ownership: 1A's LoadedModel computes use_fp16, but engine.py:97 ALSO computes self.use_fp16 before the net is built. On the spandrel path self.use_fp16 must be OVERWRITTEN by the descriptor-aware value (the AND with supports_half). 1A's item text says to AND them but the spec doesn't flag that engine.py:97 runs unconditionally first — implementer must move/override it on the spandrel branch.
- CodeFormer fidelity is CRITICAL-UNVERIFIED in 2B: face.py:93 calls net(t) with no extra args. Whether spandrel's CodeFormer descriptor accepts a per-call w is unconfirmed. 2B's acceptance criteria already include a graceful fallback to the strength blend — keep that as the shipping contract; do NOT let the fidelity slider block the batch.
- 3C removebg friendly-message extra: 2A/3A/3B all reuse the '.[face]' message, but 3C's removebg uses onnxruntime ([onnx]/[gui]), NOT [face]. 3C correctly flags this; just ensuring reviewers don't copy the '.[face]' string into removebg.
