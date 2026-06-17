# Batch 2 — FBCNN JPEG artifact removal (Clean up accordion) + CodeFormer as a second face model

**Scope:** Batch 2: Fast restoration wins — FBCNN JPEG cleanup + CodeFormer face restore

## Goal
Add two spandrel-loaded restoration wins that reuse existing plumbing: (2A) FBCNN color JPEG de-blocking exposed as a checkbox in the existing "Clean up — deblur / denoise" accordion, reusing the _structural_ok garbage-guard and the _restore blend; (2B) CodeFormer as a second selectable face model in the existing "Restore faces" accordion with a fidelity slider, reusing ALL of face.py's YuNet detect / FFHQ-512 align / feathered paste-back. Both depend on Batch 1 item 1A (a generic spandrel-based loader) because the current Deblurrer hardcodes NAFNet construction and cannot load an FBCNN .pth.

## Dependencies
- Batch 1 item 1A: the generic spandrel ModelLoader-based restorer/loader. 2A's FBCNN restorer needs spandrel.ModelLoader().load_from_file() because upscaler/deblur.py:36-43 hardcodes NAFNet(width=...,enc_blk_nums=...) and load_state_dict(strict=True) — an FBCNN checkpoint will not load through that path. If 1A lands a reusable spandrel loader, 2A's FbcnnRestorer should call it instead of re-implementing spandrel wiring (the only existing precedent is face.py:61-65).
- Both 2A and 2B require the [face] extra (spandrel + spandrel_extra_arches + opencv) declared in pyproject.toml:32 — FBCNN and CodeFormer archs live in spandrel/spandrel_extra_arches. No new pip dependency is added; only registry entries + UI + a loader.
- Independent of Batches 3/4/5.

---

## Item 2A — FBCNN color JPEG artifact removal as a Clean-up restorer  _(effort: M)_

Add an FBCNN (color) JPEG de-blocking restorer that loads via spandrel (Batch 1 item 1A's loader, or a small FbcnnRestorer modeled on Deblurrer's whole-image forward at deblur.py:48-55). Register it with a new lightweight spec type (FBCNN does NOT fit DeblurSpec's NAFNet fields per registry.py:127-136). Expose it as ONE new checkbox 'Remove JPEG artifacts (FBCNN)' inside the existing 'Clean up — deblur / denoise' accordion (app.py:1290). Run it as an extra pre-upscale restore stage that reuses _structural_ok (app.py:167-180) and the same strength-blend shape as _restore (app.py:183-198). Keep it independent of the NAFNet deblur_model dropdown so a user can do FBCNN-only, NAFNet-only, or both.

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/models/registry.py
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/models/weights.py
- /Users/anamariaradulescu/Herd/Upscaler/app.py
- /Users/anamariaradulescu/Herd/Upscaler/pyproject.toml

**New files**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/restore.py — FbcnnRestorer (spandrel-loaded, whole-image forward mirroring deblur.py:48-55, lazy weight load, RGB->tensor->net->RGB). If Batch 1 1A lands a generic spandrel restorer base, subclass/call it instead of re-implementing _deps()/ModelLoader; otherwise mirror face.py:31-44 _deps() and face.py:61-65 lazy-load.
- /Users/anamariaradulescu/Herd/Upscaler/tests/test_fbcnn.py

**Registry / data additions**
- registry.py: a NEW frozen dataclass (e.g. RestoreSpec with fields name,url,filename,sha256,notes — identical shape to FaceSpec at registry.py:193-199, because FBCNN has no NAFNet width/blk fields) plus an ARTIFACT_MODELS / RESTORE_MODELS dict and a DEFAULT, e.g. fbcnn-color = RestoreSpec(name='fbcnn-color', url='https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth', filename='fbcnn_color.pth', sha256=<PIN AT BUILD TIME via scripts/print_checksums.py>, notes='Removes JPEG blocking/ringing artifacts (FBCNN, color). Research use.').
- weights.py:18 — widen WeightSpec Union to include the new RestoreSpec (and FaceSpec/BGSpec for honesty); ensure_weights already works by duck-typing but the alias is currently a lie.
- pyproject.toml: no new dependency — confirm the [face] extra (line 32) is the home for FBCNN and document it; optionally add a one-line comment that FBCNN/CodeFormer ride the same extra.

**UI changes**
- In the 'Clean up — deblur / denoise' Accordion (app.py:1290): add fbcnn = gr.Checkbox(value=False, label='Remove JPEG artifacts (FBCNN)', info='De-blocks heavily-compressed JPEGs. Independent of the deblur/denoise model above — tick either or both.') placed after restore_strength (app.py:1310) and before restore_btn (app.py:1311), so it shares the cleanup section.
- Thread fbcnn into enhance() and restore_only(): update enhance signature (app.py:201-202) and run.click inputs (app.py:2044-2045); update restore_only signature (app.py:263) and restore_btn.click inputs (app.py:2051). When fbcnn is on, run an FBCNN restore stage (own getter+cache, NOT _get_deblurrer which builds NAFNet) guarded by _structural_ok and append 'remove JPEG artifacts (FBCNN)' to the stages list (mirroring app.py:212-215).
- Add _ARTIFACT getter+cache: a _get_fbcnn(device) modeled on _get_face_restorer (app.py:76-83) with its own module-level cache dict, since _UP_CACHE/_DB_CACHE/_FACE_CACHE are keyed per engine (app.py:34-36).

**Tests**
- tests/test_fbcnn.py::test_fbcnn_spec_pinned — assert the FBCNN RestoreSpec has sha256 of len 64 and non-empty url/filename (mirror tests/test_face.py:10-13). MUST be added to (or its dict folded into) the pinned-weights coverage so test_weights.py-style enforcement applies.
- tests/test_fbcnn.py::test_fbcnn_in_registry — assert the new ARTIFACT/RESTORE_MODELS dict is non-empty and the default name resolves (mirror resolve_deblur_model error-path test pattern).
- tests/test_fbcnn.py::test_fbcnn_resolve_unknown_raises — resolve_fbcnn_model('nope') raises ValueError listing available (mirror registry.py:184-188).
- tests/test_fbcnn.py::test_enhance_threads_fbcnn_flag — monkeypatch app._get_fbcnn to a stub returning an identity restorer and app._get_upscaler to an identity stub; call app.enhance(img, ..., fbcnn=True) and assert the result info string contains 'FBCNN' and that the stub.restore/cleanup was invoked (no network, no real weights). Mirrors how test_restore_guard.py imports app directly (tests/test_restore_guard.py:12).
- Extend test_weights.py::test_all_registered_weights_are_pinned (tests/test_weights.py:9-12) to also iterate the new ARTIFACT/RESTORE_MODELS dict, so an unpinned FBCNN hash fails CI.

**Acceptance criteria**
- A new spec type is added for FBCNN that does NOT reuse DeblurSpec's NAFNet fields; FBCNN loads through spandrel (Batch 1 1A loader or a new FbcnnRestorer), NOT through the NAFNet Deblurrer at deblur.py:36-43.
- The Clean-up accordion shows exactly one new checkbox 'Remove JPEG artifacts (FBCNN)'; it is independent of the existing deblur_model dropdown (user can run FBCNN alone, NAFNet alone, or both, in a defined order — FBCNN first then NAFNet, or document the chosen order).
- FBCNN output passes through _structural_ok (app.py:167-180) and is blended by strength exactly like _restore (app.py:197); a garbage result is skipped with the same ⚠ stage message style as app.py:215.
- FBCNN runs whole-image (no tiling) like Deblurrer; on a large image it still returns a same-resolution result without an exception.
- enhance() and restore_only() signatures plus both .click() input lists (app.py:2042-2054) are updated consistently — the app imports and the existing test suite still passes.
- The FBCNN weight is sha256-pinned (verified by the extended pin test); a wrong hash triggers the existing 'Checksum mismatch' removal path (weights.py:77-84).
- weights.py WeightSpec Union (line 18) is widened to include the new spec type.

**Risks**
- spandrel arch ID / loadability for FBCNN was NOT verified (spandrel not installed here). FBCNN may load as a 1->1 restoration model in spandrel; if spandrel does not ship an FBCNN arch in the pinned spandrel_extra_arches version, this item is blocked — must confirm at implementation time with `python -c 'import spandrel_extra_arches; spandrel_extra_arches.install(); ...'`.
- FBCNN forward in spandrel may return only the restored image (the published model also predicts a quality factor); confirm spandrel's wrapper returns an HxWx3 tensor compatible with the deblur.py:48-55 forward and does not require a QF input.
- _structural_ok with threshold 0.5 (app.py:180) may be too loose to catch a subtly-wrong FBCNN de-block (it preserves structure). It guards crashes/garbage, not quality regressions — acceptable but worth a code comment.
- The fbcnn_color.pth direct asset URL and its sha256 could NOT be confirmed/computed offline; URL pattern is the standard GitHub release path but MUST be downloaded once and pinned via scripts/print_checksums.py before merge.
- FBCNN license: verify upstream (jiaxi-jiang/FBCNN) license at build time; if non-commercial, add a '(research/non-commercial)' note like UltraSharp/Remacri (registry.py:72,80).

---

## Item 2B — CodeFormer as a second face model with a fidelity slider  _(effort: M)_

Add a 'codeformer' FaceSpec to FACE_MODELS (registry.py:202-210) so it loads through the EXISTING face.py spandrel path (face.py:61-65) and reuses ALL of detect/align/paste-back (face.py:67-107). Surface a model Dropdown ('GFPGAN' / 'CodeFormer') and a fidelity/weight slider in the 'Restore faces' accordion (app.py:1315). Make _get_face_restorer model-aware (it currently takes device only and always builds the default — app.py:76-83) and thread the chosen model + fidelity into enhance() (app.py:225-236). The fidelity slider maps to CodeFormer's w parameter (higher w = more fidelity to the original / less aggressive restoration); for GFPGAN it falls back to the existing strength blend (face.py:97-98).

**Files touched**
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/models/registry.py
- /Users/anamariaradulescu/Herd/Upscaler/upscaler/face.py
- /Users/anamariaradulescu/Herd/Upscaler/app.py

**New files**
- /Users/anamariaradulescu/Herd/Upscaler/tests/test_codeformer.py (or extend tests/test_face.py)

**Registry / data additions**
- registry.py FACE_MODELS (line 202): add codeformer = FaceSpec(name='codeformer', url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth', filename='codeformer.pth', sha256=<PIN AT BUILD TIME>, notes='CodeFormer — strong face restoration with adjustable fidelity (S-Lab License 1.0, non-commercial research). ~359MB.'). Keep DEFAULT_FACE_MODEL='gfpgan-v1.4' (registry.py:211) so behavior is unchanged unless selected.

**UI changes**
- In the 'Restore faces (GFPGAN)' Accordion (app.py:1315) — rename label to 'Restore faces' (drop the GFPGAN-only title since there are now two models): add face_model = gr.Dropdown(_FACE_CHOICES, value=DEFAULT_FACE_MODEL, label='Face model', filterable=False, info='GFPGAN is the gentle default; CodeFormer is stronger and lets you trade fidelity vs. quality.') after the face checkbox (app.py:1321). Build _FACE_CHOICES near app.py:657-658 from FACE_MODELS like _DEBLUR_CHOICES.
- Add face_fidelity = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label='CodeFormer fidelity (w)', info='Only used by CodeFormer. Higher = truer to the original face (less aggressive); lower = stronger restoration.') — keep the existing face_strength slider (app.py:1322) for the GFPGAN blend. Optionally gate fidelity visibility on the model dropdown via a .change handler (mirror _switch_method at app.py:398-404).
- Make _get_face_restorer model-aware: change signature to _get_face_restorer(model, device) and cache key from (device,) to (model, device) (app.py:76-83), passing model to FaceRestorer(model=model, device=device) (FaceRestorer already accepts model per face.py:51).
- Thread face_model + face_fidelity through enhance(): update signature (app.py:201-202), the face stage call (app.py:227 -> _get_face_restorer(face_model, device).restore(result, face_strength, fidelity=face_fidelity)), the stage label (app.py:236 -> f'faces ({face_model})...'), and run.click inputs (app.py:2044-2045).

**Tests**
- tests/test_codeformer.py::test_codeformer_registered_and_pinned — assert FACE_MODELS contains 'codeformer' with a 64-char sha256 (this is already partially enforced by the existing tests/test_face.py:10-13 test_face_models_pinned, which will now also cover CodeFormer — so an unpinned hash FAILS that existing test).
- tests/test_codeformer.py::test_default_face_model_unchanged — assert DEFAULT_FACE_MODEL == 'gfpgan-v1.4' (registry.py:211) so adding CodeFormer doesn't silently change default behavior.
- tests/test_codeformer.py::test_no_face_image_passes_through_codeformer — parametrize the existing test_face.py:33-47 pattern over model in ('gfpgan-v1.4','codeformer'): build FaceRestorer(model=m, device='cpu'), feed pure noise, assert identity output and fr._net is None (heavy net never loads when no face is detected). importorskip cv2/spandrel/spandrel_extra_arches and skip on offline detector download (test_face.py:39-42).
- tests/test_codeformer.py::test_unknown_face_model_raises — FaceRestorer(model='nope', device='cpu') raises ValueError listing available (face.py:54-55).
- tests/test_codeformer.py::test_fidelity_clamped — assert the fidelity value is clamped to [0,1] before being passed to the net (mirror the strength clamp at face.py:81); pure-Python, no weights.
- Add a test that enhance() threads face_model through: monkeypatch app._get_face_restorer to a recording stub and assert the selected model name reaches it and appears in the info string (mirror test_restore_guard.py importing app directly).

**Acceptance criteria**
- FACE_MODELS has both 'gfpgan-v1.4' and 'codeformer'; DEFAULT_FACE_MODEL is unchanged so existing behavior/tests are unaffected unless CodeFormer is explicitly selected.
- CodeFormer loads through the EXISTING face.py path with NO duplication of detect/align/paste-back — the only face.py change is to plumb a fidelity arg into the net call and (if needed) select the spandrel call signature; lines 67-107 logic is reused verbatim.
- The Restore-faces accordion shows a Face-model dropdown and a CodeFormer fidelity slider; selecting GFPGAN ignores fidelity and uses the existing strength blend (face.py:97-98).
- _get_face_restorer is keyed by (model, device) so switching models doesn't return a stale cached GFPGAN restorer (current cache key is (device,) at app.py:77 — this is a real bug the change fixes).
- enhance() signature + run.click input list (app.py:2042-2048) are updated consistently; the no-face passthrough test passes for BOTH models and the heavy net is not loaded when no face is found (fr._net is None).
- CodeFormer weight is sha256-pinned and the registry notes flag the S-Lab non-commercial license, consistent with the UltraSharp/Remacri precedent (registry.py:72,80).
- If spandrel's CodeFormer wrapper does NOT accept a per-call fidelity weight, the slider gracefully falls back to the strength blend and the info text is adjusted — feature still ships.

**Risks**
- CRITICAL UNVERIFIED: whether spandrel's CodeFormer wrapper exposes a per-call fidelity weight w. spandrel is not installed here so I could not inspect the descriptor's call signature. face.py calls net(t) with no extra args (face.py:93). If spandrel bakes in a fixed w, the fidelity slider cannot drive CodeFormer's native w and would only drive the post-blend (face.py:97-98) — the feature degrades but still ships. Must verify at build time with the installed spandrel version.
- CodeFormer expects a 512x512 aligned BGR/RGB face — the existing pipeline aligns to FFHQ-512 (_SIZE=512, face.py:28,88), which matches, so paste-back should work. Verify channel order (face.py converts BGR->RGB at 90 before the net, RGB->BGR after at 94 — confirm CodeFormer's spandrel wrapper expects RGB like GFPGAN).
- codeformer.pth sha256 could NOT be computed offline (no download in sandbox) — MUST be pinned at build time via scripts/print_checksums.py; the existing test_face.py:10-13 will fail until it is pinned, which is the desired forcing function.
- License: CodeFormer is S-Lab License 1.0 (non-commercial research), stricter than GFPGAN's Apache-2.0 (registry.py:208). Must be clearly noted in the spec notes and ideally in the UI dropdown label, matching the '(non-commercial)' precedent already set for community ESRGAN weights.
- Changing the accordion title from 'Restore faces (GFPGAN)' may affect any test/string that greps that exact label — none found in tests, but confirm.

---

## Verified facts (from reading the code)
- upscaler/deblur.py:36-43 — Deblurrer.__init__ hardcodes NAFNet(img_channel=3, width=self.spec.width, middle_blk_num=..., enc_blk_nums=..., dec_blk_nums=...) and net.load_state_dict(_load_state_dict(ensure_weights(self.spec)), strict=True). FBCNN cannot ride this constructor; it needs spandrel.
- upscaler/deblur.py:48-55 — Deblurrer.deblur(): RGB->float/255->permute(2,0,1).unsqueeze(0)->net(x).clamp_(0,1)->squeeze->permute(1,2,0)->uint8. Whole-image, no tiling (docstring lines 1-7 explain NAFNet global pooling). An FBCNN restorer can mirror this exact tensor dance.
- upscaler/face.py:51-65 — FaceRestorer.__init__ validates model against FACE_MODELS (line 54-55), stores self._spec = FACE_MODELS[model] (56), and _gfpgan() lazily loads via self._spandrel.ModelLoader().load_from_file(str(ensure_weights(self._spec))) then .to(device).eval() (63-64). This is model-agnostic — adding a CodeFormer FaceSpec to FACE_MODELS makes it loadable here with zero changes to load logic.
- upscaler/face.py:67-107 — restore() does ALL detect/align/paste-back generically: YuNet FaceDetectorYN (74-76), estimateAffinePartial2D to _TEMPLATE (85), warpAffine to 512 (88), net(t).clamp(0,1) (92-93), invertAffineTransform + feathered mask paste-back (99-104). The forward is just net(t) — no model-specific args. Strength blend at 97-98 blends restored vs aligned-original.
- upscaler/face.py:53 — _sea.install() registers GFPGAN/CodeFormer archs into spandrel's registry (comment confirms CodeFormer is among them). face.py:31-44 _deps() imports cv2/spandrel/spandrel_extra_arches behind the [face] extra with a friendly RuntimeError.
- upscaler/models/registry.py:193-211 — FaceSpec dataclass(frozen) has fields name,url,filename,sha256,notes; FACE_MODELS currently holds only 'gfpgan-v1.4' (203-209) and DEFAULT_FACE_MODEL='gfpgan-v1.4' (211). Adding a 'codeformer' entry is a pure dict addition.
- upscaler/models/registry.py:126-188 — DeblurSpec dataclass has NAFNet-shaped fields (width, middle_blk_num, enc_blk_nums, dec_blk_nums) that DO NOT fit FBCNN. resolve_deblur_model() (182-188) validates against DEBLUR_MODELS. FBCNN therefore needs a NEW spec type (e.g. RestoreSpec/FbcnnSpec with just name/url/filename/sha256/notes) OR to reuse FaceSpec's shape — it must NOT be shoehorned into DeblurSpec.
- upscaler/models/weights.py:18 — WeightSpec = Union[ModelSpec, DeblurSpec]; but ensure_weights only touches spec.filename/url/sha256 (73-85), so FaceSpec and BGSpec already pass through by duck-typing (face.py:58,63; background.py:66). A new FBCNN spec will work at runtime but the WeightSpec type alias should be widened for honesty.
- app.py:167-180 — _structural_ok(a,b) computes mean-subtracted luma correlation, returns >0.5. Reusable verbatim to guard FBCNN output (an over-aggressive de-block could still pass since it preserves structure, but it guards against total garbage).
- app.py:183-198 — _restore(src_img, deblur_model, device, onnx, strength) calls _get_deblurrer(...).deblur(rgb), runs _structural_ok, blends via Image.blend(rgb,out,strength). app.py:63-73 _get_deblurrer caches by (model,device,onnx) and builds Deblurrer or OnnxDeblurrer. FBCNN as a SEPARATE checkbox should NOT collide with the deblur_model dropdown; needs its own cache + getter.
- app.py:201-260 — enhance() signature is (image, model, device, deblur, deblur_model, restore_strength, sharpen, tile, onnx, out_size, face=False, face_strength=1.0). Face stage at 225-236 calls _get_face_restorer(device).restore(result, face_strength) — note _get_face_restorer (76-83) takes ONLY device and instantiates FaceRestorer() with the default model; it does NOT pass a model arg, so it cannot select CodeFormer yet.
- app.py:76-83 — _get_face_restorer(device) caches by (device,) only and constructs FaceRestorer(device=device) with default gfpgan. To support a model dropdown the cache key must become (model, device) and the getter must pass model through.
- app.py:1290-1314 — the 'Clean up — deblur / denoise' Accordion holds: deblur Checkbox (1291), deblur_model Dropdown (1297), restore_strength Slider (1304), restore_btn (1311). This is where the FBCNN checkbox goes.
- app.py:1315-1328 — the 'Restore faces (GFPGAN)' Accordion holds: face Checkbox (1316) and face_strength Slider (1322). This is where the face-model Dropdown + fidelity slider go.
- app.py:2042-2054 — run.click wires enhance with inputs [inp, model, device, deblur, deblur_model, restore_strength, sharpen, tile, onnx, out_size, face, face_strength]; restore_btn.click wires restore_only with [inp, deblur_model, restore_strength, sharpen, device, onnx]. Any new component must be threaded into these input lists AND the enhance/restore_only signatures in the same order.
- app.py:657-658 — _MODEL_CHOICES and _DEBLUR_CHOICES are built as [(f'{name} — {notes}', name) ...]; a _FACE_CHOICES list should follow the same shape from FACE_MODELS.
- tests/test_face.py:10-13 test_face_models_pinned iterates FACE_MODELS + FACE_DETECTOR asserting sha256 len==64 — adding CodeFormer with a real pinned hash keeps this green; an unpinned hash will FAIL this existing test. tests/test_weights.py:9-12 test_all_registered_weights_are_pinned iterates {**MODELS, **DEBLUR_MODELS} — a new FBCNN registry dict must be added to that union (or its own test) or it escapes pin-checking.
- tests/test_face.py:33-47 test_no_face_image_passes_through importorskips cv2/spandrel/spandrel_extra_arches, builds FaceRestorer(device='cpu'), feeds noise, asserts identity output and fr._net is None — this is the template for a CodeFormer-model test (parametrize the model arg).
- upscaler/cli.py:309-316 — CLI has --deblur / --deblur-model (choices=sorted(DEBLUR_MODELS)) and applies deblurrer.deblur(img) at 398-399, but NO --face flag exists. CLI parity for FBCNN/CodeFormer is out of scope for Batch 2 (the prompt scopes CLI parity to Batch 3); note it as an open item.
- spandrel is NOT installed in this environment (python3 -c 'import spandrel' -> ModuleNotFoundError), so I could NOT empirically confirm the exact spandrel arch IDs for FBCNN/CodeFormer, nor whether spandrel's CodeFormer wrapper exposes a per-call fidelity weight, nor compute any sha256. These are flagged as risks/open-questions.
- CodeFormer weight source confirmed via web: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth (359MB). CodeFormer is released under the S-Lab License 1.0 (non-commercial research) — NOT Apache-2.0 like GFPGAN — so it follows the same '(non-commercial)' notes precedent already used for 4x-UltraSharp/Remacri at registry.py:72,80.
- FBCNN weight source confirmed via web: release tag v1.0 ('model_zoo', 2022-09-15) on github.com/jiaxi-jiang/FBCNN with asset fbcnn_color.pth (color JPEG artifact removal). Standard GitHub asset URL pattern: https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth — could NOT confirm the exact live asset href or compute its sha256 offline. FBCNN is MIT-ish/research; verify license at build time.

## Open questions
- Does the pinned spandrel_extra_arches version actually register an FBCNN arch, and does its forward accept a plain HxWx3 tensor (no quality-factor input)? Could not verify — spandrel is not installed in this sandbox.
- Does spandrel's CodeFormer descriptor accept a per-call fidelity weight w, or is w fixed at load time? This determines whether the fidelity slider drives CodeFormer natively or only the post-restore blend.
- Order of operations when both FBCNN and NAFNet deblur are enabled in Clean-up: FBCNN-then-NAFNet, or expose only one at a time? Recommend FBCNN first (de-block), then optional NAFNet denoise, with a one-line UI note. Confirm with product.
- Exact pinned sha256 for fbcnn_color.pth and codeformer.pth — must be downloaded once and pinned at build time (scripts/print_checksums.py referenced at registry.py:27). The existing pin tests (test_face.py:10-13, test_weights.py:9-12) will fail until pinned, which is intended.
- Should CLI gain --fbcnn / --face / --face-model parity now? The prompt scopes CLI parity to Batch 3 and cli.py:309-327 has no --face flag today, so Batch 2 leaves CLI unchanged — confirm this scoping.
- Confirm the exact live download href for fbcnn_color.pth (GitHub release asset loading errored in fetch); the standard /releases/download/v1.0/fbcnn_color.pth pattern is assumed.
