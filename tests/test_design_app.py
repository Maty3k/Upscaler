"""App-level wiring for the Design tab."""

from __future__ import annotations

import pytest
from PIL import Image


def _photo(w=400, h=300):
    return Image.new("RGB", (w, h), (90, 140, 60))


def _inputs(tpl, texts=None, photo=None, logo=None, cutout=None, canvas=None):
    """The tab's inputs, in the order the wiring passes them."""
    from upscaler import design as dz

    texts = texts or {}
    import app

    values = [texts.get(slot, "") for slot in tpl.text_slots()]
    values += [""] * (app.DZ_SLOTS - len(values))
    pal = tpl.palette
    return ([dz.to_json(tpl), photo, logo, cutout, canvas or tpl.canvas,
             pal.primary, pal.secondary, pal.accent, pal.ink] + values)


def test_app_builds_the_template_from_the_controls():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    tpl = dz.built_in("Quote card")
    built = app._dz_build(dz.to_json(tpl), "800x400", "#111111", "#222222",
                          "#ff0000", "#ffffff")
    assert built.canvas == "800x400"                  # the picker wins over the file
    assert built.palette.accent == "#ff0000"
    assert [layer.kind for layer in built.layers] == [layer.kind for layer in tpl.layers]


def test_app_pairs_the_boxes_with_the_slots_in_order():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    tpl = dz.built_in("Event poster")
    assert tpl.text_slots() == ["title", "date", "place"]
    got = app._dz_texts(tpl, ("A", "B", "C", "", ""))
    assert got == {"title": "A", "date": "B", "place": "C"}


def test_app_labels_one_box_per_slot_and_hides_the_rest():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    fields = app._dz_fields(dz.built_in("Quote card"))
    assert len(fields) == app.DZ_SLOTS
    assert fields[0]["visible"] and fields[0]["label"] == "Quote"
    assert fields[1]["label"] == "Author" and fields[1]["value"]   # prefilled copy
    assert not fields[2]["visible"]


def test_app_pick_fans_out_every_control():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    got = app.design_pick_ui("Subject spotlight")
    assert len(got) == 7 + app.DZ_SLOTS + 2
    tpl = dz.from_json(got[0])
    assert tpl.name == "Subject spotlight"
    assert got[2] == tpl.canvas and got[5] == tpl.palette.accent
    assert got[-2]["visible"] is True                 # it wants a photo
    assert got[-1]["visible"] is False                # it doesn't want a logo


def test_app_a_broken_template_warns_instead_of_blanking_the_tab():
    pytest.importorskip("gradio")
    import app

    got = app.design_json_ui("{not json")
    assert "⚠" in got[1]
    # everything else is left alone rather than being reset to nothing
    assert got[0] == {"__type__": "update"}


def test_app_preview_renders_and_names_what_is_missing():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    tpl = dz.built_in("Quote card")
    out, note = app.design_preview_ui(*_inputs(tpl, {"quote": "Hi", "author": "Me"}))
    assert out is not None and "Quote card" in note["value"]
    assert "waiting" not in note["value"]

    poster = dz.built_in("Event poster")
    out, note = app.design_preview_ui(*_inputs(poster))
    assert out is not None and "waiting for" in note["value"] and "a photo" in note["value"]


def test_app_preview_warns_on_a_canvas_it_cannot_read():
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    out, note = app.design_preview_ui(*_inputs(dz.built_in("Quote card"),
                                               canvas="sideways"))
    assert out is None and "⚠" in note["value"]


def test_app_cutout_runs_only_for_a_template_that_wants_one(monkeypatch):
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    cut = Image.new("RGBA", (400, 300), (0, 255, 0, 255))
    monkeypatch.setattr(dz, "subject_cutout", lambda img, model="": cut)

    spotlight = dz.to_json(dz.built_in("Subject spotlight"))
    state, note = app.design_cutout_ui(_photo(), spotlight)
    assert state is cut and "covers" in note["value"]
    # a template with no subject layer never pays for the model
    quote = dz.to_json(dz.built_in("Quote card"))
    assert app.design_cutout_ui(_photo(), quote)[0] is None
    assert app.design_cutout_ui(None, spotlight)[0] is None


def test_app_apply_writes_a_png(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app
    from upscaler import design as dz

    monkeypatch.setattr(app, "_ensure_export_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app.library, "save_path", lambda *a, **k: None)
    tpl = dz.built_in("Quote card")
    out, path, info = app.design_apply_ui(*_inputs(tpl, {"quote": "Hi", "author": "Me"}))
    assert out.size == (1080, 1080) and path.endswith(".png") and "✅" in info
    with pytest.raises(app.gr.Error):
        app.design_apply_ui(*_inputs(tpl, canvas="sideways"))


def test_the_tab_is_wired_to_its_own_controls():
    """Tabs are built inside one long function, so a repeated variable prefix
    rebinds one tab's controls to another's. Check the built graph."""
    pytest.importorskip("gradio")
    import app

    wanted = {"design_pick_ui", "design_preview_ui", "design_apply_ui",
              "design_cutout_ui", "design_json_ui"}
    deps = [f for f in app.build_demo().fns.values()
            if getattr(f.fn, "__name__", "") in wanted]
    assert {f.fn.__name__ for f in deps} == wanted

    pick = next(f for f in deps if f.fn.__name__ == "design_pick_ui")
    assert [c.label for c in pick.inputs] == ["Template"]
    assert len(pick.outputs) == 7 + app.DZ_SLOTS + 2
    for f in deps:
        if f.fn.__name__ in ("design_preview_ui", "design_apply_ui"):
            assert [c.label for c in f.inputs][:3] == ["Template", "Photo",
                                                       "Logo (a transparent PNG "
                                                       "works best)"]
            assert len(f.inputs) == 9 + app.DZ_SLOTS
