"""App-level wiring for the Recipe operation in the Batch tab."""

import numpy as np
import pytest
from PIL import Image


def _photo(w=120, h=90):
    return Image.new("RGB", (w, h), (100, 120, 150))


def test_app_batch_offers_recipe():
    pytest.importorskip("gradio")
    import app

    assert "Recipe" in app._BATCH_OPS
    updates = app._switch_batch_op("Recipe")
    assert len(updates) == len(app._BATCH_OPS)
    assert updates[-1]["visible"] and not updates[0]["visible"]


def test_app_picking_a_built_in_loads_its_json():
    pytest.importorskip("gradio")
    import app
    from upscaler import recipe as rc

    box, note = app.recipe_load_built_in("Film look")
    assert '"name": "Film look"' in box["value"]
    assert "Film look:" in note["value"]
    assert rc.from_json(box["value"]).name == "Film look"
    _box, note = app.recipe_load_built_in("Nonsense")
    assert "⚠" in note["value"]


def test_app_an_edited_recipe_wins_over_the_dropdown():
    """Editing the box must never be silently ignored in favour of the
    built-in still selected above it."""
    pytest.importorskip("gradio")
    import app

    edited = '{"name": "Mine", "steps": [{"tool": "adjust", "params": {"exposure": 20}}]}'
    chosen = app._resolve_recipe_ui("Film look", edited)
    assert chosen.name == "Mine"
    assert app._resolve_recipe_ui("Film look", "   ").name == "Film look"


def test_app_shows_what_the_current_recipe_does():
    pytest.importorskip("gradio")
    import app

    _box, note = app.recipe_show_ui("Web-ready", "")
    assert "Web-ready:" in note["value"]
    _box, note = app.recipe_show_ui("Web-ready", "{not json")
    assert "⚠" in note["value"] and "JSON" in note["value"]


def test_app_batch_runs_a_recipe(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    src = tmp_path / "p.png"
    _photo().save(src)
    recipe = '{"name": "Brighten", "steps": [{"tool": "adjust", "params": {"exposure": 45}}]}'
    gallery, zip_path, info = app.batch_process(
        [str(src)], "Recipe", "realesrgan-x2plus", "Model default (×2/×4)", 0.0,
        "PNG", 90, "u2net", 1, "cpu", 512, "Film look", recipe,
        progress=lambda *a, **k: None)
    assert len(gallery) == 1
    assert np.asarray(gallery[0], np.float32).mean() > np.asarray(_photo(), np.float32).mean()
    assert zip_path and "1" in info

    with pytest.raises(gr.Error, match="JSON"):
        app.batch_process([str(src)], "Recipe", "realesrgan-x2plus",
                          "Model default (×2/×4)", 0.0, "PNG", 90, "u2net", 1,
                          "cpu", 512, "Film look", "{broken",
                          progress=lambda *a, **k: None)
