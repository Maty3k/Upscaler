"""Recipes: running a saved chain of edits, round-tripping it, and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from upscaler import blur, recipe as rc
from upscaler.cli import main


def _photo(w=200, h=150):
    img = Image.new("RGB", (w, h), (90, 110, 140))
    ImageDraw.Draw(img).rectangle([20, 20, 80, 80], fill=(230, 210, 120))
    return img


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


# ── the shape of a recipe ─────────────────────────────────────────────────────

def test_an_empty_recipe_changes_nothing():
    img = _photo()
    assert np.array_equal(_arr(rc.run(img, rc.Recipe()).image), _arr(img))
    assert "does nothing" in rc.Recipe().describe()


def test_steps_run_in_order():
    """Order matters: cropping then watermarking is not the same as the
    reverse, so the chain must be honoured as written."""
    img = _photo(200, 200)
    crop_first = rc.Recipe(steps=[
        rc.Step("crop", {"aspect": "Widescreen · 16:9"}),
        rc.Step("adjust", {"exposure": 30}),
    ])
    adjust_first = rc.Recipe(steps=[
        rc.Step("adjust", {"exposure": 30}),
        rc.Step("crop", {"aspect": "Widescreen · 16:9"}),
    ])
    a, b = rc.run(img, crop_first).image, rc.run(img, adjust_first).image
    assert a.size == b.size                       # same shape either way here
    one_step = rc.run(img, rc.Recipe(steps=[rc.Step("adjust", {"exposure": 30})])).image
    assert one_step.size == img.size              # but the crop really happened above
    assert a.size != one_step.size


@pytest.mark.parametrize("tool", rc.TOOLS)
def test_every_tool_can_be_a_step(tool):
    img = _photo()
    out = rc.run_steps(img, rc.Recipe(steps=[rc.Step(tool, {})]))
    assert out.width > 0 and out.height > 0


def test_a_step_can_be_limited_to_a_region():
    img = _photo(200, 100)
    r = rc.Recipe(steps=[rc.Step("blur", {"kind": "gaussian", "strength": 60},
                                 {"shape": "rectangle", "x": 25, "w": 30, "feather": 0})])
    out = _arr(rc.run(img, r).image)
    base = _arr(img)
    assert np.array_equal(out[:, 150:], base[:, 150:])        # untouched outside
    assert not np.array_equal(out[:, 40:60], base[:, 40:60])  # changed inside


def test_a_faces_region_is_detected_at_run_time(monkeypatch):
    from upscaler import face

    monkeypatch.setattr(face, "detect_faces",
                        lambda img, confidence=0.6: [face.Face(0.1, 0.1, 0.3, 0.3, 0.9)])
    img = _photo()
    r = rc.Recipe(steps=[rc.Step("blur", {"kind": "pixelate", "strength": 60},
                                 {"shape": "faces", "feather": 0})])
    out = _arr(rc.run(img, r).image)
    assert not np.array_equal(out, _arr(img))
    assert np.array_equal(out[130:, 170:], _arr(img)[130:, 170:])   # far corner untouched


def test_a_depth_region_is_estimated_at_run_time(monkeypatch):
    from upscaler import depth

    ramp = Image.fromarray(
        np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (150, 1)), "L")
    monkeypatch.setattr(depth, "estimate", lambda img, **kw: ramp.resize(img.size))
    r = rc.Recipe(steps=[rc.Step("blur", {"strength": 60},
                                 {"shape": blur.DEPTH, "focus": 100, "dof": 10})])
    out = _arr(rc.run(_photo(), r).image)
    assert not np.array_equal(out, _arr(_photo()))


def test_a_painted_region_is_refused_with_a_reason():
    """A brush stroke can't live in a JSON file, so say so rather than
    quietly doing nothing."""
    r = rc.Recipe(steps=[rc.Step("blur", {"strength": 40}, {"shape": "painted"})])
    with pytest.raises(rc.RecipeError, match="painted"):
        rc.run(_photo(), r)


# ── output settings ───────────────────────────────────────────────────────────

def test_output_format_produces_bytes():
    r = rc.Recipe(steps=[], output_format="JPEG")
    res = rc.run(_photo(), r)
    assert res.data and res.fmt == "JPEG" and res.extension == "jpg"


def test_a_size_budget_is_honoured():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (800, 800, 3), dtype=np.uint8), "RGB")
    r = rc.Recipe(steps=[], target_size="60KB")
    res = rc.run(noisy, r)
    assert res.data and len(res.data) <= 60 * 1024
    assert "quality" in res.note


def test_strip_metadata_cleans_the_output():
    from upscaler import metadata as md

    r = rc.Recipe(steps=[], output_format="JPEG", strip_metadata=True)
    res = rc.run(_photo(), r)
    assert md.read(res.data).is_clean


def test_no_output_settings_means_an_image_and_no_bytes():
    res = rc.run(_photo(), rc.Recipe(steps=[rc.Step("adjust", {"exposure": 10})]))
    assert res.data is None and res.image.size == (200, 150)


# ── saving and loading ────────────────────────────────────────────────────────

def test_json_round_trip_is_exact():
    for name in rc.BUILT_IN_NAMES:
        original = rc.built_in(name)
        assert rc.to_dict(rc.from_json(rc.to_json(original))) == rc.to_dict(original)


def test_save_and_load_a_file(tmp_path):
    path = rc.save(rc.built_in("Film look"), tmp_path / "sub" / "r.json")
    assert path.exists()
    assert rc.load(path).name == "Film look"
    assert json.loads(path.read_text())["schema"] == rc.SCHEMA


def test_unknown_settings_are_ignored_but_unknown_tools_are_not():
    """A recipe from another version should still run; a step naming a tool
    that doesn't exist must not be skipped silently."""
    r = rc.from_dict({"steps": [{"tool": "adjust",
                                 "params": {"exposure": 10, "future_knob": 3}}]})
    assert rc.run_steps(_photo(), r).size == (200, 150)
    with pytest.raises(rc.RecipeError, match="unknown tool"):
        rc.from_dict({"steps": [{"tool": "levitate"}]})


@pytest.mark.parametrize("bad,match", [
    ("not json at all", "valid JSON"),
    ('"a string"', "must be a JSON object"),
    ('{"steps": ["nope"]}', "must be an object"),
])
def test_malformed_recipes_are_reported_clearly(bad, match):
    with pytest.raises(rc.RecipeError, match=match):
        rc.from_json(bad)


def test_built_ins_are_copies_and_all_run():
    img = _photo()
    assert len(rc.BUILT_IN_NAMES) >= 6
    assert rc.built_in("Film look") is not rc.BUILT_IN["Film look"]
    for name in rc.BUILT_IN_NAMES:
        if "depth" in name.lower() or "face" in name.lower():
            continue                              # those need a model; covered above
        res = rc.run(img, rc.built_in(name))
        assert res.image.width > 0
    with pytest.raises(rc.RecipeError, match="unknown recipe"):
        rc.built_in("Nonsense")


def test_describe_reads_as_a_sentence():
    text = rc.built_in("Web-ready").describe()
    assert "Web-ready:" in text and "→" in text
    assert "save as WebP" in text and "under 400KB" in text and "metadata removed" in text


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_list_and_show(capsys):
    assert main(["recipe", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Web-ready" in out and "Film look" in out
    assert main(["recipe", "Film look", "--show"]) == 0
    assert "Film look:" in capsys.readouterr().out


def test_cli_runs_a_built_in(tmp_path):
    src = tmp_path / "p.jpg"
    _photo().save(src)
    assert main(["recipe", "Film look", str(src)]) == 0
    out = tmp_path / "p_film-look.png"
    assert out.exists() and not np.array_equal(_arr(Image.open(out)), _arr(_photo()))


def test_cli_saves_a_recipe_you_can_edit_and_run(tmp_path):
    path = tmp_path / "mine.json"
    assert main(["recipe", "Film look", "--save", str(path)]) == 0
    data = json.loads(path.read_text())
    data["name"] = "Edited"
    data["steps"] = [{"tool": "adjust", "params": {"exposure": 40}}]
    path.write_text(json.dumps(data))
    src = tmp_path / "p.jpg"
    _photo().save(src)
    assert main(["recipe", str(path), str(src), "-o", str(tmp_path / "out.png")]) == 0
    assert _arr(Image.open(tmp_path / "out.png")).mean() > _arr(_photo()).mean()


def test_cli_folder_and_output_naming(tmp_path):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for n in ("a", "b"):
        _photo().save(src_dir / f"{n}.jpg")
    out = tmp_path / "out"
    assert main(["recipe", "Sign and shrink", str(src_dir), "-o", str(out)]) == 0
    names = sorted(p.name for p in out.glob("*"))
    assert names == ["a_sign-and-shrink.jpg", "b_sign-and-shrink.jpg"]
    assert all((out / n).stat().st_size <= 500 * 1024 for n in names)


def test_cli_errors(tmp_path, capsys):
    assert main(["recipe"]) == 2
    assert "--list" in capsys.readouterr().err
    src = tmp_path / "p.jpg"
    _photo().save(src)
    assert main(["recipe", "nosuch", str(src)]) == 2
    assert "no recipe called" in capsys.readouterr().err
    assert main(["recipe", "Film look", str(tmp_path / "nope.jpg")]) == 2
