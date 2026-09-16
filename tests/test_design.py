"""Design templates: the layout engine, the templates that ship, and the CLI.

Layers are positioned as percentages of the canvas, so most of these tests are
"put a known block of colour at a known place and check it landed there" —
which also proves the same template holds together at any canvas size.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from upscaler import design as dz
from upscaler.cli import main


def _photo(w=400, h=300, color=(60, 120, 200)):
    return Image.new("RGB", (w, h), color)


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.int16)


def _tpl(*layers, **kw):
    base = dict(name="T", canvas="600x400", background="solid",
                palette=dz.Palette(primary="#000000", secondary="#333333",
                                   accent="#ff0000", ink="#ffffff"))
    base.update(kw)
    return dz.Template(layers=list(layers), **base)


# ── canvases ──────────────────────────────────────────────────────────────────

def test_a_canvas_is_a_name_or_a_size():
    assert dz.canvas_size("YouTube thumbnail · 1280×720") == (1280, 720)
    assert dz.canvas_size("1200x800") == (1200, 800)
    assert dz.canvas_size("1200×800") == (1200, 800)       # the pretty multiplication sign
    for bad in ("", "wide", "0x0", "12x"):
        with pytest.raises(dz.DesignError, match="canvas"):
            dz.canvas_size(bad)


def test_every_canvas_is_a_sane_size():
    for name, (w, h) in dz.CANVASES.items():
        assert 300 <= w <= 4000 and 300 <= h <= 4000, name


# ── colour ────────────────────────────────────────────────────────────────────

def test_palette_roles_and_hex_both_resolve():
    pal = dz.Palette(primary="#102030", secondary="#405060", accent="#ff8800",
                     ink="#ffffff")
    assert dz._rgba("accent", pal) == (255, 136, 0, 255)
    assert dz._rgba("#0f0", pal) == (0, 255, 0, 255)
    assert dz._rgba("#ff000080", pal) == (255, 0, 0, 128)
    assert dz._rgba("transparent", pal) == (0, 0, 0, 0)
    assert dz._rgba("not a colour", pal, (1, 2, 3, 4)) == (1, 2, 3, 4)


def test_a_role_can_be_faded_so_a_scrim_stays_themeable():
    """A template that froze #000000cc into a scrim would ignore the palette;
    "primary@80" follows it."""
    pal = dz.Palette(primary="#102030")
    assert dz._rgba("primary@80", pal) == (16, 32, 48, 204)
    assert dz._rgba("primary@0", pal)[3] == 0
    assert dz._rgba("primary@999", pal)[3] == 255          # clamped, not wrapped


def test_named_palettes_are_copies_and_complete():
    for name in dz.PALETTE_NAMES:
        pal = dz.palette(name)
        assert pal is not dz.PALETTES[name]
        for role in dz.ROLES:
            assert getattr(pal, role).startswith("#")


# ── fonts ─────────────────────────────────────────────────────────────────────

def test_font_roles_resolve_to_something_that_exists():
    from upscaler.fonts import FONTS

    for role in dz.ROLE_NAMES:
        assert dz.resolve_font(role) in FONTS
    assert dz.resolve_font("No Such Font 9000") in FONTS


# ── fitting text ──────────────────────────────────────────────────────────────

def test_text_shrinks_until_it_fits_rather_than_running_off_the_edge():
    long = "An extremely long headline that will not fit on one line at any size"
    font, lines, px, tracking = dz.fit_text(long, dz.resolve_font("sans"), 300, 120,
                                            max_px=90, line_height=1.1,
                                            tracking_pct=0)
    assert px < 90                                          # it had to come down
    assert len(lines) > 1                                   # and wrap
    for line in lines:
        assert font.getlength(line) <= 300 + 1
    assert len(lines) * px * 1.1 <= 120 + 1


def test_short_text_keeps_the_size_it_asked_for():
    _font, lines, px, _t = dz.fit_text("Hi", dz.resolve_font("sans"), 400, 200,
                                       max_px=60, line_height=1.1, tracking_pct=0)
    assert px == 60 and lines == ["Hi"]


def test_an_explicit_newline_starts_a_new_line():
    _f, lines, _px, _t = dz.fit_text("one\ntwo", dz.resolve_font("sans"), 600, 0,
                                     max_px=30, line_height=1.1, tracking_pct=0)
    assert lines == ["one", "two"]


def test_tracking_widens_a_line_and_so_lowers_the_size_that_fits():
    args = ("SPACED OUT LABEL", dz.resolve_font("sans"), 260, 0)
    _f, _l, plain, _t = dz.fit_text(*args, max_px=60, line_height=1.1, tracking_pct=0)
    _f, _l, spaced, tracking_px = dz.fit_text(*args, max_px=60, line_height=1.1,
                                              tracking_pct=25)
    assert tracking_px > 0 and spaced <= plain


# ── layers ────────────────────────────────────────────────────────────────────

def test_a_band_lands_where_the_percentages_say():
    tpl = _tpl(dz.Layer(kind=dz.BAND, x=25, y=50, w=10, h=100, color="accent"))
    out = _arr(dz.render(tpl))                              # 600×400
    assert tuple(out[200, 150]) == (255, 0, 0)              # centre of the band
    assert tuple(out[200, 450]) == (0, 0, 0)                # the other side is ground
    column = (out[:, 150, 0] > 200).sum()
    assert column == 400                                    # full height, as asked


def test_a_band_with_two_stops_is_a_gradient_and_alpha_lets_the_ground_through():
    tpl = _tpl(dz.Layer(kind=dz.BAND, x=50, y=50, w=100, h=100, color="transparent",
                        color2="#ffffff", angle=90),
               background="solid")
    out = _arr(dz.render(tpl))
    assert out[5].mean() < out[-5].mean()                   # dark at the top, light below
    assert out[2].mean() < 30                               # the ground still shows


def test_a_photo_layer_fills_its_box():
    tpl = _tpl(dz.Layer(kind=dz.PHOTO, slot="photo", x=50, y=25, w=100, h=50,
                        fit="cover"))
    out = _arr(dz.render(tpl, photo=_photo(color=(0, 255, 0))))
    assert tuple(out[100, 300]) == (0, 255, 0)              # inside the box
    assert tuple(out[300, 300]) == (0, 0, 0)                # below it, the ground


def test_contain_keeps_the_whole_picture_and_cover_crops_it():
    wide = _photo(400, 100, (0, 255, 0))
    box = dict(x=50, y=50, w=50, h=50)
    covered = _arr(dz.render(_tpl(dz.Layer(kind=dz.PHOTO, fit="cover", **box)),
                             photo=wide))
    contained = _arr(dz.render(_tpl(dz.Layer(kind=dz.PHOTO, fit="contain", **box)),
                               photo=wide))
    green = lambda a: int(((a[..., 1] > 200) & (a[..., 0] < 60)).sum())  # noqa: E731
    assert green(covered) > green(contained)                # cover fills the box


def test_a_layer_that_hangs_off_the_canvas_is_cropped_not_shifted():
    """A design that bleeds off the page is normal; sliding it back inside
    would silently break the layout."""
    tpl = _tpl(dz.Layer(kind=dz.BAND, x=0, y=50, w=40, h=20, color="accent"))
    out = _arr(dz.render(tpl))
    assert tuple(out[200, 2]) == (255, 0, 0)                # it starts at the edge
    assert (out[200, :, 0] > 200).sum() == 120              # half of 40% of 600


def test_rotation_and_opacity_both_take():
    plain = _arr(dz.render(_tpl(dz.Layer(kind=dz.BAND, w=60, h=20, color="accent"))))
    turned = _arr(dz.render(_tpl(dz.Layer(kind=dz.BAND, w=60, h=20, color="accent",
                                          rotation=30))))
    faded = _arr(dz.render(_tpl(dz.Layer(kind=dz.BAND, w=60, h=20, color="accent",
                                         opacity=40))))
    assert not np.array_equal(plain, turned)
    assert faded[200, 300, 0] < plain[200, 300, 0]


def test_text_lands_and_a_filled_slot_beats_the_template_copy():
    tpl = _tpl(dz.Layer(kind=dz.TEXT, slot="headline", text="PLACEHOLDER",
                        x=50, y=50, w=80, size=12, color="ink"))
    default = _arr(dz.render(tpl))
    filled = _arr(dz.render(tpl, texts={"headline": "REAL WORDS"}))
    assert default.max() > 200 and filled.max() > 200       # both drew something
    assert not np.array_equal(default, filled)
    # Absent and empty are different on purpose: no value at all leaves the
    # template's own copy showing, while clearing the box clears the line —
    # otherwise there would be no way to delete a line you don't want.
    assert np.array_equal(default, _arr(dz.render(tpl, texts={})))
    assert _arr(dz.render(tpl, texts={"headline": ""})).max() == 0


def test_valign_moves_the_block_inside_its_box():
    def ink_rows(valign):
        tpl = _tpl(dz.Layer(kind=dz.TEXT, text="Hi", x=50, y=50, w=80, h=60,
                            size=6, color="ink", valign=valign))
        rows = np.where(_arr(dz.render(tpl)).max(axis=(1, 2)) > 150)[0]
        return int(rows.mean())
    assert ink_rows("top") < ink_rows("middle") < ink_rows("bottom")


def test_a_subject_layer_uses_the_cutout_it_is_given():
    """The GUI computes the cut-out once and hands it over; the renderer must
    not quietly run the model again."""
    cut = Image.new("RGBA", (200, 200), (0, 255, 0, 255))
    mask = Image.new("L", (200, 200), 0)
    mask.paste(255, (50, 50, 150, 150))
    cut.putalpha(mask)
    tpl = _tpl(dz.Layer(kind=dz.SUBJECT, x=50, y=50, w=50, h=50, fit="contain"))

    def boom(*a, **k):
        raise AssertionError("the model must not be called when a cut-out is given")

    out = _arr(dz.render(tpl, photo=_photo(), cutout=cut))
    assert tuple(out[200, 300]) == (0, 255, 0)              # the subject
    assert tuple(out[200, 20]) == (0, 0, 0)                 # the ground, not the photo


# ── backgrounds ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["solid", "gradient", "mesh", "photo", "transparent"])
def test_every_background_renders(kind):
    out = dz.render(_tpl(background=kind), photo=_photo(color=(0, 255, 0)))
    assert out.size == (600, 400)
    if kind == "transparent":
        assert out.mode == "RGBA" and np.asarray(out.getchannel("A")).max() == 0
    elif kind == "photo":
        assert tuple(_arr(out)[200, 300]) == (0, 255, 0)


def test_a_photo_background_without_a_photo_falls_back_to_the_ground():
    out = dz.render(_tpl(background="photo"))
    assert tuple(_arr(out)[200, 300]) == (0, 0, 0)


# ── the canvas and scaling ────────────────────────────────────────────────────

def test_render_matches_the_canvas_and_scale_shrinks_it():
    tpl = _tpl(canvas="Social card · 1600×900")
    assert dz.render(tpl).size == (1600, 900)
    assert dz.render(tpl, scale=0.25).size == (400, 225)
    assert dz.preview(tpl, max_edge=800).size == (800, 450)


def test_the_same_design_holds_together_at_another_canvas():
    """Every measurement is a share of the canvas, so a layer covers the same
    fraction of the picture whichever size it is rendered at."""
    layer = dz.Layer(kind=dz.BAND, x=50, y=50, w=40, h=25, color="accent")
    shares = []
    for canvas in ("600x400", "1200x800", "1000x1500"):
        out = _arr(dz.render(_tpl(layer, canvas=canvas)))
        shares.append(float((out[..., 0] > 200).mean()))
    assert max(shares) - min(shares) < 0.005


# ── what it still needs ───────────────────────────────────────────────────────

def test_missing_lists_the_gaps_and_goes_quiet_when_they_are_filled():
    tpl = _tpl(dz.Layer(kind=dz.PHOTO, slot="photo", w=100, h=100),
               dz.Layer(kind=dz.TEXT, slot="headline", text="x", w=80))
    assert dz.missing(tpl) == ["a photo", "headline"]
    assert dz.missing(tpl, photo=_photo(), texts={"headline": "Hi"}) == []
    assert dz.missing(tpl, photo=_photo(), texts={"headline": "   "}) == ["headline"]


def test_slots_are_listed_once_and_in_order():
    tpl = _tpl(dz.Layer(kind=dz.TEXT, slot="a", w=10),
               dz.Layer(kind=dz.TEXT, slot="b", w=10),
               dz.Layer(kind=dz.TEXT, slot="a", w=10))
    assert tpl.slots() == [("a", dz.TEXT), ("b", dz.TEXT)]
    assert tpl.text_slots() == ["a", "b"]


# ── saving and loading ────────────────────────────────────────────────────────

def test_json_round_trip_is_exact_for_every_template():
    for name in dz.BUILT_IN_NAMES:
        original = dz.built_in(name)
        assert dz.to_dict(dz.from_json(dz.to_json(original))) == dz.to_dict(original)


def test_save_and_load_a_file(tmp_path):
    path = dz.save(dz.built_in("Quote card"), tmp_path / "sub" / "t.json")
    assert path.exists() and dz.load(path).name == "Quote card"
    assert json.loads(path.read_text())["schema"] == dz.SCHEMA


def test_unknown_settings_are_ignored_but_an_unknown_kind_is_not():
    tpl = dz.from_dict({"layers": [{"kind": "text", "text": "hi", "future_knob": 3}]})
    assert tpl.layers[0].text == "hi"
    with pytest.raises(dz.DesignError, match="unknown kind"):
        dz.from_dict({"layers": [{"kind": "hologram"}]})


@pytest.mark.parametrize("bad,match", [
    ("not json", "valid JSON"),
    ('"a string"', "must be a JSON object"),
    ('{"layers": ["nope"]}', "must be an object"),
])
def test_malformed_templates_are_reported_clearly(bad, match):
    with pytest.raises(dz.DesignError, match=match):
        dz.from_json(bad)


# ── the templates that ship ───────────────────────────────────────────────────

def test_built_ins_are_copies_and_named():
    assert len(dz.BUILT_IN_NAMES) >= 10
    assert dz.built_in("Quote card") is not dz.BUILT_IN["Quote card"]
    with pytest.raises(dz.DesignError, match="unknown template"):
        dz.built_in("Nonsense")


@pytest.mark.parametrize("name", list(dz.BUILT_IN))
def test_every_template_renders_filled_in(name):
    tpl = dz.built_in(name)
    photo = _photo(800, 600, (90, 140, 60))
    cut = photo.convert("RGBA")
    texts = {slot: f"{slot} copy" for slot in tpl.text_slots()}
    out = dz.render(tpl, photo=photo, texts=texts, cutout=cut,
                    logo=Image.new("RGBA", (80, 40), (255, 0, 0, 255)))
    assert out.size == dz.canvas_size(tpl.canvas)
    assert dz.missing(tpl, photo, texts) == []
    assert tpl.note                                          # each one says what it is for
    # It drew more than a flat ground: type and bands actually landed.
    assert len(np.unique(_arr(out).reshape(-1, 3), axis=0)) > 40


def test_describe_reads_as_a_sentence():
    text = dz.describe(dz.built_in("Quote card"))
    assert text.startswith("Quote card:") and "1080×1080" in text and "text" in text


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_lists_and_shows_slots(capsys):
    assert main(["design", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Quote card" in out and "YouTube thumbnail" in out
    assert main(["design", "Event poster", "--slots"]) == 0
    slots = capsys.readouterr().out
    assert "title" in slots and "photo" in slots


def test_cli_renders_with_filled_slots(tmp_path):
    dst = tmp_path / "card.png"
    assert main(["design", "Quote card", "--set", "quote=Make it work",
                 "--set", "author=Kent Beck", "-o", str(dst)]) == 0
    with Image.open(dst) as im:
        assert im.size == (1080, 1080)


def test_cli_palette_and_canvas_override(tmp_path):
    dst = tmp_path / "wide.png"
    assert main(["design", "Quote card", "--palette", "Ocean", "--accent", "#00ff00",
                 "--canvas", "800x400", "-o", str(dst)]) == 0
    with Image.open(dst) as im:
        assert im.size == (800, 400)


def test_cli_saves_a_template_you_can_edit_and_run(tmp_path):
    path = tmp_path / "mine.json"
    assert main(["design", "Quote card", "--save", str(path)]) == 0
    data = json.loads(path.read_text())
    data["name"] = "Edited"
    path.write_text(json.dumps(data))
    dst = tmp_path / "out.png"
    assert main(["design", str(path), "--set", "quote=Hi", "-o", str(dst)]) == 0
    assert dst.exists()


def test_cli_needs_a_photo_when_the_template_does(tmp_path, capsys):
    assert main(["design", "Event poster", "-o", str(tmp_path / "p.png")]) == 2
    assert "needs --photo" in capsys.readouterr().err
    src = tmp_path / "p.jpg"
    _photo().save(src)
    assert main(["design", "Event poster", "--photo", str(src),
                 "-o", str(tmp_path / "ok.png")]) == 0


def test_cli_errors(tmp_path, capsys):
    assert main(["design"]) == 2
    assert "--list" in capsys.readouterr().err
    assert main(["design", "Nonsense"]) == 2
    assert "unknown template" in capsys.readouterr().err
    assert main(["design", "Quote card", "--set", "nonsense"]) == 2
    assert "SLOT=TEXT" in capsys.readouterr().err
    assert main(["design", "Quote card", "--set", "nope=hi"]) == 2
    assert "no slot" in capsys.readouterr().err
    assert main(["design", str(tmp_path / "gone.json")]) == 2
    assert "no template file" in capsys.readouterr().err
