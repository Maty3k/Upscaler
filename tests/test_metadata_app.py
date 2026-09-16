"""App-level wiring for the metadata privacy mode in the Convert tab."""

import io

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational


def _jpeg(tmp_path, name="p.jpg"):
    from upscaler import metadata as md

    img = Image.new("RGB", (120, 90), (90, 120, 160))
    ex = Image.Exif()
    ex[0x010F] = "ACME"
    ex[0x0112] = 6
    gps = ex.get_ifd(md.GPS_IFD)
    gps[1] = "N"; gps[2] = (IFDRational(51), IFDRational(30), IFDRational(0))
    gps[3] = "W"; gps[4] = (IFDRational(0), IFDRational(7), IFDRational(0))
    path = tmp_path / name
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=ex.tobytes())
    path.write_bytes(buf.getvalue())
    return path


def test_app_convert_has_the_privacy_mode():
    pytest.importorskip("gradio")
    import app

    assert "Remove metadata (privacy)" in app._CONVERT_METHODS
    updates = app._switch_method("Remove metadata (privacy)")
    assert len(updates) == len(app._CONVERT_METHODS)
    assert updates[2]["visible"] and not updates[0]["visible"]


def test_app_inspect_reports_and_reveals_the_button(tmp_path):
    pytest.importorskip("gradio")
    import app

    preview, report, button = app.metadata_inspect_ui(None)
    assert preview is None and not button["visible"]

    preview, report, button = app.metadata_inspect_ui(str(_jpeg(tmp_path)))
    assert preview.size == (120, 90) and button["visible"]
    assert "records where it was taken" in report["value"]
    assert "Camera make" in report["value"]


def test_app_inspect_handles_a_file_that_is_not_an_image(tmp_path):
    pytest.importorskip("gradio")
    import app

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    preview, report, button = app.metadata_inspect_ui(str(bad))
    assert preview is None and not button["visible"]
    assert "⚠" in report["value"]


def test_app_inspect_says_so_when_a_file_is_clean(tmp_path):
    pytest.importorskip("gradio")
    import app

    clean = tmp_path / "clean.jpg"
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, "JPEG")
    clean.write_bytes(buf.getvalue())
    _preview, report, button = app.metadata_inspect_ui(str(clean))
    assert "already clean" in report["value"] and button["visible"]


def test_app_strip_writes_a_clean_file(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app
    from upscaler import library, metadata as md

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    src = _jpeg(tmp_path)
    out, note = app.metadata_strip_ui(str(src), md.REMOVE_ALL, True,
                                      progress=lambda *a, **k: None)
    assert out.endswith("_clean.jpg")
    with open(out, "rb") as fh:
        cleaned = fh.read()
    assert md.read(cleaned).gps is None
    assert "lossless" in note and "Orientation" in note      # says what it kept
    with pytest.raises(gr.Error):
        app.metadata_strip_ui(None, md.REMOVE_ALL, True, progress=lambda *a, **k: None)
