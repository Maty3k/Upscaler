"""Tests for batch processing (app.batch_process) and the 3D panel mockup."""

import zipfile

from PIL import Image

from upscaler import library, panel


def test_panel_mockup_renders(tmp_path):
    src = tmp_path / "s.png"
    Image.new("RGB", (400, 100), (30, 120, 200)).save(src)
    out = panel.mockup(str(src), panel.PanelParams())
    assert out.mode == "RGB"
    # default landscape canvas: width 1400, height = round(1400 * 0.64)
    assert out.width == 1400
    assert out.height > 400


def test_panel_mockup_portrait(tmp_path):
    src = tmp_path / "p.png"
    Image.new("RGB", (300, 300), (200, 60, 60)).save(src)
    out = panel.mockup(str(src), panel.PanelParams(orientation="Portrait · 480×1920"))
    assert out.mode == "RGB" and out.width == 1400


def test_batch_convert(tmp_path, monkeypatch):
    # keep the test's Library writes out of the real ~/.upscaler/library
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    import app  # imported here so the mockup tests don't pay the GUI import

    files = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (12, 12), (i * 40, 100, 200)).save(p)
        files.append(str(p))

    gallery, zpath, info = app.batch_process(
        files, "Convert format", "realesrgan-x4plus", "Model default (×2/×4)",
        0.0, "PNG", 90, "u2net", 0, "cpu", 0, progress=lambda *a, **k: None,
    )

    assert len(gallery) == 3
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist()) == 3
    assert "Processed" in info and "3" in info
    # results were also archived to the (temp) Library
    imgs, _vids = library.list_items()
    assert len(imgs) == 3


def test_batch_no_files_errors():
    import app

    try:
        app.batch_process([], "Upscale", "realesrgan-x4plus", "Model default (×2/×4)",
                           0.0, "PNG", 90, "u2net", 0, "cpu", 0,
                           progress=lambda *a, **k: None)
    except Exception as e:  # gr.Error
        assert "at least one image" in str(e)
    else:
        raise AssertionError("expected an error for an empty file list")
