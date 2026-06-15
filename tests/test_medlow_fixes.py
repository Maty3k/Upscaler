"""Reproduce-then-verify tests for the medium/low bug fixes:
batch temp-dir cleanup, CLI corrupt-input resilience, and friendly errors when a
model can't run (e.g. an unavailable device).
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from upscaler import cli, library


def _png(path, color=(60, 120, 200), size=(8, 8)):
    Image.new("RGB", size, color).save(path)


def test_batch_cleans_up_work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    import app

    created = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*a, **k):
        d = real_mkdtemp(*a, **k)
        created.append(d)
        return d

    monkeypatch.setattr(app.tempfile, "mkdtemp", spy)
    files = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        _png(p)
        files.append(str(p))
    app.batch_process(files, "Convert format", "realesrgan-x4plus",
                      "Model default (×2/×4)", 0.0, "PNG", 90, "u2net", 0, "cpu", 0,
                      progress=lambda *a, **k: None)
    assert created, "batch_process should create a work dir"
    assert not os.path.exists(created[0]), "work dir must be cleaned up, not leaked"


def test_cli_convert_skips_corrupt_and_continues(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    _png(d / "good.png")
    (d / "bad.png").write_bytes(b"not an image")
    out = tmp_path / "out"
    rc = cli.run_convert([str(d), "-f", "PNG", "-o", str(out)])
    assert rc == 0  # one succeeded → overall success, batch not aborted
    assert "error on bad.png" in capsys.readouterr().err
    assert (out / "good.png").exists()


def test_cli_convert_all_corrupt_returns_2(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "bad.png").write_bytes(b"nope")
    rc = cli.run_convert([str(d), "-f", "PNG", "-o", str(tmp_path / "out")])
    assert rc == 2


def test_cli_pdf_build_skips_corrupt(tmp_path, capsys):
    good = tmp_path / "good.png"
    _png(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    outpdf = tmp_path / "o.pdf"
    rc = cli.run_pdf(["build", str(good), str(bad), "-o", str(outpdf)])
    assert rc == 0 and outpdf.exists()
    assert "skipped" in capsys.readouterr().err


def test_enhance_unavailable_device_raises_friendly_error():
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is actually available here")
    import app

    img = Image.fromarray((np.random.rand(8, 8, 3) * 255).astype("uint8"), "RGB")
    with pytest.raises(Exception) as ei:  # gr.Error (not a raw AssertionError)
        app.enhance(img, "realesrgan-x4plus", "cuda", False, "nafnet-sidd-width64",
                    1.0, 0.0, 0, False, "Model default (×2/×4)")
    assert "Device" in str(ei.value) or "auto" in str(ei.value)


def test_list_items_tolerates_missing_entries(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    monkeypatch.setattr(library, "LIBRARY_DIR", lib)
    library.save_image(Image.new("RGB", (4, 4), "red"), "upscale")
    # A dangling entry (e.g. a file removed mid-listing, simulated with a broken
    # symlink) must be skipped, not crash the listing.
    try:
        (lib / "gone.png").symlink_to(lib / "does-not-exist.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    imgs, vids = library.list_items()  # must not raise
    assert len(imgs) == 1 and vids == []  # real image kept, dangling entry skipped
