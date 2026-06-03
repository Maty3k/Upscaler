"""CLI `pdf build` / `pdf extract` tests."""

import numpy as np
from PIL import Image

from upscaler.cli import main


def _make(path, size=(40, 30)):
    Image.fromarray(np.full((size[1], size[0], 3), 90, "uint8"), "RGB").save(path)


def test_pdf_build_then_extract_roundtrip(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _make(a)
    _make(b)
    pdf = tmp_path / "out.pdf"

    assert main(["pdf", "build", str(a), str(b), "-o", str(pdf)]) == 0
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"

    out = tmp_path / "pages"
    assert main(["pdf", "extract", str(pdf), "-o", str(out), "--dpi", "72"]) == 0
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 2


def test_pdf_build_from_directory(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    for n in ("1", "2", "3"):
        _make(src / f"{n}.png")
    pdf = tmp_path / "all.pdf"
    assert main(["pdf", "build", str(src), "-o", str(pdf)]) == 0
    out = tmp_path / "pages"
    main(["pdf", "extract", str(pdf), "-o", str(out), "--dpi", "72"])
    assert len(list(out.glob("*.png"))) == 3


def test_pdf_extract_default_output_dir(tmp_path):
    a = tmp_path / "a.png"
    _make(a)
    pdf = tmp_path / "doc.pdf"
    main(["pdf", "build", str(a), "-o", str(pdf)])
    assert main(["pdf", "extract", str(pdf), "--dpi", "72"]) == 0
    assert (tmp_path / "doc_pages" / "doc_p001.png").exists()


def test_pdf_build_missing_input_errors(tmp_path):
    assert main(["pdf", "build", str(tmp_path / "nope.png"), "-o", str(tmp_path / "x.pdf")]) == 2
