"""The "faces" region: the mask built from detected boxes, and the CLI
plumbing for both `blur` and `adjust`.

The mask itself takes boxes as fractions, so almost everything here runs with
no OpenCV and no model download. The one test that exercises the real detector
is skipped unless OpenCV is installed *and* the detector is already cached, so
the suite never reaches the network.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from upscaler import adjust, blur
from upscaler.cli import main
from upscaler.models.registry import FACE_DETECTOR
from upscaler.models.weights import WEIGHTS_DIR

try:
    import cv2
except ImportError:
    cv2 = None

_DETECTOR_CACHED = (WEIGHTS_DIR / FACE_DETECTOR.filename).is_file()


def _photo(w=400, h=200):
    return Image.new("RGB", (w, h), (120, 120, 120))


# ── the mask ──────────────────────────────────────────────────────────────────

def test_faces_is_a_region_shape():
    assert "faces" in blur.SHAPES


def test_faces_mask_draws_an_oval_per_box():
    boxes = [(0.10, 0.20, 0.20, 0.30), (0.60, 0.10, 0.15, 0.20)]
    m = blur.MaskParams(shape="faces", faces=boxes, feather=0, face_pad=0)
    mask = np.asarray(blur.build_mask((400, 200), m))
    assert mask[70, 80] == 255 and mask[40, 270] == 255     # inside each oval
    assert mask[190, 380] == 0 and mask[10, 10] == 0        # outside both
    # the corners of a box are outside its inscribed ellipse
    assert mask[41, 41] == 0


def test_faces_mask_scales_with_the_image_so_preview_matches_export():
    boxes = [(0.25, 0.25, 0.5, 0.5)]
    m = blur.MaskParams(shape="faces", faces=boxes, feather=0, face_pad=0)
    small = np.asarray(blur.build_mask((100, 100), m))
    big = np.asarray(blur.build_mask((800, 800), m))
    assert small[50, 50] == 255 and big[400, 400] == 255
    assert small[10, 10] == 0 and big[80, 80] == 0
    assert small.mean() == pytest.approx(big.mean(), abs=2)   # same coverage


def test_face_padding_grows_and_shrinks_the_oval():
    boxes = [(0.4, 0.4, 0.2, 0.2)]
    cover = lambda pad: np.asarray(blur.build_mask(  # noqa: E731
        (200, 200), blur.MaskParams(shape="faces", faces=boxes, feather=0, face_pad=pad))).mean()
    assert cover(-20) < cover(0) < cover(25) < cover(100)


def test_faces_mask_empty_and_outside():
    empty = np.asarray(blur.build_mask((40, 40), blur.MaskParams(shape="faces")))
    assert empty.max() == 0                                  # nothing detected → nothing masked
    boxes = [(0.25, 0.25, 0.5, 0.5)]
    inv = np.asarray(blur.build_mask((100, 100), blur.MaskParams(
        shape="faces", faces=boxes, feather=0, outside=True)))
    assert inv[50, 50] == 0 and inv[5, 5] == 255             # portrait mode: background only


def test_blur_and_adjust_apply_through_a_faces_mask():
    img = Image.new("RGB", (200, 200), (30, 60, 200))
    boxes = [(0.3, 0.3, 0.4, 0.4)]
    m = blur.MaskParams(shape="faces", faces=boxes, feather=0, face_pad=0)
    pix = blur.apply(img, blur.BlurParams(kind="pixelate", strength=60), m)
    assert np.array_equal(np.asarray(pix)[5, 5], np.asarray(img)[5, 5])      # outside untouched
    bright = adjust.apply(img, adjust.AdjustParams(exposure=60), m)
    assert np.asarray(bright)[100, 100].mean() > np.asarray(img)[100, 100].mean()
    assert np.array_equal(np.asarray(bright)[5, 5], np.asarray(img)[5, 5])


def test_describe_counts_the_faces():
    one = blur.MaskParams(shape="faces", faces=[(0.1, 0.1, 0.2, 0.2)])
    two = blur.MaskParams(shape="faces", faces=[(0.1, 0.1, 0.2, 0.2), (0.5, 0.1, 0.2, 0.2)],
                          outside=True)
    assert "inside the 1 face" in blur.describe(blur.BlurParams(), one, (100, 100))
    assert "outside the 2 faces" in blur.describe(blur.BlurParams(), two, (100, 100))
    assert "inside the 1 face" in adjust.describe(adjust.AdjustParams(exposure=5), one)


# ── CLI plumbing (detector faked, so no OpenCV and no download) ───────────────

@pytest.fixture
def fake_detect(monkeypatch):
    """Pretend one face sits in the middle of every image."""
    from upscaler import face

    calls = []

    def detect(image, confidence=0.6):
        calls.append((image.size, confidence))
        return [face.Face(x=0.3, y=0.3, w=0.4, h=0.4, confidence=0.9)]

    monkeypatch.setattr(face, "detect_faces", detect)
    return calls


def test_cli_blur_faces(tmp_path, fake_detect, capsys):
    src = tmp_path / "p.png"
    _photo().save(src)
    out = tmp_path / "hidden.png"
    assert main(["blur", str(src), "-o", str(out), "--shape", "faces", "--kind", "pixelate",
                 "--strength", "50", "--face-pad", "35", "--face-confidence", "0.4"]) == 0
    assert out.exists()
    err = capsys.readouterr().err
    assert "1 face(s) found" in err and "inside the 1 face" in err
    assert fake_detect[0][1] == 0.4                          # --face-confidence is passed through


def test_cli_adjust_faces(tmp_path, fake_detect):
    src = tmp_path / "p.png"
    _photo().save(src)
    out = tmp_path / "bright.png"
    assert main(["adjust", str(src), "-o", str(out), "--shape", "faces", "--exposure", "40"]) == 0
    a, b = np.asarray(Image.open(src), np.float32), np.asarray(Image.open(out), np.float32)
    assert b[100, 200].mean() > a[100, 200].mean()           # the face area is brighter
    assert np.array_equal(b[5, 5], a[5, 5])                  # the rest is untouched


def test_cli_skips_images_with_no_faces(tmp_path, monkeypatch, capsys):
    from upscaler import face

    monkeypatch.setattr(face, "detect_faces", lambda image, confidence=0.6: [])
    src = tmp_path / "p.png"
    _photo().save(src)
    out = tmp_path / "out.png"
    assert main(["blur", str(src), "-o", str(out), "--shape", "faces"]) == 0
    assert not out.exists()                                  # no identical copy written
    assert "no faces found (skipped)" in capsys.readouterr().err


def test_cli_reports_a_missing_opencv_instead_of_crashing(tmp_path, monkeypatch, capsys):
    from upscaler import face

    def boom(image, confidence=0.6):
        raise RuntimeError(face.DETECT_HINT)

    monkeypatch.setattr(face, "detect_faces", boom)
    src = tmp_path / "p.png"
    _photo().save(src)
    assert main(["blur", str(src), "-o", str(tmp_path / "o.png"), "--shape", "faces"]) == 1
    assert "OpenCV" in capsys.readouterr().err


# ── the real detector ─────────────────────────────────────────────────────────

@pytest.mark.skipif(cv2 is None or not _DETECTOR_CACHED,
                    reason="needs OpenCV and the cached YuNet detector")
def test_detect_faces_returns_fractions_and_finds_nothing_in_a_blank_image():
    from upscaler import face

    assert face.detect_faces(Image.new("RGB", (320, 240), (200, 190, 180))) == []
    # every returned box must be a fraction of the image, in left-to-right order
    img = Image.new("RGB", (640, 480), (128, 128, 128))
    for f in face.detect_faces(img):
        assert 0.0 <= f.x <= 1.0 and 0.0 <= f.w <= 1.0 and f.box() == (f.x, f.y, f.w, f.h)
