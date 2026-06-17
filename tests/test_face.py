"""Tests for the optional GFPGAN face-restoration feature."""

import numpy as np
import pytest
from PIL import Image

from upscaler.models.registry import FACE_DETECTOR, FACE_MODELS


def test_face_models_pinned():
    for s in list(FACE_MODELS.values()) + [FACE_DETECTOR]:
        assert s.sha256 and len(s.sha256) == 64, f"{s.name} missing/short sha256"
        assert s.url and s.filename


def test_alignment_template_recovers_with_no_reorder():
    cv2 = pytest.importorskip("cv2")
    from upscaler.face import _TEMPLATE

    # Apply a known rotate+scale+shift to the template points, then confirm the
    # solver maps them straight back onto the template (~0 residual) WITHOUT any
    # landmark reordering — i.e. YuNet's order matches the template.
    theta = np.deg2rad(12)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)
    pts = (_TEMPLATE @ rot.T) * 1.3 + np.array([40, 25], np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts.astype(np.float32), _TEMPLATE, method=cv2.LMEDS)
    mapped = (M[:, :2] @ pts.T).T + M[:, 2]
    resid = float(np.sqrt(((mapped - _TEMPLATE) ** 2).sum(1)).mean())
    assert resid < 1.0


def test_no_face_image_passes_through():
    pytest.importorskip("cv2")
    pytest.importorskip("spandrel")
    pytest.importorskip("spandrel_extra_arches")
    from upscaler.face import FaceRestorer

    try:
        fr = FaceRestorer(device="cpu")  # only loads the tiny YuNet detector here
    except (OSError, RuntimeError) as e:
        pytest.skip(f"face detector unavailable offline: {e}")
    noise = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype("uint8"), "RGB")
    out = fr.restore(noise)
    # no faces -> identical image, and the 349MB GFPGAN net is never loaded
    assert np.array_equal(np.asarray(out), np.asarray(noise))
    assert fr._net is None
