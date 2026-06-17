"""Tests for the Model Manager + Diagnostics report (upscaler/manage.py)."""

import pytest

from upscaler import background, manage
from upscaler.models import weights as W
from upscaler.models.registry import (
    ARTIFACT_MODELS,
    COLORIZE_MODELS,
    DEBLUR_MODELS,
    FACE_MODELS,
    INPAINT_MODELS,
    MODELS,
)


# -- Model Manager (4A) -----------------------------------------------------

def test_list_specs_covers_every_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    specs = manage.list_specs()
    expected = (
        len(MODELS) + len(DEBLUR_MODELS) + len(ARTIFACT_MODELS)
        + len(COLORIZE_MODELS) + len(INPAINT_MODELS)
        + len(FACE_MODELS) + 1  # FACE_DETECTOR
        + len(background.BG_MODELS)
    )
    assert len(specs) == expected
    assert {s.group for s in specs} == {
        "Upscale", "Clean-up", "Colorize", "Inpaint", "Faces", "Background"
    }


def test_present_and_size_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "RealESRGAN_x4plus.pth").write_bytes(b"x" * 1234)
    by_name = {s.filename: s for s in manage.list_specs()}
    hit = by_name["RealESRGAN_x4plus.pth"]
    assert hit.present and hit.size_bytes == 1234
    miss = by_name["RealESRGAN_x2plus.pth"]
    assert not miss.present and miss.size_bytes == 0


def test_total_bytes_sums_weight_files(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    (tmp_path / "a.pth").write_bytes(b"x" * 1000)
    (tmp_path / "b.onnx").write_bytes(b"x" * 500)
    (tmp_path / "junk.part").write_bytes(b"x" * 9999)  # not a weight ext
    assert manage.total_bytes() == 1500


def test_human_size_formats():
    assert manage.human_size(0) == "0 B"
    assert manage.human_size(2048).endswith("KB")
    assert manage.human_size(1572864).endswith("MB")


def test_remove_one_unlinks_and_is_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    f = tmp_path / "u2net.onnx"
    f.write_bytes(b"x")
    assert "Removed" in manage.remove_one("u2net.onnx")
    assert not f.exists()
    # path traversal + unknown filenames are refused, nothing else touched
    assert "Refused" in manage.remove_one("../evil")
    assert "Refused" in manage.remove_one("not_a_registered_file.pth")


def test_download_one_calls_ensure_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    seen = {}

    def _fake_ensure(spec):
        seen["spec"] = spec
        dest = tmp_path / spec.filename
        dest.write_bytes(b"x" * 10)
        return dest

    monkeypatch.setattr(W, "ensure_weights", _fake_ensure)
    status = manage.download_one("u2net.onnx")
    assert seen["spec"].filename == "u2net.onnx"
    assert "Ready" in status


def test_download_one_surfaces_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)

    def _boom(spec):
        raise RuntimeError("Couldn't download foo.pth (network down).")

    monkeypatch.setattr(W, "ensure_weights", _boom)
    status = manage.download_one("u2net.onnx")
    assert "network down" in status  # surfaced, not raised


# -- Diagnostics report (4B) ------------------------------------------------

def test_system_report_has_sections():
    r = manage.system_report()
    assert isinstance(r, str) and r
    for token in ("Platform", "Python", "ffmpeg", "Weights dir", "gradio"):
        assert token in r


def test_system_report_never_raises_and_reports_torch():
    r = manage.system_report()  # must not raise regardless of installed deps
    assert "torch" in r


def test_system_report_honors_weights_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "WEIGHTS_DIR", tmp_path)
    assert str(tmp_path) in manage.system_report()


def test_system_report_makes_no_network_calls(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("system_report must not hit the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert manage.system_report()  # succeeds with no network
