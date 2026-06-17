"""CLI parity tests — removebg, batch, and --face. All model work is faked, so
these run fully offline (no weights, no torch inference, no onnxruntime)."""

import sys

import numpy as np
import pytest
from PIL import Image

from upscaler.cli import main


def _make(path, mode="RGB", size=(20, 16)):
    arr = (np.random.rand(size[1], size[0], len(mode)) * 255).astype("uint8")
    Image.fromarray(arr.squeeze(), mode=mode).save(path)


class _FakeUp:
    spec = type("S", (), {"name": "fake"})()
    scale = 2
    device = type("D", (), {"type": "cpu"})()

    def __init__(self, **k):
        pass

    def upscale(self, img):
        return img.convert("RGB")


def _fake_removebg(img, **k):
    return img.convert("RGBA")


# -- removebg ---------------------------------------------------------------

def test_removebg_single_file(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.background.remove_background", _fake_removebg)
    src = tmp_path / "a.png"
    _make(src)
    dst = tmp_path / "out.png"
    assert main(["removebg", str(src), "-o", str(dst)]) == 0
    assert dst.exists() and Image.open(dst).mode == "RGBA"


def test_removebg_dir_to_single_file_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("upscaler.background.remove_background", _fake_removebg)
    for n in ("a", "b"):
        _make(tmp_path / f"{n}.png")
    assert main(["removebg", str(tmp_path), "-o", str(tmp_path / "x.png")]) == 2
    assert "must be a directory" in capsys.readouterr().err


def test_removebg_missing_input_errors():
    assert main(["removebg", "/no/such/file.png"]) == 2


# -- batch ------------------------------------------------------------------

def test_batch_upscale_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.cli.Upscaler", _FakeUp)
    for n in ("a", "b", "c"):
        _make(tmp_path / f"{n}.png")
    out = tmp_path / "out"
    rc = main(["batch", str(tmp_path), "-o", str(out), "--op", "upscale", "--scale", "2"])
    assert rc == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["a_x2.png", "b_x2.png", "c_x2.png"]


def test_batch_convert_skips_unreadable(tmp_path, capsys):
    _make(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes(b"not an image")  # unreadable -> skipped
    out = tmp_path / "out"
    rc = main(["batch", str(tmp_path), "-o", str(out), "--op", "convert", "-f", "WebP"])
    assert rc == 0  # one good file -> success despite the skip
    assert (out / "a.webp").exists()
    assert "skipped" in capsys.readouterr().err


def test_batch_removebg_op(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.background.remove_background", _fake_removebg)
    _make(tmp_path / "a.png")
    out = tmp_path / "out"
    assert main(["batch", str(tmp_path), "-o", str(out), "--op", "removebg"]) == 0
    assert Image.open(out / "a.png").mode == "RGBA"


def test_batch_output_must_be_dir(tmp_path):
    _make(tmp_path / "a.png")
    assert main(["batch", str(tmp_path), "-o", str(tmp_path / "x.png")]) == 2


# -- --face flag ------------------------------------------------------------

def test_face_flag_friendly_message_when_extra_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("upscaler.cli.Upscaler", _FakeUp)

    def _raise(**k):
        raise RuntimeError(
            'Face restoration needs extra packages. Install them with: '
            'pip install -e ".[face]"'
        )

    monkeypatch.setattr("upscaler.face.FaceRestorer", _raise)
    src = tmp_path / "a.png"
    _make(src)
    rc = main([str(src), "-o", str(tmp_path / "o.png"), "--face"])
    assert rc == 2
    assert 'pip install -e ".[face]"' in capsys.readouterr().err


def test_face_flag_invokes_restorer_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.cli.Upscaler", _FakeUp)
    seen = {}

    class _FakeFR:
        def __init__(self, **k):
            pass

        def restore(self, img, strength):
            seen["strength"] = strength
            return img

    monkeypatch.setattr("upscaler.face.FaceRestorer", _FakeFR)
    src = tmp_path / "a.png"
    _make(src)
    rc = main([str(src), "-o", str(tmp_path / "o.png"), "--face", "--face-strength", "0.5"])
    assert rc == 0
    assert seen.get("strength") == 0.5


def test_bare_upscale_without_face_does_not_import_face(tmp_path, monkeypatch):
    monkeypatch.setattr("upscaler.cli.Upscaler", _FakeUp)
    monkeypatch.delitem(sys.modules, "upscaler.face", raising=False)
    src = tmp_path / "a.png"
    _make(src)
    assert main([str(src), "-o", str(tmp_path / "o.png"), "--scale", "2"]) == 0
    assert "upscaler.face" not in sys.modules  # torch-only users never pull [face]
