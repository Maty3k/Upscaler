"""Tests for the export Library (upscaler.library)."""

from PIL import Image

from upscaler import library


def test_save_image_and_path_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")

    # a PIL image is archived as a timestamped PNG under its kind
    img = Image.new("RGB", (8, 8), "red")
    p = library.save_image(img, "upscale")
    assert p is not None and p.exists()
    assert p.name.startswith("upscale_") and p.suffix == ".png"

    # an existing file (e.g. a rendered mp4) is copied in, extension preserved
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x01\x02")
    q = library.save_path(clip, "video")
    assert q is not None and q.exists()
    assert q.name.startswith("video_") and q.suffix == ".mp4"

    imgs, vids = library.list_items()
    assert len(imgs) == 1 and len(vids) == 1
    assert imgs[0].endswith(".png") and vids[0].endswith(".mp4")


def test_save_path_missing_file_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "lib")
    # best-effort: a non-existent source returns None instead of raising
    assert library.save_path(tmp_path / "nope.png", "convert") is None


def test_list_items_empty_when_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "does-not-exist")
    assert library.list_items() == ([], [])
