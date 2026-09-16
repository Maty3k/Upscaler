"""Seeing and removing photo metadata, and proving the strip is lossless."""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from upscaler import metadata as md
from upscaler.cli import main


def _pixels(data: bytes) -> str:
    """A hash of the decoded pixels — the thing that must not change."""
    with Image.open(io.BytesIO(data)) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def _jpeg_with_metadata(lat_ref="N", lon_ref="W", orientation=6) -> bytes:
    """A JPEG carrying what a real phone photo carries."""
    img = Image.new("RGB", (120, 90), (90, 120, 160))
    ex = Image.Exif()
    ex[0x010F] = "ACME"
    ex[0x0110] = "ACME X100"
    ex[0x0112] = orientation
    ex[0x0131] = "PhotoEditor 3.1"
    ex[0x013B] = "Jane Doe"
    ex[0x8298] = "(c) Jane Doe"
    gps = ex.get_ifd(md.GPS_IFD)
    gps[1] = lat_ref
    gps[2] = (IFDRational(51), IFDRational(30), IFDRational(2634, 100))
    gps[3] = lon_ref
    gps[4] = (IFDRational(0), IFDRational(7), IFDRational(3900, 100))
    ex.get_ifd(0x8769)[0xA431] = "SN-123456789"
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92, exif=ex.tobytes(), comment=b"a private note")
    return buf.getvalue()


def _png_with_text() -> bytes:
    img = Image.new("RGB", (80, 60), (200, 80, 80))
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Jane Doe")
    info.add_text("Comment", "taken at home")
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


# ── reading ───────────────────────────────────────────────────────────────────

def test_read_finds_what_matters():
    report = md.read(_jpeg_with_metadata())
    labels = {f.label for f in report.findings}
    assert {"Camera make", "Camera model", "Software", "Artist", "Copyright",
            "Camera serial number", "Location"} <= labels
    assert report.gps == (51.507317, -0.1275)
    assert report.orientation == 6 and report.lossless
    assert "EXIF" in report.segments and "comment" in report.segments
    assert report.metadata_bytes > 0 and not report.is_clean
    # the serial number and the location are flagged, the camera model isn't
    sensitive = {f.label for f in report.sensitive}
    assert {"Camera serial number", "Location"} <= sensitive
    assert "Camera model" not in sensitive


def test_gps_hemispheres_are_signed_correctly():
    north_west = md.read(_jpeg_with_metadata("N", "W")).gps
    south_east = md.read(_jpeg_with_metadata("S", "E")).gps
    assert north_west[0] > 0 and north_west[1] < 0
    assert south_east[0] < 0 and south_east[1] > 0
    assert abs(north_west[0]) == abs(south_east[0])


def test_a_clean_file_reports_clean():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, "JPEG")
    report = md.read(buf.getvalue())
    assert report.is_clean and not report.findings
    assert "no metadata" in md.summary(report)


def test_summary_marks_the_sensitive_lines():
    text = md.summary(md.read(_jpeg_with_metadata()))
    assert "⚠" in text and "Location" in text and "51.507317" in text


# ── stripping a JPEG, losslessly ──────────────────────────────────────────────

def test_strip_all_removes_everything_without_touching_the_pixels():
    data = _jpeg_with_metadata()
    res = md.strip(data, md.REMOVE_ALL)
    assert res.lossless and not res.reencoded
    assert _pixels(res.data) == _pixels(data)          # the point of the exercise
    assert len(res.data) < len(data) and res.removed_bytes > 0
    after = md.read(res.data)
    assert after.gps is None
    assert {f.label for f in after.findings} <= {"Orientation"}
    assert "comment" not in after.segments


def test_orientation_is_kept_so_the_photo_stays_upright():
    data = _jpeg_with_metadata(orientation=6)
    kept = md.read(md.strip(data, md.REMOVE_ALL).data)
    assert kept.orientation == 6
    dropped = md.read(md.strip(data, md.REMOVE_ALL, keep_orientation=False).data)
    assert dropped.orientation == 1 and dropped.is_clean


def test_remove_location_only_keeps_the_rest():
    data = _jpeg_with_metadata()
    res = md.strip(data, md.REMOVE_LOCATION)
    assert res.lossless and _pixels(res.data) == _pixels(data)
    after = md.read(res.data)
    assert after.gps is None
    labels = {f.label for f in after.findings}
    assert "Location" not in labels
    assert {"Camera make", "Camera model", "Copyright"} <= labels


def test_keep_copyright_keeps_only_the_credit():
    data = _jpeg_with_metadata()
    after = md.read(md.strip(data, md.KEEP_COPYRIGHT).data)
    labels = {f.label for f in after.findings}
    assert {"Artist", "Copyright"} <= labels
    assert not ({"Camera make", "Camera serial number", "Location"} & labels)
    assert after.gps is None


def test_stripping_a_jpeg_with_no_metadata_is_harmless():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, "JPEG")
    data = buf.getvalue()
    res = md.strip(data, md.REMOVE_ALL)
    assert res.lossless and _pixels(res.data) == _pixels(data)
    assert md.read(res.data).is_clean


def test_bad_mode_and_bad_data_are_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        md.strip(_jpeg_with_metadata(), "burn it")
    with pytest.raises(ValueError, match="not a JPEG"):
        md.strip_jpeg(b"not an image at all")
    with pytest.raises(ValueError, match="not a PNG"):
        md.strip_png(b"not an image at all")


# ── PNG ───────────────────────────────────────────────────────────────────────

def test_png_text_is_found_and_removed_losslessly():
    data = _png_with_text()
    report = md.read(data)
    labels = {f.label for f in report.findings}
    assert "Text: Author" in labels and "Text: Comment" in labels
    assert any(f.sensitive for f in report.findings)
    assert report.segments.count("tEXt") == 2

    res = md.strip(data, md.REMOVE_ALL)
    assert res.lossless and _pixels(res.data) == _pixels(data)
    assert md.read(res.data).is_clean


def test_decoding_details_are_not_reported_as_metadata():
    """JFIF density and an ICC profile are not something the photographer
    wrote, so they must not clutter the report."""
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, "JPEG", dpi=(300, 300))
    assert md.read(buf.getvalue()).is_clean


# ── other formats fall back ───────────────────────────────────────────────────

def test_a_format_we_cannot_rewrite_is_re_encoded_and_says_so():
    buf = io.BytesIO()
    Image.new("RGB", (60, 60), (10, 200, 10)).save(buf, "WEBP", quality=90)
    res = md.strip(buf.getvalue(), md.REMOVE_ALL)
    assert res.reencoded and not res.lossless
    assert "re-encoded" in md.describe(res)


def test_describe_is_honest_in_each_case():
    data = _jpeg_with_metadata()
    assert "lossless" in md.describe(md.strip(data, md.REMOVE_ALL))
    assert "the location" in md.describe(md.strip(data, md.REMOVE_LOCATION))
    buf = io.BytesIO()
    Image.new("RGB", (30, 30)).save(buf, "PNG")
    assert md.describe(md.strip(buf.getvalue())) == "no metadata to remove"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_inspect_is_the_default_and_writes_nothing(tmp_path, capsys):
    src = tmp_path / "p.jpg"
    src.write_bytes(_jpeg_with_metadata())
    before = sorted(p.name for p in tmp_path.iterdir())
    assert main(["metadata", str(src)]) == 0
    out = capsys.readouterr()
    assert "Location" in out.out and "SN-123456789" in out.out
    assert "--remove" in out.err                         # it says how to act on it
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_cli_remove_writes_a_clean_copy(tmp_path, capsys):
    src = tmp_path / "p.jpg"
    original = _jpeg_with_metadata()
    src.write_bytes(original)
    assert main(["metadata", str(src), "--remove"]) == 0
    clean = tmp_path / "p_clean.jpg"
    assert clean.exists()
    assert _pixels(clean.read_bytes()) == _pixels(original)
    assert md.read(clean.read_bytes()).gps is None
    assert src.read_bytes() == original                  # the original is untouched
    assert "lossless" in capsys.readouterr().err


def test_cli_modes_and_in_place(tmp_path):
    src = tmp_path / "p.jpg"
    src.write_bytes(_jpeg_with_metadata())
    out = tmp_path / "loc.jpg"
    assert main(["metadata", str(src), "--remove", "--mode", md.REMOVE_LOCATION,
                 "-o", str(out)]) == 0
    after = md.read(out.read_bytes())
    assert after.gps is None and any(f.label == "Camera make" for f in after.findings)

    assert main(["metadata", str(src), "--remove", "--in-place"]) == 0
    assert md.read(src.read_bytes()).gps is None         # overwritten in place

    upright = tmp_path / "rot.jpg"
    upright.write_bytes(_jpeg_with_metadata())
    assert main(["metadata", str(upright), "--remove", "--allow-rotate",
                 "-o", str(tmp_path / "flat.jpg")]) == 0
    assert md.read((tmp_path / "flat.jpg").read_bytes()).orientation == 1


def test_cli_folder_and_errors(tmp_path, capsys):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "a.jpg").write_bytes(_jpeg_with_metadata())
    buf = io.BytesIO()
    Image.new("RGB", (30, 30)).save(buf, "JPEG")
    (src_dir / "b.jpg").write_bytes(buf.getvalue())
    out = tmp_path / "out"
    assert main(["metadata", str(src_dir), "--remove", "-o", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.jpg")) == ["a_clean.jpg", "b_clean.jpg"]
    assert main(["metadata", str(tmp_path / "nope.jpg")]) == 2
    assert "not found" in capsys.readouterr().err
