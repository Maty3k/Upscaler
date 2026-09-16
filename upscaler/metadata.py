"""See and remove the metadata a photo carries — EXIF, XMP, IPTC, comments.

Pure Pillow, no models. A camera or phone writes a block of data alongside the
pixels and it travels with the file: where the shot was taken to within a few
metres, when, on which body and lens, that body's serial number, the editing
software, sometimes the owner's name, and often a small embedded copy of the
picture that a crop may not have regenerated.

Most large platforms strip this on upload. Forums, direct file transfers, email
attachments and your own website do not, which is exactly the situation this
app is for.

**Stripping here is lossless where it can be.** Re-saving a JPEG through Pillow
does drop the metadata, but it re-compresses the pixels and costs quality every
time. A JPEG is a sequence of segments, so the private ones can simply be left
out while the compressed image data is copied through untouched — same pixels,
byte for byte. PNG works the same way with its ancillary chunks. Anything else
falls back to a re-encode, and says so.

Orientation is the one trap. Phones commonly store a photo sideways plus a tag
saying "rotate this on display", so stripping the tag naively lays the picture
on its side. By default that single tag is preserved in a minimal EXIF block of
about thirty bytes — it says nothing about you — which keeps both the pixels
and the orientation intact.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import ExifTags, Image

# What to do with the metadata.
REMOVE_ALL = "remove everything"
REMOVE_LOCATION = "remove location only"
KEEP_COPYRIGHT = "remove everything except copyright"
MODES = [REMOVE_ALL, REMOVE_LOCATION, KEEP_COPYRIGHT]

ORIENTATION = 0x0112
ARTIST, COPYRIGHT = 0x013B, 0x8298
GPS_IFD = 0x8825

# JPEG segments. APP0 is JFIF density, APP2 an ICC colour profile and APP14 the
# Adobe colour-transform marker: none of them say anything about the
# photographer, and dropping ICC or Adobe visibly shifts colour, so they stay.
_JPEG_KEEP = {0xE0, 0xE2, 0xEE}
# APP1 is EXIF or XMP, APP13 is IPTC/Photoshop, FFFE is a free-text comment.
_JPEG_PRIVATE = {0xE1, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB,
                 0xEC, 0xED, 0xEF, 0xFE}
_EXIF_PREFIX = b"Exif\x00\x00"

# PNG chunks that carry text or metadata rather than picture.
_PNG_PRIVATE = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME", b"dSIG"}

# Keys in Image.info that are decoding details rather than something the
# photographer wrote, so they are not reported as metadata.
_INFO_SKIP = {"jfif_version", "jfif_unit", "jfif_density", "dpi", "exif", "icc_profile",
              "adobe", "adobe_transform", "progression", "progressive", "transparency",
              "aspect", "gamma", "srgb", "chromaticity", "loop", "duration", "version",
              "background", "compression", "photoshop", "xmp"}
_SENSITIVE_TEXT_KEYS = {"comment", "description", "author", "artist", "location",
                        "gps", "usercomment", "title", "owner", "email"}

# Tags worth naming in a report, in the order a person would want to read them.
_INTERESTING: "list[tuple[int, str, bool]]" = [
    (0x010F, "Camera make", False),
    (0x0110, "Camera model", False),
    (0x0131, "Software", False),
    (0x013B, "Artist", False),
    (0x8298, "Copyright", False),
    (0x0132, "File date", False),
    (0x9003, "Date taken", True),
    (ORIENTATION, "Orientation", False),
]
_EXIF_IFD_INTERESTING: "list[tuple[int, str, bool]]" = [
    (0x9003, "Date taken", True),
    (0xA431, "Camera serial number", True),
    (0xA433, "Lens make", False),
    (0xA434, "Lens model", False),
    (0xA435, "Lens serial number", True),
]


@dataclass
class Finding:
    label: str
    value: str
    sensitive: bool = False


@dataclass
class MetadataReport:
    fmt: str
    size: "tuple[int, int]"
    file_bytes: int
    metadata_bytes: int = 0
    findings: "list[Finding]" = field(default_factory=list)
    gps: "tuple[float, float] | None" = None
    has_thumbnail: bool = False
    orientation: int = 1
    segments: "list[str]" = field(default_factory=list)
    lossless: bool = True

    @property
    def sensitive(self) -> "list[Finding]":
        return [f for f in self.findings if f.sensitive]

    @property
    def is_clean(self) -> bool:
        return not self.findings and not self.gps and not self.has_thumbnail


# ── reading ───────────────────────────────────────────────────────────────────
def _rational(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return value[0] / value[1]
        except Exception:
            return 0.0


def _dms_to_degrees(dms, ref: str) -> "float | None":
    """EXIF stores a coordinate as degrees, minutes, seconds plus N/S/E/W."""
    try:
        d, m, s = (_rational(v) for v in dms)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if str(ref).upper().strip() in ("S", "W"):
        deg = -deg
    return round(deg, 6)


def gps_coordinates(exif) -> "tuple[float, float] | None":
    """(latitude, longitude) in decimal degrees, or None if not recorded."""
    try:
        gps = exif.get_ifd(GPS_IFD)
    except (KeyError, AttributeError):
        return None
    if not gps:
        return None
    lat = _dms_to_degrees(gps.get(2), gps.get(1, "N")) if gps.get(2) else None
    lon = _dms_to_degrees(gps.get(4), gps.get(3, "E")) if gps.get(4) else None
    if lat is None or lon is None:
        return None
    return lat, lon


def _clean(value) -> str:
    text = str(value).strip().replace("\x00", "")
    return text if len(text) <= 120 else text[:117] + "…"


def read(source: "str | bytes | Image.Image") -> MetadataReport:
    """Everything the file is carrying, in terms a person can act on."""
    if isinstance(source, Image.Image):
        img, data = source, None
    else:
        data = source if isinstance(source, bytes) else open(source, "rb").read()
        img = Image.open(io.BytesIO(data))
    fmt = (img.format or "?").upper()
    report = MetadataReport(fmt=fmt, size=img.size, file_bytes=len(data) if data else 0)

    try:
        exif = img.getexif()
    except Exception:
        exif = None
    if exif:
        for tag, label, sensitive in _INTERESTING:
            if tag in exif and str(exif[tag]).strip():
                value = _clean(exif[tag])
                if tag == ORIENTATION:
                    report.orientation = int(exif[tag] or 1)
                    if report.orientation in (1, 0):
                        continue
                    value = f"{report.orientation} (the photo is stored rotated)"
                report.findings.append(Finding(label, value, sensitive))
        try:
            sub = exif.get_ifd(ExifTags.IFD.Exif)
        except Exception:
            sub = {}
        for tag, label, sensitive in _EXIF_IFD_INTERESTING:
            if sub and tag in sub and str(sub[tag]).strip():
                report.findings.append(Finding(label, _clean(sub[tag]), sensitive))
        report.gps = gps_coordinates(exif)
        if report.gps:
            report.findings.append(Finding(
                "Location", f"{report.gps[0]}, {report.gps[1]}", True))
        try:
            thumb = exif.get_ifd(ExifTags.IFD.IFD1)
            report.has_thumbnail = bool(thumb) and bool(thumb.get(0x0201))
        except Exception:
            report.has_thumbnail = False
        if report.has_thumbnail:
            report.findings.append(Finding(
                "Embedded thumbnail", "a small copy of the picture is stored inside "
                "the file; if it was not regenerated after a crop it may still show "
                "what you cut out", True))

    # PNG (and some others) carry free-text entries rather than EXIF, and they
    # are exactly where "taken at home" ends up. Pillow surfaces them in .info.
    for key, value in (getattr(img, "info", None) or {}).items():
        if not isinstance(value, str) or not value.strip() or key in _INFO_SKIP:
            continue
        report.findings.append(Finding(
            f"Text: {key}", _clean(value), key.lower() in _SENSITIVE_TEXT_KEYS))

    if data:
        if fmt == "JPEG":
            report.segments, report.metadata_bytes = _jpeg_metadata_summary(data)
        elif fmt == "PNG":
            report.segments, report.metadata_bytes = _png_metadata_summary(data)
        else:
            report.lossless = False
    return report


def summary(report: MetadataReport) -> str:
    """The report as a short block of plain text."""
    if report.is_clean:
        return f"{report.fmt} {report.size[0]}×{report.size[1]} · no metadata found."
    lines = [f"{report.fmt} {report.size[0]}×{report.size[1]} · "
             f"{len(report.findings)} item(s), {report.metadata_bytes} bytes of metadata"]
    for f in report.findings:
        lines.append(f"  {'⚠ ' if f.sensitive else '  '}{f.label}: {f.value}")
    if report.segments:
        lines.append("  carried in: " + ", ".join(report.segments))
    return "\n".join(lines)


# ── JPEG: walk the segments ───────────────────────────────────────────────────
def _jpeg_segments(data: bytes):
    """Yield (marker, start, end) for each segment before the scan data."""
    if not data.startswith(b"\xff\xd8"):
        return
    i = 2
    while i < len(data) - 3 and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2:
            return
        yield marker, i, i + 2 + length
        if marker == 0xDA:      # start of scan: compressed data follows
            return
        i += 2 + length


def _segment_name(marker: int, payload: bytes) -> str:
    if marker == 0xE1:
        if payload.startswith(_EXIF_PREFIX):
            return "EXIF"
        if payload.startswith(b"http://ns.adobe.com/xap/"):
            return "XMP"
        return "APP1"
    return {0xED: "IPTC/Photoshop", 0xFE: "comment", 0xE0: "JFIF",
            0xE2: "ICC profile", 0xEE: "Adobe"}.get(marker, f"APP{marker - 0xE0}")


def _jpeg_metadata_summary(data: bytes) -> "tuple[list[str], int]":
    names, total = [], 0
    for marker, start, end in _jpeg_segments(data):
        if marker in _JPEG_PRIVATE:
            names.append(_segment_name(marker, data[start + 4:start + 40]))
            total += end - start
    return names, total


def strip_jpeg(data: bytes, keep_tags: "set[int] | None" = None,
               keep_orientation: bool = True) -> bytes:
    """Rewrite a JPEG without its private segments, copying the compressed
    image data through untouched — the pixels are identical, byte for byte.

    ``keep_tags`` names EXIF tags to carry over in a freshly built block;
    everything else in it goes. Orientation is kept by default so the picture
    doesn't end up on its side.
    """
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG")
    keep_tags = set(keep_tags or ())
    if keep_orientation:
        keep_tags.add(ORIENTATION)

    source_exif = None
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xE1 and data[start + 4:start + 10] == _EXIF_PREFIX:
            source_exif = data[start + 4:end]
            break

    rebuilt = b""
    if keep_tags and source_exif:
        old = Image.Exif()
        try:
            old.load(source_exif)
        except Exception:
            old = Image.Exif()
        kept = Image.Exif()
        for tag in keep_tags:
            if tag in old and str(old[tag]).strip():
                kept[tag] = old[tag]
        if kept:
            payload = kept.tobytes()
            rebuilt = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload

    out = bytearray(b"\xff\xd8")
    scan_from = None
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xDA:
            scan_from = start
            break
        if marker in _JPEG_PRIVATE:
            continue                       # dropped, including the old EXIF
        if marker in _JPEG_KEEP or marker not in _JPEG_PRIVATE:
            out += data[start:end]
    if rebuilt:
        # Straight after SOI, where a reader expects it.
        out[2:2] = rebuilt
    if scan_from is None:
        raise ValueError("this JPEG has no image data")
    out += data[scan_from:]
    return bytes(out)


def remove_gps_jpeg(data: bytes) -> bytes:
    """Drop the location and nothing else, rebuilding the EXIF block around
    it. Still lossless: the compressed image data is copied through."""
    source_exif = None
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xE1 and data[start + 4:start + 10] == _EXIF_PREFIX:
            source_exif = data[start + 4:end]
            break
    if source_exif is None:
        return data                        # nothing to do

    exif = Image.Exif()
    try:
        exif.load(source_exif)
    except Exception:
        return strip_jpeg(data)            # unreadable: take the whole block out
    if GPS_IFD in exif:
        del exif[GPS_IFD]
    payload = exif.tobytes()
    rebuilt = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload

    out = bytearray(b"\xff\xd8")
    scan_from = None
    for marker, start, end in _jpeg_segments(data):
        if marker == 0xDA:
            scan_from = start
            break
        if marker == 0xE1 and data[start + 4:start + 10] == _EXIF_PREFIX:
            out += rebuilt
            continue
        out += data[start:end]
    out += data[scan_from:]
    return bytes(out)


# ── PNG: walk the chunks ──────────────────────────────────────────────────────
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunks(data: bytes):
    """Yield (name, start, end) for each chunk."""
    if not data.startswith(_PNG_SIG):
        return
    i = len(_PNG_SIG)
    while i + 8 <= len(data):
        length = int.from_bytes(data[i:i + 4], "big")
        name = data[i + 4:i + 8]
        end = i + 12 + length
        yield name, i, end
        if name == b"IEND":
            return
        i = end


def _png_metadata_summary(data: bytes) -> "tuple[list[str], int]":
    names, total = [], 0
    for name, start, end in _png_chunks(data):
        if name in _PNG_PRIVATE:
            names.append(name.decode("ascii", "replace"))
            total += end - start
    return names, total


def strip_png(data: bytes) -> bytes:
    """Drop a PNG's text and metadata chunks. Lossless by construction: the
    image chunks are copied through untouched."""
    if not data.startswith(_PNG_SIG):
        raise ValueError("not a PNG")
    out = bytearray(_PNG_SIG)
    for name, start, end in _png_chunks(data):
        if name in _PNG_PRIVATE:
            continue
        out += data[start:end]
    return bytes(out)


# ── the one entry point ───────────────────────────────────────────────────────
@dataclass
class StripResult:
    data: bytes
    lossless: bool
    removed_bytes: int
    mode: str
    report_before: MetadataReport
    reencoded: bool = False


def strip(source: "str | bytes", mode: str = REMOVE_ALL,
          keep_orientation: bool = True) -> StripResult:
    """Remove metadata according to ``mode``.

    JPEG and PNG are rewritten losslessly. Any other format is decoded and
    re-encoded without its metadata, which does cost a generation of quality —
    ``StripResult.reencoded`` says so, and the caller should pass that on.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} — expected one of {MODES}")
    data = source if isinstance(source, bytes) else open(source, "rb").read()
    before = read(data)

    if before.fmt == "JPEG":
        if mode == REMOVE_LOCATION:
            out = remove_gps_jpeg(data)
        elif mode == KEEP_COPYRIGHT:
            out = strip_jpeg(data, keep_tags={ARTIST, COPYRIGHT},
                             keep_orientation=keep_orientation)
        else:
            out = strip_jpeg(data, keep_orientation=keep_orientation)
        return StripResult(out, True, max(0, len(data) - len(out)), mode, before)

    if before.fmt == "PNG":
        # A PNG's EXIF lives in one chunk, so "location only" can't be done
        # without rewriting it; removing the lot is both simpler and safer.
        out = strip_png(data)
        return StripResult(out, True, max(0, len(data) - len(out)), mode, before)

    img = Image.open(io.BytesIO(data))
    if keep_orientation and before.orientation not in (0, 1):
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)     # bake it in; the tag is going
    buf = io.BytesIO()
    save_fmt = img.format or "PNG"
    img.save(buf, save_fmt)
    out = buf.getvalue()
    return StripResult(out, False, max(0, len(data) - len(out)), mode, before,
                       reencoded=True)


def describe(result: StripResult) -> str:
    """A short human summary of what was removed."""
    before = result.report_before
    if before.is_clean and not result.removed_bytes and not result.reencoded:
        return "no metadata to remove"
    bits = []
    if before.findings:
        bits.append(f"removed {len(before.findings)} item(s)")
    if before.gps:
        bits.append("the location" if result.mode == REMOVE_LOCATION
                    else "including the location")
    if result.removed_bytes:
        bits.append(f"{result.removed_bytes} bytes smaller")
    bits.append("lossless — the pixels are untouched" if result.lossless
                else "⚠ re-encoded, so the picture lost a generation of quality")
    return " · ".join(bits) if bits else "nothing changed"
