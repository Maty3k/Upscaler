"""Persistent library of everything the app exports.

Every result the GUI produces (upscales, restores, converted images, PDFs,
removed-background cut-outs, Lian Li panel exports, upscaled videos) is copied
here automatically so creations are easy to find and reuse later. Files live in
``~/.upscaler/library`` (override with ``UPSCALER_LIBRARY``) and are named
``<kind>_<timestamp><ext>`` so they sort newest-first and are self-describing.

All save helpers are best-effort: they never raise, so a failure to archive can
never break the export the user actually asked for.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

LIBRARY_DIR = Path(
    os.environ.get("UPSCALER_LIBRARY", str(Path.home() / ".upscaler" / "library"))
)

# What counts as a browsable image vs. a video in the Library tab.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".gif",
              ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


def ensure_dir() -> Path:
    """Create the library folder if needed and return it."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR


def _stamp() -> str:
    # millisecond precision so rapid successive saves don't collide
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _unique(dest: Path) -> Path:
    """Return ``dest``, or ``dest-1``/``dest-2``… if it already exists, so rapid
    batch saves (which can share a millisecond timestamp) never overwrite."""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 1
    while (cand := dest.with_name(f"{stem}-{i}{suffix}")).exists():
        i += 1
    return cand


def save_path(src: "str | os.PathLike", kind: str) -> "Path | None":
    """Copy an exported file into the library as ``<kind>_<timestamp><ext>``."""
    try:
        src = Path(src)
        if not src.is_file():
            return None
        dest = _unique(ensure_dir() / f"{kind}_{_stamp()}{src.suffix.lower()}")
        shutil.copyfile(src, dest)
        return dest
    except OSError:
        return None


def save_image(img, kind: str, fmt: str = "PNG") -> "Path | None":
    """Save a PIL image into the library as ``<kind>_<timestamp>.<fmt>``."""
    try:
        ext = ".png" if fmt.upper() == "PNG" else f".{fmt.lower()}"
        dest = _unique(ensure_dir() / f"{kind}_{_stamp()}{ext}")
        img.save(dest, fmt)
        return dest
    except (OSError, ValueError, KeyError):
        return None


def list_items() -> "tuple[list[str], list[str]]":
    """Return ``(images_and_gifs, videos)`` as path strings, newest first.

    Tolerant of files deleted concurrently (e.g. the user clears items from the
    library folder while the tab refreshes): such files are skipped, never raised.
    """
    if not LIBRARY_DIR.exists():
        return [], []
    pairs = []
    for p in LIBRARY_DIR.iterdir():
        try:
            if p.is_file():
                pairs.append((p.stat().st_mtime, p))
        except OSError:  # vanished between iterdir() and stat() — skip it
            continue
    pairs.sort(key=lambda mp: mp[0], reverse=True)
    files = [p for _, p in pairs]
    imgs = [str(p) for p in files if p.suffix.lower() in IMAGE_EXTS]
    vids = [str(p) for p in files if p.suffix.lower() in VIDEO_EXTS]
    return imgs, vids
