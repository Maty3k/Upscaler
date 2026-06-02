"""Lazy, cached, integrity-checked download of pretrained weights."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from typing import Union

from tqdm import tqdm

from upscaler.models.registry import DeblurSpec, ModelSpec

WeightSpec = Union[ModelSpec, DeblurSpec]

# Cache dir: env override, else alongside the package (gitignored).
WEIGHTS_DIR = Path(
    os.environ.get("UPSCALER_WEIGHTS_DIR", Path(__file__).resolve().parent.parent / "weights")
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "upscaler/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted GitHub release URL)
        total = int(resp.headers.get("Content-Length", 0))
        # Download to a temp file first so an interrupted download never leaves a
        # corrupt file in the cache.
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"↓ {dest.name}"
            ) as bar:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
            shutil.move(str(tmp), str(dest))
        finally:
            if tmp.exists():
                tmp.unlink()


def ensure_weights(spec: WeightSpec) -> Path:
    """Return a local path to the weights for ``spec``, downloading if needed.

    If ``spec.sha256`` is set it is verified; a mismatch deletes the file and raises.
    """
    dest = WEIGHTS_DIR / spec.filename
    if not dest.exists():
        _download(spec.url, dest)

    if spec.sha256:
        digest = _sha256(dest)
        if digest != spec.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {spec.filename}: expected {spec.sha256}, "
                f"got {digest}. The file was removed; re-run to retry."
            )
    return dest
