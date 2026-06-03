"""Image ⇄ PDF conversion.

  • images_to_pdf — combine one or more images into a (multi-page) PDF (Pillow).
  • pdf_to_images — render each PDF page to a PIL image (pypdfium2).

pypdfium2 is an optional dependency (the ``.[pdf]`` extra); it's imported lazily
so the rest of the package works without it.
"""

from __future__ import annotations

import io
from typing import Sequence, Union

from PIL import Image


def images_to_pdf(images: Sequence[Image.Image]) -> bytes:
    """Combine images into a single PDF (one page per image). Returns PDF bytes."""
    if not images:
        raise ValueError("No images provided.")
    # PDF has no alpha; flatten to RGB.
    pages = [im.convert("RGB") for im in images]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


def pdf_to_images(pdf: Union[str, bytes], dpi: int = 150) -> list[Image.Image]:
    """Render each page of a PDF (path or bytes) to a PIL image at ``dpi``."""
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover - exercised via GUI error path
        raise ImportError(
            'PDF rendering needs pypdfium2 — install with: pip install -e ".[pdf]"'
        ) from e

    scale = dpi / 72.0  # PDF user space is 72 dpi
    doc = pdfium.PdfDocument(pdf)
    try:
        return [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
    finally:
        doc.close()
