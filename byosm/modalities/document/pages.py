from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from . import engine
from .store import STORE

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_PDFIUM = threading.Lock()

PageList = list[tuple[str, Any]]
Pending = list[tuple[int, str]]


def _render(path: Path, max_pages: int, dpi: int) -> PageList:
    from PIL import Image

    if path.suffix.lower() != ".pdf":
        return [("", Image.open(path).convert("RGB"))]
    import pypdfium2 as pdfium

    out = []
    with _PDFIUM:
        pdf = pdfium.PdfDocument(str(path))
        for i in range(min(len(pdf), max_pages)):
            page = pdf[i]
            text = page.get_textpage().get_text_range() or ""
            out.append((text, page.render(scale=dpi / 72).to_pil().convert("RGB")))
        pdf.close()
    return out


def _stored_text(k: str, img: Any, run_ocr: bool) -> str | None:
    t = STORE.get(k)
    if t is None and run_ocr:
        t = engine.ocr(img)
        STORE.put(k, t)
    return t


def read_pages(path: Path, max_pages: int, dpi: int, ocr_mode: str, run_ocr: bool = True) -> tuple[PageList, Pending]:
    out = _render(path, max_pages, dpi)
    pending: Pending = []
    if ocr_mode == "never":
        return out, pending
    for i, (txt, img) in enumerate(out):
        if ocr_mode == "always" or len(txt.strip()) < 20:
            k = STORE.key(img)
            t = _stored_text(k, img, run_ocr)
            if t is None:
                pending.append((i, k))
            else:
                out[i] = (t, img)
    return out, pending
