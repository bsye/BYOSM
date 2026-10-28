from __future__ import annotations

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path
from typing import Any

GRID = 6
COVER = (128, 128, 128)
TRIED_DIR = "tried"
FILE_TYPES = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
MAX_BYTES = 20 * 1024 * 1024
HASH_CHARS = 16
TEXT_CHARS = 600
DIGITS = 4
BASES = 32


def _folder() -> Path:
    from ..config import cache_root

    return cache_root() / TRIED_DIR


def save_tried(body: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in FILE_TYPES:
        raise ValueError(f"send a {', '.join(FILE_TYPES)} file")
    if len(body) > MAX_BYTES:
        raise ValueError(f"the file is larger than {MAX_BYTES // 1024 // 1024} MB")
    name = hashlib.sha1(body).hexdigest()[:HASH_CHARS] + suffix
    path = _folder() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return name


def tried_path(name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    if Path(name).name != name or suffix not in FILE_TYPES or not stem.isalnum():
        raise ValueError(f"{name} is not a tried file")
    path = _folder() / name
    if not path.exists():
        raise FileNotFoundError(f"{name} was not uploaded")
    return path


def _pages(model: Any, path: Path) -> list[tuple[str, Any]]:
    from ..rt import read_document

    return read_document(path, model.m["frontend"])


def _thumb(img: Any) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def page(model: Any, path: Path) -> dict[str, Any]:
    pages = _pages(model, path)
    out = model.predict(pages)
    text, img = pages[0]
    return {**out, "page": _thumb(img), "pages": len(pages), "text": text[:TEXT_CHARS], "grid": GRID}


def _covered(img: Any, row: int, col: int) -> Any:
    from PIL import ImageDraw

    copy, w, h = img.copy(), img.width / GRID, img.height / GRID
    box = (col * w, row * h, (col + 1) * w, (row + 1) * h)
    ImageDraw.Draw(copy).rectangle(box, fill=COVER)
    return copy


@lru_cache(maxsize=BASES)
def _base(model: Any, path: Path, label: str) -> float:
    return model.predict(_pages(model, path))["probs"][label]


def occlusion_row(model: Any, path: Path, row: int, label: str) -> list[float]:
    if not 0 <= row < GRID:
        raise ValueError(f"row is 0 to {GRID - 1}")
    pages = _pages(model, path)
    (text, img), rest = pages[0], pages[1:]
    base = _base(model, path, label)
    covered = [model.predict([(text, _covered(img, row, col)), *rest])["probs"][label] for col in range(GRID)]
    return [round(base - p, DIGITS) for p in covered]


def _answer(out: dict[str, Any], label: str) -> dict[str, Any]:
    return {"label": out["label"], "p": round(out["probs"][label], DIGITS)}


def views(model: Any, path: Path, label: str) -> dict[str, Any]:
    from PIL import Image

    pages = _pages(model, path)
    no_picture = [(text, Image.new("RGB", img.size, COVER)) for text, img in pages]
    no_text = [("", img) for _, img in pages]
    return {"picture": _answer(model.predict(no_picture), label), "text": _answer(model.predict(no_text), label)}
