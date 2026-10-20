from __future__ import annotations

import hashlib
import os
from functools import cached_property
from pathlib import Path
from typing import Any


class OCRStore:
    def __init__(self, root: Path | None = None):
        base = Path(os.environ.get("BYOSM_CACHE") or Path.home() / ".cache" / "byosm")
        self.root = root or base / "ocr"

    @cached_property
    def tag(self) -> str:
        try:
            from importlib.metadata import version

            return "rapidocr-" + version("rapidocr_onnxruntime")
        except Exception:
            return "rapidocr"

    def key(self, img: Any) -> str:
        h = hashlib.sha256(f"{self.tag}|{img.mode}|{img.size}|".encode())
        h.update(img.tobytes())
        return h.hexdigest()[:40]

    def _file(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.txt"

    def get(self, key: str) -> str | None:
        f = self._file(key)
        return f.read_text() if f.exists() else None

    def put(self, key: str, text: str) -> None:
        f = self._file(key)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(text)
            tmp.replace(f)
        except OSError:
            pass


STORE = OCRStore()
