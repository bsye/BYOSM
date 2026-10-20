from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import TaskSpec
from .base import Predictor, register_baseline

IMAGE_SIDE = 1000
MIN_DOCUMENT_DPI = 100
PAGE_TEXT_CHARS = 1500
WORKERS = 8
SYSTEM = (
    "You are an expert annotator. Classify each item strictly according to the label definitions. "
    "Return exactly one label per item, in order."
)


def png_bytes(images: list[Any]) -> list[bytes]:
    out = []
    for im in images:
        im = im.copy()
        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def image_ref(first: int, last: int, count: int) -> str:
    if count > 1:
        return f" (images {first}-{last})"
    return f" (image {first})" if count else ""


@register_baseline(
    "llm",
    "The LLM itself, zero-shot, given the label definitions (role `baseline` in byosm.yaml): the System-2 bar.",
)
class ZeroShotLLM(Predictor):
    modalities, task_types = ("text", "document", "image"), ("classify",)
    wants_rows = True
    batch = {"text": 40, "document": 4, "image": 6}

    def __init__(self, spec: TaskSpec, role: str = "baseline"):
        super().__init__(spec)
        from ..llm import get_llm

        self.llm = get_llm(role)
        self.id = "llm" if role == "baseline" else f"llm:{role}"
        self.name = self.llm.name

    def _pages(self, path: Path) -> list[tuple[str, Any]]:
        if self.spec.modality == "document":
            from ..modalities.document import read_pages

            doc = self.spec.document
            return read_pages(Path(path), doc.max_pages, max(doc.dpi, MIN_DOCUMENT_DPI), doc.ocr)[0]
        from PIL import Image

        return [("", Image.open(path).convert("RGB"))]

    def _item(self, row: dict, mod: Any) -> tuple[str, list[bytes]]:
        if self.spec.modality == "text":
            return str(row["input"]), []
        path = Path(row["input"])
        if not path.is_absolute() and getattr(mod, "data_dir", None):
            path = Path(mod.data_dir) / path
        pages = self._pages(path)
        imgs = png_bytes([im for _, im in pages])
        text = "\n".join(f"[page {i + 1} text] {t[:PAGE_TEXT_CHARS]}" for i, (t, _) in enumerate(pages) if t.strip())
        return text, imgs

    def _prompt(self, chunk: list[dict], mod: Any) -> tuple[str, list[bytes]]:
        from ..data.synth_text import _context

        items, images = [], []
        for j, r in enumerate(chunk):
            text, imgs = self._item(r, mod)
            first = len(images) + 1
            images += imgs
            ref = image_ref(first, len(images), len(imgs))
            items.append(f"Item {j + 1}{ref}:" + (f"\n{text}" if text else ""))
        body = "\n\n".join(items)
        prompt = f"{_context(self.spec)}\n\n{body}\n\nReturn {len(chunk)} labels, one per item, in order."
        return prompt, images

    def _labels(self, chunk: list[dict], mod: Any, schema: dict) -> list[str]:
        prompt, images = self._prompt(chunk, mod)
        kw = {"images": images} if images else {}
        labs = self.llm.json(SYSTEM, prompt, schema, **kw).get("labels", [])
        return (labs + [self.spec.label_names[0]] * len(chunk))[: len(chunk)]

    def probs_rows(self, rows: list[dict], mod: Any) -> np.ndarray:
        from ..llm import pmap

        names = self.spec.label_names
        label_list = {"type": "array", "items": {"type": "string", "enum": names}}
        schema = {"type": "object", "properties": {"labels": label_list}}
        bs = self.batch[self.spec.modality]
        chunks = [rows[i : i + bs] for i in range(0, len(rows), bs)]
        parts = pmap(lambda chunk: self._labels(chunk, mod, schema), chunks, WORKERS)
        out = np.zeros((len(rows), len(names)), np.float32)
        for i, label in enumerate(label for part in parts for label in part):
            out[i, names.index(label) if label in names else 0] = 1.0
        return out

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        return self.probs_rows([{"input": r} for r in raws], None)

    def latency_ms(self, x: Any) -> float:
        return float("nan")
