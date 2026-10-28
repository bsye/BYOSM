from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..rt import OUTSIDE
from ..spec import TaskSpec
from .base import Predictor, register_baseline

IMAGE_SIDE = 1000
MIN_DOCUMENT_DPI = 100
PAGE_TEXT_CHARS = 1500
TAG_BATCH = 10
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


def _workers() -> int:
    from ..llm import parallel_calls

    return parallel_calls()


@register_baseline(
    "llm",
    "The LLM itself, zero-shot, given the label definitions (role `baseline` in byosm.yaml): the System-2 bar.",
)
class ZeroShotLLM(Predictor):
    modalities, task_types = ("text", "document", "image"), ("classify", "frame")
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
        from ..data.prompts import task_context

        items, images = [], []
        for j, r in enumerate(chunk):
            text, imgs = self._item(r, mod)
            first = len(images) + 1
            images += imgs
            ref = image_ref(first, len(images), len(imgs))
            items.append(f"Item {j + 1}{ref}:" + (f"\n{text}" if text else ""))
        body = "\n\n".join(items)
        prompt = f"{task_context(self.spec, self.label_space)}\n\n{body}\n\nReturn one label for each of the {len(chunk)} items, each with its item number n."
        return prompt, images

    def _labels(self, chunk: list[dict], mod: Any, schema: dict) -> list[str]:
        prompt, images = self._prompt(chunk, mod)
        kw = {"images": images} if images else {}
        answers = self.llm.json(SYSTEM, prompt, schema, **kw).get("labels", [])
        by_item = {a.get("n"): a.get("label") for a in answers if isinstance(a, dict)}
        labels = [by_item.get(j + 1) for j in range(len(chunk))]
        if len(chunk) > 1:
            return [lab or self._labels([r], mod, schema)[0] for r, lab in zip(chunk, labels, strict=True)]
        return [labels[0] or (self.label_space or self.spec.label_names)[0]]

    def probs_rows(self, rows: list[dict], mod: Any) -> np.ndarray:
        from ..llm import pmap

        names = self.spec.label_names
        allowed = self.label_space or names
        answer = {"n": {"type": "integer"}, "label": {"type": "string", "enum": allowed}}
        item = {"type": "object", "properties": answer, "required": ["n", "label"]}
        label_list = {"type": "array", "items": item}
        schema = {"type": "object", "properties": {"labels": label_list}}
        bs = self.batch[self.spec.modality]
        chunks = [rows[i : i + bs] for i in range(0, len(rows), bs)]
        parts = pmap(lambda chunk: self._labels(chunk, mod, schema), chunks, _workers())
        out = np.zeros((len(rows), len(names)), np.float32)
        for i, label in enumerate(label for part in parts for label in part):
            out[i, names.index(label) if label in names else 0] = 1.0
        return out

    def _tags(self, chunk: list[dict]) -> list[list[str]]:
        from ..data.prompts import task_context

        words = [str(r["input"]).split() for r in chunk]
        body = "\n".join(f"Item {j + 1} words: {json.dumps(w, ensure_ascii=False)}" for j, w in enumerate(words))
        ask = f"Return {len(chunk)} items; each item has exactly one label per word, in order."
        tag = {"type": "array", "items": {"type": "string", "enum": self.spec.label_names}}
        item = {"type": "object", "properties": {"tags": tag}}
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": item}}}
        got = self.llm.json(SYSTEM, f"{task_context(self.spec)}\n\n{body}\n\n{ask}", schema).get("items", [])
        return [(got[j].get("tags", []) if j < len(got) else []) for j in range(len(chunk))]

    def frame_rows(self, rows: list[dict], mod: Any) -> list[np.ndarray]:
        from ..llm import pmap

        names = self.spec.label_names
        fallback = names.index(OUTSIDE) if OUTSIDE in names else 0
        chunks = [rows[i : i + TAG_BATCH] for i in range(0, len(rows), TAG_BATCH)]
        tagged = [t for part in pmap(self._tags, chunks, _workers()) for t in part]
        out = []
        for r, tags in zip(rows, tagged, strict=True):
            n = len(str(r["input"]).split())
            p = np.zeros((n, len(names)), np.float32)
            for i in range(n):
                t = tags[i] if i < len(tags) else None
                p[i, names.index(t) if t in names else fallback] = 1.0
            out.append(p)
        return out

    def probs_batch(self, raws: list[Any]) -> np.ndarray:
        return self.probs_rows([{"input": r} for r in raws], None)

    def latency_ms(self, x: Any) -> float:
        return float("nan")

    def rows_per_round(self) -> int:
        return self.batch[self.spec.modality] * _workers()
