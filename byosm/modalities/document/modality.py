from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch

from ..base import Backbone, Modality, pad_labels, register
from ..image import norm_for
from .engine import ocr_many
from .model import DocumentBackbone
from .pages import read_pages

EXAMPLE_TEXT = "INVOICE  No. 1234  Date 2024-01-02  Total due $1,250.00  Payment terms 30 days"


@dataclass
class Page:
    text: str
    image: Any


@register
class Document(Modality):
    name = "document"

    @property
    def cfg(self) -> Any:
        return self.spec.document

    @cached_property
    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.cfg.text_backbone[3:])

    @property
    def image_size(self) -> int:
        return self.cfg.image_size

    @cached_property
    def norm(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return norm_for(self.cfg.image_backbone)

    def _path(self, ex: dict) -> Path:
        p = Path(ex["input"])
        return p if p.is_absolute() or not self.data_dir else Path(self.data_dir) / p

    def _read(self, ex: dict, run_ocr: bool = True) -> tuple[list[tuple[str, Any]], list[tuple[int, str]]]:
        ocr_mode = self.cfg.ocr if self.cfg.route != "visual" else "never"
        return read_pages(self._path(ex), self.cfg.max_pages, self.cfg.dpi, ocr_mode, run_ocr)

    def _pages(self, pages: list[tuple[str, Any]]) -> list[Page]:
        s = self.image_size
        return [Page(t, img.resize((s, s))) for t, img in pages]

    def load(self, ex: dict) -> list[Page]:
        return self._pages(self._read(ex)[0])

    def load_many(self, rows: list[dict]) -> list[list[Page]]:
        with ThreadPoolExecutor(max_workers=8) as ex:
            read = list(ex.map(lambda r: self._read(r, run_ocr=False), rows))
        todo = [(d, i, k) for d, (_, pend) in enumerate(read) for i, k in pend]
        if todo:
            texts = ocr_many([read[d][0][i][1] for d, i, _ in todo], [k for _, _, k in todo])
            for (d, i, _), t in zip(todo, texts, strict=False):
                read[d][0][i] = (t, read[d][0][i][1])
        return [self._pages(pages) for pages, _ in read]

    def frame_labels(self, ex: dict) -> list[int] | None:
        if self.spec.task_type != "frame":
            return None
        idx = self.spec.label_index
        return [idx[lab] if isinstance(lab, str) else int(lab) for lab in ex["label"]][: self.cfg.max_pages]

    def build_backbone(self) -> Backbone:
        return DocumentBackbone(self.cfg)

    def _pixels(self, imgs: list[Any]) -> torch.Tensor:
        from torchvision.transforms import v2 as T

        s = self.image_size
        tf = T.Compose(
            [
                T.ToImage(),
                T.Resize((s, s), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(*self.norm),
            ]
        )
        return torch.stack([im if isinstance(im, torch.Tensor) else tf(im) for im in imgs])

    def collate(
        self, raws: list[list[Page]], frame_labels: list[list[int]] | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        mp = self.cfg.max_pages
        pages = [(d, i, pg) for d, doc in enumerate(raws) for i, pg in enumerate(doc[:mp])]
        inputs: dict[str, torch.Tensor] = {
            "page_doc": torch.tensor([d for d, _, _ in pages]),
            "page_pos": torch.tensor([i for _, i, _ in pages]),
            "n_docs": torch.tensor(len(raws)),
        }
        if self.cfg.route in ("text", "fused"):
            texts = [pg.text or "" for _, _, pg in pages]
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.cfg.max_len, return_tensors="pt")
            inputs.update({k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        if self.cfg.route in ("visual", "fused"):
            inputs["pixel_values"] = self._pixels([pg.image for _, _, pg in pages])
        targets = pad_labels(frame_labels, max(len(d[:mp]) for d in raws)) if frame_labels is not None else None
        return inputs, targets

    def views(self, raw: list[Page], rng: random.Random) -> tuple[list[list[Page]], list[list[Page]]]:
        from torchvision.transforms import v2 as T

        torch.manual_seed(rng.randint(0, 2**31))
        s = self.image_size

        def page(pg: Page, scale: tuple[float, float], drop: float, keep: float) -> Page:
            jitter = T.RandomApply([T.ColorJitter(0.3, 0.3)], p=0.5)
            img = T.Compose([T.RandomResizedCrop(s, scale=scale, antialias=True), jitter])(pg.image)
            w = pg.text.split()
            if w:
                n = max(1, int(len(w) * keep))
                st = rng.randint(0, len(w) - n)
                w = [x for x in w[st : st + n] if rng.random() > drop]
            return Page(" ".join(w), img)

        g = [[page(pg, (0.6, 1.0), 0.1, 1.0) for pg in raw] for _ in range(2)]
        loc = [[page(rng.choice(raw), (0.1, 0.4), 0.3, 0.4)] for _ in range(4)]
        return g, loc

    def example_input(self) -> list[Page]:
        from PIL import Image

        return [Page(EXAMPLE_TEXT, Image.new("RGB", (self.image_size,) * 2, (250, 250, 250)))]
