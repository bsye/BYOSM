"""Documents: PDFs or page images. A document is a list of pages; every page has two views:

  text    the PDF text layer, or OCR for scans (RapidOCR, ONNX; cached next to the file)
  visual  the rendered page image

route=text|visual|fused picks which views are encoded. Fused pages combine both with a learned per-dimension gate
(and random view dropout while training, so a page with no text or a blank render still works). A small transformer
then mixes information across pages; classify heads pool over pages (attention pooling recommended), frame heads
label every page (e.g. "where does the next document start" when splitting a merged scan).
"""
from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import IGNORE, Backbone, Modality, masked_mean, pad_labels, register
from .image import MEAN, STD, TinyCNN

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Page:
    text: str
    image: Any  # PIL image, already resized to image_size


_ocr_engine = None
_PDFIUM = __import__("threading").Lock()  # PDFium is not thread-safe; OCR is, so only rendering is serialized


def ocr(img: Any) -> str:
    """RapidOCR (ONNX, CPU). Returns reading-order text; empty string if OCR is unavailable."""
    global _ocr_engine
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    res, _ = _ocr_engine(np.asarray(img.convert("RGB")))
    return " ".join(r[1] for r in (res or []))


def read_pages(path: Path, max_pages: int, dpi: int, ocr_mode: str) -> list[tuple[str, Any]]:
    """-> [(text, full-resolution PIL image)] for the first max_pages pages."""
    from PIL import Image

    out = []
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        with _PDFIUM:
            pdf = pdfium.PdfDocument(str(path))
            for i in range(min(len(pdf), max_pages)):
                page = pdf[i]
                out.append((page.get_textpage().get_text_range() or "", page.render(scale=dpi / 72).to_pil().convert("RGB")))
            pdf.close()
    else:
        out.append(("", Image.open(path).convert("RGB")))
    if ocr_mode == "never":
        return out
    cache = path.with_name(path.name + ".ocr.json")
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    changed = False
    for i, (txt, img) in enumerate(out):
        if ocr_mode == "always" or len(txt.strip()) < 20:
            if str(i) not in cached:
                cached[str(i)] = ocr(img)
                changed = True
            out[i] = (cached[str(i)], img)
    if changed:
        try:
            cache.write_text(json.dumps(cached))
        except OSError:
            pass
    return out


class AttnBlock(nn.Module):
    """Plain pre-norm self-attention block (exports cleanly to ONNX, unlike nn.TransformerEncoder's fast path)."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.d = heads, d
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, p, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, p, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d // self.h)
        att = att.masked_fill(~mask[:, None, None, :], -1e4).softmax(-1)
        x = x + self.o((att @ v).transpose(1, 2).reshape(b, p, self.d))
        return x + self.ff(self.n2(x))


def _hf(repo: str) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(repo)


class DocumentBackbone(Backbone):
    def __init__(self, cfg: Any):
        super().__init__()
        self.route, self.mdrop = cfg.route, cfg.modality_dropout
        d = self.dim = cfg.dim
        if self.route in ("text", "fused"):
            self.text = _hf(cfg.text_backbone[3:])
            self.tproj = nn.Linear(self.text.config.hidden_size, d)
        if self.route in ("visual", "fused"):
            if cfg.image_backbone == "tiny-cnn":
                self.image, idim = TinyCNN(), TinyCNN().dim
            else:
                self.image = _hf(cfg.image_backbone[3:])
                c = self.image.config
                idim = getattr(c, "hidden_size", None) or c.hidden_sizes[-1]
            self.iproj = nn.Linear(idim, d)
        if self.route == "fused":
            self.gate = nn.Linear(2 * d, d)
        self.pos = nn.Embedding(cfg.max_pages, d)
        self.mix = AttnBlock(d)
        self.norm = nn.LayerNorm(d)

    def _text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None) -> torch.Tensor:
        extra = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        h = self.text(input_ids=input_ids, attention_mask=attention_mask, **extra).last_hidden_state
        return self.tproj(masked_mean(h, attention_mask.bool()))

    def _image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.image, TinyCNN):
            f, m = self.image(pixel_values)
        else:
            f = self.image(pixel_values=pixel_values).last_hidden_state
            if f.dim() == 4:
                f = f.flatten(2).transpose(1, 2)
            m = torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)
        return self.iproj(masked_mean(f, m))

    def pages(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
              pixel_values: torch.Tensor | None = None, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Per-page embeddings [N_pages, D]."""
        t = self._text(input_ids, attention_mask, token_type_ids) if self.route in ("text", "fused") else None
        v = self._image(pixel_values) if self.route in ("visual", "fused") else None
        if self.route == "text":
            return t
        if self.route == "visual":
            return v
        if self.training and self.mdrop > 0:  # view dropout: drop text or image (never both) per page
            r = torch.rand(t.shape[0], 1, device=t.device)
            t = t * (r >= self.mdrop / 2)
            v = v * ((r < self.mdrop / 2) | (r >= self.mdrop))
        g = torch.sigmoid(self.gate(torch.cat([t, v], -1)))
        return g * t + (1 - g) * v

    def mix_pages(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.norm(self.mix(x, mask))

    def forward(self, page_doc: torch.Tensor, page_pos: torch.Tensor, n_docs: torch.Tensor, **views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pe = self.pages(**views)
        b, p = int(n_docs), int(page_pos.max()) + 1
        x = pe.new_zeros(b, p, pe.shape[1])
        x[page_doc, page_pos] = pe
        mask = torch.zeros(b, p, dtype=torch.bool, device=pe.device)
        mask[page_doc, page_pos] = True
        return self.mix_pages(x, mask), mask


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

    def _path(self, ex: dict) -> Path:
        p = Path(ex["input"])
        return p if p.is_absolute() or not self.data_dir else Path(self.data_dir) / p

    def load(self, ex: dict) -> list[Page]:
        pages = read_pages(self._path(ex), self.cfg.max_pages, self.cfg.dpi, self.cfg.ocr if self.cfg.route != "visual" else "never")
        s = self.image_size
        return [Page(t, img.resize((s, s))) for t, img in pages]

    def load_many(self, rows: list[dict]) -> list[list[Page]]:
        with ThreadPoolExecutor(max_workers=8) as ex:  # OCR + rendering release the GIL
            return list(ex.map(self.load, rows))

    def frame_labels(self, ex: dict) -> list[int] | None:
        if self.spec.task_type != "frame":
            return None
        idx = {n: i for i, n in enumerate(self.spec.label_names)}
        return [idx[l] if isinstance(l, str) else int(l) for l in ex["label"]][: self.cfg.max_pages]

    def build_backbone(self) -> Backbone:
        return DocumentBackbone(self.cfg)

    def _pixels(self, imgs: list[Any]) -> torch.Tensor:
        from torchvision.transforms import v2 as T

        tf = T.Compose([T.ToImage(), T.Resize((self.image_size, self.image_size), antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
        return torch.stack([im if isinstance(im, torch.Tensor) else tf(im) for im in imgs])

    def collate(self, raws: list[list[Page]], frame_labels: list[list[int]] | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        pages = [(d, i, pg) for d, doc in enumerate(raws) for i, pg in enumerate(doc[: self.cfg.max_pages])]
        inputs: dict[str, torch.Tensor] = {
            "page_doc": torch.tensor([d for d, _, _ in pages]), "page_pos": torch.tensor([i for _, i, _ in pages]), "n_docs": torch.tensor(len(raws)),
        }
        if self.cfg.route in ("text", "fused"):
            enc = self.tokenizer([pg.text or "" for _, _, pg in pages], padding=True, truncation=True, max_length=self.cfg.max_len, return_tensors="pt")
            inputs.update({k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        if self.cfg.route in ("visual", "fused"):
            inputs["pixel_values"] = self._pixels([pg.image for _, _, pg in pages])
        targets = pad_labels(frame_labels, max(len(d[: self.cfg.max_pages]) for d in raws)) if frame_labels is not None else None
        return inputs, targets

    def views(self, raw: list[Page], rng: random.Random) -> tuple[list[list[Page]], list[list[Page]]]:
        from torchvision.transforms import v2 as T

        torch.manual_seed(rng.randint(0, 2**31))
        s = self.image_size

        def page(pg: Page, scale: tuple[float, float], drop: float, keep: float) -> Page:
            img = T.Compose([T.RandomResizedCrop(s, scale=scale, antialias=True), T.RandomApply([T.ColorJitter(0.3, 0.3)], p=0.5)])(pg.image)
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

        return [Page("INVOICE  No. 1234  Date 2024-01-02  Total due $1,250.00  Payment terms 30 days", Image.new("RGB", (self.image_size,) * 2, (250, 250, 250)))]
