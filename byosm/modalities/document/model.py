from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..base import Backbone, masked_mean
from ..image import TinyCNN


class AttnBlock(nn.Module):
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
            self.iproj = nn.Linear(self._build_image(cfg.image_backbone), d)
        self.image_pool = getattr(cfg, "image_pool", "mean")
        if self.route == "fused":
            self.gate = nn.Linear(2 * d, d)
        self.pos = nn.Embedding(cfg.max_pages, d)
        self.mix = AttnBlock(d)
        self.norm = nn.LayerNorm(d)

    def _build_image(self, name: str) -> int:
        if name == "tiny-cnn":
            self.image = TinyCNN()
            return TinyCNN().dim
        self.image = _hf(name[3:])
        c = self.image.config
        return getattr(c, "hidden_size", None) or c.hidden_sizes[-1]

    def _text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None
    ) -> torch.Tensor:
        extra = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        h = self.text(input_ids=input_ids, attention_mask=attention_mask, **extra).last_hidden_state
        return self.tproj(masked_mean(h, attention_mask.bool()))

    def _image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.image, TinyCNN):
            f, m = self.image(pixel_values)
        else:
            o = self.image(pixel_values=pixel_values)
            if self.image_pool == "pooler" and getattr(o, "pooler_output", None) is not None:
                return self.iproj(o.pooler_output.flatten(1))
            f = o.last_hidden_state
            if f.dim() == 4:
                f = f.flatten(2).transpose(1, 2)
            m = torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)
        return self.iproj(masked_mean(f, m))

    def pages(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self._text(input_ids, attention_mask, token_type_ids) if self.route in ("text", "fused") else None
        v = self._image(pixel_values) if self.route in ("visual", "fused") else None
        if self.route == "text":
            return t
        if self.route == "visual":
            return v
        if self.training and self.mdrop > 0:
            r = torch.rand(t.shape[0], 1, device=t.device)
            t = t * (r >= self.mdrop / 2)
            v = v * ((r < self.mdrop / 2) | (r >= self.mdrop))
        g = torch.sigmoid(self.gate(torch.cat([t, v], -1)))
        return g * t + (1 - g) * v

    def mix_pages(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.norm(self.mix(x, mask))

    def forward(
        self, page_doc: torch.Tensor, page_pos: torch.Tensor, n_docs: torch.Tensor, **views: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pe = self.pages(**views)
        b, p = int(n_docs), int(page_pos.max()) + 1
        x = pe.new_zeros(b, p, pe.shape[1])
        x[page_doc, page_pos] = pe
        mask = torch.zeros(b, p, dtype=torch.bool, device=pe.device)
        mask[page_doc, page_pos] = True
        return self.mix_pages(x, mask), mask
