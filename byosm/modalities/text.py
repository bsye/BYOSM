from __future__ import annotations

import random
from functools import cached_property
from typing import Any

import torch

from ..rt import first_subtokens
from .base import IGNORE, Backbone, Modality, pretrained, register


class HFTextBackbone(Backbone):
    def __init__(self, repo: str):
        super().__init__()
        self.model = pretrained(repo)
        self.dim = self.model.config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extra = {"token_type_ids": kw["token_type_ids"]} if "token_type_ids" in kw else {}
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **extra)
        return out.last_hidden_state, attention_mask.bool()


@register
class Text(Modality):
    name = "text"

    @property
    def repo(self) -> str:
        bb = self.spec.backbone
        if not bb.startswith("hf:"):
            raise ValueError(f"text backbone must be hf:<repo>, got {bb!r}")
        from ..encoders import resolve

        return resolve(bb[3:])

    @cached_property
    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.repo)

    def load(self, ex: dict) -> str:
        return ex["input"]

    def frame_labels(self, ex: dict) -> list[int] | None:
        return self.frame_label_ids(ex, numeric_ok=False)

    def build_backbone(self) -> Backbone:
        return HFTextBackbone(self.repo)

    def collate(
        self, raws: list[str], frame_labels: list[list[int]] | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        max_len = self.spec.train.max_len
        if frame_labels is None and self.spec.task_type != "frame":
            enc = self.tokenizer(raws, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            return dict(enc), None
        enc = self.encode_words(raws)
        targets = None
        if frame_labels is not None:
            targets = torch.full(enc["input_ids"].shape, IGNORE, dtype=torch.long)
            for i, labs in enumerate(frame_labels):
                for w, j in enumerate(first_subtokens(enc.word_ids(i))[: len(labs)]):
                    targets[i, j] = labs[w]
        return dict(enc), targets

    def encode_words(self, raws: list[str]) -> Any:
        words = [r.split() for r in raws]
        return self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.spec.train.max_len,
            return_tensors="pt",
        )

    def views(self, raw: str, rng: random.Random) -> tuple[list[str], list[str]]:
        words = raw.split() or [raw]
        mask = self.tokenizer.mask_token or "_"

        def masked(p: float) -> str:
            return " ".join(mask if rng.random() < p else w for w in words)

        def crop(lo: float, hi: float) -> str:
            n = max(1, int(len(words) * rng.uniform(lo, hi)))
            s = rng.randint(0, max(0, len(words) - n))
            return " ".join(words[s : s + n])

        return [masked(0.10), masked(0.20)], [crop(0.3, 0.6) for _ in range(4)]

    def example_input(self) -> str:
        return "this is a fairly typical sentence of moderate length for a latency benchmark of the model"
