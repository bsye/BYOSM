"""Image: `tiny-cnn` (from scratch) or `hf:<repo>` vision encoders (DINOv2, ViT...). Classification only for now."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import Backbone, Modality, register

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TinyCNN(Backbone):
    def __init__(self, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Module:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False), nn.BatchNorm2d(cout), nn.GELU())

        self.net = nn.Sequential(block(3, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 8))
        self.dim = width * 8

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.net(pixel_values).flatten(2).transpose(1, 2)  # [B, HW, D]
        return f, torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)


class HFImageBackbone(Backbone):
    def __init__(self, repo: str):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo)
        cfg = self.model.config
        self.dim = getattr(cfg, "hidden_size", None) or cfg.hidden_sizes[-1]

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.model(pixel_values=pixel_values).last_hidden_state
        if f.dim() == 4:  # conv nets: [B, C, H, W] -> tokens
            f = f.flatten(2).transpose(1, 2)
        return f, torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)


@register
class Image(Modality):
    name = "image"

    @property
    def size(self) -> int:
        return 224 if self.spec.backbone.startswith("hf:") else 128

    def load(self, ex: dict) -> Any:
        from PIL import Image as PILImage

        p = Path(ex["input"])
        if not p.is_absolute() and self.data_dir:
            p = Path(self.data_dir) / p
        return PILImage.open(p).convert("RGB")

    def build_backbone(self) -> Backbone:
        bb = self.spec.backbone
        if bb == "tiny-cnn":
            return TinyCNN()
        if bb.startswith("hf:"):
            return HFImageBackbone(bb[3:])
        raise ValueError(f"unknown image backbone {bb!r}")

    def _tensor(self, img: Any, size: int) -> torch.Tensor:
        from torchvision.transforms import v2 as T

        return T.Compose([T.ToImage(), T.Resize((size, size), antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])(img)

    def collate(self, raws: list[Any], frame_labels: Any = None) -> tuple[dict[str, torch.Tensor], None]:
        return {"pixel_values": torch.stack([r if isinstance(r, torch.Tensor) else self._tensor(r, self.size) for r in raws])}, None

    def views(self, raw: Any, rng: random.Random) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        from torchvision.transforms import v2 as T

        torch.manual_seed(rng.randint(0, 2**31))
        g, l = self.size, (self.size // 2) // 14 * 14 if self.spec.backbone.startswith("hf:") else self.size // 2

        def tf(size: int, scale: tuple[float, float]) -> Any:
            return T.Compose([T.ToImage(), T.RandomResizedCrop(size, scale=scale, antialias=True), T.RandomHorizontalFlip(),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), T.RandomGrayscale(0.2),
                              T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

        return [tf(g, (0.3, 1.0))(raw) for _ in range(2)], [tf(l, (0.05, 0.3))(raw) for _ in range(4)]

    def example_input(self) -> Any:
        from PIL import Image as PILImage

        return PILImage.new("RGB", (self.size, self.size), (120, 130, 140))
