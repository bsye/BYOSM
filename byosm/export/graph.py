from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from ..modalities.audio import TinyCRNN
from ..runtime import Specialist


class _Full(nn.Module):
    def __init__(self, sp: Specialist, kind: str):
        super().__init__()
        self.bb, self.head, self.kind, self.frame = (
            sp.model.backbone,
            next(iter(sp.model.heads.values())),
            kind,
            sp.spec.task_type == "frame",
        )

    def _out(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frame:
            return feats, self.head.net(feats)
        return self.head.pooled(feats, mask), self.head(feats, mask)

    def _document(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        views = dict(zip(self.doc_inputs, xs, strict=False))
        pe = self.bb.pages(**views)
        mask = torch.ones(1, pe.shape[0], dtype=torch.bool)
        return self._out(self.bb.mix_pages(pe[None], mask), mask)

    def _features(self, xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "text":
            extra = {"token_type_ids": xs[2]} if len(xs) > 2 else {}
            return self.bb(input_ids=xs[0], attention_mask=xs[1], **extra)
        if self.kind == "logmel":
            bb: TinyCRNN = self.bb
            x = bb.convs(bb.in_norm(xs[0]))
            feats, _ = bb.gru(x.transpose(1, 2))
            return feats, torch.ones(feats.shape[:2], dtype=torch.bool)
        if self.kind == "waveform":
            w = xs[0]
            return self.bb(waveform=w, lengths=torch.full((w.shape[0],), w.shape[1], dtype=torch.long))
        return self.bb(pixel_values=xs[0])

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "document":
            return self._document(xs)
        feats, mask = self._features(xs)
        return self._out(feats, mask)


class _Stream(nn.Module):
    def __init__(self, sp: Specialist):
        super().__init__()
        self.bb, self.head = sp.model.backbone, next(iter(sp.model.heads.values()))

    def forward(
        self, mel: torch.Tensor, conv0: torch.Tensor, conv1: torch.Tensor, conv2: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = self.bb.in_norm(mel)
        new = []
        for c, buf in zip(self.bb.convs, (conv0, conv1, conv2), strict=False):
            xi = torch.cat([buf, x], 2)
            new.append(xi[:, :, xi.shape[2] - c.pad :])
            x = nn.functional.gelu(c.norm(c.conv(xi)))
        feats, h2 = self.bb.gru(x.transpose(1, 2), h)
        return self.head.net(feats), feats, new[0], new[1], new[2], h2


def export_model(
    sp: Specialist, kind: str, names: list[str], example: tuple, dyn: dict, out: Path, log: Callable[[str], None]
) -> list[str]:
    full = _Full(sp, kind).eval()
    full.doc_inputs = names if kind == "document" else []
    axes = {0: "batch", 1: "frames"} if sp.spec.task_type == "frame" else {0: "batch"}
    out_dyn = {"embedding": axes, "logits": dict(axes)}
    torch.onnx.export(
        full,
        example,
        str(out / "model.onnx"),
        input_names=names,
        output_names=["embedding", "logits"],
        dynamic_axes={**dyn, **out_dyn},
        opset_version=17,
        dynamo=False,
    )
    log(f"  model.onnx ({(out / 'model.onnx').stat().st_size / 1e6:.2f} MB)")
    import onnx

    return [i.name for i in onnx.load(str(out / "model.onnx"), load_external_data=False).graph.input]


def export_stream(sp: Specialist, out: Path, log: Callable[[str], None]) -> None:
    bb = sp.model.backbone
    st = _Stream(sp).eval()
    c = [torch.zeros(1, cv.conv.in_channels, cv.pad) for cv in bb.convs]
    h = torch.zeros(1, 1, bb.gru.hidden_size)
    ex = (torch.randn(1, bb.mel.n_mels, 3), *c, h)
    torch.onnx.export(
        st,
        ex,
        str(out / "stream.onnx"),
        input_names=["mel", "conv0", "conv1", "conv2", "h"],
        output_names=["logits", "embedding", "conv0_out", "conv1_out", "conv2_out", "h_out"],
        dynamic_axes={"mel": {2: "frames"}, "logits": {1: "frames"}, "embedding": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
    log("  stream.onnx (stateful streaming step)")
