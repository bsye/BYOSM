from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import Backbone, Modality, pad_labels, pretrained, register


class CausalConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.norm = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.norm(self.conv(nn.functional.pad(x, (self.pad, 0)))))


class TinyCRNN(Backbone):
    def __init__(self, sample_rate: int, hop_ms: float, n_mels: int = 40, width: int = 64, hidden: int = 96):
        super().__init__()
        import torchaudio

        self.hop = int(sample_rate * hop_ms / 1000)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate, n_fft=400, win_length=400, hop_length=self.hop, n_mels=n_mels, center=False
        )
        self.in_norm = nn.BatchNorm1d(n_mels)
        self.convs = nn.Sequential(
            CausalConv(n_mels, width, 5, 1), CausalConv(width, width, 5, 2), CausalConv(width, width, 5, 4)
        )
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.dim = hidden

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = nn.functional.pad(waveform, (400 - self.hop, 0))
        x = torch.log(self.mel(x) + 1e-6)
        x = self.convs(self.in_norm(x))
        feats, _ = self.gru(x.transpose(1, 2))
        n = feats.shape[1]
        mask = torch.arange(n, device=feats.device)[None] < (lengths[:, None] // self.hop).clamp(min=1)
        return feats, mask

    @torch.no_grad()
    def stream(self, chunk: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        if state is None:
            b = chunk.shape[0]
            state = {"wav": chunk.new_zeros(b, 400 - self.hop), "conv": [None] * len(self.convs), "h": None}
        x = torch.cat([state["wav"], chunk], 1)
        state["wav"] = x[:, -(400 - self.hop) :]
        x = self.in_norm(torch.log(self.mel(x) + 1e-6))
        for i, c in enumerate(self.convs):
            buf = state["conv"][i] if state["conv"][i] is not None else x.new_zeros(x.shape[0], x.shape[1], c.pad)
            xi = torch.cat([buf, x], 2)
            state["conv"][i] = xi[:, :, -c.pad :]
            x = nn.functional.gelu(c.norm(c.conv(xi)))
        feats, state["h"] = self.gru(x.transpose(1, 2), state["h"])
        return feats, state


class HFAudioBackbone(Backbone):
    def __init__(self, repo: str):
        super().__init__()
        self.model = pretrained(repo)
        self.dim = self.model.config.hidden_size

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (waveform - waveform.mean(-1, keepdim=True)) / (waveform.std(-1, keepdim=True) + 1e-7)
        am = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).long()
        feats = self.model(input_values=x, attention_mask=am).last_hidden_state
        out_len = self.model._get_feat_extract_output_lengths(lengths)
        mask = torch.arange(feats.shape[1], device=feats.device)[None] < out_len[:, None]
        return feats, mask


def load_wav(path: str | Path, sr: int) -> np.ndarray:
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if file_sr != sr:
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), file_sr, sr).numpy()
    return wav


@register
class Audio(Modality):
    name = "audio"

    @property
    def sr(self) -> int:
        return self.spec.data.audio.sample_rate

    def load(self, ex: dict) -> np.ndarray:
        if isinstance(ex["input"], (list, np.ndarray)):
            return np.asarray(ex["input"], dtype=np.float32)
        p = Path(ex["input"])
        if not p.is_absolute() and self.data_dir:
            p = Path(self.data_dir) / p
        return load_wav(p, self.sr)

    def frame_labels(self, ex: dict) -> list[int] | None:
        return self.frame_label_ids(ex)

    def build_backbone(self) -> Backbone:
        bb = self.spec.backbone
        if bb == "tiny-crnn":
            return TinyCRNN(self.sr, self.spec.frame_hop_ms)
        if bb.startswith("hf:"):
            return HFAudioBackbone(bb[3:])
        raise ValueError(f"unknown audio backbone {bb!r}")

    def collate(
        self, raws: list[np.ndarray], frame_labels: list[list[int]] | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        lengths = torch.tensor([len(r) for r in raws])
        wav = torch.zeros(len(raws), int(lengths.max()))
        for i, r in enumerate(raws):
            wav[i, : len(r)] = torch.from_numpy(np.asarray(r, dtype=np.float32))
        targets = pad_labels(frame_labels) if frame_labels is not None else None
        return {"waveform": wav, "lengths": lengths}, targets

    def views(self, raw: np.ndarray, rng: random.Random) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = len(raw)

        def aug(lo: float, hi: float) -> np.ndarray:
            m = max(int(n * rng.uniform(lo, hi)), self.sr // 10)
            s = rng.randint(0, max(0, n - m))
            x = raw[s : s + m] * (10 ** (rng.uniform(-12, 6) / 20))
            if rng.random() < 0.7:
                p = float(np.mean(x**2)) + 1e-9
                x = x + np.random.default_rng(rng.randint(0, 2**31)).normal(
                    0, np.sqrt(p / 10 ** (rng.uniform(0, 30) / 10)), m
                ).astype(np.float32)
            if rng.random() < 0.5:
                w = rng.randint(0, max(1, m // 10))
                t0 = rng.randint(0, max(0, m - w))
                x = x.copy()
                x[t0 : t0 + w] = 0
            return x.astype(np.float32)

        return [aug(0.8, 1.0), aug(0.8, 1.0)], [aug(0.2, 0.4) for _ in range(4)]

    def example_input(self) -> np.ndarray:
        return (
            np.random.default_rng(0)
            .normal(0, 0.05, int(self.sr * self.spec.data.audio.clip_seconds))
            .astype(np.float32)
        )
