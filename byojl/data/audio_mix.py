"""Procedural audio data: place clean positive events (e.g. speech) over negative backgrounds (noise, music, babble)
at controlled SNRs. Labels are exact by construction: per-frame (VAD-style) or per-clip (detection)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

from ..spec import TaskSpec
from .store import record


def _rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(x**2)) + 1e-12)


def frame_activity(x: np.ndarray, hop: int, rel_db: float = 35.0, floor_db: float = -55.0, fill_ms: int = 60, hop_ms: float = 10.0) -> np.ndarray:
    """1 where the clean event is active: frame energy within rel_db of its peak and above an absolute floor.
    Gaps shorter than fill_ms are filled so labels don't flicker inside words."""
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    act = (e > e.max() - rel_db) & (e > floor_db)
    gap, fill = 0, int(fill_ms / hop_ms)
    out = act.copy()
    for i, a in enumerate(act):
        if a:
            if 0 < gap <= fill and i - gap - 1 >= 0 and act[i - gap - 1]:
                out[i - gap : i] = True
            gap = 0
        else:
            gap += 1
    return out.astype(np.int64)


class Mixer:
    def __init__(self, spec: TaskSpec, positives: list[np.ndarray], negatives: list[np.ndarray], seed: int = 0):
        a = spec.data.audio
        self.spec, self.sr, self.n = spec, a.sample_rate, int(a.sample_rate * a.clip_seconds)
        self.hop = int(a.sample_rate * spec.frame_hop_ms / 1000)
        self.pos, self.neg = [p for p in positives if len(p) > self.sr // 5], [n for n in negatives if len(n) > self.sr // 10]
        if not self.pos:
            raise ValueError("audio mixing needs at least one positive source clip")
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        pos_label = spec.positive_label or spec.label_names[-1]
        self.pos_idx = spec.label_names.index(pos_label)
        self.neg_idx = next(i for i, n in enumerate(spec.label_names) if n != pos_label)

    def _background(self) -> np.ndarray:
        r = self.rng.random()
        if not self.neg or r < 0.12:  # near-silence with mic hiss
            return self.np_rng.normal(0, 10 ** (self.rng.uniform(-80, -50) / 20), self.n).astype(np.float32)
        out = np.zeros(self.n, dtype=np.float32)
        layers = 1 if r < 0.7 else 2
        for _ in range(layers):
            pos = 0
            while pos < self.n:
                c = self.rng.choice(self.neg)
                s = self.rng.randint(0, max(0, len(c) - (self.n - pos)))
                seg = c[s : s + self.n - pos]
                out[pos : pos + len(seg)] += seg * (10 ** (self.rng.uniform(-10, 0) / 20))
                pos += max(len(seg), 1)
        return out

    def clip(self) -> tuple[np.ndarray, np.ndarray]:
        bg = self._background()
        bg_db = _rms_db(bg)
        bg = bg * (10 ** ((self.rng.uniform(-45, -20) - bg_db) / 20))  # background level
        bg_db = _rms_db(bg)
        labels = np.zeros(self.n // self.hop, dtype=np.int64)
        mix = bg.copy()
        k = 0 if self.rng.random() < 0.25 else self.rng.randint(1, self.spec.data.audio.max_events)
        lo, hi = self.spec.data.audio.snr_db
        for _ in range(k):
            src = self.rng.choice(self.pos)
            m = min(len(src), int(self.sr * self.rng.uniform(0.3, self.spec.data.audio.clip_seconds * 0.8)))
            s = self.rng.randint(0, len(src) - m)
            ev = src[s : s + m].copy()
            ev *= 10 ** ((bg_db + self.rng.uniform(lo, hi) - _rms_db(ev)) / 20)
            t0 = self.rng.randint(0, self.n - m) // self.hop * self.hop
            mix[t0 : t0 + m] += ev
            act = frame_activity(ev, self.hop, fill_ms=self.spec.data.audio.fill_ms, hop_ms=self.spec.frame_hop_ms) if self.spec.data.audio.label_by_energy else np.ones(m // self.hop, dtype=np.int64)
            f0 = t0 // self.hop
            labels[f0 : f0 + len(act)] |= act[: len(labels) - f0]
        peak = np.abs(mix).max()
        if peak > 0.99:
            mix *= 0.99 / peak
        mix *= 10 ** (self.rng.uniform(-6, 0) / 20)
        frame = np.where(labels > 0, self.pos_idx, self.neg_idx)
        return mix.astype(np.float32), frame


def build(spec: TaskSpec, data_dir: Path, positives: dict[str, list[np.ndarray]], negatives: dict[str, list[np.ndarray]], sizes: dict[str, int], log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """positives/negatives are pre-partitioned per split so test clips never reuse train source audio."""
    import soundfile as sf

    out: dict[str, list[dict]] = {}
    for i, (split, n) in enumerate(sizes.items()):
        mixer = Mixer(spec, positives[split], negatives[split], seed=1000 + i)
        rows = []
        (data_dir / "audio" / split).mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            wav, frames = mixer.clip()
            r = record("", None, "mix", split=split)
            rel = f"audio/{split}/{r['id']}.wav"
            sf.write(str(data_dir / rel), wav, spec.data.audio.sample_rate, subtype="PCM_16")
            r["input"] = rel
            if spec.task_type == "frame":
                r["label"] = frames.tolist()
            else:
                r["label"] = spec.label_names[mixer.pos_idx] if (frames == mixer.pos_idx).any() else spec.label_names[mixer.neg_idx]
            rows.append(r)
        pos_share = np.mean([np.mean(np.array(r["label"]) == mixer.pos_idx) if spec.task_type == "frame" else r["label"] == spec.label_names[mixer.pos_idx] for r in rows])
        log(f"  mixed {n} {split} clips ({pos_share:.0%} positive {'frames' if spec.task_type == 'frame' else 'clips'})")
        out[split] = rows
    return out
