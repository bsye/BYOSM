"""byojl VAD specialist vs LiveKit's VAD (Silero, the exact ONNX bundled in livekit-plugins-silero).

Two test sets:
  synthetic  our held-out mixed test set (home turf for byojl: same mixer as its training data)
  ava        AVA-Speech, real film/TV audio with human utterance labels; neither model trained on it

Two views of each model:
  raw      per-10ms-frame speech probability: ROC-AUC, average precision, F1 at 0.5 and at a calibrated threshold
  livekit  LiveKit's own decision logic (ExpFilter alpha=0.35/32ms, activation 0.5, deactivation 0.35,
           min_speech 50ms, min_silence 550ms) applied identically to both models

Thresholds are calibrated per model on data neither is scored on (our dev set; 4 separate AVA clips).
Silero frames get the probability of the 32ms window containing the frame start (up to 32ms of look-ahead: favors Silero).

usage: python benchmarks/vad_vs_silero.py --ava-dir <dir with split.json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from byojl.config import Project
from byojl.data.store import Dataset
from byojl.runtime import Specialist

SR, HOP = 16000, 160


class Silero:
    name = "silero (livekit)"
    window = 512

    def __init__(self) -> None:
        from livekit.plugins.silero.onnx_model import OnnxModel, new_inference_session

        self.m = OnnxModel(onnx_session=new_inference_session(force_cpu=True), sample_rate=SR)

    def window_probs(self, wav: np.ndarray) -> np.ndarray:
        self.m.reset()
        n = len(wav) // self.window
        return np.array([self.m(wav[i * self.window : (i + 1) * self.window][None]) for i in range(n)], dtype=np.float32)

    def frame_probs(self, wav: np.ndarray) -> np.ndarray:
        wp = self.window_probs(wav)
        idx = np.minimum(np.arange(len(wav) // HOP) * HOP // self.window, len(wp) - 1)
        return wp[idx]

    def stream_chunk_ms(self, n: int = 2000) -> float:
        self.m.reset()
        x = np.random.default_rng(0).normal(0, 0.05, (1, self.window)).astype(np.float32)
        t = time.perf_counter()
        for _ in range(n):
            self.m(x)
        return (time.perf_counter() - t) * 1000 / n

    @property
    def params(self) -> int:
        import importlib.resources

        import onnx

        def walk(g) -> int:  # weights live in Constant nodes inside If-subgraphs (8k and 16k branches; both counted)
            n = sum(int(np.prod(i.dims)) for i in g.initializer)
            for node in g.node:
                for a in node.attribute:
                    if a.type == onnx.AttributeProto.TENSOR:
                        n += int(np.prod(a.t.dims))
                    elif a.type == onnx.AttributeProto.GRAPH:
                        n += walk(a.g)
            return n

        p = importlib.resources.files("livekit.plugins.silero.resources") / "silero_vad.onnx"
        return walk(onnx.load(str(p)).graph)


class Ours:
    window = 480  # stream in 30ms chunks

    def __init__(self, name: str = "vad") -> None:
        self.sp = Specialist.from_project(name, "cpu")
        self.name = f"byojl {name} ({self.sp.spec.backbone})"
        self.pos = self.sp.spec.label_names.index(self.sp.spec.positive_label)
        self.bb = self.sp.model.backbone
        self.head = next(iter(self.sp.model.heads.values()))

    @torch.no_grad()
    def frame_probs(self, wav: np.ndarray) -> np.ndarray:
        # Continuous streaming over the whole recording (same as deployment), 10s blocks to bound memory.
        state, out = None, []
        n = len(wav) // HOP * HOP
        for i in range(0, n, 160_000):
            f, state = self.bb.stream(torch.from_numpy(wav[i : min(n, i + 160_000)])[None], state)
            out.append(torch.softmax(self.head.net(f), -1)[0, :, self.pos].numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def stream_chunk_ms(self, n: int = 2000) -> float:
        x = torch.from_numpy(np.random.default_rng(0).normal(0, 0.05, (1, self.window)).astype(np.float32))
        _, state = self.bb.stream(x)
        t = time.perf_counter()
        for _ in range(n):
            f, state = self.bb.stream(x, state)
            self.head.net(f)
        return (time.perf_counter() - t) * 1000 / n

    @property
    def params(self) -> int:
        return int(self.sp.meta["params"])


def livekit_decisions(p: np.ndarray, hop_s: float, act: float = 0.5, min_speech: float = 0.05, min_silence: float = 0.55) -> np.ndarray:
    """LiveKit silero plugin state machine, offline: returns the speaking regions it would publish, per frame."""
    deact = max(act - 0.15, 0.01)
    a = 0.35 ** (hop_s / 0.032)  # same smoothing time-constant as alpha=0.35 per 32ms window
    out = np.zeros(len(p), dtype=bool)
    filt, speaking, sp_dur, sil_dur, start = None, False, 0.0, 0.0, 0
    for i, x in enumerate(p):
        filt = x if filt is None else a * filt + (1 - a) * x
        if filt >= act or (speaking and filt > deact):
            sp_dur += hop_s
            sil_dur = 0.0
            if not speaking and sp_dur >= min_speech:
                speaking, start = True, i - int(round(sp_dur / hop_s)) + 1
        else:
            sil_dur += hop_s
            sp_dur = 0.0
            if speaking and sil_dur >= min_silence:
                speaking = False
                out[start : i - int(round(sil_dur / hop_s)) + 1] = True
    if speaking:
        out[start:] = True
    return out


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = float((pred & y).sum())
    p, r = tp / max(1, pred.sum()), tp / max(1, y.sum())
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4)}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    return float(max(np.linspace(0.05, 0.95, 91), key=lambda t: prf(y, p >= t)["f1"]))


def ava_items(ava: Path, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    items = []
    for rel in json.loads((ava / "split.json").read_text())[split]:
        wav, sr = sf.read(str(ava / rel), dtype="float32")
        assert sr == SR
        j = json.loads((ava / rel.replace(".wav", ".json")).read_text())
        y = np.zeros(len(wav) // HOP, dtype=bool)
        for on, off in zip(j["onset"], j["offset"]):
            y[int(on * 100) : int(off * 100)] = True
        items.append((wav, y))
    return items


def synth_items(split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    p = Project("vad")
    spec_pos = 1  # label index of "speech"
    items = []
    for r in Dataset(p.data_dir).load(split):
        wav, _ = sf.read(str(p.data_dir / r["input"]), dtype="float32")
        items.append((wav, np.array(r["label"]) == spec_pos))
    return items


def evaluate(model, calib: list, test: list, hop_s: float = 0.01) -> dict:  # noqa: ANN001
    def run(items: list) -> tuple[np.ndarray, np.ndarray, list]:
        ps, ys, per = [], [], []
        for wav, y in items:
            p = model.frame_probs(wav)
            n = min(len(p), len(y))
            ps.append(p[:n]), ys.append(y[:n]), per.append((p[:n], y[:n]))
        return np.concatenate(ps), np.concatenate(ys), per

    pc, yc, perc = run(calib)
    thr = best_f1_threshold(yc, pc)
    # LiveKit activation threshold calibrated the same way (on calibration data, through the state machine).
    act = float(max(np.linspace(0.1, 0.9, 33), key=lambda t: prf(np.concatenate([y for _, y in perc]), np.concatenate([livekit_decisions(p, hop_s, t) for p, _ in perc]))["f1"]))
    p, y, per = run(test)
    lk_def = np.concatenate([livekit_decisions(pi, hop_s) for pi, _ in per])
    lk_cal = np.concatenate([livekit_decisions(pi, hop_s, act) for pi, _ in per])
    return {
        "frames": int(len(y)), "speech_share": round(float(y.mean()), 3),
        "raw": {"roc_auc": round(float(roc_auc_score(y, p)), 4), "avg_precision": round(float(average_precision_score(y, p)), 4),
                "at_0.5": prf(y, p >= 0.5), "calibrated": {"threshold": round(thr, 3), **prf(y, p >= thr)}},
        "livekit_logic": {"default_0.5": prf(y, lk_def), "calibrated": {"activation": round(act, 3), **prf(y, lk_cal)}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ava-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("bench/vad_vs_silero.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)  # Silero runs single-threaded in LiveKit; match it

    models = [Ours("vad"), Silero()]
    sets = {"synthetic": (synth_items("dev"), synth_items("test")), "ava": (ava_items(args.ava_dir, "calibration"), ava_items(args.ava_dir, "eval"))}
    hours = sum(len(w) for w, _ in sets["ava"][1]) / SR / 3600
    print(f"AVA eval audio: {hours:.2f} h, synthetic test: {len(sets['synthetic'][1])} clips")
    results: dict = {"models": {}}
    for m in models:
        r = {"params": m.params, "stream_chunk_ms": round(m.stream_chunk_ms(), 4), "chunk_audio_ms": m.window / SR * 1000}
        for s, (calib, test) in sets.items():
            t = time.perf_counter()
            r[s] = evaluate(m, calib, test)
            print(f"{m.name} {s}: {json.dumps(r[s])}  ({time.perf_counter() - t:.0f}s)")
        r["real_time_factor"] = round(r["stream_chunk_ms"] / r["chunk_audio_ms"], 5)
        results["models"][m.name] = r
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
