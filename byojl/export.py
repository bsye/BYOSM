"""Export a specialist to a self-contained, torch-free bundle:

  export/<name>/
    model.onnx        inputs per frontend -> outputs: embedding (JEPA representation), logits
    stream.onnx       (causal audio only) mel chunk + state -> logits, embedding, new state
    manifest.json     labels, calibrated decision policy, frontend config, metrics, provenance
    tokenizer.json | mel_fb.npy + window.npy
    byojl_rt.py      the portable runtime (numpy + onnxruntime): load / encode / predict / stream().step / serve
    requirements.txt, README.md

The bundle is verified against the PyTorch model before it is written (max abs diff on real inputs).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .config import Project
from .data.store import Dataset
from .modalities import get_modality, masked_mean
from .modalities.audio import TinyCRNN
from .runtime import Specialist


class _Full(nn.Module):
    """Frontend tensors -> (embedding, logits). Pooled embedding for classify; per-frame for frame tasks."""

    def __init__(self, sp: Specialist, kind: str):
        super().__init__()
        self.bb, self.head, self.kind, self.frame = sp.model.backbone, next(iter(sp.model.heads.values())), kind, sp.spec.task_type == "frame"

    def _out(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frame:
            return feats, self.head.net(feats)
        return masked_mean(feats, mask), self.head(feats, mask)

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "text":
            ids, am = xs[0], xs[1]
            extra = {"token_type_ids": xs[2]} if len(xs) > 2 else {}
            feats, mask = self.bb(input_ids=ids, attention_mask=am, **extra)
        elif self.kind == "logmel":
            bb: TinyCRNN = self.bb
            x = bb.convs(bb.in_norm(xs[0]))
            feats, _ = bb.gru(x.transpose(1, 2))
            mask = torch.ones(feats.shape[:2], dtype=torch.bool)
        elif self.kind == "waveform":
            w = xs[0]
            feats, mask = self.bb(waveform=w, lengths=torch.full((w.shape[0],), w.shape[1], dtype=torch.long))
        else:
            feats, mask = self.bb(pixel_values=xs[0])
        return self._out(feats, mask)


class _Stream(nn.Module):
    """One streaming step of TinyCRNN on k mel frames with explicit state."""

    def __init__(self, sp: Specialist):
        super().__init__()
        self.bb, self.head = sp.model.backbone, next(iter(sp.model.heads.values()))

    def forward(self, mel: torch.Tensor, conv0: torch.Tensor, conv1: torch.Tensor, conv2: torch.Tensor, h: torch.Tensor):  # noqa: ANN201
        x = self.bb.in_norm(mel)
        new = []
        for c, buf in zip(self.bb.convs, (conv0, conv1, conv2)):
            xi = torch.cat([buf, x], 2)
            new.append(xi[:, :, xi.shape[2] - c.pad :])
            x = nn.functional.gelu(c.norm(c.conv(xi)))
        feats, h2 = self.bb.gru(x.transpose(1, 2), h)
        return self.head.net(feats), feats, new[0], new[1], new[2], h2


def _calibrated_decision(p: Project, exp: str, spec_decision: dict) -> tuple[dict, dict]:
    board = json.loads((p.root / "board.json").read_text()) if (p.root / "board.json").exists() else {}
    from .spec import TaskSpec

    base = TaskSpec.load(p.spec_path)
    prim = base.primary_suite()
    r = board.get(exp, {}).get(prim)
    d = dict(spec_decision)
    if r and d.get("threshold") is None:
        d["threshold"] = r["metrics"].get("threshold")
    metrics = {s: {"objective": round(v["objective"], 4), **{k: round(v["metrics"][k], 4) for k in ("precision", "recall", "roc_auc") if v["metrics"].get(k) is not None}} for s, v in board.get(exp, {}).items()}
    return d, metrics


def export(name: str, exp: str | None = None, out: Path | None = None, log: Callable[[str], None] = print) -> Path:
    p = Project(name)
    exp = exp or p.promoted
    if not exp:
        raise RuntimeError("nothing promoted yet; pass --exp")
    sp = Specialist(p.exp_dir(exp), "cpu")
    spec = sp.spec
    out = out or (p.root / "export")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from .spec import TaskSpec

    base = TaskSpec.load(p.spec_path)
    decision, metrics = _calibrated_decision(p, exp, base.decision.model_dump(mode="json"))

    mod = sp.mod
    bb = sp.model.backbone
    frontend: dict[str, Any]
    if spec.modality == "text":
        kind = "text"
        mod.tokenizer.save_pretrained(out / "_tok")
        shutil.copy(out / "_tok" / "tokenizer.json", out / "tokenizer.json")
        shutil.rmtree(out / "_tok")
        enc = mod.tokenizer(["example input for export tracing"], return_tensors="pt")
        names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
        example = tuple(enc[k] for k in names)
        frontend = {"type": "tokenizer", "max_len": spec.train.max_len, "inputs": names}
        dyn = {k: {0: "batch", 1: "seq"} for k in names}
    elif isinstance(bb, TinyCRNN):
        kind = "logmel"
        mel = bb.mel
        np.save(out / "mel_fb.npy", mel.mel_scale.fb.numpy())
        np.save(out / "window.npy", mel.spectrogram.window.numpy())
        frontend = {"type": "logmel", "sample_rate": spec.data.audio.sample_rate, "n_fft": mel.n_fft, "win_length": mel.win_length, "hop": bb.hop, "left_pad": 400 - bb.hop}
        names = ["mel"]
        example = (torch.log(mel(torch.randn(1, 16000) * 0.1) + 1e-6),)
        dyn = {"mel": {0: "batch", 2: "frames"}}
    elif spec.modality == "audio":
        kind = "waveform"
        frontend = {"type": "waveform", "sample_rate": spec.data.audio.sample_rate}
        names, example, dyn = ["waveform"], (torch.randn(1, 16000) * 0.1,), {"waveform": {0: "batch", 1: "samples"}}
    else:
        kind = "image"
        from .modalities.image import MEAN, STD

        frontend = {"type": "image", "size": mod.size, "mean": list(MEAN), "std": list(STD)}
        names, example, dyn = ["pixel_values"], (torch.randn(1, 3, mod.size, mod.size),), {"pixel_values": {0: "batch"}}

    full = _Full(sp, kind).eval()
    out_dyn = {"embedding": {0: "batch", 1: "frames"} if spec.task_type == "frame" else {0: "batch"}, "logits": {0: "batch", 1: "frames"} if spec.task_type == "frame" else {0: "batch"}}
    torch.onnx.export(full, example, str(out / "model.onnx"), input_names=names, output_names=["embedding", "logits"], dynamic_axes={**dyn, **out_dyn}, opset_version=17, dynamo=False)
    log(f"  model.onnx ({(out / 'model.onnx').stat().st_size / 1e6:.2f} MB)")

    if kind == "logmel":
        st = _Stream(sp).eval()
        c = [torch.zeros(1, cv.conv.in_channels, cv.pad) for cv in bb.convs]
        h = torch.zeros(1, 1, bb.gru.hidden_size)
        ex = (torch.randn(1, mel.n_mels, 3), *c, h)
        torch.onnx.export(st, ex, str(out / "stream.onnx"), input_names=["mel", "conv0", "conv1", "conv2", "h"],
                          output_names=["logits", "embedding", "conv0_out", "conv1_out", "conv2_out", "h_out"],
                          dynamic_axes={"mel": {2: "frames"}, "logits": {1: "frames"}, "embedding": {1: "frames"}}, opset_version=17, dynamo=False)
        log("  stream.onnx (stateful streaming step)")

    neg = next(i for i in range(len(spec.labels)) if i != spec.pos_index) if len(spec.labels) > 1 else 0
    manifest = {
        "format": "byojl/1", "name": spec.name, "experiment": exp, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": spec.description, "modality": spec.modality, "task_type": spec.task_type, "labels": spec.label_names,
        "positive_index": spec.pos_index if (spec.positive_label or len(spec.labels) == 2) else None, "negative_index": neg,
        "frame_hop_ms": spec.frame_hop_ms, "decision": decision, "frontend": frontend, "backbone": spec.backbone,
        "params": sp.meta["params"], "embedding_dim": int(bb.dim), "metrics": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(Path(__file__).with_name("rt.py"), out / "byojl_rt.py")
    reqs = ["numpy", "onnxruntime"] + (["tokenizers"] if kind == "text" else []) + (["pillow"] if kind == "image" else []) + (["soundfile  # only for reading audio files"] if spec.modality == "audio" else [])
    (out / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (out / "README.md").write_text(_readme(manifest, kind))

    _verify(out, sp, p, kind, log)
    return out


def _verify(out: Path, sp: Specialist, p: Project, kind: str, log: Callable[[str], None]) -> None:
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("byojl_rt", out / "byojl_rt.py")
    rt = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rt)
    m = rt.load(out)
    rows = Dataset(p.data_dir_for(sp.spec)).load("test")[:5]
    if not rows:
        raise RuntimeError(f"no test inputs found in data/{sp.spec.data_key()} to verify the export against; run `byojl data {p.name}`")
    mod = get_modality(sp.spec, str(p.data_dir_for(sp.spec)))
    worst = 0.0
    for r in rows:
        x = mod.load(r)
        _, pr = m._run(x)
        with torch.no_grad():
            inputs, _ = mod.collate([x])
            logits, mask = sp.model(inputs)
            pt = torch.softmax(logits.float(), -1)[0]
            pt = (pt[mask[0]] if sp.spec.task_type == "frame" else pt).numpy()
        n = min(len(pt), len(pr)) if sp.spec.task_type == "frame" else None
        worst = max(worst, float(np.abs((pt[:n] if n else pt) - (pr[:n] if n else pr)).max()))
    if m.stream_sess is not None and rows:
        x = mod.load(rows[0])
        s = m.stream()
        sp_probs = np.concatenate([s.step(x[i : i + 480])["probs"] for i in range(0, len(x), 480)])
        _, full = m._run(x)
        k = min(len(sp_probs), len(full))
        worst = max(worst, float(np.abs(sp_probs[:k] - full[:k, m.m["positive_index"]]).max()))
    log(f"  verified ONNX vs PyTorch (and streaming vs offline) on {len(rows)} test inputs: max |diff| = {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError(f"export parity check failed: max diff {worst}")


def _readme(m: dict, kind: str) -> str:
    ex = {"text": 'm.predict("so what I wanted to ask was")', "logmel": "m.predict(wav)  # float32 mono @ 16kHz",
          "waveform": "m.predict(wav)", "image": 'm.predict("photo.jpg")'}[kind]
    stream = """
## Streaming (realtime audio)

```python
s = m.stream()
for chunk in mic_chunks():          # any size, float32 mono @ 16 kHz
    out = s.step(chunk)             # {"probs": [...per 10ms...], "speaking": bool, "events": [{"type": "start"|"end", "t_s": ...}]}
```
""" if kind == "logmel" else ""
    metrics = "\n".join(f"| {s} | {v['objective']} | {v.get('precision', '')} | {v.get('recall', '')} | {v.get('roc_auc', '')} |" for s, v in m["metrics"].items())
    return f"""# {m['name']} (byojl specialist, {m['experiment']})

{m['description']}

Labels: {', '.join(m['labels'])} · {m['task_type']} · {m['params'] / 1e6:.2f}M params · embedding dim {m['embedding_dim']}

```bash
pip install -r requirements.txt
python byojl_rt.py predict . <input>
python byojl_rt.py serve . --port 8800        # POST /v1/{m['name']}/predict  /v1/{m['name']}/encode
```

```python
from byojl_rt import load
m = load(".")
{ex}
m.encode(x)          # JEPA representation
```
{stream}
## Held-out results

| suite | objective | precision | recall | roc_auc |
|---|---|---|---|---|
{metrics}
"""
