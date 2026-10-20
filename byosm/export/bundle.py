from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from ..runtime import Specialist

RT_SOURCE = Path(__file__).parent.parent / "rt.py"

EXAMPLES = {
    "text": 'm.predict("so what I wanted to ask was")',
    "logmel": "m.predict(wav)  # float32 mono @ 16kHz",
    "document": 'm.predict("invoice.pdf")  # PDF or page image',
    "waveform": "m.predict(wav)",
    "image": 'm.predict("photo.jpg")',
}

STREAM_SECTION = """
## Streaming (realtime audio)

```python
s = m.stream()
for chunk in mic_chunks():          # any size, float32 mono @ 16 kHz
    out = s.step(chunk)             # {"probs": [...per 10ms...], "speaking": bool, "events": [{"type": "start"|"end", "t_s": ...}]}
```
"""


def load_runtime(out: Path) -> ModuleType:
    spec_ = importlib.util.spec_from_file_location("byosm_rt", out / "byosm_rt.py")
    rt = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rt)
    return rt


def manifest_for(sp: Specialist, exp: str, decision: dict, frontend: dict[str, Any], metrics: dict) -> dict[str, Any]:
    spec = sp.spec
    neg = next(i for i in range(len(spec.labels)) if i != spec.pos_index) if len(spec.labels) > 1 else 0
    return {
        "format": "byosm/1",
        "name": spec.name,
        "experiment": exp,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": spec.description,
        "modality": spec.modality,
        "task_type": spec.task_type,
        "labels": spec.label_names,
        "positive_index": spec.pos_index if (spec.positive_label or len(spec.labels) == 2) else None,
        "negative_index": neg,
        "frame_hop_ms": spec.frame_hop_ms,
        "decision": decision,
        "frontend": frontend,
        "backbone": spec.backbone,
        "params": sp.meta["params"],
        "embedding_dim": int(sp.model.backbone.dim),
        "metrics": metrics,
    }


def requirements(spec: Any, kind: str) -> list[str]:
    reqs = ["numpy", "onnxruntime"]
    if kind == "text" or (kind == "document" and spec.document.route != "visual"):
        reqs.append("tokenizers")
    if kind in ("image", "document"):
        reqs.append("pillow")
    if kind == "document":
        reqs += ["pypdfium2", "rapidocr_onnxruntime  # OCR for scanned pages"]
    if spec.modality == "audio":
        reqs.append("soundfile  # only for reading audio files")
    return reqs


def write_bundle(out: Path, sp: Specialist, kind: str, manifest: dict[str, Any]) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(RT_SOURCE, out / "byosm_rt.py")
    (out / "requirements.txt").write_text("\n".join(requirements(sp.spec, kind)) + "\n")
    (out / "README.md").write_text(readme(manifest, kind))


def _metrics_rows(m: dict) -> str:
    return "\n".join(
        f"| {s} | {v['objective']} | {v.get('precision', '')} | {v.get('recall', '')} | {v.get('roc_auc', '')} |"
        for s, v in m["metrics"].items()
    )


def readme(m: dict, kind: str) -> str:
    stream = STREAM_SECTION if kind == "logmel" else ""
    return f"""# {m["name"]} (byosm specialist, {m["experiment"]})

{m["description"]}

Labels: {", ".join(m["labels"])} · {m["task_type"]} · {m["params"] / 1e6:.2f}M params · embedding dim {m["embedding_dim"]}

```bash
pip install -r requirements.txt
python byosm_rt.py predict . <input>
python byosm_rt.py serve . --port 8800        # POST /v1/{m["name"]}/predict  /v1/{m["name"]}/encode
```

```python
from byosm_rt import load
m = load(".")
{EXAMPLES[kind]}
m.encode(x)          # JEPA representation
```
{stream}
## Held-out results

| suite | objective | precision | recall | roc_auc |
|---|---|---|---|---|
{_metrics_rows(m)}
"""
