from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Project
from ..data.store import Dataset
from ..modalities import get_modality
from ..runtime import Specialist
from .bundle import load_runtime

PARITY_TOLERANCE = 1e-3
STREAM_CHUNK = 480


def _torch_probs(sp: Specialist, mod: Any, x: Any) -> np.ndarray:
    with torch.no_grad():
        inputs, _ = mod.collate([x])
        logits, mask = sp.model(inputs)
        pt = torch.softmax(logits.float(), -1)[0]
        return (pt[mask[0]] if sp.spec.task_type == "frame" else pt).numpy()


def _offline_diff(m: Any, sp: Specialist, mod: Any, rows: list[dict], kind: str) -> float:
    worst = 0.0
    for r in rows:
        x = mod.load(r)
        _, pr = m._run(str(mod._path(r)) if kind == "document" else x)
        pt = _torch_probs(sp, mod, x)
        n = min(len(pt), len(pr)) if sp.spec.task_type == "frame" else None
        worst = max(worst, float(np.abs((pt[:n] if n else pt) - (pr[:n] if n else pr)).max()))
    return worst


def _stream_diff(m: Any, mod: Any, row: dict) -> float:
    x = mod.load(row)
    s = m.stream()
    sp_probs = np.concatenate([s.step(x[i : i + STREAM_CHUNK])["probs"] for i in range(0, len(x), STREAM_CHUNK)])
    _, full = m._run(x)
    k = min(len(sp_probs), len(full))
    return float(np.abs(sp_probs[:k] - full[:k, m.m["positive_index"]]).max())


def verify(out: Path, sp: Specialist, p: Project, kind: str, log: Callable[[str], None]) -> None:
    m = load_runtime(out).load(out)
    rows = Dataset(p.data_dir_for(sp.spec)).load("test")[:5]
    if not rows:
        raise RuntimeError(
            f"no test inputs found in data/{sp.spec.data_key()} to verify the export against; run `byosm data {p.name}`"
        )
    mod = get_modality(sp.spec, str(p.data_dir_for(sp.spec)))
    worst = _offline_diff(m, sp, mod, rows, kind)
    if m.stream_sess is not None and rows:
        worst = max(worst, _stream_diff(m, mod, rows[0]))
    log(f"  verified ONNX vs PyTorch (and streaming vs offline) on {len(rows)} test inputs: max |diff| = {worst:.2e}")
    if worst > PARITY_TOLERANCE:
        raise RuntimeError(f"export parity check failed: max diff {worst}")
