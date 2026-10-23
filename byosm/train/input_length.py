from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from ..data.store import Dataset
from ..spec import TaskSpec

COVER = 0.95
STEP = 64
MAX_LEN_CAP = 1024
SAMPLE = 2000


def fitted_max_len(spec: TaskSpec, data_dir: Path, log: Callable[[str], None]) -> TaskSpec:
    if spec.modality != "text" or not spec.backbone.startswith("hf:"):
        return spec
    from transformers import AutoTokenizer

    rows = Dataset(data_dir).load("train")[:SAMPLE]
    if not rows:
        return spec
    tok = AutoTokenizer.from_pretrained(spec.backbone.removeprefix("hf:"))
    lengths = sorted(len(tok(str(r["input"]))["input_ids"]) for r in rows)
    need = lengths[min(len(lengths) - 1, int(len(lengths) * COVER))]
    if need <= spec.train.max_len:
        return spec
    ceiling = min(MAX_LEN_CAP, getattr(tok, "model_max_length", MAX_LEN_CAP) or MAX_LEN_CAP)
    fitted = min(ceiling, math.ceil(need / STEP) * STEP)
    log(f"  inputs are longer than {spec.train.max_len} tokens ({COVER:.0%} fit in {need}): reading {fitted} tokens")
    return spec.model_copy(update={"train": spec.train.model_copy(update={"max_len": fitted})})
