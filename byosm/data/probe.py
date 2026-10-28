from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import TaskSpec
from ..spec.backbones import MINILM

PROBE_FOLDS = 5
PROBE_MAX_ITER = 1000
NOISE_CONFIDENCE = 0.8
MAX_JUDGED = 50
MAX_CONFUSIONS = 10
EMBED_BATCH = 32
EMBED_MAX_TOKENS = 256
TEXT_CHARS = 2000

JUDGE_SYSTEM = """You check labels in a training set. Read the input and pick the label whose definition fits it best. \
Answer with that label and one short reason."""


def input_text(spec: TaskSpec, x: str) -> str | None:
    if spec.modality == "text":
        return x
    if spec.modality != "document":
        return None
    from ..modalities.document.pages import read_pages

    dc = spec.document
    pages, _ = read_pages(Path(x), dc.max_pages, dc.dpi, dc.ocr)
    return " ".join(t for t, _ in pages)[:TEXT_CHARS]


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    from ..modalities.base import masked_mean
    from ..modalities.text import HFTextBackbone

    tok, bb = AutoTokenizer.from_pretrained(MINILM[3:]), HFTextBackbone(MINILM[3:]).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), EMBED_BATCH):
            enc = tok(texts[i : i + EMBED_BATCH], padding=True, truncation=True, max_length=EMBED_MAX_TOKENS)
            feats, mask = bb(**{k: torch.tensor(v) for k, v in enc.items()})
            out.append(masked_mean(feats, mask).numpy())
    return np.concatenate(out)


def probe(texts: list[str], labels: list[str]) -> dict[str, Any] | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    smallest = min(Counter(labels).values(), default=0)
    if len(set(labels)) < 2 or smallest < 2:
        return None
    cv = StratifiedKFold(n_splits=min(PROBE_FOLDS, smallest), shuffle=True, random_state=0)
    proba = cross_val_predict(
        LogisticRegression(max_iter=PROBE_MAX_ITER), embed(texts), labels, cv=cv, method="predict_proba"
    )
    classes = sorted(set(labels))
    pred = [classes[i] for i in proba.argmax(1)]
    wrong = [i for i, (p, y) in enumerate(zip(pred, labels, strict=True)) if p != y]
    suspects = [i for i in wrong if proba[i].max() >= NOISE_CONFIDENCE]
    pairs = Counter((labels[i], pred[i]) for i in wrong).most_common(MAX_CONFUSIONS)
    return {
        "agreement": 1 - len(wrong) / len(labels),
        "suspects": [{"index": i, "given": labels[i], "probe": pred[i]} for i in suspects],
        "confusions": [{"given": a, "probe": b, "rows": n} for (a, b), n in pairs],
    }


def judge(spec: TaskSpec, texts: list[str], suspects: list[dict], cfg: dict | None) -> list[dict]:
    from ..llm import get_llm, parallel_calls, pmap
    from .judge import JUDGE_ROLE

    llm = get_llm(JUDGE_ROLE, cfg)
    defs = "\n".join(f"- {lab.name}: {lab.definition}" for lab in spec.labels)
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": spec.label_names}, "why": {"type": "string"}},
    }

    def ask(s: dict) -> dict:
        ans = llm.json(JUDGE_SYSTEM, f"Labels:\n{defs}\n\nInput:\n{texts[s['index']][:TEXT_CHARS]}", schema)
        return {**s, "llm": ans.get("label"), "why": ans.get("why", "")}

    return pmap(ask, suspects[:MAX_JUDGED], parallel_calls(cfg))
