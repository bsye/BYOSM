from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np

from ..spec import TaskSpec

NOISE_FOLDS = 5
SUSPECT_SHARE = 0.1
MAX_SUSPECTS = 2000
MIN_FOLDS = 2


def oof_losses(rows: list[dict]) -> np.ndarray | None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels = [r["label"] for r in rows]
    folds = min(NOISE_FOLDS, min(Counter(labels).values()))
    if folds < MIN_FOLDS or len(set(labels)) < MIN_FOLDS:
        return None
    x = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(
        [r["input"] for r in rows]
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    probs = cross_val_predict(model, x, labels, cv=cv, method="predict_proba")
    order = sorted(set(labels))
    own = probs[np.arange(len(rows)), [order.index(y) for y in labels]]
    return -np.log(np.clip(own, 1e-9, 1.0))


def suspects(rows: list[dict]) -> list[dict]:
    text = [r for r in rows if isinstance(r["input"], str) and isinstance(r["label"], str)]
    losses = oof_losses(text) if text else None
    if losses is None:
        return []
    n = min(MAX_SUSPECTS, round(len(text) * SUSPECT_SHARE))
    return [text[i] for i in np.argsort(-losses)[:n]]


def cleaned(
    spec: TaskSpec, rows: list[dict], cfg: dict | None, log: Callable[[str], None], judge: Callable | None = None
) -> list[dict]:
    if not spec.data.clean_labels or spec.task_type != "classify" or spec.modality != "text":
        return rows
    from .judge import judge as llm_judge

    flagged = suspects(rows)
    kept = {id(r) for r in (judge or llm_judge)(spec, flagged, cfg, log=log)}
    dropped = {id(r) for r in flagged} - kept
    log(f"  label noise: {len(flagged)} rows with the highest out-of-fold loss checked, {len(dropped)} dropped")
    return [r for r in rows if id(r) not in dropped]
