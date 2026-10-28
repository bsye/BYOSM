from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..events import emit
from ..spec import TaskSpec
from .review import REVIEW_FILE, REVIEWED_SOURCE, review_rows, write_review
from .store import record

PANEL_FILE = "panel.json"
CACHE_DB = "teachers.sqlite"
CACHE_EVERY = 400
ACCURACY_FLOOR_MARGIN = 0.01
ACCURACY_CEILING = 0.99
PRIOR_PSEUDO_COUNT = 1.0


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _input_key(spec: TaskSpec, x: Any, data_dir: Path | None) -> str:
    if spec.modality == "text":
        return _sha(str(x).encode())
    path = Path(x) if Path(x).is_absolute() or data_dir is None else data_dir / x
    return _sha(path.read_bytes())


def _teacher_key(spec: TaskSpec, teacher: str) -> str:
    from ..evals.identity import baseline_identity

    labels = json.dumps([[lab.name, lab.definition] for lab in spec.labels])
    return _sha(f"{baseline_identity(teacher) or teacher}|{spec.description}|{labels}".encode())[:16]


def _db() -> sqlite3.Connection:
    from ..config import cache_db

    return cache_db(CACHE_DB, "probs (teacher text, item text, probs text, primary key (teacher, item))")


def teacher_probs(
    spec: TaskSpec, teacher: str, rows: list[dict], data_dir: Path | None, log: Callable[[str], None]
) -> np.ndarray:
    from ..evals.scoring import row_probs
    from ..modalities import get_modality
    from ..predictors import get_baseline

    db, tkey = _db(), _teacher_key(spec, teacher)
    keys = [_input_key(spec, r["input"], data_dir) for r in rows]
    got = dict(db.execute("select item, probs from probs where teacher = ?", (tkey,)))
    todo = [i for i, k in enumerate(keys) if k not in got]
    if todo:
        log(f"  teacher {teacher}: labelling {len(todo)} inputs ({len(rows) - len(todo)} cached)")
        pred, mod = get_baseline(teacher, spec), get_modality(spec, str(data_dir or ""))
        for start in range(0, len(todo), CACHE_EVERY):
            part = todo[start : start + CACHE_EVERY]
            probs = row_probs(pred, [rows[i] for i in part], mod)
            fresh = {keys[i]: json.dumps(np.round(p, 5).tolist()) for i, p in zip(part, probs, strict=True)}
            db.executemany("insert or replace into probs values (?, ?, ?)", [(tkey, k, v) for k, v in fresh.items()])
            db.commit()
            got.update(fresh)
            emit("teacher", teacher=teacher, done=start + len(part), total=len(todo), specialist=spec.name)
    return np.array([json.loads(got[k]) for k in keys], np.float32)


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean()) if len(y) else 0.0


def combine(probs: list[np.ndarray], accs: list[float], prior: np.ndarray) -> np.ndarray:
    k = len(prior)
    logp = np.tile(np.log(prior), (len(probs[0]), 1))
    for p, a in zip(probs, accs, strict=True):
        a = min(max(a, 1 / k + ACCURACY_FLOOR_MARGIN), ACCURACY_CEILING)
        total = p.sum(1, keepdims=True)
        p = np.where(total > 0, p / np.maximum(total, 1e-12), 1 / k)
        logp += np.log(a * p + (1 - a) / k)
    post = np.exp(logp - logp.max(1, keepdims=True))
    return post / post.sum(1, keepdims=True)


def load_panel(data_dir: Path) -> dict | None:
    path = data_dir / PANEL_FILE
    return json.loads(path.read_text()) if path.exists() else None


def label_pool(
    spec: TaskSpec,
    splits: dict[str, list[dict]],
    unlabeled: list[dict],
    data_dir: Path,
    log: Callable[[str], None],
    reviewed: dict[str, str] | None = None,
) -> list[dict]:
    if spec.task_type != "classify":
        raise ValueError("teachers label classification tasks only")
    gold = [r for r in splits["train"] + splits["dev"] if r["label"] in spec.label_index]
    seen = {str(r["input"]) for part in splits.values() for r in part}
    pool = list({str(r["input"]): r for r in unlabeled if str(r["input"]) not in seen}.values())
    random.Random(spec.name).shuffle(pool)
    human = {k: v for k, v in (reviewed or {}).items() if v in spec.label_index}
    pool = sorted(pool, key=lambda r: str(r["input"]) not in human)[: spec.data.teacher_rows]
    y = np.array([spec.label_index[r["label"]] for r in gold])
    teachers = spec.data.teachers
    on_gold = {t: teacher_probs(spec, t, gold, data_dir, log) for t in teachers}
    on_pool = {t: teacher_probs(spec, t, pool, data_dir, log) for t in teachers}
    accs = [accuracy(on_gold[t], y) for t in teachers]
    prior = np.bincount(y, minlength=len(spec.labels)) + PRIOR_PSEUDO_COUNT
    prior = prior / prior.sum()
    post = combine([on_pool[t] for t in teachers], accs, prior)
    votes = np.stack([on_pool[t].argmax(1) for t in teachers])
    summary = {
        "teachers": dict(zip(teachers, accs, strict=True)),
        "panel_on_labelled": accuracy(combine([on_gold[t] for t in teachers], accs, prior), y),
        "n_labelled": len(gold),
        "n_pool": len(pool),
        "n_reviewed": sum(str(r["input"]) in human for r in pool),
        "all_agree": float((votes == votes[0]).all(0).mean()) if len(pool) else 0.0,
        "prior": prior.round(6).tolist(),
    }
    (data_dir / PANEL_FILE).write_text(json.dumps(summary, indent=2))
    write_review(data_dir, review_rows(spec, pool, post, on_pool))
    log(
        f"  panel: {len(pool)} inputs labelled by {', '.join(teachers)}; all agree on {summary['all_agree']:.0%}; "
        f"accuracy on {len(gold)} labelled rows: "
        + ", ".join(f"{t} {a:.3f}" for t, a in summary["teachers"].items())
        + f", panel {summary['panel_on_labelled']:.3f}; least sure in {REVIEW_FILE}"
    )
    source = "teachers:" + "+".join(teachers)
    names = spec.label_names
    return [
        record(r["input"], human[str(r["input"])], REVIEWED_SOURCE)
        if str(r["input"]) in human
        else record(r["input"], names[int(p.argmax())], source, soft=p.round(5).tolist(), sure=round(float(p.max()), 4))
        for r, p in zip(pool, post, strict=True)
    ]
