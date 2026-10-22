from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Project, load_config
from ..data.store import Dataset, append, read
from ..events import emit, stage
from ..spec import TaskSpec
from .testset import WRITTEN_SPLIT, existing_rows, written_path

RESULTS_SUFFIX = ".results.json"
MODELS_FILE = "test-models.json"
OURS = "ours"
SAMPLE_SEED = 7


def results_path(p: Project, name: str) -> Path:
    return p.evals_dir / f"{name}{RESULTS_SUFFIX}"


def results(p: Project, name: str) -> dict[str, dict[str, dict]]:
    f = results_path(p, name)
    return json.loads(f.read_text()) if f.exists() else {}


def chosen_models(p: Project, spec: TaskSpec) -> list[str]:
    f = p.root / MODELS_FILE
    return json.loads(f.read_text()) if f.exists() else [OURS, *spec.baselines]


def choose_models(p: Project, models: list[str]) -> None:
    (p.root / MODELS_FILE).write_text(json.dumps(models))


def _add(p: Project, name: str, rows: list[dict]) -> int:
    path = written_path(p, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    append(path, rows)
    return len(rows)


def write_more(p: Project, spec: TaskSpec, name: str, per_label: int, log: Callable[[str], None]) -> int:
    from ..data.synth_text import generate

    against = existing_rows(p, spec) + read(written_path(p, name))
    n = per_label * len(spec.labels)
    rows = generate(spec, n, WRITTEN_SPLIT, load_config(p.ws), against, log=log, labels=spec.label_names)
    return _add(p, name, rows)


def sample_benchmark(p: Project, spec: TaskSpec, name: str, n: int) -> int:
    have = {r["id"] for r in read(written_path(p, name))}
    pool = [r for r in Dataset(p.data_dir_for(spec)).load("test") if r["id"] not in have]
    return _add(p, name, random.Random(SAMPLE_SEED + len(have)).sample(pool, min(n, len(pool))))


def _predictor(p: Project, spec: TaskSpec, model: str) -> Any:
    from ..predictors import get_baseline
    from .board import experiment_predictor

    return experiment_predictor(p, spec, p.promoted or "") if model == OURS else get_baseline(model, spec)


def run(p: Project, spec: TaskSpec, name: str, models: list[str], log: Callable[[str], None]) -> dict:
    from ..modalities import get_modality
    from .scoring import row_probs

    rows = read(written_path(p, name))
    out = results(p, name)
    mod = get_modality(spec, str(p.data_dir_for(spec)))
    for m in models:
        todo = [r for r in rows if r["id"] not in out.get(m, {})]
        if not todo:
            continue
        with stage(f"Score {p.promoted if m == OURS else m}"):
            probs = row_probs(_predictor(p, spec, m), todo, mod)
        got = out.setdefault(m, {})
        for r, pr in zip(todo, probs, strict=True):
            best = int(pr.argmax())
            got[r["id"]] = {"label": spec.label_names[best], "confidence": round(float(pr[best]), 4)}
        results_path(p, name).write_text(json.dumps(out))
        right = sum(got[r["id"]]["label"] == r["label"] for r in rows if r["id"] in got)
        emit("scored", predictor=m, suite=name, objective=round(right / max(1, len(got)), 4))
        log(f"  {m}: {right}/{len(got)} right on {name}")
    return out
