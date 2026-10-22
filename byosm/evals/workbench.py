from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from ..config import load_config
from ..data.store import Dataset, append, read, write
from ..events import emit, stage
from ..project import Project
from ..spec import TaskSpec
from .testset import WRITTEN_SPLIT, existing_rows, written_path

RESULTS_PREFIX = "workbench:"
MODELS = "test-models"
OURS = "ours"
SAMPLE_SEED = 7


def _results_report(name: str) -> str:
    return f"{RESULTS_PREFIX}{name}"


def results(p: Project, name: str) -> dict[str, dict[str, dict]]:
    return p.report(_results_report(name)) or {}


def chosen_models(p: Project, spec: TaskSpec) -> list[str]:
    return p.report(MODELS) or [OURS, *spec.baselines]


def choose_models(p: Project, models: list[str]) -> None:
    p.save_report(MODELS, models)


def _add(p: Project, name: str, rows: list[dict]) -> int:
    path = written_path(p, name)
    append(path, rows)
    return len(rows)


def write_more(
    p: Project, spec: TaskSpec, name: str, per_label: int, log: Callable[[str], None], instructions: str = ""
) -> int:
    from ..data.synth_text import generate

    path = written_path(p, name)
    shown: list[dict] = []

    def show(rows: list[dict]) -> None:
        shown.extend(rows)
        append(path, rows)

    against = existing_rows(p, spec) + read(path)
    n = per_label * len(spec.labels)
    cfg = load_config(p.ws)
    kept = generate(
        spec, n, WRITTEN_SPLIT, cfg, against, log=log, labels=spec.label_names, on_rows=show, instructions=instructions
    )
    dropped = {r["id"] for r in shown} - {r["id"] for r in kept}
    if dropped:
        write(path, [r for r in read(path) if r["id"] not in dropped])
        log(f"  removed {len(dropped)} inputs the label check rejected")
    return len(kept)


def sample_benchmark(p: Project, spec: TaskSpec, name: str, n: int) -> int:
    have = {r["id"] for r in read(written_path(p, name))}
    pool = [r for r in Dataset(p.data_dir_for(spec)).load("test") if r["id"] not in have]
    return _add(p, name, random.Random(SAMPLE_SEED + len(have)).sample(pool, min(n, len(pool))))


def _predictor(p: Project, spec: TaskSpec, model: str) -> Any:
    from ..predictors import get_baseline
    from .board import experiment_predictor

    return experiment_predictor(p, spec, p.promoted or "") if model == OURS else get_baseline(model, spec)


def _keep(out: dict, model: str, rows: list[dict], probs: Any, spec: TaskSpec, exp: str | None) -> None:
    got = out.setdefault(model, {})
    for r, pr in zip(rows, probs, strict=True):
        best = int(pr.argmax())
        got[r["id"]] = {"label": spec.label_names[best], "confidence": round(float(pr[best]), 4), "exp": exp}


def _save(p: Project, name: str, model: str, answers: dict) -> dict:
    merged = results(p, name)
    merged.setdefault(model, {}).update(answers)
    p.save_report(_results_report(name), merged)
    return merged


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
            pred = _predictor(p, spec, m)
            step = pred.rows_per_round()
            for i in range(0, len(todo), step):
                chunk = todo[i : i + step]
                _keep(out, m, chunk, row_probs(pred, chunk, mod), spec, p.promoted if m == OURS else None)
                out = _save(p, name, m, out[m])
                emit("writing", split=name, done=min(i + step, len(todo)), total=len(todo), specialist=p.name)
        got = out[m]
        right = sum(got[r["id"]]["label"] == r["label"] for r in rows if r["id"] in got)
        emit("scored", predictor=m, suite=name, objective=round(right / max(1, len(got)), 4))
        log(f"  {m}: {right}/{len(got)} right on {name}")
    return out
