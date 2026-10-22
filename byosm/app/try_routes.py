from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .errors import missing

EXAMPLES_PER_LABEL = 3
HUB_DIR = "hub"
HUB_SEPARATOR = "__"
PRODUCT = "byosm"
COMPARE_WORKERS = 4


class CompareRequest(BaseModel):
    input: str


def _examples(ws: Path, name: str) -> dict[str, Any]:
    from ..config import Project
    from ..data.store import Dataset
    from ..spec import TaskSpec

    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    spec = TaskSpec.load(p.spec_path)
    ds = Dataset(p.data_dir_for(spec))
    rows = (ds.load("test") or ds.load("dev")) if spec.modality == "text" and spec.task_type == "classify" else []
    by: dict[str, list[str]] = {}
    for r in rows:
        group = by.setdefault(str(r["label"]), [])
        if len(group) < EXAMPLES_PER_LABEL:
            group.append(str(r["input"]))
    hub = p.root / HUB_DIR
    repos = (
        sorted(
            (d.name.replace(HUB_SEPARATOR, "/", 1) for d in hub.iterdir() if d.is_dir()), key=lambda r: PRODUCT not in r
        )
        if hub.exists()
        else []
    )
    return {"examples": by, "rivals": spec.baselines, "repos": repos}


def _written(ws: Path, name: str) -> list[dict[str, Any]]:
    from ..config import Project
    from ..evals.testset import written_logs
    from ..spec import TaskSpec

    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    frozen = {e.name for e in TaskSpec.load(p.spec_path).evals}
    return [
        {"suite": s, "frozen": s in frozen, "rows": [{"input": r["input"], "label": r["label"]} for r in rows]}
        for s, rows in written_logs(p).items()
    ]


def _timed(fn: Callable[[], Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}", "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return {**out, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def try_router(ws: Path) -> APIRouter:
    from ..config import Project
    from ..spec import TaskSpec

    router = APIRouter()
    predictors: dict[tuple[str, str], Any] = {}

    def rival(name: str, rid: str, spec: TaskSpec, text: str) -> dict[str, Any]:
        from ..evals.scoring import row_probs
        from ..modalities import get_modality
        from ..predictors import get_baseline

        key = (name, rid)
        predictors[key] = predictors.get(key) or get_baseline(rid, spec)
        mod = get_modality(spec, str(Project(name, ws).data_dir_for(spec)))
        probs = row_probs(predictors[key], [{"input": text}], mod)[0]
        best = int(probs.argmax())
        return {"label": spec.label_names[best], "confidence": round(float(probs[best]), 4)}

    @router.get("/api/specialists/{name}/examples")
    def examples(name: str) -> dict:
        return missing(_examples, ws, name)

    @router.get("/api/specialists/{name}/written")
    def written(name: str) -> list:
        return missing(_written, ws, name)

    @router.post("/api/specialists/{name}/compare")
    def compare(name: str, body: CompareRequest) -> dict:
        p = Project(name, ws)
        if not p.exists():
            raise HTTPException(404, f"no specialist {name!r}")
        spec = TaskSpec.load(p.spec_path)
        if spec.modality != "text" or spec.task_type != "classify":
            raise HTTPException(422, "comparing on one input works for text classification for now")
        jobs = {rid: (lambda rid=rid: rival(name, rid, spec, body.input)) for rid in spec.baselines}
        with ThreadPoolExecutor(COMPARE_WORKERS) as pool:
            futures = {rid: pool.submit(_timed, fn) for rid, fn in jobs.items()}
            return {"rivals": {rid: f.result() for rid, f in futures.items()}}

    return router
