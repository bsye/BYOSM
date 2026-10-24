from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

COSTS_TTL_S = 60


class TryInput(BaseModel):
    input: str


def resources_router(ws: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/gpu")
    def gpu(request: Request) -> dict[str, Any]:
        from ..train.gpu import holders

        runner = request.app.state.runner
        held = holders(ws)
        holding = {h.get("job") for h in held}
        waiting = []
        for job in runner.snapshot():
            last = (runner.store.events(job.id) or [{}])[-1] if job.status == "running" else {}
            if last.get("kind") == "gpu" and job.id not in holding:
                waiting.append({"job": job.id, "args": job.args, "waited": last.get("waited", 0)})
        return {"holders": held, "waiting": waiting}

    @router.get("/api/machine")
    def machine() -> dict[str, Any]:
        from ..machine import machine as probe

        return probe(ws)

    @router.get("/api/machine/memory")
    def memory() -> dict[str, Any]:
        from ..machine import memory as probe

        return probe()

    @router.get("/api/opportunities")
    def opportunities() -> list[dict[str, Any]]:
        from ..opportunities import trending

        return trending()

    @router.get("/api/ideas")
    def ideas() -> dict[str, Any]:
        from ..ideas import saved

        return saved(ws)

    @router.get("/api/deployments")
    def deployments() -> list[dict[str, Any]]:
        from ..deploy import deployments as running

        return running()

    @router.post("/api/deployments/{name}/predict")
    def predict(name: str, body: TryInput) -> dict[str, Any]:
        from ..deploy import predict as ask

        try:
            return ask(name, body.input)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    @router.post("/api/deployments/{name}/stop")
    def stop(name: str) -> dict[str, Any]:
        from ..deploy import stop as halt

        halt(name)
        return {"stopped": name}

    @router.get("/api/jobs/{job_id}/cost")
    def job_cost(job_id: str) -> dict[str, Any]:
        from ..costs.spend import job_cost as priced

        return priced(ws, job_id)

    @router.get("/api/costs")
    def costs() -> dict[str, Any]:
        return _costs(ws, int(time.time() // COSTS_TTL_S))

    return router


@lru_cache(maxsize=1)
def _costs(ws: Path, _bucket: int) -> dict[str, Any]:
    from ..costs import economics, settings

    return {"settings": settings(ws), "specialists": economics(ws)}
