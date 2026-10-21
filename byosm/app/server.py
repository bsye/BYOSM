from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import MAX_PARALLEL_JOBS
from . import catalog, keys
from .hub import Hub
from .jobs import Runner

STATIC_DIR = Path(__file__).parent / "static"
LOCAL_HOSTS = ["127.0.0.1", "localhost"]
BEARER = "Bearer "


class JobRequest(BaseModel):
    args: list[str]


class KeyValue(BaseModel):
    value: str


class PredictRequest(BaseModel):
    input: str


def _missing(fn: Any, *a: Any) -> Any:
    try:
        return fn(*a)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _routes(app: FastAPI, ws: Path, runner: Runner, hub: Hub, auth: Any) -> None:
    models: dict[str, Any] = {}

    @app.get("/api/specialists", dependencies=[auth])
    def specialists() -> list[dict]:
        return catalog.specialists(ws)

    @app.get("/api/specialists/{name}", dependencies=[auth])
    def specialist(name: str) -> dict:
        return _missing(catalog.detail, ws, name)

    @app.put("/api/specialists/{name}/spec", dependencies=[auth])
    def save_spec(name: str, body: dict) -> dict:
        return _missing(catalog.save_spec, ws, name, body)

    @app.delete("/api/specialists/{name}", dependencies=[auth])
    def delete(name: str, confirm: str = Query(...)) -> dict:
        if confirm != name:
            raise HTTPException(422, "pass confirm=<name> to delete a specialist and everything it built")
        _missing(catalog.delete, ws, name)
        models.pop(name, None)
        return {"deleted": name}

    @app.get("/api/specialists/{name}/rows", dependencies=[auth])
    def rows(name: str, split: str = "train", offset: int = 0, limit: int = 100) -> dict:
        return _missing(catalog.rows, ws, name, split, offset, limit)

    @app.post("/api/specialists/{name}/predict", dependencies=[auth])
    def predict(name: str, body: PredictRequest) -> dict:
        from ..rt import load

        bundle = ws / "specialists" / name / "export"
        if not (bundle / "manifest.json").exists():
            raise HTTPException(404, f"{name} is not exported: run the export job first")
        models[name] = models.get(name) or load(bundle)
        return models[name].predict(body.input)

    @app.get("/api/systems", dependencies=[auth])
    def systems() -> list[dict]:
        return catalog.systems(ws)

    @app.get("/api/systems/{name}", dependencies=[auth])
    def system(name: str) -> dict:
        return _missing(catalog.system, ws, name)

    @app.get("/api/jobs", dependencies=[auth])
    def jobs() -> list[dict]:
        return [j.public() for j in reversed(runner.jobs.values())]

    @app.post("/api/jobs", dependencies=[auth])
    def submit(body: JobRequest) -> dict:
        return _missing(runner.submit, body.args).public()

    @app.get("/api/jobs/{job_id}", dependencies=[auth])
    def job(job_id: str) -> dict:
        return _missing(runner.detail, job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[auth])
    def cancel(job_id: str) -> dict:
        return _missing(runner.cancel, job_id).public()

    @app.get("/api/keys", dependencies=[auth])
    def list_keys() -> list[dict]:
        return keys.list_keys(ws)

    @app.put("/api/keys/{name}", dependencies=[auth])
    def set_key(name: str, body: KeyValue) -> dict:
        _missing(keys.set_key, ws, name, body.value)
        return {"name": name, "set": True}

    @app.delete("/api/keys/{name}", dependencies=[auth])
    def delete_key(name: str) -> dict:
        _missing(keys.delete_key, ws, name)
        return {"name": name, "set": False}

    @app.websocket("/api/events")
    async def events(sock: WebSocket, token: str = "") -> None:
        if not secrets.compare_digest(token, app.state.token):
            await sock.close(code=4401)
            return
        await sock.accept()
        sub = hub.subscribe()
        try:
            for j in runner.jobs.values():
                await sock.send_json({"type": "job", "job": j.public()})
            while True:
                await sock.send_json(await sub[1].get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(sub)


def create_app(ws: Path, token: str, hosts: list[str] | None = None, parallel: int = MAX_PARALLEL_JOBS) -> FastAPI:
    app = FastAPI(title="byosm", summary="Local API over the BYOSM pipeline")
    app.state.token = token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOCAL_HOSTS, *(hosts or [])])
    hub = Hub()
    runner = Runner(ws, hub, parallel)
    app.state.runner = runner
    app.router.on_shutdown.append(runner.shutdown)

    def check(authorization: str = Header("")) -> None:
        given = authorization[len(BEARER) :] if authorization.startswith(BEARER) else ""
        if not secrets.compare_digest(given, token):
            raise HTTPException(401, "missing or wrong token: open the link `byosm app` printed")

    _routes(app, ws, runner, hub, Depends(check))
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
