from __future__ import annotations

import importlib
import secrets
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import catalog, keys, panel_routes
from .archive_routes import archive_router
from .config_routes import config_router
from .data_routes import data_router
from .errors import missing
from .forecast_routes import forecast_router
from .hub import Hub
from .jobs import Runner
from .lab_routes import lab_router
from .live_routes import live_router
from .lock import claim, release
from .models_routes import models_router
from .opendata_routes import opendata_router
from .publish_routes import publish_router
from .resources_routes import resources_router
from .systems_routes import systems_router
from .techniques_routes import techniques_router
from .try_routes import try_router
from .workbench_routes import workbench_router

STATIC_DIR = Path(__file__).parent / "static"
LOCAL_HOSTS = ["127.0.0.1", "localhost", "::1"]
BEARER = "Bearer "
UNAUTHORIZED_CLOSE = 4401
WARM_MODULE = "byosm.export.bundle"


class JobRequest(BaseModel):
    args: list[str]
    llm: dict[str, Any] | None = None


class KeyValue(BaseModel):
    value: str


class PredictRequest(BaseModel):
    input: str


def allowed_origin(origin: str, hosts: list[str]) -> bool:
    return not origin or urlsplit(origin).hostname in [*LOCAL_HOSTS, *hosts]


def _routes(app: FastAPI, ws: Path, runner: Runner, hub: Hub, auth: Any) -> None:
    models: dict[str, Any] = {}

    @app.get("/api/specialists", dependencies=[auth])
    def specialists() -> list[dict]:
        return catalog.specialists(ws)

    @app.get("/api/next-steps", dependencies=[auth])
    def next_steps() -> dict:
        return catalog.next_steps(ws)

    @app.get("/api/health/{name}", dependencies=[auth])
    def health_of(name: str) -> dict | None:
        return missing(catalog.health_of, ws, name)

    @app.get("/api/next-steps/{name}", dependencies=[auth])
    def next_steps_of(name: str) -> list:
        return missing(catalog.steps_of, ws, name)

    @app.get("/api/specialists/{name}", dependencies=[auth])
    def specialist(name: str) -> dict:
        return missing(catalog.detail, ws, name)

    @app.put("/api/specialists/{name}/spec", dependencies=[auth])
    def save_spec(name: str, body: dict) -> dict:
        return missing(catalog.save_spec, ws, name, body)

    @app.delete("/api/specialists/{name}", dependencies=[auth])
    def delete(name: str, confirm: str = Query(...)) -> dict:
        if confirm != name:
            raise HTTPException(422, "pass confirm=<name> to delete a specialist and everything it built")
        missing(catalog.delete, ws, name)
        models.pop(name, None)
        return {"deleted": name}

    @app.get("/api/specialists/{name}/rows", dependencies=[auth])
    def rows(name: str, split: str = "train", offset: int = 0, limit: int = 100) -> dict:
        return missing(catalog.rows, ws, name, split, offset, limit)

    @app.post("/api/specialists/{name}/predict", dependencies=[auth])
    def predict(name: str, body: PredictRequest) -> dict:
        from ..rt import load

        bundle = ws / "specialists" / name / "export"
        if not (bundle / "manifest.json").exists():
            raise HTTPException(404, f"{name} is not exported: run the export job first")
        models[name] = models.get(name) or load(bundle)
        return models[name].predict(body.input)

    @app.get("/api/jobs", dependencies=[auth])
    def jobs() -> list[dict]:
        return [j.public() for j in reversed(runner.jobs.values())]

    @app.post("/api/jobs", dependencies=[auth])
    def submit(body: JobRequest) -> dict:
        return missing(runner.submit, body.args, body.llm).public()

    @app.get("/api/jobs/{job_id}", dependencies=[auth])
    def job(job_id: str) -> dict:
        return missing(runner.detail, job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[auth])
    def cancel(job_id: str) -> dict:
        return missing(runner.cancel, job_id).public()

    @app.get("/api/keys", dependencies=[auth])
    def list_keys() -> list[dict]:
        return keys.list_keys(ws)

    @app.put("/api/keys/{name}", dependencies=[auth])
    def set_key(name: str, body: KeyValue) -> dict:
        missing(keys.set_key, ws, name, body.value)
        return {"name": name, "set": True}

    @app.delete("/api/keys/{name}", dependencies=[auth])
    def delete_key(name: str) -> dict:
        missing(keys.delete_key, ws, name)
        return {"name": name, "set": False}

    @app.websocket("/api/events")
    async def events(sock: WebSocket, token: str = "") -> None:
        if not allowed_origin(sock.headers.get("origin", ""), app.state.hosts):
            return await sock.close(code=UNAUTHORIZED_CLOSE)
        await sock.accept()
        if app.state.token and not secrets.compare_digest(token, app.state.token):
            return await sock.close(code=UNAUTHORIZED_CLOSE)
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


def create_app(
    ws: Path, token: str, hosts: list[str] | None = None, parallel: int | None = None, port: int | None = None
) -> FastAPI:
    from ..predictors.turn import register_on_main_thread

    register_on_main_thread()
    app = FastAPI(title="byosm", summary="Local API over the BYOSM pipeline")
    app.state.token = token
    app.state.hosts = hosts or []
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOCAL_HOSTS, *app.state.hosts])
    lock = claim(ws, port)
    hub = Hub()
    runner = Runner(ws, hub, parallel)
    app.router.on_shutdown.append(lambda: release(lock))
    app.state.runner = runner
    app.router.on_shutdown.append(runner.shutdown)

    def check(authorization: str = Header(""), origin: str = Header("")) -> None:
        if not allowed_origin(origin, app.state.hosts):
            raise HTTPException(403, "requests from another site are not allowed")
        given = authorization[len(BEARER) :] if authorization.startswith(BEARER) else ""
        if token and not secrets.compare_digest(given, token):
            raise HTTPException(401, "missing or wrong token: open the link `byosm app` printed")

    threading.Thread(target=importlib.import_module, args=(WARM_MODULE,), daemon=True).start()
    _routes(app, ws, runner, hub, Depends(check))
    app.include_router(systems_router(ws, Depends(check)))
    app.include_router(config_router(ws, Depends(check)))
    for r in (try_router(ws), workbench_router(ws), live_router(ws, runner), models_router(ws, runner)):
        app.include_router(r, dependencies=[Depends(check)])
    for r in (resources_router(ws), opendata_router(ws), forecast_router(ws), publish_router(ws), archive_router(ws)):
        app.include_router(r, dependencies=[Depends(check)])
    for r in (data_router(ws), panel_routes.router(ws), techniques_router(ws), lab_router(ws)):
        app.include_router(r, dependencies=[Depends(check)])
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app
