from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import load_config, save_config, stored_config
from ..evals.identity import SECRET_KEYS
from ..llm import BUDGET_KEY

ROLES = ("default", "spec", "pick", "map", "generate", "eval", "judge", "analyze", "baseline", "card")
EFFORT_KEYS = {"openai": "reasoning_effort", "anthropic": "effort", "claude-code": "effort", "codex": "effort"}
EFFORT_LEVELS = {
    "openai": ["none", "low", "medium", "high"],
    "anthropic": ["low", "medium", "high", "max"],
    "claude-code": ["low", "medium", "high", "max"],
    "codex": ["minimal", "low", "medium", "high"],
}
EFFORT_DEFAULTS = {"anthropic": "medium"}
SETTINGS = ("provider", "model", "base_url", "api_key", "api_key_env", "effort", "max_parallel")


class RoleSettings(BaseModel):
    settings: dict[str, Any]


class Budget(BaseModel):
    tokens_per_job: int | None


class Unload(BaseModel):
    server: str
    model: str


class Stop(BaseModel):
    model: str


class OnLimit(BaseModel):
    mode: str


def _effort_key(conf: dict) -> str:
    return EFFORT_KEYS.get(str(conf.get("provider", "anthropic")), "effort")


def _view(conf: dict) -> dict[str, Any]:
    provider = str(conf.get("provider", "anthropic"))
    shown = {k: conf.get(k) for k in SETTINGS if k not in SECRET_KEYS}
    effort = conf.get(_effort_key(conf)) or EFFORT_DEFAULTS.get(provider)
    return {**shown, "effort": effort, "effort_levels": EFFORT_LEVELS.get(provider, [])}


def roles(ws: Path) -> dict[str, Any]:
    from ..llm import role_config

    cfg = load_config(ws)
    own = cfg.get("llm") or {}
    return {
        "roles": [{"role": r, "own": r in own, **_view(role_config(own, r))} for r in ROLES],
        "budget": {BUDGET_KEY: (cfg.get("budget") or {}).get(BUDGET_KEY)},
    }


def set_role(ws: Path, role: str, settings: dict[str, Any]) -> dict[str, Any]:
    if role not in ROLES:
        raise HTTPException(404, f"unknown role {role!r}; roles are {', '.join(ROLES)}")
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise HTTPException(422, f"unknown settings {sorted(unknown)}; allowed: {', '.join(SETTINGS)}")
    cfg = stored_config(ws)
    llm = cfg.setdefault("llm", {})
    conf = {**(llm.get(role) or {})}
    provider = settings.get("provider") or conf.get("provider") or load_config(ws)["llm"]["default"].get("provider")
    for key, value in settings.items():
        target = _effort_key({"provider": provider}) if key == "effort" else key
        if value in (None, ""):
            conf.pop(target, None)
        else:
            conf[target] = value
    if conf:
        llm[role] = conf
    else:
        llm.pop(role, None)
    save_config(ws, cfg)
    return roles(ws)


def set_budget(ws: Path, tokens: int | None) -> dict[str, Any]:
    cfg = stored_config(ws)
    if tokens:
        cfg["budget"] = {**(cfg.get("budget") or {}), BUDGET_KEY: tokens}
    else:
        cfg.get("budget", {}).pop(BUDGET_KEY, None)
    save_config(ws, cfg)
    return roles(ws)


def _write_limit(ws: Path, **values: Any) -> dict[str, Any]:
    from ..llm.limits import SECTION

    cfg = stored_config(ws)
    cfg[SECTION] = {**(cfg.get(SECTION) or {}), **values}
    save_config(ws, cfg)
    return cfg[SECTION]


def models_router(ws: Path, runner: Any) -> APIRouter:
    from ..llm import local

    router = APIRouter()
    store = runner.store

    @router.get("/api/llm/usage")
    def usage() -> list:
        return store.llm_usage()

    @router.get("/api/llm/local")
    def loaded() -> list:
        return local.loaded()

    @router.post("/api/llm/local/unload")
    def unload(body: Unload) -> list:
        try:
            local.unload(body.server, body.model)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return local.loaded()

    @router.post("/api/llm/stop")
    def stop(body: Stop) -> dict:
        using = store.jobs_using(body.model)
        active = [j for j in runner.jobs.values() if j.id in using and j.status in ("running", "queued")]
        for j in active:
            runner.cancel(j.id)
        return {"cancelled": [j.id for j in active]}

    @router.get("/api/llm/roles")
    def read_roles() -> dict:
        return roles(ws)

    @router.put("/api/llm/roles/{role}")
    def write_role(role: str, body: RoleSettings) -> dict:
        return set_role(ws, role, body.settings)

    @router.get("/api/llm/on-limit")
    def read_on_limit() -> dict:
        from ..llm.limits import LOCAL, MODES, local_llm_config, settings

        return {**settings(), "modes": list(MODES), LOCAL: local_llm_config()}

    @router.put("/api/llm/on-limit")
    def write_on_limit(body: OnLimit) -> dict:
        from ..llm.limits import MODES

        if body.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {', '.join(MODES)}")
        return _write_limit(ws, mode=body.mode)

    @router.post("/api/llm/on-limit/retry")
    def retry_now() -> dict:
        import time

        return _write_limit(ws, retry_now=time.time())

    @router.put("/api/llm/budget")
    def write_budget(body: Budget) -> dict:
        return set_budget(ws, body.tokens_per_job)

    return router
