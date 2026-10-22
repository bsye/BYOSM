from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import CONFIG_NAME, HF_TOKEN_KEY, load_config, secret
from ..evals.identity import SECRET_KEYS

MASK = "********"


class ConfigText(BaseModel):
    text: str


class DefaultLlm(BaseModel):
    config: dict[str, Any]


def _roles(cfg: dict) -> dict[str, dict]:
    return {k: v for k, v in (cfg.get("llm") or {}).items() if isinstance(v, dict)}


def _masked(cfg: dict) -> dict:
    llm = {k: {f: MASK if f in SECRET_KEYS and v else v for f, v in role.items()} for k, role in _roles(cfg).items()}
    return {**cfg, "llm": llm} if llm else cfg


def _unmasked(new: dict, old: dict) -> dict:
    before = _roles(old)
    for name, role in _roles(new).items():
        for f in SECRET_KEYS:
            if role.get(f) == MASK:
                role[f] = before.get(name, {}).get(f)
    return new


@functools.cache
def _hub_user(token: str | None) -> str | None:
    from huggingface_hub import whoami

    try:
        return whoami(token=token)["name"]
    except Exception:
        return None


def config_router(ws: Path, auth: Any) -> APIRouter:
    router = APIRouter(dependencies=[auth])
    path = ws / CONFIG_NAME

    def stored() -> dict:
        return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}

    @router.get("/api/config")
    def read() -> dict:
        return {
            "text": yaml.safe_dump(_masked(stored()), sort_keys=False),
            "effective": _masked(load_config(ws)),
        }

    @router.put("/api/config")
    def write(body: ConfigText) -> dict:
        try:
            new = yaml.safe_load(body.text) or {}
        except yaml.YAMLError as e:
            raise HTTPException(422, f"not valid YAML: {e}") from e
        if not isinstance(new, dict) or not isinstance(new.get("llm", {}), dict):
            raise HTTPException(422, "byosm.yaml is a mapping with an optional llm: section of roles")
        path.write_text(yaml.safe_dump(_unmasked(new, stored()), sort_keys=False))
        return read()

    @router.put("/api/config/default-llm")
    def default_llm(body: DefaultLlm) -> dict:
        cfg = stored()
        cfg["llm"] = {**(cfg.get("llm") or {}), "default": {k: v for k, v in body.config.items() if v is not None}}
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return read()

    @router.get("/api/hub/user")
    def hub_user() -> dict:
        return {"user": _hub_user(secret(HF_TOKEN_KEY, ws))}

    @router.get("/api/llm/providers")
    def providers() -> dict:
        from ..llm.detect import detect

        return detect()

    return router
