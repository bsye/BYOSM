from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI

SETTINGS_ENV = "BYOSM_APP_SETTINGS"


def settings_env(ws: Path, token: str, hosts: list[str], parallel: int | None, port: int) -> dict[str, str]:
    settings = {"ws": str(ws), "token": token, "hosts": hosts, "parallel": parallel, "port": port}
    return {SETTINGS_ENV: json.dumps(settings)}


def from_env() -> FastAPI:
    from .server import create_app

    s = json.loads(os.environ[SETTINGS_ENV])
    return create_app(Path(s["ws"]), s["token"], s["hosts"], s["parallel"], s["port"])
