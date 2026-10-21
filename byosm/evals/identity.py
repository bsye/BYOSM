from __future__ import annotations

import json

SECRET_KEYS = ("api_key",)


def baseline_identity(pid: str) -> str | None:
    if pid == "llm" or pid.startswith("llm:"):
        return _llm_identity(pid[4:] or "baseline")
    if pid.startswith("hf:"):
        return pid[3:]
    if pid == "jev":
        from ..predictors.jev import jev_identity

        return jev_identity()
    return None


def _llm_identity(role: str) -> str:
    from ..config import load_config
    from ..llm import role_config

    conf = role_config(load_config()["llm"], role)
    return json.dumps({k: v for k, v in conf.items() if k not in SECRET_KEYS}, sort_keys=True)
