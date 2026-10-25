from __future__ import annotations

import hashlib
import json

SECRET_KEYS = ("api_key",)
RUN_ONLY_KEYS = ("max_parallel",)


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
    return json.dumps({k: v for k, v in conf.items() if k not in (*SECRET_KEYS, *RUN_ONLY_KEYS)}, sort_keys=True)


def experiment_identity(meta: dict) -> str:
    return hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
