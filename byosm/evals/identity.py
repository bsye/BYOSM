from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from ..model_refs import is_llm

if TYPE_CHECKING:
    from ..spec import TaskSpec

SECRET_KEYS = ("api_key",)
RUN_ONLY_KEYS = ("max_parallel",)


def baseline_identity(pid: str) -> str | None:
    if is_llm(pid):
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


def task_hash(spec: TaskSpec) -> str:
    from ..data.prompts import task_context

    return hashlib.sha1(task_context(spec).encode()).hexdigest()[:12]


def rival_identity(pid: str, spec: TaskSpec) -> str | None:
    found = baseline_identity(pid)
    if found is None or not is_llm(pid):
        return found
    return json.dumps({**json.loads(found), "task": task_hash(spec)}, sort_keys=True)


def experiment_identity(meta: dict) -> str:
    return hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
