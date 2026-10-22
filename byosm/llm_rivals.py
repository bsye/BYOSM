from __future__ import annotations

from typing import Any

from .config import save_config, stored_config
from .evals.workbench import choose_models, chosen_models
from .project import Project

ROLE_PREFIX = "rival-"
LLM_ID = "llm"
KEPT_CHARS = "-_"


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() or c in KEPT_CHARS else "-" for c in text.lower())
    return "-".join(part for part in out.split("-") if part)


def add_llm_rival(p: Project, conf: dict[str, Any]) -> str:
    if not conf.get("model") and not conf.get("provider"):
        raise ValueError("pick an LLM first")
    role = ROLE_PREFIX + _slug(str(conf.get("model") or conf["provider"]))
    stored = stored_config(p.ws)
    stored.setdefault("llm", {})[role] = conf
    save_config(p.ws, stored)
    pid = f"{LLM_ID}:{role}"
    p.add_baselines([pid])
    spec = p.spec()
    models = chosen_models(p, spec)
    if pid not in models:
        choose_models(p, [*models, pid])
    return pid
