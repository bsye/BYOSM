from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import ALLOWED_IMPORTS, HOOKS, SAFE_BUILTINS
from .drafts import APPROVED, draft, draft_id
from .static import problems

CLASS_BUILDER = "__build_class__"
EXTRA_BUILTINS = ("print",)


def _guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"lab methods cannot import {name}")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _builtins() -> dict[str, Any]:
    safe = {n: getattr(builtins, n) for n in (*SAFE_BUILTINS, *EXTRA_BUILTINS)}
    return safe | {"__import__": _guarded_import, CLASS_BUILDER: getattr(builtins, CLASS_BUILDER)}


def load(p: Project, did: str) -> dict[str, Callable[..., Any]]:
    found = draft(p, did)
    if not found or found["status"] != APPROVED:
        raise PermissionError(f"lab method {did} is not approved")
    code = found["code"]
    if draft_id(code) != did or problems(code):
        raise PermissionError(f"lab method {did} changed or fails its checks since it was approved")
    namespace: dict[str, Any] = {"__builtins__": _builtins(), "__name__": f"lab_{did}"}
    exec(compile(code, f"<lab method {did}>", "exec"), namespace)
    return {h: namespace[h] for h in HOOKS if callable(namespace.get(h))}
