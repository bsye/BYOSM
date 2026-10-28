from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

from ..project import Project
from .contract import ALLOWED_IMPORTS, HOOKS, SAFE_BUILTINS
from .drafts import APPROVED, draft, draft_id
from .proven import BUILTIN_PREFIX, METHODS_DIR, registry
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


def _hooks(code: str, label: str) -> dict[str, Callable[..., Any]]:
    namespace: dict[str, Any] = {"__builtins__": _builtins(), "__name__": f"lab_{label}"}
    exec(compile(code, f"<lab method {label}>", "exec"), namespace)
    return {h: namespace[h] for h in HOOKS if callable(namespace.get(h))}


def builtin_code(name: str) -> str:
    return (METHODS_DIR / f"{name}.py").read_text()


def builtin(name: str) -> dict[str, Callable[..., Any]]:
    code = builtin_code(name)
    if problems(code):
        raise PermissionError(f"the shipped method {name} fails its checks")
    return _hooks(code, name)


def _approved(p: Project, did: str) -> str:
    found = draft(p, did)
    proven = registry(p.ws).get(did)
    if found and found["status"] == APPROVED:
        return found["code"]
    if proven:
        return proven["code"]
    raise PermissionError(f"lab method {did} is not approved")


def load(p: Project, did: str) -> dict[str, Callable[..., Any]]:
    if did.startswith(BUILTIN_PREFIX):
        return builtin(did.removeprefix(BUILTIN_PREFIX))
    code = _approved(p, did)
    if draft_id(code) != did or problems(code):
        raise PermissionError(f"lab method {did} changed or fails its checks since it was approved")
    return _hooks(code, did)
