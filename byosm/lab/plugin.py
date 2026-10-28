from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any

from .contract import ALLOWED_IMPORTS, CLASSIFIER, CONTRACTS, SAFE_BUILTINS, Hooks
from .drafts import APPROVED, draft, draft_id
from .owners import Owner, kind_of
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


def _hooks(code: str, label: str, kind: str) -> Hooks:
    namespace: dict[str, Any] = {"__builtins__": _builtins(), "__name__": f"lab_{label}"}
    exec(compile(code, f"<lab method {label}>", "exec"), namespace)
    return {h: namespace[h] for h in CONTRACTS[kind].hooks if callable(namespace.get(h))}


def builtin_code(name: str) -> str:
    return (METHODS_DIR / f"{name}.py").read_text()


def builtin(name: str, kind: str = CLASSIFIER) -> Hooks:
    code = builtin_code(name)
    if problems(code, kind):
        raise PermissionError(f"the shipped method {name} fails its checks")
    return _hooks(code, name, kind)


def _approved(owner: Owner, did: str) -> str:
    found = draft(owner, did)
    proven = registry(owner.ws).get(did)
    if found and found["status"] == APPROVED:
        return found["code"]
    if proven:
        return proven["code"]
    raise PermissionError(f"lab method {did} is not approved")


def load(owner: Owner, did: str) -> Hooks:
    kind = kind_of(owner)
    if did.startswith(BUILTIN_PREFIX):
        return builtin(did.removeprefix(BUILTIN_PREFIX), kind)
    code = _approved(owner, did)
    if draft_id(code) != did or problems(code, kind):
        raise PermissionError(f"lab method {did} changed or fails its checks since it was approved")
    return _hooks(code, did, kind)


@dataclass(frozen=True)
class Method:
    owner: Owner
    id: str

    def hooks(self) -> Hooks:
        return load(self.owner, self.id)
