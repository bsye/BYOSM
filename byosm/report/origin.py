from __future__ import annotations

from pathlib import Path

import yaml

from ..config import Project
from ..hub.facts import lineage

MANUAL = "manual"
BASE_SPEC = "spec.yaml"


def _flat(d: dict | None, pre: str = "") -> dict:
    out: dict = {}
    for k, v in (d or {}).items():
        out |= _flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v}
    return out


def _spec_path(p: Project, exp: str) -> Path:
    return p.spec_path if exp == BASE_SPEC else p.exp_dir(exp) / "spec.yaml"


def _fields(path: Path) -> dict:
    return _flat(yaml.safe_load(path.read_text())) if path.exists() else {}


def _covered(key: str, paths: list[str]) -> bool:
    return any(key == q or key.startswith(f"{q}.") or q.startswith(f"{key}.") for q in paths)


def _unexplained(p: Project, step: dict) -> list[str]:
    a, b = _fields(_spec_path(p, step["parent"])), _fields(_spec_path(p, step["exp"]))
    paths = [op["path"] for ap in step["applied"] for op in ap.get("ops", [])]
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    return [k for k in changed if not _covered(k, paths)]


def _change(applied: dict) -> str:
    return f"`{applied['why']}`" if applied["id"] == MANUAL else applied["id"]


def built_from(p: Project, exp: str) -> str:
    step = lineage(p, exp)[-1]
    if not step.get("parent"):
        return "no recorded parent"
    text = f"{step['parent']} + " + ", ".join(_change(a) for a in step["applied"])
    extra = _unexplained(p, step)
    if extra:
        sections = ", ".join(sorted({k.partition(".")[0] for k in extra}))
        text += f"; spec also differs in {len(extra)} other field{'s' if len(extra) > 1 else ''} ({sections})"
    return text
