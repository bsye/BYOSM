from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Project
from .improve.proposals import PROPOSALS_FILE
from .spec import TaskSpec

MIN_TEST_ITEMS = 100
MIN_CALIB_ITEMS = 50
WHY_CHARS = 220


def _step(sid: str, title: str, why: str, args: list[str]) -> dict[str, Any]:
    return {"id": sid, "title": title, "why": why[:WHY_CHARS], "args": args}


def _held_out(p: Project, prim: str) -> dict[str, Any] | None:
    r = p.board().get(p.promoted or "", {}).get(prim)
    if not r:
        return None
    n_test, n_calib = r.get("n_test") or 0, r.get("n_calib") or 0
    if n_test >= MIN_TEST_ITEMS and n_calib >= MIN_CALIB_ITEMS:
        return None
    why = f"The test set has {n_test} rows and calib {n_calib}: scores on so few rows say almost nothing. Rebuilding the data splits by group of similar rows."
    return _step("rebuild-data", "Rebuild the data with a bigger held-out set", why, ["build", p.name, "--force-data"])


def _rivals(p: Project, spec: TaskSpec, prim: str) -> dict[str, Any] | None:
    scored = {k for k, r in p.board().items() if prim in r and k not in p.experiments()}
    missing = [b for b in spec.baselines if b not in scored]
    if not missing:
        return None
    return _step("score-rivals", "Score the rivals", f"Not scored yet: {', '.join(missing)}.", ["eval", p.name])


def _proposal(p: Project) -> dict[str, Any] | None:
    f = p.root / PROPOSALS_FILE
    if not f.exists():
        return _step("propose", "Find what to try next", "No improvement has been proposed yet.", ["improve", p.name])
    saved = json.loads(f.read_text())
    tried = set(p.experiments())
    fresh = [pr for pr in saved.get("proposals", []) if f"{pr['id']}" not in tried]
    if saved.get("from") != p.promoted or not fresh:
        return _step(
            "propose", "Find what to try next", "The saved proposals are for an older model.", ["improve", p.name]
        )
    top = fresh[0]
    return _step(
        f"try-{top['id']}", f"Try {top['id']}", top.get("why", ""), ["improve", p.name, "--apply", top["id"], "--run"]
    )


def _overfit(p: Project) -> dict[str, Any] | None:
    report = json.loads(p.overfit_path.read_text()) if p.overfit_path.exists() else None
    if report and report.get("experiment") == p.promoted:
        return None
    return _step("overfit", "Check for overfitting", "No overfitting check for the current model.", ["overfit", p.name])


def _export(p: Project) -> dict[str, Any] | None:
    from .export.bundle import is_exported

    manifest = p.root / "export" / "manifest.json"
    if is_exported(p.root / "export") and json.loads(manifest.read_text()).get("experiment") == p.promoted:
        return None
    why = "The ONNX bundle is missing or older than the promoted model."
    return _step("export", "Export the current model", why, ["export", p.name])


def suggestions(name: str, ws: Path | None = None) -> list[dict[str, Any]]:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    spec = TaskSpec.load(p.spec_path)
    if not p.promoted:
        return [_step("build", "Build the first model", "Nothing has been trained yet.", ["build", name])]
    prim = spec.primary_suite()
    held = _held_out(p, prim) if spec.task_type == "classify" else None
    if held:
        return [held]
    steps = [_rivals(p, spec, prim), _overfit(p), _proposal(p), _export(p)]
    return [s for s in steps if s]
