from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Project
from .data.sufficiency import MIN_CALIB_ITEMS, MIN_TEST_ITEMS, data_check
from .dataset_search import saved_suggestions
from .health import OK, checks
from .improve.proposals import PROPOSALS_FILE
from .spec import TaskSpec

WHY_CHARS = 220
FIX_TITLES = {
    "overfit": "Check for overfitting",
    "robust": "Score on inputs it never saw",
    "export": "Export the current model",
}


def _step(sid: str, title: str, why: str, args: list[str]) -> dict[str, Any]:
    return {"id": sid, "title": title, "why": why[:WHY_CHARS], "args": args}


def _held_out(p: Project, check: dict | None) -> dict[str, Any] | None:
    if not check or (check["test"] >= MIN_TEST_ITEMS and check["calib"] >= MIN_CALIB_ITEMS) or not check["test"]:
        return None
    why = f"The test set has {check['test']} rows and calib {check['calib']}: scores on so few rows say almost nothing. Rebuilding the data splits by group of similar rows."
    return _step("rebuild-data", "Rebuild the data with a bigger held-out set", why, ["build", p.name, "--force-data"])


def _more_data(p: Project, check: dict | None) -> dict[str, Any] | None:
    if not check or not check["thin"]:
        return None
    labels = ", ".join(check["thin"][:4]) + (" and more" if len(check["thin"]) > 4 else "")
    found = saved_suggestions(p)
    if not found:
        why = f"{labels} have fewer training rows than recommended. The LLM searches the Hub for labelled datasets that fit."
        return _step("find-data", "Find more training data", why, ["datasets", p.name])
    top = found[0]
    why = f"{labels} have fewer training rows than recommended. {top['dataset']} fits: {top['why']}"
    return _step("more-data", f"Train with {top['dataset']}", why, top["args"])


def _rivals(p: Project, spec: TaskSpec, prim: str) -> dict[str, Any] | None:
    scored = {k for k, r in p.board().items() if prim in r and k not in p.experiments()}
    missing = [b for b in spec.baselines if b not in scored]
    if not missing:
        return None
    return _step("score-rivals", "Score the rivals", f"Not scored yet: {', '.join(missing)}.", ["eval", p.name])


def tried_proposals(p: Project) -> set[str]:
    out = set()
    for exp in p.experiments():
        f = p.exp_dir(exp) / "proposal.json"
        if f.exists():
            out |= {a["id"] for a in json.loads(f.read_text()).get("applied", [])}
    return out


def _experiments(p: Project) -> dict[str, Any]:
    f = p.root / PROPOSALS_FILE
    saved = json.loads(f.read_text()) if f.exists() else {}
    fresh = [pr["id"] for pr in saved.get("proposals", []) if pr["id"] not in tried_proposals(p)]
    if saved.get("from") != p.promoted or not fresh:
        why = "Every lined-up experiment has run. The LLM reads the board and the errors and lines up new ones."
        return _step("propose", "Find more experiments", why, ["improve", p.name])
    why = "Each one builds a new model, scores it against the rivals and is promoted only if it wins on calib."
    step = _step(
        "experiments",
        f"Run {len(fresh)} experiments",
        why,
        ["improve", p.name, "--apply", ",".join(fresh), "--each", "--run"],
    )
    return {**step, "experiments": fresh}


def _from_checks(p: Project, spec: TaskSpec) -> list[dict[str, Any]]:
    return [
        _step(c["id"], FIX_TITLES[c["id"]], c["why"], c["args"])
        for c in checks(p, spec)
        if c["args"] and c["status"] != OK
    ]


def suggestions(name: str, ws: Path | None = None) -> list[dict[str, Any]]:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    spec = TaskSpec.load(p.spec_path)
    if not p.promoted:
        return [_step("build", "Build the first model", "Nothing has been trained yet.", ["build", name])]
    prim = spec.primary_suite()
    check = data_check(p, spec)
    steps = [
        _held_out(p, check),
        _more_data(p, check),
        _rivals(p, spec, prim),
        _experiments(p),
    ]
    return [s for s in steps if s] + _from_checks(p, spec)
