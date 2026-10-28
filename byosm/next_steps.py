from __future__ import annotations

from pathlib import Path
from typing import Any

from .data.sufficiency import MIN_CALIB_ITEMS, MIN_TEST_ITEMS, data_check
from .dataset_search import saved_suggestions
from .finetuned_rivals import REPORT as FINETUNED
from .finetuned_rivals import benchmarks
from .health import checks
from .health_score import CREDIT, FAIL, OK, UNKNOWN
from .held_out import bigger_held_out
from .improve.keys import proposal_key
from .improve.proposals import PROPOSALS
from .next_fixes import faster, less_overfit
from .project import Project
from .spec import TaskSpec

WHY_CHARS = 300
EXPERIMENT_CHECKS = ("target", "rival", "weakest")
STEP_CHECKS = {
    "adopt-data": ("certainty",),
    "rebuild-data": ("certainty",),
    "find-data": ("data", "weakest"),
    "more-data": ("data", "weakest"),
    "finetuned": ("rival",),
    "score-rivals": ("rival",),
    "experiments": EXPERIMENT_CHECKS,
    "propose": EXPERIMENT_CHECKS,
    "overfit": ("overfit",),
    "fix-overfit": ("overfit",),
    "train-longer": ("overfit",),
    "faster": ("speed",),
    "robust": ("robust",),
    "export": ("export",),
    "seeds": ("seeds",),
}
FIX_TITLES = {
    "overfit": "Check for overfitting",
    "robust": "Score on inputs it never saw",
    "export": "Export the current model",
    "size": "Distill into a smaller model",
    "seeds": "Measure seed noise",
}
COMMAND_TITLES = {"continue": "Train on varied inputs"}


def _step(sid: str, title: str, why: str, args: list[str]) -> dict[str, Any]:
    return {"id": sid, "title": title, "why": why[:WHY_CHARS], "args": args}


def _held_out(p: Project, check: dict | None) -> dict[str, Any] | None:
    if not check or (check["test"] >= MIN_TEST_ITEMS and check["calib"] >= MIN_CALIB_ITEMS) or not check["test"]:
        return None
    if bigger := bigger_held_out(p):
        exp, n_test, n_calib = bigger
        why = f"Every model is scored on {check['test']} test rows, too few to tell them apart. {exp} built {n_test} test and {n_calib} calib rows: make them the benchmark, rescore every model and promote the best on calib."
        return _step("adopt-data", f"Score every model on {exp}'s data", why, ["adopt-data", p.name, exp])
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


def _finetuned(p: Project, spec: TaskSpec) -> dict[str, Any] | None:
    if spec.modality != "text" or spec.task_type != "classify" or not benchmarks(spec) or p.report(FINETUNED):
        return None
    why = f"Models fine-tuned on {benchmarks(spec)[0]} are the fairest rival. Searches Hugging Face, adds the best ones whose labels map onto ours, and scores them."
    return _step("finetuned", "Score fine-tuned rivals", why, ["rivals", p.name])


def _rivals(p: Project, spec: TaskSpec, prim: str) -> dict[str, Any] | None:
    mine = set(p.experiments())
    scored = {k for k, r in p.board().items() if prim in r and k not in mine}
    missing = [b for b in spec.baselines if b not in scored]
    if not missing:
        return None
    return _step("score-rivals", "Score the rivals", f"Not scored yet: {', '.join(missing)}.", ["eval", p.name])


def tried_proposals(p: Project) -> set[str]:
    return {proposal_key(a) for exp in p.attempts() for a in (p.proposal(exp) or {}).get("applied", [])}


def _experiments(p: Project) -> dict[str, Any]:
    saved = p.report(PROPOSALS) or {}
    tried = tried_proposals(p)
    fresh = [pr["id"] for pr in saved.get("proposals", []) if proposal_key(pr) not in tried]
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
        _step(c["id"], COMMAND_TITLES.get(c["args"][0], FIX_TITLES[c["id"]]), c["why"], c["args"])
        for c in checks(p, spec)
        if c["args"] and c["status"] != OK
    ]


def _with_gain(step: dict[str, Any], cs: list[dict]) -> dict[str, Any]:
    total = sum(c["weight"] for c in cs) or 1
    hit = [c for c in cs if c["id"] in STEP_CHECKS.get(step["id"], ())]
    gain = round(100 * sum(c["weight"] * (1 - CREDIT[c["status"]]) for c in hit) / total)
    return {**step, "gain": gain, "unblocks": [c["title"] for c in hit if c["status"] in (FAIL, UNKNOWN)]}


def unique(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    return [s for s in steps if not (tuple(s["args"]) in seen or seen.add(tuple(s["args"])))]


def suggestions(name: str, ws: Path | None = None) -> list[dict[str, Any]]:
    p = Project(name, ws)
    if not p.exists():
        raise KeyError(f"no specialist {name!r}")
    spec = p.spec()
    if not p.promoted:
        first = _step("build", "Build the first model", "Nothing has been trained yet.", ["build", name])
        return [{**first, "gain": None, "unblocks": []}]
    prim = spec.primary_suite()
    check = data_check(p, spec)
    cs = checks(p, spec)
    steps = [
        _held_out(p, check),
        _more_data(p, check),
        _finetuned(p, spec),
        _rivals(p, spec, prim),
        less_overfit(p, spec, cs),
        faster(p, spec, cs),
        _experiments(p),
    ]
    return [_with_gain(s, cs) for s in unique([s for s in steps if s] + _from_checks(p, spec))]
