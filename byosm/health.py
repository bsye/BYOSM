from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .auto.standing import best_baseline
from .config import Project
from .data.sufficiency import ENOUGH, MIN_TEST_ITEMS, THIN, data_check
from .evals.selection import compare
from .export.bundle import exported_current
from .spec import TaskSpec

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"
CREDIT = {OK: 1.0, WARN: 0.5, FAIL: 0.0, UNKNOWN: 0.0}
READY_SCORE = 80
SURE_LEAD = 0.95
TARGET_NEAR = 0.05
CI_WIDTH_OK, CI_WIDTH_WARN = 0.04, 0.10
LABEL_OK, LABEL_WARN = 0.8, 0.5
DROP_OK, DROP_WARN = 0.05, 0.15
SLOW_FACTOR = 2.0
LEAD_FILE = Path("cache") / "lead.json"


def _check(cid: str, title: str, weight: int, status: str, value: str, why: str, args: list[str] | None = None):
    return {"id": cid, "title": title, "weight": weight, "status": status, "value": value, "why": why, "args": args}


def _graded(v: float, ok: float, warn: float, higher_is_better: bool = True) -> str:
    if not higher_is_better:
        v, ok, warn = -v, -ok, -warn
    return OK if v >= ok else WARN if v >= warn else FAIL


def _lead(p: Project, spec: TaskSpec, mine: dict, theirs: dict) -> dict | None:
    f = p.root / LEAD_FILE
    saved = json.loads(f.read_text()) if f.exists() else {}
    key = f"{mine.get('fingerprint')}:{theirs.get('fingerprint')}"
    if key not in saved:
        saved = {key: compare(spec, mine, theirs, "test")}
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(saved))
    return saved[key]


def _rival(p: Project, spec: TaskSpec, board: dict, prim: str, mine: dict) -> dict:
    best = best_baseline(board, prim)
    if not best:
        return _check(
            "rival", "Beats the best rival", 3, UNKNOWN, "no rival scored", "No baseline was scored on this benchmark."
        )
    theirs = board[best[0]][prim]
    cmp = _lead(p, spec, mine, theirs)
    delta = cmp["delta"] if cmp else mine["objective"] - theirs["objective"]
    sure = cmp["p_better"] if cmp else None
    status = OK if sure is not None and sure >= SURE_LEAD else WARN if delta >= 0 else FAIL
    value = f"{delta:+.3f} vs {best[0]}" + (f", {sure:.0%} sure" if sure is not None else "")
    why = "Paired bootstrap on the test split: the lead counts when at least 95% of resamples keep it."
    return _check("rival", "Beats the best rival", 3, status, value, why)


def _target(spec: TaskSpec, mine: dict) -> dict:
    t = spec.objective.target
    status = _graded(mine["objective"], t, t - TARGET_NEAR)
    why = f"The spec asks for {spec.objective.metric} of at least {t} on test."
    return _check("target", "Meets the target", 2, status, f"{mine['objective']:.3f} of {t}", why)


def _overfit(p: Project) -> dict:
    r = json.loads(p.overfit_path.read_text()) if p.overfit_path.exists() else None
    if not r or r.get("experiment") != p.promoted:
        why = "No overfitting check for the current model."
        return _check("overfit", "No overfitting", 2, UNKNOWN, "not checked", why, ["overfit", p.name])
    return _check("overfit", "No overfitting", 2, r["verdict"], r["verdict"], r["summary"])


def _data(p: Project, spec: TaskSpec) -> dict | None:
    c = data_check(p, spec)
    if not c:
        return None
    status = OK if c["verdict"] == ENOUGH else WARN if c["verdict"] == THIN else FAIL
    why = f"{len(c['thin'])} labels below the recommended rows; test {c['test']}, calib {c['calib']}."
    return _check("data", "Enough data", 1, status, c["verdict"], why)


def _certainty(mine: dict) -> dict:
    ci, n = mine.get("metrics", {}).get("ci95"), mine.get("n_test") or 0
    if n and n < MIN_TEST_ITEMS:
        why = f"The test split has {n} rows: too few for the score to mean much."
        return _check("certainty", "Precise score", 1, FAIL, f"{n} test rows", why)
    if not ci:
        return _check("certainty", "Precise score", 1, UNKNOWN, "no interval", "The test score has no 95% interval.")
    width = ci["hi"] - ci["lo"]
    why = "Width of the 95% interval of the test score: narrow means the score can be trusted."
    return _check(
        "certainty", "Precise score", 1, _graded(width, CI_WIDTH_OK, CI_WIDTH_WARN, False), f"±{width / 2:.3f}", why
    )


def _weakest(mine: dict) -> dict | None:
    per = {k: v["f1"] for k, v in mine.get("metrics", {}).get("per_label", {}).items() if v.get("support")}
    if not per:
        return None
    name, f1 = min(per.items(), key=lambda kv: kv[1])
    why = "F1 of the label the model handles worst on test: one weak label can break a product flow."
    return _check("weakest", "No weak label", 1, _graded(f1, LABEL_OK, LABEL_WARN), f"{name} {f1:.2f}", why)


def _robustness(p: Project, spec: TaskSpec, by_suite: dict, prim: str) -> dict:
    others = {s: r["objective"] for s, r in by_suite.items() if s != prim and r.get("objective") is not None}
    unscored = [e.name for e in spec.evals if e.name != prim and e.name not in others]
    if not others:
        why = "Scored only on its own benchmark. A test set written from scratch shows how it does on new inputs."
        args = ["eval", p.name] if unscored else ["testset", p.name]
        if unscored:
            why = f"{', '.join(unscored)} not scored for the current model."
        return _check("robust", "Holds on new inputs", 2, UNKNOWN, "one suite only", why, args)
    suite, worst = min(others.items(), key=lambda kv: kv[1])
    drop = by_suite[prim]["objective"] - worst
    why = f"Drop from the primary benchmark to the hardest other suite ({len(others)} scored)."
    status = _graded(drop, DROP_OK, DROP_WARN, False)
    return _check("robust", "Holds on new inputs", 2, status, f"{-drop:+.3f} on {suite}", why)


def _speed(spec: TaskSpec, mine: dict) -> dict:
    ms, budget = mine.get("latency_ms"), spec.latency_budget_ms
    if ms is None:
        return _check("speed", "Fast enough", 1, UNKNOWN, "not timed", "No latency measured for the current model.")
    status = _graded(ms, budget, budget * SLOW_FACTOR, False)
    return _check(
        "speed",
        "Fast enough",
        1,
        status,
        f"{ms:.1f} of {budget:g} ms",
        "CPU latency per input against the spec's budget.",
    )


def _shipped(p: Project) -> dict:
    if exported_current(p.root, p.promoted):
        return _check("export", "Exported", 1, OK, "current", "The ONNX bundle matches the promoted model.")
    why = "The ONNX bundle is missing or older than the promoted model."
    return _check("export", "Exported", 1, UNKNOWN, "not current", why, ["export", p.name])


def checks(p: Project, spec: TaskSpec) -> list[dict]:
    board, prim = p.board(), spec.primary_suite()
    by_suite = board.get(p.promoted or "", {})
    mine = by_suite.get(prim)
    if not mine or mine.get("objective") is None:
        return []
    found = [
        _rival(p, spec, board, prim, mine),
        _target(spec, mine),
        _overfit(p),
        _robustness(p, spec, by_suite, prim),
        _data(p, spec),
        _certainty(mine),
        _weakest(mine),
        _speed(spec, mine),
        _shipped(p),
    ]
    return [c for c in found if c]


def health(p: Project, spec: TaskSpec) -> dict[str, Any] | None:
    cs = checks(p, spec)
    if not cs:
        return None
    score = round(100 * sum(c["weight"] * CREDIT[c["status"]] for c in cs) / sum(c["weight"] for c in cs))
    blockers = [c["title"] for c in cs if c["status"] in (FAIL, UNKNOWN)]
    return {
        "score": score,
        "ready": score >= READY_SCORE and not blockers,
        "ready_score": READY_SCORE,
        "blockers": blockers,
        "checks": cs,
    }
