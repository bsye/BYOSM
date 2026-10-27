from __future__ import annotations

from typing import Any

from .auto.standing import best_baseline
from .data.sufficiency import ENOUGH, MIN_TEST_ITEMS, THIN, data_check
from .evals.overfit import OVERFIT
from .evals.selection import SURE_LEAD, compare
from .export.bundle import exported_current
from .health_score import FAIL, OK, UNKNOWN, WARN, check, graded, report
from .metrics import CONSTRAINED, unreached
from .model_refs import family, llm_model, model_ref
from .project import Project
from .spec import TaskSpec

TARGET_NEAR = 0.05
CI_WIDTH_OK, CI_WIDTH_WARN = 0.04, 0.10
LABEL_OK, LABEL_WARN = 0.8, 0.5
DROP_OK, DROP_WARN = 0.05, 0.15
SLOW_FACTOR = 2.0
LEAD = "lead"
HUB_RIVAL = "hf:"
SMALLER_OK = 4.0


def _lead(p: Project, spec: TaskSpec, mine: dict, rival: str, theirs: dict) -> dict | None:
    saved = p.report(LEAD) or {}
    key = f"{p.promoted}:{rival}:{mine.get('fingerprint')}"
    if key not in saved:
        saved = {key: compare(spec, mine, theirs, "test")}
        p.save_report(LEAD, saved)
    return saved[key]


def _rival(p: Project, spec: TaskSpec, board: dict, prim: str, mine: dict) -> dict:
    best = best_baseline(board, prim)
    if not best:
        return check(
            "rival", "Beats the best rival", 3, UNKNOWN, "no rival scored", "No baseline was scored on this benchmark."
        )
    theirs = board[best[0]][prim]
    cmp = _lead(p, spec, mine, best[0], theirs)
    delta = cmp["delta"] if cmp else mine["objective"] - theirs["objective"]
    sure = cmp["p_better"] if cmp else None
    status = OK if sure is not None and sure >= SURE_LEAD else WARN if delta >= 0 else FAIL
    value = f"{delta:+.3f} vs {model_ref(p, best[0], theirs)['name']}" + (
        f", {sure:.0%} sure" if sure is not None else ""
    )
    why = "Paired bootstrap on the test split: the lead counts when at least 95% of resamples keep it."
    if (missed := unreached(spec.objective, theirs.get("metrics") or {})) is not None:
        o = spec.objective
        why = f"The rival's {CONSTRAINED[o.metric][1]} is {missed:.3f}, under the {o.constraint:g} the metric requires, so it scores 0 and any lead over it is empty."
        return check("rival", "Beats the best rival", 3, UNKNOWN, value + ", rival misses the constraint", why)
    if writer := _written_by_rival(p, spec, prim, best[0], theirs):
        why = f"{writer} wrote this test set, the same model family as the rival, so its score is not a fair bar."
        return check("rival", "Beats the best rival", 3, UNKNOWN, value + ", same family wrote the test", why)
    return check("rival", "Beats the best rival", 3, status, value, why)


def _written_by_rival(p: Project, spec: TaskSpec, prim: str, pid: str, theirs: dict) -> str | None:
    from .data.budget import LLM_SOURCE
    from .data.store import Dataset

    model = llm_model(pid, theirs.get("identity"))
    suite = next((e for e in spec.evals if e.name == prim), None)
    if not model or suite is None or suite.kind != "builtin":
        return None
    rows = Dataset(p.data_dir_for(p.exp_spec(p.promoted))).load("test")
    sources = {str(r.get("source", "")) for r in rows}
    writers = {s.rpartition(":")[2] for s in sources if s.startswith(LLM_SOURCE)}
    return next((w for w in sorted(writers) if family(w) == family(model)), None)


def _size(p: Project, board: dict, prim: str, mine: dict) -> dict | None:
    best = best_baseline(board, prim)
    theirs = board[best[0]][prim].get("params") if best else None
    if not theirs or not mine.get("params"):
        return None
    times = theirs / mine["params"]
    status = OK if times >= SMALLER_OK else WARN if times > 1 else FAIL
    why = f"Parameters of the best rival over this model's: the point of a specialist is to be {SMALLER_OK:g}x smaller or more."
    value = f"{times:.1f}x smaller" if times >= 1 else f"{1 / times:.1f}x bigger"
    args = None if status == OK or p.meta(p.promoted or "").get("distilled_from") else ["distill", p.name, "--run"]
    return check("size", "Smaller than the rival", 1, status, value, why, args)


def _target(spec: TaskSpec, mine: dict) -> dict:
    t = spec.objective.target
    status = graded(mine["objective"], t, t - TARGET_NEAR)
    why = f"The spec asks for {spec.objective.metric} of at least {t} on test."
    return check("target", "Meets the target", 2, status, f"{mine['objective']:.3f} of {t}", why)


def _overfit(p: Project) -> dict:
    r = p.report(OVERFIT)
    if not r or r.get("experiment") != p.promoted:
        why = "No overfitting check for the current model."
        return check("overfit", "No overfitting", 2, UNKNOWN, "not checked", why, ["overfit", p.name])
    return check("overfit", "No overfitting", 2, r["verdict"], r["verdict"], r["summary"])


def _data(p: Project, spec: TaskSpec) -> dict | None:
    c = data_check(p, spec)
    if not c:
        return None
    status = OK if c["verdict"] == ENOUGH else WARN if c["verdict"] == THIN else FAIL
    why = f"{len(c['thin'])} labels below the recommended rows; test {c['test']}, calib {c['calib']}."
    return check("data", "Enough data", 1, status, c["verdict"], why)


def _certainty(mine: dict) -> dict:
    ci, n = mine.get("metrics", {}).get("ci95"), mine.get("n_test") or 0
    if n and n < MIN_TEST_ITEMS:
        why = f"The test split has {n} rows: too few for the score to mean much."
        return check("certainty", "Precise score", 1, FAIL, f"{n} test rows", why)
    if not ci:
        return check("certainty", "Precise score", 1, UNKNOWN, "no interval", "The test score has no 95% interval.")
    width = ci["hi"] - ci["lo"]
    why = "Width of the 95% interval of the test score: narrow means the score can be trusted."
    return check(
        "certainty", "Precise score", 1, graded(width, CI_WIDTH_OK, CI_WIDTH_WARN, False), f"±{width / 2:.3f}", why
    )


def _weakest(mine: dict) -> dict | None:
    per = {k: v["f1"] for k, v in mine.get("metrics", {}).get("per_label", {}).items() if v.get("support")}
    if not per:
        return None
    name, f1 = min(per.items(), key=lambda kv: kv[1])
    why = "F1 of the label the model handles worst on test: one weak label can break a product flow."
    return check("weakest", "No weak label", 1, graded(f1, LABEL_OK, LABEL_WARN), f"{name} {f1:.2f}", why)


def _robustness(p: Project, spec: TaskSpec, board: dict, prim: str) -> dict:
    by_suite = board.get(p.promoted or "", {})
    others = {s: r["objective"] for s, r in by_suite.items() if s != prim and r.get("objective") is not None}
    folded = {"builtin"} if p.meta(p.promoted or "").get("n_test_folded") else set()
    unscored = [e.name for e in spec.evals if e.name != prim and e.name not in others and e.kind not in folded]
    if not others:
        why = "Scored only on its own benchmark. A test set written from scratch shows how it does on new inputs."
        args = ["eval", p.name] if unscored else ["testset", p.name] if spec.task_type == "classify" else None
        if unscored:
            why = f"{', '.join(unscored)} not scored for the current model."
        elif args is None:
            why = (
                "Scored only on its own benchmark. Writing a new test set is supported for classification only so far."
            )
        return check("robust", "Holds on new inputs", 2, UNKNOWN, "one suite only", why, args)
    suite, worst = min(others.items(), key=lambda kv: kv[1])
    found = [r[suite]["objective"] for pid, r in board.items() if pid.startswith(HUB_RIVAL) and suite in r]
    if found := [v for v in found if v is not None]:
        why = "On a test set written by an LLM every trained model drops, so it is judged against the best model fine-tuned on the benchmark."
        status = graded(worst - max(found), -DROP_OK, -DROP_WARN)
        return check(
            "robust", "Holds on new inputs", 2, status, f"{worst:.3f} vs {max(found):.3f} for rivals on {suite}", why
        )
    drop = by_suite[prim]["objective"] - worst
    why = f"Drop from the primary benchmark to the hardest other suite ({len(others)} scored)."
    status = graded(drop, DROP_OK, DROP_WARN, False)
    return check("robust", "Holds on new inputs", 2, status, f"{-drop:+.3f} on {suite}", why)


def _speed(spec: TaskSpec, mine: dict) -> dict:
    ms, budget = mine.get("latency_ms"), spec.latency_budget_ms
    if ms is None:
        return check("speed", "Fast enough", 1, UNKNOWN, "not timed", "No latency measured for the current model.")
    status = graded(ms, budget, budget * SLOW_FACTOR, False)
    return check(
        "speed",
        "Fast enough",
        1,
        status,
        f"{ms:.1f} of {budget:g} ms",
        "CPU latency per input against the spec's budget.",
    )


def _shipped(p: Project) -> dict:
    if exported_current(p.root, p.promoted):
        return check("export", "Exported", 1, OK, "current", "The ONNX bundle matches the promoted model.")
    why = "The ONNX bundle is missing or older than the promoted model."
    return check("export", "Exported", 1, UNKNOWN, "not current", why, ["export", p.name])


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
        _robustness(p, spec, board, prim),
        _data(p, spec),
        _certainty(mine),
        _weakest(mine),
        _speed(spec, mine),
        _size(p, board, prim, mine),
        _shipped(p),
    ]
    return [c for c in found if c]


def health(p: Project, spec: TaskSpec) -> dict[str, Any] | None:
    return report(checks(p, spec))
