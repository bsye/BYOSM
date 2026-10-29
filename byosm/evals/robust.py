from __future__ import annotations

from ..health_score import CHECK_TITLES, DROP_OK, DROP_WARN, OK, UNKNOWN, check, graded
from ..project import Project
from ..spec import TaskSpec
from .drop import worst_drop
from .grouped import cases

HUB_RIVAL = "hf:"
AIMABLE_ERRORS = 20


def weakest_case(result: dict, suite: str) -> str:
    found = {c: e for c, e in (cases(result.get("metrics") or {}) or {}).items() if e["accuracy"] is not None}
    if not found:
        return ""
    name, exact = min(found.items(), key=lambda kv: kv[1]["accuracy"])
    readable = name.replace("_", " ")
    return f" Weakest kind of request on {suite}: {readable}, {exact['accuracy']:.2f} exact over {exact['n']} requests."


def _fix(p: Project) -> tuple[list[str], str]:
    errors = len((p.experiment(p.promoted or "") or {}).get("errors") or [])
    if errors >= AIMABLE_ERRORS:
        return [
            "continue",
            p.name,
            "--run",
        ], " Keep training writes rows that vary phrasing and context, aimed at the dev errors."
    why = f" Dev has only {errors} errors, too few for Keep training to aim at, so the proposer picks a technique aimed at this check."
    return ["improve", p.name], why


def robust_check(p: Project, spec: TaskSpec, board: dict, prim: str) -> dict:
    by_suite = board.get(p.promoted or "", {})
    others = {s: r for s, r in by_suite.items() if s != prim and r.get("objective") is not None}
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
        return check("robust", CHECK_TITLES["robust"], 2, UNKNOWN, "one suite only", why, args)
    suite, drop, metric = worst_drop(by_suite[prim], others, spec.objective.metric)
    hint = weakest_case(by_suite[suite], suite)
    worst = others[suite]["objective"]
    found = [r[suite]["objective"] for pid, r in board.items() if pid.startswith(HUB_RIVAL) and suite in r]
    if found := [v for v in found if v is not None]:
        why = "On a test set written by an LLM every trained model drops, so it is judged against the best model fine-tuned on the benchmark."
        why += hint
        status = graded(worst - max(found), -DROP_OK, -DROP_WARN)
        args = ["continue", p.name, "--run"] if status != OK else None
        value = f"{worst:.3f} vs {max(found):.3f} for rivals on {suite}"
        return check("robust", CHECK_TITLES["robust"], 2, status, value, why, args)
    why = f"Drop in {metric} from the primary benchmark to the hardest other suite ({len(others)} scored).{hint}"
    status = graded(drop, DROP_OK, DROP_WARN, False)
    args = None
    if status != OK:
        args, more = _fix(p)
        why += more
    return check("robust", CHECK_TITLES["robust"], 2, status, f"{-drop:+.3f} {metric} on {suite}", why, args)
