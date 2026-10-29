from __future__ import annotations

from ..health_score import CHECK_TITLES, DROP_OK, DROP_WARN, OK, UNKNOWN, WARN, check, graded
from ..project import Project
from ..spec import TaskSpec
from .drop import worst_drop

REAL_KINDS = ("hf", "local")
TITLE = CHECK_TITLES["real"]
WEIGHT = 2
WHY = "Real inputs from a public set the model never trained on, so its score is not only about its own benchmark."


def _trained(spec: TaskSpec) -> set[str]:
    return {s.dataset for s in spec.data.hf_sources} | {spec.data.local_source}


def real_suites(spec: TaskSpec) -> list[str]:
    prim, trained = spec.primary_suite(), _trained(spec)
    return [
        e.name for e in spec.evals if e.kind in REAL_KINDS and e.name != prim and e.origin and e.origin not in trained
    ]


def _primary_is_real(spec: TaskSpec) -> bool:
    prim = next((e for e in spec.evals if e.name == spec.primary_suite()), None)
    return bool(prim and prim.kind in REAL_KINDS and prim.origin)


def _missing(p: Project, spec: TaskSpec, real: list[str]) -> dict:
    from ..benchmarks.catalog import fitting

    if real:
        args = ["eval", p.name, *(a for s in real for a in ("--suite", s))]
        return check("real", TITLE, WEIGHT, UNKNOWN, "not scored", f"{', '.join(real)} not scored yet.", args)
    if fits := [b.name for b in fitting(spec)]:
        why = f"Never scored on a public set of real inputs. Adds {', '.join(fits)} after a protocol check."
        return check("real", TITLE, WEIGHT, UNKNOWN, "not tested", why, ["suite", "public", p.name])
    return check("real", TITLE, 1, WARN, "no public set fits", "No public set of real inputs fits this task yet.")


def real_check(p: Project, spec: TaskSpec, board: dict) -> dict | None:
    if spec.task_type != "classify":
        return None
    if _primary_is_real(spec):
        origin = next(e.origin for e in spec.evals if e.name == spec.primary_suite())
        return check("real", TITLE, WEIGHT, OK, f"benchmark is {origin}", WHY)
    mine, real = board.get(p.promoted or "", {}), real_suites(spec)
    scored = {s: mine[s] for s in real if mine.get(s, {}).get("objective") is not None}
    found = worst_drop(mine[spec.primary_suite()], scored, spec.objective.metric) if scored else None
    if found is None:
        return _missing(p, spec, real)
    suite, drop, metric = found
    status = graded(drop, DROP_OK, DROP_WARN, False)
    return check("real", TITLE, WEIGHT, status, f"{-drop:+.3f} {metric} on {suite} ({len(scored)} real sets)", WHY)
