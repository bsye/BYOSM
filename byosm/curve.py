from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .evals.selection import selection_value
from .project import Project
from .replicas import seed_sd

CURVE = "learning-curve"
CURVE_STEP = "curve"
CURVE_FRACTIONS = (0.25, 0.5)
CURVE_INFIX = "-n"
MIN_GAIN = 0.002


def curve_name(exp: str, fraction: float) -> str:
    return f"{exp}{CURVE_INFIX}{round(fraction * 100)}"


def is_curve_run(exp: str) -> bool:
    head, sep, tail = exp.rpartition(CURVE_INFIX)
    return bool(sep and head and tail.isdigit() and round(int(tail) / 100, 2) in CURVE_FRACTIONS)


def _point(p: Project, exp: str, fraction: float, log: Callable[[str], None]) -> tuple[int, float] | None:
    from .pipeline import build

    run = curve_name(exp, fraction)
    if not p.meta(run):
        spec = p.exp_spec(exp).model_copy(deep=True)
        spec.train.data_fraction = fraction
        why = f"{exp} trained on {fraction:.0%} of its training rows, to see how the score grows with data"
        p.save_experiment(
            run, spec=spec, proposal={"parent": exp, "applied": [{"id": CURVE_STEP, "why": why, "ops": []}]}
        )
        build(p.name, run, log, promote=False)
    score = selection_value(p.board().get(run, {}).get(p.spec().primary_suite()))
    return (p.meta(run)["n_train"], score) if score is not None else None


def slope(points: list[tuple[int, float]]) -> float:
    xs = [math.log(n) for n, _ in points]
    mx, my = sum(xs) / len(xs), sum(s for _, s in points) / len(points)
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (s - my) for x, (_, s) in zip(xs, points, strict=True)) / var if var else 0.0


def learning_curve(name: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    p = Project(name)
    exp, prim = p.promoted, p.spec().primary_suite()
    if not exp:
        raise RuntimeError("nothing promoted yet")
    full = selection_value(p.board().get(exp, {}).get(prim))
    points = [pt for f in CURVE_FRACTIONS if (pt := _point(p, exp, f, log))] + [(p.meta(exp)["n_train"], full)]
    gain = slope(points) * math.log(2)
    noise = seed_sd(p, [prim])
    report = {
        "experiment": exp,
        "points": [[n, round(s, 4)] for n, s in points],
        "gain_at_2x": round(gain, 4),
        "seed_sd": noise,
        "data_limited": gain > max(noise or 0, MIN_GAIN),
    }
    p.save_report(CURVE, report)
    verdict = "more data should help" if report["data_limited"] else "more of the same data is unlikely to help"
    log(
        f"[curve] {exp}: calib {', '.join(f'{s:.4f} at {n}' for n, s in points)}; +{gain:.4f} expected at 2x: {verdict}"
    )
    return report
