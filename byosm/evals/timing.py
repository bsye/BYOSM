from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..machine import load_per_core
from ..project import Project
from .board import experiment_predictor
from .scoring import BUSY_LOAD, add_speed, busy
from .suites import materialize

QUIET_WAIT_S = 3600
QUIET_POLL_S = 30

Log = Callable[[str], None]


def wait_quiet(log: Log, wait_s: float = QUIET_WAIT_S, poll_s: float = QUIET_POLL_S) -> None:
    end = time.time() + wait_s
    while (load := load_per_core()) > BUSY_LOAD:
        if time.time() >= end:
            raise RuntimeError(
                f"the machine stayed above {BUSY_LOAD:g} load per core for {wait_s / 60:.0f} min: not timed"
            )
        log(f"  waiting for a quiet machine: load {load:g} per core")
        time.sleep(poll_s)


def busy_timings(p: Project) -> list[str]:
    prim = p.spec().primary_suite()
    return [e for e, by in p.board().items() if e in p.experiments() and busy(by.get(prim) or {})]


def retime(name: str, exps: list[str] | None = None, log: Log = print) -> dict[str, Any]:
    p = Project(name)
    spec = p.spec()
    suite = next(e for e in spec.evals if e.name == spec.primary_suite())
    chosen = exps or busy_timings(p)
    if not chosen:
        log(f"{name}: no timing to take again")
        return {}
    _, test, root = materialize(p, spec, suite, log)
    out = {}
    for exp in chosen:
        found = p.result(exp, suite.name)
        if found is None:
            raise KeyError(f"{exp} has no {suite.name} result to time")
        wait_quiet(log)
        speed: dict[str, Any] = {}
        add_speed(speed, experiment_predictor(p, spec, exp), spec, test, root)
        p.save_result(exp, suite.name, {**found, **speed})
        log(f"  {exp}: {speed.get('latency_ms')} ms per input at {speed.get('latency_load')} load per core")
        out[exp] = speed
    return out
