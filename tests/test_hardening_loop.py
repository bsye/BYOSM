from __future__ import annotations

import time
from types import SimpleNamespace

from byosm import improve, pipeline
from byosm.auto import loop
from byosm.auto.state import Session
from byosm.format import MISSING

STANDING = {"won": False, "exp": "e0", "ours": None, "suite": "s", "best_baseline": None, "target": None}


class FakeLog:
    def __init__(self):
        self.decisions = []

    def __call__(self, step, choice, why, **extra):
        self.decisions.append({"step": step, "choice": choice, "why": why, **extra})


def _session() -> Session:
    p = SimpleNamespace(experiments=lambda: [], attempts=lambda: [], board=lambda: {})
    return Session("t", p, FakeLog(), None, 0.0, time.time() + 3600, lambda _m: None)


def _proposal(i: int) -> dict:
    return {"id": f"llm-{i}", "ops": [{"n": i}], "cost": {"minutes": 1.0}, "why": "w"}


def test_failed_builds_stop_on_plateau(monkeypatch):
    counter = iter(range(1000))
    builds = []
    monkeypatch.setattr(loop, "standing", lambda *_a: STANDING)
    monkeypatch.setattr(improve, "propose", lambda *_a: (None, [_proposal(next(counter))]))
    monkeypatch.setattr(improve, "apply", lambda name, ids: f"exp-{ids[0]}")

    def failing_build(*_a, **_k):
        builds.append(1)
        raise RuntimeError("download failed")

    monkeypatch.setattr(pipeline, "build", failing_build)
    reason = loop.improve_until_stop(_session(), max_experiments=100, use_llm=True)
    assert reason.startswith("plateau")
    assert len(builds) == loop.PLATEAU_ROUNDS


def test_log_result_without_scores():
    s = _session()
    loop._log_result(s, STANDING, "e1", _proposal(1), [_proposal(1)], False, None, None)
    assert f"calib {MISSING} vs {MISSING}" in s.log.decisions[-1]["why"]


def test_time_spent_waiting_for_the_gpu_does_not_count_against_the_budget(monkeypatch):
    import byosm.train.gpu as gpu

    s = _session()
    monkeypatch.setattr(gpu, "_WAITED", [1800.0, 600.0])
    assert s.deadline == s.ends + 2400.0
