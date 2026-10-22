import byosm.health as health_module
import byosm.next_steps as steps_module
from byosm.auto.ready import until_ready


def test_steps_run_until_the_health_says_ready(monkeypatch):
    state = {"score": 50}
    monkeypatch.setattr("byosm.project.Project.spec", lambda self: None)
    monkeypatch.setattr(
        health_module,
        "health",
        lambda p, spec: {"score": state["score"], "ready": state["score"] >= 80, "blockers": []},
    )
    steps = [
        {"id": "data", "title": "Find data", "args": ["data", "x"]},
        {"id": "exp", "title": "Experiments", "args": ["improve", "x"]},
    ]
    monkeypatch.setattr(steps_module, "suggestions", lambda name: steps)
    ran = []

    def run(args):
        ran.append(args)
        state["score"] += 20

    out = until_ready("x", run, log=lambda _: None)
    assert out == {"ready": True, "score": 90, "rounds": 2}
    assert ran == [["data", "x"], ["improve", "x"]]


def test_it_stops_when_no_untried_step_is_left(monkeypatch):
    monkeypatch.setattr("byosm.project.Project.spec", lambda self: None)
    monkeypatch.setattr(
        health_module, "health", lambda p, spec: {"score": 40, "ready": False, "blockers": ["Enough data"]}
    )
    monkeypatch.setattr(
        steps_module, "suggestions", lambda name: [{"id": "data", "title": "Find data", "args": ["data", "x"]}]
    )
    out = until_ready("x", lambda args: None, log=lambda _: None)
    assert out["stop"] == "no untried step left" and out["rounds"] == 1
