import byosm.health as health_module
import byosm.next_steps as steps_module
from byosm.auto.ready import until_ready


def _patch(monkeypatch, steps, promoted):
    monkeypatch.setattr("byosm.project.Project.spec", lambda self: None)
    monkeypatch.setattr("byosm.project.Project.exists", lambda self: True)
    monkeypatch.setattr("byosm.project.Project.promoted", property(lambda self: promoted["v"]))
    monkeypatch.setattr(health_module, "health", lambda p, spec: {"score": 40, "ready": False, "blockers": ["X"]})
    monkeypatch.setattr(steps_module, "suggestions", lambda name: steps)


def test_a_step_runs_again_once_the_promoted_model_changes(monkeypatch):
    promoted = {"v": "v1"}
    _patch(monkeypatch, [{"id": "export", "title": "Export", "args": ["export", "x"]}], promoted)
    ran = []

    def run(args):
        ran.append(args)
        if len(ran) == 1:
            promoted["v"] = "v3"

    out = until_ready("x", run, log=lambda _: None)
    assert ran == [["export", "x"], ["export", "x"]]
    assert out["stop"] == "no untried step left"


def test_a_failing_step_is_logged_and_the_loop_moves_on(monkeypatch):
    steps = [
        {"id": "data", "title": "Find data", "args": ["datasets", "x"], "gain": 9},
        {"id": "export", "title": "Export", "args": ["export", "x"], "gain": 5},
        {"id": "eval", "title": "Eval", "args": ["eval", "x"], "gain": 1},
    ]
    _patch(monkeypatch, steps, {"v": "v1"})
    ran, logs = [], []

    def run(args):
        ran.append(args)
        if args[0] == "datasets":
            raise RuntimeError("LLM down")
        return 2 if args[0] == "export" else 0

    out = until_ready("x", run, log=logs.append)
    assert ran == [["datasets", "x"], ["export", "x"], ["eval", "x"]]
    assert "[ready] Find data failed: RuntimeError: LLM down" in logs
    assert "[ready] Export failed: exit code 2" in logs
    assert out["stop"] == "no untried step left"
