from conftest import save_board
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.project import Project
from byosm.reads import snapshot
from byosm.spec import Label, TaskSpec

AUTH = {"Authorization": "Bearer t"}


def _project():
    p = Project("t").ensure()
    p.save_spec(TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")]))
    p.save_experiment("v1", meta={})
    save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": 0.5}}})
    return p


def test_a_snapshot_reads_the_board_once_and_outside_one_every_read_is_fresh(ws, monkeypatch):
    p = _project()
    reads = []
    real = Project._board
    monkeypatch.setattr(Project, "_board", lambda self, units: reads.append(units) or real(self, units))
    with snapshot():
        assert p.board() == Project("t").board()
    assert len(reads) == 1
    save_board(p, {"v1": {"builtin": {"kind": "experiment", "objective": 0.7}}})
    assert p.board()["v1"]["builtin"]["objective"] == 0.7 and len(reads) == 2


READS = ("_spec_row", "_promoted", "_read_experiment", "_experiments", "_attempts")


def test_a_snapshot_reads_the_specialist_and_each_experiment_once(ws, monkeypatch):
    p = _project()
    calls = []
    for name in READS:
        real = getattr(Project, name)
        monkeypatch.setattr(Project, name, lambda self, *a, real=real, name=name: calls.append(name) or real(self, *a))
    with snapshot():
        for _ in range(3):
            seen = (p.spec().name, p.promoted, p.meta("v1"), p.experiments(), p.attempts())
            assert seen == ("t", None, {}, ["v1"], ["v1"])
    assert sorted(calls) == sorted(READS)


def test_outside_a_snapshot_every_read_sees_the_latest_write(ws):
    p = _project()
    assert p.promoted is None
    p.promote("v1", "test")
    p.save_experiment("v2", meta={"x": 1})
    assert p.promoted == "v1" and p.experiments() == ["v1", "v2"] and p.meta("v2") == {"x": 1}


def test_changing_a_remembered_read_does_not_change_the_next_one(ws):
    p = _project()
    with snapshot():
        p.experiment("v1")["meta"] = {"changed": True}
        p.experiments().append("v9")
        assert p.experiment("v1")["meta"] == {} and p.experiments() == ["v1"]


def test_only_read_requests_share_one_board_read(ws, monkeypatch):
    _project()
    reads = []
    real = Project._board
    monkeypatch.setattr(Project, "_board", lambda self, units: reads.append(units) or real(self, units))
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert c.get("/api/specialists/t", headers=AUTH).status_code == 200
        assert sorted(reads) == [False, True]
