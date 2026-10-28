import pytest
from conftest import save_board
from fastapi.testclient import TestClient

from byosm.app import create_app
from byosm.evals.versus import versus
from byosm.project import Project
from byosm.spec import Label, TaskSpec

ROWS = 200
AUTH = {"Authorization": "Bearer t"}


def _scored(right, wrong_first=False):
    y = [i % 2 for i in range(ROWS)]
    wrong = range(ROWS - right) if wrong_first else range(right, ROWS)
    pred = [1 - c if i in wrong else c for i, c in enumerate(y)]
    units = {split: {"y": y, "pred": pred} for split in ("calib", "test")}
    return {"kind": "experiment", "objective": right / ROWS, "calib_objective": right / ROWS, "units": units}


@pytest.fixture
def pair(ws):
    p = Project("vs").ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name="vs", description="", modality="text", labels=labels, objective={"metric": "accuracy"}))
    for name in ("old", "new", "same"):
        p.save_experiment(name, meta={"params": 1})
    save_board(
        p,
        {
            "old": {"builtin": _scored(140)},
            "new": {"builtin": _scored(180)},
            "same": {"builtin": _scored(141, wrong_first=True)},
        },
    )
    return p


def test_a_clearly_better_model_is_a_sure_lead_and_a_near_tie_is_not(pair):
    better = versus(pair, "old", "new")["suites"][0]
    assert better["suite"] == "builtin" and better["calib"]["delta"] == pytest.approx(0.2)
    assert better["calib"]["p_better"] >= 0.95 and better["test"]["lo"] > 0
    tie = versus(pair, "old", "same")["suites"][0]["calib"]
    assert tie["lo"] < 0 < tie["hi"] and tie["p_better"] < 0.95


def test_the_route_compares_two_models_on_the_board_and_refuses_an_unknown_one(ws, pair):
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        got = c.get("/api/specialists/vs/versus?a=old&b=new", headers=AUTH).json()
        assert got["a"] == "old" and got["b"] == "new" and got["sure"] == 0.95 and len(got["suites"]) == 1
        assert c.get("/api/specialists/vs/versus?a=old&b=nobody", headers=AUTH).status_code == 404
