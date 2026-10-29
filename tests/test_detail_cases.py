from conftest import save_board

from byosm.app.catalog import detail
from byosm.project import Project
from byosm.spec import Label, TaskSpec

EXACT = {"accuracy": 0.32, "ci": [0.29, 0.35], "n": 856}


def _result(requests):
    metrics = {"ci95": {"lo": 0.4, "hi": 0.6}} | ({"requests": requests} if requests else {})
    return {"kind": "experiment", "objective": 0.5, "calib_objective": 0.5, "metrics": metrics}


def test_the_detail_carries_exact_accuracy_per_kind_of_request_and_none_without_categories(ws):
    p = Project("pairs").ensure()
    labels = [Label(name="fits", definition=""), Label(name="no_fit", definition="")]
    p.save_spec(TaskSpec(name="pairs", description="", modality="text", labels=labels))
    p.save_experiment("v1", meta={})
    p.save_experiment("v2", meta={})
    requests = {"all": {"exact": EXACT}, "by_category": {"request_for_info": {"exact": EXACT, "no_tool": EXACT}}}
    save_board(p, {"v1": {"builtin": _result(requests)}, "v2": {"builtin": _result(None)}})
    board = detail(ws, "pairs")["board"]
    assert board["v1"]["builtin"]["cases"] == {"request_for_info": EXACT}
    assert board["v2"]["builtin"]["cases"] is None
