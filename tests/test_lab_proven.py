import subprocess
from pathlib import Path

from byosm.lab.drafts import APPROVED, decide, save_draft
from byosm.lab.plugin import load
from byosm.lab.proven import CANDIDATE, PROVEN_STATUS, REPO, proposals, record_win, registry
from byosm.project import Project
from byosm.spec import Label, TaskSpec

CODE = "def rows(train, rng):\n    return list(reversed(train))\n"


def _project(name, modality="text"):
    p = Project(name).ensure()
    labels = [Label(name=n, definition="") for n in "ab"]
    p.save_spec(TaskSpec(name=name, description="q", modality=modality, labels=labels))
    return p


def test_a_method_that_wins_on_two_specialists_is_proven_with_a_patch_that_applies(ws):
    a = _project("a")
    d = save_draft(a, "Reverse curriculum", "harder rows last", CODE, "test", None)
    assert record_win(ws, "a", "text", d)["status"] == CANDIDATE
    entry = record_win(ws, "b", "text", d)
    assert entry["status"] == PROVEN_STATUS and entry["won_on"] == ["a", "b"]
    patch = Path(entry["patch"]).read_text()
    assert "--- /dev/null\n+++ b/byosm/lab/methods/reverse_curriculum.py" in patch
    assert (
        "train.lab_method=builtin:reverse_curriculum" in patch
        and "tests/test_lab_method_reverse_curriculum.py" in patch
    )
    check = subprocess.run(["git", "apply", "--check", entry["patch"]], cwd=REPO, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_a_method_that_won_once_is_offered_to_other_specialists_of_its_kind_and_loads_there(ws):
    a, b, _ = _project("a"), _project("b"), _project("c", "audio")
    d = save_draft(a, "Reverse curriculum", "harder rows last", CODE, "test", None)
    decide(a, d["id"], APPROVED)
    record_win(ws, "a", "text", d)
    assert [pr["ops"][0]["value"] for pr in proposals(b, b.spec())] == [d["id"]]
    assert proposals(a, a.spec()) == [] and proposals(Project("c"), Project("c").spec()) == []
    assert set(load(b, d["id"])) == {"rows"}


def test_a_promotion_won_with_a_lab_method_records_the_win(ws):
    from conftest import save_board

    from byosm.promotion import maybe_promote

    p = _project("a")
    p.save_spec(p.spec().model_copy(update={"finalist_seeds": 1}))
    d = save_draft(p, "Reverse curriculum", "harder rows last", CODE, "test", None)
    decide(p, d["id"], APPROVED)
    p.save_experiment("v1", meta={"calib_held": True})
    p.promote("v1")
    spec = p.spec().model_copy(deep=True)
    spec.train.lab_method = d["id"]
    op = {"op": "set", "path": "train.lab_method", "value": d["id"]}
    p.save_experiment("v2", spec=spec, meta={"calib_held": True}, proposal={"applied": [{"id": "lab", "ops": [op]}]})
    result = {"kind": "experiment", "n_test": 500, "n_calib": 200}
    save_board(p, {"v1": {"builtin": result | {"objective": 0.8, "calib_objective": 0.8}}})
    save_board(p, {"v2": {"builtin": result | {"objective": 0.9, "calib_objective": 0.9}}})
    maybe_promote(p, "v2", lambda _: None)
    assert registry(ws)[d["id"]]["won_on"] == ["a"]
