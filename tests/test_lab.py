import random

import pytest

from byosm.data.store import record, write
from byosm.lab.author import unfenced
from byosm.lab.drafts import APPROVED, PENDING, REJECTED, decide, draft_id, drafts, save_draft
from byosm.lab.plugin import load
from byosm.lab.static import problems
from byosm.project import Project
from byosm.spec import Label, TaskSpec, TrainCfg

FOCAL = """import torch


def loss(logits, targets, weight, step):
    logp = torch.log_softmax(logits, -1).gather(1, targets[:, None])[:, 0]
    return (-(1 - logp.exp()) ** 2 * logp).mean()


def rows(train, rng):
    return [r for i, r in enumerate(train) if i % 4]
"""
LABELS = [Label(name="statement", definition=""), Label(name="question", definition="")]


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef loss(a, b, c, d):\n    return a",
        "def rows(train, rng):\n    return open('x').readlines()",
        "def rows(train, rng):\n    return eval('train')",
        "def rows(train, rng):\n    return train.__class__",
        "import torch\ndef loss(a, b, c, d):\n    return torch.load('x')",
        "import subprocess",
        "print('runs at import')\ndef rows(train, rng):\n    return train",
        "def helper():\n    return 1",
        "def rows(train, rng)\n    return train",
    ],
)
def test_code_that_could_reach_files_processes_or_run_at_import_fails_its_checks(code):
    assert problems(code)


def test_a_method_with_hooks_and_allowed_imports_passes(tmp_path):
    assert problems(FOCAL) == []


def _project(**train):
    p = Project("tc").ensure()
    p.save_spec(TaskSpec(name="tc", description="questions", modality="text", labels=LABELS, train=TrainCfg(**train)))
    return p


def test_only_approved_code_loads_and_a_method_that_fails_its_checks_cannot_be_approved(ws):
    p = _project()
    d = save_draft(p, "focal", "focus on hard rows", FOCAL, "test")
    assert d["id"] == draft_id(FOCAL) and d["status"] == PENDING and drafts(p)[0]["id"] == d["id"]
    with pytest.raises(PermissionError):
        load(p, d["id"])
    decide(p, d["id"], APPROVED)
    assert set(load(p, d["id"])) == {"loss", "rows"}
    bad = save_draft(p, "bad", "x", "import os\ndef rows(t, r):\n    return t", "test")
    with pytest.raises(ValueError):
        decide(p, bad["id"], APPROVED)
    assert decide(p, bad["id"], REJECTED)["status"] == REJECTED


def test_an_approved_method_shapes_training_through_its_hooks(ws, tiny_bert):
    from byosm.train import fit

    p = _project()
    d = save_draft(p, "focal", "focus on hard rows", FOCAL, "test")
    decide(p, d["id"], APPROVED)
    rng = random.Random(0)
    dd = ws / "data"
    for split, n in (("train", 80), ("dev", 20), ("calib", 20), ("test", 20)):
        rows = []
        for i in range(n):
            q = rng.random() < 0.5
            rows.append(record(f"what about {i}?" if q else f"i like {i}.", "question" if q else "statement", "t"))
        write(dd / f"{split}.jsonl", rows)
    spec = p.spec().model_copy(deep=True)
    spec.backbone = tiny_bert
    spec.train = TrainCfg(epochs=6, pretrain_steps=0, batch_size=16, lab_method=d["id"])
    meta, _ = fit(spec, dd, ws / "run", log=lambda _: None)
    assert meta["n_train"] == 60 and meta["dev"]["objective"] > 0.6


def test_running_a_method_builds_an_experiment_tagged_with_the_lab_family_and_keeps_its_error(ws, monkeypatch):
    from byosm import pipeline
    from byosm.improve.techniques import family
    from byosm.lab.run import run

    p = _project()
    d = save_draft(p, "focal", "focus on hard rows", FOCAL, "test")
    with pytest.raises(PermissionError):
        run("tc", d["id"], lambda _: None)
    decide(p, d["id"], APPROVED)

    def failing(name, exp, log, **_k):
        raise RuntimeError("loss is nan")

    monkeypatch.setattr(pipeline, "build", failing)
    with pytest.raises(RuntimeError):
        run("tc", d["id"], lambda _: None)
    found = drafts(p)[0]
    exp = found["experiment"]
    assert "loss is nan" in found["error"] and p.exp_spec(exp).train.lab_method == d["id"]
    assert family(p.proposal(exp)["applied"][0]["ops"]) == "lab-method"


def test_the_author_saves_what_the_llm_wrote_as_a_pending_draft(ws, monkeypatch):
    from byosm.lab import author

    _project()

    class Writer:
        def json(self, system, prompt, schema):
            assert "def loss(" in system and "Methods written before" in prompt
            return {"name": "focal", "hypothesis": "hard rows", "code": "```python\n" + FOCAL + "```"}

    monkeypatch.setattr("byosm.llm.get_llm", lambda role: Writer())
    monkeypatch.setattr(author, "_prompt", lambda p, instruction, paper, log: "Methods written before: []")
    d = author.write("tc", "", lambda _: None)
    assert d["status"] == PENDING and d["code"] == FOCAL.strip() and not d["problems"]
    assert unfenced("plain") == "plain"


def test_the_app_lists_methods_and_only_a_person_s_decision_approves_one(ws):
    from fastapi.testclient import TestClient

    from byosm.app import create_app

    p = _project()
    d = save_draft(p, "focal", "focus on hard rows", FOCAL, "test")
    bad = save_draft(p, "bad", "x", "import os\ndef rows(t, r):\n    return t", "test")
    auth = {"Authorization": "Bearer t"}
    with TestClient(create_app(ws, "t"), base_url="http://127.0.0.1") as c:
        assert [m["id"] for m in c.get("/api/specialists/tc/lab", headers=auth).json()] == [bad["id"], d["id"]]
        url = f"/api/specialists/tc/lab/{d['id']}"
        assert c.post(url, headers=auth, json={"status": "approved"}).json()["status"] == APPROVED
        assert (
            c.post(f"/api/specialists/tc/lab/{bad['id']}", headers=auth, json={"status": "approved"}).status_code == 422
        )
        assert c.post(url, json={"status": "approved"}).status_code == 401


def test_a_loss_hook_on_a_tagger_gets_one_row_per_token_with_padding_ignored():
    import torch

    from byosm.train.trainer import Trainer

    seen = {}

    def loss(logits, targets, weight, step):
        seen["shape"], seen["targets"] = tuple(logits.shape), targets.tolist()
        return logits.sum() * 0

    t = Trainer.__new__(Trainer)
    t.hooks, t.weight, t.steps = {"loss": loss}, None, 0
    logits = torch.zeros(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    t._task_loss(logits, torch.tensor([[1, 2, 0], [3, 0, 0]]), mask)
    assert seen["shape"] == (6, 4) and seen["targets"] == [1, 2, -100, 3, -100, -100]
