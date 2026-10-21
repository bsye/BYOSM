import json

import pytest
from conftest import FakeLLM
from typer.testing import CliRunner

from byosm import llm
from byosm.cli import app
from byosm.config import Project
from byosm.data.store import Dataset, normalize, read, record
from byosm.evals import board
from byosm.evals.testset import write_testset
from byosm.spec import Label, TaskSpec
from byosm.spec.sections import Axis, DataPlan, EvalSuite

KNOWN = ["what about pasta coffee?", "i like rain museum.", "what about bug merge flight?", "i like garden train."]
PER_LABEL = 6


class RepeatingLLM(FakeLLM):
    emitted_known = 0

    def example(self):
        if self.rng.random() < 0.3:
            RepeatingLLM.emitted_known += 1
            text = self.rng.choice(KNOWN)
            return {"text": text, "label": "question" if text.endswith("?") else "statement"}
        return super().example()


@pytest.fixture
def qa(ws):
    llm.register_provider("fake", RepeatingLLM)
    p = Project("qa").ensure()
    spec = TaskSpec(
        name="qa",
        description="is it a question",
        modality="text",
        labels=[Label(name="statement", definition="not a question"), Label(name="question", definition="asks")],
        diversity_axes=[Axis(name="topic", values=["food", "code"])],
        data=DataPlan(per_call=5, judge=True),
        baselines=["llm"],
    )
    spec.save(p.spec_path)
    Dataset(p.data_dir_for(spec)).save(
        "train", [record(t, "question" if "?" in t else "statement", "x") for t in KNOWN]
    )
    return p, spec


def test_the_llm_writes_n_new_examples_per_label_into_a_frozen_suite(qa):
    p, spec = qa
    es = write_testset(p, spec, PER_LABEL, "llm-written")
    rows = read(p.evals_dir / "llm-written" / "calib.jsonl") + read(p.evals_dir / "llm-written" / "test.jsonl")
    assert sorted(r["label"] for r in rows) == sorted(["question", "statement"] * PER_LABEL)
    assert RepeatingLLM.emitted_known > 0
    assert not {normalize(r["input"]) for r in rows} & {normalize(t) for t in KNOWN}
    assert read(p.evals_dir / "llm-written" / "calib.jsonl")
    assert (es.kind, es.written_by, es.judged_by) == ("generated", "fake", "fake")


def test_the_suite_is_registered_but_never_primary(qa):
    p, spec = qa
    write_testset(p, spec, PER_LABEL, "llm-written")
    saved = TaskSpec.load(p.spec_path)
    assert [(e.name, e.primary) for e in saved.evals] == [("builtin", True), ("llm-written", False)]
    assert TaskSpec(**{**saved.model_dump(), "evals": saved.evals[1:]}).primary_suite() == "builtin"
    with pytest.raises(ValueError):
        EvalSuite(name="x", kind="generated", primary=True)
    suite_json = json.loads((p.evals_dir / "llm-written" / "suite.json").read_text())
    assert suite_json["written_by"] == "fake"


def test_frame_tasks_are_refused(qa):
    p, spec = qa
    with pytest.raises(ValueError, match="classification only"):
        write_testset(p, spec.model_copy(update={"task_type": "frame"}), PER_LABEL, "llm-written")


def test_the_command_scores_the_baselines_on_the_new_suite_only(qa, monkeypatch):
    scored = []

    def fake_score(pred, spec, calib, test, root, space=None):
        scored.append((pred, len(test)))
        return {"kind": "baseline", "objective": 0.5, "calib_objective": 0.5, "metrics": {}}

    monkeypatch.setattr(board, "score", fake_score)
    monkeypatch.setattr(board, "get_baseline", lambda b, spec: b)
    out = CliRunner().invoke(app, ["testset", "qa", "--per-label", str(PER_LABEL)])
    assert out.exit_code == 0, out.output
    assert scored == [("llm", 2 * PER_LABEL - 4)]
    assert set(Project("qa").board()["llm"]) == {"llm-written"}
