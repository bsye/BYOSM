import pytest
from conftest import FakeLLM
from typer.testing import CliRunner

from byosm import llm
from byosm.app.catalog import detail
from byosm.cli import app
from byosm.data.store import Dataset, normalize, read, record
from byosm.evals import board, testset
from byosm.evals.testset import write_testset, written_logs, written_path
from byosm.project import Project
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
    p.save_spec(spec)
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
    saved = p.spec()
    assert [(e.name, e.primary) for e in saved.evals] == [("builtin", True), ("llm-written", False)]
    assert TaskSpec(**{**saved.model_dump(), "evals": saved.evals[1:]}).primary_suite() == "builtin"
    with pytest.raises(ValueError):
        EvalSuite(name="x", kind="generated", primary=True)
    assert saved.evals[1].written_by == "fake"


def test_the_writer_model_shows_in_the_specialist_detail(qa, ws):
    p, spec = qa
    write_testset(p, spec, PER_LABEL, "llm-written")
    suites = {s["name"]: s for s in detail(ws, "qa")["suites"]}
    assert suites["llm-written"] == {
        "name": "llm-written",
        "kind": "generated",
        "primary": False,
        "written_by": "fake",
        "judged_by": "fake",
    }


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


def test_every_written_input_is_logged_as_it_arrives_and_kept_after_the_suite_is_frozen(qa, monkeypatch):
    p, spec = qa
    sizes = []
    real = testset.append
    monkeypatch.setattr(testset, "append", lambda path, rows: (sizes.append(len(rows)), real(path, rows)))
    write_testset(p, spec, PER_LABEL, "llm-written")
    logged = written_logs(p)["llm-written"]
    assert len(sizes) > 1 and sum(sizes) == len(logged)
    frozen = read(p.evals_dir / "llm-written" / "calib.jsonl") + read(p.evals_dir / "llm-written" / "test.jsonl")
    assert {r["input"] for r in frozen} <= {r["input"] for r in logged}


def test_a_stopped_run_resumes_from_the_inputs_it_already_wrote(qa, monkeypatch):
    p, spec = qa
    done = [record(f"what about stop {i}?", "question", "llm:fake") for i in range(PER_LABEL)]
    testset.append(written_path(p, "llm-written"), done)
    asked = []
    real = testset.generate
    monkeypatch.setattr(testset, "generate", lambda *a, **k: (asked.append(k["labels"]), real(*a, **k))[1])
    monkeypatch.setattr(testset, "judge", lambda spec, rows, cfg, log: rows)
    write_testset(p, spec, PER_LABEL, "llm-written")
    assert asked[0] == ["statement"]


def test_a_spec_edit_made_while_the_inputs_are_written_survives(qa, monkeypatch):
    p, spec = qa
    real = testset.generate

    def edit_then_generate(*a, **k):
        edited = p.spec()
        edited.baselines = [*edited.baselines, "jev"]
        p.save_spec(edited)
        return real(*a, **k)

    monkeypatch.setattr(testset, "generate", edit_then_generate)
    write_testset(p, spec, PER_LABEL, "llm-written")
    saved = p.spec()
    assert "jev" in saved.baselines and "llm-written" in [e.name for e in saved.evals]
