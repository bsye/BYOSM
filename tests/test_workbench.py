import pytest
from conftest import FakeLLM

from byosm import llm
from byosm.data.store import Dataset, read, record
from byosm.evals import workbench
from byosm.evals.testset import written_path
from byosm.project import Project
from byosm.spec import Label, TaskSpec
from byosm.spec.sections import DataPlan

CALLS = []


class CountingLLM(FakeLLM):
    def json(self, system, prompt, schema, images=None):
        CALLS.append(prompt[:20])
        return super().json(system, prompt, schema)


@pytest.fixture
def qa(ws):
    llm.register_provider("fake", CountingLLM)
    p = Project("qa").ensure()
    spec = TaskSpec(
        name="qa",
        description="is it a question",
        modality="text",
        labels=[Label(name="statement", definition="not a question"), Label(name="question", definition="asks")],
        data=DataPlan(per_call=5),
        baselines=["llm"],
    )
    p.save_spec(spec)
    rows = [record(f"what about thing {i}?", "question", "x") for i in range(30)]
    Dataset(p.data_dir_for(spec)).save("test", rows)
    return p, spec


def test_writing_twice_adds_up_and_nothing_is_scored(qa):
    p, spec = qa
    first = workbench.write_more(p, spec, "bench", 3, print)
    second = workbench.write_more(p, spec, "bench", 3, print)
    assert len(read(written_path(p, "bench"))) == first + second > 0
    assert workbench.results(p, "bench") == {}


def test_a_run_scores_each_row_once_per_model_and_keeps_the_answers(qa):
    p, spec = qa
    assert workbench.sample_benchmark(p, spec, "benchmark", 10) == 10
    out = workbench.run(p, spec, "benchmark", ["llm"], print)
    rows = read(written_path(p, "benchmark"))
    assert set(out["llm"]) == {r["id"] for r in rows}
    assert all(v["label"] in spec.label_names for v in out["llm"].values())
    CALLS.clear()
    workbench.run(p, spec, "benchmark", ["llm"], print)
    assert CALLS == [] and workbench.results(p, "benchmark") == out


def test_the_model_choice_is_saved_per_specialist(qa):
    p, spec = qa
    assert workbench.chosen_models(p, spec) == ["ours", "llm"]
    workbench.choose_models(p, ["llm"])
    assert workbench.chosen_models(p, spec) == ["llm"]
