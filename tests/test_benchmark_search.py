import datasets

from byosm import benchmark_search as bs
from byosm.spec import HFSource, Label, TaskSpec
from byosm.spec.sections import DataPlan

HITS = [
    {"id": "org/turns", "downloads": 129, "tags": []},
    {"id": "org/other", "downloads": 50, "tags": []},
]


def _spec():
    labels = [Label(name="complete", definition="done"), Label(name="incomplete", definition="not done")]
    return TaskSpec(name="turn", description="turn end", modality="text", labels=labels, data=DataPlan(synthetic=True))


def _mapped(spec, cfg, log):
    return [s.model_copy(update={"input_column": "text", "label_column": "label"}) for s in spec.data.hf_sources]


def test_a_dataset_without_a_test_split_becomes_training_data_held_out_by_us(monkeypatch):
    monkeypatch.setattr(bs.hf_source, "search", lambda q: HITS)
    monkeypatch.setattr(bs, "_pick", lambda spec, hits, cfg: {"dataset": "org/turns", "why": "turn taking"})
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda ds: ["ten", "swda", "synthetic"])
    monkeypatch.setattr(bs.hf_source, "resolve", _mapped)
    spec, decisions = _spec(), []
    bs.add_benchmark(spec, ["turn taking"], {"llm": {"default": {"provider": "fake"}}}, print, decisions)
    assert [s.split for s in spec.data.hf_sources] == ["ten", "swda", "synthetic"]
    assert all(isinstance(s, HFSource) and s.role == "labeled" for s in spec.data.hf_sources)
    assert spec.primary_suite() == "builtin" and not spec.data.synthetic
    bench, data = decisions
    assert bench["choice"] == "org/turns" and "no test split" in bench["why"]
    other = bench["alternatives"][0]
    assert (
        other["option"] == "org/other" and other["args"][:3] == ["exp", "create", "turn"] and "--run" in other["args"]
    )
    assert "data.synthetic=true" in data["alternatives"][0]["args"]


def test_a_dataset_written_with_the_hub_prefix_is_read_as_its_bare_id():
    assert HFSource(dataset="hf:facebook/multiwoz").dataset == "facebook/multiwoz"


def test_an_id_that_is_not_a_valid_hub_id_does_not_exist():
    from byosm.data.hf_source.hub import exists

    assert not exists("hf:Microsoft/DialogAct")
