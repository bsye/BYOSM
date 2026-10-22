from byosm import dataset_search as ds
from byosm.spec import HFSource, Label, TaskSpec
from byosm.spec.sections import DataPlan

HITS = [{"id": f"org/set{i}", "downloads": 100 - i, "tags": []} for i in range(5)]


def _spec():
    labels = [Label(name="complete", definition="done"), Label(name="incomplete", definition="not done")]
    plan = DataPlan(synthetic=True, hf_queries=["turn taking"])
    return TaskSpec(name="turn", description="turn end", modality="text", labels=labels, data=plan)


def test_the_extra_search_adds_only_picks_that_map_and_offers_the_rest_as_paths(monkeypatch):
    monkeypatch.setattr(ds.hf_source, "search", lambda q: HITS)
    picks = [
        (HFSource(dataset="org/set0", role="labeled"), "turn labels"),
        (HFSource(dataset="org/set1", role="labeled"), "close"),
    ]
    monkeypatch.setattr(ds, "pick_sources", lambda spec, hits, cfg: picks)

    def mapped(spec, cfg, log):
        return [s.model_copy(update={"input_column": "t"}) for s in spec.data.hf_sources if s.dataset == "org/set0"]

    monkeypatch.setattr(ds.hf_source, "resolve", mapped)
    spec, decisions = _spec(), []
    ds.add_datasets(spec, {"llm": {"default": {"provider": "fake"}}}, print, decisions)
    assert [s.dataset for s in spec.data.hf_sources] == ["org/set0"] and not spec.data.synthetic
    found, data = decisions
    assert found["choice"] == "org/set0" and "turn labels" in found["why"]
    options = {a["option"]: a["why_not"] for a in found["alternatives"]}
    assert list(options) == ["org/set1"] and "do not map" in options["org/set1"]
    assert data["choice"] == "real rows only"


def test_a_failing_search_is_recorded_not_raised(monkeypatch):
    def broken(q):
        raise ConnectionError("offline")

    monkeypatch.setattr(ds.hf_source, "search", broken)
    decisions = []
    ds.add_datasets(_spec(), None, print, decisions)
    assert decisions[0]["choice"] is None and "offline" in decisions[0]["why"]


def test_datasets_found_on_the_web_come_first_and_only_if_they_exist(monkeypatch):
    from byosm.data.hf_source import research

    monkeypatch.setattr(research, "research_config", lambda cfg: {"provider": "fake"})
    monkeypatch.setattr(
        research,
        "get_llm",
        lambda role, cfg: type(
            "L", (), {"json": lambda *a: {"datasets": [{"id": "org/real", "why": "turns"}, {"id": "org/made-up"}]}}
        )(),
    )
    monkeypatch.setattr(research, "exists", lambda d: d == "org/real")
    monkeypatch.setattr(research, "role_model", lambda role, cfg: "fake")
    found = research.research(_spec(), {"llm": {"default": {"provider": "fake"}}}, print)
    assert [d["id"] for d in found] == ["org/real"] and found[0]["why"] == "turns"
