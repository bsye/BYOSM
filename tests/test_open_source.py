import pandas as pd

import byosm.data.hf_source.hub as hub
from byosm.data.hf_source import take
from byosm.spec import HFSource, Label, TaskSpec

TABLE = pd.DataFrame(
    {
        "notes": ["rodents seen in kitchen", "all clear", "no hot water", "spotless", "pests found"],
        "outcome": ["Fail", "Pass", "Fail", "Pass", "Fail"],
    }
)


def test_an_open_data_table_is_read_and_mapped_like_a_hub_dataset(monkeypatch):
    monkeypatch.setattr(hub, "open_table", lambda dataset: TABLE.astype(str))
    spec = TaskSpec(
        name="t",
        description="",
        modality="text",
        labels=[Label(name="fail", definition=""), Label(name="pass", definition="")],
    )
    src = HFSource(
        dataset="opendata:https://data.city.org/inspections.csv",
        input_column="notes",
        label_column="outcome",
        label_map=[["Fail", "fail"], ["Pass", "pass"]],
    )
    rows = take(src, spec)
    assert sorted(rows) == sorted(zip(TABLE["notes"], TABLE["outcome"].str.lower(), strict=True))
    assert hub.closest(src.dataset) == src.dataset and hub.split_sizes(src.dataset) == {"train": 5}


def test_portal_tables_are_offered_with_the_hub_results_and_downloaded_when_picked(monkeypatch):
    import byosm.data.hf_source.mapping as mapping
    import byosm.opendata.catalog as catalog

    found = [
        {
            "title": "Inspections",
            "source": "data.city.org",
            "notes": "results",
            "licence": "OGL",
            "files": [{"format": "PDF", "url": "x.pdf"}, {"format": "CSV", "url": "https://c/i.csv"}],
        }
    ]
    monkeypatch.setattr(catalog, "search", lambda term: found)
    import byosm.opendata.cache as cache

    monkeypatch.setattr(cache, "remote_mb", lambda url: 2.0)
    monkeypatch.setattr(mapping, "search", lambda queries: [])
    seen = {}
    monkeypatch.setattr(mapping, "pick_sources", lambda spec, hits, cfg: seen.setdefault("hits", hits) and [])
    spec = TaskSpec(name="t", description="", modality="text", labels=[Label(name="a", definition="")])
    spec.data.hf_queries = ["restaurant inspections"]
    mapping._pick_sources(spec, None)
    assert [h["id"] for h in seen["hits"]] == ["opendata:https://c/i.csv"]
    monkeypatch.setattr(mapping, "fetched", lambda dataset, log: False)
    assert mapping._map_source(HFSource(dataset="opendata:https://c/i.csv"), spec, None, lambda m: None) is None
