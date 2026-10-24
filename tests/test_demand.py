from types import SimpleNamespace

import byosm.hub.demand as demand


class FakeHub:
    def list_models(self, pipeline_tag, filter, sort, limit):
        downloads = 2_000_000 if filter == "it" else 500_000
        return [SimpleNamespace(id=f"m/{pipeline_tag}-{filter}", downloads=downloads)]

    def list_datasets(self, task_categories, language, sort, limit):
        return [
            SimpleNamespace(id=f"d/{task_categories}-{language}", downloads=10_000 if language == "it" else 400_000)
        ]


def test_gaps_rank_tasks_where_models_are_downloaded_far_more_than_their_datasets(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", FakeHub)
    rows = demand.scan(lambda _: None)
    assert rows[0]["language"] == "it" and rows[0]["prompt"].endswith("in Italian")
    assert all(r["demand"] >= demand.MIN_DEMAND for r in rows)
    assert rows == sorted(rows, key=lambda r: -r["score"])
