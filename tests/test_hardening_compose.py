import pytest

from byosm import llm
from byosm.tabular import compose

TABLES = [
    {
        "dataset": {"title": "register"},
        "file": {"url": "https://example.org/register.csv"},
        "columns": [
            {"name": n, "missing": 0.0, "examples": ["a", "b"]} for n in ("opened", "closed", "sector", "name")
        ],
    }
]


class FakeLLM:
    def __init__(self, out):
        self.out = out

    def json(self, system, prompt, schema):
        return self.out


def _design(monkeypatch, out):
    monkeypatch.setattr(llm, "get_llm", lambda role, cfg: FakeLLM(out))
    return compose.design("will shops close", {"horizon_years": 2}, TABLES, None)


def test_design_rejects_same_start_and_end_column(monkeypatch):
    out = {"table": 0, "start": "opened", "event": "opened", "features": ["sector"], "shown": ["name"]}
    with pytest.raises(ValueError, match="same column"):
        _design(monkeypatch, out)


def test_design_accepts_distinct_start_and_end_columns(monkeypatch):
    out = {"table": 0, "start": "opened", "event": "closed", "features": ["sector", "closed"], "shown": ["name"]}
    spec = _design(monkeypatch, out)
    assert (spec["start"], spec["event"], spec["features"]) == ("opened", "closed", ["sector"])
