import json

import pytest
from conftest import numbered_items
from option_pool import pool

from byosm.data import build
from byosm.data import judge as judge_module
from byosm.data.store import CALIB, Dataset
from byosm.data.table import read_table
from byosm.options.create import pair_plan, spec_for
from byosm.options.nearmiss import MINERS, ToolIndex
from byosm.options.pairs import Pair, write
from byosm.options.relabel import relabel
from byosm.options.situations import FITS, NO_FIT, Situation

WEATHER = {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"city": {}}}
FORECAST = {"name": "get_forecast", "description": "Weather forecast for a city over days.", "parameters": {"city": {}}}
STOCK = {"name": "get_stock", "description": "A share price.", "parameters": {"ticker": {}}}
HUMIDITY = {"name": "get_humidity", "description": "Air humidity in a place.", "parameters": {"place": {}}}
TRAIN = [
    ("0", Situation("what is the weather in Paris", [WEATHER, STOCK], {"get_weather"})),
    ("1", Situation("forecast for Rome next days", [FORECAST], {"get_forecast"})),
    ("2", Situation("humidity in Oslo", [WEATHER, HUMIDITY], {"get_humidity"})),
]
CFG = {"llm": {"default": {"provider": "fake"}}, "concurrency": 1}
SOURCE = "lockon/xlam-function-calling-60k"


def quiet(message):
    return None


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))


def _mined(kind):
    return [
        (p.group, p.option["name"], [o["name"] for o in p.offered]) for p in MINERS[kind](ToolIndex(TRAIN), CFG, quiet)
    ]


def test_each_kind_of_near_miss_looks_like_a_positive_of_its_request(ws, cache):
    assert _mined("nearest") == [
        ("0", "get_forecast", ["get_weather", "get_stock", "get_forecast"]),
        ("1", "get_weather", ["get_forecast", "get_weather"]),
    ]
    assert _mined("same_category") == [("0", "get_humidity", ["get_weather", "get_stock", "get_humidity"])]
    assert _mined("removed") == [("0", "get_stock", ["get_stock"]), ("2", "get_weather", ["get_weather"])]
    missing = MINERS["missing_parameter"](ToolIndex(TRAIN), CFG, quiet)
    assert [(p.option["name"], p.label) for p in missing] == [
        ("get_weather", NO_FIT),
        ("get_forecast", NO_FIT),
        ("get_humidity", NO_FIT),
    ]
    assert all("[city" in p.request or "[place" in p.request for p in missing)


class Judge:
    def __init__(self):
        self.calls = 0

    def json(self, system, prompt, schema):
        self.calls += 1
        items = numbered_items(prompt)
        verdict = [FITS if "get_forecast" in s else "__drop__" if "get_humidity" in s else NO_FIT for s in items]
        return {"labels": [{"n": i, "label": v} for i, v in enumerate(verdict, 1)]}


def test_every_near_miss_takes_the_judges_label_and_a_dropped_one_never_reaches_training(ws, cache, monkeypatch):
    judge = Judge()
    monkeypatch.setattr(judge_module, "get_llm", lambda role, cfg=None: judge)
    spec = spec_for("tc", ws / "t.csv", SOURCE, pair_plan(held_out=False))
    mined = [
        Pair("0", "weather in Paris", o, [WEATHER, o], NO_FIT, kind="nearest") for o in (FORECAST, STOCK, HUMIDITY)
    ]
    kept = relabel(spec, mined, CFG, quiet)
    assert [(p.option["name"], p.label, p.source) for p in kept] == [
        ("get_forecast", FITS, "near-miss:fake"),
        ("get_stock", NO_FIT, "near-miss:fake"),
    ]
    calls = judge.calls
    assert relabel(spec, mined, CFG, quiet) == kept and judge.calls == calls


def test_near_misses_enter_train_only_and_use_only_tools_training_already_offers(ws, cache, monkeypatch):
    monkeypatch.setattr(judge_module, "get_llm", lambda role, cfg=None: Judge())
    plan = pair_plan(held_out=True)
    write(ws / "pairs.csv", pool(200), plan)
    spec = spec_for("tc", ws / "pairs.csv", SOURCE, plan)
    spec.data.pairs.near_misses = list(MINERS)
    build(spec, ws / "d", CFG, quiet)
    ds = Dataset(ws / "d")
    near = [r for r in ds.load("train") if r["source"].startswith("near-miss:")]
    assert {r["meta"]["kind"] for r in near} == set(MINERS)
    held = {json.loads(r["option"])["name"] for r in read_table(ws / "pairs.csv") if r["split"] in (CALIB, "test")}
    assert held and not any(name in r["input"] for name in held for r in near)
    assert not any(r["source"].startswith("near-miss:") for s in ("dev", CALIB, "test") for r in ds.load(s))
