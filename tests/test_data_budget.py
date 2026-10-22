import time

import pytest
from conftest import FakeLLM

from byosm import data as data_stage
from byosm import llm, pipeline
from byosm.auto import autopilot
from byosm.auto.state import DATA_BUDGET_SHARE, Session
from byosm.data import budget
from byosm.spec import Label, TaskSpec
from byosm.spec.sections import Axis, DataPlan

HOUR = 3600
SLOW = 240
FAST = 1
PLAN = {"train": 600, "dev": 100, "test": 150}
CFG = {"llm": {"default": {"provider": "clocked", "max_parallel": 1}}, "concurrency": 2}


class Clock:
    now = 0.0

    def monotonic(self):
        return Clock.now


class ClockedLLM(FakeLLM):
    seconds = FAST
    calls = 0

    def json(self, system, prompt, schema):
        if "examples" in schema["properties"]:
            Clock.now += ClockedLLM.seconds
            ClockedLLM.calls += 1
        return super().json(system, prompt, schema)


@pytest.fixture
def run(ws, monkeypatch):
    llm.register_provider("clocked", ClockedLLM)
    monkeypatch.setattr(budget, "time", Clock())
    spec = TaskSpec(
        name="qa",
        description="is it a question",
        modality="text",
        labels=[Label(name="statement", definition="not a question"), Label(name="question", definition="asks")],
        diversity_axes=[Axis(name="topic", values=["food", "code"])],
        data=DataPlan(train_size=PLAN["train"], dev_size=PLAN["dev"], test_size=PLAN["test"], judge=False),
    )

    def go(seconds_per_call, allowed):
        Clock.now, ClockedLLM.seconds, ClockedLLM.calls = 0.0, seconds_per_call, 0
        lines = []
        data_stage.build(spec, ws / "data", CFG, lines.append, seconds=allowed)
        return budget.load_fit(ws / "data"), lines

    return go


def test_a_fast_llm_writes_the_planned_sizes(run):
    fit, lines = run(FAST, HOUR)
    assert fit["written"] == PLAN
    assert not fit["shrunk"]
    assert not [line for line in lines if "instead of" in line]
    assert ClockedLLM.calls == 24 + 4 + 6


def test_a_slow_llm_shrinks_every_split_to_fit_the_budget(run):
    fit, lines = run(SLOW, HOUR)
    assert fit["written"] == {"train": 278, "dev": 50, "test": 100}
    assert fit["planned"] == PLAN
    assert fit["shrunk"]
    assert fit["seconds_per_call"] == SLOW
    assert set(fit["actual"]) == set(PLAN)
    assert all(0 < fit["actual"][s] < PLAN[s] for s in PLAN)
    expected = (
        "the LLM takes 240 s per call: writing 278 train, 50 dev and 100 test rows instead of 600, 100 and 150 "
        "to fit the 1.0 h data budget"
    )
    assert f"  {expected}" in lines
    assert ClockedLLM.calls == 3 + 9 + 2 + 4


def test_no_budget_left_still_writes_the_floors(run):
    fit, _ = run(SLOW, 0)
    assert fit["written"] == {
        "train": budget.MEASURE_CALLS * DataPlan().per_call,
        "dev": budget.MIN_DEV_ROWS,
        "test": budget.MIN_TEST_ROWS,
    }


def test_the_train_floor_is_per_label():
    need = {"train": 1000, "dev": 0, "test": 0}
    have = {"train": 0, "dev": 0, "test": 0}
    floor = {"train": 2 * budget.MIN_TRAIN_PER_LABEL, "dev": 0, "test": 0}
    assert budget.shrink(need, have, 0.0, floor)["train"] == 2 * budget.MIN_TRAIN_PER_LABEL


def test_the_autopilot_gives_the_data_stage_its_share_of_the_time_left(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "build", lambda name, exp, log, data_seconds=None: seen.update(seconds=data_seconds))
    st = {"suite": "s", "metric": "m", "ours": 0.5, "best_baseline": None}
    monkeypatch.setattr(autopilot, "standing", lambda *_: st)
    s = Session("qa", None, lambda *_, **__: None, None, 0.0, time.time() + HOUR, lambda _: None)
    autopilot._build_v1(s)
    assert seen["seconds"] == pytest.approx(DATA_BUDGET_SHARE * HOUR, rel=1e-3)
