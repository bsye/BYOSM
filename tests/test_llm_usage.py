import json

import pytest

from byosm import llm
from byosm.events import EVENT_PREFIX, EVENTS_ENV
from byosm.llm.base import report_usage


class _Counted:
    name = "counted"

    def __init__(self, **_):
        pass

    def json(self, system, prompt, schema):
        report_usage(120, 30)
        return {"ok": True}


@pytest.fixture
def counted(monkeypatch):
    llm.register_provider("counted", _Counted)
    llm.clear_cache()
    monkeypatch.setitem(llm._spent, "tokens", 0)
    monkeypatch.setenv(EVENTS_ENV, "1")


def test_every_finished_call_reports_its_tokens(counted, capsys):
    cfg = {"llm": {"default": {"provider": "counted"}}}
    llm.get_llm("judge", cfg).json("s", "p", {})
    done = [json.loads(line[len(EVENT_PREFIX) :]) for line in capsys.readouterr().out.splitlines()][-1]
    assert (done["state"], done["tokens_in"], done["tokens_out"]) == ("done", 120, 30)


def test_a_job_stops_calling_the_llm_once_its_token_budget_is_spent(counted):
    cfg = {"llm": {"default": {"provider": "counted"}}, "budget": {"tokens_per_job": 300}}
    model = llm.get_llm("generate", cfg)
    model.json("s", "p", {})
    model.json("s", "p", {})
    with pytest.raises(llm.LLMError, match="token budget reached: 300 of 300"):
        model.json("s", "p", {})
