import datetime as dt

import byosm.llm.limits as limits
import byosm.llm.recover as recover
from byosm.llm import LLMError
from byosm.llm.limits import is_limit, reset_at

MESSAGE = "claude -p failed: You've hit your monthly spend limit · your session limit resets 10:50pm (Europe/Madrid)"


def test_a_limit_message_is_recognised_and_its_reset_time_read():
    assert is_limit(LLMError(MESSAGE)) and not is_limit(LLMError("claude -p gave no JSON"))
    now = dt.datetime(2026, 10, 22, 21, 0).timestamp()
    at = dt.datetime.fromtimestamp(reset_at(MESSAGE, now) - limits.RESET_SLACK_S)
    assert (at.day, at.hour, at.minute) == (22, 22, 50)
    tomorrow = dt.datetime.fromtimestamp(reset_at("resets 3am", now) - limits.RESET_SLACK_S)
    assert (tomorrow.day, tomorrow.hour) == (23, 3)
    assert reset_at("no time here", now) is None


class Flaky:
    name = "remote"

    def __init__(self, fails):
        self.fails = fails

    def json(self, *a, **k):
        if self.fails:
            self.fails -= 1
            raise LLMError(MESSAGE)
        return {"ok": True}


def _events(monkeypatch):
    seen = []
    monkeypatch.setattr("byosm.events.emit", lambda kind, **d: seen.append((kind, d)))
    return seen


def test_in_local_mode_a_limited_call_moves_to_the_local_model(monkeypatch):
    seen = _events(monkeypatch)
    local = Flaky(0)
    monkeypatch.setattr(recover, "settings", lambda: {"mode": "local"})
    monkeypatch.setattr(recover, "local_llm", lambda: local)
    out = recover.through_limits(lambda llm: (llm.json(), llm), Flaky(1), {})
    assert out == ({"ok": True}, local)
    assert seen[-1][1]["resumed"] == "local"


def test_in_wait_mode_it_waits_for_the_reset_then_retries(monkeypatch):
    seen = _events(monkeypatch)
    monkeypatch.setattr(recover, "settings", lambda: {"mode": "wait"})
    monkeypatch.setattr(limits, "settings", lambda: {"mode": "wait"})
    clock = {"t": 1000.0}
    monkeypatch.setattr(limits.time, "time", lambda: clock["t"])
    monkeypatch.setattr(limits.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(recover.time, "time", lambda: clock["t"])
    monkeypatch.setattr(limits, "reset_at", lambda msg, now: now + 120)
    remote = Flaky(1)
    assert recover.through_limits(lambda llm: llm.json(), remote, {}) == {"ok": True}
    assert any("until" in d for _, d in seen) and clock["t"] >= 1120


def test_in_fail_mode_the_error_is_raised(monkeypatch):
    _events(monkeypatch)
    monkeypatch.setattr(recover, "settings", lambda: {"mode": "fail"})
    try:
        recover.through_limits(lambda llm: llm.json(), Flaky(1), {})
    except LLMError:
        return
    raise AssertionError("expected the limit error")
