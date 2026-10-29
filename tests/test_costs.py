from pytest import approx

from byosm.costs import _compare, _per_item
from byosm.costs.prices import bare, is_local
from byosm.costs.spend import is_eval, target_of


def test_jobs_name_their_specialist_after_the_verb_or_subcommand():
    assert target_of(["improve", "intent", "--apply", "a"]) == "intent"
    assert target_of(["baseline", "add", "turn", "llm"]) == "turn"
    assert target_of(["auto", "classify emails into two kinds"]) is None


def test_rival_answers_count_as_evaluation_and_providers_are_stripped():
    assert is_eval("rival-claude-haiku-5-5") and is_eval("baseline") and not is_eval("generate")
    assert bare("claude-code:claude-opus-5-5") == "claude-opus-5-5"
    assert is_local("openai:qwen3.8:27b@http://localhost:11434/v1")


def test_break_even_is_the_build_cost_over_the_saving_per_item(monkeypatch):
    import byosm.costs as costs

    monkeypatch.setattr(costs, "price", lambda m: (1e-6, 2e-6))
    per_item = _per_item({"calls": 10, "tokens_in": 1000, "tokens_out": 500}, items=100)
    assert per_item == {"tokens_in": 10.0, "tokens_out": 5.0}
    [row] = _compare(per_item, build_usd=2.0, ours=0.0, models=["m"])
    assert abs(row["per_item_usd"] - 2e-5) < 1e-12 and row["break_even_items"] == 100_000
    assert _per_item({"calls": 0, "tokens_in": 0, "tokens_out": 0}, items=100) is None


def test_a_job_costs_its_finished_calls_at_list_price(ws, monkeypatch):
    import byosm.costs.spend as spend
    from byosm.app.store import JobStore

    monkeypatch.setattr(spend, "price", lambda m: (1e-6, 2e-6))
    store = JobStore(ws)
    store.event("j", {"kind": "llm", "model": "m", "state": "done", "tokens_in": 1000, "tokens_out": 500})
    store.event("j", {"kind": "llm", "model": "m", "state": "failed"})
    cost = spend.job_cost(ws, "j")
    assert cost["calls"] == 1 and abs(cost["usd"] - 0.002) < 1e-12


def test_a_specialist_spend_sums_finished_calls_per_bucket_and_counts_unpriced_ones(ws, monkeypatch):
    import byosm.costs.spend as spend
    from byosm.app.store import JobStore

    monkeypatch.setattr(spend, "price", lambda m: None if m == "unknown" else (1e-6, 2e-6))
    store = JobStore(ws)
    job = {"id": "j", "args": ["improve", "intent"], "status": "done", "created": 0.0, "started": 0.0}
    store.save({**job, "ended": 1.0, "code": 0, "pid": None})
    calls = [
        {"role": "generate", "tokens_in": 1000, "tokens_out": 500},
        {"role": "generate", "tokens_in": 1000, "tokens_out": 500},
        {"role": "generate", "tokens_in": 7, "tokens_out": 7, "model": "unknown"},
        {"role": "generate", "tokens_in": 99, "tokens_out": 99, "state": "failed"},
        {"role": "rival-claude", "tokens_in": 10, "tokens_out": 5},
        {"role": "card"},
    ]
    for c in calls:
        store.event("j", {"kind": "llm", "model": "m", "state": "done", **c})
    got = spend.usage(ws)["intent"]
    assert got["build"] == {"calls": 3, "tokens_in": 2007, "tokens_out": 1007, "usd": approx(0.004), "unpriced": 1}
    assert got["evaluate"] == {"calls": 2, "tokens_in": 10, "tokens_out": 5, "usd": approx(2e-5), "unpriced": 0}
    assert got["rival"]["calls"] == 1
