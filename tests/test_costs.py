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
