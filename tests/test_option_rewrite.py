import pytest
from option_pool import pool

from byosm.data import build
from byosm.data.store import CALIB, Dataset
from byosm.llm.memo import memoised
from byosm.options import rewrite
from byosm.options.create import pair_plan, spec_for
from byosm.options.pairs import write

SOURCE = "lockon/xlam-function-calling-60k"


def quiet(message):
    return None


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path / "cache"))


def test_paraphrased_requests_join_train_with_the_labels_of_the_request_they_rewrite(ws, cache):
    plan = pair_plan(held_out=True)
    write(ws / "pairs.csv", pool(200), plan)
    spec = spec_for("tc", ws / "pairs.csv", SOURCE, plan)
    spec.data.pairs.paraphrases = 2
    build(spec, ws / "d", {"llm": {"default": {"provider": "fake"}}, "concurrency": 2}, quiet)
    ds = Dataset(ws / "d")
    train = ds.load("train")
    base = {(r["meta"]["group"], r["label"]) for r in train if r["meta"]["kind"] == "table"}
    para = [r for r in train if r["meta"]["kind"] == "paraphrase"]
    assert len(para) == 2 * sum(r["meta"]["kind"] == "table" for r in train)
    assert {(r["meta"]["group"], r["label"]) for r in para} <= base and {r["source"] for r in para} == {"llm:fake"}
    assert all("(take" in r["input"] for r in para)
    assert not any("(take" in r["input"] for s in ("dev", CALIB, "test") for r in ds.load(s))


class Down:
    def json(self, system, prompt, schema):
        raise RuntimeError("no LLM")


def test_paraphrases_are_written_once_per_machine(ws, cache, monkeypatch):
    cfg = {"llm": {"default": {"provider": "fake"}}, "concurrency": 1}
    first = rewrite.paraphrases(["book a flight", "weather in rome"], 2, cfg, quiet)
    monkeypatch.setattr(rewrite, "get_llm", lambda role, cfg=None: Down())
    said = []
    again = rewrite.paraphrases(["book a flight", "weather in rome", "new one"], 2, cfg, said.append)
    assert again == [*first, []] and len(first[0]) == 2 and "no LLM" in said[0]


def test_only_missing_answers_are_computed_and_a_failed_batch_is_not_kept(cache):
    asked = []

    def compute(part):
        asked.append(part)
        if "bad" in part:
            raise RuntimeError("down")
        return {k: k.upper() for k in part}

    assert memoised(["a", "b", "a"], compute, 1, 1, quiet) == ["A", "B", "A"]
    assert memoised(["b", "bad", "c"], compute, 1, 1, quiet) == ["B", None, "C"]
    assert asked == [["a"], ["b"], ["bad"], ["c"]]
