import random

from byosm.data.noise import cleaned, suspects
from byosm.data.store import record
from byosm.spec import Label, TaskSpec

QUESTIONS = ["is the shop open on sunday", "can you send me the invoice", "where is my parcel", "how much is it"]
STATEMENTS = ["the shop opens at nine", "i sent the invoice today", "my parcel arrived", "it costs ten euros"]


def _rows(flipped):
    rng = random.Random(0)
    rows = [record(f"{rng.choice(QUESTIONS)} {i}?", "question", "llm") for i in range(60)]
    rows += [record(f"{rng.choice(STATEMENTS)} {i}.", "statement", "llm") for i in range(60)]
    wrong = [record(f"{rng.choice(QUESTIONS)} {i}?", "statement", "llm") for i in range(flipped)]
    return rows + wrong, wrong


def _spec(on=True):
    labels = [Label(name=n, definition="") for n in ("question", "statement")]
    spec = TaskSpec(name="t", description="", modality="text", labels=labels)
    spec.data.clean_labels = on
    return spec


def test_rows_whose_label_disagrees_with_their_neighbours_rank_as_suspects():
    rows, wrong = _rows(6)
    flagged = {r["id"] for r in suspects(rows)}
    assert sum(r["id"] in flagged for r in wrong) >= 5


def test_only_suspects_the_judge_rejects_leave_training_and_the_setting_is_off_by_default():
    rows, wrong = _rows(6)
    judged = []

    def judge(spec, flagged, cfg, log):
        judged.extend(flagged)
        return [r for r in flagged if r["label"] == "question" or not r["input"].endswith("?")]

    kept = cleaned(_spec(), rows, None, lambda _: None, judge)
    assert not {r["id"] for r in wrong} & {r["id"] for r in kept}
    assert len(rows) - len(kept) <= len(judged)
    assert cleaned(_spec(False), rows, None, lambda _: None, judge) == rows
