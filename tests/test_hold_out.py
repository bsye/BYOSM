import random

from byosm.data.splits import _hold_out
from byosm.data.store import record


def test_a_build_with_no_dev_or_test_rows_holds_out_a_stratified_share_of_train():
    train = [record(f"message number {i} about {lab}", lab, "llm") for lab in ("a", "b", "c") for i in range(40)]
    splits = {"train": list(train), "dev": [], "test": []}
    _hold_out(splits, random.Random(0), log=lambda m: None)
    assert splits["dev"] and splits["test"] and len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == 120
    assert {r["label"] for r in splits["test"]} == {"a", "b", "c"}
    assert not {r["input"] for r in splits["test"]} & {r["input"] for r in splits["train"]}


def test_existing_dev_and_test_rows_are_left_alone():
    splits = {
        "train": [record(f"x{i}", "a", "llm") for i in range(80)],
        "dev": [record("d", "a", "llm")],
        "test": [record("t", "a", "llm")],
    }
    _hold_out(splits, random.Random(0), log=lambda m: None)
    assert len(splits["train"]) == 80 and len(splits["dev"]) == 1
