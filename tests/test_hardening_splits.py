import random

from byosm.data.splits import _hold_out
from byosm.data.store import TEST_SHARE


def _frame_rows(n: int) -> list[dict]:
    rng = random.Random(1)
    return [
        {"input": f"clip {i} {i * 7919}", "label": [rng.choice(["speech", "none"]) for _ in range(100)]}
        for i in range(n)
    ]


def test_frame_labelled_rows_split_at_test_share_when_test_is_empty():
    splits = {"train": _frame_rows(180), "dev": _frame_rows(20), "test": []}
    _hold_out(splits, random.Random(0), lambda _: None)
    assert abs(len(splits["test"]) - 180 * TEST_SHARE) <= 3
    assert len(splits["train"]) + len(splits["test"]) == 180
    assert len(splits["dev"]) == 20


def test_frame_labelled_rows_fill_dev_and_test_when_both_are_empty():
    splits = {"train": _frame_rows(200), "dev": [], "test": []}
    _hold_out(splits, random.Random(0), lambda _: None)
    assert len(splits["train"]) > 100 and splits["dev"] and splits["test"]
