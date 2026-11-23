import numpy as np

from byosm.evals.grouped import CATEGORIES, grouped_metrics, summary
from byosm.spec import Label, TaskSpec


def test_rows_with_a_category_get_accuracy_per_category_on_the_board():
    labels = [Label(name="fold", definition=""), Label(name="call", definition="")]
    spec = TaskSpec(name="t", description="", modality="text", labels=labels)
    streets = ["preflop", "preflop", "flop", "flop", "flop"]
    truth = ["fold", "call", "fold", "fold", "call"]
    test = [
        {"id": str(i), "label": y, "meta": {"category": s}} for i, (s, y) in enumerate(zip(streets, truth, strict=True))
    ]
    dec = np.array([0, 1, 0, 1, 1])
    m = grouped_metrics(spec, [], test, None, np.zeros((5, 2)), dec)
    assert m[CATEGORIES] == {"flop": {"accuracy": 2 / 3, "n": 3}, "preflop": {"accuracy": 1.0, "n": 2}}
    assert summary(m) == "flop 0.667, preflop 1.000"
