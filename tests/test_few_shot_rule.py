from byosm.improve.cost import estimate
from byosm.improve.rules_fit import few_shot
from byosm.spec import Label, TaskSpec

SETFIT_OP = [{"op": "set", "path": "train.method", "value": "setfit"}]


def _spec(modality="text"):
    labels = [Label(name=n, definition="") for n in "abc"]
    return TaskSpec(name="t", description="", modality=modality, labels=labels)


def test_few_real_rows_per_label_propose_setfit_and_the_embedding_head():
    found = few_shot({"spec": _spec(), "n_train": 60})
    assert [pr["id"] for pr in found] == ["setfit", "embedding-head"] and "20 training rows per label" in found[0][
        "why"
    ]
    assert few_shot({"spec": _spec(), "n_train": 300}) == []
    assert few_shot({"spec": _spec("audio"), "n_train": 60}) == []


def test_an_embedding_method_costs_less_than_fine_tuning_the_same_data():
    dg = {"spec": _spec(), "train_seconds": 600}
    assert estimate(dg, {"ops": SETFIT_OP})["minutes"] == 3.0
    assert estimate(dg, {"ops": [{"op": "set", "path": "train.epochs", "value": 16}]})["minutes"] == 20.0
