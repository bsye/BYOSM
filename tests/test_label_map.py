from types import SimpleNamespace

import torch

from byosm.predictors import label_map
from byosm.spec import Label, TaskSpec


def test_each_output_maps_to_the_label_it_agrees_with_most(monkeypatch):
    spec = TaskSpec(
        name="t",
        description="t",
        modality="text",
        labels=[Label(name="no", definition=""), Label(name="yes", definition="")],
    )
    rows = [{"input": f"y{i}", "label": "yes"} for i in range(10)] + [
        {"input": f"n{i}", "label": "no"} for i in range(10)
    ]

    def model(**enc):
        return SimpleNamespace(
            logits=torch.tensor([[0.0, 1.0] if t.startswith("y") else [1.0, 0.0] for t in enc["texts"]])
        )

    model.config = SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1"})
    monkeypatch.setattr(label_map, "load_model", lambda repo, visual: (lambda texts, **k: {"texts": texts}, model))
    got = label_map.suggest_label_map(spec, rows, "org/m")
    assert got["map"] == {"LABEL_1": "yes", "LABEL_0": "no"} and got["agreement"] == 1.0 and got["rows"] == 20
